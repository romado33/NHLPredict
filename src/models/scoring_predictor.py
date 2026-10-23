"""
Playoff scoring prediction model.

Predicts individual playoff point totals using regular-season stats
and team-level expected games from the team prediction model.

Architecture:
  - Two-stage prediction: gate classifier (P(scores > 0)) + value regressors
  - Log-transform targets to handle right-skewed distributions
  - Ridge, XGBoost, LightGBM regressors with temporal CV
  - Stacking meta-learner for optimal model combination
  - Exponential sample weighting for recency
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import RANDOM_STATE
from src.utils.logging_config import get_logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = get_logger(__name__)

SAMPLE_WEIGHT_DECAY = 0.88
# Stronger L2 on stacking meta reduces overfitting to OOF base predictions
META_RIDGE_ALPHA = 3.0
# Dirichlet search: trade a small amount of raw MAE for better rank alignment (Spearman)
DIRICHLET_RANK_WEIGHT = 0.12
# Drop near-empty columns (same idea as team V2 pipeline)
MAX_NAN_FRACTION = 0.75

META_COLS = [
    "playerId", "playerName", "team_abbrev", "season", "positionCode",
    "target_points", "target_goals", "target_assists", "target_gp",
]

FEATURE_COLS = [
    # Core scoring
    "reg_points", "reg_goals", "reg_assists", "reg_gp",
    "reg_ppg", "reg_gpg", "reg_apg",
    # Situational
    "reg_ev_points", "reg_pp_points", "reg_sh_points",
    "reg_pp_goals", "reg_sh_goals", "reg_ev_goals",
    # Shooting / ice time
    "reg_shots", "reg_shots_per_game", "reg_shooting_pct",
    "reg_toi_per_game", "reg_gwg", "reg_plus_minus",
    # Position
    "is_forward", "is_center", "is_defenseman",
    # Team context
    "team_expected_games",
    "team_prob_reach_r2",
    "team_prob_reach_cf",
    "team_prob_reach_final",
    "pts_x_prob_final",
    # Historical playoff
    "prior_po_ppg", "prior_po_gp", "prior_po_pts", "career_po_seasons",
    # Interactions
    "pts_x_expected_games", "goals_x_expected_games",
    "assists_x_expected_games", "pp_pts_x_expected_games",
    "toi_x_expected_games", "shots_x_expected_games",
    "prior_po_ppg_x_games",
    # Shares
    "pp_pts_share", "ev_pts_share",
    # Position-adjusted
    "fwd_ppg", "def_ppg",
    # Team rank
    "team_pts_rank", "team_pts_share",
    "is_top3_team_scorer", "is_top6_team_scorer",
    # NEW: age and position discount
    "age", "age_squared",
    "playoff_discount_ppg",
    # NEW: team offensive context
    "team_gf_per_gp", "team_pp_pct", "team_shots_per_gp", "team_standings_pts",
    # NEW: age/opportunity interactions
    "age_x_expected_games", "age_x_ppg",
    # NEW: recent form / availability proxies
    "gp_availability", "ppg_per_toi",
    # Deep-run attenuation: diminishing returns for longer playoff runs
    "exp_games_sqrt", "pts_x_sqrt_games", "toi_x_sqrt_games", "deep_run_ratio",
    # Opponent quality: how strong are the opponents on this team's bracket path
    "opp_pts_pace", "opp_ga_per_gp",
]


def prepare_xy(
    df: pd.DataFrame, target: str = "target_points"
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split into features X and target y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[target].copy()
    return X, y, available


def temporal_cv_splits(
    df: pd.DataFrame, min_train_years: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window temporal CV splits by season."""
    seasons = sorted(df["season"].unique())
    splits = []
    for i in range(min_train_years, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        train_idx = df.index[df["season"].isin(train_seasons)].to_numpy()
        test_idx = df.index[df["season"] == test_season].to_numpy()
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def _build_sample_weights(seasons: np.ndarray) -> np.ndarray:
    """Exponential recency weights — recent seasons matter more."""
    max_s = seasons.max()
    return np.array([SAMPLE_WEIGHT_DECAY ** (max_s - s) for s in seasons])


def _train_gate_classifier(
    X: pd.DataFrame, y: pd.Series, weights: np.ndarray
) -> Pipeline:
    """
    Binary classifier predicting P(player scores > 0 points in playoffs).
    Used as stage-1 of the two-stage architecture.
    """
    y_binary = (y > 0).astype(int)
    gate = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE,
        )),
    ])
    gate.fit(X, y_binary, model__sample_weight=weights)
    pos_rate = y_binary.mean()
    logger.info("Gate classifier trained: P(scores > 0) base rate = %.3f", pos_rate)
    return gate


def train_scoring_models(
    df: pd.DataFrame,
    target: str = "target_points",
) -> dict[str, Any]:
    """
    Train all model variants with hyperparameter tuning and temporal CV.

    Architecture improvements:
      - Log-transform targets to handle right-skewed distributions
      - Two-stage: gate classifier + value regressors
      - Stacking meta-learner replaces Dirichlet blend
      - Sharper recency-weighted samples (decay=0.88)

    Returns dict with models, gate, feature names, blend weights, and eval metrics.
    """
    df = df.sort_values("season").reset_index(drop=True)

    X, y, feature_cols = prepare_xy(df, target)
    logger.info("Training scoring models for '%s': %d samples, %d features",
                target, len(X), len(feature_cols))

    # Drop all-NaN columns
    all_nan = X.columns[X.isnull().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)
        feature_cols = [c for c in feature_cols if c not in all_nan]

    # Drop very sparse columns (reduces noise + imputer warnings on test folds)
    high_nan = X.columns[X.isnull().mean() > MAX_NAN_FRACTION].tolist()
    if high_nan:
        logger.info(
            "Dropping %d columns with >%.0f%% NaN",
            len(high_nan), 100 * MAX_NAN_FRACTION,
        )
        X = X.drop(columns=high_nan)
        feature_cols = [c for c in feature_cols if c not in high_nan]

    seasons = df["season"].values
    weights = _build_sample_weights(seasons)

    # ── Stage 1: Gate classifier (P(scores > 0)) ──
    gate = _train_gate_classifier(X, y, weights)

    # ── Log-transform targets for regressors ──
    y_log = np.log1p(y)

    models = {}

    # 1) Ridge Regression
    logger.info("Training Ridge (log-target)...")
    ridge_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=RANDOM_STATE)),
    ])
    ridge_search = RandomizedSearchCV(
        ridge_pipe,
        {"model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]},
        n_iter=8, cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    ridge_search.fit(X, y_log, model__sample_weight=weights)
    models["ridge"] = ridge_search.best_estimator_
    logger.info("Ridge best CV-MAE(log)=%.3f (alpha=%.3f)",
                -ridge_search.best_score_, ridge_search.best_params_["model__alpha"])

    # 2) XGBoost
    if xgb is not None:
        logger.info("Training XGBoost Regressor (log-target)...")
        xgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(
                objective="reg:squarederror", eval_metric="mae",
                random_state=RANDOM_STATE,
            )),
        ])
        xgb_search = RandomizedSearchCV(
            xgb_pipe,
            {
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
                "model__n_estimators": [100, 200, 300, 500],
                "model__subsample": [0.6, 0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
                "model__reg_alpha": [0, 0.1, 0.5, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "model__min_child_weight": [1, 3, 5, 7],
            },
            n_iter=100, cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_absolute_error",
            n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        xgb_search.fit(X, y_log, model__sample_weight=weights)
        models["xgboost"] = xgb_search.best_estimator_
        logger.info("XGBoost best CV-MAE(log)=%.3f", -xgb_search.best_score_)

    # 3) LightGBM
    if lgb is not None:
        logger.info("Training LightGBM Regressor (log-target)...")
        lgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMRegressor(verbose=-1, random_state=RANDOM_STATE)),
        ])
        lgb_search = RandomizedSearchCV(
            lgb_pipe,
            {
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
                "model__n_estimators": [100, 200, 300, 500],
                "model__num_leaves": [7, 15, 21, 31, 47],
                "model__subsample": [0.6, 0.7, 0.8, 0.9],
                "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
                "model__reg_alpha": [0, 0.1, 0.5, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0],
                "model__min_child_samples": [3, 5, 10, 20],
            },
            n_iter=100, cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_absolute_error",
            n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        lgb_search.fit(X, y_log, model__sample_weight=weights)
        models["lightgbm"] = lgb_search.best_estimator_
        logger.info("LightGBM best CV-MAE(log)=%.3f", -lgb_search.best_score_)

    # ── Temporal CV evaluation (on original scale) ──
    eval_results = _temporal_cv_evaluate(models, gate, X, y, df, target)

    # ── Stacking meta-learner (replaces Dirichlet blend) + isotonic OOF calibration ──
    blend_weights, meta_model, output_isotonic = _train_stacking_meta(
        models, gate, X, y, df,
    )

    # ── Position-specific calibration ──
    pos_calibration = _train_position_calibration(models, gate, X, y, df, weights)
    star_calibration = _train_star_calibration(models, gate, X, y, df, weights)

    # ── Final fit on all data ──
    for name, model in models.items():
        model.fit(X, y_log)
    gate.fit(X, (y > 0).astype(int), model__sample_weight=weights)

    return {
        "models": models,
        "gate": gate,
        "meta_model": meta_model,
        "output_isotonic": output_isotonic,
        "feature_cols": feature_cols,
        "blend_weights": blend_weights,
        "eval_results": eval_results,
        "pos_calibration": pos_calibration,
        "star_calibration": star_calibration,
        "target": target,
    }


def _temporal_cv_evaluate(
    models: dict, gate: Pipeline,
    X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, target: str,
) -> dict[str, dict]:
    """Evaluate models with temporal expanding-window CV on original scale."""
    splits = temporal_cv_splits(df)
    logger.info("Temporal CV: %d splits", len(splits))
    seasons = df["season"].values
    weights = _build_sample_weights(seasons)

    results = {}

    # Evaluate individual models
    for name, model in models.items():
        all_preds = []
        all_true = []
        season_metrics = []

        for train_idx, test_idx in splits:
            y_train_log = np.log1p(y.iloc[train_idx])
            model.fit(X.iloc[train_idx], y_train_log)

            preds_log = model.predict(X.iloc[test_idx])
            preds = np.expm1(np.maximum(preds_log, 0))
            preds = np.maximum(preds, 0)
            true = y.iloc[test_idx].values

            all_preds.extend(preds)
            all_true.extend(true)
            _append_season_metrics(season_metrics, df, test_idx, true, preds)

        _store_results(results, name, all_preds, all_true, season_metrics)

    # Evaluate two-stage gated blend
    all_preds_gated = []
    all_true_gated = []
    season_metrics_gated = []

    for train_idx, test_idx in splits:
        y_train = y.iloc[train_idx]
        y_train_log = np.log1p(y_train)
        y_binary = (y_train > 0).astype(int)
        train_w = weights[train_idx]

        gate.fit(X.iloc[train_idx], y_binary, model__sample_weight=train_w)
        gate_prob = gate.predict_proba(X.iloc[test_idx])[:, 1]

        model_preds = {}
        for name, model in models.items():
            model.fit(X.iloc[train_idx], y_train_log)
            p_log = model.predict(X.iloc[test_idx])
            model_preds[name] = np.expm1(np.maximum(p_log, 0))

        avg_pred = np.mean(list(model_preds.values()), axis=0)
        gated = gate_prob * np.maximum(avg_pred, 0)
        true = y.iloc[test_idx].values

        all_preds_gated.extend(gated)
        all_true_gated.extend(true)
        _append_season_metrics(season_metrics_gated, df, test_idx, true, gated)

    _store_results(results, "gated_blend", all_preds_gated, all_true_gated,
                   season_metrics_gated)

    return results


def _topk_overlap(
    combined: pd.DataFrame, k: int, true_col: str = "true", pred_col: str = "pred",
    name_col: str = "playerName",
) -> float:
    n = min(k, len(combined))
    if n <= 0:
        return 0.0
    actual_top = set(combined.nlargest(n, true_col)[name_col])
    pred_top = set(combined.nlargest(n, pred_col)[name_col])
    return len(actual_top & pred_top) / float(n)


def _append_season_metrics(
    metrics_list: list, df: pd.DataFrame,
    test_idx: np.ndarray, true: np.ndarray, preds: np.ndarray,
):
    test_season = df.iloc[test_idx]["season"].iloc[0]
    mae = mean_absolute_error(true, preds)
    rmse = np.sqrt(mean_squared_error(true, preds))
    combined = pd.DataFrame({
        "true": true, "pred": preds,
        "playerName": df.iloc[test_idx]["playerName"].values,
    })
    metrics_list.append({
        "season": test_season, "mae": mae, "rmse": rmse,
        "top10_overlap": _topk_overlap(combined, 10),
        "top35_overlap": _topk_overlap(combined, 35),
        "top40_overlap": _topk_overlap(combined, 40),
        "n_players": len(true),
    })


def _store_results(
    results: dict, name: str,
    all_preds: list, all_true: list, season_metrics: list,
):
    all_preds_arr = np.array(all_preds)
    all_true_arr = np.array(all_true)
    overall_mae = mean_absolute_error(all_true_arr, all_preds_arr)
    overall_rmse = np.sqrt(mean_squared_error(all_true_arr, all_preds_arr))
    corr, _ = spearmanr(all_true_arr, all_preds_arr)
    season_df = pd.DataFrame(season_metrics)
    results[name] = {
        "mae": overall_mae,
        "rmse": overall_rmse,
        "spearman": corr,
        "mean_top10_overlap": season_df["top10_overlap"].mean(),
        "mean_top35_overlap": season_df["top35_overlap"].mean(),
        "mean_top40_overlap": season_df["top40_overlap"].mean(),
        "season_results": season_df,
    }


def _train_stacking_meta(
    models: dict, gate: Pipeline,
    X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
) -> tuple[dict[str, float], Ridge | None, IsotonicRegression | None]:
    """
    Train stacking meta-learner + honest nested OOF for isotonic calibration.

    Nested OOF meta predictions avoid fitting the meta-learner on its own inputs,
    then isotonic regression calibrates the stacked output on held-out seasons.

    Returns (blend_weights_dict, meta_model, output_isotonic).
    """
    splits = temporal_cv_splits(df)
    model_names = list(models.keys())
    weights = _build_sample_weights(df["season"].values)

    oof_preds = {n: np.full(len(y), np.nan) for n in model_names}
    oof_gate = np.full(len(y), np.nan)

    for train_idx, test_idx in splits:
        y_train = y.iloc[train_idx]
        y_train_log = np.log1p(y_train)
        y_binary = (y_train > 0).astype(int)
        train_w = weights[train_idx]

        gate.fit(X.iloc[train_idx], y_binary, model__sample_weight=train_w)
        oof_gate[test_idx] = gate.predict_proba(X.iloc[test_idx])[:, 1]

        for name in model_names:
            models[name].fit(X.iloc[train_idx], y_train_log)
            p_log = models[name].predict(X.iloc[test_idx])
            oof_preds[name][test_idx] = np.expm1(np.maximum(p_log, 0))

    valid = np.all([~np.isnan(oof_preds[n]) for n in model_names], axis=0)
    valid &= ~np.isnan(oof_gate)

    if valid.sum() < 100:
        logger.warning(
            "Stacking: insufficient OOF data (%d valid), using equal weights",
            valid.sum(),
        )
        equal_w = {n: 1.0 / len(model_names) for n in model_names}
        return equal_w, None, None

    y_v = y[valid].values
    gate_v = oof_gate[valid]

    # ── Nested OOF: Ridge meta fit only on train folds, predict test (honest meta_oof)
    meta_oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splits:
        tr = np.intersect1d(train_idx, np.where(valid)[0], assume_unique=True)
        te = np.intersect1d(test_idx, np.where(valid)[0], assume_unique=True)
        if len(tr) < 50 or len(te) < 10:
            continue
        meta_X_tr = np.column_stack(
            [oof_preds[n][tr] for n in model_names] + [oof_gate[tr]]
        )
        meta_X_te = np.column_stack(
            [oof_preds[n][te] for n in model_names] + [oof_gate[te]]
        )
        y_tr = y.iloc[tr].values
        ridge = Ridge(alpha=META_RIDGE_ALPHA, fit_intercept=True)
        ridge.fit(meta_X_tr, y_tr)
        meta_oof[te] = np.maximum(ridge.predict(meta_X_te), 0)

    honest = valid & ~np.isnan(meta_oof)
    if honest.sum() < 100:
        logger.warning("Nested meta OOF too sparse (%d), skipping isotonic", honest.sum())
    else:
        logger.info(
            "Nested OOF meta MAE (honest): %.3f",
            mean_absolute_error(y.iloc[honest].values, meta_oof[honest]),
        )

    # In-sample meta (for coefs / fallback weights) — slightly optimistic
    meta_X = np.column_stack([oof_preds[n][valid] for n in model_names] + [gate_v])
    meta = Ridge(alpha=META_RIDGE_ALPHA, fit_intercept=True)
    meta.fit(meta_X, y_v)
    meta_pred_ins = np.maximum(meta.predict(meta_X), 0)
    meta_mae_in = mean_absolute_error(y_v, meta_pred_ins)

    coefs = meta.coef_[: len(model_names)]
    total_c = np.abs(coefs).sum()
    blend_weights = {
        model_names[i]: float(np.abs(coefs[i]) / max(total_c, 1e-8))
        for i in range(len(model_names))
    }

    logger.info(
        "Stacking meta MAE (in-sample OOF design matrix): %.3f | Gate coef: %.3f | Model weights: %s",
        meta_mae_in,
        meta.coef_[-1],
        {n: f"{w:.3f}" for n, w in blend_weights.items()},
    )

    # Isotonic calibration on honest nested-OOF meta predictions
    output_iso: IsotonicRegression | None = None
    if honest.sum() >= 100:
        y_iso = y.iloc[honest].values
        mo = meta_oof[honest]
        output_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        output_iso.fit(mo, y_iso)
        iso_mae = mean_absolute_error(y_iso, output_iso.predict(mo))
        logger.info(
            "Output isotonic MAE (nested OOF): %.3f (was %.3f pre-iso)",
            iso_mae,
            mean_absolute_error(y_iso, mo),
        )

    # Dirichlet search on OOF: minimize MAE with a Spearman bonus (rank-aware blend)
    best_dirichlet_rank_loss = float("inf")
    best_dirichlet_mae = float("inf")
    best_dirichlet_w = {n: 1.0 / len(model_names) for n in model_names}
    np.random.seed(RANDOM_STATE)
    for _ in range(15000):
        raw = np.random.dirichlet(np.ones(len(model_names)))
        ws = {model_names[i]: raw[i] for i in range(len(model_names))}
        blended = sum(ws[n] * oof_preds[n][valid] for n in model_names)
        gated = gate_v * np.maximum(blended, 0)
        mae_d = mean_absolute_error(y_v, gated)
        spear, _ = spearmanr(y_v, gated)
        if np.isnan(spear):
            spear = 0.0
        rank_loss = mae_d - DIRICHLET_RANK_WEIGHT * spear
        if rank_loss < best_dirichlet_rank_loss:
            best_dirichlet_rank_loss = rank_loss
            best_dirichlet_mae = mae_d
            best_dirichlet_w = ws

    # Dirichlet + optional isotonic on same OOF
    dirichlet_gated = sum(
        best_dirichlet_w[n] * oof_preds[n][valid] for n in model_names
    )
    dirichlet_gated = gate_v * np.maximum(dirichlet_gated, 0)
    iso_d = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
    iso_d.fit(dirichlet_gated, y_v)
    dirichlet_iso_mae = mean_absolute_error(y_v, iso_d.predict(dirichlet_gated))

    honest_v = honest[valid]
    if honest_v.sum() > 0:
        dirichlet_iso_mae_h = mean_absolute_error(
            y_v[honest_v],
            iso_d.predict(dirichlet_gated[honest_v]),
        )
    else:
        dirichlet_iso_mae_h = dirichlet_iso_mae

    logger.info(
        "Dirichlet+gate blend MAE: %.3f | iso-cal (all valid): %.3f | Weights: %s",
        best_dirichlet_mae,
        dirichlet_iso_mae,
        {n: f"{w:.3f}" for n, w in best_dirichlet_w.items()},
    )

    meta_iso_mae_h = float("inf")
    if output_iso is not None and honest.sum() >= 100:
        meta_iso_mae_h = mean_absolute_error(
            y.iloc[honest].values,
            output_iso.predict(meta_oof[honest]),
        )

    # Prefer lower isotonic-calibrated MAE on the same nested-OOF rows when available
    if output_iso is not None and honest.sum() >= 100 and meta_iso_mae_h < float("inf"):
        use_meta = meta_iso_mae_h <= dirichlet_iso_mae_h
        logger.info(
            "Iso MAE on nested-OOF rows: meta=%.3f vs dirichlet=%.3f -> %s",
            meta_iso_mae_h,
            dirichlet_iso_mae_h,
            "meta" if use_meta else "dirichlet",
        )
    else:
        use_meta = meta_mae_in <= best_dirichlet_mae

    if use_meta:
        logger.info(
            "Using stacking meta-learner + isotonic (in-sample meta MAE %.3f, raw Dirichlet %.3f)",
            meta_mae_in,
            best_dirichlet_mae,
        )
        return blend_weights, meta, output_iso

    logger.info(
        "Using Dirichlet+gate + isotonic (raw Dirichlet %.3f vs in-sample meta %.3f)",
        best_dirichlet_mae,
        meta_mae_in,
    )
    return best_dirichlet_w, None, iso_d


def _train_position_calibration(
    models: dict,
    gate: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float]:
    """
    Learn position-specific scaling factors from temporal CV residuals.

    Forwards and defensemen have different scoring distributions —
    a single model tends to overpredict defensemen and underpredict
    elite forwards. This computes a per-position correction factor.
    """
    splits = temporal_cv_splits(df)
    if not splits:
        return {"F": 1.0, "D": 1.0}

    position_preds: dict[str, list] = {"F": [], "D": []}
    position_trues: dict[str, list] = {"F": [], "D": []}

    for train_idx, test_idx in splits:
        y_train_log = np.log1p(y.iloc[train_idx])
        y_binary = (y.iloc[train_idx] > 0).astype(int)
        train_w = weights[train_idx]

        gate.fit(X.iloc[train_idx], y_binary, model__sample_weight=train_w)
        gate_prob = gate.predict_proba(X.iloc[test_idx])[:, 1]

        model_preds_list = []
        for name, model in models.items():
            model.fit(X.iloc[train_idx], y_train_log)
            p_log = model.predict(X.iloc[test_idx])
            model_preds_list.append(np.expm1(np.maximum(p_log, 0)))

        avg_pred = np.mean(model_preds_list, axis=0)
        gated = gate_prob * np.maximum(avg_pred, 0)
        true = y.iloc[test_idx].values

        test_df = df.iloc[test_idx]
        is_fwd = test_df.get("is_forward", pd.Series([1] * len(test_idx))).values

        for i, idx in enumerate(range(len(test_idx))):
            pos = "F" if is_fwd[i] == 1 else "D"
            position_preds[pos].append(gated[i])
            position_trues[pos].append(true[i])

    calibration = {}
    for pos in ("F", "D"):
        preds = np.array(position_preds[pos])
        trues = np.array(position_trues[pos])
        if len(preds) > 0 and np.sum(preds) > 0:
            ratio = np.sum(trues) / np.sum(preds)
            calibration[pos] = round(float(np.clip(ratio, 0.7, 1.3)), 4)
        else:
            calibration[pos] = 1.0

    logger.info("Position calibration factors: F=%.4f, D=%.4f",
                calibration["F"], calibration["D"])
    return calibration


def _train_star_calibration(
    models: dict,
    gate: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float]:
    """
    Shrinkage factors for elite team scorers vs depth (is_top3_team_scorer).
    Reduces systematic MAE gap on stars vs fourth liners seen in diagnostics.
    """
    splits = temporal_cv_splits(df)
    if not splits or "is_top3_team_scorer" not in df.columns:
        return {"elite": 1.0, "depth": 1.0}

    elite_preds: list = []
    elite_trues: list = []
    depth_preds: list = []
    depth_trues: list = []

    for train_idx, test_idx in splits:
        y_train_log = np.log1p(y.iloc[train_idx])
        y_binary = (y.iloc[train_idx] > 0).astype(int)
        train_w = weights[train_idx]

        gate.fit(X.iloc[train_idx], y_binary, model__sample_weight=train_w)
        gate_prob = gate.predict_proba(X.iloc[test_idx])[:, 1]

        model_preds_list = []
        for _, model in models.items():
            model.fit(X.iloc[train_idx], y_train_log)
            p_log = model.predict(X.iloc[test_idx])
            model_preds_list.append(np.expm1(np.maximum(p_log, 0)))

        avg_pred = np.mean(model_preds_list, axis=0)
        gated = gate_prob * np.maximum(avg_pred, 0)
        true = y.iloc[test_idx].values
        top3 = df.iloc[test_idx]["is_top3_team_scorer"].values

        for i in range(len(test_idx)):
            if top3[i] == 1:
                elite_preds.append(gated[i])
                elite_trues.append(true[i])
            else:
                depth_preds.append(gated[i])
                depth_trues.append(true[i])

    out: dict[str, float] = {}
    for key, preds_l, trues_l in (
        ("elite", elite_preds, elite_trues),
        ("depth", depth_preds, depth_trues),
    ):
        preds = np.array(preds_l)
        trues = np.array(trues_l)
        if len(preds) > 0 and np.sum(preds) > 0:
            ratio = np.sum(trues) / np.sum(preds)
            out[key] = round(float(np.clip(ratio, 0.65, 1.35)), 4)
        else:
            out[key] = 1.0

    logger.info(
        "Star/depth calibration: elite(top3)=%.4f, depth=%.4f",
        out["elite"], out["depth"],
    )
    return out


def predict_blended(
    models: dict, blend_weights: dict, X: pd.DataFrame,
    gate: Pipeline | None = None,
    meta_model: Ridge | None = None,
    pos_calibration: dict[str, float] | None = None,
    star_calibration: dict[str, float] | None = None,
    output_isotonic: IsotonicRegression | None = None,
) -> np.ndarray:
    """
    Generate predictions using the best available strategy:
    1. Stacking meta-learner (if available)
    2. Gated Dirichlet blend (if gate available)
    3. Simple weighted blend (fallback)

    All base models trained on log-targets; predictions back-transformed.
    Optional isotonic calibration (OOF-fitted) then position-specific calibration.
    """
    model_names = list(models.keys())
    raw_preds = {}
    for name, model in models.items():
        p_log = model.predict(X)
        raw_preds[name] = np.maximum(np.expm1(np.maximum(p_log, 0)), 0)

    gate_prob = None
    if gate is not None:
        try:
            gate_prob = gate.predict_proba(X)[:, 1]
        except Exception:
            gate_prob = None

    if meta_model is not None and gate_prob is not None:
        meta_X = np.column_stack(
            [raw_preds[n] for n in model_names] + [gate_prob]
        )
        preds = np.maximum(meta_model.predict(meta_X), 0)
    else:
        preds = np.zeros(len(X))
        total_w = 0.0
        for name, model in models.items():
            w = blend_weights.get(name, 0)
            if w <= 0:
                continue
            preds += w * raw_preds[name]
            total_w += w
        preds = preds / max(total_w, 1e-8)

        if gate_prob is not None:
            preds = gate_prob * preds

    if output_isotonic is not None:
        preds = np.maximum(
            output_isotonic.predict(preds.reshape(-1, 1)).ravel(), 0.0
        )

    # Apply position-specific calibration
    if pos_calibration and "is_forward" in X.columns:
        is_fwd = X["is_forward"].values
        for i in range(len(preds)):
            pos_key = "F" if is_fwd[i] == 1 else "D"
            preds[i] *= pos_calibration.get(pos_key, 1.0)

    # Elite vs depth (team top-3 regular-season scorer flag)
    if star_calibration and "is_top3_team_scorer" in X.columns:
        t3 = X["is_top3_team_scorer"].values
        for i in range(len(preds)):
            sk = "elite" if t3[i] == 1 else "depth"
            preds[i] *= star_calibration.get(sk, 1.0)

    return preds


def predict_with_intervals(
    models: dict,
    blend_weights: dict,
    X: pd.DataFrame,
    gate: Pipeline | None = None,
    meta_model: Ridge | None = None,
    output_isotonic: IsotonicRegression | None = None,
    n_bootstrap: int = 200,
    ci_level: float = 0.80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate point predictions with bootstrap confidence intervals.

    Uses model disagreement + feature perturbation to estimate uncertainty.
    Returns (point_predictions, lower_bound, upper_bound) at ci_level.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(X)
    bootstrap_preds = np.zeros((n_bootstrap, n_samples))
    alpha = (1 - ci_level) / 2

    model_names = list(models.keys())

    for b in range(n_bootstrap):
        X_noisy = X.copy()
        noise_cols = [c for c in X.columns if X[c].std() > 0]
        if noise_cols:
            noise_scale = 0.03
            noise = rng.normal(0, noise_scale, size=(n_samples, len(noise_cols)))
            col_stds = X_noisy[noise_cols].std().values
            col_stds = np.where(col_stds == 0, 1, col_stds)
            X_noisy[noise_cols] = X_noisy[noise_cols].values + noise * col_stds

        chosen_models = rng.choice(model_names, size=max(1, len(model_names) - 1), replace=False)

        raw_preds = {}
        for name in chosen_models:
            p_log = models[name].predict(X_noisy)
            raw_preds[name] = np.maximum(np.expm1(np.maximum(p_log, 0)), 0)

        gate_prob = None
        if gate is not None:
            try:
                gate_prob = gate.predict_proba(X_noisy)[:, 1]
            except Exception:
                gate_prob = None

        if meta_model is not None and gate_prob is not None and len(raw_preds) == len(model_names):
            meta_X = np.column_stack(
                [raw_preds[n] for n in model_names] + [gate_prob]
            )
            preds = np.maximum(meta_model.predict(meta_X), 0)
        else:
            preds = np.zeros(n_samples)
            total_w = 0.0
            for name in chosen_models:
                w = blend_weights.get(name, 1.0 / len(chosen_models))
                preds += w * raw_preds[name]
                total_w += w
            preds = preds / max(total_w, 1e-8)
            if gate_prob is not None:
                preds = gate_prob * preds

        bootstrap_preds[b] = preds

    point = np.median(bootstrap_preds, axis=0)
    lower = np.percentile(bootstrap_preds, alpha * 100, axis=0)
    upper = np.percentile(bootstrap_preds, (1 - alpha) * 100, axis=0)

    if output_isotonic is not None:
        point = np.maximum(
            output_isotonic.predict(point.reshape(-1, 1)).ravel(), 0.0
        )
        lower = np.maximum(
            output_isotonic.predict(lower.reshape(-1, 1)).ravel(), 0.0
        )
        upper = np.maximum(
            output_isotonic.predict(upper.reshape(-1, 1)).ravel(), 0.0
        )
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)

    return np.maximum(point, 0), np.maximum(lower, 0), np.maximum(upper, 0)


def get_feature_importance(models: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Extract and average feature importance across models."""
    importances = pd.DataFrame({"feature": feature_cols})

    for name, model in models.items():
        # Get the underlying estimator from the pipeline
        if hasattr(model, "named_steps"):
            est = model.named_steps.get("model")
        else:
            est = model

        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
            if len(imp) == len(feature_cols):
                importances[f"imp_{name}"] = imp
        elif hasattr(est, "coef_"):
            coef = np.abs(est.coef_)
            if len(coef) == len(feature_cols):
                importances[f"imp_{name}"] = coef / max(coef.max(), 1e-8)

    imp_cols = [c for c in importances.columns if c.startswith("imp_")]
    if imp_cols:
        importances["avg_importance"] = importances[imp_cols].mean(axis=1)
        importances = importances.sort_values("avg_importance", ascending=False)

    return importances
