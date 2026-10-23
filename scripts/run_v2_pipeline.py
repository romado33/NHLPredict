"""
V2 Pipeline: Collect new data, build advanced features, retrain, predict.

Usage:
    python scripts/run_v2_pipeline.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config.settings import (
    CURRENT_SEASON_END_YEAR,
    FIRST_SEASON,
    LAST_HISTORICAL_SEASON,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_STATE,
)
from src.utils.model_manifest import write_run_manifest
from src.data.nhl_api_client import NHLAPIClient
from src.data.collectors_v2 import (
    collect_skater_stats,
    collect_moneypuck_recency,
    collect_head_to_head,
)
from src.features.advanced import integrate_advanced_features
from src.features.elo import add_elo_features
from src.models.improved_trainer import (
    META_COLS,
    add_interaction_features,
    prepare_xy,
    select_features,
    prune_blend_for_worst_season_auc,
    temporal_cv_evaluate,
    temporal_cv_splits,
)
from src.models.bayesian import BayesianLogistic
from src.models.monte_carlo import simulate_series, simulate_bracket, print_bracket_results
from src.utils.logging_config import get_logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = get_logger("v2_pipeline")

SEASON = CURRENT_SEASON_END_YEAR


def _single_model_importance_vector(model: Any, n_features: int) -> np.ndarray | None:
    """Normalized importance vector aligned to training feature order (length n_features)."""
    try:
        if isinstance(model, BayesianLogistic):
            cal = getattr(model, "_calibrated", None)
            if cal is not None and getattr(cal, "calibrated_classifiers_", None):
                inner = cal.calibrated_classifiers_[0].estimator
                return _single_model_importance_vector(inner, n_features)

        est = model
        if hasattr(model, "named_steps") and "model" in model.named_steps:
            est = model.named_steps["model"]

        if hasattr(est, "coef_"):
            v = np.abs(np.ravel(est.coef_))
        elif hasattr(est, "feature_importances_"):
            v = np.asarray(est.feature_importances_, dtype=float).ravel()
        else:
            return None

        if len(v) != n_features:
            return None
        s = float(v.sum())
        return v / s if s > 0 else v
    except Exception:
        return None


def v2_blend_feature_importance(
    models: dict,
    selected: list[str],
    blend_weights: dict[str, float],
    top_n: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Blend per-model importances (|coef| or tree gain) using the same weights
    as series win-probability blending. Returns top_n features + raw weights dict.
    """
    n = len(selected)
    acc = np.zeros(n)
    wsum = 0.0

    for name, w in blend_weights.items():
        if w <= 0 or name not in models:
            continue
        vec = _single_model_importance_vector(models[name], n)
        if vec is None:
            continue
        acc += float(w) * vec
        wsum += float(w)

    if wsum > 0:
        acc /= wsum
    maxv = float(acc.max()) if len(acc) else 0.0
    if maxv > 0:
        acc = acc / maxv

    order = np.argsort(-acc)
    top = [
        {"feature": selected[i], "importance": round(float(acc[i]), 6)}
        for i in order[:top_n]
    ]
    return top, {k: float(v) for k, v in blend_weights.items()}


def _stress_slice_feature_audit(
    df: pd.DataFrame,
    X: pd.DataFrame,
    stress_years: tuple[int, ...] = (2023, 2024, 2025),
    top_k: int = 15,
) -> dict[str, Any]:
    """NaN rates and worst columns for recent seasons (rule-change / COVID stress slice)."""
    out: dict[str, Any] = {"stress_years": list(stress_years)}
    for yr in stress_years:
        idx = df.index[df["season"] == yr]
        if len(idx) == 0:
            continue
        sub = X.loc[idx]
        nan_frac = sub.isnull().mean().sort_values(ascending=False)
        out[str(yr)] = {
            "n_rows": int(len(idx)),
            "mean_row_nan_frac": float(sub.isnull().mean(axis=1).mean()),
            "worst_columns_by_nan_frac": [
                {"feature": str(i), "nan_frac": round(float(v), 4)}
                for i, v in nan_frac.head(top_k).items()
            ],
        }
    return out


def _mc_summary(mc: dict) -> dict[str, Any]:
    return {
        "series_win_probability_for_higher_seed": round(float(mc["series_win_prob"]), 6),
        "expected_games": round(float(mc["expected_games"]), 3),
        "probability_higher_seed_wins_in_4": round(float(mc["prob_sweep"]), 6),
        "probability_higher_seed_wins_in_5": round(float(mc["prob_5_games"]), 6),
        "probability_higher_seed_wins_in_6": round(float(mc["prob_6_games"]), 6),
        "probability_higher_seed_wins_in_7": round(float(mc["prob_7_games"]), 6),
        "n_simulations": int(mc.get("n_simulations", 0)),
    }


def step_collect_v2():
    """Collect new data sources needed for V2 features."""
    logger.info("=" * 60)
    logger.info("V2 DATA COLLECTION")
    logger.info("=" * 60)

    client = NHLAPIClient()

    # Skater stats (for player-level + injury proxy features)
    skater_path = RAW_DIR / "skater_stats.csv"
    if not skater_path.exists():
        logger.info("Collecting skater stats (this takes a while)...")
        collect_skater_stats(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)
    else:
        logger.info("Skater stats already cached")

    # MoneyPuck recency (last 20 games)
    recency_path = RAW_DIR / "team_stats_moneypuck_last20.csv"
    if not recency_path.exists():
        logger.info("Collecting MoneyPuck recency stats...")
        collect_moneypuck_recency()
    else:
        logger.info("MoneyPuck recency already cached")

    # Head-to-head records
    h2h_path = RAW_DIR / "head_to_head.csv"
    series_df = pd.read_csv(RAW_DIR / "playoff_series.csv")
    if not h2h_path.exists():
        logger.info("Collecting head-to-head records...")
        collect_head_to_head(client, series_df)
    else:
        logger.info("H2H data already cached")


def step_build_v2():
    """Build V2 feature matrix with all advanced features."""
    logger.info("=" * 60)
    logger.info("V2 FEATURE MATRIX BUILD")
    logger.info("=" * 60)

    # Load base feature matrix
    base_path = PROCESSED_DIR / "feature_matrix.csv"
    feature_df = pd.read_csv(base_path)
    series_df = pd.read_csv(RAW_DIR / "playoff_series.csv")
    standings_df = pd.read_csv(RAW_DIR / "standings.csv")

    # Load V2 data sources
    skater_df = _load_csv("skater_stats.csv")
    recency_df = _load_csv("team_stats_moneypuck_last20.csv")
    h2h_df = _load_csv("head_to_head.csv")

    # Add ELO features
    feature_df = add_elo_features(feature_df, series_df)

    # Add interaction features
    feature_df = add_interaction_features(feature_df)

    # Integrate all advanced features
    v2_df = integrate_advanced_features(
        feature_df, series_df, skater_df, recency_df, h2h_df, standings_df
    )

    return v2_df


def step_train_v2(df: pd.DataFrame) -> dict:
    """Train V2 models with all new features."""
    logger.info("=" * 60)
    logger.info("V2 MODEL TRAINING")
    logger.info("=" * 60)

    df = df.sort_values("season").reset_index(drop=True)

    X, y, feature_cols = prepare_xy(df)
    logger.info("Total V2 features: %d", len(feature_cols))

    # Drop columns that are entirely NaN (no signal, cause imputer warnings)
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        logger.info("Dropping %d all-NaN columns", len(all_nan_cols))
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    # Drop columns with >80% NaN (too sparse to be useful)
    high_nan = X.columns[X.isnull().mean() > 0.80].tolist()
    if high_nan:
        logger.info("Dropping %d columns with >80%% NaN", len(high_nan))
        X = X.drop(columns=high_nan)
        feature_cols = [c for c in feature_cols if c not in high_nan]

    logger.info("After NaN cleanup: %d features", len(X.columns))

    # Feature selection
    selected = select_features(X, y, max_features=70)
    X_sel = X[selected]
    logger.info("After selection: %d features", len(selected))

    # Sample weights (recent seasons more important)
    seasons = df["season"].values
    max_season = seasons.max()
    weights = np.array([0.92 ** (max_season - s) for s in seasons])

    # ── Train all model types ──

    models = {}

    # 1) L1 Logistic
    logger.info("Training L1 Logistic...")
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=RANDOM_STATE)),
    ])
    log_search = RandomizedSearchCV(
        log_pipe,
        {"model__C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]},
        n_iter=9, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE,
    )
    log_search.fit(X_sel, y, model__sample_weight=weights)
    models["v2_logistic"] = log_search.best_estimator_
    logger.info("Logistic CV AUC=%.4f (C=%.5f)", log_search.best_score_, log_search.best_params_["model__C"])

    # 2) XGBoost
    if xgb is not None:
        logger.info("Training XGBoost...")
        xgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)),
        ])
        xgb_search = RandomizedSearchCV(
            xgb_pipe,
            {
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
                "model__n_estimators": [150, 200, 300, 500],
                "model__subsample": [0.5, 0.6, 0.7, 0.8],
                "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7],
                "model__reg_alpha": [0.01, 0.1, 0.5, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "model__min_child_weight": [1, 3, 5, 7],
                "model__gamma": [0, 0.1, 0.3],
            },
            n_iter=300, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc",
            n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        xgb_search.fit(X_sel, y, model__sample_weight=weights)
        models["v2_xgboost"] = xgb_search.best_estimator_
        logger.info("XGBoost CV AUC=%.4f", xgb_search.best_score_)

    # 3) LightGBM
    if lgb is not None:
        logger.info("Training LightGBM...")
        lgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lgb.LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)),
        ])
        lgb_search = RandomizedSearchCV(
            lgb_pipe,
            {
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
                "model__n_estimators": [150, 200, 300, 500],
                "model__num_leaves": [7, 11, 15, 21, 31],
                "model__subsample": [0.5, 0.6, 0.7, 0.8],
                "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7],
                "model__reg_alpha": [0.01, 0.1, 0.5, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "model__min_child_samples": [3, 5, 10, 20],
            },
            n_iter=300, cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc",
            n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
        )
        lgb_search.fit(X_sel, y, model__sample_weight=weights)
        models["v2_lightgbm"] = lgb_search.best_estimator_
        logger.info("LightGBM CV AUC=%.4f", lgb_search.best_score_)

    # 4) Bayesian logistic
    logger.info("Training Bayesian Logistic...")
    bayes = BayesianLogistic(n_calibration_folds=5)
    bayes.fit(X_sel, y)
    models["v2_bayesian"] = bayes
    logger.info("Bayesian model trained")

    # ── Temporal CV evaluation ──
    print(f"\n{'='*70}")
    print("V2 TEMPORAL CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")

    best_auc = 0.0
    best_name = ""
    all_results = {}

    for name, model in models.items():
        metrics = temporal_cv_evaluate(model, X_sel, y, df, min_train_years=8)
        all_results[name] = metrics
        season_df = metrics["season_results"]

        print(f"\n  {name}:")
        print(f"    Accuracy:    {metrics['accuracy']:.4f}")
        print(f"    Log Loss:    {metrics['log_loss']:.4f}")
        print(f"    ROC AUC:     {metrics['roc_auc']:.4f}")
        print(f"    Brier Score: {metrics['brier_score']:.4f}")
        print(f"    Mean season accuracy: {season_df['accuracy'].mean():.4f}")
        print(f"    Mean season AUC:      {season_df['auc'].mean():.4f}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name

    print(f"\n{'*'*70}")
    print(f"  Best V2 model: {best_name} (AUC={best_auc:.4f})")
    print(f"{'*'*70}")

    # ── Optimal blending (calibration-aware: AUC + Brier + log loss) ──
    blend_weights = _optimize_blend(models, X_sel, y, df)
    blend_weights, _blend_eval = prune_blend_for_worst_season_auc(
        models, blend_weights, X_sel, y, df, min_train_years=8,
    )
    print(f"  Blend weights after worst-season prune: {blend_weights}")

    # ── Stress-slice audit (2023–2025): feature NaN profile ──
    try:
        audit = _stress_slice_feature_audit(df, X_sel)
        audit_path = PROCESSED_DIR / "v2_stress_slice_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "audit": audit,
                },
                f,
                indent=2,
            )
        logger.info("Wrote %s", audit_path)
    except Exception as e:
        logger.warning("Stress-slice audit failed: %s", e)

    # ── Season-by-season for best model ──
    if best_name in all_results:
        season_df = all_results[best_name]["season_results"]
        print(f"\nSeason-by-Season ({best_name}):")
        print(season_df.to_string(index=False))
        season_df.to_csv(PROCESSED_DIR / "v2_season_accuracy.csv", index=False)

    try:
        eval_blob: dict[str, Any] = {}
        for name, metrics in all_results.items():
            d: dict[str, Any] = {}
            for k, v in metrics.items():
                if k == "season_results" and hasattr(v, "to_dict"):
                    d[k] = v.to_dict(orient="records")
                elif isinstance(v, (float, int, np.floating, np.integer)):
                    d[k] = float(v)
                else:
                    d[k] = v
            eval_blob[name] = d
        with open(PROCESSED_DIR / "v2_temporal_eval_full.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "models": eval_blob,
                },
                f,
                indent=2,
            )
        logger.info("Wrote %s", PROCESSED_DIR / "v2_temporal_eval_full.json")
    except Exception as e:
        logger.warning("Could not save v2_temporal_eval_full.json: %s", e)

    # Train final models on all data
    for name, model in models.items():
        model.fit(X_sel, y)

    # Save
    joblib.dump(models, MODELS_DIR / "v2_models.joblib")
    joblib.dump(selected, MODELS_DIR / "v2_features.joblib")
    joblib.dump(blend_weights, MODELS_DIR / "v2_blend_weights.joblib")

    return {
        "models": models,
        "selected": selected,
        "blend_weights": blend_weights,
        "best_name": best_name,
        "best_auc": best_auc,
        "X": X_sel,
        "y": y,
        "df": df,
    }


def _optimize_blend(models, X, y, df):
    """Find optimal model blend weights via temporal CV."""
    splits = temporal_cv_splits(df, min_train_years=8)
    model_names = [n for n in models if models[n] is not None]
    probas = {n: np.full(len(y), np.nan) for n in model_names}

    for train_idx, test_idx in splits:
        for name in model_names:
            model = models[name]
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            p = model.predict_proba(X.iloc[test_idx])[:, 1]
            probas[name][test_idx] = p

    valid = np.all([~np.isnan(probas[n]) for n in model_names], axis=0)
    y_v = y[valid].values
    p_v = {n: probas[n][valid] for n in model_names}

    # Maximize discrimination + calibration (lower Brier is better)
    best_score = -1.0
    best_auc = 0.0
    best_brier = 1.0
    best_w = {n: 1.0 / len(model_names) for n in model_names}

    for _ in range(5000):
        raw = np.random.dirichlet(np.ones(len(model_names)))
        ws = {model_names[i]: raw[i] for i in range(len(model_names))}
        blended = sum(ws[n] * p_v[n] for n in model_names)
        try:
            auc = roc_auc_score(y_v, blended)
            brier = brier_score_loss(y_v, blended)
            ll = log_loss(y_v, blended, labels=[0, 1])
            ll_norm = 1.0 - min(float(ll) / 1.5, 1.0)
            score = 0.38 * auc + 0.32 * (1.0 - brier) + 0.30 * ll_norm
            if score > best_score:
                best_score = score
                best_auc = auc
                best_brier = brier
                best_w = ws
        except ValueError:
            continue

    print(f"\n  Optimal blend: score={best_score:.4f} (0.38*AUC + 0.32*(1-Brier) + 0.30*norm_logloss)")
    print(f"    AUC={best_auc:.4f}  Brier={best_brier:.4f}")
    print(f"  Weights: {', '.join(f'{n}={w:.3f}' for n, w in best_w.items())}")
    return best_w


def step_predict_v2(train_result: dict):
    """Generate V2 predictions for current season using proper NHL bracket format."""
    logger.info("=" * 60)
    logger.info("V2 PREDICTIONS FOR %d", SEASON)
    logger.info("=" * 60)

    models = train_result["models"]
    selected = train_result["selected"]
    blend_weights = train_result["blend_weights"]

    from src.data.collectors import (
        collect_standings, collect_team_stats_nhl,
        collect_moneypuck_team_stats, collect_goalie_stats,
    )
    from src.data.collectors_v2 import collect_moneypuck_recency, collect_skater_stats
    from src.features.engineer import (
        _build_team_season_features, _compute_differential, _get_team_features,
    )
    from src.features.advanced import (
        build_injury_proxy,
        build_player_features,
        build_recency_features,
        build_top_skater_strength_features,
    )

    client = NHLAPIClient(cache_ttl_hours=0)
    standings = collect_standings(client, SEASON, SEASON)
    nhl_stats = collect_team_stats_nhl(client, SEASON, SEASON)
    moneypuck = collect_moneypuck_team_stats(SEASON, SEASON)
    goalies = collect_goalie_stats(client, SEASON, SEASON)
    recency = collect_moneypuck_recency(SEASON, SEASON)

    team_features = _build_team_season_features(standings, nhl_stats, moneypuck, goalies)
    if team_features.empty:
        logger.error("Could not build current season features")
        return

    skater_current = collect_skater_stats(client, SEASON, SEASON)
    plyr_feats = build_player_features(skater_current)
    skel_feats = build_top_skater_strength_features(skater_current)
    injury_feats = build_injury_proxy(skater_current)
    recency_feats = build_recency_features(recency)

    for extra_df in [plyr_feats, skel_feats, injury_feats, recency_feats]:
        if not extra_df.empty and "team_abbrev" in extra_df.columns:
            team_features = team_features.merge(
                extra_df, on=["team_abbrev", "season"], how="left", suffixes=("", "_extra")
            )
            team_features = team_features.loc[:, ~team_features.columns.duplicated()]

    # ── Build proper NHL division-based bracket ──
    # NHL format (since 2013-14):
    #   Each division sends top 3 teams. Two wild cards per conference.
    #   Division winners (D1) face wild cards.
    #   D2 vs D3 play within the same division.
    #   The better division winner gets the weaker wild card.
    stnd = standings[standings["season"] == SEASON].copy()

    def _extract_abbrev(x):
        if isinstance(x, dict):
            return x.get("default", "")
        if isinstance(x, str) and "{" in x:
            try:
                return eval(x).get("default", "")
            except Exception:
                return x
        return str(x)

    stnd["team"] = stnd["teamAbbrev"].apply(_extract_abbrev)

    def _build_conference_bracket(conf_df):
        """
        Build NHL-accurate bracket for one conference using division seeding.
        Returns list of (higher_seed, lower_seed) tuples for Round 1.
        """
        divs = sorted(conf_df["divisionAbbrev"].unique())
        div_teams = {}
        div_winners = {}

        for div in divs:
            div_df = conf_df[conf_df["divisionAbbrev"] == div].sort_values(
                "divisionSequence"
            )
            teams = div_df["team"].tolist()
            div_teams[div] = teams[:3]  # top 3 qualify directly
            div_winners[div] = teams[0] if teams else ""

        # Wild cards: conference rank 7 and 8 (not in top-3 of either division)
        all_div_qualified = set()
        for teams in div_teams.values():
            all_div_qualified.update(teams)

        wild_cards = conf_df[
            ~conf_df["team"].isin(all_div_qualified)
        ].sort_values("conferenceSequence")["team"].head(2).tolist()

        # Determine which division winner is better (lower conferenceSequence)
        div_winner_ranks = {}
        for div, winner in div_winners.items():
            row = conf_df[conf_df["team"] == winner]
            rank = row["conferenceSequence"].values[0] if not row.empty else 99
            div_winner_ranks[div] = rank

        sorted_divs = sorted(divs, key=lambda d: div_winner_ranks.get(d, 99))
        better_div = sorted_divs[0]
        worse_div = sorted_divs[1]

        # WC1 = higher wild card, WC2 = lower wild card
        wc1 = wild_cards[0] if len(wild_cards) > 0 else ""
        wc2 = wild_cards[1] if len(wild_cards) > 1 else ""

        # Better division winner gets the weaker wild card (WC2)
        # Worse division winner gets the stronger wild card (WC1)
        matchups = [
            (div_winners[better_div], wc2),   # D1 of better div vs WC2
            (div_teams[better_div][1], div_teams[better_div][2]),  # D2 vs D3 within better div
            (div_winners[worse_div], wc1),    # D1 of worse div vs WC1
            (div_teams[worse_div][1], div_teams[worse_div][2]),   # D2 vs D3 within worse div
        ]

        bracket_info = {
            "div_teams": div_teams,
            "div_winners": div_winners,
            "wild_cards": wild_cards,
            "better_div": better_div,
            "worse_div": worse_div,
        }

        return matchups, bracket_info

    east_df = stnd[stnd["conferenceAbbrev"] == "E"].copy()
    west_df = stnd[stnd["conferenceAbbrev"] == "W"].copy()

    east_matchups, east_info = _build_conference_bracket(east_df)
    west_matchups, west_info = _build_conference_bracket(west_df)

    def _get_stnd_val(team_name, col):
        row = stnd[stnd["team"] == team_name]
        return row[col].values[0] if not row.empty and col in row.columns else "?"

    def _get_seed_label(team_name, bracket_info):
        for div, teams in bracket_info["div_teams"].items():
            if team_name in teams:
                pos = teams.index(team_name) + 1
                return f"D{pos}"
        if team_name in bracket_info["wild_cards"]:
            wc_pos = bracket_info["wild_cards"].index(team_name) + 1
            return f"WC{wc_pos}"
        return "?"

    def _print_bracket(conf_name, matchups, bracket_info):
        print(f"\n{'='*70}")
        print(f"  {conf_name} CONFERENCE BRACKET")
        print(f"{'='*70}")
        for div in sorted(bracket_info["div_teams"]):
            div_name = {"A": "Atlantic", "M": "Metropolitan", "C": "Central", "P": "Pacific"}.get(div, div)
            print(f"  {div_name}: ", end="")
            for i, t in enumerate(bracket_info["div_teams"][div]):
                pts = _get_stnd_val(t, "points")
                print(f"  {t}(D{i+1},{pts}pts)", end="")
            print()
        wc = bracket_info["wild_cards"]
        print(f"  Wild Cards:", end="")
        for i, t in enumerate(wc):
            pts = _get_stnd_val(t, "points")
            print(f"  {t}(WC{i+1},{pts}pts)", end="")
        print()
        print(f"\n  First Round Matchups:")
        for hi, lo in matchups:
            hi_lbl = _get_seed_label(hi, bracket_info)
            lo_lbl = _get_seed_label(lo, bracket_info)
            print(f"    {hi} ({hi_lbl}) vs {lo} ({lo_lbl})")

    _print_bracket("EASTERN", east_matchups, east_info)
    _print_bracket("WESTERN", west_matchups, west_info)

    def predict_series_blended(hi, lo):
        hi_feats = _get_team_features(team_features, hi, SEASON)
        lo_feats = _get_team_features(team_features, lo, SEASON)
        if hi_feats is None or lo_feats is None:
            return 0.5
        diff = _compute_differential(hi_feats, lo_feats, prefix="diff_")
        diff["is_bubble_year"] = 0
        diff["is_shortened_season"] = 0
        diff.setdefault("diff_elo_rating", 0)
        diff.setdefault("elo_expected_win", 0.5)
        X_pred = pd.DataFrame([diff])
        for col in selected:
            if col not in X_pred.columns:
                X_pred[col] = 0.0
        X_pred = X_pred[selected]

        blended = 0.0
        total_w = 0.0
        for name, model in models.items():
            w = blend_weights.get(name, 0)
            if w <= 0:
                continue
            try:
                p = model.predict_proba(X_pred)[0][1]
                blended += w * p
                total_w += w
            except Exception:
                continue
        return blended / total_w if total_w > 0 else 0.5

    series_summaries: list[dict[str, Any]] = []

    def _print_series(
        hi: str,
        lo: str,
        base_prob: float,
        mc: dict,
        *,
        round_name: str,
        conference: str,
    ) -> str:
        winner = hi if mc["series_win_prob"] >= 0.5 else lo
        conf = mc["series_win_prob"] * 100 if winner == hi else (1 - mc["series_win_prob"]) * 100
        print(f"\n  {hi} vs {lo}")
        print(f"    Predicted winner: {winner} ({conf:.1f}%)")
        print(f"    Model prob: {base_prob*100:.1f}%  |  MC series prob: {mc['series_win_prob']*100:.1f}%")
        print(f"    In 4: {mc['prob_sweep']*100:.0f}%  In 5: {mc['prob_5_games']*100:.0f}%  "
              f"In 6: {mc['prob_6_games']*100:.0f}%  In 7: {mc['prob_7_games']*100:.0f}%")
        print(f"    Expected games: {mc['expected_games']:.1f}")
        series_summaries.append(
            {
                "round": round_name,
                "conference": conference,
                "higher_seed": hi,
                "lower_seed": lo,
                "predicted_winner": winner,
                "confidence_in_predicted_winner_pct": round(float(conf), 2),
                "blended_model_prob_higher_seed_wins_series": round(float(base_prob), 6),
                "monte_carlo": _mc_summary(mc),
            }
        )
        return winner

    N_SIM = 50000

    # ── ROUND 1 ──
    print(f"\n{'='*70}")
    print(f"  ROUND 1: EASTERN CONFERENCE")
    print(f"{'='*70}")
    east_r1_winners = []
    east_r1_probs = []
    for hi, lo in east_matchups:
        bp = predict_series_blended(hi, lo)
        mc = simulate_series(bp, N_SIM)
        w = _print_series(hi, lo, bp, mc, round_name="First Round", conference="Eastern")
        east_r1_winners.append(w)
        east_r1_probs.append(mc["series_win_prob"] if w == hi else 1 - mc["series_win_prob"])

    print(f"\n{'='*70}")
    print(f"  ROUND 1: WESTERN CONFERENCE")
    print(f"{'='*70}")
    west_r1_winners = []
    west_r1_probs = []
    for hi, lo in west_matchups:
        bp = predict_series_blended(hi, lo)
        mc = simulate_series(bp, N_SIM)
        w = _print_series(hi, lo, bp, mc, round_name="First Round", conference="Western")
        west_r1_winners.append(w)
        west_r1_probs.append(mc["series_win_prob"] if w == hi else 1 - mc["series_win_prob"])

    # ── ROUND 2 ──
    # NHL bracket sides: matchups 0,1 are one side (better div), matchups 2,3 are the other (worse div)
    # Winner of matchup 0 (D1 vs WC) plays winner of matchup 1 (D2 vs D3) on same side
    east_r2 = [(east_r1_winners[0], east_r1_winners[1]), (east_r1_winners[2], east_r1_winners[3])]
    west_r2 = [(west_r1_winners[0], west_r1_winners[1]), (west_r1_winners[2], west_r1_winners[3])]

    print(f"\n{'='*70}")
    print(f"  ROUND 2: EASTERN CONFERENCE")
    print(f"{'='*70}")
    east_r2_winners = []
    for hi, lo in east_r2:
        bp = predict_series_blended(hi, lo)
        mc = simulate_series(bp, N_SIM)
        w = _print_series(hi, lo, bp, mc, round_name="Second Round", conference="Eastern")
        east_r2_winners.append(w)

    print(f"\n{'='*70}")
    print(f"  ROUND 2: WESTERN CONFERENCE")
    print(f"{'='*70}")
    west_r2_winners = []
    for hi, lo in west_r2:
        bp = predict_series_blended(hi, lo)
        mc = simulate_series(bp, N_SIM)
        w = _print_series(hi, lo, bp, mc, round_name="Second Round", conference="Western")
        west_r2_winners.append(w)

    # ── CONFERENCE FINALS ──
    print(f"\n{'='*70}")
    print(f"  CONFERENCE FINALS")
    print(f"{'='*70}")
    ecf_bp = predict_series_blended(east_r2_winners[0], east_r2_winners[1])
    ecf_mc = simulate_series(ecf_bp, N_SIM)
    east_champ = _print_series(
        east_r2_winners[0], east_r2_winners[1], ecf_bp, ecf_mc,
        round_name="Conference Finals", conference="Eastern",
    )

    wcf_bp = predict_series_blended(west_r2_winners[0], west_r2_winners[1])
    wcf_mc = simulate_series(wcf_bp, N_SIM)
    west_champ = _print_series(
        west_r2_winners[0], west_r2_winners[1], wcf_bp, wcf_mc,
        round_name="Conference Finals", conference="Western",
    )

    # ── STANLEY CUP FINAL ──
    print(f"\n{'='*70}")
    print(f"  STANLEY CUP FINAL")
    print(f"{'='*70}")
    scf_bp = predict_series_blended(east_champ, west_champ)
    scf_mc = simulate_series(scf_bp, N_SIM)
    champion = _print_series(
        east_champ, west_champ, scf_bp, scf_mc,
        round_name="Stanley Cup Final", conference="NHL",
    )

    print(f"\n{'*'*70}")
    print(f"  PREDICTED 2026 STANLEY CUP CHAMPION: {champion}")
    scf_conf = scf_mc["series_win_prob"] * 100 if champion == east_champ else (1 - scf_mc["series_win_prob"]) * 100
    print(f"  Win probability: {scf_conf:.1f}%")
    print(f"{'*'*70}")

    # ── Bayesian uncertainty for key matchups ──
    bayes_model = models.get("v2_bayesian")
    if bayes_model is not None and hasattr(bayes_model, "predict_with_uncertainty"):
        print(f"\n{'='*70}")
        print(f"  BAYESIAN UNCERTAINTY ANALYSIS (90% credible intervals)")
        print(f"{'='*70}")
        for hi, lo in east_matchups + west_matchups:
            hi_feats = _get_team_features(team_features, hi, SEASON)
            lo_feats = _get_team_features(team_features, lo, SEASON)
            if hi_feats is None or lo_feats is None:
                continue
            diff = _compute_differential(hi_feats, lo_feats, prefix="diff_")
            diff["is_bubble_year"] = 0
            diff["is_shortened_season"] = 0
            diff.setdefault("diff_elo_rating", 0)
            diff.setdefault("elo_expected_win", 0.5)
            X_u = pd.DataFrame([diff])
            for col in selected:
                if col not in X_u.columns:
                    X_u[col] = 0.0
            X_u = X_u[selected]
            unc = bayes_model.predict_with_uncertainty(X_u)
            print(f"  {hi} vs {lo}: "
                  f"{hi} win prob = {unc['mean'][0]*100:.1f}% "
                  f"(90% CI: {unc['ci_low'][0]*100:.1f}%-{unc['ci_high'][0]*100:.1f}%)")

    top_features, blend_w_saved = v2_blend_feature_importance(
        models, selected, blend_weights, top_n=25
    )
    print(f"\n{'='*70}")
    print(f"  TOP FEATURES (V2 blend, same weights as series predictions)")
    print(f"{'='*70}")
    for row in top_features[:15]:
        print(f"    {row['feature']:45s}  {row['importance']:.4f}")

    # Save predictions (matchups, winners, per-series games distribution, feature importance)
    from src.utils.model_manifest import build_v2_model_manifest

    preds = {
        "season_end_year": SEASON,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_manifest": build_v2_model_manifest(),
        "east_r1": east_matchups,
        "west_r1": west_matchups,
        "east_r1_winners": east_r1_winners,
        "west_r1_winners": west_r1_winners,
        "east_r2_winners": east_r2_winners,
        "west_r2_winners": west_r2_winners,
        "east_champ": east_champ,
        "west_champ": west_champ,
        "champion": champion,
        "series": series_summaries,
        "top_features": top_features,
        "blend_weights": blend_w_saved,
        "feature_importance_method": (
            "Per submodel: normalized abs(logistic coef) or tree feature_importances_; "
            "blended with the same weights as win-probability; global importance scaled to max=1."
        ),
    }
    out_path = PROCESSED_DIR / "predictions_v2_2026.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    logger.info("Wrote %s", out_path)

    try:
        import subprocess

        export_py = Path(__file__).resolve().parent / "export_v2_predictions_md.py"
        subprocess.run(
            [sys.executable, str(export_py)],
            cwd=str(Path(__file__).resolve().parent.parent),
            check=False,
        )
    except Exception as e:
        logger.warning("Markdown export skipped: %s", e)


def _load_csv(filename):
    path = RAW_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip collect/build/train, just re-predict with saved models")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  NHL PLAYOFF PREDICTOR V2 - FULL PIPELINE")
    print(f"{'#'*70}")

    if args.predict_only:
        models = joblib.load(MODELS_DIR / "v2_models.joblib")
        selected = joblib.load(MODELS_DIR / "v2_features.joblib")
        blend_weights = joblib.load(MODELS_DIR / "v2_blend_weights.joblib")
        result = {"models": models, "selected": selected, "blend_weights": blend_weights}
        step_predict_v2(result)
    else:
        step_collect_v2()
        v2_df = step_build_v2()
        result = step_train_v2(v2_df)
        step_predict_v2(result)

    write_run_manifest(extra={"pipeline": "v2", "mode": "predict_only" if args.predict_only else "full"})

    print(f"\n{'#'*70}")
    print(f"  V2 PIPELINE COMPLETE")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
