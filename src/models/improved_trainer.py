"""
Improved training pipeline with:
1. Temporal (expanding-window) cross-validation — no future data leakage
2. Feature selection integrated into the pipeline
3. Better hyperparameter search ranges informed by round-1 results
4. Interaction and derived features
5. Stacking ensemble
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


LABEL_COL = "higher_seed_won"
META_COLS = [
    "season", "round", "higher_seed", "lower_seed",
    "higher_seed_won", "is_bubble_year", "is_shortened_season",
]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [
        c for c in df.columns
        if c not in META_COLS and df[c].dtype in (np.float64, np.int64, float, int)
    ]
    X = df[feature_cols].copy()
    y = df[LABEL_COL].copy()
    return X, y, feature_cols


# ── Temporal Cross-Validation ───────────────────────────────────────────────

def temporal_cv_splits(
    df: pd.DataFrame, min_train_years: int = 8
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window temporal CV.
    Train on seasons [first..year-1], test on season [year].
    """
    seasons = sorted(df["season"].unique())
    splits = []

    for i in range(min_train_years, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df[df["season"].isin(train_seasons)].index.values
        test_idx = df[df["season"] == test_season].index.values

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    logger.info(
        "Temporal CV: %d splits (train from %d seasons, test 1 season each)",
        len(splits), min_train_years,
    )
    return splits


def temporal_cv_evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    min_train_years: int = 8,
) -> dict[str, Any]:
    """
    Evaluate model with temporal CV, returning metrics and per-season breakdown.
    """
    splits = temporal_cv_splits(df, min_train_years)
    seasons = sorted(df["season"].unique())

    all_preds = np.full(len(y), np.nan)
    all_probas = np.full(len(y), np.nan)
    season_results = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        all_probas[test_idx] = proba
        all_preds[test_idx] = preds

        test_season = df.iloc[test_idx]["season"].iloc[0]
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = np.nan

        season_results.append({
            "season": test_season,
            "n_series": len(test_idx),
            "accuracy": acc,
            "auc": auc,
        })

    valid_mask = ~np.isnan(all_probas)
    y_valid = y[valid_mask]
    probas_valid = all_probas[valid_mask]
    preds_valid = all_preds[valid_mask].astype(int)

    metrics = {
        "accuracy": accuracy_score(y_valid, preds_valid),
        "log_loss": log_loss(y_valid, probas_valid),
        "roc_auc": roc_auc_score(y_valid, probas_valid),
        "brier_score": brier_score_loss(y_valid, probas_valid),
        "n_test_samples": int(valid_mask.sum()),
        "season_results": pd.DataFrame(season_results),
    }

    return metrics


def temporal_cv_blend_evaluate(
    models: dict[str, Any],
    blend_weights: dict[str, float],
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    min_train_years: int = 8,
) -> dict[str, Any]:
    """
    Temporal CV for a fixed convex blend of models (same splits as temporal_cv_evaluate).
    """
    active = {n: w for n, w in blend_weights.items() if w > 0 and n in models}
    sw = sum(active.values())
    if sw <= 0:
        raise ValueError("temporal_cv_blend_evaluate: no positive weights for known models")
    norm_w = {n: w / sw for n, w in active.items()}

    splits = temporal_cv_splits(df, min_train_years)
    all_preds = np.full(len(y), np.nan)
    all_probas = np.full(len(y), np.nan)
    season_results = []

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        blended = np.zeros(len(X_test))
        for name, w in norm_w.items():
            m = models[name]
            m.fit(X_train, y_train)
            blended += w * m.predict_proba(X_test)[:, 1]

        proba = blended
        preds = (proba >= 0.5).astype(int)

        all_probas[test_idx] = proba
        all_preds[test_idx] = preds

        test_season = df.iloc[test_idx]["season"].iloc[0]
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = np.nan

        season_results.append({
            "season": test_season,
            "n_series": len(test_idx),
            "accuracy": acc,
            "auc": auc,
        })

    valid_mask = ~np.isnan(all_probas)
    y_valid = y[valid_mask]
    probas_valid = all_probas[valid_mask]
    preds_valid = all_preds[valid_mask].astype(int)

    return {
        "accuracy": accuracy_score(y_valid, preds_valid),
        "log_loss": log_loss(y_valid, probas_valid),
        "roc_auc": roc_auc_score(y_valid, probas_valid),
        "brier_score": brier_score_loss(y_valid, probas_valid),
        "n_test_samples": int(valid_mask.sum()),
        "season_results": pd.DataFrame(season_results),
    }


def prune_blend_for_worst_season_auc(
    models: dict[str, Any],
    blend_weights: dict[str, float],
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    min_train_years: int = 8,
    min_improvement: float = 0.002,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Drop at most one ensemble member if renormalizing the blend improves
    the worst (minimum) per-season AUC — targets bad years, not average CV.
    """
    base = temporal_cv_blend_evaluate(
        models, blend_weights, X, y, df, min_train_years,
    )
    base_min = float(base["season_results"]["auc"].min())
    names = [n for n in blend_weights if blend_weights.get(n, 0) > 0 and n in models]
    if len(names) <= 1:
        return blend_weights, base

    best_w: dict[str, float] | None = None
    best_min = base_min
    best_eval: dict[str, Any] = base

    for drop in names:
        trial = {k: v for k, v in blend_weights.items() if k != drop and v > 0}
        s = sum(trial.values())
        if s <= 0:
            continue
        trial = {k: v / s for k, v in trial.items()}
        ev = temporal_cv_blend_evaluate(models, trial, X, y, df, min_train_years)
        mn = float(ev["season_results"]["auc"].min())
        if mn > best_min + min_improvement:
            best_min = mn
            best_w = trial
            best_eval = ev

    if best_w is None:
        return blend_weights, base

    logger.info(
        "Pruned blend: worst-season AUC (min over seasons) %.4f -> %.4f",
        float(base["season_results"]["auc"].min()),
        best_min,
    )
    return best_w, best_eval


# ── Add interaction features ────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create targeted interaction features based on domain knowledge
    of what matters in playoff hockey.
    """
    df = df.copy()

    # Offense vs Defense balance
    for prefix in ("diff_stnd_", "diff_nhl_"):
        gf_col = f"{prefix}goals_for_pg" if f"{prefix}goals_for_pg" in df.columns else None
        ga_col = f"{prefix}goals_against_pg" if f"{prefix}goals_against_pg" in df.columns else None
        if not gf_col:
            gf_col = next((c for c in df.columns if "goalsForPerGame" in c and c.startswith("diff_")), None)
        if not ga_col:
            ga_col = next((c for c in df.columns if "goalsAgainstPerGame" in c and c.startswith("diff_")), None)

    # Key playoff indicators: penalty kill × save percentage
    pk_col = next((c for c in df.columns if "penaltyKillPct" in c and c.startswith("diff_")), None)
    sv_col = next((c for c in df.columns if "starter_savePct" in c and c.startswith("diff_")), None)
    if pk_col and sv_col:
        df["interact_pk_x_sv"] = df[pk_col] * df[sv_col]

    # Goal differential × point percentage
    gd_col = next((c for c in df.columns if "goal_diff_pg" in c and c.startswith("diff_")), None)
    pp_col = next((c for c in df.columns if "point_pct" in c and c.startswith("diff_")), None)
    if gd_col and pp_col:
        df["interact_goaldiff_x_pctg"] = df[gd_col] * df[pp_col]

    # Pythagorean × conference rank
    pyth_col = next((c for c in df.columns if "pythag" in c and c.startswith("diff_")), None)
    rank_col = next((c for c in df.columns if "conference_rank" in c and c.startswith("diff_")), None)
    if pyth_col and rank_col:
        df["interact_pythag_x_rank"] = df[pyth_col] * df[rank_col]

    # Playoff experience × recent performance
    exp_col = next((c for c in df.columns if "playoff_appearances" in c), None)
    l10_col = next((c for c in df.columns if "l10_wins" in c and c.startswith("diff_")), None)
    if exp_col and l10_col:
        df["interact_exp_x_recent"] = df[exp_col] * df[l10_col]

    # xGoals percentage (MoneyPuck advanced)
    xg_col = next((c for c in df.columns if "xGoalsPercentage" in c and c.startswith("diff_")), None)
    corsi_col = next((c for c in df.columns if "corsiPercentage" in c and c.startswith("diff_")), None)
    if xg_col and corsi_col:
        df["interact_xg_x_corsi"] = df[xg_col] * df[corsi_col]

    # Road performance indicator (important for playoff series)
    road_col = next((c for c in df.columns if "road_pct" in c and c.startswith("diff_")), None)
    if road_col and gd_col:
        df["interact_road_x_goaldiff"] = df[road_col] * df[gd_col]

    n_new = sum(1 for c in df.columns if c.startswith("interact_"))
    logger.info("Added %d interaction features", n_new)
    return df


# ── Feature selection (variance + correlation + importance) ─────────────────

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 60,
) -> list[str]:
    """
    Multi-stage feature selection:
    1. Remove low-variance features
    2. Remove highly correlated features (keep the one with higher target correlation)
    3. Rank by gradient boosting importance
    4. Take top N
    """
    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 1) Variance filter
    variances = X_filled.var()
    low_var = variances[variances < 1e-6].index.tolist()
    X_filtered = X_filled.drop(columns=low_var)
    logger.info("Variance filter: dropped %d, kept %d", len(low_var), len(X_filtered.columns))

    # 2) Correlation filter
    corr_matrix = X_filtered.corr().abs()
    target_corr = X_filtered.corrwith(y).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > 0.90].tolist()
        for c in correlated:
            if target_corr.get(col, 0) >= target_corr.get(c, 0):
                drop_cols.add(c)
            else:
                drop_cols.add(col)
    X_decorr = X_filtered.drop(columns=list(drop_cols))
    logger.info("Correlation filter: dropped %d, kept %d", len(drop_cols), len(X_decorr.columns))

    # 3) Gradient boosting importance ranking
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    gb.fit(X_decorr, y)
    importances = pd.Series(gb.feature_importances_, index=X_decorr.columns)
    importances = importances.sort_values(ascending=False)

    selected = importances.head(max_features).index.tolist()
    logger.info("Importance selection: kept top %d features", len(selected))

    print(f"\nTop 20 selected features:")
    for i, (feat, imp) in enumerate(importances.head(20).items()):
        print(f"  {i+1:2d}. {feat:<55s} {imp:.5f}")

    return selected


# ── Improved training with temporal CV ──────────────────────────────────────

def train_improved_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1", solver="saga",
            max_iter=5000, random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__C": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    }
    cv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y)
    logger.info("Improved Logistic: C=%.5f  CV AUC=%.4f", search.best_params_["model__C"], search.best_score_)
    best = search.best_estimator_
    joblib.dump(best, MODELS_DIR / "improved_logistic.joblib")
    return best


def train_improved_xgboost(X: pd.DataFrame, y: pd.Series) -> Pipeline | None:
    if not HAS_XGB:
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.005, 0.01, 0.03, 0.05],
        "model__n_estimators": [100, 200, 300, 500],
        "model__subsample": [0.6, 0.7, 0.8],
        "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "model__reg_alpha": [0.01, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
        "model__min_child_weight": [1, 3, 5],
    }

    cv = TimeSeriesSplit(n_splits=5)
    from sklearn.model_selection import RandomizedSearchCV
    search = RandomizedSearchCV(
        pipe, param_grid, n_iter=200, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X, y)
    logger.info("Improved XGBoost: params=%s  CV AUC=%.4f", search.best_params_, search.best_score_)
    best = search.best_estimator_
    joblib.dump(best, MODELS_DIR / "improved_xgboost.joblib")
    return best


def train_improved_lightgbm(X: pd.DataFrame, y: pd.Series) -> Pipeline | None:
    if not HAS_LGB:
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMClassifier(
            verbose=-1, random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.005, 0.01, 0.03, 0.05],
        "model__n_estimators": [100, 200, 300, 500],
        "model__num_leaves": [7, 15, 31],
        "model__subsample": [0.6, 0.7, 0.8],
        "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "model__reg_alpha": [0.01, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
        "model__min_child_samples": [5, 10, 20],
    }

    cv = TimeSeriesSplit(n_splits=5)
    from sklearn.model_selection import RandomizedSearchCV
    search = RandomizedSearchCV(
        pipe, param_grid, n_iter=200, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X, y)
    logger.info("Improved LightGBM: params=%s  CV AUC=%.4f", search.best_params_, search.best_score_)
    best = search.best_estimator_
    joblib.dump(best, MODELS_DIR / "improved_lightgbm.joblib")
    return best


def train_stacking_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    logistic: Pipeline | None = None,
    xgb_model: Pipeline | None = None,
    lgb_model: Pipeline | None = None,
) -> Any:
    """
    Stacking ensemble: base models' predictions fed into a logistic meta-learner.
    """
    estimators = []
    if logistic:
        estimators.append(("logistic", logistic))
    if xgb_model:
        estimators.append(("xgboost", xgb_model))
    if lgb_model:
        estimators.append(("lightgbm", lgb_model))

    if len(estimators) < 2:
        return estimators[0][1] if estimators else logistic

    from sklearn.model_selection import StratifiedKFold
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        cv=StratifiedKFold(n_splits=5, shuffle=False),
        stack_method="predict_proba",
        passthrough=False,
    )
    stack.fit(X, y)

    calibrated = CalibratedClassifierCV(
        stack, cv=StratifiedKFold(n_splits=3, shuffle=False), method="isotonic"
    )
    calibrated.fit(X, y)

    joblib.dump(calibrated, MODELS_DIR / "improved_ensemble.joblib")
    logger.info("Stacking ensemble saved")
    return calibrated


# ── Full improved pipeline ──────────────────────────────────────────────────

def train_improved(df: pd.DataFrame) -> dict[str, Any]:
    """
    Full improved training pipeline:
    1. Add interaction features
    2. Feature selection
    3. Train all models with temporal CV
    4. Build stacking ensemble
    5. Evaluate with proper temporal validation
    """
    logger.info("="*60)
    logger.info("IMPROVED TRAINING PIPELINE")
    logger.info("="*60)

    # Sort by season for temporal consistency
    df = df.sort_values("season").reset_index(drop=True)

    # Add interactions
    df = add_interaction_features(df)

    # Prepare features
    X, y, feature_cols = prepare_xy(df)
    logger.info("Before selection: %d features", len(feature_cols))

    # Feature selection
    selected_features = select_features(X, y, max_features=60)
    X_sel = X[selected_features]
    logger.info("After selection: %d features", len(selected_features))

    # Save selected feature matrix
    keep_meta = [c for c in META_COLS if c in df.columns]
    keep_feats = [c for c in selected_features if c in df.columns and c not in keep_meta]
    df_selected = df[keep_meta + keep_feats].copy()
    df_selected = df_selected.loc[:, ~df_selected.columns.duplicated()]
    df_selected.to_csv(PROCESSED_DIR / "feature_matrix_improved.csv", index=False)

    # Train individual models
    print(f"\n{'='*60}")
    print("Training Improved Logistic Regression (L1)...")
    print(f"{'='*60}")
    logistic = train_improved_logistic(X_sel, y)

    print(f"\n{'='*60}")
    print("Training Improved XGBoost...")
    print(f"{'='*60}")
    xgb_model = train_improved_xgboost(X_sel, y)

    print(f"\n{'='*60}")
    print("Training Improved LightGBM...")
    print(f"{'='*60}")
    lgb_model = train_improved_lightgbm(X_sel, y)

    print(f"\n{'='*60}")
    print("Training Stacking Ensemble...")
    print(f"{'='*60}")
    ensemble = train_stacking_ensemble(X_sel, y, logistic, xgb_model, lgb_model)

    # Evaluate all with temporal CV
    print(f"\n{'='*60}")
    print("TEMPORAL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")

    models = {
        "improved_logistic": logistic,
        "improved_xgboost": xgb_model,
        "improved_lightgbm": lgb_model,
        "improved_ensemble": ensemble,
    }

    best_auc = 0.0
    best_name = ""
    results = {}

    for name, model in models.items():
        if model is None:
            continue
        metrics = temporal_cv_evaluate(model, X_sel, y, df, min_train_years=8)
        results[name] = metrics

        print(f"\n  {name}:")
        print(f"    Accuracy:    {metrics['accuracy']:.4f}")
        print(f"    Log Loss:    {metrics['log_loss']:.4f}")
        print(f"    ROC AUC:     {metrics['roc_auc']:.4f}")
        print(f"    Brier Score: {metrics['brier_score']:.4f}")
        print(f"    Test samples: {metrics['n_test_samples']}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name

    print(f"\n{'*'*60}")
    print(f"  Best model: {best_name} (AUC={best_auc:.4f})")
    print(f"{'*'*60}")

    # Season-by-season for best model
    if best_name in results:
        season_df = results[best_name]["season_results"]
        print(f"\n{'='*60}")
        print(f"Season-by-Season Accuracy ({best_name})")
        print(f"{'='*60}")
        print(season_df.to_string(index=False))
        print(f"\nMean accuracy: {season_df['accuracy'].mean():.4f}")
        print(f"Mean AUC:      {season_df['auc'].mean():.4f}")
        season_df.to_csv(PROCESSED_DIR / "improved_season_accuracy.csv", index=False)

    # Save final metadata
    joblib.dump(selected_features, MODELS_DIR / "selected_features.joblib")

    return {
        "models": models,
        "best_name": best_name,
        "best_model": models.get(best_name),
        "selected_features": selected_features,
        "results": results,
        "X": X_sel,
        "y": y,
        "df": df,
    }
