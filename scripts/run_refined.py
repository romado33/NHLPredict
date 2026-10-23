"""
Refined training iteration with:
- ELO ratings
- Sample weighting (recent seasons weighted more)
- Focused feature set informed by round-2 results
- Bayesian-style optimization via larger random search

Usage:
    python scripts/run_refined.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, RAW_DIR
from src.features.elo import add_elo_features
from src.models.improved_trainer import (
    META_COLS,
    add_interaction_features,
    prepare_xy,
    select_features,
    temporal_cv_evaluate,
    temporal_cv_splits,
)
from src.utils.logging_config import get_logger

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = get_logger("run_refined")


def compute_sample_weights(df: pd.DataFrame, decay: float = 0.92) -> np.ndarray:
    """
    Exponential decay weighting: recent seasons get higher weight.
    decay=0.92 means each year older gets 92% of the next year's weight.
    """
    seasons = sorted(df["season"].unique())
    max_season = max(seasons)
    weights = np.array([
        decay ** (max_season - s) for s in df["season"]
    ])
    return weights


def train_weighted_xgboost(
    X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray
) -> Pipeline | None:
    if xgb is None:
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "model__n_estimators": [150, 200, 300, 400, 500],
        "model__subsample": [0.5, 0.6, 0.7, 0.8],
        "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7],
        "model__reg_alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.3, 0.5],
    }

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=500, cv=TimeSeriesSplit(n_splits=5),
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X, y, model__sample_weight=sample_weights)

    logger.info("Weighted XGBoost: CV AUC=%.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_


def train_weighted_lightgbm(
    X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray
) -> Pipeline | None:
    if lgb is None:
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMClassifier(
            verbose=-1, random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
        "model__n_estimators": [150, 200, 300, 400, 500],
        "model__num_leaves": [7, 11, 15, 21, 31],
        "model__subsample": [0.5, 0.6, 0.7, 0.8],
        "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7],
        "model__reg_alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "model__min_child_samples": [3, 5, 10, 15, 20],
    }

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=500, cv=TimeSeriesSplit(n_splits=5),
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X, y, model__sample_weight=sample_weights)

    logger.info("Weighted LightGBM: CV AUC=%.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_


def train_weighted_logistic(
    X: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray
) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1", solver="saga",
            max_iter=5000, random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "model__C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    }

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=10, cv=TimeSeriesSplit(n_splits=5),
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    search.fit(X, y, model__sample_weight=sample_weights)

    logger.info("Weighted Logistic: C=%.5f  CV AUC=%.4f", search.best_params_["model__C"], search.best_score_)
    return search.best_estimator_


def build_blended_predictions(
    models: dict, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Blend model predictions with temporal CV, returning optimally weighted ensemble.
    """
    splits = temporal_cv_splits(df, min_train_years=8)

    model_probas = {name: np.full(len(y), np.nan) for name in models if models[name] is not None}

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        for name, model in models.items():
            if model is None:
                continue
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            model_probas[name][test_idx] = proba

    # Find optimal blend weights using grid search over valid indices
    valid_mask = np.all([~np.isnan(p) for p in model_probas.values()], axis=0)
    y_valid = y[valid_mask].values
    probas_valid = {name: p[valid_mask] for name, p in model_probas.items()}

    best_auc = 0.0
    best_weights = None
    model_names = list(probas_valid.keys())

    for w1 in np.arange(0.0, 1.05, 0.1):
        for w2 in np.arange(0.0, 1.05 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue
            ws = [w1, w2, w3] if len(model_names) >= 3 else [w1, 1.0 - w1]

            blended = sum(
                ws[i] * probas_valid[model_names[i]]
                for i in range(min(len(ws), len(model_names)))
            )

            try:
                auc = roc_auc_score(y_valid, blended)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {model_names[i]: ws[i] for i in range(min(len(ws), len(model_names)))}
            except ValueError:
                continue

    logger.info("Optimal blend weights: %s  AUC=%.4f", best_weights, best_auc)

    # Compute blended metrics
    blended_proba = sum(
        best_weights[name] * probas_valid[name] for name in model_names
    )
    blended_preds = (blended_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_valid, blended_preds),
        "log_loss": log_loss(y_valid, blended_proba),
        "roc_auc": roc_auc_score(y_valid, blended_proba),
        "brier_score": brier_score_loss(y_valid, blended_proba),
        "blend_weights": best_weights,
    }

    return metrics


def main() -> None:
    feature_path = PROCESSED_DIR / "feature_matrix.csv"
    df = pd.read_csv(feature_path)

    series_path = RAW_DIR / "playoff_series.csv"
    series_df = pd.read_csv(series_path)

    logger.info("="*60)
    logger.info("REFINED TRAINING (Iteration 3)")
    logger.info("="*60)

    # Add ELO ratings
    logger.info("Computing ELO ratings...")
    df = add_elo_features(df, series_df)

    # Add interaction features
    df = add_interaction_features(df)

    # Sort temporally
    df = df.sort_values("season").reset_index(drop=True)

    # Prepare features
    X, y, feature_cols = prepare_xy(df)
    logger.info("Total features before selection: %d", len(feature_cols))

    # Feature selection
    selected = select_features(X, y, max_features=50)
    X_sel = X[selected]
    logger.info("Features after selection: %d", len(selected))

    # Compute sample weights
    sample_weights = compute_sample_weights(df, decay=0.92)

    # Train models with sample weighting
    print(f"\n{'='*60}")
    print("Training weighted models...")
    print(f"{'='*60}")

    logistic = train_weighted_logistic(X_sel, y, sample_weights)
    xgb_model = train_weighted_xgboost(X_sel, y, sample_weights)
    lgb_model = train_weighted_lightgbm(X_sel, y, sample_weights)

    # Temporal CV evaluation for each model
    models = {
        "weighted_logistic": logistic,
        "weighted_xgboost": xgb_model,
        "weighted_lightgbm": lgb_model,
    }

    print(f"\n{'='*60}")
    print("TEMPORAL CV RESULTS (per model)")
    print(f"{'='*60}")

    for name, model in models.items():
        if model is None:
            continue
        metrics = temporal_cv_evaluate(model, X_sel, y, df, min_train_years=8)
        season_df = metrics["season_results"]

        print(f"\n  {name}:")
        print(f"    Accuracy:    {metrics['accuracy']:.4f}")
        print(f"    Log Loss:    {metrics['log_loss']:.4f}")
        print(f"    ROC AUC:     {metrics['roc_auc']:.4f}")
        print(f"    Brier Score: {metrics['brier_score']:.4f}")
        print(f"    Seasons: {len(season_df)}")
        print(f"    Mean season accuracy: {season_df['accuracy'].mean():.4f}")

    # Optimal blending
    print(f"\n{'='*60}")
    print("OPTIMIZING BLEND WEIGHTS...")
    print(f"{'='*60}")

    blend_metrics = build_blended_predictions(models, X_sel, y, df)
    print(f"\n  Blended Ensemble:")
    print(f"    Accuracy:    {blend_metrics['accuracy']:.4f}")
    print(f"    Log Loss:    {blend_metrics['log_loss']:.4f}")
    print(f"    ROC AUC:     {blend_metrics['roc_auc']:.4f}")
    print(f"    Brier Score: {blend_metrics['brier_score']:.4f}")
    print(f"    Weights:     {blend_metrics['blend_weights']}")

    # Train final models on ALL data for deployment
    print(f"\n{'='*60}")
    print("Training final models on full dataset...")
    print(f"{'='*60}")

    for name, model in models.items():
        if model is not None:
            model.fit(X_sel, y)

    # Save everything
    joblib.dump(models, MODELS_DIR / "refined_models.joblib")
    joblib.dump(selected, MODELS_DIR / "refined_features.joblib")
    joblib.dump(blend_metrics["blend_weights"], MODELS_DIR / "blend_weights.joblib")
    joblib.dump(sample_weights, MODELS_DIR / "sample_weights.joblib")

    # Save refined feature matrix
    keep_meta = [c for c in META_COLS if c in df.columns]
    keep_feats = [c for c in selected if c in df.columns and c not in keep_meta]
    df_out = df[keep_meta + keep_feats].copy()
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.to_csv(PROCESSED_DIR / "feature_matrix_refined.csv", index=False)

    print(f"\n{'='*60}")
    print("REFINED TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Selected features: {len(selected)}")
    print(f"Best blend AUC:    {blend_metrics['roc_auc']:.4f}")
    print(f"Blend weights:     {blend_metrics['blend_weights']}")

    # Print top features
    print(f"\nTop 25 selected features:")
    for i, f in enumerate(selected[:25]):
        print(f"  {i+1:2d}. {f}")


if __name__ == "__main__":
    main()
