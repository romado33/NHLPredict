"""
Model training pipeline.

Trains baseline (logistic regression) and ensemble (XGBoost, LightGBM)
models with cross-validation, hyperparameter tuning, and calibration.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import CV_FOLDS, MODELS_DIR, RANDOM_STATE
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


# ── Feature / target split ──────────────────────────────────────────────────

LABEL_COL = "higher_seed_won"
META_COLS = [
    "season", "round", "higher_seed", "lower_seed",
    "higher_seed_won", "is_bubble_year", "is_shortened_season",
]


def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split feature matrix into X (numeric features) and y (label)."""
    feature_cols = [
        c for c in df.columns
        if c not in META_COLS and df[c].dtype in (np.float64, np.int64, float, int)
    ]
    X = df[feature_cols].copy()
    y = df[LABEL_COL].copy()
    return X, y, feature_cols


# ── Baseline: Logistic Regression ───────────────────────────────────────────

def train_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y)

    logger.info(
        "Logistic best C=%.4f  CV AUC=%.4f",
        search.best_params_["model__C"], search.best_score_,
    )
    best = search.best_estimator_
    joblib.dump(best, MODELS_DIR / "logistic_baseline.joblib")
    return best


# ── XGBoost ─────────────────────────────────────────────────────────────────

def train_xgboost(X: pd.DataFrame, y: pd.Series) -> Pipeline | None:
    if not HAS_XGB:
        logger.warning("xgboost not installed, skipping")
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [200, 300, 500],
        "model__subsample": [0.7, 0.8, 0.9],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    search.fit(X, y)

    logger.info("XGBoost best params: %s  CV AUC=%.4f", search.best_params_, search.best_score_)
    best = search.best_estimator_
    joblib.dump(best, MODELS_DIR / "xgboost.joblib")
    return best


# ── LightGBM ───────────────────────────────────────────────────────────────

def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> Pipeline | None:
    if not HAS_LGB:
        logger.warning("lightgbm not installed, skipping")
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [200, 300, 500],
        "model__num_leaves": [15, 31, 63],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
    )
    search.fit(X, y)

    logger.info("LightGBM best params: %s  CV AUC=%.4f", search.best_params_, search.best_score_)
    best = search.best_estimator_
    joblib.dump(best, MODELS_DIR / "lightgbm.joblib")
    return best


# ── Calibrated Ensemble ─────────────────────────────────────────────────────

def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    logistic: Pipeline | None = None,
    xgb_model: Pipeline | None = None,
    lgb_model: Pipeline | None = None,
) -> Pipeline:
    """
    Build a soft-voting ensemble from available models,
    wrapped in isotonic calibration.
    """
    estimators: list[tuple[str, Pipeline]] = []
    if logistic is not None:
        estimators.append(("logistic", logistic))
    if xgb_model is not None:
        estimators.append(("xgboost", xgb_model))
    if lgb_model is not None:
        estimators.append(("lightgbm", lgb_model))

    if len(estimators) < 2:
        logger.warning("Fewer than 2 models for ensemble; returning best single model")
        model = estimators[0][1] if estimators else logistic
        return model

    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X, y)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(ensemble, cv=cv, method="isotonic")
    calibrated.fit(X, y)

    joblib.dump(calibrated, MODELS_DIR / "ensemble_calibrated.joblib")
    logger.info("Calibrated ensemble saved")
    return calibrated


# ── Cross-validated predictions ─────────────────────────────────────────────

def cv_predictions(
    model: Any, X: pd.DataFrame, y: pd.Series
) -> np.ndarray:
    """Return out-of-fold probability predictions for evaluation."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]


# ── Full pipeline ───────────────────────────────────────────────────────────

def train_all(df: pd.DataFrame) -> dict[str, Any]:
    """
    End-to-end training: prepares data, trains all models,
    builds ensemble, returns dict of trained models.
    """
    X, y, feature_cols = prepare_xy(df)
    logger.info("Training on %d samples, %d features", len(X), len(feature_cols))

    logistic = train_logistic(X, y)
    xgb_model = train_xgboost(X, y)
    lgb_model = train_lightgbm(X, y)
    ensemble = train_ensemble(X, y, logistic, xgb_model, lgb_model)

    return {
        "logistic": logistic,
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "ensemble": ensemble,
        "feature_cols": feature_cols,
        "X": X,
        "y": y,
    }
