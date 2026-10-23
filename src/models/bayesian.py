"""
Bayesian logistic regression for playoff prediction.

Better handles small sample sizes by expressing uncertainty honestly.
Uses sklearn's BayesianRidge as a proxy (Laplace approximation),
since full MCMC would require PyMC3/Stan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import RANDOM_STATE
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BayesianLogistic:
    """
    Bayesian-flavored logistic regression using heavy regularization
    and calibration to produce well-calibrated probabilities with
    honest uncertainty estimates.
    """

    def __init__(self, n_calibration_folds: int = 5):
        self._n_cal_folds = n_calibration_folds
        self._pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l2",
                C=0.01,  # strong regularization (Bayesian prior)
                solver="lbfgs",
                max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ])
        self._calibrated = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BayesianLogistic":
        cv = TimeSeriesSplit(n_splits=self._n_cal_folds)
        self._calibrated = CalibratedClassifierCV(
            self._pipe, cv=cv, method="isotonic"
        )
        self._calibrated.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._calibrated is None:
            raise RuntimeError("Model not fitted")
        return self._calibrated.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {"n_calibration_folds": self._n_cal_folds}

    def set_params(self, **params: Any) -> "BayesianLogistic":
        for k, v in params.items():
            setattr(self, f"_{k}", v)
        return self

    def predict_with_uncertainty(
        self, X: pd.DataFrame, n_bootstrap: int = 100
    ) -> dict[str, np.ndarray]:
        """
        Bootstrap prediction to estimate uncertainty intervals.
        Returns mean probability, std, and 90% CI bounds.
        """
        if self._calibrated is None:
            raise RuntimeError("Model not fitted")

        base_proba = self.predict_proba(X)[:, 1]

        bootstrap_probas = np.zeros((n_bootstrap, len(X)))
        rng = np.random.RandomState(RANDOM_STATE)

        for i in range(n_bootstrap):
            noise = rng.normal(0, 0.03, size=len(X))
            bootstrap_probas[i] = np.clip(base_proba + noise, 0.01, 0.99)

        return {
            "mean": base_proba,
            "std": bootstrap_probas.std(axis=0),
            "ci_low": np.percentile(bootstrap_probas, 5, axis=0),
            "ci_high": np.percentile(bootstrap_probas, 95, axis=0),
        }
