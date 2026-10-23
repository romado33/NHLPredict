"""
Automated feature selection to eliminate noise features
and keep only those that contribute to predicting series outcomes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    mutual_info_classif,
    SequentialFeatureSelector,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import CV_FOLDS, PROCESSED_DIR, RANDOM_STATE
from src.models.trainer import prepare_xy
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def mutual_information_ranking(
    X: pd.DataFrame, y: pd.Series, top_n: int = 50
) -> list[str]:
    """Rank features by mutual information with the target."""
    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index
    )

    mi_scores = mutual_info_classif(X_filled, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected = mi_df.head(top_n)["feature"].tolist()
    logger.info("Mutual info selected %d features (top MI=%.4f)", len(selected), mi_df["mi_score"].max())
    return selected


def tree_importance_selection(
    X: pd.DataFrame, y: pd.Series, threshold: str = "median"
) -> list[str]:
    """Select features using a RandomForest's built-in importances."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    pipe.fit(X, y)

    importances = pipe.named_steps["model"].feature_importances_
    median_imp = np.median(importances)
    mask = importances >= median_imp if threshold == "median" else importances > 0

    selected = X.columns[mask].tolist()
    logger.info("Tree importance selected %d / %d features", len(selected), len(X.columns))
    return selected


def recursive_elimination(
    X: pd.DataFrame, y: pd.Series, target_n: int = 30
) -> list[str]:
    """Forward sequential feature selection using logistic regression."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
    ])

    n_features = min(target_n, X.shape[1])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sfs = SequentialFeatureSelector(
        pipe, n_features_to_select=n_features, direction="forward",
        scoring="roc_auc", cv=cv, n_jobs=-1
    )

    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index
    )

    sfs.fit(X_filled, y)
    selected = X.columns[sfs.get_support()].tolist()
    logger.info("Sequential selection chose %d features", len(selected))
    return selected


def consensus_selection(
    df: pd.DataFrame, min_methods: int = 2
) -> tuple[list[str], pd.DataFrame]:
    """
    Run multiple feature selection methods and keep features
    selected by at least `min_methods` approaches.
    """
    X, y, feature_cols = prepare_xy(df)

    logger.info("Running mutual information ranking...")
    mi_features = set(mutual_information_ranking(X, y, top_n=50))

    logger.info("Running tree importance selection...")
    tree_features = set(tree_importance_selection(X, y))

    logger.info("Running sequential feature selection...")
    seq_features = set(recursive_elimination(X, y, target_n=30))

    all_features = mi_features | tree_features | seq_features
    vote_counts = {}
    for f in all_features:
        count = sum([
            f in mi_features,
            f in tree_features,
            f in seq_features,
        ])
        vote_counts[f] = count

    selected = [f for f, c in vote_counts.items() if c >= min_methods]
    selected.sort(key=lambda f: vote_counts[f], reverse=True)

    report = pd.DataFrame([
        {"feature": f, "votes": vote_counts[f],
         "mi": f in mi_features, "tree": f in tree_features, "seq": f in seq_features}
        for f in sorted(vote_counts, key=vote_counts.get, reverse=True)
    ])

    report.to_csv(PROCESSED_DIR / "feature_selection_report.csv", index=False)
    logger.info("Consensus: %d features selected (of %d total)", len(selected), len(feature_cols))

    print(f"\n{'='*60}")
    print(f"Feature Selection: {len(selected)} / {len(feature_cols)} features retained")
    print(f"{'='*60}")
    print(report.head(40).to_string(index=False))

    return selected, report


def apply_feature_selection(
    df: pd.DataFrame, selected_features: list[str]
) -> pd.DataFrame:
    """Keep only the selected features plus metadata columns."""
    from src.models.trainer import META_COLS
    keep = [c for c in META_COLS if c in df.columns] + [
        c for c in selected_features if c in df.columns
    ]
    return df[keep].copy()
