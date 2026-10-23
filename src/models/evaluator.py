"""
Model evaluation framework.

Computes accuracy, log-loss, AUC, Brier score, and calibration metrics.
Generates diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from config.settings import MODELS_DIR, PROCESSED_DIR
from src.models.trainer import cv_predictions, prepare_xy, META_COLS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

PLOTS_DIR = PROCESSED_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a single model using cross-validated predictions."""
    proba = cv_predictions(model, X, y)
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "log_loss": log_loss(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "brier_score": brier_score_loss(y, proba),
    }

    logger.info(
        "%s | Accuracy=%.4f | LogLoss=%.4f | AUC=%.4f | Brier=%.4f",
        name, metrics["accuracy"], metrics["log_loss"],
        metrics["roc_auc"], metrics["brier_score"],
    )
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"\n{classification_report(y, preds, target_names=['Lower seed won', 'Higher seed won'])}")

    return metrics


def evaluate_all(
    models: dict[str, Any],
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame."""
    X, y, _ = prepare_xy(feature_df)
    results: list[dict[str, Any]] = []

    for name, model in models.items():
        if model is None:
            continue
        metrics = evaluate_model(name, model, X, y)
        metrics["model"] = name
        results.append(metrics)

    comparison = pd.DataFrame(results).set_index("model")
    comparison.to_csv(PROCESSED_DIR / "model_comparison.csv")
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    print(comparison.to_string())
    return comparison


# ── Feature importance ──────────────────────────────────────────────────────

def feature_importance_report(
    model: Any, feature_cols: list[str], top_n: int = 30
) -> pd.DataFrame:
    """
    Extract feature importances from a trained model.
    Handles pipelines with imputer → model structure.
    """
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False

    actual_model = _extract_model(model)

    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
    elif hasattr(actual_model, "coef_"):
        importances = np.abs(actual_model.coef_[0])
    else:
        logger.warning("Model has no feature importances")
        return pd.DataFrame()

    n_features = len(importances)
    cols = feature_cols[:n_features] if len(feature_cols) >= n_features else feature_cols

    imp_df = pd.DataFrame({
        "feature": cols,
        "importance": importances[:len(cols)],
    }).sort_values("importance", ascending=False)

    top = imp_df.head(top_n)
    print(f"\nTop {top_n} Features:")
    print(top.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    ax.barh(range(len(top)), top["importance"].values[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    imp_df.to_csv(PROCESSED_DIR / "feature_importances.csv", index=False)
    return imp_df


def _extract_model(pipeline: Any) -> Any:
    """Walk a pipeline/calibration wrapper to find the base estimator."""
    if hasattr(pipeline, "estimator"):
        return _extract_model(pipeline.estimator)
    if hasattr(pipeline, "named_steps"):
        last_step = list(pipeline.named_steps.values())[-1]
        return _extract_model(last_step)
    if hasattr(pipeline, "estimators_"):
        for _, est in pipeline.estimators_:
            inner = _extract_model(est)
            if hasattr(inner, "feature_importances_"):
                return inner
    return pipeline


# ── Calibration plot ────────────────────────────────────────────────────────

def plot_calibration(
    models: dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, model in models.items():
        if model is None:
            continue
        proba = cv_predictions(model, X, y)

        # calibration curve
        fraction_pos, mean_predicted = calibration_curve(y, proba, n_bins=10)
        axes[0].plot(mean_predicted, fraction_pos, "s-", label=name)

        # ROC curve
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Calibration Curves")
    axes[0].legend()

    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "calibration_and_roc.png", dpi=150)
    plt.close(fig)
    logger.info("Calibration and ROC plots saved")


# ── Season-by-season accuracy ──────────────────────────────────────────────

def season_accuracy_report(
    model: Any, feature_df: pd.DataFrame
) -> pd.DataFrame:
    """Show accuracy broken down by season to check for temporal drift."""
    X, y, _ = prepare_xy(feature_df)
    proba = cv_predictions(model, X, y)
    preds = (proba >= 0.5).astype(int)

    feature_df = feature_df.copy()
    feature_df["predicted"] = preds
    feature_df["correct"] = (feature_df["predicted"] == feature_df["higher_seed_won"]).astype(int)
    feature_df["pred_proba"] = proba

    by_season = (
        feature_df.groupby("season")
        .agg(
            n_series=("correct", "count"),
            accuracy=("correct", "mean"),
            avg_proba=("pred_proba", "mean"),
        )
        .reset_index()
    )

    print(f"\n{'='*60}")
    print("Accuracy by Season")
    print(f"{'='*60}")
    print(by_season.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(by_season["season"], by_season["accuracy"], color="steelblue")
    ax.axhline(y=by_season["accuracy"].mean(), color="red", linestyle="--", label="Mean")
    ax.set_xlabel("Season (end year)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Prediction Accuracy by Playoff Season")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_by_season.png", dpi=150)
    plt.close(fig)

    return by_season
