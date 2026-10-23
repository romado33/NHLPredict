"""
Run the improved training pipeline with temporal CV,
feature selection, and interaction features.

Usage:
    python scripts/run_improved.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import PROCESSED_DIR, MODELS_DIR
from src.models.improved_trainer import train_improved
from src.utils.logging_config import get_logger

logger = get_logger("run_improved")


def main() -> None:
    feature_path = PROCESSED_DIR / "feature_matrix.csv"
    if not feature_path.exists():
        logger.error("Feature matrix not found. Run 'python main.py build' first.")
        sys.exit(1)

    df = pd.read_csv(feature_path)
    logger.info("Loaded feature matrix: %d rows × %d columns", len(df), len(df.columns))

    results = train_improved(df)

    print(f"\n{'='*60}")
    print("IMPROVED PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Best model:       {results['best_name']}")
    print(f"Selected features: {len(results['selected_features'])}")
    print(f"Outputs saved to:  {PROCESSED_DIR}")
    print(f"Models saved to:   {MODELS_DIR}")


if __name__ == "__main__":
    main()
