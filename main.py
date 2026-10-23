"""
NHL Playoff Predictor — Main Pipeline

Usage:
    python main.py collect       # Step 1: Collect all historical data
    python main.py build         # Step 2: Build feature matrix
    python main.py train         # Step 3: Train models (baseline + ensemble)
    python main.py evaluate      # Step 4: Evaluate and compare models
    python main.py select        # Step 5: Feature selection and retrain
    python main.py predict       # Step 6: Predict current season playoffs
    python main.py full          # Run entire pipeline end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    CURRENT_SEASON_END_YEAR,
    FIRST_SEASON,
    LAST_HISTORICAL_SEASON,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.utils.logging_config import get_logger

logger = get_logger("main")


def step_collect() -> None:
    """Collect all historical data from APIs and MoneyPuck."""
    from src.data.nhl_api_client import NHLAPIClient
    from src.data.collectors import (
        collect_playoff_series,
        collect_team_stats_nhl,
        collect_moneypuck_team_stats,
        collect_goalie_stats,
        collect_standings,
    )

    client = NHLAPIClient()

    logger.info("="*60)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("="*60)

    logger.info("Collecting playoff series results...")
    collect_playoff_series(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)

    logger.info("Collecting standings...")
    collect_standings(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)

    logger.info("Collecting NHL team stats...")
    collect_team_stats_nhl(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)

    logger.info("Collecting MoneyPuck advanced stats...")
    collect_moneypuck_team_stats()

    logger.info("Collecting goalie stats...")
    collect_goalie_stats(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)

    logger.info("Data collection complete.")


def step_build() -> pd.DataFrame:
    """Build feature matrix from collected data."""
    from src.features.engineer import build_feature_matrix

    logger.info("="*60)
    logger.info("STEP 2: BUILD FEATURE MATRIX")
    logger.info("="*60)

    series_df = _load_csv("playoff_series.csv")
    standings_df = _load_csv("standings.csv")
    nhl_stats_df = _load_csv("team_stats_nhl.csv")
    moneypuck_df = _load_csv("team_stats_moneypuck.csv")
    goalie_df = _load_csv("goalie_stats.csv")

    feature_df = build_feature_matrix(
        series_df, standings_df, nhl_stats_df, moneypuck_df, goalie_df
    )
    return feature_df


def step_train() -> dict:
    """Train all models on the feature matrix."""
    from src.models.trainer import train_all

    logger.info("="*60)
    logger.info("STEP 3: TRAIN MODELS")
    logger.info("="*60)

    feature_df = pd.read_csv(PROCESSED_DIR / "feature_matrix.csv")
    results = train_all(feature_df)
    return results


def step_evaluate() -> None:
    """Evaluate and compare all trained models."""
    import joblib
    from src.models.evaluator import (
        evaluate_all,
        feature_importance_report,
        plot_calibration,
        season_accuracy_report,
    )
    from src.models.trainer import prepare_xy

    logger.info("="*60)
    logger.info("STEP 4: EVALUATE MODELS")
    logger.info("="*60)

    feature_df = pd.read_csv(PROCESSED_DIR / "feature_matrix.csv")

    models = {}
    for name in ("logistic_baseline", "xgboost", "lightgbm", "ensemble_calibrated"):
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    if not models:
        logger.error("No trained models found. Run 'train' first.")
        return

    evaluate_all(models, feature_df)

    X, y, feature_cols = prepare_xy(feature_df)
    plot_calibration(models, X, y)

    best_name = _pick_best_model(models, feature_df)
    best_model = models[best_name]
    feature_importance_report(best_model, feature_cols)
    season_accuracy_report(best_model, feature_df)


def step_select() -> None:
    """Run feature selection and retrain with selected features."""
    from src.models.feature_selection import consensus_selection, apply_feature_selection
    from src.models.trainer import train_all

    logger.info("="*60)
    logger.info("STEP 5: FEATURE SELECTION + RETRAIN")
    logger.info("="*60)

    feature_df = pd.read_csv(PROCESSED_DIR / "feature_matrix.csv")
    selected, report = consensus_selection(feature_df)

    reduced_df = apply_feature_selection(feature_df, selected)
    reduced_df.to_csv(PROCESSED_DIR / "feature_matrix_selected.csv", index=False)

    logger.info("Retraining on %d selected features...", len(selected))
    results = train_all(reduced_df)

    from src.models.evaluator import evaluate_all, feature_importance_report
    from src.models.trainer import prepare_xy

    models = {k: v for k, v in results.items() if k not in ("feature_cols", "X", "y") and v is not None}
    evaluate_all(models, reduced_df)

    X, y, feature_cols = prepare_xy(reduced_df)
    best_name = _pick_best_model(models, reduced_df)
    feature_importance_report(models[best_name], feature_cols)


def step_predict() -> None:
    """Predict current season playoff outcomes."""
    import joblib
    from src.data.nhl_api_client import NHLAPIClient
    from src.models.predictor import (
        build_current_season_features,
        predict_bracket,
        simulate_full_bracket,
    )
    from src.models.trainer import prepare_xy

    logger.info("="*60)
    logger.info("STEP 6: PREDICT %d PLAYOFFS", CURRENT_SEASON_END_YEAR)
    logger.info("="*60)

    model_path = MODELS_DIR / "ensemble_calibrated.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "xgboost.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "logistic_baseline.joblib"
    if not model_path.exists():
        logger.error("No trained model found. Run 'train' first.")
        return

    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path.name)

    feature_matrix_path = PROCESSED_DIR / "feature_matrix_selected.csv"
    if not feature_matrix_path.exists():
        feature_matrix_path = PROCESSED_DIR / "feature_matrix.csv"
    feature_df = pd.read_csv(feature_matrix_path)
    _, _, feature_cols = prepare_xy(feature_df)

    client = NHLAPIClient()
    team_features = build_current_season_features(client, CURRENT_SEASON_END_YEAR)

    if team_features.empty:
        logger.error("Could not build current season features")
        return

    print(f"\nCurrent season teams with features: {sorted(team_features['team_abbrev'].unique())}")

    # Try to infer likely matchups from standings
    matchups = _infer_playoff_matchups(team_features, CURRENT_SEASON_END_YEAR)

    if matchups:
        print(f"\nInferred {len(matchups)} first-round matchups from standings")
        simulate_full_bracket(matchups, model, feature_cols, team_features, CURRENT_SEASON_END_YEAR)
    else:
        logger.warning("Could not infer matchups; provide them manually")


def _infer_playoff_matchups(
    team_features: pd.DataFrame, season: int
) -> list[tuple[str, str]]:
    """
    Infer first-round playoff matchups from conference standings.
    NHL bracket: 1v8(WC2), 2v7(WC1), 3v6(div), 4v5(div) in each conference.
    This is an approximation — actual matchups depend on division winners.
    """
    if "stnd_conference_rank" not in team_features.columns:
        return []

    tf = team_features[team_features["season"] == season].copy()
    if tf.empty:
        return []

    if "stnd_points" not in tf.columns:
        return []

    tf = tf.sort_values("stnd_points", ascending=False)
    top_16 = tf.head(16)

    teams_by_rank = top_16.sort_values("stnd_conference_rank")
    teams_list = teams_by_rank["team_abbrev"].tolist()

    if len(teams_list) < 16:
        return []

    # Pair 1v16, 2v15, 3v14, ... (simplified)
    matchups = []
    for i in range(8):
        matchups.append((teams_list[i], teams_list[15 - i]))

    return matchups


def _pick_best_model(models: dict, feature_df: pd.DataFrame) -> str:
    """Return the key of the best model by AUC."""
    from src.models.evaluator import evaluate_model
    from src.models.trainer import prepare_xy

    X, y, _ = prepare_xy(feature_df)
    best_auc = -1.0
    best_name = list(models.keys())[0]

    for name, model in models.items():
        if model is None:
            continue
        try:
            metrics = evaluate_model(name, model, X, y)
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = name
        except Exception:
            continue

    return best_name


def _load_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    logger.warning("File not found: %s", path)
    return pd.DataFrame()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    commands = {
        "collect": step_collect,
        "build": step_build,
        "train": step_train,
        "evaluate": step_evaluate,
        "select": step_select,
        "predict": step_predict,
        "full": lambda: (step_collect(), step_build(), step_train(),
                         step_evaluate(), step_select(), step_predict()),
    }

    if command not in commands:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    commands[command]()


if __name__ == "__main__":
    main()
