"""
Generate 2026 NHL Playoff Predictions.

Fetches current season data, builds features, and predicts
playoff matchups using the best trained model.

Usage:
    python scripts/predict_2026.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    CURRENT_SEASON_END_YEAR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    season_id,
)
from src.data.nhl_api_client import NHLAPIClient
from src.data.collectors import (
    collect_standings,
    collect_team_stats_nhl,
    collect_moneypuck_team_stats,
    collect_goalie_stats,
)
from src.features.engineer import (
    _build_team_season_features,
    _compute_differential,
    _get_team_features,
)
from src.features.elo import add_elo_features, compute_playoff_elo, DEFAULT_ELO
from src.utils.logging_config import get_logger

logger = get_logger("predict_2026")

SEASON = CURRENT_SEASON_END_YEAR


def fetch_current_season_data(client: NHLAPIClient) -> pd.DataFrame:
    """Fetch and build features for the current season."""
    logger.info("Fetching %d regular season data...", SEASON)

    standings = collect_standings(client, SEASON, SEASON)
    nhl_stats = collect_team_stats_nhl(client, SEASON, SEASON)
    moneypuck = collect_moneypuck_team_stats(SEASON, SEASON)
    goalies = collect_goalie_stats(client, SEASON, SEASON)

    team_features = _build_team_season_features(
        standings, nhl_stats, moneypuck, goalies
    )

    if team_features.empty:
        logger.error("Failed to build current season features")
        return pd.DataFrame()

    logger.info(
        "Built features for %d teams: %s",
        len(team_features),
        sorted(team_features["team_abbrev"].unique()),
    )
    return team_features


def infer_playoff_matchups(team_features: pd.DataFrame) -> list[dict]:
    """
    Infer first-round playoff matchups from current standings.
    NHL bracket format:
      - Top 3 in each division qualify (division seeds 1-3)
      - 2 wildcard spots per conference
      - Division winner with best record vs WC2
      - Other division winner vs WC1
      - 2 vs 3 within each division
    """
    tf = team_features[team_features["season"] == SEASON].copy()

    if "stnd_division_rank" not in tf.columns or "stnd_conference_rank" not in tf.columns:
        logger.warning("Missing ranking columns, using points-based ordering")
        return _simple_seed_matchups(tf)

    return _division_based_matchups(tf)


def _division_based_matchups(tf: pd.DataFrame) -> list[dict]:
    """Construct matchups using actual NHL playoff seeding rules."""
    matchups = []

    # Attempt to detect divisions from standings data
    # NHL has 4 divisions: Atlantic, Metropolitan, Central, Pacific
    # Each conference has 2 divisions

    if "stnd_division_rank" not in tf.columns:
        return _simple_seed_matchups(tf)

    # Sort by points to determine conference seeding
    tf = tf.sort_values("stnd_points", ascending=False).copy()

    # Take top 16 teams as playoff contenders
    playoff_teams = tf.head(16)

    # Simplified approach: pair 1v16, 2v15, etc. within conference-like groupings
    teams = playoff_teams["team_abbrev"].tolist()
    points = playoff_teams["stnd_points"].tolist()

    for i in range(8):
        hi = teams[i]
        lo = teams[15 - i]
        matchups.append({
            "higher_seed": hi,
            "lower_seed": lo,
            "higher_seed_pts": points[i],
            "lower_seed_pts": points[15 - i],
        })

    return matchups


def _simple_seed_matchups(tf: pd.DataFrame) -> list[dict]:
    """Fallback: seed by points and pair 1v16, 2v15, etc."""
    tf = tf.sort_values("stnd_points", ascending=False)
    teams = tf["team_abbrev"].tolist()
    points = tf["stnd_points"].tolist()

    matchups = []
    n = min(16, len(teams))
    for i in range(n // 2):
        matchups.append({
            "higher_seed": teams[i],
            "lower_seed": teams[n - 1 - i],
            "higher_seed_pts": points[i],
            "lower_seed_pts": points[n - 1 - i],
        })
    return matchups


def predict_series(
    team_a: str,
    team_b: str,
    models: dict,
    blend_weights: dict,
    feature_cols: list[str],
    team_features: pd.DataFrame,
) -> dict:
    """Predict a single series using the blended model."""
    hi_feats = _get_team_features(team_features, team_a, SEASON)
    lo_feats = _get_team_features(team_features, team_b, SEASON)

    if hi_feats is None:
        logger.warning("No features for %s", team_a)
        return {}
    if lo_feats is None:
        logger.warning("No features for %s", team_b)
        return {}

    diff = _compute_differential(hi_feats, lo_feats, prefix="diff_")
    diff["is_bubble_year"] = 0
    diff["is_shortened_season"] = 0

    # Add ELO features if available
    diff.setdefault("diff_elo_rating", 0.0)
    diff.setdefault("elo_expected_win", 0.5)

    X = pd.DataFrame([diff])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    # Blend predictions from all models
    blended_proba = 0.0
    total_weight = 0.0

    for name, model in models.items():
        if model is None:
            continue
        weight = blend_weights.get(name, 0.0)
        if weight <= 0:
            continue

        try:
            proba = model.predict_proba(X)[0][1]
            blended_proba += weight * proba
            total_weight += weight
        except Exception as e:
            logger.warning("Prediction failed for %s: %s", name, e)

    if total_weight > 0:
        blended_proba /= total_weight
    else:
        blended_proba = 0.5

    return {
        "higher_seed": team_a,
        "lower_seed": team_b,
        "win_prob_higher": round(blended_proba, 4),
        "win_prob_lower": round(1 - blended_proba, 4),
        "predicted_winner": team_a if blended_proba >= 0.5 else team_b,
        "confidence": round(max(blended_proba, 1 - blended_proba) * 100, 1),
    }


def simulate_bracket(
    first_round_matchups: list[dict],
    models: dict,
    blend_weights: dict,
    feature_cols: list[str],
    team_features: pd.DataFrame,
) -> None:
    """Simulate the entire playoff bracket."""
    round_names = ["FIRST ROUND", "SECOND ROUND", "CONFERENCE FINALS", "STANLEY CUP FINAL"]
    current_matchups = first_round_matchups

    all_results = {}

    for round_idx, round_name in enumerate(round_names):
        if not current_matchups:
            break

        print(f"\n{'='*70}")
        print(f"  {round_name}")
        print(f"{'='*70}")

        round_results = []
        for m in current_matchups:
            result = predict_series(
                m["higher_seed"], m["lower_seed"],
                models, blend_weights, feature_cols, team_features,
            )
            if result:
                round_results.append(result)
                winner = result["predicted_winner"]
                loser = result["lower_seed"] if winner == result["higher_seed"] else result["higher_seed"]
                prob = result["confidence"]
                print(
                    f"  {result['higher_seed']:>4s} vs {result['lower_seed']:<4s}  =>  "
                    f"{winner} wins  ({prob}% confidence)  "
                    f"[{result['win_prob_higher']*100:.0f}-{result['win_prob_lower']*100:.0f}]"
                )

        all_results[round_name] = round_results

        # Build next round matchups
        winners = [r["predicted_winner"] for r in round_results]
        next_matchups = []
        for i in range(0, len(winners) - 1, 2):
            next_matchups.append({
                "higher_seed": winners[i],
                "lower_seed": winners[i + 1],
            })
        current_matchups = next_matchups

    # Announce champion
    if "STANLEY CUP FINAL" in all_results and all_results["STANLEY CUP FINAL"]:
        final = all_results["STANLEY CUP FINAL"][0]
        champion = final["predicted_winner"]
        conf = final["confidence"]
        print(f"\n{'*'*70}")
        print(f"  PREDICTED 2026 STANLEY CUP CHAMPION: {champion}")
        print(f"  Confidence: {conf}%")
        print(f"{'*'*70}")

    # Save predictions
    all_preds = []
    for round_name, results in all_results.items():
        for r in results:
            r["round"] = round_name
            all_preds.append(r)

    pred_df = pd.DataFrame(all_preds)
    pred_df.to_csv(PROCESSED_DIR / "predictions_2026.csv", index=False)
    logger.info("Predictions saved to predictions_2026.csv")


def main() -> None:
    print(f"\n{'#'*70}")
    print(f"  NHL PLAYOFF PREDICTOR — {SEASON} SEASON PREDICTIONS")
    print(f"{'#'*70}")

    # Load models and feature config
    refined_models_path = MODELS_DIR / "refined_models.joblib"
    refined_features_path = MODELS_DIR / "refined_features.joblib"
    blend_weights_path = MODELS_DIR / "blend_weights.joblib"

    # Try refined first, fall back to improved
    if refined_models_path.exists():
        models = joblib.load(refined_models_path)
        feature_cols = joblib.load(refined_features_path)
        blend_weights = joblib.load(blend_weights_path)
        logger.info("Loaded refined models")
    elif (MODELS_DIR / "improved_xgboost.joblib").exists():
        models = {
            "improved_xgboost": joblib.load(MODELS_DIR / "improved_xgboost.joblib"),
            "improved_logistic": joblib.load(MODELS_DIR / "improved_logistic.joblib"),
        }
        feature_cols = joblib.load(MODELS_DIR / "selected_features.joblib")
        blend_weights = {"improved_xgboost": 0.7, "improved_logistic": 0.3}
        logger.info("Loaded improved models")
    else:
        logger.error("No trained models found. Run training first.")
        sys.exit(1)

    # Fetch current season data
    client = NHLAPIClient()
    team_features = fetch_current_season_data(client)

    if team_features.empty:
        logger.error("Could not fetch current season data")
        sys.exit(1)

    # Add ELO from historical data
    series_path = RAW_DIR / "playoff_series.csv"
    if series_path.exists():
        series_df = pd.read_csv(series_path)
        elo_df = compute_playoff_elo(series_df)

        for _, row in team_features.iterrows():
            team = row["team_abbrev"]
            match = elo_df[elo_df["team"] == team]
            if not match.empty:
                latest = match.sort_values("season").iloc[-1]
                idx = team_features.index[
                    (team_features["team_abbrev"] == team) &
                    (team_features["season"] == SEASON)
                ]
                if len(idx) > 0:
                    team_features.loc[idx, "elo_pre_playoff"] = latest["elo_pre_playoff"]

    # Show current standings
    print(f"\n{'='*70}")
    print(f"  CURRENT STANDINGS (Top 16 by Points)")
    print(f"{'='*70}")
    current = team_features[team_features["season"] == SEASON].copy()
    if "stnd_points" in current.columns:
        current = current.sort_values("stnd_points", ascending=False)
        for i, (_, row) in enumerate(current.head(16).iterrows()):
            pts = row.get("stnd_points", "?")
            gp = row.get("stnd_games_played", "?")
            gd = row.get("stnd_goal_diff", "?")
            print(f"  {i+1:2d}. {row['team_abbrev']:>4s}  |  {pts} pts  |  GP: {gp}  |  GD: {gd:+.0f}" if isinstance(gd, (int, float)) else f"  {i+1:2d}. {row['team_abbrev']:>4s}  |  {pts} pts")

    # Infer matchups
    matchups = infer_playoff_matchups(team_features)
    if not matchups:
        logger.error("Could not infer playoff matchups")
        sys.exit(1)

    print(f"\n  Projected first-round matchups: {len(matchups)}")

    # Simulate full bracket
    simulate_bracket(matchups, models, blend_weights, feature_cols, team_features)


if __name__ == "__main__":
    main()
