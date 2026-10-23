"""
Prediction module for current-season playoff matchups.

Fetches current standings, builds feature vectors for hypothetical
matchups, and outputs win probabilities.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import CURRENT_SEASON_END_YEAR, MODELS_DIR, PROCESSED_DIR, season_id
from src.data.nhl_api_client import NHLAPIClient
from src.features.engineer import (
    _build_team_season_features,
    _compute_differential,
    _get_team_features,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def predict_series(
    team_a: str,
    team_b: str,
    model: Any,
    feature_cols: list[str],
    team_features: pd.DataFrame,
    season: int = CURRENT_SEASON_END_YEAR,
) -> dict[str, Any]:
    """
    Predict the outcome of a single playoff series.

    team_a is treated as the higher seed (home ice).
    Returns dict with probabilities for each team.
    """
    hi_feats = _get_team_features(team_features, team_a, season)
    lo_feats = _get_team_features(team_features, team_b, season)

    if hi_feats is None:
        logger.error("No features found for %s in season %d", team_a, season)
        return {}
    if lo_feats is None:
        logger.error("No features found for %s in season %d", team_b, season)
        return {}

    diff = _compute_differential(hi_feats, lo_feats, prefix="diff_")
    diff["is_bubble_year"] = 0
    diff["is_shortened_season"] = 0

    X = pd.DataFrame([diff])
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    proba = model.predict_proba(X)[0]
    higher_seed_prob = proba[1]

    return {
        "higher_seed": team_a,
        "lower_seed": team_b,
        "higher_seed_win_prob": round(higher_seed_prob, 4),
        "lower_seed_win_prob": round(1 - higher_seed_prob, 4),
        "predicted_winner": team_a if higher_seed_prob >= 0.5 else team_b,
        "confidence": round(max(higher_seed_prob, 1 - higher_seed_prob), 4),
    }


def build_current_season_features(
    client: NHLAPIClient,
    season: int = CURRENT_SEASON_END_YEAR,
) -> pd.DataFrame:
    """
    Fetch current-season team stats and build the feature set
    needed for predictions.
    """
    from src.data.collectors import (
        collect_standings,
        collect_team_stats_nhl,
        collect_moneypuck_team_stats,
        collect_goalie_stats,
    )

    logger.info("Fetching current season (%d) data...", season)

    standings = collect_standings(client, season, season)
    nhl_stats = collect_team_stats_nhl(client, season, season)
    moneypuck = collect_moneypuck_team_stats(season, season)
    goalies = collect_goalie_stats(client, season, season)

    team_features = _build_team_season_features(
        standings, nhl_stats, moneypuck, goalies
    )
    return team_features


def predict_bracket(
    matchups: list[tuple[str, str]],
    model: Any,
    feature_cols: list[str],
    team_features: pd.DataFrame,
    season: int = CURRENT_SEASON_END_YEAR,
) -> pd.DataFrame:
    """
    Predict outcomes for a list of (higher_seed, lower_seed) matchups.
    Returns a DataFrame with predictions and probabilities.
    """
    results: list[dict[str, Any]] = []
    for team_a, team_b in matchups:
        result = predict_series(
            team_a, team_b, model, feature_cols, team_features, season
        )
        if result:
            results.append(result)
        else:
            logger.warning("Could not predict %s vs %s", team_a, team_b)

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(PROCESSED_DIR / "predictions.csv", index=False)
        print(f"\n{'='*60}")
        print(f"Playoff Series Predictions ({season})")
        print(f"{'='*60}")
        for _, row in df.iterrows():
            winner = row["predicted_winner"]
            loser = row["lower_seed"] if winner == row["higher_seed"] else row["higher_seed"]
            print(
                f"  {row['higher_seed']} vs {row['lower_seed']}  →  "
                f"{winner} wins ({row['confidence']*100:.1f}% confidence)"
            )
    return df


def simulate_full_bracket(
    first_round: list[tuple[str, str]],
    model: Any,
    feature_cols: list[str],
    team_features: pd.DataFrame,
    season: int = CURRENT_SEASON_END_YEAR,
) -> dict[str, Any]:
    """
    Simulate an entire playoff bracket from first round through finals.
    Feeds winners of each round into the next.
    """
    bracket: dict[str, list[dict[str, Any]]] = {}
    round_names = ["First Round", "Second Round", "Conference Finals", "Stanley Cup Final"]
    current_matchups = first_round

    for round_idx, round_name in enumerate(round_names):
        if not current_matchups:
            break

        round_results = []
        for team_a, team_b in current_matchups:
            result = predict_series(
                team_a, team_b, model, feature_cols, team_features, season
            )
            if result:
                round_results.append(result)

        bracket[round_name] = round_results

        print(f"\n{'='*60}")
        print(f"{round_name}")
        print(f"{'='*60}")
        for r in round_results:
            winner = r["predicted_winner"]
            print(
                f"  {r['higher_seed']} vs {r['lower_seed']}  →  "
                f"{winner} ({r['confidence']*100:.1f}%)"
            )

        winners = [r["predicted_winner"] for r in round_results]
        next_matchups = []
        for i in range(0, len(winners) - 1, 2):
            next_matchups.append((winners[i], winners[i + 1]))
        current_matchups = next_matchups

    if bracket.get("Stanley Cup Final"):
        champion = bracket["Stanley Cup Final"][0]["predicted_winner"]
        print(f"\n{'*'*60}")
        print(f"  Predicted Stanley Cup Champion: {champion}")
        print(f"{'*'*60}")

    return bracket
