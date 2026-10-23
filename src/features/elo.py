"""
ELO rating system for NHL teams based on playoff performance.

Maintains a running ELO score that updates after each playoff series,
providing a strong signal of team quality that accumulates over time.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_ELO = 1500
K_FACTOR = 32
HOME_ADVANTAGE = 30


def compute_playoff_elo(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ELO ratings based on historical playoff series results.

    Returns a DataFrame with columns: team, season, elo_before_playoffs, elo_after_playoffs
    """
    elo: dict[str, float] = {}
    records: list[dict] = []

    for season in sorted(series_df["season"].unique()):
        season_series = series_df[series_df["season"] == season].copy()

        # Capture ELO going into playoffs
        season_teams = set(season_series["higher_seed"]) | set(season_series["lower_seed"])
        for team in season_teams:
            if team not in elo:
                elo[team] = DEFAULT_ELO

        # Record pre-playoff ELO
        for team in season_teams:
            records.append({
                "team": team,
                "season": season,
                "elo_pre_playoff": elo[team],
            })

        # Update ELO based on series results
        for _, row in season_series.iterrows():
            winner = row["series_winner"]
            loser = row["lower_seed"] if winner == row["higher_seed"] else row["higher_seed"]

            winner_elo = elo.get(winner, DEFAULT_ELO)
            loser_elo = elo.get(loser, DEFAULT_ELO)

            expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
            expected_loser = 1 - expected_winner

            # Scale K by margin (4-0 sweep is more decisive than 4-3)
            winner_wins = max(row["higher_seed_wins"], row["lower_seed_wins"])
            loser_wins = min(row["higher_seed_wins"], row["lower_seed_wins"])
            margin_mult = 1.0 + 0.1 * (winner_wins - loser_wins)

            elo[winner] = winner_elo + K_FACTOR * margin_mult * (1 - expected_winner)
            elo[loser] = loser_elo + K_FACTOR * margin_mult * (0 - expected_loser)

        # Regression to mean between seasons (prevent drift)
        for team in elo:
            elo[team] = elo[team] * 0.85 + DEFAULT_ELO * 0.15

    elo_df = pd.DataFrame(records)
    return elo_df


def add_elo_features(
    feature_df: pd.DataFrame, series_df: pd.DataFrame
) -> pd.DataFrame:
    """Add ELO differential to the feature matrix."""
    elo_df = compute_playoff_elo(series_df)

    feature_df = feature_df.copy()
    hi_elos = []
    lo_elos = []

    for _, row in feature_df.iterrows():
        season = row["season"]
        hi = row["higher_seed"]
        lo = row["lower_seed"]

        hi_match = elo_df[(elo_df["team"] == hi) & (elo_df["season"] == season)]
        lo_match = elo_df[(elo_df["team"] == lo) & (elo_df["season"] == season)]

        hi_elo = hi_match["elo_pre_playoff"].values[0] if not hi_match.empty else DEFAULT_ELO
        lo_elo = lo_match["elo_pre_playoff"].values[0] if not lo_match.empty else DEFAULT_ELO

        hi_elos.append(hi_elo)
        lo_elos.append(lo_elo)

    feature_df["diff_elo_rating"] = [h - l for h, l in zip(hi_elos, lo_elos)]
    feature_df["elo_expected_win"] = [
        1 / (1 + 10 ** ((l - h) / 400)) for h, l in zip(hi_elos, lo_elos)
    ]

    logger.info("Added ELO features (mean diff=%.1f)", feature_df["diff_elo_rating"].mean())
    return feature_df
