"""
Playoff skater stats collector.

Fetches per-player scoring data from NHL playoffs (gameTypeId=3)
for historical training and current-season roster verification.
Also collects player biographical data (birthDate, draft info).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from config.settings import (
    FIRST_SEASON,
    LAST_HISTORICAL_SEASON,
    RAW_DIR,
    season_id,
)
from src.data.nhl_api_client import NHLAPIClient
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

GAME_TYPE_PLAYOFFS = 3
GAME_TYPE_REGULAR = 2


def collect_playoff_skater_stats(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """
    Collect per-player playoff stats for every season in the range.
    Uses gameTypeId=3 to get playoff-only data from the NHL Stats API.
    """
    all_frames: list[pd.DataFrame] = []

    for year in tqdm(range(start_year, end_year + 1), desc="Playoff skater stats"):
        if year == 2005:
            continue
        sid = season_id(year)
        for report in ("summary", "realtime"):
            try:
                raw = client._stats(
                    f"/skater/{report}",
                    {
                        "isAggregate": "false",
                        "isGame": "false",
                        "limit": -1,
                        "start": 0,
                        "cayenneExp": (
                            f"gameTypeId={GAME_TYPE_PLAYOFFS} "
                            f"and seasonId<={sid} and seasonId>={sid}"
                        ),
                    },
                )
                rows = raw.get("data", [])
                if rows:
                    df = pd.DataFrame(rows)
                    df["season"] = year
                    df["skater_report"] = report
                    all_frames.append(df)
            except Exception as e:
                logger.warning("Playoff skater %s failed for %d: %s", report, year, e)

    if not all_frames:
        logger.error("No playoff skater stats collected")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(RAW_DIR / "skater_stats_playoffs.csv", index=False)
    logger.info(
        "Saved playoff skater stats: %d rows, %d columns", len(df), len(df.columns)
    )
    return df


def collect_player_bios(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """
    Collect player biographical data (birthDate, draftYear, height, weight)
    for all skaters across all seasons. Deduplicates by playerId, keeping
    the most recent record for each player.
    """
    all_frames: list[pd.DataFrame] = []

    for year in tqdm(range(start_year, end_year + 1), desc="Player bios"):
        if year == 2005:
            continue
        sid = season_id(year)
        try:
            raw = client._stats(
                "/skater/bios",
                {
                    "isAggregate": "false",
                    "isGame": "false",
                    "limit": -1,
                    "start": 0,
                    "cayenneExp": (
                        f"gameTypeId={GAME_TYPE_REGULAR} "
                        f"and seasonId<={sid} and seasonId>={sid}"
                    ),
                },
            )
            rows = raw.get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                df["season"] = year
                all_frames.append(df)
        except Exception as e:
            logger.warning("Player bios failed for %d: %s", year, e)

    if not all_frames:
        logger.error("No player bios collected")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values("season", ascending=False).drop_duplicates(
        subset=["playerId"], keep="first"
    )

    keep_cols = ["playerId", "birthDate", "draftYear", "draftRound",
                 "draftOverall", "height", "weight", "nationalityCode"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    out_path = RAW_DIR / "player_bios.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved player bios: %d unique players", len(df))
    return df


def collect_regular_season_skater_stats(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
    output_file: str = "skater_stats_regular.csv",
) -> pd.DataFrame:
    """
    Collect per-player regular-season stats. Separate from the V2 collector
    to avoid coupling; writes to its own file for the scoring pipeline.
    """
    all_frames: list[pd.DataFrame] = []

    for year in tqdm(range(start_year, end_year + 1), desc="Regular skater stats"):
        if year == 2005:
            continue
        sid = season_id(year)
        try:
            raw = client._stats(
                "/skater/summary",
                {
                    "isAggregate": "false",
                    "isGame": "false",
                    "limit": -1,
                    "start": 0,
                    "cayenneExp": (
                        f"gameTypeId={GAME_TYPE_REGULAR} "
                        f"and seasonId<={sid} and seasonId>={sid}"
                    ),
                },
            )
            rows = raw.get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                df["season"] = year
                all_frames.append(df)
        except Exception as e:
            logger.warning("Regular skater summary failed for %d: %s", year, e)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(RAW_DIR / output_file, index=False)
    logger.info(
        "Saved regular skater stats: %d rows, %d columns", len(df), len(df.columns)
    )
    return df
