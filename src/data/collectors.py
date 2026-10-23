"""
Data collection orchestrator.

Pulls playoff series results and regular-season team stats
from the NHL API and MoneyPuck for all seasons in scope.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from config.settings import (
    FIRST_SEASON,
    LAST_HISTORICAL_SEASON,
    MONEYPUCK_TEAM_URL_TEMPLATE,
    RAW_DIR,
    season_id,
)
from src.data.nhl_api_client import NHLAPIClient
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Playoff Series Results ──────────────────────────────────────────────────

def collect_playoff_series(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """
    Collect all playoff series results from start_year through end_year.

    Returns a DataFrame with columns:
        season, round, series_letter, higher_seed, lower_seed,
        higher_seed_wins, lower_seed_wins, series_winner
    """
    all_series: list[dict[str, Any]] = []

    for year in tqdm(range(start_year, end_year + 1), desc="Playoff brackets"):
        if year == 2005:
            logger.info("Skipping 2004-05 (lockout season)")
            continue

        try:
            bracket = client.get_playoff_bracket(year)
        except Exception:
            logger.warning("Could not fetch bracket for %d, trying schedule fallback", year)
            series = _series_from_schedule(client, year)
            all_series.extend(series)
            continue

        series = _parse_bracket(bracket, year)
        if series:
            all_series.extend(series)
        else:
            logger.warning("Empty bracket for %d, trying schedule fallback", year)
            all_series.extend(_series_from_schedule(client, year))

    df = pd.DataFrame(all_series)
    if not df.empty:
        df.to_csv(RAW_DIR / "playoff_series.csv", index=False)
        logger.info("Saved %d playoff series to playoff_series.csv", len(df))
    return df


def _parse_bracket(data: dict[str, Any], year: int) -> list[dict[str, Any]]:
    """Parse the api-web playoff-bracket response."""
    series_list: list[dict[str, Any]] = []

    for round_data in data.get("rounds", data.get("series", [])):
        if isinstance(round_data, dict) and "series" in round_data:
            round_num = round_data.get("roundNumber", 0)
            for s in round_data["series"]:
                parsed = _parse_single_series(s, year, round_num)
                if parsed:
                    series_list.append(parsed)
        elif isinstance(round_data, dict):
            parsed = _parse_single_series(round_data, year, 0)
            if parsed:
                series_list.append(parsed)

    return series_list


def _parse_single_series(s: dict[str, Any], year: int, round_num: int) -> dict[str, Any] | None:
    top = s.get("topSeedTeam", s.get("topSeed", {}))
    bottom = s.get("bottomSeedTeam", s.get("bottomSeed", {}))

    top_abbrev = top.get("abbrev", top.get("teamAbbrev", {}).get("default", ""))
    bottom_abbrev = bottom.get("abbrev", bottom.get("teamAbbrev", {}).get("default", ""))
    top_wins = top.get("wins", s.get("topSeedWins", 0))
    bottom_wins = bottom.get("wins", s.get("bottomSeedWins", 0))

    if not top_abbrev or not bottom_abbrev:
        return None

    if top_wins == 0 and bottom_wins == 0:
        return None

    winner = top_abbrev if top_wins > bottom_wins else bottom_abbrev

    return {
        "season": year,
        "round": round_num if round_num else s.get("roundNumber", 0),
        "series_letter": s.get("seriesLetter", s.get("seriesTitle", "")),
        "higher_seed": top_abbrev,
        "lower_seed": bottom_abbrev,
        "higher_seed_wins": top_wins,
        "lower_seed_wins": bottom_wins,
        "series_winner": winner,
    }


def _series_from_schedule(client: NHLAPIClient, year: int) -> list[dict[str, Any]]:
    """
    Fallback: reconstruct series from playoff game results in the schedule.
    We look at playoff games (gameType=3) and group by matchup.
    """
    sid = season_id(year)
    series_map: dict[tuple[str, str], dict[str, int]] = {}

    schedule_dates = _get_playoff_date_range(year)
    for date_str in schedule_dates:
        try:
            sched = client.get_schedule(date_str)
        except Exception:
            continue

        for week in sched.get("gameWeek", []):
            for game in week.get("games", []):
                if game.get("gameType") != 3:
                    continue
                if game.get("gameState") not in ("FINAL", "OFF", "OVER"):
                    continue

                away = game.get("awayTeam", {}).get("abbrev", "")
                home = game.get("homeTeam", {}).get("abbrev", "")
                if not away or not home:
                    continue

                matchup = tuple(sorted([away, home]))
                if matchup not in series_map:
                    series_map[matchup] = {away: 0, home: 0}

                away_score = game["awayTeam"].get("score", 0)
                home_score = game["homeTeam"].get("score", 0)
                if away_score > home_score:
                    series_map[matchup][away] = series_map[matchup].get(away, 0) + 1
                elif home_score > away_score:
                    series_map[matchup][home] = series_map[matchup].get(home, 0) + 1

    results: list[dict[str, Any]] = []
    for (t1, t2), wins in series_map.items():
        w1 = wins.get(t1, 0)
        w2 = wins.get(t2, 0)
        if w1 == 0 and w2 == 0:
            continue
        winner = t1 if w1 > w2 else t2
        results.append({
            "season": year,
            "round": 0,
            "series_letter": "",
            "higher_seed": t1,
            "lower_seed": t2,
            "higher_seed_wins": w1,
            "lower_seed_wins": w2,
            "series_winner": winner,
        })

    return results


def _get_playoff_date_range(year: int) -> list[str]:
    """Generate weekly date samples spanning April–June for a given year."""
    import datetime
    dates: list[str] = []
    start = datetime.date(year, 4, 1)
    end = datetime.date(year, 7, 1)
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += datetime.timedelta(days=7)
    return dates


# ── Regular Season Team Stats (NHL Stats API) ──────────────────────────────

def collect_team_stats_nhl(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """
    Collect regular-season team summary stats from the NHL stats API
    for all seasons in range.
    """
    all_frames: list[pd.DataFrame] = []

    reports = ["summary", "percentages", "realtime", "penalties",
               "penaltykill", "powerplay", "faceoffpercentages",
               "goalsbyperiod", "goalsforbystrength", "goalsagainstbystrength",
               "leadingtrailing", "outshootoutshotby", "summaryshooting",
               "scoretrailfirst"]

    for year in tqdm(range(start_year, end_year + 1), desc="Team stats (NHL API)"):
        if year == 2005:
            continue
        sid = season_id(year)
        season_frames: list[pd.DataFrame] = []

        for report in reports:
            try:
                data = client.get_team_stats(
                    report=report, season_id=sid, game_type=2
                )
                rows = data.get("data", [])
                if rows:
                    df = pd.DataFrame(rows)
                    df["season"] = year
                    df["report"] = report
                    season_frames.append(df)
            except Exception as e:
                logger.debug("Stats report %s failed for %d: %s", report, year, e)

        if season_frames:
            merged = _merge_team_reports(season_frames, year)
            all_frames.append(merged)

    if not all_frames:
        logger.error("No team stats collected from NHL API")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(RAW_DIR / "team_stats_nhl.csv", index=False)
    logger.info("Saved NHL team stats: %d rows, %d columns", len(df), len(df.columns))
    return df


def _merge_team_reports(frames: list[pd.DataFrame], year: int) -> pd.DataFrame:
    """Merge multiple report DataFrames for a single season on teamFullName."""
    base = frames[0].copy()
    team_col = _find_team_column(base)

    for f in frames[1:]:
        other_team_col = _find_team_column(f)
        overlap = set(base.columns) & set(f.columns) - {team_col, "season"}
        f_deduped = f.drop(columns=[c for c in overlap if c in f.columns], errors="ignore")
        if other_team_col and other_team_col != team_col:
            f_deduped = f_deduped.rename(columns={other_team_col: team_col})
        if team_col in f_deduped.columns:
            base = base.merge(f_deduped, on=[team_col, "season"], how="outer", suffixes=("", "_dup"))
            dup_cols = [c for c in base.columns if c.endswith("_dup")]
            base = base.drop(columns=dup_cols, errors="ignore")

    return base


def _find_team_column(df: pd.DataFrame) -> str:
    for candidate in ("teamFullName", "teamName", "franchiseName"):
        if candidate in df.columns:
            return candidate
    return "teamFullName"


# ── MoneyPuck Advanced Stats ───────────────────────────────────────────────

def collect_moneypuck_team_stats(
    start_year: int = max(FIRST_SEASON, 2009),
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """
    Download team-level advanced stats CSVs from MoneyPuck.
    Available from 2008-09 season onwards.
    """
    all_frames: list[pd.DataFrame] = []

    for year in tqdm(range(start_year, end_year + 1), desc="MoneyPuck stats"):
        if year == 2005:
            continue
        start_year_val = year - 1
        url = MONEYPUCK_TEAM_URL_TEMPLATE.format(start_year=start_year_val)
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/csv,text/plain,*/*",
                "Referer": "https://moneypuck.com/data.htm",
            }
            resp = requests.get(url, timeout=30, headers=headers)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df["season"] = year
            all_frames.append(df)
        except Exception as e:
            logger.warning("MoneyPuck download failed for %d: %s", year, e)

    if not all_frames:
        logger.warning("No MoneyPuck data collected")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(RAW_DIR / "team_stats_moneypuck.csv", index=False)
    logger.info("Saved MoneyPuck team stats: %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Goalie Stats ────────────────────────────────────────────────────────────

def collect_goalie_stats(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """Collect regular-season goalie summary stats for all seasons."""
    all_frames: list[pd.DataFrame] = []

    for year in tqdm(range(start_year, end_year + 1), desc="Goalie stats"):
        if year == 2005:
            continue
        sid = season_id(year)
        for report in ("summary", "advanced"):
            try:
                data = client.get_goalie_stats(
                    report=report, season_id=sid, game_type=2, limit=-1
                )
                rows = data.get("data", [])
                if rows:
                    df = pd.DataFrame(rows)
                    df["season"] = year
                    df["goalie_report"] = report
                    all_frames.append(df)
            except Exception as e:
                logger.debug("Goalie %s failed for %d: %s", report, year, e)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(RAW_DIR / "goalie_stats.csv", index=False)
    logger.info("Saved goalie stats: %d rows", len(df))
    return df


# ── Standings ───────────────────────────────────────────────────────────────

def collect_standings(
    client: NHLAPIClient,
    start_year: int = FIRST_SEASON,
    end_year: int = LAST_HISTORICAL_SEASON,
) -> pd.DataFrame:
    """Collect end-of-regular-season standings for each year."""
    all_frames: list[pd.DataFrame] = []

    reg_season_end_dates = {
        2001: "2001-04-08", 2002: "2002-04-14", 2003: "2003-04-06",
        2004: "2004-04-04", 2006: "2006-04-18", 2007: "2007-04-08",
        2008: "2008-04-06", 2009: "2009-04-12", 2010: "2010-04-11",
        2011: "2011-04-10", 2012: "2012-04-07", 2013: "2013-04-28",
        2014: "2014-04-13", 2015: "2015-04-11", 2016: "2016-04-10",
        2017: "2017-04-09", 2018: "2018-04-08", 2019: "2019-04-06",
        2020: "2020-08-01", 2021: "2021-05-19", 2022: "2022-04-29",
        2023: "2023-04-14", 2024: "2024-04-18", 2025: "2025-04-17",
        2026: "2026-04-15",
    }

    for year in tqdm(range(start_year, end_year + 1), desc="Standings"):
        if year == 2005:
            continue
        date = reg_season_end_dates.get(year)
        if not date:
            import datetime
            today = datetime.date.today()
            if year == today.year or (year == today.year + 1 and today.month < 9):
                date = today.isoformat()
            else:
                date = f"{year}-04-15"
        try:
            data = client.get_standings(date)
            standings = data.get("standings", [])
            if standings:
                df = pd.DataFrame(standings)
                df["season"] = year
                all_frames.append(df)
        except Exception as e:
            logger.warning("Standings failed for %d: %s", year, e)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(RAW_DIR / "standings.csv", index=False)
    logger.info("Saved standings: %d rows", len(df))
    return df
