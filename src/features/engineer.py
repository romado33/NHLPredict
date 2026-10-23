"""
Feature engineering pipeline.

Builds a matchup-level feature matrix for playoff series prediction.
Each row represents one playoff series with differential features
(higher_seed_stat - lower_seed_stat) plus contextual features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Team-name normalisation ─────────────────────────────────────────────────

ABBREV_MAP = {
    "PHX": "ARI", "ARI": "UTA",
    "ATL": "WPG",
    "S.J": "SJS", "N.J": "NJD", "T.B": "TBL", "L.A": "LAK",
}

# MoneyPuck uses full team names; map to abbreviations
FULL_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA", "Arizona Coyotes": "ARI",
    "Atlanta Thrashers": "ATL", "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL", "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL", "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
    "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Phoenix Coyotes": "ARI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR", "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


def normalize_team(name: str, season: int | None = None) -> str:
    """Convert any team name variant to a canonical abbreviation."""
    if not name:
        return name
    name = name.strip()
    if name in FULL_NAME_TO_ABBREV:
        return FULL_NAME_TO_ABBREV[name]
    if len(name) == 3:
        return name.upper()
    return name


# ── Feature builders ────────────────────────────────────────────────────────

def build_feature_matrix(
    series_df: pd.DataFrame,
    standings_df: pd.DataFrame,
    team_stats_nhl_df: pd.DataFrame,
    moneypuck_df: pd.DataFrame,
    goalie_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Master function: joins all data sources and computes
    differential features for each playoff series.
    """
    logger.info("Building feature matrix from %d series", len(series_df))

    team_features = _build_team_season_features(
        standings_df, team_stats_nhl_df, moneypuck_df, goalie_df
    )

    rows: list[dict[str, Any]] = []
    unmatched = 0

    for _, s in series_df.iterrows():
        season = s["season"]
        hi = s["higher_seed"]
        lo = s["lower_seed"]
        winner = s["series_winner"]

        hi_feats = _get_team_features(team_features, hi, season)
        lo_feats = _get_team_features(team_features, lo, season)

        if hi_feats is None or lo_feats is None:
            unmatched += 1
            logger.debug(
                "Missing features for %s vs %s (%d)", hi, lo, season
            )
            continue

        row = _compute_differential(hi_feats, lo_feats, prefix="diff_")
        row["season"] = season
        row["round"] = s.get("round", 0)
        row["higher_seed"] = hi
        row["lower_seed"] = lo
        row["higher_seed_won"] = 1 if winner == hi else 0

        # absolute features that don't make sense as differentials
        row["is_bubble_year"] = 1 if season == 2020 else 0
        row["is_shortened_season"] = 1 if season in (2013, 2020, 2021) else 0

        rows.append(row)

    if unmatched:
        logger.warning("%d series had missing features and were dropped", unmatched)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("Feature matrix is empty!")
        return df

    df = _add_historical_playoff_experience(df, series_df)
    df = _clean_features(df)

    df.to_csv(PROCESSED_DIR / "feature_matrix.csv", index=False)
    logger.info(
        "Feature matrix: %d rows × %d columns", len(df), len(df.columns)
    )
    return df


# ── Team-level feature aggregation ──────────────────────────────────────────

def _build_team_season_features(
    standings_df: pd.DataFrame,
    nhl_stats_df: pd.DataFrame,
    moneypuck_df: pd.DataFrame,
    goalie_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a single DataFrame: one row per (team_abbrev, season)."""

    frames: list[pd.DataFrame] = []

    # 1) Standings features
    if not standings_df.empty:
        st = _extract_standings_features(standings_df)
        frames.append(st)

    # 2) NHL Stats API features
    if not nhl_stats_df.empty:
        ns = _extract_nhl_stats_features(nhl_stats_df)
        frames.append(ns)

    # 3) MoneyPuck advanced features
    if not moneypuck_df.empty:
        mp = _extract_moneypuck_features(moneypuck_df)
        frames.append(mp)

    # 4) Goalie features (aggregated per team)
    if not goalie_df.empty:
        gl = _extract_goalie_features(goalie_df)
        frames.append(gl)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(
            f, on=["team_abbrev", "season"], how="outer", suffixes=("", "_y")
        )
        y_cols = [c for c in merged.columns if c.endswith("_y")]
        merged = merged.drop(columns=y_cols, errors="ignore")

    return merged


def _extract_standings_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the standings DataFrame."""
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        team_name = row.get("teamName", {})
        if isinstance(team_name, dict):
            name = team_name.get("default", "")
        else:
            name = str(team_name)

        abbrev = row.get("teamAbbrev", {})
        if isinstance(abbrev, dict):
            abbrev = abbrev.get("default", "")
        else:
            abbrev = str(abbrev)

        if not abbrev:
            abbrev = normalize_team(name, row.get("season"))

        rec = {
            "team_abbrev": abbrev,
            "season": row.get("season"),
            "stnd_wins": row.get("wins", 0),
            "stnd_losses": row.get("losses", 0),
            "stnd_ot_losses": row.get("otLosses", 0),
            "stnd_points": row.get("points", 0),
            "stnd_point_pct": row.get("pointPctg", 0),
            "stnd_reg_wins": row.get("regulationWins", 0),
            "stnd_reg_plus_ot_wins": row.get("regulationPlusOtWins", 0),
            "stnd_goals_for": row.get("goalFor", 0),
            "stnd_goals_against": row.get("goalAgainst", 0),
            "stnd_goal_diff": row.get("goalDifferential", 0),
            "stnd_games_played": row.get("gamesPlayed", 0),
            "stnd_home_wins": row.get("homeWins", 0),
            "stnd_home_losses": row.get("homeLosses", 0),
            "stnd_home_ot_losses": row.get("homeOtLosses", 0),
            "stnd_road_wins": row.get("roadWins", 0),
            "stnd_road_losses": row.get("roadLosses", 0),
            "stnd_road_ot_losses": row.get("roadOtLosses", 0),
            "stnd_streak_count": row.get("streakCount", 0),
            "stnd_l10_wins": row.get("l10Wins", 0),
            "stnd_l10_losses": row.get("l10Losses", 0),
            "stnd_l10_ot_losses": row.get("l10OtLosses", 0),
            "stnd_conference_rank": row.get("conferenceSequence", 0),
            "stnd_division_rank": row.get("divisionSequence", 0),
            "stnd_league_rank": row.get("leagueSequence", 0),
            "stnd_wildcard_rank": row.get("wildcardSequence", 0),
            "stnd_shootout_wins": row.get("shootoutWins", 0),
            "stnd_shootout_losses": row.get("shootoutLosses", 0),
        }

        gp = rec["stnd_games_played"] or 1
        rec["stnd_goals_for_pg"] = rec["stnd_goals_for"] / gp
        rec["stnd_goals_against_pg"] = rec["stnd_goals_against"] / gp
        rec["stnd_goal_diff_pg"] = rec["stnd_goal_diff"] / gp

        # Pythagorean expectation (Bill James adapted for hockey, exponent ~2)
        gf = max(rec["stnd_goals_for"], 1)
        ga = max(rec["stnd_goals_against"], 1)
        rec["stnd_pythag_pct"] = gf ** 2 / (gf ** 2 + ga ** 2)

        rec["stnd_home_pct"] = (
            rec["stnd_home_wins"]
            / max(rec["stnd_home_wins"] + rec["stnd_home_losses"] + rec["stnd_home_ot_losses"], 1)
        )
        rec["stnd_road_pct"] = (
            rec["stnd_road_wins"]
            / max(rec["stnd_road_wins"] + rec["stnd_road_losses"] + rec["stnd_road_ot_losses"], 1)
        )

        records.append(rec)

    return pd.DataFrame(records)


def _extract_nhl_stats_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the NHL stats API team data."""
    team_col = None
    for c in ("teamFullName", "teamName", "franchiseName"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        return pd.DataFrame()

    df = df.copy()
    df["team_abbrev"] = df[team_col].apply(normalize_team)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip_cols = {"season", "seasonId", "teamId", "franchiseId", "gameTypeId"}
    feature_cols = [c for c in numeric_cols if c not in skip_cols]

    agg = df.groupby(["team_abbrev", "season"])[feature_cols].first().reset_index()

    rename_map = {c: f"nhl_{c}" for c in feature_cols}
    agg = agg.rename(columns=rename_map)
    return agg


def _extract_moneypuck_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract team-level MoneyPuck advanced stats."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    team_col = None
    for c in ("team", "Team", "name", "Name"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        logger.warning("MoneyPuck: could not find team column in %s", list(df.columns)[:10])
        return pd.DataFrame()

    df["team_abbrev"] = df[team_col].apply(normalize_team)

    # Filter to 5-on-5 or all situations where applicable
    if "situation" in df.columns:
        df_all = df[df["situation"] == "all"].copy()
        df_5v5 = df[df["situation"] == "5on5"].copy()

        all_numeric = df_all.select_dtypes(include=[np.number]).columns.tolist()
        all_numeric = [c for c in all_numeric if c not in ("season",)]
        all_rename = {c: f"mp_all_{c}" for c in all_numeric}
        df_all = df_all[["team_abbrev", "season"] + all_numeric].rename(columns=all_rename)

        ev_numeric = df_5v5.select_dtypes(include=[np.number]).columns.tolist()
        ev_numeric = [c for c in ev_numeric if c not in ("season",)]
        ev_rename = {c: f"mp_5v5_{c}" for c in ev_numeric}
        df_5v5 = df_5v5[["team_abbrev", "season"] + ev_numeric].rename(columns=ev_rename)

        merged = df_all.merge(df_5v5, on=["team_abbrev", "season"], how="outer")
        return merged
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if c not in ("season",)]
        rename = {c: f"mp_{c}" for c in numeric}
        return df[["team_abbrev", "season"] + numeric].rename(columns=rename)


def _extract_goalie_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate goalie stats per team (starter-weighted)."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    team_col = None
    for c in ("teamAbbrevs", "teamFullName", "teamName"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        return pd.DataFrame()

    df["team_abbrev"] = df[team_col].apply(lambda x: normalize_team(str(x).split(",")[0] if x else ""))

    gp_col = "gamesPlayed" if "gamesPlayed" in df.columns else None
    if not gp_col:
        for c in df.columns:
            if "games" in c.lower() and "played" in c.lower():
                gp_col = c
                break

    agg_cols: dict[str, str] = {}
    for stat, agg_type in [
        ("savePct", "max"), ("goalsAgainstAverage", "min"),
        ("wins", "sum"), ("losses", "sum"), ("shutouts", "sum"),
        ("gamesPlayed", "sum"), ("timeOnIce", "sum"),
    ]:
        if stat in df.columns:
            agg_cols[stat] = agg_type

    if not agg_cols:
        return pd.DataFrame()

    # Starter = goalie with most GP per team/season
    if gp_col:
        idx = df.groupby(["team_abbrev", "season"])[gp_col].idxmax()
        starters = df.loc[idx].copy()
        starter_stats = {}
        for col in ("savePct", "goalsAgainstAverage", "wins", "shutouts"):
            if col in starters.columns:
                starter_stats[col] = "first"
        starter_agg = starters.groupby(["team_abbrev", "season"]).agg(starter_stats).reset_index()
        starter_agg = starter_agg.rename(
            columns={c: f"goalie_starter_{c}" for c in starter_stats}
        )
    else:
        starter_agg = pd.DataFrame()

    team_agg = df.groupby(["team_abbrev", "season"]).agg(agg_cols).reset_index()
    team_agg = team_agg.rename(columns={c: f"goalie_team_{c}" for c in agg_cols})

    if not starter_agg.empty:
        return team_agg.merge(starter_agg, on=["team_abbrev", "season"], how="outer")
    return team_agg


# ── Differential computation ────────────────────────────────────────────────

def _get_team_features(
    team_df: pd.DataFrame, team: str, season: int
) -> pd.Series | None:
    """Look up the feature row for a team/season, trying abbreviation variants."""
    if team_df.empty:
        return None
    mask = (team_df["team_abbrev"] == team) & (team_df["season"] == season)
    match = team_df.loc[mask]
    if match.empty:
        for alt in _abbreviation_variants(team, season):
            mask = (team_df["team_abbrev"] == alt) & (team_df["season"] == season)
            match = team_df.loc[mask]
            if not match.empty:
                break
    if match.empty:
        return None
    return match.iloc[0]


def _abbreviation_variants(team: str, season: int) -> list[str]:
    """Return alternative abbreviations for a franchise."""
    variants = []
    mapping = {
        "WPG": ["ATL"], "UTA": ["ARI", "PHX"], "ARI": ["PHX"],
        "ATL": ["WPG"], "PHX": ["ARI"],
        "CAR": ["HFD"],
    }
    if team in mapping:
        variants.extend(mapping[team])
    return variants


def _compute_differential(
    hi: pd.Series, lo: pd.Series, prefix: str = "diff_"
) -> dict[str, Any]:
    """Compute hi - lo for all shared numeric features."""
    result: dict[str, Any] = {}
    shared = set(hi.index) & set(lo.index)
    skip = {"team_abbrev", "season", "report", "goalie_report"}

    for col in sorted(shared):
        if col in skip:
            continue
        hi_val = hi[col]
        lo_val = lo[col]
        if isinstance(hi_val, (int, float, np.integer, np.floating)):
            try:
                result[f"{prefix}{col}"] = float(hi_val) - float(lo_val)
            except (TypeError, ValueError):
                continue

    return result


# ── Historical playoff experience ───────────────────────────────────────────

def _add_historical_playoff_experience(
    features_df: pd.DataFrame, series_df: pd.DataFrame
) -> pd.DataFrame:
    """Add features for recent playoff experience (appearances and series wins in last 5 years)."""
    features_df = features_df.copy()

    po_years: dict[str, list[int]] = {}
    po_wins: dict[str, dict[int, int]] = {}

    for _, row in series_df.iterrows():
        s = row["season"]
        for team in (row["higher_seed"], row["lower_seed"]):
            po_years.setdefault(team, []).append(s)
        winner = row["series_winner"]
        po_wins.setdefault(winner, {})
        po_wins[winner][s] = po_wins[winner].get(s, 0) + 1

    hi_exp, lo_exp, hi_sw, lo_sw = [], [], [], []
    for _, row in features_df.iterrows():
        season = row["season"]
        window = range(season - 5, season)

        for team, exp_list, sw_list in [
            (row["higher_seed"], hi_exp, hi_sw),
            (row["lower_seed"], lo_exp, lo_sw),
        ]:
            years = po_years.get(team, [])
            exp_list.append(sum(1 for y in years if y in window))
            wins = po_wins.get(team, {})
            sw_list.append(sum(wins.get(y, 0) for y in window))

    features_df["diff_playoff_appearances_5yr"] = [h - l for h, l in zip(hi_exp, lo_exp)]
    features_df["diff_playoff_series_wins_5yr"] = [h - l for h, l in zip(hi_sw, lo_sw)]

    return features_df


# ── Cleaning ────────────────────────────────────────────────────────────────

def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with too many NaNs, infinite values, and constant columns."""
    nan_thresh = 0.5
    nan_frac = df.isnull().mean()
    drop_cols = nan_frac[nan_frac > nan_thresh].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns with >%.0f%% NaN", len(drop_cols), nan_thresh * 100)
        df = df.drop(columns=drop_cols)

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric = df.select_dtypes(include=[np.number])
    constant_cols = numeric.columns[numeric.std() == 0].tolist()
    if constant_cols:
        logger.info("Dropping %d constant columns", len(constant_cols))
        df = df.drop(columns=constant_cols)

    return df
