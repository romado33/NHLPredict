"""
Advanced feature engineering (V2).

Adds: player-level depth, recency/momentum, head-to-head,
round-specific, injury proxy, strength-of-schedule,
style-matchup, and conference/division features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DIR, RAW_DIR
from src.features.engineer import normalize_team, FULL_NAME_TO_ABBREV
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# HIGH IMPACT
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Player-Level Features ───────────────────────────────────────────────

def build_player_features(skater_df: pd.DataFrame) -> pd.DataFrame:
    """
    From per-player stats, compute team-level aggregates:
    - Scoring depth (top-6 vs bottom-6 contribution)
    - Star concentration (% of points from top-3 scorers)
    - Playoff-experienced player count
    - Roster age profile
    """
    if skater_df.empty:
        return pd.DataFrame()

    df = skater_df.copy()

    # Normalize team names
    team_col = None
    for c in ("teamAbbrevs", "teamFullName", "teamName", "lastTeamAbbrev"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        logger.warning("Player features: no team column found in %s", list(df.columns)[:15])
        return pd.DataFrame()

    df["team_abbrev"] = df[team_col].apply(
        lambda x: normalize_team(str(x).split(",")[0].strip()) if pd.notna(x) else ""
    )

    # Use only summary report rows
    if "skater_report" in df.columns:
        df = df[df["skater_report"] == "summary"].copy()

    points_col = "points" if "points" in df.columns else None
    goals_col = "goals" if "goals" in df.columns else None
    gp_col = "gamesPlayed" if "gamesPlayed" in df.columns else None
    toi_col = "timeOnIce" if "timeOnIce" in df.columns else None
    pos_col = "positionCode" if "positionCode" in df.columns else None

    if not points_col or not gp_col:
        logger.warning("Player features: missing points or gamesPlayed columns")
        return pd.DataFrame()

    records: list[dict] = []

    for (team, season), group in df.groupby(["team_abbrev", "season"]):
        if not team or len(group) < 5:
            continue

        group = group.sort_values(points_col, ascending=False)
        total_points = group[points_col].sum()
        total_goals = group[goals_col].sum() if goals_col else 0
        n_players = len(group)

        # Top-3 scorer concentration
        top3_pts = group.head(3)[points_col].sum()
        top3_pct = top3_pts / max(total_points, 1)

        # Top-6 vs bottom-6 forwards (rough split)
        top6 = group.head(6)
        bottom = group.iloc[6:]
        top6_pts = top6[points_col].sum()
        bottom_pts = bottom[points_col].sum()

        # Scoring depth ratio
        depth_ratio = bottom_pts / max(top6_pts, 1)

        # Games played by top scorers (availability proxy)
        top5_avg_gp = top6.head(5)[gp_col].mean()

        # Points per game for top players
        top3_ppg = (top3_pts / top6.head(3)[gp_col].sum()) if top6.head(3)[gp_col].sum() > 0 else 0

        rec = {
            "team_abbrev": team,
            "season": season,
            "plyr_total_points": total_points,
            "plyr_total_goals": total_goals,
            "plyr_top3_pts_pct": top3_pct,
            "plyr_scoring_depth": depth_ratio,
            "plyr_top5_avg_gp": top5_avg_gp,
            "plyr_top3_ppg": top3_ppg,
            "plyr_n_skaters": n_players,
        }

        # Roster age (if birthDate available)
        if "birthDate" in group.columns:
            try:
                ages = pd.to_datetime(group["birthDate"], errors="coerce")
                season_date = pd.Timestamp(f"{season}-04-01")
                computed_ages = (season_date - ages).dt.days / 365.25
                rec["plyr_avg_age"] = computed_ages.mean()
                rec["plyr_age_std"] = computed_ages.std()
            except Exception:
                pass

        # Position breakdown
        if pos_col and pos_col in group.columns:
            n_fwd = len(group[group[pos_col].isin(["C", "L", "R"])])
            n_def = len(group[group[pos_col] == "D"])
            rec["plyr_n_forwards"] = n_fwd
            rec["plyr_n_defensemen"] = n_def

        records.append(rec)

    result = pd.DataFrame(records)
    logger.info("Built player features: %d team-seasons", len(result))
    return result


def build_top_skater_strength_features(skater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Team-season aggregates of elite skater scoring (PPG concentration).
    Feeds V2 series model with a player→team bridge (retrain required to load
    diff_skel_* into selected features).
    """
    if skater_df.empty:
        return pd.DataFrame()

    df = skater_df.copy()
    team_col = None
    for c in ("teamAbbrevs", "teamFullName", "teamName", "lastTeamAbbrev"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        return pd.DataFrame()

    df["team_abbrev"] = df[team_col].apply(
        lambda x: normalize_team(str(x).split(",")[0].strip()) if pd.notna(x) else ""
    )
    if "skater_report" in df.columns:
        df = df[df["skater_report"] == "summary"].copy()

    gp_col = "gamesPlayed" if "gamesPlayed" in df.columns else None
    pts_col = "points" if "points" in df.columns else None
    pos_col = "positionCode" if "positionCode" in df.columns else None
    if not gp_col or not pts_col:
        return pd.DataFrame()

    df["_ppg"] = df[pts_col] / df[gp_col].replace(0, np.nan)

    records: list[dict] = []
    for (team, season), group in df.groupby(["team_abbrev", "season"]):
        if not team or len(group) < 3:
            continue
        g2 = group.sort_values(pts_col, ascending=False)
        top5 = g2.head(5)
        fwd_mask = g2[pos_col].isin(["C", "L", "R"]) if pos_col in g2.columns else pd.Series(True, index=g2.index)
        top3_fwd = g2[fwd_mask].head(3)

        records.append({
            "team_abbrev": team,
            "season": season,
            "skel_top5_ppg_sum": float(top5["_ppg"].sum()),
            "skel_top3_fwd_ppg_sum": float(top3_fwd["_ppg"].sum()) if len(top3_fwd) else 0.0,
        })

    result = pd.DataFrame(records)
    logger.info("Built top-skater strength features: %d team-seasons", len(result))
    return result


# ── 2. Recency / Momentum Features ─────────────────────────────────────────

def build_recency_features(recency_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract team-level last-20-games stats from MoneyPuck recency data.
    Captures late-season momentum heading into playoffs.
    """
    if recency_df.empty:
        return pd.DataFrame()

    df = recency_df.copy()

    team_col = None
    for c in ("team", "Team", "name"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        return pd.DataFrame()

    df["team_abbrev"] = df[team_col].apply(normalize_team)

    if "situation" in df.columns:
        df_all = df[df["situation"] == "all"].copy()
    else:
        df_all = df.copy()

    numeric = df_all.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != "season"]
    rename = {c: f"recent_{c}" for c in numeric}

    result = df_all[["team_abbrev", "season"] + numeric].rename(columns=rename)
    result = result.groupby(["team_abbrev", "season"]).first().reset_index()

    logger.info("Built recency features: %d team-seasons, %d columns", len(result), len(result.columns))
    return result


# ── 3. Head-to-Head Features ───────────────────────────────────────────────

def build_h2h_features(
    h2h_df: pd.DataFrame, series_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each playoff series, compute H2H regular-season features:
    - Win differential
    - Goal differential
    - Dominance ratio
    """
    if h2h_df.empty:
        return _build_h2h_from_default(series_df)

    records: list[dict] = []

    for _, row in series_df.iterrows():
        season = row["season"]
        hi = row["higher_seed"]
        lo = row["lower_seed"]

        match = h2h_df[
            (h2h_df["season"] == season)
            & (h2h_df["team_a"] == hi)
            & (h2h_df["team_b"] == lo)
        ]

        if match.empty:
            match = h2h_df[
                (h2h_df["season"] == season)
                & (h2h_df["team_a"] == lo)
                & (h2h_df["team_b"] == hi)
            ]
            if not match.empty:
                m = match.iloc[0]
                records.append({
                    "season": season,
                    "higher_seed": hi,
                    "lower_seed": lo,
                    "h2h_games": m.get("games", 0),
                    "h2h_hi_wins": m.get("b_wins", 0),
                    "h2h_lo_wins": m.get("a_wins", 0),
                    "h2h_hi_goals": m.get("b_goals", 0),
                    "h2h_lo_goals": m.get("a_goals", 0),
                })
                continue

        if not match.empty:
            m = match.iloc[0]
            records.append({
                "season": season,
                "higher_seed": hi,
                "lower_seed": lo,
                "h2h_games": m.get("games", 0),
                "h2h_hi_wins": m.get("a_wins", 0),
                "h2h_lo_wins": m.get("b_wins", 0),
                "h2h_hi_goals": m.get("a_goals", 0),
                "h2h_lo_goals": m.get("b_goals", 0),
            })
        else:
            records.append({
                "season": season,
                "higher_seed": hi,
                "lower_seed": lo,
                "h2h_games": 0,
                "h2h_hi_wins": 0,
                "h2h_lo_wins": 0,
                "h2h_hi_goals": 0,
                "h2h_lo_goals": 0,
            })

    df = pd.DataFrame(records)
    games = df["h2h_games"].replace(0, 1)
    df["h2h_win_diff"] = df["h2h_hi_wins"] - df["h2h_lo_wins"]
    df["h2h_goal_diff"] = df["h2h_hi_goals"] - df["h2h_lo_goals"]
    df["h2h_win_pct"] = df["h2h_hi_wins"] / games
    df["h2h_dominance"] = (df["h2h_hi_wins"] - df["h2h_lo_wins"]) / games

    logger.info("Built H2H features: %d series", len(df))
    return df


def _build_h2h_from_default(series_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no H2H data collected - return empty features."""
    records = []
    for _, row in series_df.iterrows():
        records.append({
            "season": row["season"],
            "higher_seed": row["higher_seed"],
            "lower_seed": row["lower_seed"],
            "h2h_games": 0, "h2h_hi_wins": 0, "h2h_lo_wins": 0,
            "h2h_hi_goals": 0, "h2h_lo_goals": 0,
            "h2h_win_diff": 0, "h2h_goal_diff": 0,
            "h2h_win_pct": 0.5, "h2h_dominance": 0,
        })
    return pd.DataFrame(records)


# ── 4. Round-Specific Features ──────────────────────────────────────────────

def build_round_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add round-specific indicators. Different dynamics in different rounds:
    - Round 1: more blowouts (1v8 mismatches)
    - Round 2+: tighter matchups between proven teams
    """
    df = series_df[["season", "higher_seed", "lower_seed", "round"]].copy()
    df["is_round_1"] = (df["round"] == 1).astype(int)
    df["is_round_2"] = (df["round"] == 2).astype(int)
    df["is_conf_final"] = (df["round"] == 3).astype(int)
    df["is_cup_final"] = (df["round"] == 4).astype(int)
    df["is_late_round"] = (df["round"] >= 3).astype(int)
    return df


# ── 5. Injury Proxy Features ───────────────────────────────────────────────

def build_injury_proxy(skater_df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy for injury impact: compare games played by top players
    vs the team's total games. High variance = key players missed time.
    """
    if skater_df.empty:
        return pd.DataFrame()

    df = skater_df.copy()
    if "skater_report" in df.columns:
        df = df[df["skater_report"] == "summary"].copy()

    team_col = None
    for c in ("teamAbbrevs", "teamFullName", "teamName", "lastTeamAbbrev"):
        if c in df.columns:
            team_col = c
            break
    if not team_col:
        return pd.DataFrame()

    df["team_abbrev"] = df[team_col].apply(
        lambda x: normalize_team(str(x).split(",")[0].strip()) if pd.notna(x) else ""
    )

    gp_col = "gamesPlayed" if "gamesPlayed" in df.columns else None
    pts_col = "points" if "points" in df.columns else None
    if not gp_col:
        return pd.DataFrame()

    records: list[dict] = []

    for (team, season), group in df.groupby(["team_abbrev", "season"]):
        if not team or len(group) < 5:
            continue

        # Sort by points to identify "key players"
        if pts_col:
            group = group.sort_values(pts_col, ascending=False)

        top10 = group.head(10)
        top10_gp = top10[gp_col]
        team_max_gp = group[gp_col].max()

        # Healthy% = avg GP of top players / max GP
        healthy_pct = top10_gp.mean() / max(team_max_gp, 1)

        # GP variance among top players (high = injuries disrupted lineup)
        gp_variance = top10_gp.std() / max(top10_gp.mean(), 1)

        # Man-games-lost proxy: sum of (max_gp - actual_gp) for top players
        man_games_lost = sum(max(0, team_max_gp - gp) for gp in top10_gp)

        records.append({
            "team_abbrev": team,
            "season": season,
            "inj_top10_healthy_pct": healthy_pct,
            "inj_gp_variance": gp_variance,
            "inj_man_games_lost": man_games_lost,
            "inj_top10_avg_gp": top10_gp.mean(),
        })

    result = pd.DataFrame(records)
    logger.info("Built injury proxy features: %d team-seasons", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MEDIUM IMPACT
# ═══════════════════════════════════════════════════════════════════════════

# ── 6. Strength of Schedule ─────────────────────────────────────────────────

def build_sos_features(standings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strength of schedule by looking at a team's results
    relative to the quality of their opponents.
    SOS = average opponent point% weighted by games played.
    Since we don't have per-game opponent data here, we use a proxy:
    team's point% minus the league average (residual performance).
    """
    if standings_df.empty:
        return pd.DataFrame()

    df = standings_df.copy()

    abbrev_col = None
    pts_col = None
    gp_col = None

    for c in df.columns:
        val = df[c].iloc[0] if len(df) > 0 else None
        if isinstance(val, dict) and "default" in val:
            if "abbrev" in c.lower():
                abbrev_col = c

    if abbrev_col:
        df["team_abbrev"] = df[abbrev_col].apply(
            lambda x: x.get("default", "") if isinstance(x, dict) else str(x)
        )
    else:
        if "team_abbrev" not in df.columns:
            return pd.DataFrame()

    for c in ("points", "pointPctg", "pointPct"):
        if c in df.columns:
            pts_col = c
            break
    for c in ("gamesPlayed",):
        if c in df.columns:
            gp_col = c
            break

    if not pts_col:
        return pd.DataFrame()

    records: list[dict] = []
    for season, group in df.groupby("season"):
        league_avg_pctg = group[pts_col].mean() if pts_col else 0.5
        league_std_pctg = group[pts_col].std() if pts_col else 0.1

        for _, row in group.iterrows():
            team = row.get("team_abbrev", "")
            if not team:
                continue
            team_pctg = row.get(pts_col, 0)

            # Z-score relative to league
            z_score = (team_pctg - league_avg_pctg) / max(league_std_pctg, 0.001)

            records.append({
                "team_abbrev": team,
                "season": season,
                "sos_league_z_score": z_score,
                "sos_above_avg": team_pctg - league_avg_pctg,
            })

    result = pd.DataFrame(records)
    logger.info("Built SOS features: %d team-seasons", len(result))
    return result


# ── 7. Rest Days / Scheduling ──────────────────────────────────────────────

def build_rest_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest-related features based on round number and season context.
    Later rounds mean previous series were played; sweeps mean more rest.
    """
    records: list[dict] = []

    # Build a lookup of series results by season and team
    series_by_team: dict[tuple, list[dict]] = {}
    for _, row in series_df.iterrows():
        for team in (row["higher_seed"], row["lower_seed"]):
            key = (row["season"], team)
            series_by_team.setdefault(key, []).append(dict(row))

    for _, row in series_df.iterrows():
        season = row["season"]
        hi = row["higher_seed"]
        lo = row["lower_seed"]
        round_num = row.get("round", 0)

        hi_prev_games = _count_prev_round_games(series_by_team, season, hi, round_num)
        lo_prev_games = _count_prev_round_games(series_by_team, season, lo, round_num)

        records.append({
            "season": season,
            "higher_seed": hi,
            "lower_seed": lo,
            "rest_hi_prev_games": hi_prev_games,
            "rest_lo_prev_games": lo_prev_games,
            "rest_game_diff": lo_prev_games - hi_prev_games,
            "rest_hi_had_sweep": 1 if hi_prev_games == 4 and round_num > 1 else 0,
            "rest_lo_had_sweep": 1 if lo_prev_games == 4 and round_num > 1 else 0,
        })

    return pd.DataFrame(records)


def _count_prev_round_games(
    series_by_team: dict, season: int, team: str, current_round: int
) -> int:
    """Count total games played in the previous round by a team."""
    key = (season, team)
    if key not in series_by_team:
        return 0
    for s in series_by_team[key]:
        if s.get("round", 0) == current_round - 1:
            return s.get("higher_seed_wins", 0) + s.get("lower_seed_wins", 0)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# LOW IMPACT
# ═══════════════════════════════════════════════════════════════════════════

# ── 8. Style-Matchup Features ──────────────────────────────────────────────

def build_style_features(team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team style vectors and matchup compatibility.
    Style dimensions: offensive, defensive, physical, possession, goaltending.
    """
    if team_features.empty:
        return pd.DataFrame()

    df = team_features.copy()
    style_cols: dict[str, list[str]] = {
        "offense": [],
        "defense": [],
        "physical": [],
        "possession": [],
    }

    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["goalsfor", "shotsfor", "xgoalsfor", "scoring"]):
            style_cols["offense"].append(c)
        elif any(k in cl for k in ["goalsagainst", "savepct", "gaa", "penalty"]):
            style_cols["defense"].append(c)
        elif any(k in cl for k in ["hit", "block", "takeaway", "giveaway", "penalty"]):
            style_cols["physical"].append(c)
        elif any(k in cl for k in ["corsi", "fenwick", "sat", "usat", "possession"]):
            style_cols["possession"].append(c)

    for style_name, cols in style_cols.items():
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            numeric = df[valid_cols].select_dtypes(include=[np.number])
            if not numeric.empty:
                # Normalize to z-scores within each season
                for season in df["season"].unique():
                    mask = df["season"] == season
                    season_data = numeric.loc[mask]
                    means = season_data.mean()
                    stds = season_data.std().replace(0, 1)
                    z_scores = (season_data - means) / stds
                    df.loc[mask, f"style_{style_name}_z"] = z_scores.mean(axis=1)

    keep_cols = ["team_abbrev", "season"] + [c for c in df.columns if c.startswith("style_")]
    result = df[keep_cols].copy() if any(c.startswith("style_") for c in df.columns) else pd.DataFrame()

    if not result.empty:
        logger.info("Built style features: %d team-seasons", len(result))
    return result


# ── 9. Conference / Division Features ───────────────────────────────────────

def build_conference_features(
    standings_df: pd.DataFrame, series_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add conference/division indicators for each series:
    - Same division? (more familiarity)
    - Cross-conference? (less familiarity, Cup Final only)
    """
    if standings_df.empty:
        return pd.DataFrame()

    # Build team -> conference/division mapping
    team_info: dict[tuple, dict] = {}
    for _, row in standings_df.iterrows():
        abbrev = row.get("teamAbbrev", {})
        if isinstance(abbrev, dict):
            abbrev = abbrev.get("default", "")
        if not abbrev:
            continue
        season = row.get("season", 0)
        conf = row.get("conferenceAbbrev", row.get("conferenceName", ""))
        div = row.get("divisionAbbrev", row.get("divisionName", ""))
        team_info[(str(abbrev), season)] = {"conference": str(conf), "division": str(div)}

    records: list[dict] = []
    for _, row in series_df.iterrows():
        season = row["season"]
        hi = row["higher_seed"]
        lo = row["lower_seed"]

        hi_info = team_info.get((hi, season), {})
        lo_info = team_info.get((lo, season), {})

        same_div = int(hi_info.get("division", "X") == lo_info.get("division", "Y"))
        same_conf = int(hi_info.get("conference", "X") == lo_info.get("conference", "Y"))

        records.append({
            "season": season,
            "higher_seed": hi,
            "lower_seed": lo,
            "div_same_division": same_div,
            "div_same_conference": same_conf,
            "div_cross_conference": 1 - same_conf,
        })

    result = pd.DataFrame(records)
    logger.info("Built conference features: %d series", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MASTER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def integrate_advanced_features(
    feature_df: pd.DataFrame,
    series_df: pd.DataFrame,
    skater_df: pd.DataFrame,
    recency_df: pd.DataFrame,
    h2h_df: pd.DataFrame,
    standings_df: pd.DataFrame,
    team_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Master function: build all advanced features and merge them
    into the existing feature matrix.
    """
    logger.info("=" * 60)
    logger.info("INTEGRATING ADVANCED V2 FEATURES")
    logger.info("=" * 60)

    df = feature_df.copy()
    original_cols = len(df.columns)

    # ── HIGH IMPACT ──

    # 1) Player-level features
    plyr = build_player_features(skater_df)
    if not plyr.empty:
        df = _merge_team_diff(df, plyr, "plyr_")

    # 1b) Elite skater PPG stack (player → team bridge for V2)
    skel = build_top_skater_strength_features(skater_df)
    if not skel.empty:
        df = _merge_team_diff(df, skel, "skel_")

    # 2) Recency features
    rec = build_recency_features(recency_df)
    if not rec.empty:
        df = _merge_team_diff(df, rec, "recent_")

    # 3) Head-to-head
    h2h = build_h2h_features(h2h_df, series_df)
    if not h2h.empty:
        merge_cols = [c for c in h2h.columns if c not in ("higher_seed", "lower_seed", "season")]
        df = df.merge(
            h2h[["season", "higher_seed", "lower_seed"] + merge_cols],
            on=["season", "higher_seed", "lower_seed"],
            how="left",
        )

    # 4) Round-specific
    round_feats = build_round_features(series_df)
    if not round_feats.empty:
        df = df.merge(
            round_feats,
            on=["season", "higher_seed", "lower_seed"],
            how="left",
            suffixes=("", "_round"),
        )

    # 5) Injury proxy
    injury = build_injury_proxy(skater_df)
    if not injury.empty:
        df = _merge_team_diff(df, injury, "inj_")

    # ── MEDIUM IMPACT ──

    # 6) Strength of schedule
    sos = build_sos_features(standings_df)
    if not sos.empty:
        df = _merge_team_diff(df, sos, "sos_")

    # 7) Rest days
    rest = build_rest_features(series_df)
    if not rest.empty:
        merge_cols = [c for c in rest.columns if c not in ("higher_seed", "lower_seed", "season")]
        df = df.merge(
            rest[["season", "higher_seed", "lower_seed"] + merge_cols],
            on=["season", "higher_seed", "lower_seed"],
            how="left",
        )

    # ── LOW IMPACT ──

    # 8) Style matchup (needs team_features to be passed)
    if team_features is not None and not team_features.empty:
        style = build_style_features(team_features)
        if not style.empty:
            df = _merge_team_diff(df, style, "style_")

    # 9) Conference/division
    conf = build_conference_features(standings_df, series_df)
    if not conf.empty:
        merge_cols = [c for c in conf.columns if c not in ("higher_seed", "lower_seed", "season")]
        df = df.merge(
            conf[["season", "higher_seed", "lower_seed"] + merge_cols],
            on=["season", "higher_seed", "lower_seed"],
            how="left",
        )

    new_cols = len(df.columns) - original_cols
    logger.info("Added %d new columns (total: %d)", new_cols, len(df.columns))

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_frac = df.isnull().mean()
    drop = nan_frac[nan_frac > 0.6].index.tolist()
    meta = ["season", "higher_seed", "lower_seed", "higher_seed_won"]
    drop = [c for c in drop if c not in meta]
    if drop:
        logger.info("Dropping %d high-NaN columns", len(drop))
        df = df.drop(columns=drop)

    df.to_csv(PROCESSED_DIR / "feature_matrix_v2.csv", index=False)
    logger.info("V2 feature matrix saved: %d rows x %d columns", len(df), len(df.columns))
    return df


def _merge_team_diff(
    feature_df: pd.DataFrame,
    team_df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge team-level features into the series-level feature matrix
    by computing higher_seed - lower_seed differentials.
    """
    if team_df.empty or "team_abbrev" not in team_df.columns:
        return feature_df

    numeric_cols = [
        c for c in team_df.columns
        if c not in ("team_abbrev", "season") and team_df[c].dtype in (np.float64, np.int64, float, int)
    ]

    if not numeric_cols:
        return feature_df

    hi = team_df.rename(columns={c: f"_hi_{c}" for c in numeric_cols})
    hi = hi.rename(columns={"team_abbrev": "higher_seed"})
    lo = team_df.rename(columns={c: f"_lo_{c}" for c in numeric_cols})
    lo = lo.rename(columns={"team_abbrev": "lower_seed"})

    merged = feature_df.merge(
        hi[["higher_seed", "season"] + [f"_hi_{c}" for c in numeric_cols]],
        on=["higher_seed", "season"], how="left",
    )
    merged = merged.merge(
        lo[["lower_seed", "season"] + [f"_lo_{c}" for c in numeric_cols]],
        on=["lower_seed", "season"], how="left",
    )

    for c in numeric_cols:
        col_name = f"diff_{c}" if c.startswith(prefix) else f"diff_{prefix}{c}"
        # Avoid duplicate column names
        if col_name in merged.columns:
            col_name = f"diff_{prefix}v2_{c}"
        hi_col = f"_hi_{c}"
        lo_col = f"_lo_{c}"
        if hi_col in merged.columns and lo_col in merged.columns:
            merged[col_name] = merged[hi_col] - merged[lo_col]

    # Drop temp columns
    temp = [c for c in merged.columns if c.startswith("_hi_") or c.startswith("_lo_")]
    merged = merged.drop(columns=temp)

    return merged
