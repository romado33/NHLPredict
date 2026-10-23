"""
Player-level feature engineering for playoff scoring prediction.

Builds features from regular-season stats and team-level predictions
to predict individual playoff point totals.

Feature categories:
  - Core scoring (PPG, GPG, APG)
  - Situational (EV/PP/SH splits)
  - Shooting / ice time
  - Position (forward, center, defenseman)
  - Team context (expected games, team offense, standings)
  - Historical playoff performance
  - Composite interactions
  - Age and experience
  - Position-specific playoff discount
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DIR, RAW_DIR
from src.features.engineer import normalize_team
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

PLAYOFF_DISCOUNT = {
    "C": 0.786, "L": 0.754, "R": 0.762, "D": 0.814,
    "F": 0.767,
}

# Empirical share of top-40 playoff scorer slots by team's deepest round (2001-2025, excl. 2005)
HIST_TOP40_ROUND_MIX_PCT = {
    "R1": 3.1,
    "R2": 23.8,
    "CF": 30.1,
    "F": 43.0,
}


def series_letter_to_playoff_round(letter: str) -> int:
    """Map playoff series letter (NHL bracket file) to round index 1..4."""
    L = str(letter).upper()
    if L in "ABCDEFGH":
        return 1
    if L in "IJKL":
        return 2
    if L in "MN":
        return 3
    if L == "O":
        return 4
    return 0


def compute_team_round_reach_actuals(
    series_df: pd.DataFrame,
) -> dict[tuple[int, str], tuple[float, float, float]]:
    """
    Per (season, team), actual 0/1 indicators: reached R2+, CF+, Cup Final.
    Training aligns with inference where bracket model supplies probabilities.
    """
    if series_df.empty:
        return {}

    out: dict[tuple[int, str], tuple[float, float, float]] = {}
    for season, sdf in series_df.groupby("season"):
        depth: dict[str, int] = defaultdict(int)
        for _, r in sdf.iterrows():
            rnd = series_letter_to_playoff_round(str(r.get("series_letter", "")))
            hi = str(r.get("higher_seed", "") or "")
            lo = str(r.get("lower_seed", "") or "")
            for t in (hi, lo):
                if t:
                    depth[t] = max(depth[t], rnd)
        sy = int(season)
        for team, d in depth.items():
            pr2 = 1.0 if d >= 2 else 0.0
            pcf = 1.0 if d >= 3 else 0.0
            pfn = 1.0 if d >= 4 else 0.0
            out[(sy, normalize_team(team))] = (pr2, pcf, pfn)
    return out


def round_mix_bucket_from_probs(p_r2: float, p_cf: float, p_final: float) -> str:
    """Most likely exit depth from nested probabilities (p_r2 >= p_cf >= p_final)."""
    p_r1 = max(0.0, 1.0 - p_r2)
    p_r2_only = max(0.0, p_r2 - p_cf)
    p_cf_only = max(0.0, p_cf - p_final)
    p_f = max(0.0, p_final)
    probs = {"R1": p_r1, "R2": p_r2_only, "CF": p_cf_only, "F": p_f}
    return max(probs, key=probs.get)


def _normalize_team_col(df: pd.DataFrame) -> pd.DataFrame:
    """Find and normalize the team abbreviation column."""
    df = df.copy()
    for c in ("teamAbbrevs", "teamFullName", "teamName", "lastTeamAbbrev"):
        if c in df.columns:
            df["team_abbrev"] = df[c].apply(
                lambda x: normalize_team(str(x).split(",")[0].strip())
                if pd.notna(x) else ""
            )
            return df
    if "team_abbrev" not in df.columns:
        logger.warning("No team column found in DataFrame")
        df["team_abbrev"] = ""
    return df


def build_training_dataset(
    reg_season_df: pd.DataFrame,
    playoff_df: pd.DataFrame,
    series_df: pd.DataFrame,
    expected_games_by_season: dict[int, dict[str, float]] | None = None,
    player_bios_df: pd.DataFrame | None = None,
    team_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the full training dataset: one row per player-season for players
    who appeared in the playoffs.

    Features come from regular-season stats.
    Targets come from playoff stats.
    """
    if reg_season_df.empty or playoff_df.empty:
        logger.error("Empty input DataFrames")
        return pd.DataFrame()

    reg = reg_season_df.copy()
    po = playoff_df.copy()

    if "skater_report" in reg.columns:
        reg = reg[reg["skater_report"] == "summary"].copy()
    if "skater_report" in po.columns:
        po = po[po["skater_report"] == "summary"].copy()

    reg = _normalize_team_col(reg)
    po = _normalize_team_col(po)

    bios_lookup = _build_bios_lookup(player_bios_df)
    team_stats_lookup = _build_team_stats_lookup(team_stats_df, reg)
    team_playoff_games = _compute_team_playoff_games(series_df)
    opponent_quality = _compute_opponent_quality(series_df, reg)
    team_round_actuals = compute_team_round_reach_actuals(series_df)

    # Fix train-test distribution shift: generate noisy expected games for
    # historical seasons so the model trains on values that resemble the
    # team model's predictions at inference time (not exact actuals).
    noisy_expected = _generate_noisy_expected_games(team_playoff_games)
    if expected_games_by_season:
        noisy_expected.update(expected_games_by_season)

    records: list[dict] = []
    seasons = sorted(set(reg["season"].unique()) & set(po["season"].unique()))

    for season in seasons:
        reg_s = reg[reg["season"] == season]
        po_s = po[po["season"] == season]

        if po_s.empty:
            continue

        prior_playoffs = po[po["season"] < season]
        team_ranks = _compute_team_scoring_ranks(reg_s)

        for _, po_row in po_s.iterrows():
            pid = po_row.get("playerId")
            if pd.isna(pid):
                continue

            reg_match = reg_s[reg_s["playerId"] == pid]
            if reg_match.empty:
                continue
            reg_row = reg_match.iloc[0]

            team = po_row.get("team_abbrev", "")
            if not team:
                team = reg_row.get("team_abbrev", "")
            team_n = normalize_team(team) if team else ""
            pr2, pcf, pfn = team_round_actuals.get(
                (int(season), team_n), (0.0, 0.0, 0.0)
            )

            features = _extract_player_features(
                reg_row, po_row, team, season,
                team_playoff_games, prior_playoffs,
                noisy_expected,
                bios_lookup=bios_lookup,
                team_stats_lookup=team_stats_lookup,
                opponent_quality_lookup=opponent_quality,
                team_prob_reach_r2=pr2,
                team_prob_reach_cf=pcf,
                team_prob_reach_final=pfn,
            )
            if features:
                rank_info = team_ranks.get(pid, {})
                features["team_pts_rank"] = rank_info.get("rank", 20)
                features["team_pts_share"] = rank_info.get("share", 0.0)
                features["is_top3_team_scorer"] = 1 if rank_info.get("rank", 20) <= 3 else 0
                features["is_top6_team_scorer"] = 1 if rank_info.get("rank", 20) <= 6 else 0
                records.append(features)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No training records built")
        return df

    df = df.replace([np.inf, -np.inf], np.nan)
    logger.info("Built training dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


def _extract_player_features(
    reg_row: pd.Series,
    po_row: pd.Series,
    team: str,
    season: int,
    team_playoff_games: dict,
    prior_playoffs: pd.DataFrame,
    expected_games_by_season: dict | None,
    bios_lookup: dict | None = None,
    team_stats_lookup: dict | None = None,
    opponent_quality_lookup: dict | None = None,
    team_prob_reach_r2: float = 0.0,
    team_prob_reach_cf: float = 0.0,
    team_prob_reach_final: float = 0.0,
) -> dict | None:
    """Extract all features for a single player-season."""
    pid = reg_row.get("playerId")
    if pd.isna(pid):
        return None

    reg_gp = _safe_float(reg_row.get("gamesPlayed", 0))
    if reg_gp < 10:
        return None

    # ── Target variables (from playoff data) ──
    po_points = _safe_float(po_row.get("points", 0))
    po_goals = _safe_float(po_row.get("goals", 0))
    po_assists = _safe_float(po_row.get("assists", 0))
    po_gp = _safe_float(po_row.get("gamesPlayed", 0))

    # ── Team expected games ──
    team_exp_games = 0.0
    if expected_games_by_season and season in expected_games_by_season:
        team_exp_games = expected_games_by_season[season].get(team, 0)
    if team_exp_games == 0:
        team_exp_games = team_playoff_games.get((season, team), po_gp)

    # ── Core scoring features (regular season) ──
    reg_pts = _safe_float(reg_row.get("points", 0))
    reg_goals = _safe_float(reg_row.get("goals", 0))
    reg_assists = _safe_float(reg_row.get("assists", 0))
    reg_ppg = reg_pts / max(reg_gp, 1)
    reg_gpg = reg_goals / max(reg_gp, 1)
    reg_apg = reg_assists / max(reg_gp, 1)

    ev_pts = _safe_float(reg_row.get("evPoints", 0))
    pp_pts = _safe_float(reg_row.get("ppPoints", 0))
    sh_pts = _safe_float(reg_row.get("shPoints", 0))
    pp_goals = _safe_float(reg_row.get("ppGoals", 0))
    sh_goals = _safe_float(reg_row.get("shGoals", 0))
    ev_goals = _safe_float(reg_row.get("evGoals", 0))

    shots = _safe_float(reg_row.get("shots", 0))
    shoot_pct = _safe_float(reg_row.get("shootingPct", 0))
    toi_per_game = _safe_float(reg_row.get("timeOnIcePerGame", 0))
    gwg = _safe_float(reg_row.get("gameWinningGoals", 0))
    plus_minus = _safe_float(reg_row.get("plusMinus", 0))

    # Position
    pos = str(reg_row.get("positionCode", "")).upper()
    is_forward = 1 if pos in ("C", "L", "R", "F") else 0
    is_center = 1 if pos == "C" else 0
    is_defenseman = 1 if pos == "D" else 0

    # ── Age (from bios lookup) ──
    age = np.nan
    if bios_lookup:
        bio = bios_lookup.get(int(pid))
        if bio and bio.get("birthDate"):
            try:
                birth = pd.Timestamp(bio["birthDate"])
                playoff_date = pd.Timestamp(f"{season}-04-15")
                age = (playoff_date - birth).days / 365.25
            except Exception:
                pass

    # ── Position-specific playoff scoring discount ──
    discount = PLAYOFF_DISCOUNT.get(pos, 0.77)
    playoff_discount_ppg = reg_ppg * discount * team_exp_games

    # ── Team offensive context ──
    team_gf_per_gp = np.nan
    team_pp_pct = np.nan
    team_shots_per_gp = np.nan
    team_standings_pts = np.nan
    if team_stats_lookup:
        tst = team_stats_lookup.get((team, season), {})
        team_gf_per_gp = tst.get("gf_per_gp", np.nan)
        team_pp_pct = tst.get("pp_pct", np.nan)
        team_shots_per_gp = tst.get("shots_per_gp", np.nan)
        team_standings_pts = tst.get("standings_pts", np.nan)

    # ── Historical playoff performance ──
    prior_po_ppg = 0.0
    prior_po_gp = 0.0
    prior_po_pts = 0.0
    career_po_seasons = 0

    if not prior_playoffs.empty:
        player_prior = prior_playoffs[prior_playoffs["playerId"] == pid]
        if not player_prior.empty:
            prior_po_pts = player_prior["points"].sum()
            prior_po_gp = player_prior["gamesPlayed"].sum()
            prior_po_ppg = prior_po_pts / max(prior_po_gp, 1)
            career_po_seasons = player_prior["season"].nunique()

    # ── Computed features ──
    pts_x_games = reg_ppg * team_exp_games
    goals_x_games = reg_gpg * team_exp_games
    shots_per_game = shots / max(reg_gp, 1)

    pp_pts_share = pp_pts / max(reg_pts, 1)
    ev_pts_share = ev_pts / max(reg_pts, 1)

    assists_x_games = reg_apg * team_exp_games
    pp_pts_x_games = (pp_pts / max(reg_gp, 1)) * team_exp_games
    toi_x_games = toi_per_game * team_exp_games
    prior_po_ppg_x_games = prior_po_ppg * team_exp_games
    shots_x_games = shots_per_game * team_exp_games

    fwd_ppg = reg_ppg * is_forward
    def_ppg = reg_ppg * is_defenseman

    # ── Age interactions ──
    age_sq = age ** 2 if not np.isnan(age) else np.nan
    age_x_exp = age * team_exp_games if not np.isnan(age) else np.nan
    age_x_ppg = age * reg_ppg if not np.isnan(age) else np.nan

    # ── Recent form / availability proxies ──
    gp_availability = reg_gp / 82.0
    ppg_per_toi = reg_ppg / max(toi_per_game / 1200.0, 0.1) if toi_per_game > 0 else 0

    # ── Deep-run attenuation ──
    # Scoring rates drop in later playoff rounds as opponents improve.
    # Model this with sqrt(expected_games) interaction which has diminishing
    # returns: doubling from 6 to 12 games doesn't double scoring.
    exp_games_sqrt = np.sqrt(max(team_exp_games, 0))
    pts_x_sqrt_games = reg_ppg * exp_games_sqrt
    toi_x_sqrt_games = toi_per_game * exp_games_sqrt

    # Ratio feature: how much of the expected games are "deep rounds"?
    # First round ~ 5.5 games. Beyond that is increasingly hard scoring.
    deep_run_ratio = max(team_exp_games - 5.5, 0) / max(team_exp_games, 1)

    pr2 = float(np.clip(team_prob_reach_r2, 0.0, 1.0))
    pcf = float(np.clip(team_prob_reach_cf, 0.0, 1.0))
    pfn = float(np.clip(team_prob_reach_final, 0.0, 1.0))
    pts_x_prob_final = reg_ppg * pfn

    # ── Opponent quality ──
    opp_quality = 0.0
    opp_ga_rate = 0.0
    if opponent_quality_lookup:
        oq = opponent_quality_lookup.get((season, team), {})
        opp_quality = oq.get("opp_pts_pace", 0.0)
        opp_ga_rate = oq.get("opp_ga_per_gp", 0.0)

    return {
        # Identity / metadata
        "playerId": int(pid),
        "playerName": reg_row.get("skaterFullName", ""),
        "team_abbrev": team,
        "season": season,
        "positionCode": pos,

        # Target
        "target_points": po_points,
        "target_goals": po_goals,
        "target_assists": po_assists,
        "target_gp": po_gp,

        # Core scoring
        "reg_points": reg_pts,
        "reg_goals": reg_goals,
        "reg_assists": reg_assists,
        "reg_gp": reg_gp,
        "reg_ppg": reg_ppg,
        "reg_gpg": reg_gpg,
        "reg_apg": reg_apg,

        # Situational
        "reg_ev_points": ev_pts,
        "reg_pp_points": pp_pts,
        "reg_sh_points": sh_pts,
        "reg_pp_goals": pp_goals,
        "reg_sh_goals": sh_goals,
        "reg_ev_goals": ev_goals,

        # Shooting / ice time
        "reg_shots": shots,
        "reg_shots_per_game": shots_per_game,
        "reg_shooting_pct": shoot_pct,
        "reg_toi_per_game": toi_per_game,
        "reg_gwg": gwg,
        "reg_plus_minus": plus_minus,

        # Position
        "is_forward": is_forward,
        "is_center": is_center,
        "is_defenseman": is_defenseman,

        # Team context
        "team_expected_games": team_exp_games,
        "team_prob_reach_r2": pr2,
        "team_prob_reach_cf": pcf,
        "team_prob_reach_final": pfn,
        "pts_x_prob_final": pts_x_prob_final,

        # Historical playoff
        "prior_po_ppg": prior_po_ppg,
        "prior_po_gp": prior_po_gp,
        "prior_po_pts": prior_po_pts,
        "career_po_seasons": career_po_seasons,

        # Composite / interaction
        "pts_x_expected_games": pts_x_games,
        "goals_x_expected_games": goals_x_games,
        "assists_x_expected_games": assists_x_games,
        "pp_pts_x_expected_games": pp_pts_x_games,
        "toi_x_expected_games": toi_x_games,
        "shots_x_expected_games": shots_x_games,
        "prior_po_ppg_x_games": prior_po_ppg_x_games,
        "pp_pts_share": pp_pts_share,
        "ev_pts_share": ev_pts_share,
        "fwd_ppg": fwd_ppg,
        "def_ppg": def_ppg,

        # NEW: Age
        "age": age,
        "age_squared": age_sq,

        # NEW: Position-specific playoff discount
        "playoff_discount_ppg": playoff_discount_ppg,

        # NEW: Team offensive context
        "team_gf_per_gp": team_gf_per_gp,
        "team_pp_pct": team_pp_pct,
        "team_shots_per_gp": team_shots_per_gp,
        "team_standings_pts": team_standings_pts,

        # NEW: Age interactions
        "age_x_expected_games": age_x_exp,
        "age_x_ppg": age_x_ppg,

        # NEW: Recent form / availability proxies
        "gp_availability": gp_availability,
        "ppg_per_toi": ppg_per_toi,
        # Deep-run attenuation
        "exp_games_sqrt": exp_games_sqrt,
        "pts_x_sqrt_games": pts_x_sqrt_games,
        "toi_x_sqrt_games": toi_x_sqrt_games,
        "deep_run_ratio": deep_run_ratio,
        # Opponent quality
        "opp_pts_pace": opp_quality,
        "opp_ga_per_gp": opp_ga_rate,
    }


def build_prediction_features(
    current_reg_df: pd.DataFrame,
    team_expected_games: dict[str, float],
    playoff_teams: list[str],
    prior_playoff_df: pd.DataFrame | None = None,
    season: int = 2026,
    player_bios_df: pd.DataFrame | None = None,
    team_stats_df: pd.DataFrame | None = None,
    team_probs: dict[str, dict] | None = None,
    team_round_reach_probs: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Build features for prediction on current-season players.
    Only includes players on playoff/contender teams.
    """
    reg = current_reg_df.copy()
    if "skater_report" in reg.columns:
        reg = reg[reg["skater_report"] == "summary"].copy()
    reg = _normalize_team_col(reg)

    reg = reg[reg["team_abbrev"].isin(playoff_teams)].copy()

    if reg.empty:
        logger.error("No players found on playoff teams")
        return pd.DataFrame()

    bios_lookup = _build_bios_lookup(player_bios_df)
    team_stats_lookup = _build_team_stats_lookup(team_stats_df, reg)

    # Build opponent quality for prediction year
    pred_opp_quality = _compute_prediction_opponent_quality(
        team_expected_games, team_probs, season
    )

    records: list[dict] = []
    prior = prior_playoff_df if prior_playoff_df is not None else pd.DataFrame()
    if "skater_report" in prior.columns:
        prior = prior[prior["skater_report"] == "summary"]

    team_ranks = _compute_team_scoring_ranks(reg)

    for _, row in reg.iterrows():
        pid = row.get("playerId")
        if pd.isna(pid):
            continue

        team = row.get("team_abbrev", "")
        team_exp = team_expected_games.get(team, 5.0)

        po_row = pd.Series({"gamesPlayed": 0, "points": 0, "goals": 0, "assists": 0})

        trp = team_round_reach_probs or {}
        rp = trp.get(team, trp.get(normalize_team(team), {}))
        pr2 = float(rp.get("p_r2", 0.0))
        pcf = float(rp.get("p_cf", 0.0))
        pfn = float(rp.get("p_final", 0.0))

        features = _extract_player_features(
            row, po_row, team, season,
            {}, prior, {season: team_expected_games},
            bios_lookup=bios_lookup,
            team_stats_lookup=team_stats_lookup,
            opponent_quality_lookup=pred_opp_quality,
            team_prob_reach_r2=pr2,
            team_prob_reach_cf=pcf,
            team_prob_reach_final=pfn,
        )
        if features:
            rank_info = team_ranks.get(pid, {})
            features["team_pts_rank"] = rank_info.get("rank", 20)
            features["team_pts_share"] = rank_info.get("share", 0.0)
            features["is_top3_team_scorer"] = 1 if rank_info.get("rank", 20) <= 3 else 0
            features["is_top6_team_scorer"] = 1 if rank_info.get("rank", 20) <= 6 else 0
            features["target_points"] = np.nan
            features["target_goals"] = np.nan
            features["target_assists"] = np.nan
            features["target_gp"] = np.nan
            records.append(features)

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan)
    logger.info("Built prediction features: %d players on %d teams",
                len(df), df["team_abbrev"].nunique())
    return df


def _build_bios_lookup(
    player_bios_df: pd.DataFrame | None,
) -> dict[int, dict] | None:
    """Build a playerId -> {birthDate, draftYear, ...} lookup dict."""
    if player_bios_df is None or player_bios_df.empty:
        return None
    lookup = {}
    for _, row in player_bios_df.iterrows():
        pid = row.get("playerId")
        if pd.notna(pid):
            lookup[int(pid)] = row.to_dict()
    return lookup


def _build_team_stats_lookup(
    team_stats_df: pd.DataFrame | None,
    reg_df: pd.DataFrame | None = None,
) -> dict[tuple, dict] | None:
    """
    Build a (team_abbrev, season) -> {gf_per_gp, pp_pct, shots_per_gp, standings_pts}
    lookup from team stats or aggregated regular season data.
    """
    if team_stats_df is not None and not team_stats_df.empty:
        lookup = {}
        for _, row in team_stats_df.iterrows():
            team = str(row.get("team_abbrev", ""))
            season = row.get("season", 0)
            if team and season:
                lookup[(team, season)] = {
                    "gf_per_gp": _safe_float(row.get("gf_per_gp", np.nan)),
                    "pp_pct": _safe_float(row.get("pp_pct", np.nan)),
                    "shots_per_gp": _safe_float(row.get("shots_per_gp", np.nan)),
                    "standings_pts": _safe_float(row.get("standings_pts", np.nan)),
                }
        return lookup if lookup else None

    if reg_df is not None and not reg_df.empty:
        lookup = {}
        reg = reg_df.copy()
        reg = _normalize_team_col(reg)
        for (team, season), group in reg.groupby(["team_abbrev", "season"]):
            if not team:
                continue
            total_goals = group["goals"].sum() if "goals" in group.columns else 0
            total_gp = group["gamesPlayed"].max() if "gamesPlayed" in group.columns else 82
            total_shots = group["shots"].sum() if "shots" in group.columns else 0
            total_pp_goals = group["ppGoals"].sum() if "ppGoals" in group.columns else 0
            total_pp_pts = group["ppPoints"].sum() if "ppPoints" in group.columns else 0
            n_players = len(group)
            gf_per_gp = total_goals / max(total_gp, 1)
            shots_per_gp = total_shots / max(total_gp * n_players / 20, 1)
            pp_pct = total_pp_goals / max(total_goals, 1) if total_goals > 0 else 0.2
            lookup[(team, season)] = {
                "gf_per_gp": gf_per_gp,
                "pp_pct": pp_pct,
                "shots_per_gp": shots_per_gp,
                "standings_pts": np.nan,
            }
        return lookup if lookup else None

    return None


def _compute_team_scoring_ranks(reg_s: pd.DataFrame) -> dict:
    """
    For each player, compute their rank within their team by points
    and their share of team total points.
    """
    result = {}
    reg_s = reg_s.copy()
    reg_s = _normalize_team_col(reg_s)

    for team, group in reg_s.groupby("team_abbrev"):
        if not team:
            continue
        sorted_g = group.sort_values("points", ascending=False)
        team_total = sorted_g["points"].sum()
        for rank_idx, (_, row) in enumerate(sorted_g.iterrows()):
            pid = row.get("playerId")
            if pd.notna(pid):
                result[pid] = {
                    "rank": rank_idx + 1,
                    "share": float(row.get("points", 0)) / max(team_total, 1),
                }
    return result


def _compute_prediction_opponent_quality(
    team_expected_games: dict[str, float],
    team_probs: dict[str, dict] | None,
    season: int,
) -> dict[tuple, dict[str, float]]:
    """
    Compute opponent quality for the prediction year based on standings data.

    Uses pts_per_game from standings as a proxy for opponent strength.
    Teams with more expected games face tougher opponents on average (they
    advance further), so we scale opponent quality by the team's expected depth.
    """
    result: dict[tuple, dict[str, float]] = {}

    if not team_probs:
        return result

    all_teams_ppg = {t: info.get("pts_per_game", 1.0) for t, info in team_probs.items()}
    mean_ppg = np.mean(list(all_teams_ppg.values())) if all_teams_ppg else 1.0

    playoff_teams_ppg = {
        t: ppg for t, ppg in all_teams_ppg.items()
        if team_probs.get(t, {}).get("in_playoff_position", False)
    }
    playoff_avg_ppg = np.mean(list(playoff_teams_ppg.values())) if playoff_teams_ppg else mean_ppg

    for team in team_expected_games:
        team_ppg = all_teams_ppg.get(team, mean_ppg)
        # Teams that go deeper face opponents closer to the league's best
        exp_games = team_expected_games.get(team, 5.0)
        depth_factor = min(exp_games / 20.0, 1.0)
        opp_quality = playoff_avg_ppg * (1 + depth_factor * 0.15)

        result[(season, team)] = {
            "opp_pts_pace": round(float(opp_quality), 4),
            "opp_ga_per_gp": round(float(opp_quality), 4),
        }

    return result


def _compute_opponent_quality(
    series_df: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> dict[tuple, dict[str, float]]:
    """
    For each (season, team) that appeared in the playoffs, compute the
    average quality of their opponents based on regular-season metrics.

    Returns dict[(season, team)] -> {opp_pts_pace, opp_ga_per_gp}.
    """
    if series_df.empty:
        return {}

    # Build per-season team stats from regular season
    team_stats_by_season: dict[tuple, dict] = {}
    for season in reg_df["season"].unique():
        season_df = reg_df[reg_df["season"] == season]
        for team in season_df["team_abbrev"].unique():
            if not team:
                continue
            team_players = season_df[season_df["team_abbrev"] == team]
            total_gp = team_players["gamesPlayed"].max() if "gamesPlayed" in team_players.columns else 0
            total_goals = team_players["goals"].sum() if "goals" in team_players.columns else 0
            pts_col = "points" if "points" in team_players.columns else None
            total_pts = team_players[pts_col].sum() if pts_col else 0
            team_stats_by_season[(season, team)] = {
                "gp": max(total_gp, 1),
                "goals_for": total_goals,
                "player_pts": total_pts,
            }

    # For each series, map team -> opponents and average their quality
    team_opponents: dict[tuple, list[str]] = {}
    for _, row in series_df.iterrows():
        season = row["season"]
        hi = row.get("higher_seed", "")
        lo = row.get("lower_seed", "")
        if hi and lo:
            team_opponents.setdefault((season, hi), []).append(lo)
            team_opponents.setdefault((season, lo), []).append(hi)

    result: dict[tuple, dict[str, float]] = {}
    for (season, team), opponents in team_opponents.items():
        opp_pts_paces = []
        opp_ga_rates = []
        for opp in opponents:
            opp_stats = team_stats_by_season.get((season, opp))
            if opp_stats:
                gp = opp_stats["gp"]
                opp_pts_paces.append(opp_stats["goals_for"] / gp)
                # GA approximated as inverse of scoring (opponents score against them)
                opp_ga_rates.append(opp_stats["goals_for"] / gp)
        result[(season, team)] = {
            "opp_pts_pace": float(np.mean(opp_pts_paces)) if opp_pts_paces else 0.0,
            "opp_ga_per_gp": float(np.mean(opp_ga_rates)) if opp_ga_rates else 0.0,
        }

    return result


def _compute_team_playoff_games(series_df: pd.DataFrame) -> dict[tuple, float]:
    """
    From historical series results, compute total playoff games each team
    played in each season.
    """
    if series_df.empty:
        return {}

    result: dict[tuple, float] = {}
    for _, row in series_df.iterrows():
        season = row["season"]
        hi = row["higher_seed"]
        lo = row["lower_seed"]
        hi_w = row.get("higher_seed_wins", 0)
        lo_w = row.get("lower_seed_wins", 0)
        games = hi_w + lo_w

        key_hi = (season, hi)
        key_lo = (season, lo)
        result[key_hi] = result.get(key_hi, 0) + games
        result[key_lo] = result.get(key_lo, 0) + games

    return result


def _generate_noisy_expected_games(
    team_playoff_games: dict[tuple, float],
    noise_sigma_per_round: float = 1.8,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """
    Generate noisy expected games for historical seasons to align
    training features with the distribution seen at inference time.

    The team model outputs probability-weighted expected games which are
    smooth, continuous values (e.g., 10.2 instead of exactly 12). We
    simulate this by adding calibrated Gaussian noise to actual playoff
    games, scaled by the number of rounds the team played.

    This fixes the train-test distribution shift where the model trains
    on exact actuals but predicts using noisy model outputs.
    """
    rng = np.random.RandomState(seed)
    by_season: dict[int, dict[str, float]] = {}

    for (season, team), actual_games in team_playoff_games.items():
        estimated_rounds = max(actual_games / 5.5, 1)
        noise_std = noise_sigma_per_round * np.sqrt(estimated_rounds)
        noise = rng.normal(0, noise_std)
        noisy = actual_games + noise
        noisy = np.clip(noisy, 4.0, 28.0)

        if season not in by_season:
            by_season[season] = {}
        by_season[season][team] = round(float(noisy), 1)

    return by_season


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default
