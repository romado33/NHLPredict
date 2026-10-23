"""
Roster verification and injury detection for the scoring pipeline.

Uses the NHL Web API to:
- Fetch current rosters for verification
- Detect potentially injured/inactive players via game log recency
- Compute playoff probability for bubble teams based on points pace

Uses ESPN's public NHL injuries feed for up-to-date injury status, type,
and expected return (when available). The NHL's own APIs do not expose
a stable injury report endpoint.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import requests

from config.settings import PROJECT_ROOT
from src.data.nhl_api_client import NHLAPIClient
from src.data.player_health import espn_playoff_production_multiplier
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

TOTAL_REGULAR_SEASON_GAMES = 82
DAYS_INACTIVE_THRESHOLD = 14

ESPN_NHL_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
)
ESPN_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

INJURY_OVERRIDES_PATH = PROJECT_ROOT / "config" / "injury_overrides.json"


def load_injury_overrides_config() -> dict[str, Any]:
    if not INJURY_OVERRIDES_PATH.exists():
        return {}
    with open(INJURY_OVERRIDES_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_espn_rest_maintenance(row: dict[str, Any]) -> bool:
    """
    ESPN often lists end-of-season maintenance as Day-To-Day + Rest — not a
    playoff injury for modeling purposes.
    """
    st = (row.get("espn_injury_status") or "").lower()
    it = (row.get("espn_injury_type") or "").lower()
    cm = (row.get("espn_injury_comment") or "").lower()
    if "rest" not in it and "rest" not in cm:
        return False
    if "day" in st or "day-to-day" in st or "day to day" in st:
        return True
    return False


def _clear_espn_injury_columns(pred_df: pd.DataFrame, idx: Any) -> None:
    for col in (
        "espn_injury_status",
        "espn_injury_type",
        "espn_injury_comment",
        "espn_injury_return_date",
        "espn_injury_updated",
    ):
        if col in pred_df.columns:
            pred_df.at[idx, col] = ""


def apply_injury_context_overrides(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply config/injury_overrides.json: strip misleading ESPN rows (e.g. Rest),
    and per-player playoff production multipliers from beat reporting.
    """
    pred_df = pred_df.copy()
    pred_df["injury_override_mult"] = np.nan
    pred_df["injury_override_note"] = ""
    pred_df["skip_game_log_inactive"] = False

    cfg = load_injury_overrides_config()
    strip_rest = bool(
        (cfg.get("strip_espn_when") or {}).get("rest_maintenance", True)
    )

    if strip_rest:
        for idx in pred_df.index:
            row = pred_df.loc[idx].to_dict()
            if not is_espn_rest_maintenance(row):
                continue
            _clear_espn_injury_columns(pred_df, idx)
            pred_df.at[idx, "injury_report_source"] = "Override (ESPN Rest cleared)"

    for p in cfg.get("players", []) or []:
        pid = p.get("playerId")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        mask = pred_df["playerId"] == pid if "playerId" in pred_df.columns else pd.Series(False, index=pred_df.index)
        for idx in pred_df.index[mask]:
            if p.get("clear_espn_injury"):
                _clear_espn_injury_columns(pred_df, idx)
            if p.get("skip_game_log_inactive"):
                pred_df.at[idx, "skip_game_log_inactive"] = True
            mult = p.get("playoff_production_multiplier")
            if mult is not None:
                pred_df.at[idx, "injury_override_mult"] = float(mult)
            reason = (p.get("reason") or "").strip()
            if reason:
                pred_df.at[idx, "injury_override_note"] = reason
            pred_df.at[idx, "injury_report_source"] = "Manual override"
    return pred_df


def fetch_live_standings(client: NHLAPIClient) -> list[dict[str, Any]]:
    """Fetch current standings with near-zero cache TTL."""
    data = client._web("/standings/now")
    return data.get("standings", [])


def compute_playoff_probabilities(
    standings: list[dict],
) -> dict[str, dict[str, Any]]:
    """
    For each team, compute a simplified playoff probability based on
    current points, points pace, and distance from the cutline.

    Returns dict keyed by team abbreviation with:
    - points, gp, games_remaining, points_pace
    - conf_rank, in_playoff_position (bool)
    - playoff_probability (0.0 - 1.0)
    - conference, division
    """
    result = {}

    for conf_name in ("Eastern", "Western"):
        conf_teams = [
            t for t in standings if t.get("conferenceName") == conf_name
        ]
        conf_teams.sort(key=lambda t: t.get("conferenceSequence", 99))

        cutline_pts = 0
        if len(conf_teams) >= 8:
            cutline_pts = conf_teams[7].get("points", 0)

        for t in conf_teams:
            abbrev = t.get("teamAbbrev", {})
            if isinstance(abbrev, dict):
                abbrev = abbrev.get("default", "")
            else:
                abbrev = str(abbrev)

            pts = t.get("points", 0)
            gp = t.get("gamesPlayed", 0)
            w = t.get("wins", 0)
            l = t.get("losses", 0)
            otl = t.get("otLosses", 0)
            conf_rank = t.get("conferenceSequence", 99)
            div = t.get("divisionAbbrev", "?")
            div_rank = t.get("divisionSequence", 99)

            games_remaining = max(TOTAL_REGULAR_SEASON_GAMES - gp, 0)
            pts_per_game = pts / max(gp, 1)
            projected_pts = pts + pts_per_game * games_remaining

            in_position = conf_rank <= 8
            pts_from_cutline = pts - cutline_pts

            # Simplified playoff probability using logistic-style curve
            # based on distance from cutline and games remaining
            if in_position:
                if pts_from_cutline >= 15:
                    prob = 0.99
                elif pts_from_cutline >= 10:
                    prob = 0.95
                elif pts_from_cutline >= 5:
                    prob = 0.85
                elif pts_from_cutline >= 2:
                    prob = 0.70
                else:
                    prob = 0.60
            else:
                gap = abs(pts_from_cutline)
                max_possible_gain = games_remaining * 2
                if max_possible_gain <= 0:
                    prob = 0.0
                elif gap > max_possible_gain:
                    prob = 0.0
                else:
                    prob = max(0.0, 0.50 - gap * 0.07)
                    if games_remaining < 5:
                        prob *= 0.3

            result[abbrev] = {
                "points": pts,
                "gp": gp,
                "wins": w,
                "losses": l,
                "otl": otl,
                "games_remaining": games_remaining,
                "pts_per_game": round(pts_per_game, 3),
                "projected_pts": round(projected_pts, 1),
                "conference": conf_name,
                "division": div,
                "conf_rank": conf_rank,
                "div_rank": div_rank,
                "in_playoff_position": in_position,
                "pts_from_cutline": pts_from_cutline,
                "playoff_probability": round(prob, 3),
            }

    return result


def identify_contender_teams(
    team_probs: dict[str, dict],
    min_probability: float = 0.05,
) -> tuple[list[str], list[str]]:
    """
    Identify playoff teams and bubble/contender teams.

    Returns:
        (playoff_teams, bubble_teams) — both as lists of abbreviations
    """
    playoff = []
    bubble = []

    for team, info in team_probs.items():
        if info["in_playoff_position"]:
            playoff.append(team)
        elif info["playoff_probability"] >= min_probability:
            bubble.append(team)

    playoff.sort(key=lambda t: team_probs[t]["conf_rank"])
    bubble.sort(key=lambda t: -team_probs[t]["playoff_probability"])

    return playoff, bubble


def fetch_team_roster(
    client: NHLAPIClient, team_abbrev: str
) -> dict[int, dict[str, Any]]:
    """
    Fetch the current roster for a team.
    Returns dict keyed by player ID.
    """
    try:
        data = client._web(f"/roster/{team_abbrev}/current")
    except Exception as e:
        logger.warning("Failed to fetch roster for %s: %s", team_abbrev, e)
        return {}

    players = {}
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            pid = p.get("id")
            if pid:
                first = p.get("firstName", {})
                last = p.get("lastName", {})
                fname = first.get("default", "") if isinstance(first, dict) else str(first)
                lname = last.get("default", "") if isinstance(last, dict) else str(last)
                players[pid] = {
                    "name": f"{fname} {lname}".strip(),
                    "position": p.get("positionCode", ""),
                    "team": team_abbrev,
                    "sweater": p.get("sweaterNumber", 0),
                    "birthDate": p.get("birthDate", ""),
                }
    return players


def check_player_activity(
    client: NHLAPIClient,
    player_id: int,
    season_id: str = "20252026",
    game_type: int = 2,
) -> dict[str, Any]:
    """
    Check a player's recent game activity to detect possible injuries.
    Uses the game log endpoint.
    """
    try:
        data = client._web(f"/player/{player_id}/game-log/{season_id}/{game_type}")
        logs = data.get("gameLog", [])
    except Exception:
        return {"active": True, "games_logged": 0, "last_game_date": None, "days_since_last_game": None}

    if not logs:
        return {"active": False, "games_logged": 0, "last_game_date": None, "days_since_last_game": None}

    last_date_str = logs[0].get("gameDate", "")
    today = datetime.date.today()

    days_since = None
    if last_date_str:
        try:
            last_date = datetime.date.fromisoformat(last_date_str)
            days_since = (today - last_date).days
        except ValueError:
            pass

    inactive = days_since is not None and days_since > DAYS_INACTIVE_THRESHOLD

    return {
        "active": not inactive,
        "games_logged": len(logs),
        "last_game_date": last_date_str,
        "days_since_last_game": days_since,
    }


def normalize_player_name(name: str) -> str:
    """Lowercase, strip suffixes, collapse spaces for name matching."""
    s = (name or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    for suffix in (" jr.", " sr.", " iii", " ii", " iv", " jr", " sr"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


def fetch_espn_nhl_injuries() -> tuple[list[dict[str, Any]], str | None, str | None]:
    """
    Fetch current NHL injury list from ESPN's public API.

    Returns (flat_records, api_timestamp_iso, error_message).
    Each record: team_abbrev, player_name, status, injury_type, comment,
    return_date, updated_iso, espn_injury_id.
    """
    try:
        resp = requests.get(
            ESPN_NHL_INJURIES_URL,
            headers=ESPN_REQUEST_HEADERS,
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("ESPN injuries fetch failed: %s", e)
        return [], None, str(e)

    ts = data.get("timestamp")
    flat: list[dict[str, Any]] = []
    for team_block in data.get("injuries", []):
        team_abbrev = ""
        for inj in team_block.get("injuries", []):
            ath = inj.get("athlete") or {}
            team = ath.get("team") or {}
            team_abbrev = str(team.get("abbreviation", "") or "").strip().upper()
            if not team_abbrev:
                continue
            name = str(ath.get("displayName", "") or "").strip()
            if not name:
                continue
            details = inj.get("details") or {}
            inj_type = str(details.get("type", "") or "").strip()
            ret = details.get("returnDate")
            return_date = str(ret).strip() if ret else ""
            status = str(inj.get("status", "") or "").strip()
            comment = str(inj.get("longComment") or inj.get("shortComment") or "").strip()
            updated = str(inj.get("date", "") or "").strip()
            flat.append(
                {
                    "team_abbrev": team_abbrev,
                    "player_name": name,
                    "status": status,
                    "injury_type": inj_type,
                    "comment": comment,
                    "return_date": return_date,
                    "updated_iso": updated,
                    "espn_injury_id": str(inj.get("id", "")),
                }
            )

    logger.info("ESPN injuries: %d player entries (timestamp=%s)", len(flat), ts)
    return flat, str(ts) if ts is not None else None, None


def build_espn_injury_lookup(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Map TEAM|normalized_name -> most recent injury record (by updated_iso).
    """
    best: dict[str, tuple[str, dict[str, Any]]] = {}
    for rec in records:
        team = rec.get("team_abbrev", "")
        name = rec.get("player_name", "")
        key = f"{team}|{normalize_player_name(name)}"
        cur_ts = rec.get("updated_iso") or ""
        if key not in best or cur_ts >= best[key][0]:
            best[key] = (cur_ts, rec)
    return {k: v[1] for k, v in best.items()}


def lookup_espn_injury(
    team_abbrev: str,
    player_name: str,
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    key = f"{(team_abbrev or '').upper()}|{normalize_player_name(player_name)}"
    return lookup.get(key)


def enrich_pred_df_with_espn_injuries(
    pred_df: pd.DataFrame,
    lookup: dict[str, dict[str, Any]],
    api_timestamp: str | None,
) -> pd.DataFrame:
    """Add ESPN injury columns for every row (matched by team + player name)."""
    pred_df = pred_df.copy()
    pred_df["espn_injury_status"] = ""
    pred_df["espn_injury_type"] = ""
    pred_df["espn_injury_comment"] = ""
    pred_df["espn_injury_return_date"] = ""
    pred_df["espn_injury_updated"] = ""
    pred_df["injury_report_source"] = ""
    if api_timestamp:
        pred_df["injury_report_fetched_at"] = api_timestamp
    else:
        pred_df["injury_report_fetched_at"] = ""

    for idx in pred_df.index:
        team = str(pred_df.at[idx, "team_abbrev"] if "team_abbrev" in pred_df.columns else "")
        pname = str(pred_df.at[idx, "playerName"] if "playerName" in pred_df.columns else "")
        hit = lookup_espn_injury(team, pname, lookup)
        if not hit:
            continue
        pred_df.at[idx, "espn_injury_status"] = hit.get("status", "")
        pred_df.at[idx, "espn_injury_type"] = hit.get("injury_type", "")
        pred_df.at[idx, "espn_injury_comment"] = hit.get("comment", "")
        pred_df.at[idx, "espn_injury_return_date"] = hit.get("return_date", "")
        pred_df.at[idx, "espn_injury_updated"] = hit.get("updated_iso", "")
        pred_df.at[idx, "injury_report_source"] = "ESPN"

    return pred_df


def espn_injury_discount_multiplier(status: str, injury_type: str) -> float | None:
    """
    Playoff production multiplier from ESPN status + type.

    Delegates to ``espn_playoff_production_multiplier`` (playoff presumption:
    only IR/LTIR/long-term/suspension → 0; most game labels → 1.0).
    Returns None only when ESPN fields are empty (caller uses 1.0).
    """
    if not (status or "").strip() and not (injury_type or "").strip():
        return None
    return espn_playoff_production_multiplier(status, injury_type)


def verify_and_flag_players(
    client: NHLAPIClient,
    pred_df: pd.DataFrame,
    team_rosters: dict[str, dict[int, dict]],
    check_top_n: int = 50,
) -> pd.DataFrame:
    """
    Verify top predicted players against rosters and check for injuries.

    Adds columns:
    - on_roster: whether player is on current team roster
    - injury_flag: whether player appears to be injured/inactive
    - last_game_date: date of most recent game
    - days_inactive: days since last game
    """
    pred_df = pred_df.copy()
    pred_df["on_roster"] = True
    pred_df["injury_flag"] = ""
    pred_df["last_game_date"] = ""
    pred_df["days_inactive"] = 0

    # Only check the top N players to avoid excessive API calls
    top_indices = pred_df.head(check_top_n).index

    for idx in top_indices:
        row = pred_df.loc[idx]
        pid = int(row.get("playerId", 0))
        team = row.get("team_abbrev", "")
        name = row.get("playerName", "")

        # Roster verification
        roster = team_rosters.get(team, {})
        if roster and pid not in roster:
            pred_df.at[idx, "on_roster"] = False
            pred_df.at[idx, "injury_flag"] = "NOT ON ROSTER"
            logger.warning(
                "Player %s (ID=%d) NOT found on %s roster", name, pid, team
            )

        # Activity check (game-log recency — used when no ESPN injury row)
        activity = check_player_activity(client, pid)
        if activity["last_game_date"]:
            pred_df.at[idx, "last_game_date"] = activity["last_game_date"]
        if activity["days_since_last_game"] is not None:
            pred_df.at[idx, "days_inactive"] = int(activity["days_since_last_game"])

        espn_st = ""
        if "espn_injury_status" in pred_df.columns:
            espn_st = str(pred_df.at[idx, "espn_injury_status"] or "").strip()

        # Game-log inactivity only if not on ESPN injury list (avoid false "inactive"
        # when a healthy scratch or rest is misread vs a listed injury).
        skip_gl = bool(
            pred_df.at[idx, "skip_game_log_inactive"]
            if "skip_game_log_inactive" in pred_df.columns
            else False
        )
        if (
            not skip_gl
            and not espn_st
            and activity["days_since_last_game"] is not None
            and activity["days_since_last_game"] > DAYS_INACTIVE_THRESHOLD
        ):
            flag = f"INACTIVE ({activity['days_since_last_game']}d since last game)"
            pred_df.at[idx, "injury_flag"] = flag
            logger.warning(
                "Player %s appears inactive: last game %s (%d days ago)",
                name, activity["last_game_date"],
                activity["days_since_last_game"],
            )

    return pred_df


def fill_espn_injury_flags_all_rows(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Set injury_flag from ESPN columns for any row with a status, unless the
    player was already marked NOT ON ROSTER.
    """
    pred_df = pred_df.copy()
    if "espn_injury_status" not in pred_df.columns:
        return pred_df

    for idx in pred_df.index:
        st = str(pred_df.at[idx, "espn_injury_status"] or "").strip()
        if not st:
            continue
        if str(pred_df.at[idx, "injury_flag"] or "") == "NOT ON ROSTER":
            continue
        ty = str(pred_df.at[idx, "espn_injury_type"] or "").strip()
        flag = f"ESPN: {st}"
        if ty:
            flag += f" ({ty})"
        pred_df.at[idx, "injury_flag"] = flag

    return pred_df
