"""
Optional prediction-time adjustments to team expected games and round-reach probs.

The bracket sim uses probability-weighted paths but can under-weight plausible upsets
(e.g. a wild-card team winning multiple rounds). Blending toward a manual scenario
updates scorer features that depend on ``team_expected_games`` and reach probabilities.

Configured in ``config/scoring_bracket_adjustments.json``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _enforce_nested_reach(p_r2: float, p_cf: float, p_final: float) -> tuple[float, float, float]:
    """Ensure p_r2 >= p_cf >= p_final in [0, 1]."""
    p_r2 = _clip01(p_r2)
    p_cf = _clip01(min(p_cf, p_r2))
    p_final = _clip01(min(p_final, p_cf))
    return p_r2, p_cf, p_final


def _apply_team_scenario(
    base_eg: dict[str, float],
    base_trp: dict[str, dict[str, float]],
    team: str,
    spec: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Single-team mutation for scenario branch (copies dicts)."""
    eg = dict(base_eg)
    trp = deepcopy(base_trp)

    mult = float(spec.get("expected_games_mult", 1.0))
    if team in eg:
        eg[team] = max(0.0, float(eg[team]) * mult)

    rp = trp.get(team, {"p_r2": 0.0, "p_cf": 0.0, "p_final": 0.0})
    pr2 = float(rp.get("p_r2", 0.0)) + float(spec.get("p_r2_add", 0.0))
    pcf = float(rp.get("p_cf", 0.0)) + float(spec.get("p_cf_add", 0.0))
    pfn = float(rp.get("p_final", 0.0)) + float(spec.get("p_final_add", 0.0))
    pr2, pcf, pfn = _enforce_nested_reach(pr2, pcf, pfn)
    trp[team] = {"p_r2": pr2, "p_cf": pcf, "p_final": pfn}

    return eg, trp


def apply_scoring_bracket_adjustments(
    team_expected_games: dict[str, float],
    team_round_reach_probs: dict[str, dict[str, float]],
    config_path: Path | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any] | None]:
    """
    Blend base bracket outputs toward a configurable upset / alternate-path scenario.

    Returns:
        (adjusted_expected_games, adjusted_reach_probs, meta_or_none)
    """
    if config_path is None or not config_path.exists():
        return team_expected_games, team_round_reach_probs, None

    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not load bracket adjustments %s: %s", config_path, e)
        return team_expected_games, team_round_reach_probs, None

    if not cfg.get("enabled", False):
        return team_expected_games, team_round_reach_probs, None

    w = float(cfg.get("upset_blend", 0.0))
    if w <= 0.0:
        return team_expected_games, team_round_reach_probs, None
    w = min(w, 1.0)

    teams_cfg = cfg.get("teams") or {}
    if not teams_cfg:
        return team_expected_games, team_round_reach_probs, None

    # Build full scenario: sequentially apply per-team deltas on a copy of base
    eg_scen = dict(team_expected_games)
    trp_scen = deepcopy(team_round_reach_probs)
    for team, spec in teams_cfg.items():
        if not isinstance(spec, dict):
            continue
        eg_scen, trp_scen = _apply_team_scenario(eg_scen, trp_scen, team, spec)

    # Blend
    eg_out: dict[str, float] = {}
    all_teams = set(team_expected_games) | set(eg_scen)
    for t in all_teams:
        b = float(team_expected_games.get(t, 0.0))
        s = float(eg_scen.get(t, b))
        eg_out[t] = (1.0 - w) * b + w * s

    trp_out: dict[str, dict[str, float]] = {}
    all_trp = set(team_round_reach_probs) | set(trp_scen)
    for t in all_trp:
        rb = team_round_reach_probs.get(t, {"p_r2": 0.0, "p_cf": 0.0, "p_final": 0.0})
        rs = trp_scen.get(t, rb)
        pr2 = (1.0 - w) * float(rb.get("p_r2", 0.0)) + w * float(rs.get("p_r2", 0.0))
        pcf = (1.0 - w) * float(rb.get("p_cf", 0.0)) + w * float(rs.get("p_cf", 0.0))
        pfn = (1.0 - w) * float(rb.get("p_final", 0.0)) + w * float(rs.get("p_final", 0.0))
        pr2, pcf, pfn = _enforce_nested_reach(pr2, pcf, pfn)
        trp_out[t] = {"p_r2": pr2, "p_cf": pcf, "p_final": pfn}

    meta = {
        "config_path": str(config_path),
        "upset_blend": w,
        "teams_tuned": list(teams_cfg.keys()),
        "comment": cfg.get("comment", ""),
    }
    logger.info(
        "Scoring bracket adjustments applied (upset_blend=%.2f teams=%s)",
        w,
        meta["teams_tuned"],
    )
    return eg_out, trp_out, meta
