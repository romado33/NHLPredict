"""Lightweight consistency check between team expected games and player predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import PROCESSED_DIR


def write_team_player_consistency_report(
    pred_df: pd.DataFrame,
    team_expected_games: dict[str, float],
    *,
    top_n_per_team: int = 6,
    out_path: Path | None = None,
) -> Path:
    """
    For each team, compare sum of top-N predicted playoff points vs team_expected_games.
    Does not modify predictions — diagnostic JSON only.
    """
    out_path = out_path or (PROCESSED_DIR / "team_player_consistency_2026.json")
    if pred_df.empty or "team_abbrev" not in pred_df.columns:
        payload = {"error": "empty pred_df or missing team_abbrev", "teams": {}}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path

    rows: list[dict[str, Any]] = []
    for team in sorted(team_expected_games.keys()):
        sub = pred_df[pred_df["team_abbrev"] == team].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("pred_points", ascending=False).head(top_n_per_team)
        s = float(sub["pred_points"].sum())
        eg = float(team_expected_games.get(team, 0.0))
        implied_team_ppg = s / max(eg * top_n_per_team, 1e-6)
        rows.append({
            "team": team,
            "team_expected_games": eg,
            f"top{top_n_per_team}_pred_points_sum": round(s, 2),
            "implied_points_per_team_game_per_top_player_slot": round(implied_team_ppg, 4),
        })

    payload = {
        "description": (
            "Sum of top-N player pred_points vs team expected playoff games. "
            "High implied_ppg may indicate star-heavy weighting or injury noise."
        ),
        "top_n": top_n_per_team,
        "teams": {r["team"]: r for r in rows},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
