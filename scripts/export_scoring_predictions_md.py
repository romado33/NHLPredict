"""
Write data/processed/top_scorers_2026.md from top40_scorers_2026.csv (+ optional JSON context).
Also writes data/processed/top_scorers_2026_simple.csv (rank, playerName, team_abbrev only).

Usage:
    python scripts/export_scoring_predictions_md.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CURRENT_SEASON_END_YEAR, PROCESSED_DIR


def _safe_str(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    s = str(x).strip()
    return "" if s in ("nan", "None") else s


def main() -> None:
    csv_path = PROCESSED_DIR / "top40_scorers_2026.csv"
    dst = PROCESSED_DIR / "top_scorers_2026.md"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}; run scripts/run_scoring_pipeline.py first.")

    df = pd.read_csv(csv_path)

    simple_df = pd.DataFrame({
        "rank": range(1, len(df) + 1),
        "playerName": df["playerName"],
        "team_abbrev": df["team_abbrev"],
    })
    simple_csv = PROCESSED_DIR / "top_scorers_2026_simple.csv"
    simple_df.to_csv(simple_csv, index=False)
    print(f"Wrote {simple_csv}")

    simple_md = PROCESSED_DIR / "top_scorers_2026_simple.md"
    md_lines = [
        f"# Top scorers — {CURRENT_SEASON_END_YEAR} (rank, player, team)",
        "",
        f"*From `{csv_path.name}` — regenerate with `python scripts/export_scoring_predictions_md.py`*",
        "",
        "| Rank | Player | Team |",
        "|------|--------|------|",
    ]
    for _, r in simple_df.iterrows():
        md_lines.append(
            f"| {int(r['rank'])} | {r['playerName']} | {r['team_abbrev']} |"
        )
    simple_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {simple_md}")

    eg_path = PROCESSED_DIR / "team_expected_games_2026.json"
    manifest_path = PROCESSED_DIR / "model_run_manifest.json"
    eval_path = PROCESSED_DIR / "scoring_temporal_eval.json"

    gen = ""
    health_v = ""
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        gen = man.get("generated_at_utc", "")
        health_v = man.get("health_policy_version", "")

    lines: list[str] = []
    season = CURRENT_SEASON_END_YEAR
    lines.append(f"# Playoff top scorers — {season} (model output)")
    lines.append("")
    lines.append(f"- **Source CSV:** `{csv_path.name}`")
    if gen:
        lines.append(f"- **Run manifest:** `{manifest_path.name}` ({gen})")
    if health_v:
        lines.append(f"- **Health policy:** `{health_v}`")
    lines.append("")
    lines.append(
        "Predictions use the **gated** scoring stack (walk-forward trained on historical "
        "playoff points). **Pred points** combine goals + assists model with reconciliation. "
        "**Exp GP** is expected team playoff games from the V2 bracket model (+ bubble logic). "
        "Injury flags and overrides apply at **predict time** (live ESPN + `injury_overrides.json`)."
    )
    lines.append("")

    if eg_path.exists():
        lines.append("## Team expected games (snapshot)")
        lines.append("")
        with open(eg_path, encoding="utf-8") as f:
            eg = json.load(f)
        ts = eg.get("timestamp", "")
        if ts:
            lines.append(f"- **Bracket / standings snapshot:** {ts}")
        pt = eg.get("playoff_teams")
        bu = eg.get("bubble_teams")
        if pt:
            lines.append(f"- **Playoff field (16):** {', '.join(sorted(pt))}")
        if bu:
            lines.append(f"- **Bubble / contenders:** {', '.join(sorted(bu))}")
        lines.append("")

    # Optional: brief walk-forward headline metrics for points (gated_blend)
    if eval_path.exists():
        try:
            with open(eval_path, encoding="utf-8") as f:
                ev = json.load(f)
            gb = ev.get("eval_by_target", {}).get("target_points", {}).get("gated_blend", {})
            if gb:
                lines.append("## Historical backtest (walk-forward, `target_points`, gated blend)")
                lines.append("")
                lines.append(
                    f"- **MAE:** {gb.get('mae', 0):.3f}  \n"
                    f"- **Spearman:** {gb.get('spearman', 0):.3f}  \n"
                    f"- **Mean top-10 / 35 / 40 name overlap:** "
                    f"{gb.get('mean_top10_overlap', 0)*100:.1f}% / "
                    f"{gb.get('mean_top35_overlap', 0)*100:.1f}% / "
                    f"{gb.get('mean_top40_overlap', 0)*100:.1f}%"
                )
                lines.append("")
        except (json.JSONDecodeError, OSError):
            pass

    lines.append(f"## Top {len(df)} predicted playoff scorers")
    lines.append("")
    lines.append(
        "| Rk | Player | Team | Pos | Pred Pts | 80% CI | G | A | Exp GP | Reg Pts | Notes |"
    )
    lines.append("|----|--------|------|-----|----------|--------|---|---|--------|---------|-------|")

    note_cols = ("injury_flag", "injury_override_note")
    for i, row in df.iterrows():
        rk = int(i) + 1
        name = _safe_str(row.get("playerName", ""))
        team = _safe_str(row.get("team_abbrev", ""))
        pos = _safe_str(row.get("positionCode", ""))
        pp = row.get("pred_points", 0)
        lo = row.get("points_lo80", "")
        hi = row.get("points_hi80", "")
        try:
            ci = f"{float(lo):.1f}–{float(hi):.1f}"
        except (TypeError, ValueError):
            ci = ""
        pg = row.get("pred_goals", "")
        pa = row.get("pred_assists", "")
        try:
            eg = float(row.get("team_expected_games", 0))
            eg_s = f"{eg:.1f}"
        except (TypeError, ValueError):
            eg_s = ""
        try:
            rp = int(float(row.get("reg_points", 0)))
        except (TypeError, ValueError):
            rp = ""
        notes = ""
        for c in note_cols:
            s = _safe_str(row.get(c))
            if s:
                notes = s[:120] + ("…" if len(s) > 120 else "")
                break
        lines.append(
            f"| {rk} | {name} | {team} | {pos} | **{float(pp):.1f}** | {ci} | "
            f"{pg} | {pa} | {eg_s} | {rp} | {notes} |"
        )

    lines.append("")
    lines.append(
        f"*Full player list (all contender teams):* `{PROCESSED_DIR.name}/scoring_predictions_2026.csv`"
    )

    dst.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
