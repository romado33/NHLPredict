"""
Write data/processed/predictions_v2_2026.md from predictions_v2_2026.json.

Usage:
    python scripts/export_v2_predictions_md.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DIR


def main() -> None:
    src = PROCESSED_DIR / "predictions_v2_2026.json"
    dst = PROCESSED_DIR / "predictions_v2_2026.md"
    if not src.exists():
        raise SystemExit(f"Missing {src}; run scripts/run_v2_pipeline.py --predict-only first.")

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    lines: list[str] = []
    season = data.get("season_end_year", "")
    gen = data.get("generated_at", "")

    lines.append(f"# NHL Playoff Predictor V2 — {season} Playoffs")
    lines.append("")
    lines.append(f"- **Generated:** {gen}")
    lines.append(f"- **Source:** `{src.name}`")
    lines.append("")
    lines.append(
        "Probabilities below are **series win probabilities** (best-of-7). "
        "**Higher seed** is listed first in each matchup (home-ice format in the model). "
        "**Monte Carlo** uses 50k simulated series from blended per-game win probability."
    )
    lines.append("")

    # ── Series tables ──
    by_round: dict[str, list] = {}
    for s in data.get("series", []):
        r = s.get("round", "?")
        by_round.setdefault(r, []).append(s)

    round_order = [
        "First Round",
        "Second Round",
        "Conference Finals",
        "Stanley Cup Final",
    ]

    for rnd in round_order:
        items = by_round.get(rnd)
        if not items:
            continue
        lines.append(f"## {rnd}")
        lines.append("")
        lines.append(
            "| Matchup | "
            "P(higher seed wins) | "
            "P(lower seed wins) | "
            "Predicted winner | "
            "Expected games | "
            "Ends in 4 / 5 / 6 / 7 (higher seed wins) |"
        )
        lines.append(
            "|---------|----------------------|----------------------|------------------|----------------|----------------------------------------|"
        )

        for s in items:
            hi = s["higher_seed"]
            lo = s["lower_seed"]
            mc = s.get("monte_carlo", {})
            p_hi = float(mc.get("series_win_probability_for_higher_seed", 0))
            p_lo = 1.0 - p_hi
            pw = s.get("predicted_winner", "")
            eg = mc.get("expected_games", 0)
            p4 = float(mc.get("probability_higher_seed_wins_in_4", 0)) * 100
            p5 = float(mc.get("probability_higher_seed_wins_in_5", 0)) * 100
            p6 = float(mc.get("probability_higher_seed_wins_in_6", 0)) * 100
            p7 = float(mc.get("probability_higher_seed_wins_in_7", 0)) * 100
            conf = f"{hi} vs {lo}"
            lines.append(
                f"| {conf} | **{p_hi * 100:.2f}%** | **{p_lo * 100:.2f}%** | "
                f"**{pw}** | {float(eg):.2f} | "
                f"{p4:.1f}% / {p5:.1f}% / {p6:.1f}% / {p7:.1f}% |"
            )

        lines.append("")

    # Bracket shortcut
    lines.append("## Bracket outcome (model progression)")
    lines.append("")
    lines.append(
        f"- **Eastern champion:** {data.get('east_champ', '')}  \n"
        f"- **Western champion:** {data.get('west_champ', '')}  \n"
        f"- **Stanley Cup:** {data.get('champion', '')}"
    )
    lines.append("")

    # Features
    lines.append("## Most influential features (V2 blend)")
    lines.append("")
    lines.append(data.get("feature_importance_method", ""))
    lines.append("")
    bw = data.get("blend_weights") or {}
    if bw:
        lines.append("**Submodel blend weights:**")
        for k, v in sorted(bw.items(), key=lambda x: -x[1]):
            lines.append(f"- `{k}`: {float(v) * 100:.2f}%")
        lines.append("")

    lines.append("| Rank | Feature | Importance (max=1) |")
    lines.append("|------|---------|-------------------|")
    for i, row in enumerate(data.get("top_features", []), 1):
        feat = row.get("feature", "")
        imp = row.get("importance", 0)
        lines.append(f"| {i} | `{feat}` | {imp:.6f} |")
    lines.append("")

    dst.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
