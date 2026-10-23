"""
Playoff availability policy for production multipliers.

Historical scoring training/eval does not use ESPN injury tags — those apply only
at predict time (plus overrides), so backtest metrics are not polluted by noisy
feed labels.

Default presumption: unless a player is *definitively* ruled out for an extended
absence (IR/LTIR/long-term/suspension), treat them as likely available for the
playoffs. Day-to-day / questionable / maintenance listings are not heavily
discounted — use config/injury_overrides.json for beat-reported partial timelines.

Latest injury *status* should still come from the live ESPN feed at predict time;
overrides layer on top for known mislabels.
"""

from __future__ import annotations

import re

PLAYOFF_HEALTH_POLICY_VERSION = "2026-04_playoff_presumption_v1"


def is_definitively_ruled_out(status: str, injury_type: str) -> bool:
    """
    True when feed indicates season-long / IR-class absence or suspension.
    Plain 'game out' without IR is NOT treated as definitive here.
    """
    st = (status or "").lower().strip()
    it = (injury_type or "").lower().strip()

    if "suspended" in st or "suspended" in it:
        return True
    if "ltir" in st or "ltir" in it or "lti" in st:
        return True
    if "long-term" in st or "long term" in st:
        return True
    if "injured reserve" in st:
        return True
    # Standalone IR (avoid matching 'fir' etc.)
    if re.search(r"\bir\b", st):
        return True
    if "season" in it and any(
        x in it for x in ("ending", "ended", "miss remainder", "out for season")
    ):
        return True
    return False


def espn_playoff_production_multiplier(status: str, injury_type: str) -> float | None:
    """
    Multiplier for predicted *playoff* points from ESPN text alone.

    Returns:
        0.0 — definitive long absence / suspension (no playoff production baseline)
        1.0 — default: expected to participate in playoffs unless override says otherwise
        None — rarely used; defer to inactive heuristics only when no ESPN status
    """
    if is_definitively_ruled_out(status, injury_type):
        return 0.0

    st = (status or "").lower()

    if "doubtful" in st:
        return 0.92

    # Probable / questionable / day-to-day / game-time / out (short) → playoffs presumption
    return 1.0
