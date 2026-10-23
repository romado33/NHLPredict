"""Jeff's Hockey Yahoo draft board 2026-27. Do not touch top50 CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\RobDods\Apps\Cursor\Predictors\Cursor_NHLPO_Predict")
OLD_CSV = ROOT / "data" / "processed" / "yahoo_jeffs_draft_board.csv"
NHL = ROOT / "data" / "raw" / "skater_stats.csv"
HIST = ROOT / "data" / "raw" / "skater_stats_regular_historical.csv"
TOP50 = ROOT / "data" / "processed" / "nhl_2026_27_top50_scorers.csv"
ATHLETIC = Path(r"c:\Users\RobDods\Downloads\2026-27-Fantasy-Projections-Yahoo-3.xlsx")
OUT_CSV = ROOT / "data" / "processed" / "yahoo_jeffs_draft_board.csv"
OUT_JSON = ROOT / "data" / "processed" / "yahoo_jeffs_draft_board.json"

# Skater: G=7 A=5 +/-=2 PIM=1 PPG=2 PPA=1 SHG=3 GWG=5 SOG=0.5
def fp(g, a, pm, pim, ppg, ppa, shg, gwg, sog):
    return 7 * g + 5 * a + 2 * pm + pim + 2 * ppg + ppa + 3 * shg + 5 * gwg + 0.5 * sog


# Goalie: W=10 L=-7 GA=-2 SV=0.5 SHO=7
def gfp(w, l, ga, sv, sho):
    return 10 * w + (-7) * l + (-2) * ga + 0.5 * sv + 7 * sho


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def nk(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )


# 2024-25 last healthy mix for Barkov (Yahoo last year is 0 GP).
BARKOV_PRIOR = {
    "gp": 67,
    "g": 20,
    "a": 51,
    "pm": 1,
    "pim": 16,
    "ppg": 9,
    "ppa": 19,
    "shg": 1,
    "gwg": 4,
    "sog": 149,
    "pts": 71,
}

# Live Sep 28 2026: injury/roster verified.
# DTD = minus 2-5 GP from healthy cap, still draftable.
# IR-LT Marchand = minus ~12 GP, draftable mid/late.
# IR-NR Barzal = stash only.
# Yahoo NA last year != NA now.
OVERRIDES = {
    "zach werenski": {"gp": 82, "pred": 83, "g_share": 0.27},
    "mark scheifele": {"gp": 82, "pred": 87, "g_share": 0.345},
    "alex debrincat": {"gp": 84, "pred": 80, "g_share": 0.48},
    "aleksander barkov": {
        "gp": 82,
        "pred": 80,
        "g_share": 20 / 71,
        "flag": "healthy",
        "injury_note": "healthy, 82 GP",
        "use_prior": True,
    },
    "artemi panarin": {"gp": 82, "pred": 88, "g_share": 0.34},
    "juraj slafkovsky": {"gp": 84, "pred": 72, "g_share": 0.42},
    "kirill marchenko": {"gp": 80, "pred": 68, "g_share": 0.40},
    "filip forsberg": {"gp": 84, "pred": 77, "g_share": 0.52},
    "will smith": {"gp": 80, "pred": 76, "g_share": 0.40},
    "jake sanderson": {"gp": 78, "pred": 58, "g_share": 0.24},
    "jakob chychrun": {"gp": 82, "pred": 58, "g_share": 0.44},
    "lucas raymond": {"gp": 82, "pred": 79, "g_share": 0.34},
    "porter martone": {"gp": 70, "pred": 45, "g_share": 0.40, "flag": "tiny"},
    "pavel dorofeyev": {"gp": 82, "pred": 66, "g_share": 0.56},
    "adam fox": {"gp": 72, "pred": 62, "g_share": 0.18},
    "jesper bratt": {"gp": 82, "pred": 68, "g_share": 0.32},
    "roope hintz": {"gp": 72, "pred": 58, "g_share": 0.36},
    "moritz seider": {"gp": 84, "pred": 58, "g_share": 0.18},
    "drake batherson": {"gp": 80, "pred": 68, "g_share": 0.46},
    "beckett sennecke": {"gp": 82, "pred": 58, "g_share": 0.38},
    "logan cooley": {"gp": 78, "pred": 74, "g_share": 0.42},
    "mika zibanejad": {"gp": 82, "pred": 77, "g_share": 0.44},
    "nick schmaltz": {"gp": 82, "pred": 70, "g_share": 0.44},
    "timo meier": {"gp": 80, "pred": 50, "g_share": 0.54},
    "dylan holloway": {"gp": 76, "pred": 62, "g_share": 0.42},
    "erik karlsson": {"gp": 78, "pred": 62, "g_share": 0.23},
    "bo horvat": {"gp": 78, "pred": 64, "g_share": 0.54},
    "jt miller": {"gp": 76, "pred": 62, "g_share": 0.32},
    "zach hyman": {"gp": 74, "pred": 62, "g_share": 0.58},
    "jackson lacombe": {"gp": 82, "pred": 52, "g_share": 0.18},
    "gavin mckenna": {"gp": 72, "pred": 48, "g_share": 0.36, "flag": "tiny"},
    "miro heiskanen": {"gp": 80, "pred": 62, "g_share": 0.15},
    "roman josi": {"gp": 74, "pred": 58, "g_share": 0.24},
    "sam bennett": {"gp": 78, "pred": 56, "g_share": 0.45},
    "mark stone": {"gp": 68, "pred": 68, "g_share": 0.38},
    "john tavares": {"gp": 82, "pred": 66, "g_share": 0.44},
    "bryan rust": {"gp": 76, "pred": 66, "g_share": 0.446, "flag": "DTD", "injury_note": "DTD -4 GP"},
    "brock faber": {"gp": 78, "pred": 49, "g_share": 0.28, "flag": "DTD", "injury_note": "DTD -4 GP"},
    "alex ovechkin": {"gp": 79, "pred": 60, "g_share": 0.52, "flag": "DTD", "injury_note": "DTD -3 GP"},
    "matthew knies": {"gp": 80, "pred": 66, "g_share": 0.36},
    "quinton byfield": {"gp": 80, "pred": 54, "g_share": 0.48},
    "rickard rakell": {"gp": 72, "pred": 56, "g_share": 0.50},
    "morgan geekie": {"gp": 80, "pred": 62, "g_share": 0.54},
    "steven stamkos": {"gp": 80, "pred": 64, "g_share": 0.62},
    "joel eriksson ek": {"gp": 76, "pred": 54, "g_share": 0.38},
    "andrei svechnikov": {"gp": 80, "pred": 68, "g_share": 0.45},
    "nico hischier": {"gp": 82, "pred": 64, "g_share": 0.42},
    "nikolaj ehlers": {"gp": 82, "pred": 68, "g_share": 0.37},
    "cole hutson": {"gp": 70, "pred": 38, "g_share": 0.22, "flag": "tiny"},
    "alexis lafreniere": {"gp": 82, "pred": 56, "g_share": 0.42},
    "john carlson": {"gp": 76, "pred": 58, "g_share": 0.24},
    "anton frondell": {"gp": 65, "pred": 38, "g_share": 0.38, "flag": "tiny"},
    "trevor zegras": {"gp": 82, "pred": 66, "g_share": 0.39},
    "alex tuch": {"gp": 80, "pred": 64, "g_share": 0.50},
    "dylan larkin": {"gp": 79, "pred": 71, "g_share": 0.507, "flag": "DTD", "injury_note": "DTD -3 GP"},
    "logan stankoven": {"gp": 82, "pred": 48, "g_share": 0.47},
    "jackson blake": {"gp": 82, "pred": 54, "g_share": 0.42},
    "jordan kyrou": {"gp": 76, "pred": 50, "g_share": 0.40},
    "travis konecny": {"gp": 80, "pred": 66, "g_share": 0.40},
    "nazem kadri": {"gp": 78, "pred": 54, "g_share": 0.32},
    "victor hedman": {"gp": 72, "pred": 46, "g_share": 0.12, "flag": "healthy", "injury_note": "healthy, 72 GP"},
    "gabriel vilardi": {"gp": 80, "pred": 64, "g_share": 0.44},
    "brock nelson": {"gp": 82, "pred": 64, "g_share": 0.50},
    "jimmy snuggerud": {"gp": 76, "pred": 54, "g_share": 0.42},
    "mathew barzal": {"gp": 55, "pred": 50, "g_share": 0.28, "flag": "IR-NR", "injury_note": "IR stash"},
    "evgeni malkin": {"gp": 68, "pred": 66, "g_share": 0.32},
    "ryan oreilly": {"gp": 80, "pred": 68, "g_share": 0.35},
    "mikhail sergachev": {"gp": 80, "pred": 56, "g_share": 0.18},
    "dylan strome": {"gp": 80, "pred": 56, "g_share": 0.34},
    "vincent trocheck": {"gp": 74, "pred": 54, "g_share": 0.32},
    "josh doan": {"gp": 82, "pred": 52, "g_share": 0.48},
    "zach benson": {"gp": 74, "pred": 52, "g_share": 0.32},
    "elias pettersson": {"gp": 76, "pred": 58, "g_share": 0.30},
    "dylan cozens": {"gp": 82, "pred": 58, "g_share": 0.47},
    "ivar stenberg": {"gp": 70, "pred": 42, "g_share": 0.38, "flag": "tiny"},
    "shea theodore": {"gp": 74, "pred": 42, "g_share": 0.26},
    "matt duchene": {"gp": 68, "pred": 50, "g_share": 0.36, "flag": "DTD", "injury_note": "DTD -2 GP"},
    "shayne gostisbehere": {"gp": 70, "pred": 52, "g_share": 0.26},
    "william eklund": {"gp": 80, "pred": 56, "g_share": 0.30},
    "josh morrissey": {"gp": 80, "pred": 54, "g_share": 0.26},
    "charlie mcavoy": {
        "gp": 76,
        "pred": 60,
        "g_share": 0.18,
        "flag": "SUSP",
        "injury_note": "healthy, -6 GP susp",
    },
    "noah dobson": {"gp": 82, "pred": 48, "g_share": 0.26},
    "brandt clarke": {"gp": 82, "pred": 42, "g_share": 0.20},
    "adam fantilli": {"gp": 84, "pred": 64, "g_share": 0.42},
    "jj peterka": {"gp": 82, "pred": 52, "g_share": 0.52},
    "dougie hamilton": {"gp": 78, "pred": 42, "g_share": 0.30},
    "carter verhaeghe": {"gp": 78, "pred": 56, "g_share": 0.46},
    "valeri nichushkin": {"gp": 74, "pred": 50, "g_share": 0.36},
    "egor chinakhov": {"gp": 74, "pred": 44, "g_share": 0.50},
    "gabe perreault": {"gp": 68, "pred": 42, "g_share": 0.44, "flag": "tiny"},
    "aliaksei protas": {"gp": 78, "pred": 54, "g_share": 0.48},
    "mason marchment": {"gp": 74, "pred": 48, "g_share": 0.42},
    "mackenzie weegar": {"gp": 80, "pred": 32, "g_share": 0.16},
    "brandon montour": {"gp": 72, "pred": 38, "g_share": 0.34},
    "ryan nugent hopkins": {"gp": 76, "pred": 58, "g_share": 0.36},
    "pavel zacha": {"gp": 80, "pred": 62, "g_share": 0.46},
    "seth jones": {"gp": 68, "pred": 38, "g_share": 0.22},
    "gabriel landeskog": {"gp": 68, "pred": 42, "g_share": 0.42},
    "brad marchand": {"gp": 66, "pred": 58, "g_share": 0.50, "flag": "IR-LT", "injury_note": "IR-LT -12 GP"},
    "michael misa": {"gp": 70, "pred": 40, "g_share": 0.42, "flag": "tiny"},
}

NOTES = {
    "zach werenski": "Healthy. D1 on board. Norris year, 260 SOG. Take as 1D/early F-flex.",
    "mark scheifele": "Healthy. Our 87 pts (SH% down from 20.6). Still 1C next to Connor.",
    "alex debrincat": "Healthy. Goal-heavy 41 G / 287 SOG. This scoring loves him more than our 80 NHL pts.",
    "aleksander barkov": "BACK healthy 82 GP. Yahoo NA was 0 GP last year, not now. FLA roster. 2024-25: 20/51/71 in 67; mix*80 pts. 1C.",
    "artemi panarin": "Healthy. Our 88. Fiala out to Nov = extra PP. Best available LW talent.",
    "juraj slafkovsky": "Healthy. 30 G / 15 PPG last year. Volume 3rd-year leap, not a lock.",
    "kirill marchenko": "Healthy. CBJ PP1 with Werenski. 7 GWG. Goal/SOG profile.",
    "filip forsberg": "Healthy. 40 G, 246 SOG. Our 77 pts; G=7/GWG/SOG keep him elite here.",
    "will smith": "Healthy. Celebrini stack. Our 76. Only 69 GP last year; 80 GP is the bet.",
    "jake sanderson": "Healthy. 14 G / 67 GP. GP bounce; not Werenski/Chychrun shots.",
    "jakob chychrun": "Healthy. 26 G / 221 SOG / 8 GWG. Best goal-scoring D after Werenski.",
    "lucas raymond": "Healthy. Our 79. Larkin DTD can bump early usage with DeBrincat.",
    "porter martone": "Tiny sample: 9 GP. Flyer after real NHL wingers.",
    "pavel dorofeyev": "Healthy. 37 G / 20 PPG / 230 SOG. Goal/PPG monster; assists thin.",
    "adam fox": "Healthy. 55 GP last year. Rate still PP1 NYR; GP is the tax.",
    "jesper bratt": "Healthy. Hughes stack. 22 G only; this format wants more shot/goal.",
    "roope hintz": "Healthy. 53 GP last year. DAL 2C; GP bounce if healthy, not a 70-pt lock.",
    "moritz seider": "Healthy. 82 GP, 187 SOG, 57 PIM. D volume even if 10 G.",
    "drake batherson": "Healthy. 33 G / 13 PPG. Stutzle still feeds him; Brady gone is a nibble.",
    "beckett sennecke": "Healthy. Year-2 ANA. 23 G / 197 SOG; assists more than finishing.",
    "logan cooley": "Healthy. Our 74. 23.5 SH% last year will not repeat; Keller/Guenther PP.",
    "mika zibanejad": "Healthy. Our 77. Do NOT copy -20; PP1 NYR with Dorofeyev shooter.",
    "nick schmaltz": "Healthy. Last-year FP 683 is GWG/+17 luck. Regress; still useful C/W.",
    "timo meier": "Healthy. 269 SOG, only 24 G. Shot streamer; finishing died. Not a star.",
    "dylan holloway": "Healthy. 59 GP rate was real. STL 1-line; not a 70-pt projection.",
    "erik karlsson": "Healthy. 15 G / 176 SOG / 22 PPA. PIT PP1 shots+points D.",
    "bo horvat": "Healthy. 31 G / 7 GWG / 226 SOG. Goal C; 68 GP last year.",
    "jt miller": "Healthy. Do NOT copy -30. NYR 2C/W; Athletic 69 pts. 68 GP last year.",
    "zach hyman": "Healthy. 31 G in 58. McDavid PP net-front. GP 74 not 84.",
    "jackson lacombe": "Healthy. 48 A / 82 GP ANA PP. Assist D; fewer goals than Chychrun.",
    "gavin mckenna": "Tiny sample: zero NHL GP. Athletic likes him; wait. Not an injury.",
    "miro heiskanen": "Healthy. 54 A / 26 PPA. DAL PP; 9 G. Stable 2D.",
    "roman josi": "Healthy. Age/NSH. 68 GP last year, -15. Still PP1 shots; not 70 pts.",
    "sam bennett": "Healthy. 82 PIM + 5 GWG. Cats help; 26 G not a 1C.",
    "mark stone": "Healthy. 28 G / 45 A in 60. Elite rate, 68 GP cap. VGK PP.",
    "john tavares": "Healthy. 31 G ironman. Do NOT copy -28. TOR 2C; age 36.",
    "bryan rust": "DTD -4 GP (76). Evaluated after Sat preseason; not on IR. 29 G / 16 PPA when healthy. DRAFTABLE.",
    "brock faber": "DTD -4 GP (78). Precautionary puck-to-head Fri. MIN 2D behind Hughes. DRAFTABLE.",
    "alex ovechkin": "DTD -3 GP (79). Carbery expects Oct 2 opener. 32 G / 244 SOG still. DRAFTABLE.",
    "matthew knies": "Healthy. Do NOT copy -30. Matthews stack; 23 G not 35.",
    "quinton byfield": "Healthy. 24 G / 25 A. Usage up with Panarin; finishing still meh.",
    "rickard rakell": "Healthy. 60 GP last year, 24 G. PIT PP; not a full-season lock.",
    "morgan geekie": "Healthy. 39 G on likely SH% spike. 30 G is the mean, not 39.",
    "steven stamkos": "Healthy. 42 G / 14 PPG / 10 GWG. This scoring's age cheat code.",
    "joel eriksson ek": "Healthy. 194 SOG, two-way. 19 G; MIN 2C not a scorer.",
    "andrei svechnikov": "Healthy. 31 G / 66 PIM / 12 PPG. CAR PP; Jarvis out weeks = bump.",
    "nico hischier": "Healthy. 28 G / 211 SOG. NJ 1C; Hughes takes PP cream.",
    "nikolaj ehlers": "Healthy. 26 G / 19 PPA / 6 GWG. CAR PP2; Jarvis bump.",
    "cole hutson": "Tiny sample: 14 GP. Lane's brother, not Lane.",
    "alexis lafreniere": "Healthy. 24 G floor. NYR depth; Dorofeyev/Zibanejad eat PP.",
    "john carlson": "Healthy. TB now. 14 G / 46 A in 71. PP with Kucherov leftovers.",
    "anton frondell": "Tiny sample: 12 GP CHI. Prospect, not tonight.",
    "trevor zegras": "Healthy. 26 G / 62 PIM. PHI PP1 with Konecny; bounce-back year.",
    "alex tuch": "Healthy. WSH now. 33 G / 57 PIM / 3 SHG. Goal+PIM profile.",
    "dylan larkin": "DTD -3 GP (79). DET opener Oct 2 uncertain; 1 extra game not a fade. 34 G / 9 GWG. KEEP.",
    "logan stankoven": "Healthy. CAR depth. 21 G; Jarvis out helps a little.",
    "jackson blake": "Healthy. CAR 22 G / 174 SOG. Depth scorer, not PP1.",
    "jordan kyrou": "Healthy. WSH now. 18 G in 72. Role down; don't chase name.",
    "travis konecny": "Healthy. 27 G / 59 PIM / +14. PHI PP with Zegras.",
    "nazem kadri": "Healthy. Do NOT copy -30. COL 2C behind MacKinnon/Necas.",
    "victor hedman": "Healthy 72 GP. Last year 33 GP was leave/elbow, not current. Still not a 1D.",
    "gabriel vilardi": "Healthy. 30 G / 13 PPG next to Scheifele/Connor. WPG PP.",
    "brock nelson": "Healthy. 33 G / 9 GWG on COL 2C. MacKinnon leftovers still count.",
    "jimmy snuggerud": "Healthy. Year-2 STL. 21 G in 70; upside if 1-line sticks.",
    "mathew barzal": "IR-NR stash. Not on NYI active roster (knee). 55 GP cut. 53 A when healthy. IR slot only.",
    "evgeni malkin": "Healthy. 19 G / 42 A in 56. Age 40; 68 GP. PIT PP still.",
    "ryan oreilly": "Healthy. 25 G / 49 A / 8 GWG. NSH 1C; quiet 68-pt year.",
    "mikhail sergachev": "Healthy. UTA PP1. 10 G / 49 A. Stable 2D.",
    "dylan strome": "Healthy. WSH 1C with Tuch. 19 G; PP but not a 30-goal guy.",
    "vincent trocheck": "Healthy. UTA now. 67 GP last year, 16 G. 3C/2C; PIM help.",
    "josh doan": "Healthy. BUF 25 G in 82. Depth; Tuch gone may add PP crumbs.",
    "zach benson": "Healthy. 65 GP last year, +27 luck. 13 G; don't buy the plus-minus.",
    "elias pettersson": "Healthy. Do NOT copy -30. VAN reset; 15 G is the floor risk.",
    "dylan cozens": "Healthy. 28 G / 13 PPG / 59 PIM. OTT 2C; goal/PIM C.",
    "ivar stenberg": "Tiny sample: zero NHL GP. SJ prospect. Not an injury.",
    "shea theodore": "Healthy. 70 GP last year, 10 G. VGK 2D behind Pietrangelo role.",
    "matt duchene": "DTD -2 GP (68). Gulutzan expects Oct 2 opener. DAL 3C. DRAFTABLE.",
    "shayne gostisbehere": "Healthy. 55 GP last year, 13 G / 12 PPA. CAR PP2; GP bounce.",
    "william eklund": "Healthy. OTT now with Stutzle. Do NOT copy -31. Usage up.",
    "josh morrissey": "Healthy. WPG PP. 14 G / 41 A stable.",
    "charlie mcavoy": "Healthy. Yahoo NA = 6-game slash suspension, not injury. On BOS roster. 76 GP.",
    "noah dobson": "Healthy. MTL 2D. Hutson eats PP1; 12 G floor.",
    "brandt clarke": "Healthy. LA 8 G / 63 PIM. PP2; volume PIM D.",
    "adam fantilli": "Healthy. 24 G / 214 SOG. CBJ 1C; year-3 leap possible.",
    "jj peterka": "Healthy. BOS 25 G. Pasta leftovers; 22 A thin.",
    "dougie hamilton": "Healthy. 12 G / 203 SOG. NJ PP2; shot D not 50 pts.",
    "carter verhaeghe": "Healthy. Barkov BACK helps. 25 G; FLA crowded with Tkachuks.",
    "valeri nichushkin": "Healthy. CBJ now, not COL. 17 G; role down, don't chase old name.",
    "egor chinakhov": "Healthy. PIT 21 G in 13:28. Depth; Crosby leftovers.",
    "gabe perreault": "Tiny sample: 49 GP / 12 G. NYR prospect.",
    "aliaksei protas": "Healthy. 25 G / +20 / 2 SHG. WSH even-strength; no PP.",
    "mason marchment": "Healthy. SJ 19 G / 56 PIM. Depth winger.",
    "mackenzie weegar": "Healthy. Do NOT copy -33. 4 G / 88 PIM. PIM streamer only.",
    "brandon montour": "Healthy. SEA 11 G in 64. PP; not a 50-pt D.",
    "ryan nugent hopkins": "Healthy. EDM PP with McDavid. 20 G / 18 PPA. 3C/W.",
    "pavel zacha": "Healthy. 30 G / 11 PPG. BOS 2C next to Pasta; sneaky C/W.",
    "seth jones": "Healthy. 52 GP last year FLA. PP2; 7 G. Not a 1D.",
    "gabriel landeskog": "Healthy. COL 14 G in 60. Depth next to MacKinnon; 68 GP.",
    "brad marchand": "IR-LT -12 GP (66). Out October, return ~Nov 2. Draftable mid/late with IR slot, not AVOID-forever.",
    "michael misa": "Tiny sample: 45 GP / 9 G at 12:48. SJ 3C.",
}

NHL_POS_HINT = {
    "alex debrincat": "RW",
    "artemi panarin": "LW",
    "filip forsberg": "LW",
    "pavel dorofeyev": "LW",
    "steven stamkos": "C",
    "kirill marchenko": "RW",
    "juraj slafkovsky": "LW",
    "lucas raymond": "LW",
    "will smith": "RW",
    "jesper bratt": "LW",
    "drake batherson": "RW",
    "beckett sennecke": "RW",
    "timo meier": "LW",
    "zach hyman": "LW",
    "gavin mckenna": "LW",
    "mark stone": "RW",
    "bryan rust": "RW",
    "alex ovechkin": "LW",
    "matthew knies": "LW",
    "andrei svechnikov": "LW",
    "nikolaj ehlers": "LW",
    "alexis lafreniere": "LW",
    "alex tuch": "RW",
    "jackson blake": "RW",
    "jordan kyrou": "RW",
    "travis konecny": "RW",
    "jimmy snuggerud": "RW",
    "josh doan": "RW",
    "zach benson": "LW",
    "ivar stenberg": "LW",
    "william eklund": "LW",
    "jj peterka": "RW",
    "carter verhaeghe": "LW",
    "valeri nichushkin": "RW",
    "egor chinakhov": "RW",
    "gabe perreault": "RW",
    "aliaksei protas": "LW",
    "mason marchment": "LW",
    "gabriel landeskog": "LW",
    "brad marchand": "LW",
    "porter martone": "RW",
    "quinton byfield": "C",
    "elias pettersson": "C",
    "logan cooley": "C",
    "nick schmaltz": "C",
    "mika zibanejad": "C",
    "morgan geekie": "C",
    "trevor zegras": "C",
    "gabriel vilardi": "C",
    "pavel zacha": "C",
    "ryan nugent hopkins": "C",
    "jt miller": "C",
    "mathew barzal": "C",
    "evgeni malkin": "C",
    "rickard rakell": "RW",
    "dylan holloway": "LW",
    "logan stankoven": "C",
    "nazem kadri": "C",
    "matt duchene": "C",
    "victor hedman": "D",
}

# 10-team 1G+BN: assume ~8-10 elite starters kept. These ADP stars are likely NOT FA.
LIKELY_KEPT_G = {
    "andrei vasilevskiy",
    "connor hellebuyck",
    "igor shesterkin",
    "logan thompson",
    "jake oettinger",
    "ilya sorokin",
    "jeremy swayman",
    "scott wedgewood",
}

TEAM_MAP = {"T.B": "TBL", "TB": "TBL", "N.J": "NJD", "NJ": "NJD", "L.A": "LAK", "LA": "LAK", "S.J": "SJS", "SJ": "SJS"}


def load_last_year_stats():
    nhl = pd.read_csv(NHL)
    if "skater_report" in nhl.columns:
        nhl = nhl[nhl["skater_report"].fillna("summary") == "summary"]
    nhl["nk"] = nhl["skaterFullName"].astype(str).map(nk)
    out = {}
    for _, r in nhl.iterrows():
        out[r["nk"]] = r
    return out


def load_hist_2025():
    hist = pd.read_csv(HIST)
    hist = hist[hist["season"] == 2025]
    hist["nk"] = hist["skaterFullName"].astype(str).map(nk)
    return {r["nk"]: r for _, r in hist.iterrows()}


def project_skater(p, last, ov, our_pred):
    key = nk(p["player"])
    flag = ov.get("flag") or "healthy"
    exp_gp = float(ov.get("gp") or 80)
    pred = ov.get("pred")
    if pred is None:
        pred = our_pred if our_pred else 40.0
    g_share = ov.get("g_share", 0.40)

    if ov.get("use_prior"):
        mix = BARKOV_PRIOR
        last_gp = mix["gp"]
        last_g, last_a = mix["g"], mix["a"]
        last_pm, last_pim = mix["pm"], mix["pim"]
        last_ppg, last_ppa = mix["ppg"], mix["ppa"]
        last_shg, last_gwg, last_sog = mix["shg"], mix["gwg"], mix["sog"]
        last_pts = mix["pts"]
    elif last is not None and float(last.get("gamesPlayed") or 0) > 0:
        last_gp = float(last["gamesPlayed"])
        last_g = float(last.get("goals") or 0)
        last_a = float(last.get("assists") or 0)
        last_pm = float(last.get("plusMinus") or 0)
        last_pim = float(last.get("penaltyMinutes") or 0)
        last_ppg = float(last.get("ppGoals") or 0)
        ppp = float(last.get("ppPoints") or 0)
        last_ppa = max(0.0, ppp - last_ppg)
        last_shg = float(last.get("shGoals") or 0)
        last_gwg = float(last.get("gameWinningGoals") or 0)
        last_sog = float(last.get("shots") or 0)
        last_pts = last_g + last_a
        if "g_share" not in ov and last_pts > 0:
            g_share = last_g / last_pts
    else:
        last_gp = 0
        last_g = last_a = last_pm = last_pim = 0
        last_ppg = last_ppa = last_shg = last_gwg = last_sog = 0
        last_pts = 0

    G = pred * g_share
    A = pred - G
    ppg_frac = (last_ppg / last_g) if last_g > 0 else 0.28
    ppa_frac = (last_ppa / last_a) if last_a > 0 else 0.30
    PPG = G * clip(ppg_frac, 0.05, 0.60)
    PPA = A * clip(ppa_frac, 0.05, 0.55)

    if last_gp >= 20 and last_sog:
        SOG = (last_sog / last_gp) * exp_gp
    else:
        SOG = 2.2 * exp_gp if p["yahoo_pos"] != "D" else 1.8 * exp_gp

    if last_g > 0 and last_gwg:
        GWG = G * (last_gwg / last_g)
    elif last_gp >= 20:
        GWG = (last_gwg / last_gp) * exp_gp
    else:
        GWG = G * 0.12

    if last_gp >= 20:
        SHG = (last_shg / last_gp) * exp_gp
        PIM = (last_pim / last_gp) * exp_gp
        raw = last_pm * (exp_gp / last_gp)
    else:
        SHG = 0.3 if p["yahoo_pos"] != "D" else 0.1
        PIM = 0.4 * exp_gp
        raw = 0
    PM = clip(0.28 * raw, -8.0, 8.0)

    proj = fp(G, A, PM, PIM, PPG, PPA, SHG, GWG, SOG)
    return {
        "proj": round(proj, 1),
        "pred": round(pred, 1),
        "exp_gp": int(round(exp_gp)),
        "flag": flag,
        "proj_g": round(G, 1),
        "proj_a": round(A, 1),
        "g_share": round(g_share, 3),
        "last_gp_used": int(last_gp) if last_gp else 0,
    }


def inferred_goalies():
    lst = pd.read_excel(ATHLETIC, sheet_name="The List")
    g = lst[lst["POS"].astype(str).str.upper() == "G"].copy()
    rows = []
    for _, r in g.iterrows():
        name = str(r["NAME"]).strip()
        key = nk(name)
        w, l = float(r["W"]), float(r["L"])
        ga, sv, sho = float(r["GA"]), float(r["SV"]), float(r["SO"])
        gp = float(r["GP.1"]) if pd.notna(r["GP.1"]) else 0
        proj = gfp(w, l, ga, sv, sho)
        team = TEAM_MAP.get(str(r["TEAM"]).strip(), str(r["TEAM"]).strip())
        kept = key in LIKELY_KEPT_G
        if kept:
            continue
        # Most-likely FA pool: everyone after the 8 likely-kept stars.
        # Cap the board at useful starters/1As (proj FP >= 450 or GP >= 45).
        if proj < 430:
            continue
        note = (
            f"INFERRED FA (not in Yahoo F/D paste). Athletic {int(round(gp))} GP, "
            f"{w:.0f} W / {sv:.0f} SV. This league: W=10 L=-7 SV=0.5. Stars (Vasy/"
            f"Hellebuyck/Shesterkin/Thompson/Oettinger/Sorokin/Swayman/Wedgewood) assumed kept."
        )
        rows.append(
            {
                "player": name,
                "team": team,
                "yahoo_pos": "G",
                "nhl_pos": "G",
                "proj_fp": round(proj, 1),
                "last_year_fp": 0,
                "last_year_gp": 0,
                "our_nhl_pred_if_any": "",
                "pred_pts": "",
                "exp_gp": int(round(gp)),
                "flag": "inferred",
                "injury": "inferred",
                "proj_gp": int(round(gp)),
                "avoid": False,
                "note": note,
                "source": "inferred",
                "c": False,
                "lw": False,
                "rw": False,
                "f": False,
                "d": False,
                "g": True,
                "goal": False,
            }
        )
    return rows


# Only recast names whose GP/injury treatment changed. Keep prior FP for everyone else.
# Snapshot is the board BEFORE this injury pass (CSV was overwritten once).
RECALC = {
    "aleksander barkov",
    "dylan larkin",
    "bryan rust",
    "brock faber",
    "alex ovechkin",
    "matt duchene",
    "mathew barzal",
    "brad marchand",
    "charlie mcavoy",
    "victor hedman",
}
ORIGINAL_SKATER_FP = {
    "alex debrincat": 711.6,
    "artemi panarin": 684.1,
    "filip forsberg": 677.6,
    "mark scheifele": 675.0,
    "dylan larkin": 663.9,
    "zach werenski": 661.7,
    "bo horvat": 638.0,
    "aleksander barkov": 636.1,
    "mika zibanejad": 635.9,
    "juraj slafkovsky": 634.1,
    "steven stamkos": 633.0,
    "nick schmaltz": 629.3,
    "andrei svechnikov": 626.2,
    "lucas raymond": 622.9,
    "will smith": 619.9,
    "kirill marchenko": 614.8,
    "pavel dorofeyev": 611.9,
    "logan cooley": 601.4,
    "brock nelson": 594.1,
    "jakob chychrun": 593.9,
    "evgeni malkin": 589.8,
    "alex tuch": 588.4,
    "trevor zegras": 586.1,
    "drake batherson": 581.3,
    "nikolaj ehlers": 578.8,
    "zach hyman": 569.4,
    "travis konecny": 567.9,
    "nico hischier": 566.5,
    "dylan cozens": 557.9,
    "ryan oreilly": 556.9,
    "sam bennett": 554.5,
    "adam fantilli": 550.1,
    "mark stone": 546.9,
    "john tavares": 545.9,
    "morgan geekie": 543.5,
    "bryan rust": 532.7,
    "dylan holloway": 527.8,
    "alex ovechkin": 526.5,
    "jesper bratt": 525.2,
    "pavel zacha": 515.7,
    "beckett sennecke": 514.3,
    "moritz seider": 514.1,
    "gabriel vilardi": 510.2,
    "jt miller": 509.4,
    "timo meier": 503.5,
    "joel eriksson ek": 501.6,
    "matthew knies": 497.4,
    "quinton byfield": 496.5,
    "roope hintz": 494.3,
    "miro heiskanen": 490.0,
    "carter verhaeghe": 488.7,
    "erik karlsson": 485.8,
    "roman josi": 485.4,
    "rickard rakell": 480.1,
    "jimmy snuggerud": 478.3,
    "jackson blake": 476.9,
    "alexis lafreniere": 473.2,
    "dylan strome": 473.0,
    "aliaksei protas": 471.5,
    "ryan nugent hopkins": 470.8,
    "mathew barzal": 470.7,
    "jake sanderson": 466.7,
    "josh doan": 463.6,
    "nazem kadri": 463.6,
    "mikhail sergachev": 462.8,
    "adam fox": 462.3,
    "john carlson": 459.8,
    "zach benson": 459.4,
    "vincent trocheck": 458.5,
    "jj peterka": 455.3,
    "brad marchand": 447.3,
    "william eklund": 446.4,
    "mason marchment": 442.9,
    "josh morrissey": 436.3,
    "logan stankoven": 431.5,
    "porter martone": 430.1,
    "elias pettersson": 429.8,
    "shayne gostisbehere": 426.9,
    "dougie hamilton": 422.6,
    "jordan kyrou": 417.5,
    "valeri nichushkin": 413.6,
    "brock faber": 410.9,
    "gavin mckenna": 410.7,
    "gabriel landeskog": 410.5,
    "brandt clarke": 401.7,
    "noah dobson": 398.1,
    "charlie mcavoy": 397.5,
    "jackson lacombe": 396.2,
    "brandon montour": 380.6,
    "egor chinakhov": 374.4,
    "ivar stenberg": 370.5,
    "shea theodore": 361.1,
    "matt duchene": 358.6,
    "gabe perreault": 348.5,
    "anton frondell": 345.0,
    "mackenzie weegar": 328.6,
    "cole hutson": 323.8,
    "michael misa": 313.2,
    "seth jones": 308.7,
    "victor hedman": 276.8,
}


def main():
    old = pd.read_csv(OLD_CSV)
    last_map = load_last_year_stats()
    our = pd.read_csv(TOP50)
    our["nk"] = our["player"].astype(str).map(nk)
    our_map = {r["nk"]: r for _, r in our.iterrows()}

    rows = []
    for _, p in old.iterrows():
        if str(p.get("yahoo_pos", "")).upper() == "G":
            continue
        key = nk(str(p["player"]))
        ov = OVERRIDES.get(key, {})
        last = last_map.get(key)
        our_r = our_map.get(key)
        our_pred = float(our_r["pred_2026_27"]) if our_r is not None else None
        calc = project_skater(p, last, ov, our_pred)
        if key not in RECALC:
            if key in ORIGINAL_SKATER_FP:
                calc["proj"] = ORIGINAL_SKATER_FP[key]
            calc["exp_gp"] = int(ov.get("gp") or calc["exp_gp"])
            calc["flag"] = ov.get("flag") or "healthy"
        flag = calc["flag"]
        injury = {
            "healthy": "healthy",
            "DTD": "DTD",
            "IR-LT": "IR-LT",
            "IR-NR": "IR-NR",
            "tiny": "tiny",
            "SUSP": "SUSP",
            "inferred": "inferred",
        }.get(flag, flag or "healthy")
        avoid = flag in {"tiny", "IR-NR"}
        yahoo = str(p["yahoo_pos"])
        nhl_pos = str(p["nhl_pos"])
        elig_c = yahoo in {"C", "C,W"}
        elig_d = yahoo == "D"
        elig_f = not elig_d
        elig_lw = (not elig_d) and nhl_pos == "LW"
        elig_rw = (not elig_d) and nhl_pos == "RW"
        last_fp = float(p["last_year_fp"]) if pd.notna(p["last_year_fp"]) else 0
        last_gp_yahoo = int(p["last_year_gp"]) if pd.notna(p["last_year_gp"]) else 0
        our_out = our_pred if our_pred is not None else ""
        rows.append(
            {
                "player": p["player"],
                "team": p["team"],
                "yahoo_pos": yahoo,
                "nhl_pos": nhl_pos,
                "proj_fp": calc["proj"],
                "last_year_fp": last_fp,
                "last_year_gp": last_gp_yahoo,
                "our_nhl_pred_if_any": our_out,
                "pred_pts": calc["pred"],
                "exp_gp": calc["exp_gp"],
                "flag": flag,
                "injury": injury,
                "proj_gp": calc["exp_gp"],
                "avoid": avoid,
                "note": NOTES.get(key, p.get("note", "")),
                "source": "pasted",
                "c": elig_c,
                "lw": elig_lw,
                "rw": elig_rw,
                "f": elig_f,
                "d": elig_d,
                "g": False,
                "goal": calc["proj_g"] >= 34,
                "proj_g": calc["proj_g"],
            }
        )

    rows.extend(inferred_goalies())
    rows.sort(key=lambda r: (-r["proj_fp"], r["player"]))
    for i, r in enumerate(rows, 1):
        r["rank_overall"] = i

    def assign_pos(key, pred):
        n = 0
        for r in rows:
            if pred(r):
                n += 1
                r[key] = n
            else:
                r[key] = ""

    assign_pos("rank_f", lambda r: r["f"])
    assign_pos("rank_c", lambda r: r["c"])
    assign_pos("rank_lw", lambda r: r["lw"])
    assign_pos("rank_rw", lambda r: r["rw"])
    assign_pos("rank_d", lambda r: r["d"])
    assign_pos("rank_g", lambda r: r["g"])

    fields = [
        "rank_overall",
        "rank_f",
        "rank_c",
        "rank_lw",
        "rank_rw",
        "rank_d",
        "rank_g",
        "player",
        "team",
        "yahoo_pos",
        "nhl_pos",
        "proj_fp",
        "last_year_fp",
        "last_year_gp",
        "proj_gp",
        "injury",
        "note",
    ]
    out_v2 = OUT_CSV.with_name("yahoo_jeffs_draft_board_v2.csv")

    def write_csv(path):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    write_csv(out_v2)
    try:
        write_csv(OUT_CSV)
        written = OUT_CSV
    except PermissionError:
        written = out_v2
        print("main CSV locked; wrote v2 only")

    OUT_JSON.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {written} n={len(rows)}")
    print("--- OVERALL TOP 15 ---")
    for r in rows[:15]:
        print(
            f"{r['rank_overall']:3d} {r['player']:22s} {r['yahoo_pos']:4s} "
            f"{r['proj_fp']:6.1f} gp={r['exp_gp']:2d} {r['flag']}"
        )
    print("--- F TOP 12 ---")
    fs = [r for r in rows if r["f"]]
    for i, r in enumerate(fs[:12], 1):
        print(f"{i:2d} {r['player']:22s} {r['proj_fp']:6.1f} {r['flag']}")
    print("--- D TOP 8 ---")
    ds = [r for r in rows if r["d"]]
    for i, r in enumerate(ds[:8], 1):
        print(f"{i:2d} {r['player']:22s} {r['proj_fp']:6.1f} {r['flag']}")
    print("--- G ---")
    gs = [r for r in rows if r["g"]]
    for i, r in enumerate(gs, 1):
        print(f"{i:2d} {r['player']:22s} {r['proj_fp']:6.1f} {r['team']}")
    named = [
        "Aleksander Barkov",
        "Dylan Larkin",
        "Bryan Rust",
        "Brock Faber",
        "Alex Ovechkin",
        "Matt Duchene",
        "Mathew Barzal",
        "Brad Marchand",
        "Charlie McAvoy",
        "Victor Hedman",
        "Brady Tkachuk",
    ]
    print("--- NAMED ---")
    by = {r["player"]: r for r in rows}
    for n in named:
        r = by.get(n)
        if not r:
            print(n, "NOT ON BOARD")
            continue
        print(f"{r['rank_overall']:3d} {n:22s} {r['proj_fp']:6.1f} gp={r['exp_gp']} {r['flag']} {r['note'][:70]}")


if __name__ == "__main__":
    main()
