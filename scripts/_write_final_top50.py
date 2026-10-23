# One-off writer for the final Sep 28 2026-27 top-50 board. Not a pipeline step.
import csv
import os
from pathlib import Path

ROWS = [
    (1, "Connor McDavid", "EDM", "C", 138, 82, 136, 135, "Y",
     "IN. Last year 138 in 82 (1.68). Career 1.53 and last-5 1.66; 138 is the mean, not a spike. Camp healthy, Draisaitl/Bouchard still PP1, every points sheet 128-135. DA did not contest the 1 lock. Pred 136 is 82 GP at ~1.66."),
    (2, "Nathan MacKinnon", "COL", "C", 127, 80, 131, 130, "Y",
     "IN. Last year 127 in 80 (1.59), 53 G. Career 1.20 is old; last-5 1.54 and last-3 1.59 are the player. Necas still on the wing, opener vs LA Sep 29, no health flag. DA left him. Pred 131 is 84 healthy games at ~1.55."),
    (3, "Nikita Kucherov", "TBL", "RW", 130, 76, 126, 125, "Y",
     "IN. Last year 130 in 76 (1.71), Hart. Career 1.28; last-3 1.68 so 1.7 is the baseline when he plays. Healthy camp; Guentzel/Hagel/Point stay. Unsigned, still TBL. Pred 126 is the 76-GP / age-33 haircut."),
    (4, "Macklin Celebrini", "SJS", "C", 115, 82, 116, 122, "Y",
     "IN. Last year 115 in 82 (1.40) at 19. Two NHL seasons (0.90 then 1.38); career 1.16 still climbing. Will Smith remains the linemate. Pred 116 splits NHL.com 122 and models 104-111. DA did not force a rank change."),
    (5, "Leon Draisaitl", "EDM", "C", 97, 65, 115, 117, "Y",
     "IN. Last year 97 in 65 (1.49). Career 1.23; last-5 1.45 so 1.49 is recent-normal. Camp healthy; McDavid/Bouchard stack intact. GP has declined three straight years -- that is the only negotiation. Pred 115 is ~80 games at 1.45."),
    (6, "David Pastrnak", "BOS", "RW", 100, 77, 105, 102, "Y",
     "IN. Last year 100 in 77 (1.30) on 11.1 SH% and 29 G. Career 1.12; last-3 1.31 through linemate roulette. Named captain Sep 27; no health flag. Pred 105 is 84 games at ~1.27 with goals bouncing toward 40."),
    (7, "Kirill Kaprizov", "MIN", "LW", 89, 78, 99, 99, "Y",
     "IN. Last year 89 in 78 (1.14), 45 G -- a DOWN counting year vs career 1.20 and last-5 1.24. Full season of Quinn Hughes on PP1. No new injury. Pred 99 is Hughes-on-PP1, not last year's 89."),
    (8, "Jason Robertson", "DAL", "LW", 96, 82, 97, 98, "Y",
     "IN. Last year 96 in 82 (1.17), 45 G, 294 shots, zero missed games four years. Career 1.08; last-3 1.04. Johnston/Rantanen stay; Seguin tracking Oct 2 is not PP1 theft. Pred 97 is 84 games at last year's rate minus a little Seguin."),
    (9, "Martin Necas", "COL", "C", 100, 78, 98, 99, "Y",
     "IN. Last year 100 in 78 (1.28) next to MacKinnon. Career 0.82 is Carolina; last-3 1.01 includes the spike. Year-two with MacKinnon. FreshSheets 85 is the mean-reversion; Nate 103 still a 32. Pred 98 stops copying the first 100. DA did not force him out."),
    (10, "Jack Eichel", "VGK", "C", 90, 74, 97, 96, "Y",
     "IN. Last year 90 in 74 (1.22). Career 1.01 is Buffalo; last-3 1.18. Marner year-two; Dorofeyev leaving cuts a finisher. GP 74/77/63 last three. Pred 97 is ~80 GP at 1.20. Camp healthy."),
    (11, "Nick Suzuki", "MTL", "C", 101, 82, 96, 98, "Y",
     "IN. Last year 101 in 82 (1.23) plus Selke. Career 0.89; last-3 1.09 so the first 100 is the spike. Caufield/Hutson/Demidov (foot bruise, expected opener) still around him; he never sits. Every points sheet 90-98. Pred 96 is regression from 101."),
    (12, "Mikko Rantanen", "DAL", "RW", 77, 64, 94, 96, "Y",
     "IN. Last year 77 in 64 (1.20) until Olympic injury. Career 1.09; last-5 1.22. Healthy camp; Robertson/Johnston stack. Pred 94 is 78 games at 1.20. 22 G in 64 was a finishing drop already baked in."),
    (13, "Wyatt Johnston", "DAL", "C", 86, 82, 92, 96, "Y",
     "IN. Last year 86 in 82 (1.05), 45 G, 27 PPG, still 23, ironman three years. Career 0.80 is year-1-2; last-3 0.90. Pred 92 splits NHL 96 and models 81-86: high-30s G, mid-80s pts. Seguin/Hintz extra mouths."),
    (14, "Kyle Connor", "WPG", "LW", 92, 82, 91, 93, "Y",
     "IN. Last year 92 in 82 (1.12). Career 0.97; last-5 1.09 so 92 is recent Connor. Never sits; Scheifele still 1C. Hellebuyck still suspended late Sep is environment, not ice time. Pred 91 is 84-game ironman at ~1.10."),
    (15, "William Nylander", "TOR", "RW", 79, 65, 90, 88, "Y",
     "IN. Last year 79 in 65 (1.22). Career 0.92; last-3 1.14 so 1.22 is a high year. Matthews practiced and is opening-night healthy; Raddysh is the new PP QB. Pred 90 needs that stack and ~77 GP, not an 84-game lock after 65 last year."),
    (16, "Auston Matthews", "TOR", "C", 53, 60, 88, 86, "Y",
     "IN. Last year 53 in 60 (0.88) on a Grade-3 MCL. Career 1.13, last-5 1.21, last-3 1.14. Chayka 100% green-light, practiced, opener vs MTL Sep 29. DA would not force him out; author keeps him. Pred 88 because he has not played 82 since 2016-17."),
    (17, "Cole Caufield", "MTL", "RW", 88, 81, 87, 92, "Y",
     "IN. Last year 88 in 81 (1.09), 51 G. Career 0.83; last-3 0.91 so 51 G is the spike. Suzuki/Hutson still feed him; Demidov expected opener. DA treated this as a toss-up vs models 75-84 and did not force him out. Pred 87 overrules NHL.com 92 with the model pile; shot volume keeps him in."),
    (18, "Matt Boldy", "MIN", "LW", 85, 76, 90, 90, "Y",
     "IN. Last year 85 in 76 (1.12), 42 G. Career 0.91; last-3 0.97 so 1.12 is a step up. Full year with Hughes and Kaprizov. NHL and Nate agree at 90. Minnesota PP cannot feed three 90-point skaters equally; Pred 90 is already crowded-PP."),
    (19, "Cale Makar", "COL", "D", 79, 75, 89, 87, "Y",
     "IN. Last year 79 in 75 (1.05) -- a DOWN year vs career 1.08 and last-5 1.12. Played 232/246 last three; MacKinnon PP1. DA left the D 1:1. Pred 89. Hutson stays out of the 32."),
    (20, "Evan Bouchard", "EDM", "D", 95, 82, 87, 86, "Y",
     "IN. Last year 95 in 82, only D there. Career 0.77; last-3 0.99 so 95 is the high and ~1.00 is the established PP1 D. McDavid/Draisaitl still run it. Pred 87 is 95 minus regression. Last year's counting does not outrank Makar's career 1.08."),
    (21, "Mitch Marner", "VGK", "RW", 80, 81, 88, 89, "Y",
     "IN. Last year 80 in 81 (0.99) year-one Vegas. Career 1.11; last-5 1.21 so 80 was the system haircut. Year-two with Eichel, same PP. Pred 88 is 84 games at 1.05-1.10. Overrule DFO 34 (category)."),
    (22, "Artemi Panarin", "LAK", "LW", 84, 78, 88, 89, "Y",
     "IN. Last year 84 in 78 (1.08) year-one LA. Career 1.15; last-3 1.23 so LA was a mild step down. Fiala second left-leg procedure Sep 10, Nov 1 hope -- more Panarin PP in October. Every points sheet 88-91. Age 34; pie splits when Fiala returns."),
    (23, "Jake Guentzel", "TBL", "C", 88, 81, 86, 83, "Y",
     "IN. Last year 88 in 81 (1.09) on Kucherov's line. Career 0.97; last-5 1.05 so 88 is the high end. Raddysh left TOR; Kucherov/Hagel/Point remain. NHL 83 over-discounts PP. Pred 86 assumes PP1 on a crowded unit."),
    (24, "Sidney Crosby", "PIT", "C", 74, 68, 87, 90, "Y",
     "IN. Last year 74 in 68 (1.09). Career 1.24; last-5 1.14 and last-3 1.13 so age 39 has settled near 1.10. Still a Penguin, opener Sep 30 at PHI; scored in the preseason finale. Rust being evaluated does not pull him. DA would not force out; author keeps him. Pred 87 is ~80 GP at 1.09."),
    (25, "Mark Scheifele", "WPG", "C", 103, 82, 87, 92, "Y",
     "IN. Last year 103 in 82 (1.26) on 20.6 SH%. Career 0.95; last-3 1.11 so 103 is the spike. Still 21-minute 1C next to Connor. Even after SH% regression he is 85-90. Hellebuyck suspension is why Pred 87 sits under NHL 92. Sheets 83-92."),
    (26, "Quinn Hughes", "MIN", "D", 76, 74, 86, 86, "Y",
     "IN. Last year 76 in 74 (1.03) VAN/MIN split. Career 0.96; last-3 1.09. Full season running Kaprizov PP; unsigned, still MIN, no holdout. DA left D 1:1. Pred 86 is the full-Minnesota year. Hutson/Dahlin stay out."),
    (27, "Clayton Keller", "UTA", "C", 88, 82, 86, 87, "Y",
     "IN. Last year 88 in 82 (1.07). Career 0.87; last-5 1.03 so 88 is recent Keller. Never sits; Utah still runs through him. Guenther/Cooley can steal PP finishes. Pred 86 is 84-game ironman at ~1.02."),
    (28, "Matthew Tkachuk", "FLA", "LW", 34, 31, 84, 87, "Y",
     "IN. Last year 34 in 31 (1.10) after hernia. Career 1.00; last-5 1.21; last-3 1.10. Barkov healthy Sep 29; Marchand out October. Brady (DTD, expected Tue vs CAR) starts with Barkov-Reinhart so Matthew's 5v5 is still Bennett/Verhaeghe plus crowded PP. DA would not force him out. Pred 84 is ~76 GP at 1.10."),
    (29, "Brandon Hagel", "TBL", "LW", 74, 71, 83, 82, "Y",
     "IN. Last year 74 in 71 (1.04) on Kucherov's line. Career 0.83 is early years; last-3 1.02 so 1.04 is now the player. Same teammates, healthy. DA would not force him out; author keeps him on 4/5 points sheets (Nate 91, FanPts 86, NHL 82, VOR 81). DFO 51 is category. Pred 83 is ~80 GP at 1.04."),
    (30, "Zach Werenski", "CBJ", "D", 81, 75, 83, 83, "Y",
     "IN. Last year 81 in 75 (1.08), Norris, 240+ shots. Career 0.72; last-3 0.97 so this is a two-year breakout. DA left him as the fourth D. Pred 83 matches NHL.com. Columbus offense is thinner; 83 is not 90. Hutson stays out of the 32."),
    (31, "Tage Thompson", "BUF", "C", 81, 81, 83, 83, "Y",
     "IN. Last year 81 in 81 (1.00), 40 G. Career 0.77; last-5 0.97; last-3 0.92 so 81 is the recent floor. Lost Tuch (sign-and-trade Jun 24) is a nibble; still BUF PP1. DA would not force him out; author keeps him on 81/81 plus NHL 83. Pred 83 is 84 games at ~1.00."),
    (32, "Tim Stutzle", "OTT", "LW", 83, 80, 82, 86, "Y",
     "IN. Last year 83 in 80 (1.04) WITH Brady. Career 0.91, last-5 0.96, last-3 0.98 on 75/82/80 GP. New teammates: Batherson/Burakovsky/Eklund, not Brady. DA forced Stutzle IN / DeBrincat OUT: Pred 82 and last-year 83/80 and NHL.com 86 and 4/5 sheets IN cannot sit behind a Pred-80 DeBrincat. Author ACCEPTS that swap. Pred 82 already prices the Brady loss. On OTT roster, not a ghost."),
    (33, "Alex DeBrincat", "DET", "RW", 85, 82, 80, 79, "N",
     "OUT. Last year 85 in 82 (1.04), 41 G, ironman -- a career high. Career 0.86, last-5 0.89, last-3 0.90 (67/70/85). Larkin missed preseason, Oct 2 uncertain. DA forced DeBrincat OUT / Stutzle IN: Pred 80 cannot stay in the 32 while Stutzle is 82 with 83/80 last year, NHL 86, and 4/5 sheets IN. Author ACCEPTS; ironman 41 G is recency on a career high, not a sharper overrule. Sheets 75-79. Pred 80. First out, 33rd."),
    (34, "Brady Tkachuk", "FLA", "LW", 59, 60, 79, 87, "N",
     "OUT. Last year 59 in 60 (0.98) in Ottawa. Career 0.81, last-5 0.90, last-3 0.88. Florida upgrade is real (Barkov-Reinhart 5v5, brothers on PP1). DTD since Sep 22; Maurice expects him Tue vs CAR. DA would not force him in; author keeps him out. Pred 79 (was 83) trims toward the 63-69 points-model cluster plus a Florida bump so rank 34 and Pred stop lying. DFO 17 is hits."),
    (35, "Jack Hughes", "NJD", "C", 77, 61, 82, 84, "N",
     "OUT. Last year 77 in 61 (1.26). Career 1.00 (428 pts in 429 GP); last-3 1.20. GP tax: 61-56-49-78-62-62-61 -- last-3 mean 61.7, only one 70+ season in seven years. Camp healthy, Mantha/Bratt on his line. DA would not force him in (5/5 sheets is the miss-you-hate; 7-year GP still prices him as the next 82, not a 32). Author keeps him out. Pred 82 is ~65 GP at 1.26."),
    (36, "Sebastian Aho", "CAR", "C", 80, 79, 82, 81, "N",
     "OUT vs the new 32nd (Stutzle Pred 82). Last year 80 in 79 (1.01). Career 0.89; last-3 1.03. Jarvis injured non-roster Sep 27, late-Oct return -- usage bump is weeks. Sheets 79-84 sit on the cutline; Stutzle's 83/80 plus 4/5 sheets win the chair. DA did not force Aho in. Pred 82."),
    (37, "Brayden Point", "TBL", "C", 50, 63, 79, 85, "N",
     "OUT. Last year 50 in 63 (0.79), 18 G. Career 0.95; last-5 1.02; last-3 1.01 so 0.79 is the outlier and prior years were 40-goal seasons. Still Kucherov PP; no new camp injury. DA would not force in; OUT matches the model cluster. Pred 79 does not treat the 40-goal bounce as the mean."),
    (38, "Dylan Guenther", "UTA", "RW", 73, 79, 79, 83, "N",
     "OUT. Last year 73 in 79 (0.92), already 40 G. Career 0.81; last-3 0.87 so assists have not caught the shot. Keller/Cooley remain. NHL 83 is the only 32-ish number. DA would not force in. Pred 79 is the no-leap."),
    (39, "Aleksander Barkov", "FLA", "C", 0, 0, 80, 83, "N",
     "OUT. Zero NHL games last year (ACL/MCL). Prior 71 in 67 (1.06); career 0.97; last-5 1.15. Healthy Sep 29 at CAR. Brady (DTD, expected opener) and Reinhart on his line; Matthew on the other. DA would not force in. Pred 80. Nate 88 would put him in; a full healthy year is the miss you hate."),
    (40, "Sam Reinhart", "FLA", "C", 61, 64, 80, 84, "N",
     "OUT. Last year 61 in 64 (0.95), 29 G. Career 0.81; last-3 1.05 so 0.95 was down without Barkov. Barkov back helps; Brady now starts on the same line. DA would not force in. Fourth Panther in this band. Pred 80. Recheck vs 32nd Stutzle: Stutzle's 83/80 and 4/5 sheets win."),
    (41, "Lane Hutson", "MTL", "D", 78, 82, 81, 80, "N",
     "OUT vs the new 32nd (Stutzle Pred 82). Last year 78 in 82 (0.95), 66 assists. Career 0.88 in 166 GP. Suzuki/Caufield PP1. D 1:1 already has Bouchard/Makar/Q Hughes/Werenski; DA would not force Hutson in. Pred 81 is a 50 lock. First D replacement if one of those four misses time."),
    (42, "Leo Carlsson", "ANA", "C", 67, 70, 80, 82, "N",
     "OUT. Last year 67 in 70 (0.96). Career 0.70 in 200 GP; last year is the leap not the mean. McTavish gone, full 1C next to Gauthier. Needs another leap, not just 84 games. Nate 85 is the boom; no-leap is 69-72. Pred 80."),
    (43, "Cutter Gauthier", "ANA", "LW", 69, 76, 78, 80, "N",
     "OUT. Last year 69 in 76 (0.91), 41 G, 286 shots. Career 0.72; last year is the only real sample. Carlsson 1C. DFO 32 is category (shots). Points-only 75-80. Pred 78. Goal-only boards already rank him near 32; we do not."),
    (44, "Lucas Raymond", "DET", "LW", 76, 80, 79, 77, "N",
     "OUT. Last year 76 in 80 (0.95). Career 0.83; last-3 0.93 so 76 is the player. Larkin cloudy Oct 2 can raise early usage next to DeBrincat. Detroit ceiling. Pred 79. Came-from-nowhere is Larkin missing a chunk."),
    (45, "Mika Zibanejad", "NYR", "C", 78, 81, 77, 79, "N",
     "OUT. Last year 78 in 81 (0.96). Career 0.81; last-3 0.87 after Panarin left midyear. Still NYR PP1; Dorofeyev is a shooter not a playmaker replacement. 77 is floor and ceiling. Pred 77."),
    (46, "Filip Forsberg", "NSH", "LW", 75, 82, 77, 80, "N",
     "OUT. Last year 75 in 82 (0.91), 40 G 35 A. Career 0.88; last-5 1.01; last-3 0.99 so 75 is a down counting year on a bad team. Same Nashville usage. FreshSheets 81 is the IN-adjacent number. Pred 77."),
    (47, "Will Smith", "SJS", "C", 59, 69, 76, 75, "N",
     "OUT. Last year 59 in 69 (0.86) as Celebrini's linemate. Career 0.72 in 142 GP. Same stack. Needs 15-20 more points. Pred 76. Came-from-nowhere is 18:11 TOI jumping."),
    (48, "Logan Cooley", "UTA", "C", 43, 54, 74, 80, "N",
     "OUT. Last year 43 in 54 (0.80) on 23.5 SH%. Career 0.72; 2024-25 was 65 in 75 (0.87) -- the healthier year is the better read. Keller/Guenther PP. 23.5 SH% will not repeat. NHL 80 is the boom; 80 GP at 0.87 is 70. Pred 74."),
    (49, "Ivan Demidov", "MTL", "RW", 62, 82, 74, 70, "N",
     "OUT. Rookie 62 in 82 (0.76); that year is the career. Shot off the foot in preseason; Gazette/RotoWire expect opening night. Suzuki/Caufield PP; he played 15:30. Mean is not 90. Pred 74. Not a ghost; on the MTL roster."),
    (50, "Rasmus Dahlin", "BUF", "D", 74, 77, 75, 75, "N",
     "OUT. Last year 74 in 77 (0.96). Career 0.74; last-3 0.87 so 74 is the high. Tuch/Byram gone; Thompson remains PP trigger. Four D already in. DFO 23 is category. Pred 75. Nate 78 plus healthy Thompson PP is how 75 becomes 85."),
]

PRIMARY = Path(r"C:\Users\RobDods\Apps\Cursor\Predictors\Cursor_NHLPO_Predict\data\processed\nhl_2026_27_top50_scorers.csv")
FALLBACK = Path(r"C:\Users\RobDods\Apps\Cursor\Predictors\Cursor_NHLPO_Predict\data\processed\nhl_2026_27_top50_scorers_final.csv")
HEADER = ["rank", "player", "team", "pos", "last_year_pts", "last_year_gp", "pred_2026_27", "nhl_com", "chosen_top32", "justification"]


def write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(HEADER)
        for r in ROWS:
            w.writerow(r)


assert len(ROWS) == 50
assert [r[8] for r in ROWS] == ["Y"] * 32 + ["N"] * 18
assert ROWS[31][1] == "Tim Stutzle" and ROWS[31][6] == 82 and ROWS[31][8] == "Y"
assert ROWS[32][1] == "Alex DeBrincat" and ROWS[32][6] == 80 and ROWS[32][8] == "N"
assert ROWS[30][1] == "Tage Thompson" and ROWS[30][8] == "Y"
assert ROWS[33][1] == "Brady Tkachuk" and 78 <= ROWS[33][6] <= 80
assert ROWS[34][1] == "Jack Hughes" and ROWS[34][8] == "N"

written = None
lock_note = "none"
try:
    write_csv(PRIMARY)
    written = PRIMARY
except PermissionError as e:
    lock_note = f"primary locked ({e}); wrote fallback then replace"
    write_csv(FALLBACK)
    try:
        os.replace(FALLBACK, PRIMARY)
        written = PRIMARY
        lock_note = "primary locked on direct write; replace from fallback succeeded"
    except PermissionError as e2:
        written = FALLBACK
        lock_note = f"primary still locked after replace ({e2}); shipped fallback"

print(f"path={written}")
print(f"rows={len(ROWS)}")
print(f"32nd={ROWS[31][1]} pred={ROWS[31][6]} chosen={ROWS[31][8]}")
print(f"33rd={ROWS[32][1]} pred={ROWS[32][6]} chosen={ROWS[32][8]}")
print(f"lock={lock_note}")
