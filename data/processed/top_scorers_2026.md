# Playoff top scorers — 2026 (model output)

- **Source CSV:** `top40_scorers_2026.csv`
- **Run manifest:** `model_run_manifest.json` (2026-04-16T16:39:10.173550+00:00)
- **Health policy:** `2026-04_playoff_presumption_v1`

Predictions use the **gated** scoring stack (walk-forward trained on historical playoff points). **Pred points** combine goals + assists model with reconciliation. **Exp GP** is expected team playoff games from the V2 bracket model (+ bubble logic). Injury flags and overrides apply at **predict time** (live ESPN + `injury_overrides.json`).

## Team expected games (snapshot)

- **Bracket / standings snapshot:** 2026-04-16T12:38:23.598139
- **Playoff field (16):** ANA, BOS, BUF, CAR, COL, DAL, EDM, LAK, MIN, MTL, OTT, PHI, PIT, TBL, UTA, VGK

## Historical backtest (walk-forward, `target_points`, gated blend)

- **MAE:** 1.876  
- **Spearman:** 0.750  
- **Mean top-10 / 35 / 40 name overlap:** 58.3% / 70.0% / 71.8%

## Top 40 predicted playoff scorers

| Rk | Player | Team | Pos | Pred Pts | 80% CI | G | A | Exp GP | Reg Pts | Notes |
|----|--------|------|-----|----------|--------|---|---|--------|---------|-------|
| 1 | Nathan MacKinnon | COL | C | **38.5** | 22.6–36.3 | 12.9 | 25.0 | 18.0 | 127 |  |
| 2 | Martin Necas | COL | C | **27.9** | 18.5–20.7 | 9.0 | 18.3 | 18.0 | 100 |  |
| 3 | Cale Makar | COL | D | **25.7** | 18.5–19.2 | 6.2 | 18.9 | 18.0 | 78 |  |
| 4 | Connor McDavid | EDM | C | **21.3** | 15.9–16.3 | 6.5 | 13.7 | 9.9 | 134 |  |
| 5 | Brock Nelson | COL | C | **20.7** | 12.4–12.8 | 8.7 | 11.2 | 18.0 | 65 | ESPN: Day-To-Day (Undisclosed) |
| 6 | Jack Eichel | VGK | C | **20.2** | 12.8–15.9 | 6.9 | 12.6 | 13.7 | 90 |  |
| 7 | Seth Jarvis | CAR | C | **19.8** | 9.5–12.4 | 8.6 | 9.9 | 14.3 | 66 |  |
| 8 | Valeri Nichushkin | COL | R | **18.9** | 10.1–12.2 | 6.8 | 9.9 | 18.0 | 47 |  |
| 9 | Artturi Lehkonen | COL | L | **18.1** | 10.6–12.1 | 7.2 | 9.9 | 18.0 | 48 |  |
| 10 | Nikita Kucherov | TBL | R | **17.5** | 12.4–12.8 | 5.0 | 12.0 | 10.7 | 130 |  |
| 11 | Sebastian Aho | CAR | C | **16.7** | 10.6–12.8 | 6.2 | 9.8 | 14.3 | 80 |  |
| 12 | Mark Stone | VGK | R | **16.6** | 11.0–12.4 | 6.2 | 9.8 | 13.7 | 73 |  |
| 13 | Leon Draisaitl | EDM | C | **16.3** | 11.3–11.5 | 5.8 | 10.5 | 9.9 | 97 | Lower-body; expected early R1 — Sportsnet/NHL Apr 2026. |
| 14 | Mitch Marner | VGK | R | **16.2** | 12.1–12.2 | 5.0 | 10.2 | 13.7 | 80 |  |
| 15 | Tage Thompson | BUF | C | **16.0** | 9.5–12.1 | 7.8 | 8.7 | 13.8 | 81 |  |
| 16 | Andrei Svechnikov | CAR | R | **15.3** | 8.9–12.4 | 6.2 | 7.3 | 14.3 | 70 |  |
| 17 | Pavel Dorofeyev | VGK | R | **15.1** | 8.7–9.5 | 6.8 | 7.3 | 13.7 | 64 |  |
| 18 | Gabriel Landeskog | COL | L | **14.7** | 8.7–9.4 | 5.0 | 8.2 | 18.0 | 35 |  |
| 19 | Rasmus Dahlin | BUF | D | **14.5** | 9.5–12.1 | 3.2 | 10.2 | 13.8 | 74 |  |
| 20 | Evan Bouchard | EDM | D | **14.0** | 8.8–12.1 | 2.8 | 10.2 | 9.9 | 92 |  |
| 21 | Evgeni Malkin | PIT | C | **13.8** | 6.4–10.6 | 4.1 | 7.3 | 10.3 | 61 |  |
| 22 | Nikolaj Ehlers | CAR | L | **13.6** | 8.4–12.1 | 5.0 | 7.3 | 14.3 | 71 |  |
| 23 | Jake Guentzel | TBL | C | **13.6** | 9.2–10.1 | 5.0 | 8.7 | 10.7 | 88 |  |
| 24 | Shayne Gostisbehere | CAR | D | **13.6** | 6.3–11.0 | 2.5 | 8.3 | 14.3 | 50 |  |
| 25 | Tomas Hertl | VGK | C | **13.4** | 7.6–8.7 | 5.0 | 7.3 | 13.7 | 58 |  |
| 26 | Mikko Rantanen | DAL | R | **13.3** | 8.4–11.0 | 3.7 | 8.7 | 9.5 | 77 |  |
| 27 | Jason Robertson | DAL | L | **13.1** | 9.4–10.1 | 5.5 | 7.3 | 9.5 | 96 |  |
| 28 | Alex Tuch | BUF | R | **12.6** | 8.4–10.6 | 5.4 | 6.3 | 13.8 | 66 |  |
| 29 | Sidney Crosby | PIT | C | **12.4** | 8.4–10.4 | 4.2 | 7.2 | 10.3 | 74 |  |
| 30 | Darren Raddysh | TBL | D | **12.3** | 6.4–8.4 | 3.1 | 8.2 | 10.7 | 70 |  |
| 31 | Wyatt Johnston | DAL | C | **12.3** | 8.1–9.5 | 5.0 | 6.3 | 9.5 | 86 |  |
| 32 | Brent Burns | COL | D | **11.8** | 6.3–7.6 | 3.1 | 7.1 | 18.0 | 34 |  |
| 33 | Jackson Blake | CAR | R | **11.8** | 6.4–8.4 | 4.5 | 5.9 | 14.3 | 53 |  |
| 34 | Zach Hyman | EDM | L | **11.6** | 6.0–7.9 | 5.2 | 5.5 | 9.9 | 51 | Oilers late-RS caution; Knoblauch Apr 2026 — playoff-focused rest. |
| 35 | Brandon Hagel | TBL | L | **11.5** | 7.6–8.7 | 5.0 | 6.2 | 10.7 | 74 |  |
| 36 | Erik Karlsson | PIT | D | **11.3** | 6.4–9.4 | 1.9 | 8.7 | 10.3 | 66 |  |
| 37 | Ryan Nugent-Hopkins | EDM | C | **10.8** | 6.3–6.9 | 3.5 | 7.1 | 9.9 | 55 |  |
| 38 | Ivan Barbashev | VGK | L | **10.8** | 6.4–7.6 | 3.5 | 7.1 | 13.7 | 61 |  |
| 39 | Josh Doan | BUF | R | **10.7** | 6.3–6.9 | 4.5 | 5.8 | 13.8 | 52 |  |
| 40 | Jason Zucker | BUF | L | **10.7** | 6.3–6.4 | 4.5 | 5.8 | 13.8 | 45 |  |

*Full player list (all contender teams):* `processed/scoring_predictions_2026.csv`