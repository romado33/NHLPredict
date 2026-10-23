# 2026 NHL Playoff Predictions

> **Last updated: April 7, 2026** (fresh API data: standings, team stats, skaters, rosters)  
> **Runs:** `python scripts/run_v2_pipeline.py --predict-only` and `python scripts/run_scoring_pipeline.py --predict-only`  
> Models use saved weights; inputs are live from NHL API + MoneyPuck.

---

## Current standings snapshot (April 7, 2026)

### Eastern Conference

| Rk | Team | Pts | GP | Proj Pts | Playoff |
|----|------|-----|-----|----------|---------|
| 1 | CAR | 104 | 77 | 110.8 | In (95%) |
| 2 | TBL | 102 | 77 | 108.6 | In (95%) |
| 3 | BUF | 102 | 78 | 107.2 | In (95%) |
| 4 | MTL | 100 | 77 | 106.5 | In (95%) |
| 5 | PIT | 96 | 78 | 100.9 | In (85%) |
| 6 | BOS | 95 | 78 | 99.9 | In (85%) |
| 7 | OTT | 90 | 77 | 95.8 | In (60%) |
| 8 | PHI | 90 | 77 | 95.8 | In (60%) |
| — | NYI, DET, CBJ, WSH, … | — | — | — | Bubble / out |

### Western Conference

| Rk | Team | Pts | GP | Proj Pts | Playoff |
|----|------|-----|-----|----------|---------|
| 1 | COL | 110 | 76 | 118.7 | In (99%) |
| 2 | DAL | 102 | 77 | 108.6 | In (99%) |
| 3 | MIN | 100 | 77 | 106.5 | In (99%) |
| 4 | EDM | 87 | 77 | 92.6 | In (70%) |
| 5 | ANA | 87 | 77 | 92.6 | In (70%) |
| 6 | UTA | 86 | 76 | 92.8 | In (70%) |
| 7 | VGK | 86 | 77 | 91.6 | In (70%) |
| 8 | LAK | 83 | 77 | 88.4 | In (60%) |
| — | NSH, SJS, WPG, STL, … | — | — | — | Bubble / out |

**Scoring pipeline playoff teams (16):** ANA, BOS, BUF, CAR, COL, DAL, EDM, LAK, MIN, MTL, OTT, PHI, PIT, TBL, UTA, VGK  
**Bubble (example):** CBJ, DET, NSH, NYI, SJS, STL, WPG, WSH

---

## Bracket (division seeding — as modeled)

### East

- **Atlantic:** TBL (D1), BUF (D2), MTL (D3)
- **Metro:** CAR (D1), PIT (D2), PHI (D3)
- **Wild cards:** BOS (WC1), OTT (WC2)

**First round:** CAR vs OTT | PIT vs PHI | TBL vs BOS | BUF vs MTL  

### West

- **Central:** COL (D1), DAL (D2), MIN (D3)
- **Pacific:** EDM (D1), ANA (D2), VGK (D3)
- **Wild cards:** UTA (WC1), LAK (WC2)

**First round:** COL vs LAK | DAL vs MIN | EDM vs UTA | ANA vs VGK  

---

## Series predictions (V2 ensemble + Monte Carlo, 50k sims / series)

### Round 1 — East

| Series | Predicted winner | MC win % | Exp. games | Sweep / 5 / 6 / 7 |
|--------|-----------------|----------|------------|-------------------|
| CAR vs OTT | **CAR** | 89.6% | 5.3 | 25% / 31% / 20% / 14% |
| PIT vs PHI | **PIT** | 95.0% | 5.1 | 34% / 34% / 17% / 10% |
| TBL vs BOS | **TBL** | 87.9% | 5.4 | 23% / 30% / 20% / 15% |
| BUF vs MTL | **BUF** | 92.7% | 5.2 | 29% / 33% / 19% / 12% |

### Round 1 — West

| Series | Predicted winner | MC win % | Exp. games | Sweep / 5 / 6 / 7 |
|--------|-----------------|----------|------------|-------------------|
| COL vs LAK | **COL** | 98.5% | 4.5 | 49% / 33% / 13% / 4% |
| DAL vs MIN | **DAL** | 61.8% | 5.8 | 9% / 17% / 17% / 19% |
| EDM vs UTA | **EDM** | 91.1% | 5.3 | 27% / 32% / 19% / 13% |
| ANA vs VGK | **VGK** | 94.8% (VGK) | 5.0 | 0% / 1% / 1% / 3% |

*Note: For ANA vs VGK, Monte Carlo assigns **VGK** ~94.8% to win the series (ANA ~5.2%); this differs strongly from the raw model matchup probability.*

### Round 2

| Series | Predicted winner | MC win % | Exp. games |
|--------|-----------------|----------|------------|
| CAR vs PIT | **CAR** | 74.3% | 5.7 |
| TBL vs BUF | **TBL** | 54.0% | 5.8 |
| COL vs DAL | **COL** | 98.1% | 4.6 |
| EDM vs VGK | **VGK** | 53.3% | 5.8 |

### Conference finals

| Series | Predicted winner | MC win % | Exp. games |
|--------|-----------------|----------|------------|
| CAR vs TBL | **CAR** | 74.5% | 5.7 |
| COL vs VGK | **COL** | 98.4% | 4.6 |

### Stanley Cup Final

| Series | Predicted winner | Cup win % | Exp. games |
|--------|-----------------|-----------|------------|
| CAR vs **COL** | **Colorado Avalanche** | **95.5%** | 5.0 |

### Bayesian Round 1 uncertainty (90% credible intervals, higher seed win prob)

| Series | Point est. | 90% CI |
|--------|------------|--------|
| CAR vs OTT | 68.1% | 62.9% - 72.5% |
| PIT vs PHI | 74.7% | 69.5% - 79.2% |
| TBL vs BOS | 65.6% | 60.5% - 70.1% |
| BUF vs MTL | 72.2% | 67.0% - 76.7% |
| COL vs LAK | 89.0% | 83.8% - 93.4% |
| DAL vs MIN | 49.5% | 44.3% - 53.9% |
| EDM vs UTA | 69.6% | 64.4% - 74.0% |
| ANA vs VGK | 20.6% (ANA) | 15.4% - 25.0% |

---

## Probability-weighted expected playoff games (April 7)

Used by the scoring model (sample):

| Team | Exp. GP | Team | Exp. GP |
|------|---------|------|---------|
| COL | 18.2 | CAR | 16.7 |
| TBL | 13.2 | EDM | 10.6 |
| VGK | 12.8 | BUF | 10.6 |
| PIT | 10.5 | DAL | 8.7 |
| OTT | 5.3 | MIN | 5.8 |
| … | … | … | … |

Full table: `data/processed/team_expected_games_2026.json`

---

## Top 35 predicted playoff scorers (April 7)

Data: `data/processed/top35_scorers_2026.csv`  
Fetched: April 7, 2026 ~09:12 local (see `last_scoring_predict_run.txt`).

| Rk | Player | Team | Pos | Pts | 80% CI | G | A | Exp GP | Reg Pts |
|----|--------|------|-----|-----|--------|---|---|--------|---------|
| 1 | Nathan MacKinnon | COL | C | 32.9 | 19.7-38.6 | 10.3 | 21.2 | 18.2 | 122 |
| 2 | Martin Necas | COL | C | 23.0 | 15.9-19.7 | 7.7 | 14.9 | 18.2 | 94 |
| 3 | Cale Makar | COL | D | 21.3 | 15.9-16.6 | 3.1 | 17.2 | 18.2 | 75 |
| 4 | Nikita Kucherov | TBL | R | 20.7 | 15.4-18.3 | 4.9 | 14.0 | 13.2 | 126 |
| 5 | Connor McDavid | EDM | C | 18.0 | 12.9-15.5 | 4.0 | 13.2 | 10.6 | 126 |
| 6 | Sebastian Aho | CAR | C | 16.7 | 12.0-12.5 | 5.1 | 11.3 | 16.7 | 78 |
| 7 | Seth Jarvis | CAR | C | 16.5 | 11.3-12.5 | 6.3 | 9.9 | 16.7 | 66 |
| 8 | Jack Eichel | VGK | C | 16.4 | 11.7-12.6 | 4.7 | 10.9 | 12.8 | 82 |
| 9 | Brock Nelson | COL | C | 15.7 | 11.4-11.8 | 5.8 | 9.2 | 18.2 | 63 |
| 10 | Jake Guentzel | TBL | C | 14.5 | 10.1-11.3 | 4.2 | 9.5 | 13.2 | 86 |
| 11 | Andrei Svechnikov | CAR | R | 14.3 | 9.7-10.6 | 5.1 | 8.8 | 16.7 | 67 |
| 12 | Mark Stone | VGK | R | 13.6 | 8.9-10.2 | 4.3 | 9.1 | 12.8 | 67 |
| 13 | Artturi Lehkonen | COL | L | 13.6 | 9.2-10.1 | 4.5 | 8.7 | 18.2 | 46 |
| 14 | Mitch Marner | VGK | R | 13.1 | 8.9-10.4 | 3.4 | 8.9 | 12.8 | 77 |
| 15 | Nikolaj Ehlers | CAR | L | 12.7 | 8.8-9.4 | 4.3 | 8.1 | 16.7 | 65 |
| 16 | Brandon Hagel | TBL | L | 12.6 | 8.5-9.5 | 4.3 | 7.7 | 13.2 | 73 |
| 17 | Evan Bouchard | EDM | D | 12.6 | 7.9-9.8 | 1.3 | 10.7 | 10.6 | 88 |
| 18 | Valeri Nichushkin | COL | R | 12.5 | 8.2-9.5 | 3.7 | 8.2 | 18.2 | 45 |
| 19 | Darren Raddysh | TBL | D | 11.7 | 7.3-9.2 | 1.8 | 9.0 | 13.2 | 69 |
| 20 | Sidney Crosby | PIT | C | 11.0 | 7.0-9.2 | 2.7 | 6.9 | 10.5 | 72 |
| 21 | Shayne Gostisbehere | CAR | D | 10.9 | 6.4-9.0 | 1.6 | 8.4 | 16.7 | 47 |
| 22 | Evgeni Malkin | PIT | C | 10.8 | 6.0-9.1 | 2.3 | 7.1 | 10.5 | 59 |
| 23 | Gabriel Landeskog | COL | L | 9.9 | 6.3-7.4 | 2.7 | 6.7 | 18.2 | 33 |
| 24 | Pavel Dorofeyev | VGK | R | 9.8 | 6.4-7.2 | 3.9 | 5.2 | 12.8 | 61 |
| 25 | Jackson Blake | CAR | R | 9.6 | 6.5-6.8 | 3.0 | 6.3 | 16.7 | 49 |
| 26 | Brayden Point | TBL | C | 9.6 | 6.2-7.0 | 2.7 | 6.4 | 13.2 | 48 |
| 27 | Mikko Rantanen | DAL | R | 9.4 | 6.3-7.0 | 1.9 | 6.8 | 8.7 | 73 |
| 28 | Erik Karlsson | PIT | D | 9.4 | 5.4-7.9 | 0.9 | 7.4 | 10.5 | 64 |
| 29 | Jason Robertson | DAL | L | 9.2 | 6.2-7.1 | 2.9 | 5.7 | 8.7 | 89 |
| 30 | Zach Hyman | EDM | L | 9.1 | 5.8-6.5 | 3.8 | 4.9 | 10.6 | 51 |
| 31 | Tomas Hertl | VGK | C | 9.0 | 5.5-6.9 | 3.0 | 5.4 | 12.8 | 55 |
| 32 | Bryan Rust | PIT | R | 8.9 | 5.6-7.0 | 2.7 | 5.3 | 10.5 | 64 |
| 33 | Rasmus Dahlin | BUF | D | 8.8 | 5.4-6.4 | 0.8 | 7.5 | 10.6 | 70 |
| 34 | Tage Thompson | BUF | C | 8.5 | 5.6-5.9 | 3.0 | 5.1 | 10.6 | 78 |
| 35 | Wyatt Johnston | DAL | C | 8.5 | 5.8-6.2 | 3.1 | 4.9 | 8.7 | 80 |

### Roster / injury adjustments (this run)

| Player | Issue | Action |
|--------|--------|--------|
| Leon Draisaitl | Inactive 23d (last game 2026-03-15) | Points scaled **50%** (e.g. 15.2 → **7.6**); falls out of top 35 after re-rank |
| Victor Olofsson | Not on COL roster | **Excluded** (predictions set to 0) |

---

## Output artifacts

| File | Contents |
|------|----------|
| `data/processed/last_v2_predict_run.txt` | Full V2 bracket log (Apr 7) |
| `data/processed/last_scoring_predict_run.txt` | Full scoring log (Apr 7) |
| `data/processed/team_expected_games_2026.json` | Expected games + team lists |
| `data/processed/scoring_predictions_2026.csv` | All contender skaters |
| `data/processed/top35_scorers_2026.csv` | Top 35 after injury/roster pass |

---

## How to refresh again

```powershell
cd C:\Users\RobDods\Apps\Cursor_NHLPO_Predict
$env:PYTHONIOENCODING = "utf-8"
python scripts/run_v2_pipeline.py --predict-only
python scripts/run_scoring_pipeline.py --predict-only
```

To **retrain** scoring models (longer run), omit `--predict-only` on `run_scoring_pipeline.py`.
