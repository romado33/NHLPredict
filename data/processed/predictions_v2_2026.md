# NHL Playoff Predictor V2 — 2026 Playoffs

- **Generated:** 2026-04-16T16:37:45.089831+00:00
- **Source:** `predictions_v2_2026.json`

Probabilities below are **series win probabilities** (best-of-7). **Higher seed** is listed first in each matchup (home-ice format in the model). **Monte Carlo** uses 50k simulated series from blended per-game win probability.

## First Round

| Matchup | P(higher seed wins) | P(lower seed wins) | Predicted winner | Expected games | Ends in 4 / 5 / 6 / 7 (higher seed wins) |
|---------|----------------------|----------------------|------------------|----------------|----------------------------------------|
| CAR vs OTT | **69.50%** | **30.50%** | **CAR** | 5.73 | 11.3% / 20.4% / 18.6% / 19.2% |
| PIT vs PHI | **88.32%** | **11.68%** | **PIT** | 5.38 | 23.2% / 30.4% / 20.0% / 14.7% |
| BUF vs BOS | **95.44%** | **4.56%** | **BUF** | 5.02 | 35.3% / 34.2% / 16.9% / 9.1% |
| TBL vs MTL | **77.30%** | **22.70%** | **TBL** | 5.63 | 15.1% / 24.1% / 20.0% / 18.1% |
| COL vs LAK | **98.43%** | **1.57%** | **COL** | 4.53 | 48.0% / 32.9% / 12.9% / 4.6% |
| DAL vs MIN | **74.82%** | **25.18%** | **DAL** | 5.67 | 13.6% / 23.0% / 19.4% / 18.8% |
| VGK vs UTA | **92.16%** | **7.84%** | **VGK** | 5.23 | 28.4% / 32.5% / 19.1% / 12.1% |
| EDM vs ANA | **85.32%** | **14.68%** | **EDM** | 5.47 | 20.5% / 28.5% / 20.3% / 16.1% |

## Second Round

| Matchup | P(higher seed wins) | P(lower seed wins) | Predicted winner | Expected games | Ends in 4 / 5 / 6 / 7 (higher seed wins) |
|---------|----------------------|----------------------|------------------|----------------|----------------------------------------|
| CAR vs PIT | **82.46%** | **17.54%** | **CAR** | 5.54 | 18.0% / 27.3% / 20.2% / 16.9% |
| BUF vs TBL | **65.17%** | **34.83%** | **BUF** | 5.77 | 9.8% / 18.6% / 17.8% / 19.0% |
| COL vs DAL | **97.54%** | **2.46%** | **COL** | 4.77 | 42.6% / 34.1% / 14.6% / 6.3% |
| VGK vs EDM | **94.58%** | **5.42%** | **VGK** | 5.09 | 33.0% / 33.8% / 17.7% / 10.1% |

## Conference Finals

| Matchup | P(higher seed wins) | P(lower seed wins) | Predicted winner | Expected games | Ends in 4 / 5 / 6 / 7 (higher seed wins) |
|---------|----------------------|----------------------|------------------|----------------|----------------------------------------|
| CAR vs BUF | **55.99%** | **44.01%** | **CAR** | 5.81 | 7.1% / 15.0% / 15.2% / 18.7% |
| COL vs VGK | **97.92%** | **2.08%** | **COL** | 4.68 | 44.4% / 34.1% / 13.7% / 5.6% |

## Stanley Cup Final

| Matchup | P(higher seed wins) | P(lower seed wins) | Predicted winner | Expected games | Ends in 4 / 5 / 6 / 7 (higher seed wins) |
|---------|----------------------|----------------------|------------------|----------------|----------------------------------------|
| CAR vs COL | **6.01%** | **93.99%** | **COL** | 5.06 | 0.3% / 0.9% / 1.4% / 3.4% |

## Bracket outcome (model progression)

- **Eastern champion:** CAR  
- **Western champion:** COL  
- **Stanley Cup:** COL

## Most influential features (V2 blend)

Per submodel: normalized abs(logistic coef) or tree feature_importances_; blended with the same weights as win-probability; global importance scaled to max=1.

**Submodel blend weights:**
- `v2_bayesian`: 70.31%
- `v2_xgboost`: 19.35%
- `v2_logistic`: 8.73%
- `v2_lightgbm`: 1.62%

| Rank | Feature | Importance (max=1) |
|------|---------|-------------------|
| 1 | `diff_nhl_satPctAhead` | 1.000000 |
| 2 | `diff_mp_5v5_goalsFor` | 0.850960 |
| 3 | `diff_mp_5v5_blockedShotAttemptsAgainst` | 0.807835 |
| 4 | `diff_nhl_winsInRegulation` | 0.804159 |
| 5 | `diff_nhl_ppFaceoffs` | 0.774419 |
| 6 | `diff_nhl_period2GoalsAgainst` | 0.763830 |
| 7 | `diff_nhl_goalsAgainstPenaltyShots` | 0.751826 |
| 8 | `h2h_games` | 0.744156 |
| 9 | `diff_mp_5v5_reboundGoalsAgainst` | 0.735322 |
| 10 | `diff_mp_5v5_mediumDangerxGoalsFor` | 0.724402 |
| 11 | `diff_nhl_goalsAgainst5On5` | 0.713797 |
| 12 | `diff_nhl_points` | 0.700274 |
| 13 | `diff_nhl_lossesScoringFirst` | 0.684122 |
| 14 | `diff_nhl_shFaceoffs` | 0.677422 |
| 15 | `diff_nhl_period1GoalsFor` | 0.675068 |
| 16 | `diff_nhl_takeawaysPer60` | 0.661985 |
| 17 | `diff_nhl_ppFaceoffPct` | 0.660453 |
| 18 | `diff_nhl_winPctTrailPeriod2` | 0.658182 |
| 19 | `diff_nhl_penaltyKillNetPct` | 0.653868 |
| 20 | `diff_nhl_lossTrailPeriod1` | 0.649705 |
| 21 | `diff_mp_5v5_blockedShotAttemptsFor` | 0.611209 |
| 22 | `diff_nhl_winPctScoringFirst` | 0.585373 |
| 23 | `diff_goalie_team_savePct` | 0.580886 |
| 24 | `diff_nhl_goalsAgainstPerGame` | 0.578500 |
| 25 | `diff_nhl_winPctLeadPeriod1` | 0.578341 |
