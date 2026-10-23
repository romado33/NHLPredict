# NHL Playoff Series Predictor

Machine learning model that predicts NHL playoff series outcomes based on 25 years of historical data (2001-2025). Uses regular season team statistics, advanced analytics, goaltending metrics, and playoff ELO ratings to generate win probabilities for playoff matchups.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (collect data, build features, train, evaluate, predict)
python main.py full

# Or run individual steps
python main.py collect     # Fetch data from NHL API + MoneyPuck
python main.py build       # Build feature matrix
python main.py train       # Train baseline models
python main.py evaluate    # Evaluate and compare models

# Run improved training pipeline with temporal CV
python scripts/run_improved.py

# Run refined pipeline with ELO + sample weighting
python scripts/run_refined.py

# Generate current season predictions
python scripts/predict_2026.py
```

## Architecture

```
├── config/settings.py          # Central configuration
├── main.py                     # Pipeline orchestrator
├── src/
│   ├── data/
│   │   ├── nhl_api_client.py   # NHL API client (cached, rate-limited)
│   │   └── collectors.py       # Data collection from NHL API + MoneyPuck
│   ├── features/
│   │   ├── engineer.py         # Feature engineering pipeline
│   │   └── elo.py              # Playoff ELO rating system
│   ├── models/
│   │   ├── trainer.py          # Baseline model training
│   │   ├── improved_trainer.py # Improved training with temporal CV
│   │   ├── evaluator.py        # Evaluation framework
│   │   ├── feature_selection.py# Multi-method feature selection
│   │   └── predictor.py        # Prediction module
│   └── utils/
│       └── logging_config.py   # Structured logging
├── scripts/
│   ├── run_improved.py         # Improved training script
│   ├── run_refined.py          # Refined training with ELO + weighting
│   └── predict_2026.py         # Current season predictions
└── data/
    ├── raw/                    # Raw API data
    ├── processed/              # Feature matrices, reports, plots
    ├── cache/                  # API response cache
    └── models/                 # Trained model artifacts
```

## Data Sources

| Source | Data | Coverage |
|--------|------|----------|
| NHL Web API (`api-web.nhle.com`) | Standings, playoff brackets, rosters | 2001-present |
| NHL Stats API (`api.nhle.com/stats`) | 14 team stat reports (196 columns) | 2001-present |
| MoneyPuck | Advanced stats (Corsi, Fenwick, xGoals) | 2008-present |
| Computed | ELO ratings, Pythagorean expectation | Derived |

## Feature Matrix

**368 historical playoff series** with **400+ features** reduced to **50-60 selected features** through multi-method consensus selection:

**Top predictive features (from gradient boosting importance):**
1. Shots for per game differential
2. Win % when leading after 1st period
3. Shot attempt % when ahead (Corsi when leading)
4. Win % when trailing after 2nd period
5. Power play faceoff %
6. Low danger goals against
7. Goalie save percentage
8. Penalty kill efficiency
9. Goals against per game
10. Regulation wins

## Model Performance

Evaluated with **temporal cross-validation** (expanding window, no future data leakage):

| Model | CV AUC | Accuracy | Brier Score |
|-------|--------|----------|-------------|
| Logistic (L1) | 0.635 | 59.7% | 0.269 |
| XGBoost | 0.657 | 59.7% | 0.241 |
| LightGBM | 0.643 | 58.9% | 0.263 |
| Blended Ensemble | 0.625 | 59.3% | 0.246 |

Season-by-season accuracy ranges from 33% (2012 - major upsets) to 80% (2013).

## How It Works

1. **Data Collection**: Fetches team stats, standings, goalie stats, and advanced analytics from the NHL API and MoneyPuck
2. **Feature Engineering**: Computes differential features (higher seed minus lower seed) for all metrics, adds Pythagorean expectation, ELO ratings, playoff experience, and interaction terms
3. **Feature Selection**: Three-method consensus (mutual information, tree importance, sequential selection) eliminates noise features
4. **Model Training**: Logistic regression (L1), XGBoost, and LightGBM with temporal cross-validation and hyperparameter optimization
5. **Calibration**: Isotonic calibration ensures predicted probabilities are well-calibrated
6. **Prediction**: Blends model outputs with optimized weights to predict series outcomes with calibrated win probabilities

## Important Notes

- The 2004-05 season is excluded (NHL lockout)
- The 2020 bubble playoffs are included but flagged as a special circumstance
- Shortened seasons (2013, 2020, 2021) are flagged
- All API responses are cached locally to avoid redundant requests
- MoneyPuck data is used with attribution per their terms of service
