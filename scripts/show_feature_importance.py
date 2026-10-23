"""
Display the top features from the refined model pipeline
and explain their meaning in plain English.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from config.settings import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE


FEATURE_DESCRIPTIONS = {
    "diff_nhl_shotsForPerGame": "Shots on goal per game",
    "diff_nhl_winPctLeadPeriod1": "Win % when leading after the 1st period",
    "diff_nhl_satPctAhead": "Shot attempt % (Corsi) when leading in-game",
    "diff_nhl_winPctTrailPeriod2": "Win % when trailing after the 2nd period (comeback ability)",
    "diff_nhl_ppFaceoffPct": "Power play faceoff win percentage",
    "diff_mp_all_lowDangerGoalsAgainst": "Low-danger goals allowed (defensive lapses)",
    "diff_nhl_period2GoalsAgainst": "Goals allowed in the 2nd period",
    "diff_nhl_winsInRegulation": "Regulation wins (decisive victories, no OT needed)",
    "diff_goalie_team_savePct": "Team save percentage",
    "diff_nhl_losses": "Total losses (fewer = better)",
    "diff_nhl_goalsAgainstPerGame": "Goals allowed per game",
    "diff_nhl_period1GoalsFor": "Goals scored in the 1st period (fast starts)",
    "diff_nhl_pkNetGoalsPerGame": "Penalty kill net goal efficiency per game",
    "diff_nhl_netPenalties": "Net penalty differential (discipline)",
    "diff_nhl_penaltyKillNetPct": "Penalty kill net % (PK minus opponent PP)",
    "diff_goalie_starter_goalsAgainstAverage": "Starting goalie's GAA",
    "diff_nhl_missedShots": "Missed shots (shot quality indicator)",
    "diff_goalie_starter_wins": "Starting goalie's win total",
    "diff_nhl_ppFaceoffs": "Power play faceoff count (volume of PP opportunities)",
    "diff_mp_5v5_mediumDangerxGoalsFor": "Expected goals from medium-danger chances at 5v5",
    "diff_nhl_winPctScoringFirst": "Win % when scoring first",
    "diff_nhl_takeawaysPer60": "Takeaways per 60 minutes (puck-stealing ability)",
    "diff_nhl_lossTrailPeriod1": "Losses when trailing after 1st period",
    "diff_mp_5v5_reboundGoalsAgainst": "Rebound goals allowed at 5v5 (goalie control)",
    "diff_mp_all_takeawaysFor": "Total takeaways (all situations)",
    "diff_nhl_goalsAgainstPenaltyShots": "Goals against on penalty shots",
    "diff_nhl_tiesLeadPeriod2": "Tied/leading after 2nd period frequency",
    "diff_nhl_matchPenalties": "Match penalties (major discipline issues)",
    "diff_mp_5v5_lowDangerGoalsAgainst": "Low-danger goals against at 5v5",
    "diff_mp_5v5_reboundGoalsAgainst": "Rebound goals against at 5v5",
    "diff_nhl_winsOutshotByOpponent": "Wins when outshot (resilience/goaltending quality)",
    "diff_nhl_winsOutshootOpponent": "Wins when outshooting opponent",
    "diff_nhl_lossesScoringFirst": "Losses despite scoring first (blown leads)",
    "diff_nhl_goalsAgainst5On5": "5-on-5 goals against (even strength defense)",
    "diff_nhl_savePct5v5": "5-on-5 save percentage",
    "diff_nhl_offensiveZoneFaceoffPct": "Offensive zone faceoff %",
    "diff_elo_rating": "Playoff ELO rating (accumulated playoff success)",
    "elo_expected_win": "ELO-based expected win probability",
    "diff_stnd_pythag_pct": "Pythagorean win expectation (goals-based expected win%)",
    "diff_stnd_point_pct": "Regular season point percentage",
    "diff_stnd_goal_diff_pg": "Goal differential per game",
    "diff_playoff_appearances_5yr": "Playoff appearances in last 5 years",
    "diff_playoff_series_wins_5yr": "Playoff series wins in last 5 years",
}


def main():
    # Load the refined feature list
    features_path = MODELS_DIR / "refined_features.joblib"
    if not features_path.exists():
        features_path = MODELS_DIR / "selected_features.joblib"
    selected = joblib.load(features_path)

    # Load feature matrix and train a GB model to get importances
    matrix_path = PROCESSED_DIR / "feature_matrix_refined.csv"
    if not matrix_path.exists():
        matrix_path = PROCESSED_DIR / "feature_matrix_improved.csv"
    if not matrix_path.exists():
        matrix_path = PROCESSED_DIR / "feature_matrix.csv"

    df = pd.read_csv(matrix_path)

    META = ["season", "round", "higher_seed", "lower_seed",
            "higher_seed_won", "is_bubble_year", "is_shortened_season"]
    feature_cols = [c for c in df.columns if c not in META and c in selected]
    X = df[feature_cols].copy()
    y = df["higher_seed_won"].copy()

    imputer = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    gb.fit(X_filled, y)

    importances = pd.Series(gb.feature_importances_, index=X_filled.columns)
    importances = importances.sort_values(ascending=False)

    # Also compute correlation with target
    correlations = X_filled.corrwith(y).abs()

    print("=" * 90)
    print("TOP FEATURES FOR PREDICTING NHL PLAYOFF SERIES WINNERS")
    print("=" * 90)
    print()
    print(f"{'Rank':<5s} {'Feature':<55s} {'Importance':>10s} {'Corr':>6s}")
    print("-" * 90)

    for i, (feat, imp) in enumerate(importances.head(35).items()):
        corr = correlations.get(feat, 0)
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        print(f" {i+1:2d}.  {feat:<55s} {imp:10.5f} {corr:6.3f}")
        if desc:
            print(f"       >> {desc}")

    # Group by category
    print()
    print("=" * 90)
    print("FEATURE IMPORTANCE BY CATEGORY")
    print("=" * 90)

    categories = {
        "Goal Prevention / Defense": [f for f in importances.index if any(k in f.lower() for k in ["against", "savepct", "gaa", "pk", "penalty"])],
        "Offense / Scoring": [f for f in importances.index if any(k in f.lower() for k in ["goalsfor", "period1goals", "scoring", "shots", "xgoals"])],
        "Situational Play": [f for f in importances.index if any(k in f.lower() for k in ["lead", "trail", "ahead", "behind", "scoring"])],
        "Advanced Analytics (MoneyPuck)": [f for f in importances.index if f.startswith("diff_mp_")],
        "Goaltending": [f for f in importances.index if "goalie" in f.lower()],
        "Experience / Intangibles": [f for f in importances.index if any(k in f.lower() for k in ["elo", "playoff", "pythag", "bubble", "shortened"])],
    }

    for cat_name, cat_features in categories.items():
        total_imp = sum(importances.get(f, 0) for f in cat_features)
        pct = total_imp / importances.sum() * 100 if importances.sum() > 0 else 0
        print(f"\n  {cat_name}: {pct:.1f}% of total importance ({len(cat_features)} features)")


if __name__ == "__main__":
    main()
