"""
Deep analysis of NHL playoff scoring prediction model.
Analyzes training data, 2026 predictions, and top 35 realism.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_training_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "scoring_training_data.csv"
    df = pd.read_csv(path)
    return df


def load_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "scoring_predictions_2026.csv"
    df = pd.read_csv(path)
    return df


def load_top35() -> pd.DataFrame:
    path = PROCESSED_DIR / "top35_scorers_2026.csv"
    df = pd.read_csv(path)
    return df


def analyze_training_data(df: pd.DataFrame) -> dict:
    """Analyze training data: team_expected_games vs target_gp, residuals, biases."""
    results = {}

    # 1. team_expected_games vs target_gp
    valid = df[["team_expected_games", "target_gp"]].dropna()
    if len(valid) > 0:
        corr_pearson, _ = pearsonr(valid["team_expected_games"], valid["target_gp"])
        corr_spearman, _ = spearmanr(valid["team_expected_games"], valid["target_gp"])
        mad = np.abs(valid["team_expected_games"] - valid["target_gp"]).mean()
        results["team_exp_vs_gp"] = {
            "correlation_pearson": corr_pearson,
            "correlation_spearman": corr_spearman,
            "mean_absolute_difference": mad,
            "n": len(valid),
        }

    # 2. Prediction residuals by position (using pts_x_expected_games as baseline)
    df = df.copy()
    df["baseline_pred"] = df["pts_x_expected_games"]
    df["residual"] = df["target_points"] - df["baseline_pred"]

    pos_stats = {}
    for pos in ["C", "L", "R", "D"]:
        subset = df[df["positionCode"] == pos].dropna(subset=["residual"])
        if len(subset) > 0:
            pos_stats[pos] = {
                "n": len(subset),
                "mean_residual": subset["residual"].mean(),
                "median_residual": subset["residual"].median(),
                "std_residual": subset["residual"].std(),
                "mae": np.abs(subset["residual"]).mean(),
            }
    results["residuals_by_position"] = pos_stats

    # 3. Stars vs depth accuracy
    stars = df[df["reg_ppg"] > 0.8].dropna(subset=["residual"])
    depth = df[df["reg_ppg"] < 0.4].dropna(subset=["residual"])
    results["stars_vs_depth"] = {
        "stars": {
            "n": len(stars),
            "mean_residual": stars["residual"].mean() if len(stars) > 0 else np.nan,
            "mae": np.abs(stars["residual"]).mean() if len(stars) > 0 else np.nan,
        },
        "depth": {
            "n": len(depth),
            "mean_residual": depth["residual"].mean() if len(depth) > 0 else np.nan,
            "mae": np.abs(depth["residual"]).mean() if len(depth) > 0 else np.nan,
        },
    }

    # 4. Systematic biases by team_expected_games bucket
    df["exp_games_bucket"] = pd.cut(
        df["team_expected_games"],
        bins=[0, 4, 8, 12, 16, 25],
        labels=["4 or fewer", "5-8", "9-12", "13-16", "17+"],
    )
    bucket_stats = {}
    for bucket in ["4 or fewer", "5-8", "9-12", "13-16", "17+"]:
        subset = df[df["exp_games_bucket"] == bucket].dropna(subset=["residual"])
        if len(subset) > 0:
            bucket_stats[bucket] = {
                "n": len(subset),
                "mean_residual": subset["residual"].mean(),
                "median_residual": subset["residual"].median(),
                "bias": "over" if subset["residual"].mean() < 0 else "under",
            }
    results["bias_by_expected_games"] = bucket_stats

    return results


def analyze_predictions_2026(pred_df: pd.DataFrame, top35_df: pd.DataFrame) -> dict:
    """Check flags, suspicious predictions, missing players."""
    results = {}

    top35_names = set(top35_df["playerName"].tolist())

    # 1. NOT ON ROSTER or INACTIVE in top 35
    flagged_in_top35 = top35_df[
        (top35_df["on_roster"] == False)
        | (top35_df["injury_flag"].astype(str).str.contains("NOT ON ROSTER|INACTIVE", na=False))
    ]
    results["flagged_in_top35"] = [
        {
            "player": row["playerName"],
            "rank": i + 1,
            "pred_points": row["pred_points"],
            "on_roster": row["on_roster"],
            "injury_flag": str(row.get("injury_flag", "")),
        }
        for i, (_, row) in enumerate(flagged_in_top35.iterrows())
    ]

    # 2. Suspicious: low reg stats but high predictions (in top 35)
    top35 = top35_df.copy()
    top35["reg_ppg"] = top35["reg_points"] / top35["reg_gp"].replace(0, np.nan)
    suspicious = top35[
        (top35["reg_ppg"] < 0.5) & (top35["pred_points"] > 5)
    ]
    results["suspicious_low_stats_high_pred"] = [
        {
            "player": row["playerName"],
            "rank": i + 1,
            "pred_points": row["pred_points"],
            "reg_ppg": row["reg_ppg"],
            "reg_points": row["reg_points"],
            "reg_gp": row["reg_gp"],
        }
        for i, (_, row) in enumerate(suspicious.iterrows())
    ]

    # 3. Major players missing from top 35 (check full predictions)
    # High reg_ppg players (stars) that should be in top 35
    high_ppg = pred_df[pred_df["reg_points"] / pred_df["reg_gp"].replace(0, np.nan) > 1.0]
    high_ppg = high_ppg.sort_values("pred_points", ascending=False)
    missing_stars = []
    for _, row in high_ppg.iterrows():
        if row["playerName"] not in top35_names and row["pred_points"] > 4:
            missing_stars.append({
                "player": row["playerName"],
                "pred_points": row["pred_points"],
                "reg_ppg": row["reg_points"] / row["reg_gp"] if row["reg_gp"] else 0,
                "team": row["team_abbrev"],
            })
    results["potentially_missing_stars"] = missing_stars[:15]

    return results


def verify_top35_realism(top35_df: pd.DataFrame) -> dict:
    """Verify realism of top 35 predictions."""
    results = {}

    # Sum of predicted points
    total_pred = top35_df["pred_points"].sum()
    results["total_predicted_points"] = total_pred

    # Team distribution
    team_counts = top35_df["team_abbrev"].value_counts()
    results["team_distribution"] = team_counts.to_dict()

    # Position distribution
    pos_counts = top35_df["positionCode"].value_counts()
    results["position_distribution"] = pos_counts.to_dict()

    # Predicted points range
    results["pred_points_range"] = (top35_df["pred_points"].min(), top35_df["pred_points"].max())

    # Reg PPG vs pred PPG (implied)
    top35 = top35_df.copy()
    top35["reg_ppg"] = top35["reg_points"] / top35["reg_gp"].replace(0, np.nan)
    top35["implied_po_ppg"] = top35["pred_points"] / top35["team_expected_games"].replace(0, np.nan)
    results["reg_ppg_vs_implied_po"] = top35[["playerName", "reg_ppg", "implied_po_ppg", "pred_points"]].head(15).to_dict("records")

    return results


def main():
    print("=" * 80)
    print("NHL PLAYOFF SCORING MODEL - DEEP ANALYSIS")
    print("=" * 80)

    # Load data
    train_df = load_training_data()
    pred_df = load_predictions()
    top35_df = load_top35()

    print(f"\nTraining data: {len(train_df)} rows")
    print(f"Predictions 2026: {len(pred_df)} rows")
    print(f"Top 35: {len(top35_df)} rows")

    # 1. Training data analysis
    print("\n" + "=" * 80)
    print("1. TRAINING DATA ANALYSIS")
    print("=" * 80)

    train_results = analyze_training_data(train_df)

    print("\n1.1 team_expected_games vs actual target_gp:")
    r = train_results["team_exp_vs_gp"]
    print(f"   Pearson correlation:  {r['correlation_pearson']:.4f}")
    print(f"   Spearman correlation:  {r['correlation_spearman']:.4f}")
    print(f"   Mean absolute diff:    {r['mean_absolute_difference']:.2f} games")
    print(f"   N:                    {r['n']}")

    print("\n1.2 Prediction residuals by position (baseline: pts_x_expected_games):")
    for pos, stats in train_results["residuals_by_position"].items():
        print(f"   {pos}: n={stats['n']}, mean_residual={stats['mean_residual']:.3f}, "
              f"median={stats['median_residual']:.3f}, MAE={stats['mae']:.3f}")

    print("\n1.3 Stars (reg_ppg>0.8) vs Depth (reg_ppg<0.4):")
    s = train_results["stars_vs_depth"]["stars"]
    d = train_results["stars_vs_depth"]["depth"]
    print(f"   Stars: n={s['n']}, mean_residual={s['mean_residual']:.3f}, MAE={s['mae']:.3f}")
    print(f"   Depth: n={d['n']}, mean_residual={d['mean_residual']:.3f}, MAE={d['mae']:.3f}")

    print("\n1.4 Systematic bias by team_expected_games bucket:")
    for bucket, stats in train_results["bias_by_expected_games"].items():
        print(f"   {bucket}: n={stats['n']}, mean_residual={stats['mean_residual']:.3f} ({stats['bias']} prediction)")

    # 2. Predictions 2026 checks
    print("\n" + "=" * 80)
    print("2. PREDICTIONS 2026 - FLAGS & ISSUES")
    print("=" * 80)

    pred_results = analyze_predictions_2026(pred_df, top35_df)

    print("\n2.1 Players flagged NOT ON ROSTER or INACTIVE in top 35:")
    if pred_results["flagged_in_top35"]:
        for p in pred_results["flagged_in_top35"]:
            print(f"   #{p['rank']} {p['player']}: pred={p['pred_points']:.1f}, "
                  f"on_roster={p['on_roster']}, flag={p['injury_flag']}")
    else:
        print("   None")

    print("\n2.2 Suspicious: low reg stats but high predictions (in top 35):")
    if pred_results["suspicious_low_stats_high_pred"]:
        for p in pred_results["suspicious_low_stats_high_pred"]:
            print(f"   #{p['rank']} {p['player']}: pred={p['pred_points']:.1f}, "
                  f"reg_ppg={p['reg_ppg']:.3f}, reg_pts={p['reg_points']}, reg_gp={p['reg_gp']}")
    else:
        print("   None")

    print("\n2.3 Major players potentially missing from top 35:")
    if pred_results["potentially_missing_stars"]:
        for p in pred_results["potentially_missing_stars"][:10]:
            print(f"   {p['player']} ({p['team']}): pred={p['pred_points']:.1f}, reg_ppg={p['reg_ppg']:.3f}")
    else:
        print("   None identified")

    # 3. Top 35 realism
    print("\n" + "=" * 80)
    print("3. TOP 35 REALISM VERIFICATION")
    print("=" * 80)

    realism = verify_top35_realism(top35_df)
    print(f"\nTotal predicted points (top 35): {realism['total_predicted_points']:.1f}")
    print(f"Pred points range: {realism['pred_points_range'][0]:.1f} - {realism['pred_points_range'][1]:.1f}")
    print("\nTeam distribution:")
    for team, count in sorted(realism["team_distribution"].items(), key=lambda x: -x[1])[:10]:
        print(f"   {team}: {count}")
    print("\nPosition distribution:")
    for pos, count in realism["position_distribution"].items():
        print(f"   {pos}: {count}")

    # Save full report
    report = {
        "training": train_results,
        "predictions_2026": pred_results,
        "realism": {k: v for k, v in realism.items() if k != "reg_ppg_vs_implied_po"},
    }
    out_path = PROCESSED_DIR / "scoring_analysis_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {out_path}")

    return report


if __name__ == "__main__":
    main()
