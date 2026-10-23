"""
Playoff Top Scorer Prediction Pipeline.

Predicts playoff point totals for every player on a playoff-contending team,
then ranks and displays the top 40.

Includes:
- Bubble/contender teams beyond the current top 16
- Playoff probability weighting for expected games
- Live roster verification
- Injury/inactivity detection
- Data freshness validation

Built on the foundation of the team prediction model (V2).

Usage:
    python scripts/run_scoring_pipeline.py
    python scripts/run_scoring_pipeline.py --predict-only
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    CURRENT_SEASON_END_YEAR,
    FIRST_SEASON,
    LAST_HISTORICAL_SEASON,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RANDOM_STATE,
    season_id,
)
from src.data.nhl_api_client import NHLAPIClient
from src.data.collectors_playoff_scoring import (
    collect_player_bios,
    collect_playoff_skater_stats,
    collect_regular_season_skater_stats,
)
from src.analysis.team_player_consistency import write_team_player_consistency_report
from src.data.player_health import PLAYOFF_HEALTH_POLICY_VERSION
from src.data.roster_injury import (
    apply_injury_context_overrides,
    build_espn_injury_lookup,
    compute_playoff_probabilities,
    enrich_pred_df_with_espn_injuries,
    espn_injury_discount_multiplier,
    fetch_espn_nhl_injuries,
    fetch_live_standings,
    fetch_team_roster,
    fill_espn_injury_flags_all_rows,
    identify_contender_teams,
    verify_and_flag_players,
)
from src.features.bracket_adjustments import apply_scoring_bracket_adjustments
from src.features.player_scoring import (
    HIST_TOP40_ROUND_MIX_PCT,
    build_prediction_features,
    build_training_dataset,
    round_mix_bucket_from_probs,
)
from src.models.scoring_predictor import (
    FEATURE_COLS,
    get_feature_importance,
    predict_blended,
    predict_with_intervals,
    prepare_xy,
    train_scoring_models,
)
from src.models.monte_carlo import simulate_series
from src.utils.logging_config import get_logger
from src.utils.model_manifest import (
    build_scoring_model_manifest,
    build_v2_model_manifest,
    write_run_manifest,
)

logger = get_logger("scoring_pipeline")

SEASON = CURRENT_SEASON_END_YEAR
TOP_PLAYOFF_SCORERS = 40


def _dataframe_to_csv_atomic(df: pd.DataFrame, path: Path) -> Path:
    """
    Write CSV via a same-directory temp file, then replace (avoids partial reads).
    If the target is locked (e.g. open in Excel on Windows), writes alongside as
    ``<name>_export.csv`` and returns the path actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
        return path
    except OSError:
        fallback = path.with_name(path.stem + "_export" + path.suffix)
        try:
            os.replace(tmp, fallback)
        except OSError:
            df.to_csv(fallback, index=False)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        logger.warning(
            "Could not replace %s (file in use?). Wrote %s instead.",
            path,
            fallback,
        )
        print(
            f"\n  WARNING: {path.name} is locked; predictions written to "
            f"{fallback.name} — close the original and rename, or delete it and re-run."
        )
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def step_collect():
    """Collect all data needed for the scoring model."""
    logger.info("=" * 60)
    logger.info("SCORING DATA COLLECTION")
    logger.info("=" * 60)

    client = NHLAPIClient()

    # Playoff skater stats (historical targets)
    po_path = RAW_DIR / "skater_stats_playoffs.csv"
    if not po_path.exists():
        logger.info("Collecting historical playoff skater stats...")
        collect_playoff_skater_stats(client)
    else:
        logger.info("Playoff skater stats already cached (%d KB)", po_path.stat().st_size // 1024)

    # Regular season skater stats (historical features)
    reg_path = RAW_DIR / "skater_stats.csv"
    if not reg_path.exists():
        logger.info("Collecting historical regular-season skater stats...")
        from src.data.collectors_v2 import collect_skater_stats
        collect_skater_stats(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)
    else:
        logger.info("Regular skater stats already cached")

    # Player bios (birthDate, draft info for age features)
    bios_path = RAW_DIR / "player_bios.csv"
    if not bios_path.exists():
        logger.info("Collecting player biographical data...")
        collect_player_bios(client, FIRST_SEASON, LAST_HISTORICAL_SEASON)
    else:
        logger.info("Player bios already cached (%d KB)", bios_path.stat().st_size // 1024)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE TEAM EXPECTED GAMES (from team model)
# ═══════════════════════════════════════════════════════════════════════════

def step_compute_team_expected_games() -> tuple[dict[str, float], dict[str, dict]]:
    """
    Compute expected playoff games for all teams that could make the playoffs.

    1. Fetches live standings
    2. Computes playoff probability for every team
    3. Identifies bubble/contender teams beyond the top 16
    4. Runs the team model bracket simulation for the current top 16
    5. Assigns probability-weighted expected games to bubble teams

    Returns (team_expected_games, team_probabilities)
    """
    logger.info("=" * 60)
    logger.info("COMPUTING TEAM EXPECTED GAMES (with bubble teams)")
    logger.info("=" * 60)

    # ── 1. Live standings + playoff probabilities ──
    fresh_client = NHLAPIClient(cache_ttl_hours=0.01)
    live_standings = fetch_live_standings(fresh_client)
    team_probs = compute_playoff_probabilities(live_standings)
    playoff_teams, bubble_teams = identify_contender_teams(team_probs)

    print(f"\n{'='*70}")
    print("  CURRENT STANDINGS & PLAYOFF PROBABILITIES")
    print(f"{'='*70}")
    for conf in ("Eastern", "Western"):
        print(f"\n  {conf} Conference:")
        conf_teams = {t: v for t, v in team_probs.items() if v["conference"] == conf}
        for team in sorted(conf_teams, key=lambda t: conf_teams[t]["conf_rank"]):
            info = conf_teams[team]
            status = "IN " if info["in_playoff_position"] else "OUT"
            bubble_tag = " [BUBBLE]" if team in bubble_teams else ""
            print(f"    {info['conf_rank']:2d}. {team:4s}  {status}  "
                  f"PTS={info['points']:3d}  GP={info['gp']:2d}  "
                  f"Proj={info['projected_pts']:5.1f}  "
                  f"P(playoff)={info['playoff_probability']*100:5.1f}%{bubble_tag}")

    all_contenders = playoff_teams + bubble_teams
    logger.info("Playoff teams (%d): %s", len(playoff_teams), ", ".join(playoff_teams))
    logger.info("Bubble teams (%d): %s", len(bubble_teams), ", ".join(bubble_teams))

    # ── 2. Load team models for bracket simulation ──
    try:
        models = joblib.load(MODELS_DIR / "v2_models.joblib")
        selected = joblib.load(MODELS_DIR / "v2_features.joblib")
        blend_weights = joblib.load(MODELS_DIR / "v2_blend_weights.joblib")
    except FileNotFoundError:
        logger.error("Team V2 models not found. Run run_v2_pipeline.py first.")
        return {}, team_probs

    from src.data.collectors import (
        collect_standings, collect_team_stats_nhl,
        collect_moneypuck_team_stats, collect_goalie_stats,
    )
    from src.features.engineer import (
        _build_team_season_features, _compute_differential, _get_team_features,
    )

    client = NHLAPIClient(cache_ttl_hours=0)
    standings = collect_standings(client, SEASON, SEASON)
    nhl_stats = collect_team_stats_nhl(client, SEASON, SEASON)
    moneypuck = collect_moneypuck_team_stats(SEASON, SEASON)
    goalies = collect_goalie_stats(client, SEASON, SEASON)
    team_features = _build_team_season_features(standings, nhl_stats, moneypuck, goalies)

    if team_features.empty:
        logger.error("Could not build team features")
        return {}, team_probs

    # Build bracket from current top 16
    stnd = standings[standings["season"] == SEASON].copy()

    def _extract_abbrev(x):
        if isinstance(x, dict):
            return x.get("default", "")
        if isinstance(x, str) and "{" in x:
            try:
                return eval(x).get("default", "")
            except Exception:
                return x
        return str(x)

    stnd["team"] = stnd["teamAbbrev"].apply(_extract_abbrev)

    def _build_conference_bracket(conf_df):
        divs = sorted(conf_df["divisionAbbrev"].unique())
        div_teams = {}
        div_winners = {}
        for div in divs:
            div_df = conf_df[conf_df["divisionAbbrev"] == div].sort_values("divisionSequence")
            teams = div_df["team"].tolist()
            div_teams[div] = teams[:3]
            div_winners[div] = teams[0] if teams else ""
        all_div_qualified = set()
        for teams in div_teams.values():
            all_div_qualified.update(teams)
        wild_cards = conf_df[
            ~conf_df["team"].isin(all_div_qualified)
        ].sort_values("conferenceSequence")["team"].head(2).tolist()
        div_winner_ranks = {}
        for div, winner in div_winners.items():
            row = conf_df[conf_df["team"] == winner]
            rank = row["conferenceSequence"].values[0] if not row.empty else 99
            div_winner_ranks[div] = rank
        sorted_divs = sorted(divs, key=lambda d: div_winner_ranks.get(d, 99))
        better_div = sorted_divs[0]
        worse_div = sorted_divs[1]
        wc1 = wild_cards[0] if len(wild_cards) > 0 else ""
        wc2 = wild_cards[1] if len(wild_cards) > 1 else ""
        matchups = [
            (div_winners[better_div], wc2),
            (div_teams[better_div][1], div_teams[better_div][2]),
            (div_winners[worse_div], wc1),
            (div_teams[worse_div][1], div_teams[worse_div][2]),
        ]
        return matchups

    east_df = stnd[stnd["conferenceAbbrev"] == "E"].copy()
    west_df = stnd[stnd["conferenceAbbrev"] == "W"].copy()
    east_matchups = _build_conference_bracket(east_df)
    west_matchups = _build_conference_bracket(west_df)

    def predict_series(hi, lo):
        hi_feats = _get_team_features(team_features, hi, SEASON)
        lo_feats = _get_team_features(team_features, lo, SEASON)
        if hi_feats is None or lo_feats is None:
            return 0.5
        diff = _compute_differential(hi_feats, lo_feats, prefix="diff_")
        diff["is_bubble_year"] = 0
        diff["is_shortened_season"] = 0
        diff.setdefault("diff_elo_rating", 0)
        diff.setdefault("elo_expected_win", 0.5)
        X_pred = pd.DataFrame([diff])
        for col in selected:
            if col not in X_pred.columns:
                X_pred[col] = 0.0
        X_pred = X_pred[selected]
        blended = 0.0
        total_w = 0.0
        for name, model in models.items():
            w = blend_weights.get(name, 0)
            if w <= 0:
                continue
            try:
                p = model.predict_proba(X_pred)[0][1]
                blended += w * p
                total_w += w
            except Exception:
                continue
        return blended / total_w if total_w > 0 else 0.5

    # ── 3. Probability-weighted bracket simulation ──
    # Each team's expected games = sum across rounds of:
    #   E[games_in_round] * P(team reaches that round)
    # This properly accounts for advancement probability instead of
    # deterministically advancing the predicted winner.

    team_games: dict[str, float] = {}
    team_reach_prob: dict[str, float] = {t: 1.0 for matchups in
        [east_matchups, west_matchups] for hi, lo in matchups for t in (hi, lo)}
    N_SIM = 50000

    def _prob_weighted_round(matchups, team_reach):
        """Simulate round and return probability-weighted results."""
        results = []
        for hi, lo in matchups:
            bp = predict_series(hi, lo)
            mc = simulate_series(bp, N_SIM)
            exp_g = mc["expected_games"]
            hi_win_p = mc["series_win_prob"]

            hi_reach = team_reach.get(hi, 1.0)
            lo_reach = team_reach.get(lo, 1.0)
            team_games[hi] = team_games.get(hi, 0) + exp_g * hi_reach
            team_games[lo] = team_games.get(lo, 0) + exp_g * lo_reach

            results.append({
                "hi": hi, "lo": lo,
                "hi_win_p": hi_win_p,
                "hi_reach": hi_reach, "lo_reach": lo_reach,
                "winner": hi if hi_win_p >= 0.5 else lo,
            })
        return results

    east_r1_res = _prob_weighted_round(east_matchups, team_reach_prob)
    west_r1_res = _prob_weighted_round(west_matchups, team_reach_prob)

    def _update_reach_and_build_r2(r1_results):
        """Build R2 matchups with probability of reaching R2."""
        reach_r2 = {}
        for r in r1_results:
            reach_r2[r["hi"]] = r["hi_reach"] * r["hi_win_p"]
            reach_r2[r["lo"]] = r["lo_reach"] * (1 - r["hi_win_p"])
        w0 = r1_results[0]["winner"]
        w1 = r1_results[1]["winner"]
        return [(w0, w1)], reach_r2

    east_r2_matchups, east_r2_reach = _update_reach_and_build_r2(east_r1_res[:2])
    east_r2b_matchups, east_r2b_reach = _update_reach_and_build_r2(east_r1_res[2:])
    east_all_r2_reach = {**east_r2_reach, **east_r2b_reach}
    east_r2_matchups = east_r2_matchups + east_r2b_matchups

    west_r2_matchups, west_r2_reach = _update_reach_and_build_r2(west_r1_res[:2])
    west_r2b_matchups, west_r2b_reach = _update_reach_and_build_r2(west_r1_res[2:])
    west_all_r2_reach = {**west_r2_reach, **west_r2b_reach}
    west_r2_matchups = west_r2_matchups + west_r2b_matchups

    east_r2_res = _prob_weighted_round(east_r2_matchups, east_all_r2_reach)
    west_r2_res = _prob_weighted_round(west_r2_matchups, west_all_r2_reach)

    east_cf_reach = {}
    for r in east_r2_res:
        east_cf_reach[r["hi"]] = east_all_r2_reach.get(r["hi"], 0) * r["hi_win_p"]
        east_cf_reach[r["lo"]] = east_all_r2_reach.get(r["lo"], 0) * (1 - r["hi_win_p"])
    ecf_matchups = [(east_r2_res[0]["winner"], east_r2_res[1]["winner"])]
    ecf_res = _prob_weighted_round(ecf_matchups, east_cf_reach)

    west_cf_reach = {}
    for r in west_r2_res:
        west_cf_reach[r["hi"]] = west_all_r2_reach.get(r["hi"], 0) * r["hi_win_p"]
        west_cf_reach[r["lo"]] = west_all_r2_reach.get(r["lo"], 0) * (1 - r["hi_win_p"])
    wcf_matchups = [(west_r2_res[0]["winner"], west_r2_res[1]["winner"])]
    wcf_res = _prob_weighted_round(wcf_matchups, west_cf_reach)

    scf_reach = {}
    for r in ecf_res:
        scf_reach[r["hi"]] = east_cf_reach.get(r["hi"], 0) * r["hi_win_p"]
        scf_reach[r["lo"]] = east_cf_reach.get(r["lo"], 0) * (1 - r["hi_win_p"])
    for r in wcf_res:
        scf_reach[r["hi"]] = west_cf_reach.get(r["hi"], 0) * r["hi_win_p"]
        scf_reach[r["lo"]] = west_cf_reach.get(r["lo"], 0) * (1 - r["hi_win_p"])
    scf_matchups = [(ecf_res[0]["winner"], wcf_res[0]["winner"])]
    # P(reach Cup Final) before SCF — same state used for team_round_reach_probs
    p_reach_final_pre_scf = dict(scf_reach)
    _prob_weighted_round(scf_matchups, scf_reach)

    p_reach_r2 = {**east_all_r2_reach, **west_all_r2_reach}
    p_reach_cf = {**east_cf_reach, **west_cf_reach}

    def _bubble_round_probs(pp: float) -> dict[str, float]:
        """Rough survival probs if a bubble team sneaks into the bracket."""
        p = max(0.0, min(1.0, pp))
        return {
            "p_r2": min(1.0, p * 0.22),
            "p_cf": min(1.0, p * 0.04),
            "p_final": min(1.0, p * 0.01),
        }

    team_round_reach_probs: dict[str, dict[str, float]] = {}
    for t, pr2 in p_reach_r2.items():
        team_round_reach_probs[t] = {
            "p_r2": float(pr2),
            "p_cf": float(p_reach_cf.get(t, 0.0)),
            "p_final": float(p_reach_final_pre_scf.get(t, 0.0)),
        }
    for t in bubble_teams:
        bp = _bubble_round_probs(team_probs[t]["playoff_probability"])
        team_round_reach_probs[t] = bp

    # ── 4. Assign expected games to bubble teams ──
    # Bubble teams get: base first-round expected games * playoff probability
    # Average first-round series length is about 5.5 games
    BASE_FIRST_ROUND_GAMES = 5.5
    for team in bubble_teams:
        prob = team_probs[team]["playoff_probability"]
        team_games[team] = BASE_FIRST_ROUND_GAMES * prob

    # Mild shrinkage toward a points-based prior (reduces bracket-sim noise propagation)
    _prior_w = 0.08
    for t in list(team_games.keys()):
        info = team_probs.get(t) or {}
        pts = float(info.get("points", 95))
        eg_prior = min(26.0, 3.2 + (pts / 115.0) * 9.0)
        team_games[t] = (1.0 - _prior_w) * float(team_games[t]) + _prior_w * eg_prior

    print(f"\n{'='*70}")
    print("  TEAM EXPECTED PLAYOFF GAMES")
    print(f"{'='*70}")
    for team in sorted(team_games, key=team_games.get, reverse=True):
        prob_str = f"  P(playoff)={team_probs[team]['playoff_probability']*100:.0f}%"
        bubble = " [BUBBLE]" if team in bubble_teams else ""
        print(f"  {team:>4s}: {team_games[team]:5.1f} expected games{prob_str}{bubble}")

    # ── Save results ──
    result = {
        "team_expected_games": team_games,
        "team_round_reach_probs": team_round_reach_probs,
        "playoff_teams": playoff_teams,
        "bubble_teams": bubble_teams,
        "all_contenders": all_contenders,
        "team_probabilities": team_probs,
        "timestamp": datetime.datetime.now().isoformat(),
        "health_policy_version": PLAYOFF_HEALTH_POLICY_VERSION,
        "model_manifest": build_v2_model_manifest(),
        "scoring_models_manifest": build_scoring_model_manifest(),
    }
    out_path = PROCESSED_DIR / "team_expected_games_2026.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Saved expected games for %d teams (%d playoff + %d bubble)",
                len(team_games), len(playoff_teams), len(bubble_teams))
    return team_games, team_probs


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: BUILD FEATURES & TRAIN
# ═══════════════════════════════════════════════════════════════════════════

def step_build_and_train() -> dict:
    """Build training data and train scoring models."""
    logger.info("=" * 60)
    logger.info("BUILDING TRAINING DATA & TRAINING MODELS")
    logger.info("=" * 60)

    # Load data — use historical regular season file for training
    reg_path = RAW_DIR / "skater_stats_regular_historical.csv"
    if not reg_path.exists():
        logger.info("Historical regular season stats missing, collecting now...")
        client = NHLAPIClient()
        collect_regular_season_skater_stats(client, FIRST_SEASON, LAST_HISTORICAL_SEASON,
                                            "skater_stats_regular_historical.csv")
    reg_df = pd.read_csv(reg_path)
    po_df = pd.read_csv(RAW_DIR / "skater_stats_playoffs.csv")
    series_df = pd.read_csv(RAW_DIR / "playoff_series.csv")

    # Load player bios for age features
    bios_path = RAW_DIR / "player_bios.csv"
    bios_df = pd.read_csv(bios_path) if bios_path.exists() else None
    if bios_df is not None:
        logger.info("Loaded player bios: %d players", len(bios_df))

    logger.info("Regular season: %d rows, Playoff: %d rows, Series: %d rows",
                len(reg_df), len(po_df), len(series_df))

    # Build training dataset with new feature sources
    train_df = build_training_dataset(
        reg_df, po_df, series_df,
        player_bios_df=bios_df,
    )

    if train_df.empty:
        logger.error("Training dataset is empty")
        return {}

    train_df.to_csv(PROCESSED_DIR / "scoring_training_data.csv", index=False)
    logger.info("Training data: %d rows, %d columns", len(train_df), len(train_df.columns))

    # Train models for each target
    results = {}

    for target in ("target_points", "target_goals", "target_assists"):
        print(f"\n{'='*70}")
        print(f"  TRAINING: {target}")
        print(f"{'='*70}")

        result = train_scoring_models(train_df, target=target)
        results[target] = result

        # Print evaluation
        for name, metrics in result["eval_results"].items():
            print(f"\n  {name}:")
            print(f"    MAE:            {metrics['mae']:.3f}")
            print(f"    RMSE:           {metrics['rmse']:.3f}")
            print(f"    Spearman corr:  {metrics['spearman']:.3f}")
            print(f"    Top-10 overlap: {metrics['mean_top10_overlap']*100:.1f}%")
            print(f"    Top-35 overlap: {metrics['mean_top35_overlap']*100:.1f}%")
            print(f"    Top-40 overlap: {metrics['mean_top40_overlap']*100:.1f}%")

        # Feature importance
        fi = get_feature_importance(result["models"], result["feature_cols"])
        print(f"\n  Top 10 features for {target}:")
        for _, row in fi.head(10).iterrows():
            print(f"    {row['feature']:35s}  {row.get('avg_importance', 0):.4f}")

    # Save models
    joblib.dump(results, MODELS_DIR / "scoring_models.joblib")
    logger.info("Saved scoring models to %s", MODELS_DIR / "scoring_models.joblib")

    # Persist temporal CV metrics for each target (test-set style walk-forward)
    eval_payload = {}
    for target, res in results.items():
        er = res.get("eval_results") or {}
        eval_payload[target] = {}
        for model_name, metrics in er.items():
            row = {}
            for k, v in metrics.items():
                if k == "season_results" and hasattr(v, "to_dict"):
                    row[k] = v.to_dict(orient="records")
                elif isinstance(v, (float, int, np.floating, np.integer)):
                    row[k] = float(v)
                else:
                    row[k] = v
            eval_payload[target][model_name] = row
    eval_path = PROCESSED_DIR / "scoring_temporal_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.datetime.now().isoformat(),
                "health_policy_version": PLAYOFF_HEALTH_POLICY_VERSION,
                "eval_by_target": eval_payload,
            },
            f,
            indent=2,
        )
    logger.info("Wrote temporal eval metrics to %s", eval_path)

    arch_dir = PROCESSED_DIR / "eval_archives"
    arch_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    arch_copy = arch_dir / f"scoring_temporal_eval_{ts}.json"
    shutil.copy2(eval_path, arch_copy)
    logger.info("Archived copy for regression tracking: %s", arch_copy)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: PREDICT 2026 TOP SCORERS
# ═══════════════════════════════════════════════════════════════════════════

def step_predict(
    model_results: dict,
    team_expected_games: dict[str, float],
    team_probs: dict[str, dict] | None = None,
):
    """
    Predict 2026 playoff scoring for all players on playoff + bubble teams.
    Includes roster verification, injury detection, and data freshness checks.
    """
    logger.info("=" * 60)
    logger.info("PREDICTING 2026 PLAYOFF TOP SCORERS")
    logger.info("=" * 60)

    # ── Data freshness: fetch LIVE current-season stats ──
    fetch_time = datetime.datetime.now()
    client = NHLAPIClient(cache_ttl_hours=0.01)
    sid = season_id(SEASON)

    logger.info("Fetching LIVE current-season player stats (timestamp: %s)", fetch_time.isoformat())
    raw = client._stats(
        "/skater/summary",
        {
            "isAggregate": "false",
            "isGame": "false",
            "limit": -1,
            "start": 0,
            "cayenneExp": f"gameTypeId=2 and seasonId<={sid} and seasonId>={sid}",
        },
    )
    rows = raw.get("data", [])
    if not rows:
        logger.error("No current season skater data returned")
        return

    current_df = pd.DataFrame(rows)
    current_df["season"] = SEASON
    logger.info("Fetched %d players for %s season", len(current_df), sid)

    # ── Load prior playoff data for historical features ──
    po_path = RAW_DIR / "skater_stats_playoffs.csv"
    prior_po = pd.read_csv(po_path) if po_path.exists() else pd.DataFrame()

    # ── Load player bios for age features ──
    bios_path = RAW_DIR / "player_bios.csv"
    bios_df = pd.read_csv(bios_path) if bios_path.exists() else None

    # ── Get all contender teams (playoff + bubble) ──
    team_round_reach_probs: dict[str, dict[str, float]] = {}
    eg_path = PROCESSED_DIR / "team_expected_games_2026.json"
    if eg_path.exists():
        with open(eg_path) as f:
            eg_data = json.load(f)
        playoff_teams = eg_data.get("playoff_teams", [])
        bubble_teams = eg_data.get("bubble_teams", [])
        all_contenders = eg_data.get("all_contenders", playoff_teams + bubble_teams)
        team_round_reach_probs = eg_data.get("team_round_reach_probs") or {}
        if not team_round_reach_probs:
            logger.warning(
                "team_round_reach_probs missing from %s; round features default to 0",
                eg_path,
            )
    else:
        all_contenders = list(team_expected_games.keys())
        playoff_teams = all_contenders
        bubble_teams = []

    if not all_contenders:
        logger.error("No contender teams identified")
        return

    # Optional: blend bracket sim toward upset paths (e.g. TBL deeper run vs BUF/CAR)
    bracket_adj_path = PROJECT_ROOT / "config" / "scoring_bracket_adjustments.json"
    team_expected_games, team_round_reach_probs, bracket_adj_meta = (
        apply_scoring_bracket_adjustments(
            team_expected_games, team_round_reach_probs, bracket_adj_path,
        )
    )
    if bracket_adj_meta:
        print(f"\n  Bracket upset blend: upset_blend={bracket_adj_meta['upset_blend']:.2f} "
              f"teams={', '.join(bracket_adj_meta['teams_tuned'])}")
        if bracket_adj_meta.get("comment"):
            print(f"    ({bracket_adj_meta['comment'][:120]})")

    logger.info("Playoff teams (%d): %s", len(playoff_teams), ", ".join(sorted(playoff_teams)))
    logger.info("Bubble teams (%d): %s", len(bubble_teams), ", ".join(sorted(bubble_teams)))
    logger.info("Total contenders: %d teams", len(all_contenders))

    # ── Fetch rosters for all contender teams ──
    print(f"\n{'='*70}")
    print(f"  ROSTER VERIFICATION")
    print(f"{'='*70}")
    team_rosters: dict[str, dict] = {}
    for team in all_contenders:
        roster = fetch_team_roster(client, team)
        team_rosters[team] = roster
        if roster:
            print(f"  {team}: {len(roster)} players on roster")
        else:
            print(f"  {team}: WARNING - could not fetch roster")

    # ── Build prediction features (for ALL contender teams) ──
    pred_df = build_prediction_features(
        current_df, team_expected_games, all_contenders, prior_po, SEASON,
        player_bios_df=bios_df,
        team_probs=team_probs,
        team_round_reach_probs=team_round_reach_probs,
    )

    if pred_df.empty:
        logger.error("No prediction features built")
        return

    # ── Predict with each target model (two-stage: gate + value + intervals) ──
    for target_name, display_col in [
        ("target_points", "pred_points"),
        ("target_goals", "pred_goals"),
        ("target_assists", "pred_assists"),
    ]:
        if target_name not in model_results:
            continue
        res = model_results[target_name]
        feature_cols = res["feature_cols"]
        available = [c for c in feature_cols if c in pred_df.columns]
        X_pred = pred_df[available].copy()
        for c in feature_cols:
            if c not in X_pred.columns:
                X_pred[c] = 0.0
        X_pred = X_pred[feature_cols]

        preds = predict_blended(
            res["models"], res["blend_weights"], X_pred,
            gate=res.get("gate"),
            meta_model=res.get("meta_model"),
            pos_calibration=res.get("pos_calibration"),
            star_calibration=res.get("star_calibration"),
            output_isotonic=res.get("output_isotonic"),
        )
        pred_df[display_col] = np.maximum(preds, 0)

        # Bootstrap prediction intervals (80% CI)
        _, lower, upper = predict_with_intervals(
            res["models"], res["blend_weights"], X_pred,
            gate=res.get("gate"),
            meta_model=res.get("meta_model"),
            output_isotonic=res.get("output_isotonic"),
            n_bootstrap=200,
            ci_level=0.80,
        )
        short = display_col.replace("pred_", "")
        pred_df[f"{short}_lo80"] = np.round(lower, 1)
        pred_df[f"{short}_hi80"] = np.round(upper, 1)

    # Reconcile: goals + assists ~ points
    if "pred_goals" in pred_df.columns and "pred_assists" in pred_df.columns:
        ga_sum = pred_df["pred_goals"] + pred_df["pred_assists"]
        pred_df["pred_points"] = (pred_df.get("pred_points", ga_sum) + ga_sum) / 2

    pred_df["pred_points"] = pred_df["pred_points"].round(1)
    pred_df["pred_goals"] = pred_df.get("pred_goals", 0).round(1)
    pred_df["pred_assists"] = pred_df.get("pred_assists", 0).round(1)

    # ── Rank by predicted points ──
    pred_df = pred_df.sort_values("pred_points", ascending=False).reset_index(drop=True)

    # ── ESPN injury report (all teams; merged by name + team) ──
    print(f"\n{'='*70}")
    print(f"  ESPN INJURY REPORT (live feed)")
    print(f"{'='*70}")
    espn_records, espn_api_ts, espn_err = fetch_espn_nhl_injuries()
    if espn_err:
        logger.warning("ESPN injuries unavailable: %s", espn_err)
        print(f"  WARNING: Could not load ESPN injuries ({espn_err}).")
    else:
        print(f"  Loaded {len(espn_records)} injured-player entries (API ts: {espn_api_ts})")
    espn_lookup = build_espn_injury_lookup(espn_records)
    pred_df = enrich_pred_df_with_espn_injuries(pred_df, espn_lookup, espn_api_ts)
    pred_df = apply_injury_context_overrides(pred_df)

    # ── Roster verification and game-log check for top players ──
    print(f"\n{'='*70}")
    print(f"  ROSTER & GAME-LOG CHECK (top 50 players)")
    print(f"{'='*70}")
    pred_df = verify_and_flag_players(client, pred_df, team_rosters, check_top_n=50)
    pred_df = fill_espn_injury_flags_all_rows(pred_df)

    flagged_count = 0
    excluded_count = 0
    for idx in pred_df.head(50).index:
        row = pred_df.loc[idx]
        flag = row.get("injury_flag", "")
        if flag:
            flagged_count += 1
            print(f"  ** {row['playerName']:25s} ({row['team_abbrev']}) - {flag}")

    if flagged_count == 0:
        print("  All top 50 players appear active and on their team rosters.")
    else:
        print(f"\n  {flagged_count} player(s) flagged.")

    # ── Apply injury discount and roster exclusions ──
    print(f"\n{'='*70}")
    print(f"  APPLYING INJURY DISCOUNTS & ROSTER EXCLUSIONS")
    print(f"{'='*70}")

    espn_discounted = 0
    inactive_discounted = 0
    override_discounted = 0
    for idx in pred_df.index:
        flag = pred_df.at[idx, "injury_flag"] if "injury_flag" in pred_df.columns else ""
        days = pred_df.at[idx, "days_inactive"] if "days_inactive" in pred_df.columns else 0
        on_roster = pred_df.at[idx, "on_roster"] if "on_roster" in pred_df.columns else True
        espn_st = ""
        espn_ty = ""
        if "espn_injury_status" in pred_df.columns:
            espn_st = str(pred_df.at[idx, "espn_injury_status"] or "").strip()
        if "espn_injury_type" in pred_df.columns:
            espn_ty = str(pred_df.at[idx, "espn_injury_type"] or "").strip()

        ci_cols = [c for c in pred_df.columns
                   if c.endswith("_lo80") or c.endswith("_hi80")]

        if flag == "NOT ON ROSTER" or (not on_roster and flag):
            name = pred_df.at[idx, "playerName"]
            old_pts = pred_df.at[idx, "pred_points"]
            for col in ["pred_points", "pred_goals", "pred_assists"] + ci_cols:
                if col in pred_df.columns:
                    pred_df.at[idx, col] = 0.0
            excluded_count += 1
            print(f"  EXCLUDED: {name} (NOT ON ROSTER) - was {old_pts:.1f} pts")

        elif (
            "injury_override_mult" in pred_df.columns
            and pd.notna(pred_df.at[idx, "injury_override_mult"])
        ):
            mult = float(pred_df.at[idx, "injury_override_mult"])
            name = pred_df.at[idx, "playerName"]
            old_pts = pred_df.at[idx, "pred_points"]
            note = str(pred_df.at[idx, "injury_override_note"] or "").strip()
            if mult <= 0.0:
                for col in ["pred_points", "pred_goals", "pred_assists"] + ci_cols:
                    if col in pred_df.columns:
                        pred_df.at[idx, col] = 0.0
                excluded_count += 1
                print(f"  EXCLUDED: {name} (override) - was {old_pts:.1f} pts")
            else:
                for col in ["pred_points", "pred_goals", "pred_assists"] + ci_cols:
                    if col in pred_df.columns:
                        pred_df.at[idx, col] = round(pred_df.at[idx, col] * mult, 1)
                override_discounted += 1
                suffix = f" — {note}" if note else ""
                print(
                    f"  DISCOUNTED: {name} (manual injury context x{mult:.0%}){suffix} - "
                    f"{old_pts:.1f} => {pred_df.at[idx, 'pred_points']:.1f} pts"
                )

        elif espn_st:
            mult = espn_injury_discount_multiplier(espn_st, espn_ty)
            if mult is None:
                mult = 1.0
            name = pred_df.at[idx, "playerName"]
            old_pts = pred_df.at[idx, "pred_points"]
            if mult <= 0.0:
                for col in ["pred_points", "pred_goals", "pred_assists"] + ci_cols:
                    if col in pred_df.columns:
                        pred_df.at[idx, col] = 0.0
                excluded_count += 1
                print(f"  EXCLUDED: {name} (ESPN: {espn_st}) - was {old_pts:.1f} pts")
            elif mult >= 0.999:
                pass
            else:
                for col in ["pred_points", "pred_goals", "pred_assists"] + ci_cols:
                    if col in pred_df.columns:
                        pred_df.at[idx, col] = round(pred_df.at[idx, col] * mult, 1)
                espn_discounted += 1
                print(f"  DISCOUNTED: {name} (ESPN {espn_st}) - "
                      f"{old_pts:.1f} => {pred_df.at[idx, 'pred_points']:.1f} pts "
                      f"(x{mult:.0%})")

        elif "INACTIVE" in str(flag) and days > 0:
            name = pred_df.at[idx, "playerName"]
            old_pts = pred_df.at[idx, "pred_points"]

            if days >= 60:
                discount = 0.0
            elif days >= 30:
                discount = 0.15
            elif days >= 21:
                discount = 0.50
            elif days >= 14:
                discount = 0.75
            else:
                discount = 0.90

            for col in ["pred_points", "pred_goals", "pred_assists"] + ci_cols:
                if col in pred_df.columns:
                    pred_df.at[idx, col] = round(pred_df.at[idx, col] * discount, 1)
            inactive_discounted += 1
            print(f"  DISCOUNTED: {name} ({days}d inactive) - "
                  f"{old_pts:.1f} => {pred_df.at[idx, 'pred_points']:.1f} pts "
                  f"(x{discount:.0%})")

    if (
        excluded_count == 0
        and espn_discounted == 0
        and inactive_discounted == 0
        and override_discounted == 0
    ):
        print("  No adjustments needed.")
    else:
        print(
            f"\n  Summary: {excluded_count} excluded, "
            f"{override_discounted} manual injury-context adjustments, "
            f"{espn_discounted} ESPN injury adjustments, "
            f"{inactive_discounted} game-log inactive adjustments."
        )

    # Re-rank after adjustments
    pred_df = pred_df.sort_values("pred_points", ascending=False).reset_index(drop=True)

    # ── Data freshness summary ──
    print(f"\n{'='*70}")
    print(f"  DATA FRESHNESS VERIFICATION")
    print(f"{'='*70}")
    print(f"  Data fetched at: {fetch_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if espn_api_ts:
        print(f"  ESPN injury feed timestamp: {espn_api_ts}")
    print(f"  Season: {sid} (regular season)")
    print(f"  Total players fetched: {len(current_df)}")
    print(f"  Players on contender teams: {len(pred_df)}")
    print(f"  Playoff teams ({len(playoff_teams)}): {', '.join(sorted(playoff_teams))}")
    if bubble_teams:
        print(f"  Bubble teams ({len(bubble_teams)}): {', '.join(sorted(bubble_teams))}")

    # ── Output top N (rankings use model preds + team expected games from bracket) ──
    top_out = pred_df.head(TOP_PLAYOFF_SCORERS)
    print(f"\n{'='*70}")
    print(f"  PREDICTED TOP {TOP_PLAYOFF_SCORERS} PLAYOFF SCORERS - 2026 NHL PLAYOFFS")
    print(f"{'='*70}")
    print(f"  {'Rk':>3s}  {'Player':<25s} {'Team':>4s} {'Pos':>3s} "
          f"{'Pts':>5s} {'80% CI':>11s} {'G':>4s} {'A':>4s} "
          f"{'ExpGP':>5s} {'RegPts':>6s} {'Notes'}")
    print(f"  {'-'*3}  {'-'*25} {'-'*4} {'-'*3} {'-'*5} {'-'*11} "
          f"{'-'*4} {'-'*4} {'-'*5} {'-'*6} {'-'*15}")

    for i, (_, row) in enumerate(top_out.iterrows()):
        notes = ""
        team = row.get("team_abbrev", "")
        if team in bubble_teams:
            prob_pct = team_probs.get(team, {}).get("playoff_probability", 0) * 100 if team_probs else 0
            notes = f"BUBBLE({prob_pct:.0f}%)"
        flag = row.get("injury_flag", "")
        if flag:
            fs = str(flag)
            notes = (fs[:42] + "…") if len(fs) > 42 else fs

        lo = row.get("points_lo80", 0)
        hi = row.get("points_hi80", 0)
        ci_str = f"[{lo:4.1f}-{hi:4.1f}]"

        print(f"  {i+1:3d}  {row['playerName']:<25s} {team:>4s} "
              f"{row.get('positionCode','?'):>3s} "
              f"{row['pred_points']:5.1f} {ci_str:>11s} "
              f"{row['pred_goals']:4.1f} {row['pred_assists']:4.1f} "
              f"{row.get('team_expected_games', 0):5.1f} "
              f"{row.get('reg_points', 0):6.0f} {notes}")

    # ── Top-N round mix vs historical distribution ──
    mix_buckets: list[str] = []
    for _, row in top_out.iterrows():
        tm = row.get("team_abbrev", "")
        rp = team_round_reach_probs.get(str(tm), {})
        pr2 = float(rp.get("p_r2", 0.0))
        pcf = float(rp.get("p_cf", 0.0))
        pfn = float(rp.get("p_final", 0.0))
        mix_buckets.append(round_mix_bucket_from_probs(pr2, pcf, pfn))
    mix_ct = Counter(mix_buckets)
    n_mix = len(mix_buckets)
    print(f"\n{'='*70}")
    print(
        f"  TOP {TOP_PLAYOFF_SCORERS} — ROUND MIX (by team bracket reach probs)"
    )
    print(f"{'='*70}")
    print(
        f"  {'Bucket':6s} {'This run':>10s}  {'Hist top-40':>14s}  "
        f"(share of slots; hist = 2001-2025 excl. lockout)"
    )
    for key in ("R1", "R2", "CF", "F"):
        run_pct = 100.0 * mix_ct.get(key, 0) / max(n_mix, 1)
        hist = HIST_TOP40_ROUND_MIX_PCT.get(key, 0.0)
        print(f"  {key:6s} {run_pct:9.1f}%  {hist:13.1f}%")

    # ── Save full predictions ──
    out_cols = [
        "playerName", "team_abbrev", "positionCode",
        "pred_points", "points_lo80", "points_hi80",
        "pred_goals", "goals_lo80", "goals_hi80",
        "pred_assists", "assists_lo80", "assists_hi80",
        "team_expected_games", "reg_points", "reg_ppg", "reg_gp",
        "on_roster", "injury_flag", "last_game_date", "days_inactive",
        "espn_injury_status", "espn_injury_type", "espn_injury_comment",
        "espn_injury_return_date", "espn_injury_updated",
        "injury_report_source", "injury_report_fetched_at",
        "injury_override_mult", "injury_override_note",
    ]
    out_cols = [c for c in out_cols if c in pred_df.columns]
    full_path = _dataframe_to_csv_atomic(
        pred_df[out_cols], PROCESSED_DIR / "scoring_predictions_2026.csv"
    )
    top40_path = _dataframe_to_csv_atomic(
        top_out[out_cols],
        PROCESSED_DIR / f"top{TOP_PLAYOFF_SCORERS}_scorers_2026.csv",
    )
    top35_path = _dataframe_to_csv_atomic(
        pred_df[out_cols].head(35),
        PROCESSED_DIR / "top35_scorers_2026.csv",
    )

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    print(f"\n  Predictions (full) saved to {_rel(full_path)}")
    print(f"  Top {TOP_PLAYOFF_SCORERS} saved to {_rel(top40_path)}")
    print(f"  Top 35 saved to {_rel(top35_path)}")

    write_team_player_consistency_report(pred_df, team_expected_games)
    manifest_extra: dict = {
        "health_policy_version": PLAYOFF_HEALTH_POLICY_VERSION,
        "pipeline": "scoring_predict",
    }
    if bracket_adj_meta:
        manifest_extra["scoring_bracket_adjustments"] = bracket_adj_meta
    write_run_manifest(extra=manifest_extra)
    print(f"  Consistency report: data/processed/team_player_consistency_2026.json")
    print(f"  Model manifest: data/processed/model_run_manifest.json")

    try:
        export_md = Path(__file__).resolve().parent / "export_scoring_predictions_md.py"
        subprocess.run(
            [sys.executable, str(export_md)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except Exception as e:
        logger.warning("top_scorers markdown export skipped: %s", e)

    return pred_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training, use saved models")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  NHL PLAYOFF TOP SCORER PREDICTOR")
    print(f"  (with bubble teams, roster verification, injury checks)")
    print(f"{'#'*70}")

    if args.predict_only:
        model_results = joblib.load(MODELS_DIR / "scoring_models.joblib")
        # Always recompute team expected games for freshest standings + bubble teams
        team_exp, team_probs = step_compute_team_expected_games()
        step_predict(model_results, team_exp, team_probs)
    else:
        step_collect()
        team_exp, team_probs = step_compute_team_expected_games()
        model_results = step_build_and_train()
        if model_results:
            step_predict(model_results, team_exp, team_probs)

    print(f"\n{'#'*70}")
    print(f"  SCORING PIPELINE COMPLETE")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
