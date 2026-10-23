"""
Monte Carlo series simulation.

Instead of predicting series winners directly, simulates
individual games within a best-of-7 series thousands of times
to produce more nuanced probability distributions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import RANDOM_STATE
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

HOME_ICE_BOOST = 0.04  # ~4% boost for home team
DESPERATION_BOOST = 0.02  # small boost when facing elimination
GAME_VARIANCE = 0.06  # per-game random noise


def simulate_series(
    higher_seed_base_prob: float,
    n_simulations: int = 50000,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Simulate a best-of-7 series using per-game probabilities.

    NHL format: higher seed has home ice in games 1,2,5,7.
    Accounts for home-ice advantage and elimination-game dynamics.

    Returns distribution of outcomes.
    """
    rng = np.random.RandomState(seed)

    # Home ice schedule: True = higher seed is home
    home_ice = [True, True, False, False, True, False, True]

    hi_wins_count = np.zeros(5)  # index = games won by loser (0-4)
    hi_total_wins = 0

    for _ in range(n_simulations):
        hi_w, lo_w = 0, 0
        game_num = 0

        while hi_w < 4 and lo_w < 4:
            p = higher_seed_base_prob

            # Home ice adjustment
            if home_ice[game_num]:
                p += HOME_ICE_BOOST
            else:
                p -= HOME_ICE_BOOST

            # Desperation adjustment (facing elimination)
            if lo_w == 3:
                p += DESPERATION_BOOST  # higher seed closes out
            if hi_w == 3:
                p -= DESPERATION_BOOST  # lower seed fights back

            # Game-level randomness
            p += rng.normal(0, GAME_VARIANCE)
            p = np.clip(p, 0.15, 0.85)

            if rng.random() < p:
                hi_w += 1
            else:
                lo_w += 1

            game_num += 1

        if hi_w == 4:
            hi_total_wins += 1
            hi_wins_count[lo_w] += 1

    series_win_prob = hi_total_wins / n_simulations

    return {
        "series_win_prob": series_win_prob,
        "prob_sweep": hi_wins_count[0] / n_simulations,
        "prob_5_games": hi_wins_count[1] / n_simulations,
        "prob_6_games": hi_wins_count[2] / n_simulations,
        "prob_7_games": hi_wins_count[3] / n_simulations,
        "expected_games": _expected_games(higher_seed_base_prob),
        "n_simulations": n_simulations,
    }


def simulate_bracket(
    matchups: list[dict[str, Any]],
    n_simulations: int = 50000,
) -> dict[str, Any]:
    """
    Simulate an entire playoff bracket using Monte Carlo.

    matchups: list of dicts with 'higher_seed', 'lower_seed', 'base_prob'
    Returns round-by-round results with probability distributions.
    """
    round_names = ["First Round", "Second Round", "Conference Finals", "Stanley Cup Final"]
    bracket_results: dict[str, list[dict]] = {}
    current_matchups = matchups

    for round_idx, round_name in enumerate(round_names):
        if not current_matchups:
            break

        round_results = []
        for m in current_matchups:
            sim = simulate_series(m["base_prob"], n_simulations)
            result = {
                "higher_seed": m["higher_seed"],
                "lower_seed": m["lower_seed"],
                "base_prob": m["base_prob"],
                **sim,
            }
            result["predicted_winner"] = (
                m["higher_seed"] if sim["series_win_prob"] >= 0.5
                else m["lower_seed"]
            )
            round_results.append(result)

        bracket_results[round_name] = round_results

        # Build next round
        winners = []
        for r in round_results:
            winners.append({
                "team": r["predicted_winner"],
                "prob": r["series_win_prob"],
            })

        next_matchups = []
        for i in range(0, len(winners) - 1, 2):
            t1 = winners[i]
            t2 = winners[i + 1]
            # Higher seed in next round = team with higher probability path
            if t1["prob"] >= t2["prob"]:
                next_matchups.append({
                    "higher_seed": t1["team"],
                    "lower_seed": t2["team"],
                    "base_prob": 0.5 + abs(t1["prob"] - t2["prob"]) * 0.2,
                })
            else:
                next_matchups.append({
                    "higher_seed": t2["team"],
                    "lower_seed": t1["team"],
                    "base_prob": 0.5 + abs(t2["prob"] - t1["prob"]) * 0.2,
                })

        current_matchups = next_matchups

    return bracket_results


def _expected_games(p: float) -> float:
    """Compute expected number of games in a best-of-7 given per-game win prob p."""
    q = 1 - p
    # Probability of ending in exactly n games
    expected = 0
    for n in range(4, 8):
        if n == 4:
            prob_n = p**4 + q**4
        elif n == 5:
            from math import comb
            prob_n = comb(4, 3) * (p**4 * q + q**4 * p)
        elif n == 6:
            from math import comb
            prob_n = comb(5, 3) * (p**4 * q**2 + q**4 * p**2)
        elif n == 7:
            from math import comb
            prob_n = comb(6, 3) * (p**4 * q**3 + q**4 * p**3)
        else:
            prob_n = 0
        expected += n * prob_n
    return expected


def print_bracket_results(results: dict[str, list[dict]]) -> None:
    """Pretty-print Monte Carlo bracket simulation results."""
    for round_name, round_results in results.items():
        print(f"\n{'='*75}")
        print(f"  {round_name}")
        print(f"{'='*75}")

        for r in round_results:
            winner = r["predicted_winner"]
            prob = r["series_win_prob"] * 100
            hi = r["higher_seed"]
            lo = r["lower_seed"]

            print(f"\n  {hi} vs {lo}")
            print(f"    Winner: {winner} ({prob:.1f}%)")
            print(f"    Expected games: {r['expected_games']:.1f}")
            print(f"    In 4: {r['prob_sweep']*100:.1f}%  "
                  f"In 5: {r['prob_5_games']*100:.1f}%  "
                  f"In 6: {r['prob_6_games']*100:.1f}%  "
                  f"In 7: {r['prob_7_games']*100:.1f}%")

    if "Stanley Cup Final" in results and results["Stanley Cup Final"]:
        final = results["Stanley Cup Final"][0]
        champion = final["predicted_winner"]
        print(f"\n{'*'*75}")
        print(f"  PREDICTED CHAMPION: {champion} ({final['series_win_prob']*100:.1f}%)")
        print(f"{'*'*75}")
