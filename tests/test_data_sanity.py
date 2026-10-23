"""
Smoke tests for loaders, feature shapes, and bounded NaNs (one synthetic season).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.models.scoring_predictor import (
    FEATURE_COLS,
    META_COLS,
    prepare_xy,
    temporal_cv_splits,
)


def _minimal_scoring_df(n: int = 80, season: int = 2019) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "playerId": i,
                "playerName": f"P{i}",
                "team_abbrev": "TES",
                "season": season,
                "positionCode": "C",
                "target_points": float(np.random.poisson(4)),
                "target_goals": 1.0,
                "target_assists": 1.0,
                "target_gp": 10.0,
            }
        )
    df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.random.randn(n) + 0.5
    df["is_forward"] = 1
    df["is_center"] = 1
    df["is_defenseman"] = 0
    df["is_top3_team_scorer"] = (df.index < 3).astype(int)
    return df


def test_prepare_xy_shapes():
    df = _minimal_scoring_df()
    X, y, cols = prepare_xy(df)
    assert len(y) == len(X) == len(df)
    assert all(c in df.columns for c in cols)


def test_temporal_cv_splits_walk_forward():
    parts = []
    for yr in (2017, 2018, 2019, 2020, 2021):
        d = _minimal_scoring_df(season=yr)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    splits = temporal_cv_splits(df, min_train_years=2)
    assert len(splits) >= 1
    for tr, te in splits:
        assert len(tr) > 0 and len(te) > 0
        assert df.iloc[te]["season"].nunique() == 1


def test_no_nan_explosion_after_impute():
    df = _minimal_scoring_df()
    df.loc[:5, "reg_ppg"] = np.nan
    df.loc[10:15, "team_expected_games"] = np.nan
    X, _, _ = prepare_xy(df)
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(X)
    assert not np.isnan(Xt).any()
    assert np.isfinite(Xt).all()


def test_meta_columns_present():
    df = _minimal_scoring_df()
    for c in META_COLS:
        assert c in df.columns
