"""
Unit tests for quant.volatility.

All tests use synthetic data — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from quant.volatility import realized_vol, vol_term_structure

_RNG = np.random.default_rng(42)


def make_returns(n: int = 252, sigma: float = 0.01) -> pd.Series:
    """Synthetic daily log returns with known daily std."""
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    values = _RNG.normal(0, sigma, n)
    return pd.Series(values, index=dates, name="ASSET")


class TestRealizedVol:
    def test_annualized_magnitude(self):
        # Daily sigma=0.01 → annualized vol ≈ 0.01 * sqrt(252) ≈ 0.1587
        returns = make_returns(n=252, sigma=0.01)
        vol = realized_vol(returns, window=252, min_periods=252)
        result = vol.dropna()
        assert len(result) == 1
        expected = 0.01 * np.sqrt(252)
        assert abs(result.iloc[0] - expected) / expected < 0.10  # within 10% (sample vs population std)

    def test_not_annualized(self):
        returns = make_returns(n=252, sigma=0.01)
        vol_raw = realized_vol(returns, window=252, min_periods=252, annualize=False)
        vol_ann = realized_vol(returns, window=252, min_periods=252, annualize=True)
        ratio = (vol_ann / vol_raw).dropna().iloc[0]
        assert abs(ratio - np.sqrt(252)) < 0.01

    def test_rolling_window_length(self):
        returns = make_returns(n=100)
        vol = realized_vol(returns, window=21, min_periods=10)
        # First 9 rows should be NaN (fewer than min_periods=10)
        assert vol.iloc[:9].isna().all()
        assert vol.iloc[9:].notna().all()

    def test_dataframe_input(self):
        n = 100
        df = pd.DataFrame(
            {"A": make_returns(n).values, "B": make_returns(n).values},
            index=make_returns(n).index,
        )
        vol = realized_vol(df, window=21)
        assert isinstance(vol, pd.DataFrame)
        assert list(vol.columns) == ["A", "B"]

    def test_constant_returns_zero_vol(self):
        dates = pd.date_range("2023-01-02", periods=50, freq="B")
        returns = pd.Series(np.zeros(50), index=dates)
        vol = realized_vol(returns, window=21, min_periods=10)
        assert (vol.dropna() == 0.0).all()

    def test_non_negative(self):
        returns = make_returns(n=200)
        vol = realized_vol(returns, window=21)
        assert (vol.dropna() >= 0).all()


class TestVolTermStructure:
    def test_output_columns(self):
        returns = make_returns(n=150)
        ts = vol_term_structure(returns, short_window=21, long_window=63)
        assert "ASSET_21d" in ts.columns
        assert "ASSET_63d" in ts.columns
        assert "ASSET_ratio" in ts.columns

    def test_ratio_is_short_over_long(self):
        returns = make_returns(n=200)
        ts = vol_term_structure(returns, short_window=21, long_window=63)
        ratio = ts["ASSET_ratio"].dropna()
        v21 = ts["ASSET_21d"].loc[ratio.index]
        v63 = ts["ASSET_63d"].loc[ratio.index]
        np.testing.assert_allclose(ratio.values, (v21 / v63).values, rtol=1e-6)

    def test_dataframe_input(self):
        n = 200
        df = pd.DataFrame(
            {"X": make_returns(n).values, "Y": make_returns(n).values},
            index=make_returns(n).index,
        )
        ts = vol_term_structure(df)
        for col in ["X", "Y"]:
            assert f"{col}_21d" in ts.columns
            assert f"{col}_63d" in ts.columns
            assert f"{col}_ratio" in ts.columns
