"""
Unit tests for quant.factors.

All tests use synthetic data — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from quant.factors import (
    OLSResult,
    beta_to_spy,
    factor_exposures,
    vug_vtv_tilt,
    hhi,
)

_RNG = np.random.default_rng(42)


def make_index(n: int = 252) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-02", periods=n, freq="B")


def make_market_returns(n: int = 252, sigma: float = 0.01) -> pd.Series:
    dates = make_index(n)
    return pd.Series(_RNG.normal(0, sigma, n), index=dates, name="SPY")


def make_asset_returns(
    spy_returns: pd.Series, beta: float = 1.0, noise: float = 0.005
) -> pd.Series:
    """Synthetic asset with known beta."""
    noise_series = _RNG.normal(0, noise, len(spy_returns))
    return pd.Series(
        beta * spy_returns.values + noise_series,
        index=spy_returns.index,
        name="ASSET",
    )


class TestBetaToSPY:
    def test_beta_one_for_spy_itself(self):
        spy = make_market_returns()
        result = beta_to_spy(spy, spy)
        assert abs(result.beta - 1.0) < 0.01
        assert abs(result.r_squared - 1.0) < 1e-6

    def test_known_beta(self):
        spy = make_market_returns(n=500)
        asset = make_asset_returns(spy, beta=1.5, noise=0.002)
        result = beta_to_spy(asset, spy)
        assert abs(result.beta - 1.5) < 0.05  # within 0.05 due to noise

    def test_low_beta(self):
        spy = make_market_returns(n=500)
        asset = make_asset_returns(spy, beta=0.3, noise=0.002)
        result = beta_to_spy(asset, spy)
        assert abs(result.beta - 0.3) < 0.05

    def test_returns_ols_result(self):
        spy = make_market_returns()
        asset = make_asset_returns(spy)
        result = beta_to_spy(asset, spy)
        assert isinstance(result, OLSResult)
        assert 0 <= result.r_squared <= 1
        assert result.n_obs == len(spy)
        assert result.stderr >= 0

    def test_handles_nan_alignment(self):
        spy = make_market_returns(n=100)
        asset = make_asset_returns(spy, beta=1.0)
        # Introduce NaN in asset
        asset.iloc[10:15] = np.nan
        result = beta_to_spy(asset, spy)
        assert result.n_obs == 95  # 5 NaN rows dropped


class TestFactorExposures:
    def test_excludes_spy_column(self):
        spy = make_market_returns(n=200)
        df = pd.DataFrame(
            {"AAPL": make_asset_returns(spy, 1.2).values,
             "SPY": spy.values},
            index=spy.index,
        )
        table = factor_exposures(df, spy)
        assert "SPY" not in table.index
        assert "AAPL" in table.index

    def test_output_columns(self):
        spy = make_market_returns(n=200)
        df = pd.DataFrame(
            {"AAPL": make_asset_returns(spy, 1.2).values,
             "NVDA": make_asset_returns(spy, 1.8).values},
            index=spy.index,
        )
        table = factor_exposures(df, spy)
        for col in ["beta", "alpha", "r_squared", "stderr", "t_stat", "p_value", "n_obs"]:
            assert col in table.columns

    def test_higher_beta_asset(self):
        spy = make_market_returns(n=500)
        df = pd.DataFrame(
            {"HIGH": make_asset_returns(spy, 2.0, noise=0.001).values,
             "LOW": make_asset_returns(spy, 0.5, noise=0.001).values},
            index=spy.index,
        )
        table = factor_exposures(df, spy)
        assert table.loc["HIGH", "beta"] > table.loc["LOW", "beta"]


class TestVugVtvTilt:
    def test_positive_tilt(self):
        idx = make_index(300)
        vug = pd.Series(_RNG.normal(0, 0.01, 300), index=idx)
        vtv = pd.Series(_RNG.normal(0, 0.01, 300), index=idx)
        spread = vug - vtv
        # Asset moves strongly with spread → growth tilt
        asset = 1.5 * spread + pd.Series(_RNG.normal(0, 0.002, 300), index=idx)
        result = vug_vtv_tilt(asset, vug, vtv)
        assert result.beta > 0.5

    def test_returns_ols_result(self):
        idx = make_index(200)
        vug = pd.Series(_RNG.normal(0, 0.01, 200), index=idx)
        vtv = pd.Series(_RNG.normal(0, 0.01, 200), index=idx)
        asset = pd.Series(_RNG.normal(0, 0.01, 200), index=idx)
        result = vug_vtv_tilt(asset, vug, vtv)
        assert isinstance(result, OLSResult)


class TestHHI:
    def test_equal_weights_n4(self):
        # HHI of 4 equal weights = 4 * (0.25)^2 = 0.25
        result = hhi({"A": 1, "B": 1, "C": 1, "D": 1})
        assert abs(result - 0.25) < 1e-9

    def test_full_concentration(self):
        # One asset with weight 1 → HHI = 1.0
        result = hhi({"A": 1.0, "B": 0.0})
        assert abs(result - 1.0) < 1e-9

    def test_normalizes_unnormalized_weights(self):
        # Weights summing to 2 should give same result as summing to 1
        result_raw = hhi({"A": 2, "B": 2})
        result_norm = hhi({"A": 0.5, "B": 0.5})
        assert abs(result_raw - result_norm) < 1e-9

    def test_series_input(self):
        weights = pd.Series({"X": 0.3, "Y": 0.7})
        result = hhi(weights)
        expected = 0.3 ** 2 + 0.7 ** 2
        assert abs(result - expected) < 1e-9

    def test_range(self):
        for n in [2, 5, 10]:
            equal = {str(i): 1 for i in range(n)}
            result = hhi(equal)
            assert abs(result - 1 / n) < 1e-9  # minimum HHI = 1/n
