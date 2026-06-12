"""
Unit tests for quant.correlation.

All tests use synthetic data — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from quant.correlation import rolling_corr_matrix, spot_corr_matrix, avg_pairwise_corr

_RNG = np.random.default_rng(42)


def make_returns(n: int = 150, n_assets: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    data = _RNG.normal(0, 0.01, (n, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


class TestSpotCorrMatrix:
    def test_shape(self):
        returns = make_returns(n=100, n_assets=4)
        corr = spot_corr_matrix(returns, window=63)
        assert corr.shape == (4, 4)

    def test_diagonal_is_one(self):
        returns = make_returns(n=100)
        corr = spot_corr_matrix(returns, window=63)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)

    def test_symmetric(self):
        returns = make_returns(n=100)
        corr = spot_corr_matrix(returns, window=63)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)

    def test_values_in_range(self):
        returns = make_returns(n=200)
        corr = spot_corr_matrix(returns, window=63)
        assert (corr.values >= -1.0).all()
        assert (corr.values <= 1.0).all()

    def test_perfect_correlation(self):
        dates = pd.date_range("2023-01-02", periods=100, freq="B")
        s = pd.Series(_RNG.normal(0, 0.01, 100), index=dates)
        df = pd.DataFrame({"X": s.values, "Y": s.values}, index=dates)
        corr = spot_corr_matrix(df, window=63)
        # X and Y are identical → corr[X,Y] == 1.0
        assert abs(corr.loc["X", "Y"] - 1.0) < 1e-9

    def test_as_of_parameter(self):
        returns = make_returns(n=200)
        mid_date = returns.index[100]
        corr_mid = spot_corr_matrix(returns, window=63, as_of=mid_date)
        corr_end = spot_corr_matrix(returns, window=63)
        # Different windows → different results (almost certain with random data)
        assert not corr_mid.equals(corr_end)


class TestRollingCorrMatrix:
    def test_returns_dict(self):
        returns = make_returns(n=100)
        result = rolling_corr_matrix(returns, window=21)
        assert isinstance(result, dict)

    def test_dict_keys_are_timestamps(self):
        returns = make_returns(n=100)
        result = rolling_corr_matrix(returns, window=21)
        for key in result:
            assert isinstance(key, pd.Timestamp)

    def test_each_matrix_is_symmetric(self):
        returns = make_returns(n=100)
        result = rolling_corr_matrix(returns, window=21)
        for _, corr in list(result.items())[:5]:
            np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)

    def test_min_periods_respected(self):
        returns = make_returns(n=50)
        # With window=21 and min_periods=21, first key should be at index 20
        result = rolling_corr_matrix(returns, window=21, min_periods=21)
        first_key = min(result.keys())
        assert first_key == returns.index[20]


class TestAvgPairwiseCorr:
    def test_identity_matrix_is_zero(self):
        corr = pd.DataFrame(np.eye(4), columns=list("ABCD"), index=list("ABCD"))
        assert avg_pairwise_corr(corr) == 0.0

    def test_all_ones_off_diagonal(self):
        corr = pd.DataFrame(np.ones((3, 3)), columns=list("ABC"), index=list("ABC"))
        assert abs(avg_pairwise_corr(corr) - 1.0) < 1e-9

    def test_range(self):
        returns = make_returns(n=150)
        corr = spot_corr_matrix(returns, window=63)
        avg = avg_pairwise_corr(corr)
        assert -1.0 <= avg <= 1.0
