"""
Unit tests for quant.drawdown.

All tests use synthetic data — no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from quant.drawdown import (
    drawdown_series,
    max_drawdown,
    drawdown_duration,
    drawdown_table,
    underwater_curve,
    portfolio_drawdown,
)

_RNG = np.random.default_rng(42)


def make_prices(n: int = 100, start: float = 100.0, drift: float = 0.0005,
                sigma: float = 0.01) -> pd.Series:
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    log_ret = _RNG.normal(drift, sigma, n)
    prices = start * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=dates, name="ASSET")


class TestDrawdownSeries:
    def test_all_nonpositive(self):
        prices = make_prices()
        dd = drawdown_series(prices)
        assert (dd <= 0).all()

    def test_zero_at_new_high(self):
        # Monotonically increasing prices → always at ATH → dd == 0
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.Series(np.arange(1.0, 11.0), index=dates)
        dd = drawdown_series(prices)
        assert (dd == 0).all()

    def test_known_drawdown(self):
        # Peak at 100, drops to 80 → dd = -0.20
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        prices = pd.Series([100.0, 90.0, 80.0], index=dates)
        dd = drawdown_series(prices)
        assert abs(dd.iloc[-1] - (-0.20)) < 1e-9

    def test_underwater_curve_alias(self):
        prices = make_prices()
        pd.testing.assert_series_equal(drawdown_series(prices), underwater_curve(prices))


class TestMaxDrawdown:
    def test_monotone_increase(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.Series(np.arange(1.0, 11.0), index=dates)
        assert max_drawdown(prices) == 0.0

    def test_known_max_drawdown(self):
        dates = pd.date_range("2023-01-02", periods=4, freq="B")
        # Peak=100, low=60 → max_dd = -0.40
        prices = pd.Series([100.0, 90.0, 60.0, 70.0], index=dates)
        assert abs(max_drawdown(prices) - (-0.40)) < 1e-9

    def test_dataframe_returns_series(self):
        df = pd.DataFrame({"A": make_prices().values, "B": make_prices().values},
                          index=make_prices().index)
        result = max_drawdown(df)
        assert isinstance(result, pd.Series)
        assert set(result.index) == {"A", "B"}
        assert (result <= 0).all()


class TestDrawdownDuration:
    def test_no_drawdown(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        prices = pd.Series(np.arange(1.0, 11.0), index=dates)
        assert drawdown_duration(prices) == 0

    def test_single_dip(self):
        # ATH → 2 underwater periods → new ATH
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        prices = pd.Series([100.0, 90.0, 80.0, 110.0, 120.0], index=dates)
        # Periods 1 and 2 are underwater (indices 1, 2)
        assert drawdown_duration(prices) == 2

    def test_longer_drawdown(self):
        n = 10
        dates = pd.date_range("2023-01-02", periods=n, freq="B")
        prices = pd.Series([100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 80.0, 85.0, 100.0, 110.0],
                           index=dates)
        assert drawdown_duration(prices) == 7  # indices 1-7 underwater


class TestDrawdownTable:
    def test_shape_and_columns(self):
        df = pd.DataFrame(
            {"A": make_prices().values, "B": make_prices().values},
            index=make_prices().index,
        )
        table = drawdown_table(df)
        assert set(table.index) == {"A", "B"}
        assert "max_drawdown" in table.columns
        assert "duration_days" in table.columns

    def test_values_in_range(self):
        df = pd.DataFrame(
            {"A": make_prices().values, "B": make_prices().values},
            index=make_prices().index,
        )
        table = drawdown_table(df)
        assert (table["max_drawdown"] <= 0).all()
        assert (table["duration_days"] >= 0).all()


class TestPortfolioDrawdown:
    def test_equal_weight_two_assets(self):
        prices = pd.DataFrame(
            {"A": make_prices().values, "B": make_prices().values},
            index=make_prices().index,
        )
        dd = portfolio_drawdown(prices)
        assert isinstance(dd, pd.Series)
        assert (dd <= 0).all()

    def test_custom_weights(self):
        prices = pd.DataFrame(
            {"A": make_prices().values, "B": make_prices().values},
            index=make_prices().index,
        )
        dd = portfolio_drawdown(prices, weights={"A": 0.8, "B": 0.2})
        assert isinstance(dd, pd.Series)
        assert (dd <= 0).all()
