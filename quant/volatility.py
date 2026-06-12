"""
Realized volatility metrics.

Assumptions:
    - Return type: daily log returns (ln(P_t / P_{t-1}))
    - Annualization: sqrt(252) for equities/ETFs; sqrt(365) for crypto-only series.
      This module always uses sqrt(252) since the return series is aligned to the
      equity calendar (see data.align_returns).
    - Lookback windows: 21 trading days ≈ 1 month; 63 trading days ≈ 1 quarter.
    - Minimum periods: window // 2 — avoids NaN-heavy leading rows while still
      flagging genuinely sparse data at the start of a series.
"""

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def realized_vol(
    returns: pd.Series | pd.DataFrame,
    window: int = 21,
    min_periods: int | None = None,
    annualize: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    Rolling realized volatility (annualized standard deviation of log returns).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily log returns. One series per asset if DataFrame.
    window : int
        Rolling window in trading days. Common values: 21 (1-month), 63 (1-quarter).
    min_periods : int, optional
        Minimum observations required. Defaults to window // 2.
    annualize : bool
        Multiply by sqrt(252) to express as annualized vol. Default: True.

    Returns
    -------
    Same shape as input — rolling standard deviation (annualized if requested).
    """
    if min_periods is None:
        min_periods = window // 2

    vol = returns.rolling(window=window, min_periods=min_periods).std()
    if annualize:
        vol = vol * np.sqrt(_TRADING_DAYS)
    return vol


def vol_term_structure(
    returns: pd.Series | pd.DataFrame,
    short_window: int = 21,
    long_window: int = 63,
) -> pd.DataFrame:
    """
    Compare short-term vs long-term realized volatility.

    Returns a DataFrame with columns '<asset>_21d' and '<asset>_63d'
    (or '21d'/'63d' for a single Series input). A ratio column
    '<asset>_ratio' (short / long) shows whether recent vol is elevated.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily log returns.
    short_window : int
        Short lookback in trading days. Default: 21.
    long_window : int
        Long lookback in trading days. Default: 63.

    Returns
    -------
    pd.DataFrame
        Columns: [<asset>_21d, <asset>_63d, <asset>_ratio] per asset.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    frames = {}
    for col in returns.columns:
        s = returns[col]
        v21 = realized_vol(s, window=short_window)
        v63 = realized_vol(s, window=long_window)
        frames[f"{col}_{short_window}d"] = v21
        frames[f"{col}_{long_window}d"] = v63
        frames[f"{col}_ratio"] = v21 / v63

    return pd.DataFrame(frames)
