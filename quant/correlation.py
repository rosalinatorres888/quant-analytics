"""
Rolling correlation matrix.

Assumptions:
    - Input returns are already calendar-aligned (use data.align_returns first).
    - Pearson correlation computed on daily log returns.
    - Lookback window: 63 trading days (≈1 quarter) by default.
    - Minimum periods: window // 2.
    - Crypto/equity calendar misalignment is handled upstream in data.align_returns;
      this module receives a clean, aligned DataFrame.
"""

import numpy as np
import pandas as pd


def rolling_corr_matrix(
    returns: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute rolling Pearson correlation matrices.

    Returns a dict mapping each date to a (n_assets × n_assets) correlation
    DataFrame. Only dates where enough observations exist (>= min_periods) are
    included.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, assets as columns, dates as index.
    window : int
        Lookback window in trading days. Default: 63.
    min_periods : int, optional
        Minimum observations required. Defaults to window // 2.

    Returns
    -------
    dict[pd.Timestamp, pd.DataFrame]
        Keys: dates; values: symmetric correlation matrix with asset names.

    Notes
    -----
    For large datasets, this can be memory-intensive. Use spot_corr_matrix
    for a single snapshot without the full dict overhead.
    """
    if min_periods is None:
        min_periods = window // 2

    result = {}
    for i in range(window - 1, len(returns)):
        window_data = returns.iloc[max(0, i - window + 1) : i + 1]
        if window_data.shape[0] < min_periods:
            continue
        corr = window_data.corr(method="pearson")
        result[returns.index[i]] = corr

    return result


def spot_corr_matrix(
    returns: pd.DataFrame,
    window: int = 63,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Single-snapshot correlation matrix for a given date.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns.
    window : int
        Lookback window in trading days. Default: 63.
    as_of : pd.Timestamp or str, optional
        End date for the window. Defaults to the last available date.

    Returns
    -------
    pd.DataFrame
        Symmetric Pearson correlation matrix.
    """
    if as_of is None:
        as_of = returns.index[-1]
    else:
        as_of = pd.Timestamp(as_of)

    idx = returns.index.get_indexer([as_of], method="ffill")[0]
    start = max(0, idx - window + 1)
    window_data = returns.iloc[start : idx + 1]
    return window_data.corr(method="pearson")


def avg_pairwise_corr(corr_matrix: pd.DataFrame) -> float:
    """
    Mean of all off-diagonal correlation entries (upper triangle).

    Useful as a single-number measure of portfolio diversification.
    Lower → more diversified.
    """
    n = len(corr_matrix)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    values = corr_matrix.values[mask]
    return float(np.nanmean(values))
