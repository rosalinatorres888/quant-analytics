"""
Drawdown metrics — per-asset and portfolio.

Assumptions:
    - Input: price series (not returns). Prices must be positive.
    - Drawdown defined as (P_t - peak_t) / peak_t where peak_t is the
      running maximum of prices up to time t.
    - Expressed as a negative fraction (e.g. -0.35 = 35% drawdown).
    - Duration measured in calendar/trading days depending on index frequency.
    - Portfolio drawdown uses equal weights by default; pass explicit weights
      for a custom portfolio.
"""

import numpy as np
import pandas as pd


def drawdown_series(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Compute drawdown at each point in time.

    DD_t = (P_t - max(P_0..P_t)) / max(P_0..P_t)

    Returns negative values in [−1, 0]; 0 when at an all-time high.
    """
    rolling_peak = prices.expanding().max()
    return (prices - rolling_peak) / rolling_peak


def max_drawdown(prices: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Maximum drawdown over the full price history.

    Returns a scalar for Series input; a pd.Series (one value per asset)
    for DataFrame input.
    """
    dd = drawdown_series(prices)
    return dd.min()


def drawdown_duration(prices: pd.Series) -> int:
    """
    Maximum drawdown duration in periods (days if daily data).

    Duration = number of periods from the drawdown peak to the trough,
    measured as the longest consecutive stretch below a prior high.

    Parameters
    ----------
    prices : pd.Series
        Single-asset price series.

    Returns
    -------
    int
        Longest drawdown duration in periods.
    """
    dd = drawdown_series(prices)
    is_underwater = dd < 0

    max_dur = 0
    current_dur = 0
    for under in is_underwater:
        if under:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def drawdown_table(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: max drawdown and duration for each asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Price history, one column per asset.

    Returns
    -------
    pd.DataFrame
        Index: asset names. Columns: max_drawdown, duration_days.
    """
    rows = {}
    for col in prices.columns:
        s = prices[col].dropna()
        rows[col] = {
            "max_drawdown": max_drawdown(s),
            "duration_days": drawdown_duration(s),
        }
    return pd.DataFrame(rows).T


# underwater_curve is an alias — same as drawdown_series, named for plot clarity
def underwater_curve(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Alias for drawdown_series — returns the underwater equity curve."""
    return drawdown_series(prices)


def portfolio_drawdown(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Drawdown of an equally-weighted (or custom-weighted) portfolio.

    Parameters
    ----------
    prices : pd.DataFrame
        Price history. NaN rows forward-filled before portfolio construction.
    weights : dict, optional
        {ticker: weight}. Weights need not sum to 1 — they are normalized.
        Default: equal weight across all columns.

    Returns
    -------
    pd.Series
        Portfolio drawdown time series.
    """
    p = prices.ffill().dropna()
    if weights is None:
        w = np.ones(len(p.columns)) / len(p.columns)
    else:
        raw = np.array([weights.get(c, 0.0) for c in p.columns])
        w = raw / raw.sum()

    # Normalize each asset to start at 1.0, then compute portfolio NAV
    normalized = p / p.iloc[0]
    portfolio_nav = normalized.dot(w)
    return drawdown_series(portfolio_nav)
