"""
Factor exposure metrics.

Metrics implemented:
    1. Market beta (OLS of asset returns on SPY returns)
    2. Growth-vs-value tilt (OLS on VUG−VTV return spread)
    3. Sector concentration (Herfindahl-Hirschman Index on portfolio weights)

Assumptions:
    - Beta estimated via OLS on daily log returns (not excess returns — risk-free
      rate omitted as a simplification; acceptable for equity beta estimation at
      daily frequency with near-zero daily rf).
    - Lookback: full sample by default; use returns.iloc[-window:] to window it.
    - R² and stderr reported so reader can assess regression quality.
    - Growth/value tilt: regress asset returns on (VUG_returns − VTV_returns).
      Positive coefficient → growth tilt; negative → value tilt.
    - HHI: sum of squared portfolio weights. Range [1/n, 1].
      HHI < 0.15 → diversified; > 0.25 → concentrated (US DOJ thresholds).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSResult:
    """Container for a single-factor OLS regression result."""
    beta: float
    alpha: float
    r_squared: float
    stderr: float
    t_stat: float
    p_value: float
    n_obs: int


def beta_to_spy(
    asset_returns: pd.Series,
    spy_returns: pd.Series,
) -> OLSResult:
    """
    Estimate market beta via OLS: R_asset = alpha + beta * R_SPY + epsilon.

    Parameters
    ----------
    asset_returns : pd.Series
        Daily log returns of the asset.
    spy_returns : pd.Series
        Daily log returns of SPY (market proxy).

    Returns
    -------
    OLSResult
        OLS coefficients with goodness-of-fit statistics.
    """
    # Align on common dates, drop NaN
    common = pd.concat([asset_returns, spy_returns], axis=1).dropna()
    y = common.iloc[:, 0].values
    x = common.iloc[:, 1].values

    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    t_stat = slope / stderr if stderr > 0 else float("nan")

    return OLSResult(
        beta=slope,
        alpha=intercept,
        r_squared=r_value ** 2,
        stderr=stderr,
        t_stat=t_stat,
        p_value=p_value,
        n_obs=len(y),
    )


def factor_exposures(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
) -> pd.DataFrame:
    """
    Apply beta_to_spy to every asset column.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, one column per asset.
    spy_returns : pd.Series
        Daily log returns of SPY.

    Returns
    -------
    pd.DataFrame
        Index: asset names. Columns: beta, alpha, r_squared, stderr, t_stat,
        p_value, n_obs.
    """
    rows = {}
    for col in returns.columns:
        if col == "SPY":
            continue
        result = beta_to_spy(returns[col], spy_returns)
        rows[col] = {
            "beta": result.beta,
            "alpha": result.alpha,
            "r_squared": result.r_squared,
            "stderr": result.stderr,
            "t_stat": result.t_stat,
            "p_value": result.p_value,
            "n_obs": result.n_obs,
        }
    return pd.DataFrame(rows).T


def vug_vtv_tilt(
    asset_returns: pd.Series,
    vug_returns: pd.Series,
    vtv_returns: pd.Series,
) -> OLSResult:
    """
    Estimate growth-vs-value tilt: R_asset = alpha + coef * (R_VUG − R_VTV).

    Interpretation:
        coefficient > 0: asset moves with growth stocks (VUG) → growth tilt
        coefficient < 0: asset moves with value stocks (VTV) → value tilt
        |coefficient| ≈ 0: no meaningful tilt

    Parameters
    ----------
    asset_returns : pd.Series
        Daily log returns of the asset.
    vug_returns : pd.Series
        Daily log returns of VUG (Vanguard Growth ETF).
    vtv_returns : pd.Series
        Daily log returns of VTV (Vanguard Value ETF).

    Returns
    -------
    OLSResult
        OLS result where `beta` is the growth-value tilt coefficient.
    """
    spread = (vug_returns - vtv_returns).rename("gv_spread")
    common = pd.concat([asset_returns, spread], axis=1).dropna()
    y = common.iloc[:, 0].values
    x = common.iloc[:, 1].values

    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    t_stat = slope / stderr if stderr > 0 else float("nan")

    return OLSResult(
        beta=slope,       # growth-value tilt coefficient
        alpha=intercept,
        r_squared=r_value ** 2,
        stderr=stderr,
        t_stat=t_stat,
        p_value=p_value,
        n_obs=len(y),
    )


def hhi(weights: dict[str, float] | pd.Series) -> float:
    """
    Herfindahl-Hirschman Index for portfolio concentration.

    HHI = sum(w_i^2) where weights are normalized to sum to 1.

    Range: [1/n, 1]
        1/n → perfectly diversified (equal weight)
        1.0 → fully concentrated in one asset

    DOJ guidelines (adapted):
        HHI < 0.15  → unconcentrated (diversified)
        0.15–0.25   → moderately concentrated
        > 0.25      → highly concentrated

    Parameters
    ----------
    weights : dict or pd.Series
        Portfolio weights (need not sum to 1; they are normalized).

    Returns
    -------
    float
        HHI value in [1/n, 1].
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)
    w = weights.values.astype(float)
    w = w / w.sum()
    return float((w ** 2).sum())
