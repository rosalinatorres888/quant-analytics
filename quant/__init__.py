"""
quant-analytics — risk metrics and factor analysis library.

Modules:
    data        — price fetching (yfinance + CoinGecko) and CSV caching
    volatility  — realized volatility (21d/63d) and VIX context
    correlation — rolling correlation matrix with equity/crypto calendar alignment
    drawdown    — max drawdown, duration, underwater curves
    factors     — beta to SPY, VUG/VTV tilt, sector concentration (HHI)

Typical usage:
    from quant.data import get_prices, align_returns
    from quant.volatility import realized_vol
    from quant.drawdown import max_drawdown, drawdown_series
    from quant.factors import beta_to_spy, factor_exposures
"""

from .data import get_prices, align_returns, EQUITY_TICKERS, CRYPTO_IDS, ALL_ASSETS
from .volatility import realized_vol, vol_term_structure
from .correlation import rolling_corr_matrix, spot_corr_matrix
from .drawdown import drawdown_series, max_drawdown, drawdown_duration, underwater_curve
from .factors import beta_to_spy, factor_exposures, vug_vtv_tilt, hhi

__all__ = [
    "get_prices",
    "align_returns",
    "EQUITY_TICKERS",
    "CRYPTO_IDS",
    "ALL_ASSETS",
    "realized_vol",
    "vol_term_structure",
    "rolling_corr_matrix",
    "spot_corr_matrix",
    "drawdown_series",
    "max_drawdown",
    "drawdown_duration",
    "underwater_curve",
    "beta_to_spy",
    "factor_exposures",
    "vug_vtv_tilt",
    "hhi",
]
