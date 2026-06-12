"""
Price fetching and caching for the 11-asset universe.

Sources:
    Equities/ETFs: yfinance (no API key)
    Crypto (BTC/ETH/SOL): CoinGecko free API (no key)

Cache: CSV files at data/cache/<TICKER>.csv (DatetimeIndex + 'close' column).
Delete a file to force a re-fetch on next run.

Assumptions:
    - Return type: daily log returns (ln(P_t / P_{t-1}))
    - Calendar: equity trading days used as the anchor (≈252/year).
      Crypto trades 7d/week — forward-filled to equity calendar, then
      weekend/holiday rows dropped via SPY anchor.
    - Missing data: forward-fill at most 5 consecutive NaN days (handles
      short holiday gaps); rows with remaining NaN dropped before return calc.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"

EQUITY_TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "SPY", "QQQ", "VTI"]
FACTOR_TICKERS = ["^VIX", "VUG", "VTV"]  # implied-vol proxy + growth/value
CRYPTO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
ALL_ASSETS = EQUITY_TICKERS + list(CRYPTO_IDS.keys())

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("^", "_")
    return CACHE_DIR / f"{safe}.csv"


def _load_cache(ticker: str) -> pd.Series | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty or "close" not in df.columns:
        return None
    s = df["close"].rename(ticker)
    s.index.name = "date"
    return s


def _save_cache(ticker: str, series: pd.Series) -> None:
    series.to_frame("close").to_csv(_cache_path(ticker))


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _fetch_equity(ticker: str, start: str) -> pd.Series:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r}")
    close = df["Close"].squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    close.name = ticker
    return close


def _fetch_crypto(ticker: str, days: int = 730) -> pd.Series:
    cg_id = CRYPTO_IDS[ticker]
    url = f"{_COINGECKO_BASE}/coins/{cg_id}/market_chart"
    resp = requests.get(
        url,
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["prices"]  # [[epoch_ms, price], ...]
    series = pd.Series(
        {pd.Timestamp(ts, unit="ms").normalize(): p for ts, p in raw},
        name=ticker,
    )
    series.index.name = "date"
    # CoinGecko sometimes includes today's partial candle — drop it
    series = series.iloc[:-1]
    return series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prices(
    tickers: list[str] | None = None,
    extra_tickers: list[str] | None = None,
    start: str = "2022-01-01",
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily closing prices.

    Parameters
    ----------
    tickers : list of str, optional
        Asset symbols to fetch. Defaults to ALL_ASSETS (11-asset universe).
    extra_tickers : list of str, optional
        Additional equity/ETF tickers (e.g. FACTOR_TICKERS) appended to tickers.
    start : str
        Start date for history, YYYY-MM-DD. Default: 2022-01-01.
    refresh : bool
        If True, bypass cache and re-fetch all data.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex × tickers, sorted ascending.
    """
    if tickers is None:
        tickers = ALL_ASSETS
    if extra_tickers:
        tickers = tickers + [t for t in extra_tickers if t not in tickers]

    series_list = []
    for ticker in tickers:
        cached = None if refresh else _load_cache(ticker)
        if cached is not None:
            series_list.append(cached)
            continue

        if ticker in CRYPTO_IDS:
            series = _fetch_crypto(ticker)
            series = series[series.index >= pd.Timestamp(start)]
        else:
            series = _fetch_equity(ticker, start)

        _save_cache(ticker, series)
        series_list.append(series)
        time.sleep(0.25)  # polite rate-limiting

    df = pd.concat(series_list, axis=1).sort_index()
    df.index.name = "date"
    return df


def align_returns(prices: pd.DataFrame, max_fill: int = 5) -> pd.DataFrame:
    """
    Compute daily log returns aligned to equity trading days.

    Crypto gaps (weekends/holidays) are forward-filled up to `max_fill`
    consecutive days before computing returns. Rows where SPY (the equity
    calendar anchor) is NaN are dropped, then log returns are computed and
    the first row (always NaN) is discarded.

    Parameters
    ----------
    prices : pd.DataFrame
        Output of get_prices() — mixed equity + crypto closes.
    max_fill : int
        Maximum consecutive NaN days to forward-fill. Default: 5.

    Returns
    -------
    pd.DataFrame
        Daily log returns, equity-calendar aligned, no NaN rows.
    """
    filled = prices.ffill(limit=max_fill)

    # Anchor to equity trading days via SPY
    anchor = "SPY" if "SPY" in filled.columns else filled.columns[0]
    filled = filled[filled[anchor].notna()]

    log_ret = np.log(filled / filled.shift(1))
    return log_ret.iloc[1:].dropna(how="all")
