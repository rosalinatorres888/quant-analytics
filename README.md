# quant-analytics

Python library for realized risk metrics and factor analysis across an 11-asset universe.

**Requires Python ≥ 3.10** (uses PEP 604 union types).

**Assets:** AAPL NVDA MSFT TSLA GOOGL SPY QQQ VTI BTC ETH SOL

**Interview framing:** Signal generation and outcome tracking live in an LLM-driven dashboard; risk and factor analytics live here — a Python library with tested, documented methodology.

---

## Structure

```
quant-analytics/
├── quant/
│   ├── data.py          # yfinance + CoinGecko fetching, CSV caching, calendar alignment
│   ├── volatility.py    # Realized vol (21d/63d annualized), VIX context, term structure
│   ├── correlation.py   # Rolling Pearson correlation matrix, avg pairwise corr
│   ├── drawdown.py      # Max drawdown, duration, underwater curves, portfolio drawdown
│   └── factors.py       # Beta to SPY (OLS), VUG/VTV tilt, HHI concentration
├── tests/               # pytest unit tests — all use synthetic data, no network calls
├── notebooks/
│   └── risk_factor_analysis.ipynb  # Narrative analysis with stated assumptions
├── data/cache/          # CSV price cache (gitignored — auto-created on first run)
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook notebooks/risk_factor_analysis.ipynb
```

First run fetches prices from yfinance and CoinGecko and caches to `data/cache/`.
Subsequent runs load from cache. Delete a CSV to force re-fetch.

```python
from quant.data import get_prices, align_returns, ALL_ASSETS
from quant.volatility import realized_vol
from quant.drawdown import max_drawdown, drawdown_table
from quant.factors import factor_exposures, hhi

prices = get_prices(start='2022-01-01')
returns = align_returns(prices)

# Annualized realized vol
vol = realized_vol(returns, window=21)

# Max drawdown per asset
dd = drawdown_table(prices)

# Market beta (OLS on SPY)
betas = factor_exposures(returns, returns['SPY'])
```

---

## Run Tests

```bash
pytest tests/ -v
```

All tests use synthetic data with known mathematical properties (e.g., constant returns → zero vol, diagonal correlation matrix → zero avg pairwise corr).

---

## Key Assumptions

| Assumption | Choice |
|---|---|
| Return type | Daily log returns: ln(P_t / P_{t-1}) |
| Annualization | √252 (equity calendar anchor) |
| Short vol window | 21 trading days (≈1 month) |
| Long vol window | 63 trading days (≈1 quarter) |
| Correlation window | 63 trading days |
| Beta estimation | OLS, full sample, no risk-free rate |
| Calendar alignment | Forward-fill crypto ≤5 days, anchor to SPY trading days |
| Missing data | Forward-fill ≤5 consecutive NaN; drop remaining |

---

## Non-Goals (Per PRD)

- No options pricing / Greeks
- No live trading integration
- Factor models are proxy-based (ETF spreads), not Fama-French downloads (v2)

---

## Related Projects

- **Career Intelligence System** — semantic job matching + resume generation ([repo](https://github.com/rosalinatorres888/career-intelligence-system))
- **ARIA** — autonomous 7-stage career intelligence pipeline ([repo](https://github.com/rosalinatorres888/aria-career-assistant))

**Author:** Rosalina Torres — MS Data Analytics Engineering @ Northeastern University
