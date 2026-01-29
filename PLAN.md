# ClaudeTrader 2026 Optimization Plan

## Executive Summary
This document outlines the strategic enhancements for the Alpaca trading bot optimized for the 2026 market regime. The plan introduces regime detection, relative strength filtering, and volatility-based position sizing to maximize alpha while managing downside risk.

---

## Phase 1: 2026 Market Research Summary

### Portfolio Tickers & Current Performance (January 2026)

| Ticker | Company | Sector | YTD Performance | Analyst Consensus |
|--------|---------|--------|-----------------|-------------------|
| **NVDA** | NVIDIA | Semiconductors | +5.39% (1W) | Strong Buy (PT: $255.82, +37% upside) |
| **AVGO** | Broadcom | Semiconductors | +58% (12M vs NVDA +44%) | Buy - Strong AI backlog |
| **ANET** | Arista Networks | Networking | +15.97% (1W), +33.82% (1Y) | Overweight (PT: $159) |
| **LLY** | Eli Lilly | Pharma/Biotech | Trading at $1,043 | Buy (PT: $1,251.50) |
| **PLTR** | Palantir | AI/Defense | +120% (1Y), -20% from ATH | Hold (PT: $190.06) |
| **MSFT** | Microsoft | Big Tech | ~$481.63 | Strong Buy |
| **AXON** | Axon Enterprise | Defense Tech | -4% (2025), $636 | Strong Buy (PT: $800.23, +26% upside) |
| **SPY** | S&P 500 ETF | Index | Regime Indicator | N/A |

### Key 2026 Themes
1. **AI Infrastructure Boom**: NVDA, AVGO, ANET benefiting from continued AI capex
2. **Healthcare Innovation**: LLY's GLP-1 drugs driving pharmaceutical dominance
3. **Defense Tech**: AXON, PLTR positioned for government AI contracts
4. **Volatility Regime**: Post-ATH pullbacks in PLTR (-20%), AXON (-27%) create entry opportunities

---

## Phase 2: Strategic Rules Implementation

### Rule 1: Regime Detection (SPY-Based Market Filter)

**Logic:**
```
IF SPY 5-day return < -2%:
    MODE = "DEFENSIVE"
    ALLOWED_ACTIONS = [SELL, HOLD]
ELSE:
    MODE = "OFFENSIVE"
    ALLOWED_ACTIONS = [BUY, SELL, HOLD]
```

**Rationale:** When the broad market is in correction mode, individual stock picks have higher failure rates due to systematic selling pressure. Switching to defensive mode preserves capital.

**Implementation:**
- Fetch SPY close prices for T-5 to T
- Calculate: `spy_5d_return = (spy_close_today - spy_close_5d_ago) / spy_close_5d_ago`
- If `spy_5d_return < -0.02`: Block all BUY signals

---

### Rule 2: Relative Strength Filter (vs QQQ Benchmark)

**Logic:**
```
stock_14d_return = (stock_price_today - stock_price_14d_ago) / stock_price_14d_ago
qqq_14d_return = (qqq_price_today - qqq_price_14d_ago) / qqq_price_14d_ago

IF stock_14d_return > qqq_14d_return:
    relative_strength = "OUTPERFORMING"
    BUY_ELIGIBLE = True
ELSE:
    relative_strength = "UNDERPERFORMING"
    BUY_ELIGIBLE = False
```

**Rationale:** Stocks exhibiting relative strength vs the tech benchmark (QQQ) demonstrate institutional accumulation and momentum. Buying underperformers = catching falling knives.

**Implementation:**
- Fetch 14-day historical data for both stock and QQQ
- Compare returns before any BUY decision
- Include relative strength ratio in AI prompt context

---

### Rule 3: Volatility-Based Position Sizing (ATR Filter)

**Logic:**
```
atr_30d = Average True Range over 30 days
atr_percent = atr_30d / current_price

IF atr_percent > 0.05 (5%):
    position_multiplier = 0.5  # Reduce size by 50%
ELSE:
    position_multiplier = 1.0  # Full position
```

**True Range Calculation:**
```
TR = max(
    high - low,
    abs(high - previous_close),
    abs(low - previous_close)
)
ATR = SMA(TR, 30)
```

**Rationale:** High-ATR stocks (>5% daily range) carry elevated risk. Reducing position size maintains consistent portfolio risk exposure across different volatility regimes.

**Implementation:**
- Fetch 31 days of OHLC data
- Calculate daily True Range
- Compute 30-day ATR
- Apply 0.5x multiplier if ATR% > 5%

---

## Phase 3: Architecture Design

### trader.py Module Structure

```
trader.py
├── Configuration
│   ├── SYMBOLS = ["NVDA", "AVGO", "ANET", "LLY", "PLTR", "MSFT", "AXON"]
│   ├── BENCHMARK = "QQQ"
│   └── REGIME_INDICATOR = "SPY"
│
├── Data Functions
│   ├── get_historical_bars(symbol, days)
│   ├── calculate_return(prices, days)
│   ├── calculate_atr(ohlc_data, period=30)
│   └── get_current_positions()
│
├── Strategy Filters
│   ├── check_regime_mode()          # Returns "OFFENSIVE" or "DEFENSIVE"
│   ├── check_relative_strength()    # Returns True/False
│   └── calculate_position_size()    # Returns multiplier (0.5 or 1.0)
│
├── AI Analysis
│   ├── build_analysis_prompt()      # Includes regime, RS, ATR context
│   └── get_ai_recommendation()      # Claude API call
│
└── Execution
    ├── execute_trade()
    ├── log_trade()
    └── main()
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         MARKET OPEN                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: REGIME CHECK                                           │
│  - Fetch SPY 5-day data                                         │
│  - Calculate 5-day return                                       │
│  - Set MODE = DEFENSIVE if return < -2%                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: FOR EACH SYMBOL IN PORTFOLIO                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2a: RELATIVE STRENGTH CHECK                               │
│  - Fetch stock 14-day return                                    │
│  - Fetch QQQ 14-day return                                      │
│  - If stock < QQQ: Skip BUY consideration                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2b: VOLATILITY SIZING                                     │
│  - Calculate 30-day ATR                                         │
│  - If ATR% > 5%: Apply 0.5x position multiplier                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2c: AI ANALYSIS                                           │
│  - Build prompt with regime, RS, ATR context                    │
│  - Get Claude recommendation: BUY/SELL/HOLD                     │
│  - Apply regime filter (block BUY if DEFENSIVE)                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: EXECUTE TRADES                                         │
│  - Submit orders via Alpaca API                                 │
│  - Log to trades.csv                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 4: Implementation Checklist

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `trader.py` | CREATE | Main trading bot with all strategic rules |
| `symbols.json` | CREATE | Portfolio configuration file |
| `requirements.txt` | CREATE | Dependencies (alpaca-trade-api, anthropic, pandas) |
| `tests/test_strategy.py` | CREATE | Unit tests for strategy functions |
| `CLAUDE.md` | CREATE | Project documentation |
| `STRATEGY_REPORT.md` | CREATE | Backtest results for NVDA/LLY |

### Dependencies

```
alpaca-trade-api>=3.0.0
anthropic>=0.18.0
pandas>=2.0.0
python-dotenv>=1.0.0
pytest>=7.0.0
```

---

## Phase 5: Risk Controls

### Position Limits
- Max 15% of portfolio in any single position
- Max 40% in any single sector (Semiconductors)
- Cash reserve: Minimum 10% always

### Stop-Loss Rules
- Hard stop: -8% from entry
- Trailing stop: -5% from peak (after +10% gain)

### Defensive Mode Triggers
1. SPY 5-day return < -2% (primary)
2. VIX > 25 (optional enhancement)
3. Portfolio drawdown > -5% intraday

---

## Approval Checkpoint

**Before proceeding with implementation, please confirm:**

1. [ ] Portfolio tickers approved: NVDA, AVGO, ANET, LLY, PLTR, MSFT, AXON, SPY
2. [ ] Regime Detection threshold: -2% SPY over 5 days
3. [ ] Relative Strength benchmark: QQQ (14-day lookback)
4. [ ] ATR volatility threshold: 5% triggers 50% position reduction
5. [ ] Position sizing and risk controls acceptable

---

## Sources

- [Top Semiconductor Stocks for 2026 - MarketBeat](https://www.marketbeat.com/stock-ideas/beyond-nvidia-5-semiconductor-stocks-set-to-dominate-2026/)
- [NVDA Stock Analysis](https://stockanalysis.com/stocks/nvda/)
- [Arista Networks 2026 Outlook - Piper Sandler](https://finance.yahoo.com/news/why-piper-sandler-sees-2026-220143024.html)
- [Eli Lilly Price Prediction - 24/7 Wall St](https://247wallst.com/forecasts/2026/01/12/eli-lilly-lly-stock-price-prediction-and-forecast-2025-2030/)
- [Palantir Stock Analysis - Finbold](https://finbold.com/ai-predicts-palantir-stock-price-for-january-31-2026/)
- [Axon 2026 Outlook - Motley Fool](https://www.fool.com/investing/2026/01/08/is-axon-axon-stock-a-buy-in-2026/)
- [Chip Stocks Rally - CNBC](https://www.cnbc.com/2026/01/28/global-chip-stocks-today-nvidia-china-asml-sk-hynix.html)

---

*Document Version: 1.0*
*Last Updated: January 29, 2026*
*Status: AWAITING APPROVAL*
