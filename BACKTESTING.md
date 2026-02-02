# Backtesting Framework

Comprehensive backtesting system for validating ClaudeTrader strategy performance on historical data.

## Overview

The backtesting framework simulates the ClaudeTrader strategy on historical market data to:
- Validate strategy profitability before live trading
- Measure risk-adjusted returns (Sharpe ratio, max drawdown)
- Model realistic transaction costs (slippage, spread, fees)
- Compare performance vs. buy-and-hold benchmark (QQQ)
- Calculate win rate, average win/loss, and other trading statistics

## Key Features

### 1. Transaction Cost Modeling
Realistic cost modeling prevents overstated backtest returns:
- **Bid-Ask Spread**: 0.03% (typical for liquid stocks)
- **Slippage**: 0.02% market impact on orders
- **SEC Fees**: $0.000008 per dollar
- **Commission**: $0 (Alpaca is commission-free)
- **Total Round-Trip Cost**: ~0.07%

### 2. Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns (target: >1.0)
- **CAGR**: Compound Annual Growth Rate  
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation
- **Win Rate**: Percentage of winning trades
- **Avg Win/Loss**: Mean return per trade type

### 3. Strategy Implementation
The backtester replicates live trading logic:
- Regime detection (SPY 5-day momentum filter)
- Relative strength filtering (vs QQQ benchmark)
- Entry signal generation (4 independent signals)
- Stop loss enforcement (-8% hard stop)
- Profit-taking rules (+15% scale-out)
- Position sizing (10% base, volatility-adjusted)

## Quick Start

### Basic Usage

```python
from backtest import Backtester
from datetime import datetime

# Initialize backtester
backtester = Backtester(
    initial_capital=100000.0,
    use_ai=False  # Set False to avoid API costs
)

# Run backtest
performance = backtester.run_backtest(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 12, 31)
)

# Print results
backtester.print_performance_report(performance)
```

### Run from Command Line

```bash
# Run default backtest (2023-2025)
python backtest.py

# Results saved to backtest_results.json
```

### Customize Transaction Costs

```python
from backtest import Backtester, TransactionCosts

# Custom cost model (more conservative)
costs = TransactionCosts(
    spread_pct=0.0005,  # 0.05% spread (less liquid stocks)
    slippage_pct=0.0003,  # 0.03% slippage
)

backtester = Backtester(
    initial_capital=100000.0,
    transaction_costs=costs
)
```

### Backtest Specific Symbols

```python
from trader import Config

# Custom symbol list
config = Config()
config.symbols = ["NVDA", "MSFT", "AAPL"]

backtester = Backtester(config=config)
performance = backtester.run_backtest(start_date, end_date)
```

## Interpreting Results

### Sample Output

```
======================================================================
BACKTEST PERFORMANCE REPORT
======================================================================

Period: 2023-01-01 to 2025-12-31 (3.0 years)

--- RETURNS ---
Initial Capital:    $  100,000.00
Final Equity:       $  125,430.50
Total Return:             25.43%
CAGR:                      7.85%

--- RISK METRICS ---
Sharpe Ratio:              0.89
Max Drawdown:            -12.45%
Volatility:               18.23%

--- TRADING STATS ---
Total Trades:                 42
Win Rate:                  45.24%
Avg Win:                   12.50%
Avg Loss:                  -6.80%

--- COSTS ---
Total Tx Costs:     $    1,234.56
Cost per Trade:     $       29.39

--- BENCHMARK COMPARISON ---
Benchmark (QQQ):          22.34% (6.95% CAGR)
Strategy:                 25.43% (7.85% CAGR)
Alpha:                     0.90%
```

### What to Look For

**Good Signs:**
- ✅ Sharpe ratio > 1.0 (excellent risk-adjusted returns)
- ✅ Alpha > 0% (outperforming benchmark)
- ✅ Max drawdown < 20% (manageable risk)
- ✅ Win rate 40-60% with avg win > avg loss

**Red Flags:**
- ❌ Sharpe ratio < 0.5 (poor risk-adjusted returns)
- ❌ Alpha < 0% (underperforming buy-and-hold)
- ❌ Max drawdown > 40% (excessive risk)
- ❌ Transaction costs > 50% of gains (over-trading)

## Backtest Limitations

### What the Backtest Does NOT Account For
1. **Market Impact**: Large orders may move prices more than modeled
2. **Liquidity Gaps**: Assumes all stocks are equally liquid
3. **Dividend Adjustments**: Does not include dividend income
4. **Overnight Gaps**: Only uses daily close prices
5. **Execution Delays**: Assumes instant fills at close price
6. **Regime Changes**: Past performance ≠ future results
7. **Overfitting Risk**: Strategy may be optimized for historical data

### Best Practices
- Test on multiple time periods (bull, bear, sideways markets)
- Use out-of-sample data (don't optimize on test period)
- Add 50% buffer to transaction costs for safety
- Reduce backtest Sharpe by 20-30% for live trading expectations
- Paper trade for 30+ days before going live

## Advanced Usage

### Run Multiple Scenarios

```python
scenarios = [
    ("2020-2022", datetime(2020,1,1), datetime(2022,12,31)),  # COVID crash + recovery
    ("2023-2024", datetime(2023,1,1), datetime(2024,12,31)),  # Recent bull market
    ("2022 only", datetime(2022,1,1), datetime(2022,12,31)),  # Bear market
]

for name, start, end in scenarios:
    print(f"\n=== Scenario: {name} ===")
    backtester = Backtester(initial_capital=100000)
    performance = backtester.run_backtest(start, end)
    backtester.print_performance_report(performance)
```

### Export Results for Analysis

```python
import json

performance = backtester.run_backtest(start_date, end_date)

# Save full results
with open("backtest_results.json", "w") as f:
    json.dump(performance, f, indent=2, default=str)

# Extract equity curve for plotting
equity_curve = performance["equity_curve"]
dates = [point["date"] for point in equity_curve]
values = [point["equity"] for point in equity_curve]

# Use matplotlib, plotly, etc. to visualize
```

## Recommended Backtest Periods

### Minimum Validation
- **3 months**: Quick sanity check (too short for significance)
- **1 year**: Captures seasonal effects, basic validation
- **3 years**: Recommended minimum (multiple market regimes)

### Comprehensive Validation  
- **5+ years**: Captures full market cycle
- **2008-2025**: Includes financial crisis, QE era, COVID, recent AI boom
- **2020-2025**: Recent data most relevant to current regime

### Critical Periods to Test
- **March 2020**: COVID crash (-35% SPY in 30 days)
- **2022**: Bear market (QQQ -33% for year)
- **2023-2024**: AI-driven bull market

## Performance Targets

Based on STRATEGY_ASSESSMENT.md analysis:

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| CAGR | <5% | 5-10% | 10-15% | >15% |
| Sharpe | <0.5 | 0.5-1.0 | 1.0-2.0 | >2.0 |
| Max DD | >40% | 20-40% | 10-20% | <10% |
| Alpha | <0% | 0-2% | 2-5% | >5% |
| Win Rate | <35% | 35-45% | 45-55% | >55% |

**Minimum to Trade Live:**
- Sharpe > 0.5
- Alpha > 0%
- Max DD < 30%
- 30+ day paper trading period with positive results

## Troubleshooting

### "Missing data for symbol X"
- Symbol may have IPO'd after backtest start date
- Remove symbol or adjust start date

### "No trades executed"
- Check date range (need 35+ days of data for indicators)
- Verify symbols are in config
- Review regime detection (may be in DEFENSIVE mode entire period)

### Backtest too slow
- Reduce date range
- Test with fewer symbols
- Set `use_ai=False` (no API calls)

### Transaction costs seem high
- This is realistic! 0.07% round-trip × 100 trades/year = 7% drag
- Reduce trading frequency if costs exceed 5% of gains
- Consider limit orders in live trading (not modeled in backtest)

## Next Steps

After running backtests:

1. **If backtest PASSES (Sharpe > 0.5, Alpha > 0%):**
   - Paper trade for 30 days
   - Monitor slippage vs. backtest assumptions
   - Start live with small capital (<$10K)

2. **If backtest FAILS (Sharpe < 0.5 or Alpha < 0%):**
   - Review STRATEGY_ASSESSMENT.md for improvements
   - Consider Phase 2 enhancements (Kelly sizing, multi-regime)
   - Test different symbols/universes

3. **Ongoing Validation:**
   - Re-run quarterly with latest data
   - Compare live results to backtest expectations
   - Adjust parameters if market regime shifts

## API Cost Note

Setting `use_ai=False` makes backtesting **FREE** (no Anthropic API calls). The backtester uses pure quantitative rules to replicate strategy logic. This is recommended for initial testing.

For AI-powered backtesting (more realistic but costly), set `use_ai=True` and expect ~$0.015 per symbol per day.

---

**Related Files:**
- `backtest.py` - Main backtesting engine
- `trader.py` - Live trading strategy
- `STRATEGY_ASSESSMENT.md` - Strategy analysis & roadmap
- `CLAUDE.md` - Project overview

**Phase 1 Status (from STRATEGY_ASSESSMENT.md):**
- ✅ Build backtesting framework
- ✅ Add transaction cost modeling
- ✅ Implement performance metrics
- ⏳ Backtest on 2020-2025 data (ready to run)
- ⏳ Paper trade 30 days (after backtest validation)
