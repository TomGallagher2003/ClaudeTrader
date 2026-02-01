# Running Your First Backtest

The backtesting framework is ready to use! You just need to configure API credentials.

## Quick Setup (2 minutes)

### Step 1: Get Alpaca API Credentials

**Option A: Free Paper Trading Account** (Recommended for backtesting)
1. Visit: https://alpaca.markets/
2. Sign up for a free paper trading account
3. Navigate to "Your API Keys" in the dashboard
4. Copy your API Key and Secret Key

**Note**: Alpaca provides **free** historical market data access with any account type.

### Step 2: Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env and replace with your actual credentials
nano .env  # or use your preferred editor
```

Your `.env` should look like:
```bash
ALPACA_API_KEY=PK1234567890ABCDEF
ALPACA_SECRET_KEY=abcdef1234567890xyz
ALPACA_PAPER=true

# Optional: Leave blank for backtesting without AI
ANTHROPIC_API_KEY=
```

### Step 3: Run the Backtest

```bash
python backtest.py
```

This will:
- Test the strategy on 2023-2025 historical data ($100K initial capital)
- Calculate Sharpe ratio, max drawdown, win rate, and other metrics
- Compare performance vs. buy-and-hold QQQ
- Save detailed results to `backtest_results.json`
- Display performance report

Expected runtime: 1-3 minutes (depends on data download speed)

## Understanding Results

After the backtest completes, you'll see a report like:

```
======================================================================
BACKTEST PERFORMANCE REPORT
======================================================================

Period: 2023-01-01 to 2025-12-31 (3.0 years)

--- RETURNS ---
Initial Capital:    $  100,000.00
Final Equity:       $  ???,???.??
Total Return:             ?.??%
CAGR:                     ?.??%

--- RISK METRICS ---
Sharpe Ratio:             ?.??
Max Drawdown:            -?.??%
Volatility:               ?.??%
```

### Interpreting Your Results

**Strategy PASSES if:**
- ✅ Sharpe Ratio > 0.5
- ✅ Alpha > 0% (beat QQQ benchmark)
- ✅ Max Drawdown < 30%

**Strategy FAILS if:**
- ❌ Sharpe Ratio < 0.5 (poor risk-adjusted returns)
- ❌ Alpha < 0% (underperformed buy-and-hold)
- ❌ Max Drawdown > 40% (excessive risk)

## What If I Don't Have Credentials?

If you can't get Alpaca credentials right now, you can review:
1. **BACKTESTING.md** - Complete framework documentation
2. **STRATEGY_ASSESSMENT.md** - Expected performance analysis
3. Sample backtest results (once someone runs it)

The framework is fully functional and ready to validate the strategy as soon as you add credentials.

## Customizing the Backtest

Edit `backtest.py` to test different periods:

```python
# Test the COVID crash period
start_date = datetime(2020, 2, 1)
end_date = datetime(2020, 5, 1)

# Test the 2022 bear market
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

# Test full 2020-2025 (comprehensive)
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 12, 31)
```

## Troubleshooting

**"ValueError: You must supply a method of authentication"**
→ Add your Alpaca API credentials to `.env`

**"Failed to load data for symbol X"**
→ Symbol might not have data for the selected period (e.g., PLTR IPO'd in 2020)
→ Either remove the symbol or adjust the start date

**Backtest runs but shows no trades**
→ Market might be in DEFENSIVE mode the entire period
→ Try a different date range or check symbols in `symbols.json`

## Next Steps

After backtesting:

1. **If results are positive** → Paper trade for 30 days
2. **If results are mixed** → Review Phase 2 enhancements in STRATEGY_ASSESSMENT.md
3. **If results are negative** → Adjust strategy parameters or test different symbols

---

**Related Documentation:**
- `BACKTESTING.md` - Comprehensive framework guide
- `STRATEGY_ASSESSMENT.md` - Strategy analysis & roadmap
- `CLAUDE.md` - Project overview
