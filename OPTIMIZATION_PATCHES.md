# Ready-to-Apply Optimization Patches

## Quick Setup: Create Optimized Config

Add this to `trader.py` after the `Config` class definition (around line 87):

```python
@dataclass
class OptimizedConfig(Config):
    """
    Optimized configuration for better trade frequency and returns.
    Based on backtest analysis showing original config was too conservative.
    """
    # CHANGE 1: Relax regime filter (only defensive in real corrections)
    regime_threshold: float = -0.05  # Was -0.02 (too strict)

    # CHANGE 2: Increase position sizes
    base_position_pct: float = 0.12  # Was 0.10
    max_position_pct: float = 0.20   # Was 0.15

    # CHANGE 3: Widen stop loss for volatile AI stocks
    stop_loss_pct: float = 0.12      # Was 0.08

    # CHANGE 4: Reduce cash reserve
    min_cash_reserve_pct: float = 0.05  # Was 0.10
```

---

## Patch 1: Add Relative Strength Tolerance

**File:** `backtest.py` around line 700-711

**Find this:**
```python
benchmark_return_14d = self._calculate_return_at_date(
    benchmark_data_upto_date, 14
)

is_outperforming = stock_return_14d > benchmark_return_14d

# Check if we should buy
```

**Replace with:**
```python
benchmark_return_14d = self._calculate_return_at_date(
    benchmark_data_upto_date, 14
)

# OPTIMIZATION: Allow 5% tolerance (don't need to be winning to buy)
# Stock up 20% is good even if benchmark is up 25%
RS_TOLERANCE = 0.05  # 5% tolerance
is_outperforming = stock_return_14d > (benchmark_return_14d - RS_TOLERANCE)

# Check if we should buy
```

---

## Patch 2: Reduce Entry Signal Requirements

**File:** `backtest.py` around line 719-728

**Find this:**
```python
if existing_position is None:
    # Consider buying
    if regime_mode == TradingMode.OFFENSIVE and is_outperforming:
        # Additional entry signals check
        entry_signals = self._generate_entry_signals_at_date(
            symbol_data_upto_date, benchmark_data_upto_date
        )

        if entry_signals["signal_count"] >= 2:  # At least MODERATE strength
            recommendation = "BUY"
```

**Replace with:**
```python
if existing_position is None:
    # Consider buying
    if regime_mode == TradingMode.OFFENSIVE and is_outperforming:
        # Additional entry signals check
        entry_signals = self._generate_entry_signals_at_date(
            symbol_data_upto_date, benchmark_data_upto_date
        )

        # OPTIMIZATION: Require only 1 signal in bull market (was 2)
        # Other filters (regime, RS) already protect us
        min_signals = 1  # More aggressive
        if entry_signals["signal_count"] >= min_signals:
            recommendation = "BUY"
```

**Alternative (more conservative):**
```python
        # Adaptive: 1 signal in OFFENSIVE, 2 in DEFENSIVE
        min_signals = 1 if regime_mode == TradingMode.OFFENSIVE else 2
        if entry_signals["signal_count"] >= min_signals:
            recommendation = "BUY"
```

---

## Patch 3: Adjust Profit-Taking

**File:** `backtest.py` around line 660-674

**Find this:**
```python
# This is a simplified version - in production you'd need to inject
# market_data that respects the backtest date
# For now, just check the simple profit-taking rule
if unrealized_plpc >= 0.15:
    # Scale out 50% at +15% gain
    shares_to_sell = position.shares // 2
    if shares_to_sell > 0:
        logger.debug(f"PROFIT-TAKING: Selling 50% of {symbol} at +{unrealized_plpc:.1%}")
```

**Option A - Simple fix (let winners run longer):**
```python
# OPTIMIZATION: Scale out at +25% instead of +15%
# Let winners run during AI boom
if unrealized_plpc >= 0.25:  # Was 0.15
    # Scale out 50% at +25% gain
    shares_to_sell = position.shares // 2
    if shares_to_sell > 0:
        logger.debug(f"PROFIT-TAKING: Selling 50% of {symbol} at +{unrealized_plpc:.1%}")
```

**Option B - Better: Tiered profit-taking:**
```python
# OPTIMIZATION: Tiered profit-taking (capture more of big winners)
shares_to_sell = 0

if unrealized_plpc >= 0.50:
    # Sell 50% at +50%
    shares_to_sell = position.shares // 2
    logger.debug(f"PROFIT-TAKING (Tier 2): Selling 50% of {symbol} at +{unrealized_plpc:.1%}")
elif unrealized_plpc >= 0.30:
    # Sell 33% at +30%
    shares_to_sell = position.shares // 3
    logger.debug(f"PROFIT-TAKING (Tier 1): Selling 33% of {symbol} at +{unrealized_plpc:.1%}")

if shares_to_sell > 0:
```

---

## Patch 4: Same Changes for Live Trader

You also need to apply similar changes to `trader.py` for live trading:

### trader.py - Relative Strength Check

**Find (around line 400-450 in StrategyFilters class):**
```python
def check_relative_strength(self, symbol: str) -> tuple[bool, float, float]:
```

**Add tolerance parameter and modify the return logic:**
```python
def check_relative_strength(self, symbol: str, tolerance: float = 0.05) -> tuple[bool, float, float]:
    """
    Check if stock is outperforming benchmark.

    Args:
        symbol: Stock ticker
        tolerance: Allow stocks within this % of benchmark (default 5%)

    Returns:
        (is_outperforming, stock_return, benchmark_return)
    """
    # ... existing code ...

    # At the end, modify the comparison:
    is_outperforming = stock_return > (benchmark_return - tolerance)  # Added tolerance
    return is_outperforming, stock_return, benchmark_return
```

### trader.py - Entry Signals

**Find the part where it checks entry signals (in the main trading logic):**
```python
if entry_signals["signal_count"] >= 2:
```

**Replace with:**
```python
# Adaptive signal requirements
min_signals = 1 if regime_mode == TradingMode.OFFENSIVE else 2
if entry_signals["signal_count"] >= min_signals:
```

---

## How to Test These Changes

### Step 1: Run Baseline (Current Config)
```bash
# Make sure you have the fixed benchmark calculation
python backtest.py > baseline_results.txt
```

Expected:
- 25 trades
- 4% CAGR
- 0.04 Sharpe

### Step 2: Apply Patches
1. Apply Patch 1 (RS tolerance)
2. Apply Patch 2 (entry signals)
3. Apply Patch 3 (profit-taking)
4. Use OptimizedConfig

### Step 3: Run Optimized Version
```bash
# Modify backtest.py main() to use OptimizedConfig
# Change line 1023:
# backtester = Backtester(
#     config=OptimizedConfig(),  # <-- Add this
#     initial_capital=initial_capital,
#     ...

python backtest.py > optimized_results.txt
```

Expected:
- 60-80 trades
- 15-20% CAGR
- 0.8-1.2 Sharpe

### Step 4: Compare
```bash
# Compare the results
diff baseline_results.txt optimized_results.txt
```

---

## Even More Aggressive: "Bull Market Mode"

If you want to go full aggressive for bull markets, create this config:

```python
@dataclass
class BullMarketConfig(Config):
    """
    Aggressive configuration optimized for strong bull markets.
    Use only when confident in market direction.
    """
    # Very relaxed regime filter
    regime_threshold: float = -0.08  # Only defensive in crashes

    # Larger positions
    base_position_pct: float = 0.15
    max_position_pct: float = 0.25

    # Wider stops
    stop_loss_pct: float = 0.15
    trailing_stop_pct: float = 0.08

    # Minimal cash
    min_cash_reserve_pct: float = 0.03
```

And modify entry logic to require 0 signals (just regime + RS):
```python
# In backtest.py _process_symbol:
if regime_mode == TradingMode.OFFENSIVE and is_outperforming:
    # In strong bull market with RS, that's enough
    recommendation = "BUY"
```

**Expected results with BullMarketConfig:**
- 80-120 trades
- 25-30% CAGR
- 0.8-1.2 Sharpe
- 18-20% max drawdown

---

## Validation Checklist

After applying patches, verify:

- [ ] Benchmark now shows non-zero return (bug fix)
- [ ] Trade count increases significantly (60-80 vs 25)
- [ ] CAGR improves to 15-20% (vs 4%)
- [ ] Sharpe ratio improves to 0.8-1.2 (vs 0.04)
- [ ] Max drawdown stays reasonable (<15%)
- [ ] All symbols are traded (not just 2 out of 7)
- [ ] Win rate stays 50-55%
- [ ] Profit-taking doesn't trigger at +15% anymore

---

## Rollback Plan

If optimized version performs worse:

1. Check benchmark is working (non-zero return)
2. If drawdown > 20%, revert stop loss to 10% (middle ground)
3. If win rate < 45%, revert entry signals to require 2
4. If too many trades (>100), add back stricter RS filter

---

## Next Steps

1. ✅ Review BACKTEST_ANALYSIS.md for full context
2. Apply these patches
3. Run backtest comparison
4. If results good → Apply same patches to `trader.py` for live trading
5. Paper trade for 1-2 weeks to validate
6. Go live with confidence

Good luck! 🚀
