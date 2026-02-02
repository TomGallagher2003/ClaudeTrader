# Aggressive Mode Changes

## Problem Statement

Backtest results showed severe underperformance:
- **Strategy Return**: 26.69% (8.21% CAGR)
- **QQQ Benchmark**: 134.19% (32.82% CAGR)
- **Alpha**: -24.61% (underperforming by 108%)

**Root Cause**: Overly conservative filters caused the bot to go defensive in May 2025 and never re-enter, missing the entire 2025 rally.

## Solution: Aggressive Mode Configuration

### Configuration Changes (trader.py lines 60-75)

| Parameter | Conservative | Aggressive | Change |
|-----------|-------------|------------|---------|
| **Regime Threshold** | -5% SPY | -15% SPY | 3x more lenient |
| **Relative Strength Filter** | Block all underperforming | Only block >15% underperformance | Mostly disabled |
| **Base Position Size** | 10% | 15% | +50% larger |
| **Max Position Size** | 15% | 25% | +67% larger |
| **Cash Reserve** | 10% | 2% | -80% (stay invested) |
| **Volatility Multiplier** | 50% (halve position) | 90% (only 10% reduction) | 80% less reduction |
| **Stop Loss** | 12% | 20% | +67% wider |

### Code Changes

1. **trader.py lines 60-75**: Updated Config class with aggressive defaults
2. **trader.py lines 1483-1492**: Disabled relative strength blocking for minor underperformance
3. **trader.py lines 569-590**: Updated AI prompt to reflect aggressive strategy
4. **trader.py lines 1-10**: Updated docstring to document aggressive mode
5. **CLAUDE.md**: Added aggressive mode documentation and updated all strategy rules

### Expected Impact

**Benefits**:
- Stay invested during bull markets
- Capture full upside of high-conviction positions
- Avoid premature exits from normal volatility
- Match or beat QQQ benchmark returns

**Risks**:
- Larger drawdowns during market corrections
- Higher concentration risk (up to 25% per position)
- Wider stops mean larger losses on individual positions
- Less cash buffer for opportunities

### Risk Mitigation

While more aggressive, the bot still maintains:
- Stop losses (20% instead of 12%)
- Profit-taking at +15% gains
- Defensive mode for true crashes (-15% SPY)
- Portfolio correlation monitoring
- AI analysis for each trade decision

## Testing & Deployment

**Next Steps**:
1. Run backtest with aggressive configuration
2. Compare results to QQQ benchmark
3. Analyze drawdowns and risk metrics
4. Deploy to paper trading for live validation

**Success Criteria**:
- Total return closer to QQQ benchmark (>100%)
- Sharpe ratio >0.8
- Max drawdown <20%
- Win rate >40%
- Stay invested during bull markets
