# ClaudeTrader

Autonomous AI-powered trading bot using Alpaca API and Claude for market analysis.

## Overview

ClaudeTrader is a quantitative trading system optimized for the 2026 market regime. It combines rule-based filters with AI-driven analysis to make trading decisions. The system features comprehensive risk management, portfolio optimization, and cost-efficient AI model selection.

## ⚡ AGGRESSIVE MODE (Current Configuration)

**Status**: Backtest revealed severe underperformance (26.69% vs QQQ's 134.19%) due to overly conservative filters. The bot went defensive in May 2025 and never re-entered, missing the entire rally.

**Aggressive Mode Changes**:
- **Regime Filter**: Relaxed from -5% to -15% SPY threshold (only go defensive in major crashes)
- **Relative Strength Filter**: DISABLED for minor underperformance (only blocks if >15% worse than QQQ)
- **Position Sizing**: Increased from 10% → 15% base, 15% → 25% max
- **Cash Reserve**: Reduced from 10% → 2% (stay fully invested)
- **Volatility Multiplier**: Changed from 50% → 90% (only 10% size reduction for high volatility)
- **Stop Loss**: Widened from 12% → 20% (let winners run, accommodate normal volatility)

**Goal**: Match or beat QQQ benchmark returns by staying invested during bull markets while still protecting against true bear markets.

## Implementation Status

✅ **Tier 1 (Critical Fixes)**: Complete
- Fixed ATR calculation
- Implemented stop loss enforcement
- Added position context to AI prompts

✅ **Tier 2 (Strategy Enhancements)**: Complete
- Multi-timeframe analysis (5d, 14d, 30d)
- Entry signal generation (momentum, MA crossover, RSI, volume)
- Profit-taking rules (scale out at +15%, trailing stops)
- Expanded AI analysis with technical indicators

✅ **Tier 3 (Advanced Features)**: Complete
- AI model configuration (Opus/Sonnet/Haiku for cost control)
- Portfolio optimization with correlation analysis
- Dynamic universe screening (quantitative)
- News sentiment analysis (placeholder for future API integration)

## Architecture

```
trader.py
├── Configuration (Config, load_symbols, load_universe)
├── Data Layer (MarketData)
│   ├── Historical data fetching
│   ├── Technical indicators (ATR, RSI, SMA)
│   └── Volume analysis
├── Strategy Filters (StrategyFilters)
│   ├── Regime Detection
│   ├── Relative Strength
│   ├── Volatility Sizing
│   ├── Multi-timeframe Analysis
│   ├── Entry Signal Generation
│   └── Profit-taking Rules
├── AI Analysis (AIAnalyzer)
├── News Sentiment (NewsSentimentAnalyzer) [Placeholder]
├── Universe Screening (UniverseScreener)
├── Portfolio Optimization (PortfolioOptimizer)
├── Trade Execution (TradeExecutor)
└── Main Loop (run_trading_cycle)
```

## Core Strategy Rules (AGGRESSIVE MODE)

### 1. Regime Detection (RELAXED)
- Monitors SPY 5-day return
- If SPY < -15% over 5 days: DEFENSIVE mode (no new buys) ⚡ **CHANGED from -5%**
- Otherwise: OFFENSIVE mode (full trading)
- **Result**: Only goes defensive in major market crashes, not minor pullbacks

### 2. Relative Strength Filter (MOSTLY DISABLED)
- Compares each stock's 14-day return vs QQQ
- ⚡ **AGGRESSIVE MODE**: Minor underperformance is ALLOWED
- Only blocks BUY if stock underperforms by >15%
- **Result**: Can buy during recoveries even if lagging benchmark slightly

### 3. Volatility Sizing (ATR) (RELAXED)
- Calculates 30-day Average True Range
- If ATR > 5% of price: reduces position by only 10% ⚡ **CHANGED from 50%**
- **Result**: Maintains larger positions even in volatile stocks

### 4. Entry Signal Generation (Tier 2)
- **Momentum Flip**: Short-term outperformance vs benchmark
- **MA Crossover**: Price above 20-day SMA after decline
- **RSI Recovery**: RSI in 30-60 range (oversold recovery)
- **Volume Confirmation**: Elevated volume (>1.5x average)
- Signal strength: STRONG (3+ signals), MODERATE (2), WEAK (1)

### 5. Profit-Taking Rules (Tier 2)
- **Scale Out**: Sell 50% at +15% gain
- **Trailing Stop**: 5% from highs after +10% gain
- **Stop Loss**: Hard exit at -20% loss ⚡ **CHANGED from -12%** (wider stops)

### 6. Portfolio Optimization (Tier 3)
- Analyzes position concentration (flags positions >25%) ⚡ **CHANGED from >15%**
- Calculates correlation matrix between holdings
- Identifies high correlation pairs (>0.7)
- AI-powered risk assessment and rebalancing recommendations

### 7. Universe Screening (Tier 3)
- Weekly scans of broader stock universe
- Quantitative scoring: Relative strength (40pts) + Momentum (30pts) + RSI (20pts) + Volume (10pts)
- Identifies rotation candidates and emerging opportunities
- Zero additional API cost (pure quantitative)

## Portfolio

Current high-conviction 2026 tickers:
- **NVDA** - AI/GPU leader
- **AVGO** - AI infrastructure
- **ANET** - AI networking
- **LLY** - GLP-1 pharma leader
- **PLTR** - AI/Defense software
- **MSFT** - Big tech AI play
- **AXON** - Defense tech

## Setup

### Environment Variables

Create a `.env` file:

```bash
# Alpaca API Configuration
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_PAPER=true

# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_key

# AI Model Selection (optional - defaults to sonnet for cost efficiency)
# Options:
# - claude-opus-4-5-20251101 (Premium: ~$0.075/call)
# - claude-sonnet-4-20250514 (Balanced: ~$0.015/call) [DEFAULT]
# - claude-haiku-4-20250514 (Fast & Cheap: ~$0.003/call)
CLAUDE_MODEL=claude-sonnet-4-20250514
```

### Configuration Options

The bot behavior can be customized by editing the `Config` class in `trader.py`:

```python
# Tier 3 Feature Toggles
enable_news_analysis: bool = True          # Enable news/sentiment (placeholder)
enable_portfolio_optimization: bool = True  # Enable portfolio correlation checks
enable_universe_screening: bool = False     # Enable weekly universe screening
```

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python trader.py
```

## Testing

```bash
pytest tests/ -v
```

## Files

- `trader.py` - Main trading bot (1300+ lines)
- `symbols.json` - Portfolio configuration (active trading symbols)
- `universe.json` - Broader universe for screening (70+ symbols)
- `trades.csv` - Trade log (auto-generated)
- `latest_decisions.json` - Most recent job decisions (auto-generated)
- `trader.log` - Application logs
- `STRATEGY_ASSESSMENT.md` - Comprehensive strategy analysis and roadmap

## Risk Controls (AGGRESSIVE MODE)

- Max 25% of portfolio in single position ⚡ **(was 15%)**
- Min 2% cash reserve ⚡ **(was 10%)**
- 20% hard stop loss (enforced on every cycle) ⚡ **(was 12%)**
- 5% trailing stop after +10% gains
- Defensive mode blocks new buys only at -15% SPY ⚡ **(was -5%)**
- Relative strength filter disabled for minor underperformance
- Portfolio correlation monitoring
- Concentration risk alerts

## API Cost Optimization

**Before Tier 3** (using Opus for all calls):
- 7 symbols × ~$0.075/call = ~$0.53/day = ~$16/month

**After Tier 3** (using Sonnet by default):
- 7 symbols × ~$0.015/call = ~$0.11/day = ~$3.30/month
- Portfolio optimization: +~$0.02/day
- **Total: ~$3.50/month**
- **Savings: ~80% cost reduction**

**With Haiku** (for maximum cost savings):
- 7 symbols × ~$0.003/call = ~$0.02/day = ~$0.60/month
- **Savings: ~96% cost reduction**

Set `CLAUDE_MODEL` environment variable to control costs vs. quality trade-off.
