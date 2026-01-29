# ClaudeTrader

Autonomous AI-powered trading bot using Alpaca API and Claude for market analysis.

## Overview

ClaudeTrader is a quantitative trading system optimized for the 2026 market regime. It combines rule-based filters with AI-driven analysis to make trading decisions.

## Architecture

```
trader.py
├── Configuration (Config, load_symbols)
├── Data Layer (MarketData)
├── Strategy Filters (StrategyFilters)
├── AI Analysis (AIAnalyzer)
├── Trade Execution (TradeExecutor)
└── Main Loop (run_trading_cycle)
```

## Core Strategy Rules

### 1. Regime Detection
- Monitors SPY 5-day return
- If SPY < -2% over 5 days: DEFENSIVE mode (no new buys)
- Otherwise: OFFENSIVE mode (full trading)

### 2. Relative Strength Filter
- Compares each stock's 14-day return vs QQQ
- Only allows BUY if stock is outperforming benchmark
- Prevents buying laggards/falling knives

### 3. Volatility Sizing (ATR)
- Calculates 30-day Average True Range
- If ATR > 5% of price: reduces position by 50%
- Maintains consistent risk across different volatility regimes

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

```
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_PAPER=true
ANTHROPIC_API_KEY=your_anthropic_key
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

- `trader.py` - Main trading bot
- `symbols.json` - Portfolio configuration
- `trades.csv` - Trade log (auto-generated)
- `trader.log` - Application logs

## Risk Controls

- Max 15% of portfolio in single position
- Min 10% cash reserve
- 8% hard stop loss
- Defensive mode blocks new buys in downtrends
