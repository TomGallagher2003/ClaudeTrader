# ClaudeTrader Implementation Plan

## Overview
Build an autonomous paper trading bot that:
1. Fetches SPY market data via Alpaca API
2. Uses Claude (Anthropic API) to make BUY/SELL/HOLD decisions
3. Executes paper trades on Alpaca
4. Runs daily via GitHub Actions at market open

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions (Cron)                       │
│                   Weekdays 14:30 UTC (9:30 EST)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        trader.py                                │
├─────────────────────────────────────────────────────────────────┤
│  1. Fetch 14-day OHLCV data for SPY (Alpaca)                   │
│  2. Format data → Anthropic prompt                              │
│  3. Get decision: BUY / SELL / HOLD                            │
│  4. Execute market order (Alpaca Paper)                         │
│  5. Log to trades.csv                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      trades.csv                                 │
│            (Committed back to repo for tracking)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ClaudeTrader/
├── CLAUDE.md              # Tech stack & rules
├── PROGRESS.md            # Build roadmap & status
├── PLAN.md                # This file
├── README.md              # Project overview
├── requirements.txt       # Python dependencies
├── trader.py              # Core trading logic
├── trades.csv             # Trade log (auto-updated)
├── tests/
│   └── test_trader.py     # Unit tests with mocked APIs
└── .github/
    └── workflows/
        └── trade.yml      # Daily automation
```

---

## Implementation Phases

### Phase 1: Setup & Context
- [x] Initialize repository
- [ ] Create CLAUDE.md (tech stack, rules)
- [ ] Create PROGRESS.md (roadmap)
- [ ] Create requirements.txt

### Phase 2: Core Trading Logic (`trader.py`)
- [ ] Alpaca client setup (paper trading endpoint)
- [ ] Fetch 14-day OHLCV bars for SPY
- [ ] Format data into Claude prompt
- [ ] Parse Claude response (BUY/SELL/HOLD)
- [ ] Execute market order based on decision
- [ ] Log trade to CSV

### Phase 3: Testing
- [ ] Mock Alpaca API responses
- [ ] Mock Anthropic API responses
- [ ] Test edge cases:
  - Market closed
  - API failures
  - Invalid Claude response
  - Insufficient buying power
  - Position already held (for BUY)
  - No position to sell (for SELL)

### Phase 4: Automation
- [ ] Create GitHub Action workflow
- [ ] Configure secrets injection
- [ ] Auto-commit trades.csv back to repo
- [ ] Test workflow execution

---

## Edge Cases & Handling

| Scenario | Handling |
|----------|----------|
| Market closed | Check `clock.is_open`, skip if closed |
| Alpaca API down | Catch exception, log error, exit gracefully |
| Anthropic API down | Catch exception, log error, exit gracefully |
| Invalid Claude response | Default to HOLD, log warning |
| Insufficient buying power | Log warning, skip trade |
| Already holding SPY (BUY) | Log info, skip duplicate buy |
| No SPY position (SELL) | Log info, skip invalid sell |
| Weekend/holiday run | GitHub cron handles weekdays only |

---

## Secrets Required (GitHub)

| Secret | Purpose |
|--------|---------|
| `ALPACA_API_KEY` | Alpaca paper trading API key |
| `ALPACA_SECRET_KEY` | Alpaca paper trading secret |
| `ANTHROPIC_API_KEY` | Claude API key |

---

## Trading Parameters

- **Symbol:** SPY (S&P 500 ETF)
- **Data Window:** 14 days OHLCV
- **Order Type:** Market order
- **Quantity:** 1 share per signal
- **Execution Time:** 9:30 AM EST (market open)

---

## Risk Considerations

1. **Paper Trading Only** - Uses Alpaca paper endpoint, no real money
2. **Single Share** - Minimal position sizing
3. **HOLD as default** - Conservative fallback on errors
4. **Daily logging** - Full audit trail in trades.csv

---

## Next Steps

1. Create CLAUDE.md and PROGRESS.md
2. Implement trader.py with proper error handling
3. Write comprehensive tests
4. Set up GitHub Action
5. Run dry-run to validate

