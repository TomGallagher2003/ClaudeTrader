# ClaudeTrader - Development Guidelines

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.11+ |
| Trading API | alpaca-py | Latest |
| AI API | anthropic | Latest |
| Testing | pytest | Latest |
| CI/CD | GitHub Actions | v4 |

## API Endpoints

### Alpaca Paper Trading
```
Base URL: https://paper-api.alpaca.markets
Data URL: https://data.alpaca.markets
```

**CRITICAL:** Never use the live trading endpoint. Always verify `paper=True` in client initialization.

### Anthropic
```
Model: claude-opus-4-20250514
Max Tokens: 500 (allows detailed reasoning)
```

## Trading Logic

### Position Sizing
- Positions are expressed as **percentage of total portfolio equity**
- Claude returns a `target_allocation_pct` (0-100%)
- System calculates shares to buy/sell to reach target
- HOLD if target differs by <1% from current allocation

### Decision Flow
1. Fetch portfolio state (equity, cash, current position)
2. Fetch 14-day OHLCV data for SPY
3. Send to Claude with full portfolio context
4. Parse target allocation from JSON response
5. Calculate shares needed to reach target
6. Execute BUY/SELL/HOLD accordingly

### Safeguards
- Cannot exceed buying power
- Cannot sell more shares than owned
- Defaults to HOLD on parse errors
- Logs all decisions with reasoning

## Development Rules

### 1. Testing
- **Always use unit tests for logic.**
- Mock all external API calls in tests
- Test edge cases: market closed, API failures, invalid responses
- Run tests before every commit: `pytest tests/ -v`

### 2. Trading
- **Strictly follow the Alpaca Paper Trading base URL.**
- Never execute real trades without explicit authorization
- Default to HOLD on any ambiguous signal
- Log every trade decision to `trades.csv`

### 3. Progress Tracking
- **Update `PROGRESS.md` after every successful script run.**
- Include timestamp, action taken, and result
- Commit trades.csv changes in GitHub Action

### 4. Error Handling
- Catch and log all API exceptions
- Exit gracefully on failures (exit code 0 to not fail workflow)
- Never crash silently - always log the reason

### 5. Code Quality
- Type hints on all functions
- Docstrings for public functions
- No hardcoded credentials (use environment variables)

## Environment Variables

```bash
ALPACA_API_KEY      # Alpaca API key
ALPACA_SECRET_KEY   # Alpaca secret key
ANTHROPIC_API_KEY   # Anthropic API key
```

## File Conventions

| File | Purpose | Update Frequency |
|------|---------|------------------|
| trader.py | Core logic | On code changes |
| trades.csv | Trade log | Every execution |
| PROGRESS.md | Build status | After milestones |
| tests/*.py | Unit tests | With code changes |

## CSV Log Format

| Column | Description |
|--------|-------------|
| timestamp | UTC ISO timestamp |
| symbol | Trading symbol (SPY) |
| action | BUY, SELL, HOLD, or ERROR |
| shares | Number of shares traded |
| price | Current share price |
| prev_alloc_pct | Allocation % before trade |
| target_alloc_pct | Target allocation % |
| equity | Total portfolio equity |
| reasoning | Claude's reasoning (truncated) |
| result | Execution result message |

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Dry run (mocked)
python -m pytest tests/test_trader.py -v -k "dry_run"

# Manual execution (requires env vars)
python trader.py
```

## Commit Message Format

```
[component] action: description

Examples:
[trader] feat: add portfolio-aware sizing
[tests] fix: mock Alpaca API correctly
[workflow] chore: update cron schedule
```
