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
Model: claude-sonnet-4-20250514
Max Tokens: 50 (decisions should be terse)
```

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
- Keep trader.py under 200 lines

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
[trader] feat: add OHLCV data fetching
[tests] fix: mock Alpaca API correctly
[workflow] chore: update cron schedule
```
