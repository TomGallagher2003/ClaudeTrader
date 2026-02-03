# ClaudeTrader
Autonomous Claude trader using alpaca api

## Quick Start

### GitHub Actions Setup (Automated Trading)

To run the bot automatically via GitHub Actions:

1. **Configure GitHub Secrets** (required for authentication)
   - See [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md) for detailed instructions
   - Add: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER`, `ANTHROPIC_API_KEY`

2. **Workflow runs automatically**
   - Scheduled: Weekdays at 9:30 AM ET (market open)
   - Manual: Go to Actions tab → Run workflow

### Local Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API credentials
# Then run:
python trader.py
```

## Troubleshooting

**Getting 401 Unauthorized errors?**
→ See [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md) for credential setup

**Full Documentation:**
- [CLAUDE.md](CLAUDE.md) - Strategy details and architecture
- [STRATEGY_ASSESSMENT.md](STRATEGY_ASSESSMENT.md) - Analysis and roadmap
