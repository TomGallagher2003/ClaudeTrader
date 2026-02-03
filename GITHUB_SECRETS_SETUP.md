# GitHub Secrets Setup Guide

This guide explains how to configure GitHub repository secrets for the ClaudeTrader automated trading bot.

## The 401 Unauthorized Error

If you're seeing this error in your GitHub Actions logs:

```
alpaca.common.exceptions.APIError: {"code":40110000,"message":"request is not authorized"}
```

This means your Alpaca API credentials are either:
- Not configured in GitHub Secrets
- Invalid or expired
- Incorrectly formatted

## Step-by-Step Fix

### 1. Get Your Alpaca API Credentials

1. Log in to your Alpaca account at [https://alpaca.markets/](https://alpaca.markets/)
2. Navigate to **Your API Keys** section
3. For paper trading (recommended for testing):
   - Select **Paper Trading** at the top
   - Generate new API keys if needed
   - Copy both the **API Key** and **Secret Key**

⚠️ **Important**: API keys are shown only once. If you've lost them, you'll need to regenerate them.

### 2. Configure GitHub Repository Secrets

1. Go to your GitHub repository
2. Click **Settings** (top menu)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret** for each of the following:

#### Required Secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `ALPACA_API_KEY` | Your Alpaca API Key | Starts with `PK...` (paper) or `AK...` (live) |
| `ALPACA_SECRET_KEY` | Your Alpaca Secret Key | Long alphanumeric string |
| `ALPACA_PAPER` | `true` or `false` | Set to `true` for paper trading |
| `ANTHROPIC_API_KEY` | Your Claude API Key | Starts with `sk-ant-...` |

#### Adding Each Secret:

1. Click **New repository secret**
2. Enter the **Name** (e.g., `ALPACA_API_KEY`)
3. Enter the **Value** (paste your key)
4. Click **Add secret**
5. Repeat for all four secrets

### 3. Verify Setup

After adding all secrets, you can verify they're configured correctly:

1. Go to **Actions** tab in your repository
2. Select the **Run Trading Bot** workflow
3. Click **Run workflow** → **Run workflow** (manual trigger)
4. Wait for the job to start
5. Check the logs for:
   ```
   ✓ Alpaca credentials validated (Account: PA...)
   ```

If you see this message, your credentials are working!

### 4. Common Issues

#### Issue: "Missing environment variables"
**Fix**: Make sure all four secrets are added in GitHub Settings

#### Issue: "Failed to validate Alpaca credentials"
**Fix**: Your API keys are invalid. Generate new ones in Alpaca dashboard and update GitHub secrets

#### Issue: Keys were regenerated
**Fix**: When you regenerate keys in Alpaca, the old keys stop working immediately. Update GitHub secrets with the new keys.

#### Issue: Using live trading keys in paper mode
**Fix**: Make sure you're using Paper Trading keys (start with `PK...`) and `ALPACA_PAPER=true`

## Testing Locally

To test the bot on your local machine:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```bash
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_secret_here
   ALPACA_PAPER=true
   ANTHROPIC_API_KEY=your_claude_key_here
   ```

3. Run the bot:
   ```bash
   python trader.py
   ```

## Security Best Practices

- ✅ **DO**: Use GitHub Secrets for credentials (never commit to code)
- ✅ **DO**: Use Paper Trading keys for testing
- ✅ **DO**: Regenerate keys if you suspect they're compromised
- ❌ **DON'T**: Share your API keys publicly
- ❌ **DON'T**: Commit `.env` file to git (it's in `.gitignore`)
- ❌ **DON'T**: Use Live Trading keys until you've thoroughly tested in paper mode

## Need Help?

If you're still having issues:

1. Check the GitHub Actions logs for detailed error messages
2. Verify your Alpaca account is active and in good standing
3. Try generating fresh API keys
4. Open an issue in this repository with the error details (never include actual API keys!)
