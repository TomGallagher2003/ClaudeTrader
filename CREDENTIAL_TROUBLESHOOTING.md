# Credential Troubleshooting Guide

## Quick Fix Steps

### 1. Run the Verification Script

```bash
python verify_credentials.py
```

This will tell you exactly what's wrong with your credentials.

### 2. Create Your `.env` File (if not exists)

```bash
# Copy the example
cp .env.example .env

# Edit with your real credentials
# Use notepad, vim, or any text editor
notepad .env
```

### 3. Get New Alpaca API Keys

If your keys are invalid:

1. Go to [https://alpaca.markets/](https://alpaca.markets/)
2. Log in to your account
3. Navigate to: **Paper Trading** (for testing) or **Live Trading**
4. Click on **"Your API Keys"** or **"API Keys"** section
5. You'll see your keys:
   - **API Key ID** (this is your `ALPACA_API_KEY`)
   - **Secret Key** (this is your `ALPACA_SECRET_KEY`)
6. If keys don't work, click **"Regenerate Keys"** (WARNING: This invalidates old keys)

### 4. Update Your `.env` File

Your `.env` file should look like this:

```bash
# Alpaca API Configuration
ALPACA_API_KEY=PK1234567890ABCDEF      # Your actual key
ALPACA_SECRET_KEY=abcdef1234567890     # Your actual secret
ALPACA_PAPER=true                       # 'true' for paper trading

# Anthropic API Configuration
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx   # Your Claude API key

# AI Model Selection (optional)
CLAUDE_MODEL=claude-sonnet-4-20250514
```

### 5. Test Again

```bash
python verify_credentials.py
```

If it passes, you can run the bot:

```bash
python trader.py
```

## Common Issues

### Issue 1: "ALPACA_API_KEY not found"
**Solution:** You don't have a `.env` file. Create one using step 2 above.

### Issue 2: "401 Authentication Error"
**Causes:**
- Keys are invalid or expired
- Keys were regenerated (old keys are revoked)
- You copied the keys incorrectly (extra spaces, incomplete)
- You're using live keys but `ALPACA_PAPER=true` (or vice versa)

**Solution:**
1. Double-check you copied the FULL key (no spaces, no truncation)
2. Verify you're using the right mode (paper vs live)
3. Generate new keys if needed

### Issue 3: "Market is closed"
**Not an error!** The bot only runs during market hours (9:30 AM - 4:00 PM ET, Monday-Friday).

**To check market status:**
```bash
python verify_credentials.py
```
It will show market status in the output.

### Issue 4: GitHub Actions Failing
If running via GitHub Actions:

1. Go to your repository on GitHub
2. Click: **Settings** > **Secrets and variables** > **Actions**
3. Add these **Repository Secrets**:
   - `ALPACA_API_KEY` - Your Alpaca API key
   - `ALPACA_SECRET_KEY` - Your Alpaca secret key
   - `ALPACA_PAPER` - Set to `true`
   - `ANTHROPIC_API_KEY` - Your Claude API key

**Important:** After adding secrets, re-run the GitHub Action.

## Testing Your Setup Step-by-Step

### Step 1: Check .env exists
```bash
# Windows
dir .env

# Mac/Linux
ls -la .env
```

### Step 2: Check .env contents (sanitized view)
```bash
# Windows
type .env

# Mac/Linux
cat .env
```

Make sure:
- Keys don't have quotes around them
- No extra spaces before/after the `=`
- No blank lines between variables
- Keys are not the placeholder values (like `your_alpaca_api_key_here`)

### Step 3: Run verification
```bash
python verify_credentials.py
```

### Step 4: If successful, run the bot
```bash
python trader.py
```

## Still Having Issues?

### Debug: Check what Python is loading

Create a test file `check_env.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

print("ALPACA_API_KEY:", os.getenv('ALPACA_API_KEY', 'NOT SET'))
print("ALPACA_SECRET_KEY:", os.getenv('ALPACA_SECRET_KEY', 'NOT SET'))
print("ALPACA_PAPER:", os.getenv('ALPACA_PAPER', 'NOT SET'))
```

Run it:
```bash
python check_env.py
```

This shows you what Python is actually loading.

### Get Fresh Keys

If nothing works:
1. Go to Alpaca dashboard
2. **Regenerate** your API keys (this invalidates old ones)
3. Copy the NEW keys
4. Update your `.env` file
5. Run `python verify_credentials.py` again

## Need More Help?

Check the logs:
- Local: `trader.log`
- GitHub Actions: Go to Actions tab > Click the failed run > View logs

The error messages will tell you exactly what's wrong.
