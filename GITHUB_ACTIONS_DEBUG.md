# GitHub Actions Credential Debugging Guide

## Problem
Your credentials work locally (`python verify_credentials.py` passes), but GitHub Actions jobs are failing with "401 Authentication Error".

## Why This Happens

**GitHub Actions doesn't use your local `.env` file!**

GitHub Actions runs in a clean Ubuntu container and gets credentials from **Repository Secrets**, which you configure in GitHub's web interface.

## Step-by-Step Fix

### Step 1: Run the Credential Verification Workflow

1. Go to your GitHub repository
2. Click on **Actions** tab
3. Click on **Verify Credentials** workflow (left sidebar)
4. Click **Run workflow** button (top right)
5. Wait for it to complete
6. Check the output - it will tell you exactly which secrets are missing/invalid

### Step 2: Check Your Repository Secrets

1. Go to your GitHub repository
2. Click **Settings** tab
3. Click **Secrets and variables** > **Actions** (left sidebar)
4. Under "Repository secrets", you should see:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - `ALPACA_PAPER`
   - `ANTHROPIC_API_KEY`

**If any are missing, add them!**

### Step 3: Verify Secret Values

**Common mistakes:**

#### ❌ Wrong Key Format
```
# DON'T include quotes or any extra characters
ALPACA_API_KEY="PK1234567890"  # WRONG
ALPACA_API_KEY=PK1234567890    # WRONG (this is how .env looks, but not how GitHub secrets work)
```

#### ✅ Correct Format in GitHub Secrets
In GitHub's "New secret" form:
- **Name:** `ALPACA_API_KEY`
- **Value:** `PK1234567890` (just the key itself, no quotes, no `ALPACA_API_KEY=` prefix)

#### ❌ Truncated Keys
Make sure you copy the **entire** key. It's easy to accidentally miss characters when copying.

Your keys should be:
- **ALPACA_API_KEY**: Starts with `PK`, around 20-30 characters
- **ALPACA_SECRET_KEY**: Around 40-50 characters
- **ALPACA_PAPER**: Exactly `true` or `false` (lowercase)
- **ANTHROPIC_API_KEY**: Starts with `sk-ant-api03-`, around 100+ characters

### Step 4: Update Secrets (If Needed)

To update a secret:
1. Go to **Settings** > **Secrets and variables** > **Actions**
2. Click on the secret name
3. Click **Update secret**
4. Paste the new value
5. Click **Update secret**

**DO NOT** add extra spaces, quotes, or the variable name (like `ALPACA_API_KEY=`)

### Step 5: Re-run the Trading Bot

After updating secrets:
1. Go to **Actions** tab
2. Click **Run Trading Bot** workflow
3. Click **Run workflow**
4. Wait and check the results

## Detailed Troubleshooting

### Issue 1: "ALPACA_API_KEY is NOT set in repository secrets"

**Cause:** The secret doesn't exist or is named incorrectly.

**Fix:**
1. Check secret name is **exactly**: `ALPACA_API_KEY` (case-sensitive!)
2. No spaces, no typos
3. If wrong, delete it and create a new one with the correct name

### Issue 2: "401 Authentication Error" even though secrets are set

**Causes:**
1. **Wrong API keys in GitHub secrets** (different from your local `.env`)
2. **Truncated keys** (didn't copy the full key)
3. **Extra spaces** before/after the key value
4. **Using live keys with paper=true** or vice versa

**Fix:**
1. Go to [Alpaca dashboard](https://alpaca.markets/)
2. Log in
3. Navigate to **Paper Trading** section
4. Click **View API Keys** or **Your API Keys**
5. **Copy the exact keys** (make sure you get ALL characters)
6. Update GitHub secrets with these exact values

**Pro tip:** Generate new keys to ensure they're fresh:
1. In Alpaca dashboard, click **Regenerate Keys**
2. Copy the new keys immediately
3. Update both your local `.env` AND GitHub secrets

### Issue 3: Secrets are set but still getting credential errors

**Possible causes:**
1. **Alpaca is rate-limiting you** - wait a few minutes and try again
2. **Alpaca API is down** - check [Alpaca status page](https://status.alpaca.markets/)
3. **IP restrictions** - GitHub Actions IPs might be blocked (unlikely for paper trading)

**Debug:**
Run the verification workflow to see the exact error message.

### Issue 4: "Market is closed" - Not Running

This is **NOT an error!** The bot only runs during market hours:
- **Monday - Friday**
- **9:30 AM - 4:00 PM ET**

To test outside market hours, you can temporarily modify the code (not recommended for production).

## How to Test Your Setup

### Test 1: Verify Secrets Exist
```bash
# Run the verification workflow
# Go to: Actions > Verify Credentials > Run workflow
```

This will show:
- ✅ Which secrets are configured
- ✅ Which are missing
- ✅ Whether they're valid

### Test 2: Manual Trading Bot Run
```bash
# Go to: Actions > Run Trading Bot > Run workflow
```

Click on the job to see detailed logs. Look for:
- `✓ Alpaca credentials validated` - Good!
- `AUTHENTICATION ERROR: Failed to validate` - Bad, check secrets

### Test 3: Check Recent Workflow Runs
```bash
# Go to: Actions tab
# Click on the most recent failed run
# Click on "trade" job
# Expand "Run trading bot" step
# Read the error message
```

The error will tell you exactly what's wrong.

## Visual Guide: Adding Secrets in GitHub

1. **Settings** tab (top of repository page)
2. **Secrets and variables** (left sidebar)
3. **Actions** (under Secrets and variables)
4. **New repository secret** button
5. Enter:
   - Name: `ALPACA_API_KEY` (exactly as shown)
   - Value: `PK...` (your actual key, no quotes)
6. Click **Add secret**
7. Repeat for all 4 secrets

## Expected Secrets Configuration

Your GitHub secrets should look like this:

| Name | Value Example | Notes |
|------|---------------|-------|
| `ALPACA_API_KEY` | `PKDJCW3GR7HMD5N6MOBIP4IBU5` | Starts with PK |
| `ALPACA_SECRET_KEY` | `DkCJ2v8MPwrbxF3RxXQBRznnVFTiB7...` | Long string |
| `ALPACA_PAPER` | `true` | Lowercase, no quotes |
| `ANTHROPIC_API_KEY` | `sk-ant-api03-_7kyj269GQw9sIw...` | Starts with sk-ant-api03- |

**Important:** These are just the VALUES. Don't include the `=` or variable names.

## Common Mistakes Checklist

- [ ] Secret names are **exactly** as shown (case-sensitive)
- [ ] No quotes around values
- [ ] No `VARIABLE_NAME=` prefix in the value
- [ ] Copied the **entire** key (no truncation)
- [ ] No extra spaces before/after the value
- [ ] Using **paper trading** keys (not live) if `ALPACA_PAPER=true`
- [ ] Keys are from the correct Alpaca account

## Still Not Working?

### Generate Fresh Keys

1. Go to [Alpaca dashboard](https://alpaca.markets/)
2. Navigate to **Paper Trading** > **API Keys**
3. Click **Regenerate Keys** (this will invalidate old keys!)
4. **Immediately copy** both keys
5. Update:
   - Local `.env` file
   - GitHub repository secrets
6. Run verification workflow again

### Check Workflow Logs

The error message in GitHub Actions will tell you exactly what's wrong:
- `ALPACA_API_KEY is NOT set` → Secret doesn't exist
- `401 Authentication Error` → Invalid credentials
- `Market is closed` → Not an error, just not market hours

### Compare Local vs GitHub

**Local (working):**
```bash
python verify_credentials.py
```

**GitHub (failing):**
```bash
# Actions > Verify Credentials > Run workflow
```

If local works but GitHub fails, the issue is **definitely** with your repository secrets configuration.

## Need More Help?

1. Run the **Verify Credentials** workflow
2. Copy the full output
3. Check which specific step is failing
4. The error message will guide you to the exact problem

Remember: The most common issue is that the GitHub secrets don't match your local `.env` file, or they were entered incorrectly (with quotes, truncated, etc.).
