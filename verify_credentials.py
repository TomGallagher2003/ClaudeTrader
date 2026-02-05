#!/usr/bin/env python3
"""
Quick credential verification script for ClaudeTrader.
Run this to test your API credentials before running the full bot.
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()

def verify_alpaca_credentials():
    """Test Alpaca API credentials."""
    print("=" * 60)
    print("ALPACA CREDENTIAL VERIFICATION")
    print("=" * 60)

    # Check if environment variables are set
    alpaca_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
    alpaca_paper = os.getenv('ALPACA_PAPER', 'true')

    print("\n1. Checking environment variables...")
    if not alpaca_key or alpaca_key == 'your_alpaca_api_key_here':
        print("   ❌ ALPACA_API_KEY is not set or still has default value")
        print("   → Edit your .env file and add your real API key")
        return False
    else:
        print(f"   ✓ ALPACA_API_KEY is set (length: {len(alpaca_key)})")

    if not alpaca_secret or alpaca_secret == 'your_alpaca_secret_key_here':
        print("   ❌ ALPACA_SECRET_KEY is not set or still has default value")
        print("   → Edit your .env file and add your real secret key")
        return False
    else:
        print(f"   ✓ ALPACA_SECRET_KEY is set (length: {len(alpaca_secret)})")

    print(f"   ✓ ALPACA_PAPER is set to: {alpaca_paper}")

    # Test API connection
    print("\n2. Testing API connection...")
    try:
        client = TradingClient(
            api_key=alpaca_key,
            secret_key=alpaca_secret,
            paper=alpaca_paper.lower() == 'true'
        )

        # Try to get account info
        account = client.get_account()

        print("   ✓ Successfully connected to Alpaca API!")
        print("\n3. Account Information:")
        print(f"   Account Number: {account.account_number}")
        print(f"   Status: {account.status}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")

        # Check market status
        clock = client.get_clock()
        print(f"\n4. Market Status:")
        print(f"   Market is: {'OPEN' if clock.is_open else 'CLOSED'}")
        print(f"   Next Open: {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")

        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED - Your credentials are valid!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"   ❌ API Connection Failed!")
        print(f"\n   Error Details: {e}")
        print("\n   Possible issues:")
        print("   1. API keys are invalid or expired")
        print("   2. API keys were regenerated (old keys revoked)")
        print("   3. Network connectivity issues")
        print("   4. Wrong paper/live mode setting")
        print("\n   To fix:")
        print("   1. Go to: https://alpaca.markets/")
        print("   2. Navigate to your API Keys section")
        print("   3. Verify your keys or generate new ones")
        print("   4. Update your .env file with the correct keys")
        print("\n" + "=" * 60)
        return False

def verify_anthropic_credentials():
    """Test Anthropic API credentials."""
    print("\n" + "=" * 60)
    print("ANTHROPIC CREDENTIAL VERIFICATION")
    print("=" * 60)

    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if not anthropic_key or anthropic_key == 'your_anthropic_api_key_here':
        print("   ⚠️  ANTHROPIC_API_KEY is not set or still has default value")
        print("   → AI analysis will not work without this")
        print("   → Get your key from: https://console.anthropic.com/")
        return False
    else:
        print(f"   ✓ ANTHROPIC_API_KEY is set (length: {len(anthropic_key)})")
        print("   Note: Not testing API connection to avoid charges")
        return True

def main():
    """Run all credential checks."""
    print("\nClaudeTrader Credential Verification Tool\n")

    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ ERROR: .env file not found!")
        print("\nTo fix:")
        print("1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("2. Edit .env and add your real API credentials")
        print("3. Run this script again")
        return

    alpaca_valid = verify_alpaca_credentials()
    anthropic_valid = verify_anthropic_credentials()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Alpaca API: {'✅ Valid' if alpaca_valid else '❌ Invalid'}")
    print(f"Anthropic API: {'✅ Valid' if anthropic_valid else '❌ Invalid'}")

    if alpaca_valid:
        print("\n✅ You're ready to run the trading bot!")
        print("   Run: python trader.py")
    else:
        print("\n❌ Fix the credential issues above before running the bot.")

if __name__ == "__main__":
    main()
