#!/usr/bin/env python3
"""
ClaudeTrader - Autonomous Paper Trading Bot
Uses Alpaca for execution and Anthropic Claude for decision-making.
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import anthropic
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# Constants
SYMBOL = "SPY"
LOOKBACK_DAYS = 14
TRADE_QTY = 1
CSV_FILE = Path(__file__).parent / "trades.csv"

Decision = Literal["BUY", "SELL", "HOLD"]


def get_alpaca_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    """Initialize Alpaca clients for paper trading."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_client = StockHistoricalDataClient(api_key, secret_key)

    return trading_client, data_client


def get_anthropic_client() -> anthropic.Anthropic:
    """Initialize Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set")

    return anthropic.Anthropic(api_key=api_key)


def fetch_ohlcv_data(data_client: StockHistoricalDataClient) -> str:
    """Fetch last 14 days of OHLCV data for SPY and format as string."""
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )

    bars = data_client.get_stock_bars(request)

    if SYMBOL not in bars.data or not bars.data[SYMBOL]:
        raise ValueError(f"No data returned for {SYMBOL}")

    # Format bars as readable text
    lines = [f"Last {LOOKBACK_DAYS} days of {SYMBOL} daily OHLCV data:"]
    lines.append("Date | Open | High | Low | Close | Volume")
    lines.append("-" * 60)

    for bar in bars.data[SYMBOL]:
        lines.append(
            f"{bar.timestamp.strftime('%Y-%m-%d')} | "
            f"${bar.open:.2f} | ${bar.high:.2f} | ${bar.low:.2f} | "
            f"${bar.close:.2f} | {bar.volume:,}"
        )

    return "\n".join(lines)


def get_claude_decision(client: anthropic.Anthropic, market_data: str) -> Decision:
    """Ask Claude to analyze market data and return BUY/SELL/HOLD decision."""
    prompt = f"""You are a trading assistant. Analyze the following market data and make a decision.

{market_data}

Based on this price action, respond ONLY with 'BUY', 'SELL', or 'HOLD'.
Reasoning must be under 20 words after your decision.

Format: DECISION: [reasoning]"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip().upper()

    # Parse decision from response
    if response_text.startswith("BUY"):
        return "BUY"
    elif response_text.startswith("SELL"):
        return "SELL"
    else:
        # Default to HOLD for any ambiguous response
        return "HOLD"


def check_market_open(trading_client: TradingClient) -> bool:
    """Check if the market is currently open."""
    clock = trading_client.get_clock()
    return clock.is_open


def get_current_position(trading_client: TradingClient) -> int:
    """Get current SPY position quantity. Returns 0 if no position."""
    try:
        position = trading_client.get_open_position(SYMBOL)
        return int(position.qty)
    except Exception:
        return 0


def execute_trade(
    trading_client: TradingClient, decision: Decision, current_position: int
) -> str:
    """Execute trade based on decision. Returns result message."""
    if decision == "HOLD":
        return "HOLD - No action taken"

    if decision == "BUY":
        if current_position > 0:
            return f"SKIP BUY - Already holding {current_position} shares"

        order_request = MarketOrderRequest(
            symbol=SYMBOL,
            qty=TRADE_QTY,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_request)
        return f"BUY ORDER SUBMITTED - Order ID: {order.id}"

    if decision == "SELL":
        if current_position <= 0:
            return "SKIP SELL - No position to sell"

        order_request = MarketOrderRequest(
            symbol=SYMBOL,
            qty=min(TRADE_QTY, current_position),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_request)
        return f"SELL ORDER SUBMITTED - Order ID: {order.id}"

    return "UNKNOWN DECISION"


def log_trade(decision: Decision, result: str) -> None:
    """Log trade to CSV file."""
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "decision", "result"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            SYMBOL,
            decision,
            result,
        ])


def main() -> int:
    """Main entry point for the trading bot."""
    print(f"ClaudeTrader starting at {datetime.utcnow().isoformat()} UTC")
    print(f"Target symbol: {SYMBOL}")
    print("-" * 50)

    try:
        # Initialize clients
        trading_client, data_client = get_alpaca_clients()
        anthropic_client = get_anthropic_client()

        # Check if market is open
        if not check_market_open(trading_client):
            print("Market is closed. Exiting.")
            log_trade("HOLD", "Market closed - no action")
            return 0

        # Fetch market data
        print("Fetching market data...")
        market_data = fetch_ohlcv_data(data_client)
        print(market_data)
        print("-" * 50)

        # Get Claude's decision
        print("Asking Claude for decision...")
        decision = get_claude_decision(anthropic_client, market_data)
        print(f"Claude's decision: {decision}")

        # Check current position
        current_position = get_current_position(trading_client)
        print(f"Current position: {current_position} shares")

        # Execute trade
        print("Executing trade...")
        result = execute_trade(trading_client, decision, current_position)
        print(f"Result: {result}")

        # Log trade
        log_trade(decision, result)
        print(f"Trade logged to {CSV_FILE}")

        print("-" * 50)
        print("ClaudeTrader completed successfully")
        return 0

    except Exception as e:
        error_msg = f"ERROR: {type(e).__name__}: {e}"
        print(error_msg)
        log_trade("HOLD", error_msg)
        # Return 0 to not fail the GitHub Action
        return 0


if __name__ == "__main__":
    sys.exit(main())
