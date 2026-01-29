#!/usr/bin/env python3
"""
ClaudeTrader - Autonomous Paper Trading Bot
Uses Alpaca for execution and Anthropic Claude for decision-making.
Portfolio-aware with percentage-based position sizing.
"""

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

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
CSV_FILE = Path(__file__).parent / "trades.csv"
MODEL = "claude-opus-4-20250514"
MAX_TOKENS = 500


@dataclass
class PortfolioState:
    """Current portfolio state."""
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    position_qty: int
    position_value: Decimal
    position_pct: Decimal
    current_price: Decimal


@dataclass
class TradeDecision:
    """Claude's trading decision."""
    target_pct: Decimal
    reasoning: str
    action: str  # BUY, SELL, or HOLD
    shares_to_trade: int


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


def get_portfolio_state(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
) -> PortfolioState:
    """Get current portfolio and position state."""
    # Get account info
    account = trading_client.get_account()
    equity = Decimal(str(account.equity))
    cash = Decimal(str(account.cash))
    buying_power = Decimal(str(account.buying_power))

    # Get current price
    end = datetime.now()
    start = end - timedelta(days=1)
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = data_client.get_stock_bars(request)
    if SYMBOL in bars.data and bars.data[SYMBOL]:
        current_price = Decimal(str(bars.data[SYMBOL][-1].close))
    else:
        raise ValueError(f"Could not fetch current price for {SYMBOL}")

    # Get position
    try:
        position = trading_client.get_open_position(SYMBOL)
        position_qty = int(position.qty)
        position_value = Decimal(str(position.market_value))
    except Exception:
        position_qty = 0
        position_value = Decimal("0")

    # Calculate position as percentage of equity
    position_pct = (position_value / equity * 100) if equity > 0 else Decimal("0")

    return PortfolioState(
        equity=equity,
        cash=cash,
        buying_power=buying_power,
        position_qty=position_qty,
        position_value=position_value,
        position_pct=position_pct.quantize(Decimal("0.01")),
        current_price=current_price,
    )


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
    lines.append("Date       | Open    | High    | Low     | Close   | Volume")
    lines.append("-" * 65)

    for bar in bars.data[SYMBOL]:
        lines.append(
            f"{bar.timestamp.strftime('%Y-%m-%d')} | "
            f"${bar.open:>7.2f} | ${bar.high:>7.2f} | ${bar.low:>7.2f} | "
            f"${bar.close:>7.2f} | {bar.volume:>10,}"
        )

    return "\n".join(lines)


def get_claude_decision(
    client: anthropic.Anthropic,
    market_data: str,
    portfolio: PortfolioState,
) -> TradeDecision:
    """Ask Claude to analyze market data and portfolio, return target allocation."""
    prompt = f"""You are an autonomous trading assistant managing a paper trading portfolio.
Your task is to analyze market data and current portfolio state, then decide the optimal
allocation percentage for {SYMBOL} in this portfolio.

## Current Portfolio State
- Total Equity: ${portfolio.equity:,.2f}
- Cash Available: ${portfolio.cash:,.2f}
- Buying Power: ${portfolio.buying_power:,.2f}
- Current {SYMBOL} Position: {portfolio.position_qty} shares (${portfolio.position_value:,.2f})
- Current {SYMBOL} Allocation: {portfolio.position_pct}% of portfolio
- Current {SYMBOL} Price: ${portfolio.current_price:.2f}

## Market Data
{market_data}

## Instructions
Analyze the price action, trends, and momentum. Consider:
1. Recent price trends (bullish, bearish, or sideways)
2. Volatility and trading volume patterns
3. Support and resistance levels
4. Current position and whether to increase, decrease, or maintain exposure

## Response Format
Respond with a JSON object only (no markdown, no code blocks):
{{
    "target_allocation_pct": <number 0-100>,
    "reasoning": "<your analysis in 2-3 sentences>"
}}

The target_allocation_pct is what percentage of total portfolio equity should be allocated to {SYMBOL}.
- 0 means fully exit the position (sell all)
- 100 means go all-in on {SYMBOL}
- Current allocation is {portfolio.position_pct}%

Be decisive but prudent. Consider risk management."""

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()

    # Parse JSON response
    try:
        # Try to extract JSON if wrapped in code blocks
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()

        data = json.loads(response_text)
        target_pct = Decimal(str(data.get("target_allocation_pct", 0)))
        reasoning = data.get("reasoning", "No reasoning provided")
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Warning: Could not parse Claude response: {e}")
        print(f"Raw response: {response_text}")
        # Default to current allocation (HOLD)
        target_pct = portfolio.position_pct
        reasoning = f"Parse error, maintaining current allocation. Raw: {response_text[:100]}"

    # Clamp to valid range
    target_pct = max(Decimal("0"), min(Decimal("100"), target_pct))

    # Calculate shares to trade
    target_value = portfolio.equity * target_pct / 100
    current_value = portfolio.position_value
    value_diff = target_value - current_value

    if portfolio.current_price > 0:
        shares_to_trade = int(abs(value_diff) / portfolio.current_price)
    else:
        shares_to_trade = 0

    # Determine action
    if shares_to_trade == 0 or abs(target_pct - portfolio.position_pct) < Decimal("1"):
        action = "HOLD"
        shares_to_trade = 0
    elif value_diff > 0:
        action = "BUY"
        # Ensure we don't exceed buying power
        max_affordable = int(portfolio.buying_power / portfolio.current_price)
        shares_to_trade = min(shares_to_trade, max_affordable)
    else:
        action = "SELL"
        # Ensure we don't sell more than we own
        shares_to_trade = min(shares_to_trade, portfolio.position_qty)

    return TradeDecision(
        target_pct=target_pct,
        reasoning=reasoning,
        action=action,
        shares_to_trade=shares_to_trade,
    )


def check_market_open(trading_client: TradingClient) -> bool:
    """Check if the market is currently open."""
    clock = trading_client.get_clock()
    return clock.is_open


def execute_trade(
    trading_client: TradingClient,
    decision: TradeDecision,
    portfolio: PortfolioState,
) -> str:
    """Execute trade based on decision. Returns result message."""
    if decision.action == "HOLD" or decision.shares_to_trade == 0:
        return (
            f"HOLD - Maintaining {portfolio.position_pct}% allocation "
            f"({portfolio.position_qty} shares)"
        )

    if decision.action == "BUY":
        if decision.shares_to_trade <= 0:
            return "SKIP BUY - Insufficient buying power or zero shares calculated"

        order_request = MarketOrderRequest(
            symbol=SYMBOL,
            qty=decision.shares_to_trade,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_request)
        return (
            f"BUY {decision.shares_to_trade} shares @ ~${portfolio.current_price:.2f} "
            f"(target: {decision.target_pct}%) - Order ID: {order.id}"
        )

    if decision.action == "SELL":
        if decision.shares_to_trade <= 0:
            return "SKIP SELL - No shares to sell"

        order_request = MarketOrderRequest(
            symbol=SYMBOL,
            qty=decision.shares_to_trade,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = trading_client.submit_order(order_request)
        return (
            f"SELL {decision.shares_to_trade} shares @ ~${portfolio.current_price:.2f} "
            f"(target: {decision.target_pct}%) - Order ID: {order.id}"
        )

    return "UNKNOWN ACTION"


def log_trade(
    decision: TradeDecision,
    portfolio: PortfolioState,
    result: str,
) -> None:
    """Log trade to CSV file with detailed portfolio context."""
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "symbol",
                "action",
                "shares",
                "price",
                "prev_alloc_pct",
                "target_alloc_pct",
                "equity",
                "reasoning",
                "result",
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            SYMBOL,
            decision.action,
            decision.shares_to_trade,
            float(portfolio.current_price),
            float(portfolio.position_pct),
            float(decision.target_pct),
            float(portfolio.equity),
            decision.reasoning[:200],  # Truncate for CSV
            result,
        ])


def log_error(error_msg: str) -> None:
    """Log error to CSV when no decision could be made."""
    file_exists = CSV_FILE.exists()

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "symbol",
                "action",
                "shares",
                "price",
                "prev_alloc_pct",
                "target_alloc_pct",
                "equity",
                "reasoning",
                "result",
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            SYMBOL,
            "ERROR",
            0,
            0,
            0,
            0,
            0,
            error_msg[:200],
            "Failed",
        ])


def main() -> int:
    """Main entry point for the trading bot."""
    print(f"ClaudeTrader starting at {datetime.utcnow().isoformat()} UTC")
    print(f"Target symbol: {SYMBOL}")
    print(f"Model: {MODEL}")
    print("-" * 60)

    try:
        # Initialize clients
        trading_client, data_client = get_alpaca_clients()
        anthropic_client = get_anthropic_client()

        # Check if market is open
        if not check_market_open(trading_client):
            print("Market is closed. Exiting.")
            log_error("Market closed - no action")
            return 0

        # Get portfolio state
        print("Fetching portfolio state...")
        portfolio = get_portfolio_state(trading_client, data_client)
        print(f"  Equity: ${portfolio.equity:,.2f}")
        print(f"  Cash: ${portfolio.cash:,.2f}")
        print(f"  {SYMBOL} Position: {portfolio.position_qty} shares "
              f"(${portfolio.position_value:,.2f}, {portfolio.position_pct}%)")
        print(f"  {SYMBOL} Price: ${portfolio.current_price:.2f}")
        print("-" * 60)

        # Fetch market data
        print("Fetching market data...")
        market_data = fetch_ohlcv_data(data_client)
        print(market_data)
        print("-" * 60)

        # Get Claude's decision
        print(f"Asking Claude ({MODEL}) for decision...")
        decision = get_claude_decision(anthropic_client, market_data, portfolio)
        print(f"  Target Allocation: {decision.target_pct}%")
        print(f"  Action: {decision.action}")
        print(f"  Shares to Trade: {decision.shares_to_trade}")
        print(f"  Reasoning: {decision.reasoning}")
        print("-" * 60)

        # Execute trade
        print("Executing trade...")
        result = execute_trade(trading_client, decision, portfolio)
        print(f"Result: {result}")

        # Log trade
        log_trade(decision, portfolio, result)
        print(f"Trade logged to {CSV_FILE}")

        print("-" * 60)
        print("ClaudeTrader completed successfully")
        return 0

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"ERROR: {error_msg}")
        log_error(error_msg)
        # Return 0 to not fail the GitHub Action
        return 0


if __name__ == "__main__":
    sys.exit(main())
