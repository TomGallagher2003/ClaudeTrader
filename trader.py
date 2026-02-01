#!/usr/bin/env python3
"""
ClaudeTrader - Autonomous AI Trading Bot for Alpaca
Optimized for 2026 market regime with regime detection, relative strength, and volatility sizing.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from enum import Enum

import anthropic
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class TradingMode(Enum):
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"

@dataclass
class Config:
    """Trading configuration parameters."""
    # Portfolio
    symbols: list[str] = None
    benchmark: str = "QQQ"
    regime_indicator: str = "SPY"

    # Strategy thresholds
    regime_threshold: float = -0.02  # -2% SPY 5-day triggers defensive
    regime_lookback_days: int = 5
    relative_strength_days: int = 14
    atr_period: int = 30
    atr_volatility_threshold: float = 0.05  # 5% ATR triggers size reduction
    volatility_size_multiplier: float = 0.5  # 50% reduction

    # Position sizing
    base_position_pct: float = 0.10  # 10% of portfolio per position
    max_position_pct: float = 0.15  # 15% max single position
    min_cash_reserve_pct: float = 0.10  # Keep 10% cash

    # Risk management
    stop_loss_pct: float = 0.08  # 8% hard stop
    trailing_stop_pct: float = 0.05  # 5% trailing after 10% gain

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = load_symbols()


def load_symbols() -> list[str]:
    """Load trading symbols from symbols.json or return defaults."""
    try:
        with open('symbols.json', 'r') as f:
            data = json.load(f)
            return data.get('symbols', [])
    except FileNotFoundError:
        return ["NVDA", "AVGO", "ANET", "LLY", "PLTR", "MSFT", "AXON"]


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

class MarketData:
    """Handles all market data retrieval from Alpaca."""

    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )

    def get_historical_bars(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical OHLCV data for a symbol."""
        end = datetime.now()
        start = end - timedelta(days=days + 5)  # Extra buffer for weekends/holidays

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]

        return df.tail(days)

    def get_current_price(self, symbol: str) -> float:
        """Get the most recent closing price for a symbol."""
        bars = self.get_historical_bars(symbol, 1)
        return float(bars['close'].iloc[-1])

    def calculate_return(self, symbol: str, days: int) -> float:
        """Calculate percentage return over specified days."""
        bars = self.get_historical_bars(symbol, days + 1)
        if len(bars) < 2:
            return 0.0

        start_price = float(bars['close'].iloc[0])
        end_price = float(bars['close'].iloc[-1])

        return (end_price - start_price) / start_price

    def calculate_atr(self, symbol: str, period: int = 30) -> tuple[float, float]:
        """
        Calculate Average True Range over specified period.
        Returns (ATR value, ATR as percentage of current price).
        """
        # Fetch extra data to ensure we have enough after dropping NaN
        bars = self.get_historical_bars(symbol, period + 10)

        if len(bars) < period:
            logger.warning(f"Insufficient data for ATR calculation on {symbol}: {len(bars)} bars < {period} required")
            # Return reasonable defaults to avoid NaN
            current_price = float(bars['close'].iloc[-1])
            return 0.0, 0.0

        high = bars['high']
        low = bars['low']
        close = bars['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Drop NaN values that occur from the shift operation
        true_range = true_range.dropna()

        if len(true_range) < period:
            logger.warning(f"Insufficient valid data for ATR on {symbol} after dropping NaN")
            current_price = float(close.iloc[-1])
            return 0.0, 0.0

        # Calculate ATR using the last 'period' days of valid true range data
        atr = true_range.tail(period).mean()

        current_price = float(close.iloc[-1])
        atr_percent = atr / current_price

        return float(atr), float(atr_percent)


# =============================================================================
# STRATEGY FILTERS
# =============================================================================

class StrategyFilters:
    """Implements the three core strategy filters."""

    def __init__(self, config: Config, market_data: MarketData):
        self.config = config
        self.data = market_data

    def check_regime_mode(self) -> tuple[TradingMode, float]:
        """
        Check SPY 5-day trend to determine market regime.
        Returns (TradingMode, spy_return).
        Returns DEFENSIVE if SPY is down >2% over 5 days.
        """
        spy_return = self.data.calculate_return(
            self.config.regime_indicator,
            self.config.regime_lookback_days
        )

        logger.info(f"SPY {self.config.regime_lookback_days}-day return: {spy_return:.2%}")

        if spy_return < self.config.regime_threshold:
            logger.warning(f"DEFENSIVE MODE: SPY down {spy_return:.2%} over {self.config.regime_lookback_days} days")
            return TradingMode.DEFENSIVE, spy_return

        logger.info("OFFENSIVE MODE: Market regime favorable")
        return TradingMode.OFFENSIVE, spy_return

    def check_relative_strength(self, symbol: str) -> tuple[bool, float, float]:
        """
        Compare stock's 14-day performance against QQQ benchmark.
        Returns (is_outperforming, stock_return, benchmark_return).
        """
        stock_return = self.data.calculate_return(
            symbol,
            self.config.relative_strength_days
        )
        benchmark_return = self.data.calculate_return(
            self.config.benchmark,
            self.config.relative_strength_days
        )

        is_outperforming = stock_return > benchmark_return

        logger.info(
            f"{symbol} RS: {stock_return:.2%} vs {self.config.benchmark}: {benchmark_return:.2%} "
            f"-> {'OUTPERFORMING' if is_outperforming else 'UNDERPERFORMING'}"
        )

        return is_outperforming, stock_return, benchmark_return

    def calculate_position_multiplier(self, symbol: str) -> tuple[float, float]:
        """
        Calculate position size multiplier based on ATR volatility.
        Returns (multiplier, atr_percent).
        """
        atr, atr_percent = self.data.calculate_atr(symbol, self.config.atr_period)

        if atr_percent > self.config.atr_volatility_threshold:
            multiplier = self.config.volatility_size_multiplier
            logger.warning(
                f"{symbol} HIGH VOLATILITY: ATR {atr_percent:.2%} > {self.config.atr_volatility_threshold:.0%} "
                f"-> Position reduced to {multiplier:.0%}"
            )
        else:
            multiplier = 1.0
            logger.info(f"{symbol} ATR: {atr_percent:.2%} -> Full position size")

        return multiplier, atr_percent


# =============================================================================
# AI ANALYSIS
# =============================================================================

class AIAnalyzer:
    """Handles Claude AI analysis for trading decisions."""

    def __init__(self, config: Config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def build_analysis_prompt(
        self,
        symbol: str,
        current_price: float,
        regime_mode: TradingMode,
        relative_strength: tuple[bool, float, float],
        atr_data: tuple[float, float],
        position_multiplier: float,
        position_data: dict = None
    ) -> str:
        """Build comprehensive analysis prompt with all strategy context."""
        is_outperforming, stock_return, benchmark_return = relative_strength
        atr, atr_percent = atr_data

        # Build position context if we hold the stock
        position_context = ""
        if position_data:
            qty = position_data["qty"]
            avg_entry = position_data["avg_entry_price"]
            unrealized_pl = position_data["unrealized_pl"]
            unrealized_plpc = position_data["unrealized_plpc"]
            market_value = position_data["market_value"]

            position_context = f"""
## Current Position in {symbol}
- **Holding**: YES - {qty:.0f} shares
- **Entry Price**: ${avg_entry:.2f}
- **Current P&L**: ${unrealized_pl:+.2f} ({unrealized_plpc:+.2%})
- **Position Value**: ${market_value:.2f}
- **Stop Loss Alert**: {'⚠️ APPROACHING STOP (-8%)' if unrealized_plpc < -0.06 else 'No concern'}
"""
        else:
            position_context = f"""
## Current Position in {symbol}
- **Holding**: NO - Not currently in position
"""

        prompt = f"""You are an expert quantitative trader. Analyze {symbol} and provide a trading recommendation.

## Current Market Context
- **Regime Mode**: {regime_mode.value.upper()}
- **SPY Trend**: {'Bearish (>2% down over 5 days)' if regime_mode == TradingMode.DEFENSIVE else 'Neutral/Bullish'}
{position_context}
## {symbol} Market Analysis
- **Current Price**: ${current_price:.2f}
- **14-Day Return**: {stock_return:.2%}
- **QQQ 14-Day Return**: {benchmark_return:.2%}
- **Relative Strength**: {'OUTPERFORMING' if is_outperforming else 'UNDERPERFORMING'} vs QQQ
- **30-Day ATR**: ${atr:.2f} ({atr_percent:.2%} of price)
- **Volatility Classification**: {'HIGH (>5%)' if atr_percent > 0.05 else 'NORMAL'}
- **Position Size Multiplier**: {position_multiplier:.0%}

## Trading Rules
1. If DEFENSIVE mode: Only SELL or HOLD allowed (no new BUYs)
2. If UNDERPERFORMING vs QQQ: Do not BUY
3. If HIGH volatility: Position size already reduced by 50%
4. Stop loss triggers at -8% unrealized loss

## Your Task
Based on the above context and current market conditions, provide:
1. Your recommendation: BUY, SELL, or HOLD
   - If we hold the position: HOLD means stay in, SELL means exit
   - If we don't hold: HOLD means stay out, BUY means enter
2. Confidence level: HIGH, MEDIUM, or LOW
3. Brief rationale (2-3 sentences)

Format your response as:
RECOMMENDATION: [BUY/SELL/HOLD]
CONFIDENCE: [HIGH/MEDIUM/LOW]
RATIONALE: [Your reasoning]
"""
        return prompt

    def get_recommendation(
        self,
        symbol: str,
        current_price: float,
        regime_mode: TradingMode,
        relative_strength: tuple[bool, float, float],
        atr_data: tuple[float, float],
        position_multiplier: float,
        position_data: dict = None
    ) -> dict:
        """Get AI trading recommendation for a symbol."""
        prompt = self.build_analysis_prompt(
            symbol, current_price, regime_mode,
            relative_strength, atr_data, position_multiplier,
            position_data
        )

        message = self.client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Parse response
        recommendation = "HOLD"
        confidence = "LOW"
        rationale = ""

        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith("RECOMMENDATION:"):
                rec = line.replace("RECOMMENDATION:", "").strip().upper()
                if rec in ["BUY", "SELL", "HOLD"]:
                    recommendation = rec
            elif line.startswith("CONFIDENCE:"):
                conf = line.replace("CONFIDENCE:", "").strip().upper()
                if conf in ["HIGH", "MEDIUM", "LOW"]:
                    confidence = conf
            elif line.startswith("RATIONALE:"):
                rationale = line.replace("RATIONALE:", "").strip()

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "raw_response": response_text
        }


# =============================================================================
# TRADE EXECUTION
# =============================================================================

class TradeExecutor:
    """Handles trade execution via Alpaca API."""

    def __init__(self, config: Config):
        self.config = config
        self.client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
        )

    def get_account(self) -> dict:
        """Get current account information."""
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power)
        }

    def get_positions(self) -> dict:
        """Get all current positions."""
        positions = self.client.get_all_positions()
        return {
            pos.symbol: {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "avg_entry_price": float(pos.avg_entry_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc)
            }
            for pos in positions
        }

    def calculate_order_qty(
        self,
        symbol: str,
        current_price: float,
        position_multiplier: float
    ) -> int:
        """Calculate number of shares to buy based on position sizing rules."""
        account = self.get_account()
        equity = account["equity"]
        cash = account["cash"]

        # Ensure minimum cash reserve
        available_cash = cash - (equity * self.config.min_cash_reserve_pct)
        if available_cash <= 0:
            logger.warning("Insufficient cash after reserve requirement")
            return 0

        # Calculate base position size
        base_position_value = equity * self.config.base_position_pct

        # Apply volatility multiplier
        adjusted_position_value = base_position_value * position_multiplier

        # Cap at max position size
        max_position_value = equity * self.config.max_position_pct
        position_value = min(adjusted_position_value, max_position_value, available_cash)

        # Calculate shares
        qty = int(position_value / current_price)

        logger.info(
            f"Position sizing for {symbol}: ${position_value:.2f} / ${current_price:.2f} = {qty} shares"
        )

        return qty

    def execute_buy(self, symbol: str, qty: int) -> Optional[str]:
        """Execute a market buy order."""
        if qty <= 0:
            logger.warning(f"Invalid quantity for {symbol}: {qty}")
            return None

        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = self.client.submit_order(order_data)
            logger.info(f"BUY order submitted: {symbol} x {qty} - Order ID: {order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to execute BUY for {symbol}: {e}")
            return None

    def execute_sell(self, symbol: str, qty: int = None) -> Optional[str]:
        """Execute a market sell order. If qty is None, sells entire position."""
        try:
            positions = self.get_positions()
            if symbol not in positions:
                logger.warning(f"No position in {symbol} to sell")
                return None

            if qty is None:
                qty = int(positions[symbol]["qty"])

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = self.client.submit_order(order_data)
            logger.info(f"SELL order submitted: {symbol} x {qty} - Order ID: {order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to execute SELL for {symbol}: {e}")
            return None


# =============================================================================
# JOB DECISION LOGGING
# =============================================================================

class JobDecisionLogger:
    """
    Logs all decisions from a single trading job to a JSON file.
    Overwrites on each run so you always see the most recent job's decisions.
    """

    def __init__(self):
        self.job_start = datetime.now().isoformat()
        self.regime_mode = None
        self.spy_return = None
        self.decisions = []
        self.trades_executed = 0

    def set_regime(self, mode: TradingMode, spy_return: float):
        """Record the market regime for this job."""
        self.regime_mode = mode.value
        self.spy_return = spy_return

    def log_decision(
        self,
        symbol: str,
        current_price: float,
        ai_recommendation: str,
        ai_confidence: str,
        ai_rationale: str,
        final_action: str,
        block_reason: str = None,
        is_outperforming: bool = None,
        stock_return: float = None,
        benchmark_return: float = None,
        atr_percent: float = None,
        position_multiplier: float = None,
        qty: int = 0,
        order_id: str = None
    ):
        """Log a decision for a single symbol."""
        trade_executed = order_id is not None and qty > 0

        decision = {
            "symbol": symbol,
            "price": round(current_price, 2),
            "ai_recommendation": ai_recommendation,
            "ai_confidence": ai_confidence,
            "ai_rationale": ai_rationale,
            "final_action": final_action,
            "trade_executed": trade_executed,
            "block_reason": block_reason,
            "metrics": {
                "relative_strength": "OUTPERFORMING" if is_outperforming else "UNDERPERFORMING",
                "stock_14d_return": f"{stock_return:.2%}" if stock_return is not None else None,
                "benchmark_14d_return": f"{benchmark_return:.2%}" if benchmark_return is not None else None,
                "atr_percent": f"{atr_percent:.2%}" if atr_percent is not None else None,
                "position_multiplier": position_multiplier
            },
            "execution": {
                "qty": qty,
                "order_id": order_id
            } if trade_executed else None
        }

        self.decisions.append(decision)

        if trade_executed:
            self.trades_executed += 1

    def write_log(self, filepath: str = "latest_decisions.json"):
        """Write the job decisions to a JSON file."""
        # Calculate summary
        actions = [d["final_action"] for d in self.decisions]
        trades = [d for d in self.decisions if d["trade_executed"]]

        summary = {
            "total_symbols_analyzed": len(self.decisions),
            "buys": actions.count("BUY"),
            "sells": actions.count("SELL"),
            "holds": actions.count("HOLD"),
            "trades_executed": self.trades_executed,
            "executed_trades": [
                {
                    "symbol": t["symbol"],
                    "action": t["final_action"],
                    "qty": t["execution"]["qty"],
                    "price": t["price"]
                }
                for t in trades
            ]
        }

        job_log = {
            "job_timestamp": self.job_start,
            "job_completed": datetime.now().isoformat(),
            "market_regime": {
                "mode": self.regime_mode,
                "spy_5d_return": f"{self.spy_return:.2%}" if self.spy_return is not None else None
            },
            "summary": summary,
            "decisions": self.decisions
        }

        with open(filepath, 'w') as f:
            json.dump(job_log, f, indent=2)

        logger.info(f"Job decisions written to {filepath}")
        logger.info(f"Summary: {summary['trades_executed']} trades executed out of {summary['total_symbols_analyzed']} symbols analyzed")


# =============================================================================
# TRADE LOGGING
# =============================================================================

def log_trade(
    symbol: str,
    action: str,
    qty: int,
    price: float,
    regime_mode: str,
    relative_strength: str,
    atr_percent: float,
    position_multiplier: float,
    ai_confidence: str,
    ai_rationale: str,
    order_id: str = None
):
    """Log trade to CSV file."""
    trade_data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "price": price,
        "regime_mode": regime_mode,
        "relative_strength": relative_strength,
        "atr_percent": f"{atr_percent:.4f}",
        "position_multiplier": position_multiplier,
        "ai_confidence": ai_confidence,
        "ai_rationale": ai_rationale,
        "order_id": order_id or "N/A"
    }

    df = pd.DataFrame([trade_data])

    file_exists = os.path.exists('trades.csv')
    df.to_csv('trades.csv', mode='a', header=not file_exists, index=False)

    logger.info(f"Trade logged: {action} {symbol} x {qty} @ ${price:.2f}")


# =============================================================================
# MAIN TRADING LOOP
# =============================================================================

def run_trading_cycle():
    """Execute one complete trading cycle."""
    logger.info("=" * 60)
    logger.info("Starting trading cycle")
    logger.info("=" * 60)

    # Initialize components
    config = Config()
    market_data = MarketData()
    filters = StrategyFilters(config, market_data)
    analyzer = AIAnalyzer(config)
    executor = TradeExecutor(config)
    job_logger = JobDecisionLogger()

    # Step 1: Check market regime
    regime_mode, spy_return = filters.check_regime_mode()
    job_logger.set_regime(regime_mode, spy_return)

    # Step 2: Enforce stop losses on existing positions
    logger.info("-" * 40)
    logger.info("Checking stop losses on existing positions")
    positions = executor.get_positions()

    for symbol, position_data in positions.items():
        unrealized_plpc = position_data["unrealized_plpc"]
        unrealized_pl = position_data["unrealized_pl"]
        avg_entry_price = position_data["avg_entry_price"]

        logger.info(f"{symbol}: P&L {unrealized_plpc:.2%} (${unrealized_pl:.2f})")

        # Check if stop loss threshold is breached
        if unrealized_plpc < -config.stop_loss_pct:
            logger.warning(
                f"STOP LOSS TRIGGERED for {symbol}: {unrealized_plpc:.2%} loss exceeds {config.stop_loss_pct:.0%} threshold"
            )
            qty = int(position_data["qty"])
            order_id = executor.execute_sell(symbol)

            if order_id:
                logger.info(f"Stop loss executed: SELL {symbol} x {qty}")
                # Log the stop loss trade
                current_price = market_data.get_current_price(symbol)
                log_trade(
                    symbol=symbol,
                    action="SELL",
                    qty=qty,
                    price=current_price,
                    regime_mode=regime_mode.value,
                    relative_strength="N/A",
                    atr_percent=0.0,
                    position_multiplier=1.0,
                    ai_confidence="N/A",
                    ai_rationale=f"Stop loss triggered at {unrealized_plpc:.2%}",
                    order_id=order_id
                )

    # Step 3: Process each symbol
    for symbol in config.symbols:
        logger.info("-" * 40)
        logger.info(f"Analyzing {symbol}")

        try:
            # Get current price
            current_price = market_data.get_current_price(symbol)

            # Check relative strength
            is_outperforming, stock_return, benchmark_return = filters.check_relative_strength(symbol)
            relative_strength = (is_outperforming, stock_return, benchmark_return)

            # Calculate volatility-adjusted position size
            position_multiplier, atr_percent = filters.calculate_position_multiplier(symbol)
            atr_data = (market_data.calculate_atr(symbol)[0], atr_percent)

            # Get current position data for this symbol (if exists)
            positions = executor.get_positions()
            position_data = positions.get(symbol, None)

            # Get AI recommendation with position context
            ai_result = analyzer.get_recommendation(
                symbol, current_price, regime_mode,
                relative_strength, atr_data, position_multiplier,
                position_data
            )

            recommendation = ai_result["recommendation"]
            confidence = ai_result["confidence"]
            rationale = ai_result["rationale"]

            logger.info(f"AI Recommendation: {recommendation} ({confidence})")
            logger.info(f"Rationale: {rationale}")

            # Apply strategy filters
            final_action = recommendation
            block_reason = None

            # Filter 1: Block BUYs in defensive mode
            if regime_mode == TradingMode.DEFENSIVE and recommendation == "BUY":
                logger.warning(f"BUY blocked for {symbol}: DEFENSIVE mode active")
                final_action = "HOLD"
                block_reason = "DEFENSIVE mode - market down >2% over 5 days"

            # Filter 2: Block BUYs if underperforming benchmark
            if not is_outperforming and recommendation == "BUY":
                logger.warning(f"BUY blocked for {symbol}: Underperforming {config.benchmark}")
                final_action = "HOLD"
                block_reason = f"Underperforming {config.benchmark} benchmark"

            # Execute trade
            order_id = None
            qty = 0

            if final_action == "BUY":
                qty = executor.calculate_order_qty(symbol, current_price, position_multiplier)
                if qty > 0:
                    order_id = executor.execute_buy(symbol, qty)
            elif final_action == "SELL":
                positions = executor.get_positions()
                if symbol in positions:
                    qty = int(positions[symbol]["qty"])
                    order_id = executor.execute_sell(symbol)

            # Log trade to CSV
            log_trade(
                symbol=symbol,
                action=final_action,
                qty=qty,
                price=current_price,
                regime_mode=regime_mode.value,
                relative_strength="OUTPERFORMING" if is_outperforming else "UNDERPERFORMING",
                atr_percent=atr_percent,
                position_multiplier=position_multiplier,
                ai_confidence=confidence,
                ai_rationale=rationale,
                order_id=order_id
            )

            # Log decision to job log
            job_logger.log_decision(
                symbol=symbol,
                current_price=current_price,
                ai_recommendation=recommendation,
                ai_confidence=confidence,
                ai_rationale=rationale,
                final_action=final_action,
                block_reason=block_reason,
                is_outperforming=is_outperforming,
                stock_return=stock_return,
                benchmark_return=benchmark_return,
                atr_percent=atr_percent,
                position_multiplier=position_multiplier,
                qty=qty,
                order_id=order_id
            )

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    # Write job decisions log
    job_logger.write_log()

    logger.info("=" * 60)
    logger.info("Trading cycle complete")
    logger.info("=" * 60)


def is_market_open() -> bool:
    """Check if the market is currently open using Alpaca's clock API."""
    try:
        client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=os.getenv('ALPACA_PAPER', 'true').lower() == 'true'
        )
        clock = client.get_clock()
        return clock.is_open
    except Exception as e:
        logger.error(f"Failed to check market status: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("ClaudeTrader initialized")

    if not is_market_open():
        logger.warning("Market is not open. Skipping trading cycle.")
        return

    run_trading_cycle()


if __name__ == "__main__":
    main()
