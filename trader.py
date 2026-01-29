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
            end=end
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
        bars = self.get_historical_bars(symbol, period + 1)

        high = bars['high']
        low = bars['low']
        close = bars['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]

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

    def check_regime_mode(self) -> TradingMode:
        """
        Check SPY 5-day trend to determine market regime.
        Returns DEFENSIVE if SPY is down >2% over 5 days.
        """
        spy_return = self.data.calculate_return(
            self.config.regime_indicator,
            self.config.regime_lookback_days
        )

        logger.info(f"SPY {self.config.regime_lookback_days}-day return: {spy_return:.2%}")

        if spy_return < self.config.regime_threshold:
            logger.warning(f"DEFENSIVE MODE: SPY down {spy_return:.2%} over {self.config.regime_lookback_days} days")
            return TradingMode.DEFENSIVE

        logger.info("OFFENSIVE MODE: Market regime favorable")
        return TradingMode.OFFENSIVE

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
        position_multiplier: float
    ) -> str:
        """Build comprehensive analysis prompt with all strategy context."""
        is_outperforming, stock_return, benchmark_return = relative_strength
        atr, atr_percent = atr_data

        prompt = f"""You are an expert quantitative trader. Analyze {symbol} and provide a trading recommendation.

## Current Market Context
- **Regime Mode**: {regime_mode.value.upper()}
- **SPY Trend**: {'Bearish (>2% down over 5 days)' if regime_mode == TradingMode.DEFENSIVE else 'Neutral/Bullish'}

## {symbol} Analysis
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

## Your Task
Based on the above context and current market conditions, provide:
1. Your recommendation: BUY, SELL, or HOLD
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
        position_multiplier: float
    ) -> dict:
        """Get AI trading recommendation for a symbol."""
        prompt = self.build_analysis_prompt(
            symbol, current_price, regime_mode,
            relative_strength, atr_data, position_multiplier
        )

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
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

    # Step 1: Check market regime
    regime_mode = filters.check_regime_mode()

    # Step 2: Process each symbol
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

            # Get AI recommendation
            ai_result = analyzer.get_recommendation(
                symbol, current_price, regime_mode,
                relative_strength, atr_data, position_multiplier
            )

            recommendation = ai_result["recommendation"]
            confidence = ai_result["confidence"]
            rationale = ai_result["rationale"]

            logger.info(f"AI Recommendation: {recommendation} ({confidence})")
            logger.info(f"Rationale: {rationale}")

            # Apply strategy filters
            final_action = recommendation

            # Filter 1: Block BUYs in defensive mode
            if regime_mode == TradingMode.DEFENSIVE and recommendation == "BUY":
                logger.warning(f"BUY blocked for {symbol}: DEFENSIVE mode active")
                final_action = "HOLD"

            # Filter 2: Block BUYs if underperforming benchmark
            if not is_outperforming and recommendation == "BUY":
                logger.warning(f"BUY blocked for {symbol}: Underperforming {config.benchmark}")
                final_action = "HOLD"

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

            # Log trade
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

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    logger.info("=" * 60)
    logger.info("Trading cycle complete")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    logger.info("ClaudeTrader initialized")
    run_trading_cycle()


if __name__ == "__main__":
    main()
