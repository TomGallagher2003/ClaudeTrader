#!/usr/bin/env python3
"""
Backtesting framework for ClaudeTrader.
Validates strategy performance on historical data with transaction cost modeling.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import existing trader components
from trader import (
    Config, MarketData, StrategyFilters, AIAnalyzer,
    TradingMode, load_symbols
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TRANSACTION COST MODEL
# =============================================================================

@dataclass
class TransactionCosts:
    """Models realistic transaction costs for backtesting."""

    # Bid-ask spread (percentage)
    spread_pct: float = 0.0003  # 0.03% typical for liquid stocks

    # Slippage (percentage) - market impact
    slippage_pct: float = 0.0002  # 0.02% for market orders

    # SEC fees (per share)
    sec_fee_per_dollar: float = 0.000008  # $0.000008 per dollar

    # Commission (Alpaca is commission-free)
    commission_per_trade: float = 0.0

    def calculate_buy_cost(self, shares: int, price: float) -> float:
        """Calculate total cost to buy shares including all transaction costs."""
        gross_value = shares * price

        # Pay the ask (higher price due to spread)
        spread_cost = gross_value * (self.spread_pct / 2)

        # Market impact slippage (moves price against us)
        slippage_cost = gross_value * self.slippage_pct

        # SEC fees
        sec_fees = gross_value * self.sec_fee_per_dollar

        total_cost = gross_value + spread_cost + slippage_cost + sec_fees + self.commission_per_trade

        return total_cost

    def calculate_sell_proceeds(self, shares: int, price: float) -> float:
        """Calculate net proceeds from selling shares after transaction costs."""
        gross_value = shares * price

        # Receive the bid (lower price due to spread)
        spread_cost = gross_value * (self.spread_pct / 2)

        # Market impact slippage
        slippage_cost = gross_value * self.slippage_pct

        # SEC fees
        sec_fees = gross_value * self.sec_fee_per_dollar

        net_proceeds = gross_value - spread_cost - slippage_cost - sec_fees - self.commission_per_trade

        return net_proceeds

    def get_total_cost_pct(self) -> float:
        """Get total round-trip cost as percentage."""
        # Buy spread + sell spread + buy slippage + sell slippage
        return self.spread_pct + (2 * self.slippage_pct)


# =============================================================================
# BACKTEST PORTFOLIO
# =============================================================================

@dataclass
class Position:
    """Represents a single position in the portfolio."""
    symbol: str
    shares: int
    avg_entry_price: float
    entry_date: datetime

    def get_market_value(self, current_price: float) -> float:
        """Get current market value of position."""
        return self.shares * current_price

    def get_unrealized_pl(self, current_price: float) -> float:
        """Get unrealized profit/loss."""
        return self.shares * (current_price - self.avg_entry_price)

    def get_unrealized_plpc(self, current_price: float) -> float:
        """Get unrealized profit/loss as percentage."""
        if self.avg_entry_price == 0:
            return 0.0
        return (current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class Trade:
    """Represents a completed trade."""
    date: datetime
    symbol: str
    action: str  # BUY or SELL
    shares: int
    price: float
    cost: float  # Total cost including transaction costs
    regime_mode: str
    ai_confidence: str
    ai_rationale: str


class BacktestPortfolio:
    """Simulates portfolio state during backtesting."""

    def __init__(self, initial_capital: float = 100000.0, transaction_costs: TransactionCosts = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.transaction_costs = transaction_costs or TransactionCosts()

        # Track portfolio value over time
        self.equity_curve = []
        self.daily_returns = []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio equity."""
        positions_value = sum(
            pos.get_market_value(current_prices[pos.symbol])
            for pos in self.positions.values()
            if pos.symbol in current_prices
        )
        return self.cash + positions_value

    def get_position_value_pct(self, symbol: str, current_price: float, total_equity: float) -> float:
        """Get position value as percentage of total equity."""
        if symbol not in self.positions:
            return 0.0
        market_value = self.positions[symbol].get_market_value(current_price)
        return market_value / total_equity if total_equity > 0 else 0.0

    def execute_buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        date: datetime,
        regime_mode: str,
        ai_confidence: str,
        ai_rationale: str
    ) -> bool:
        """Execute a buy order with transaction costs."""
        if shares <= 0:
            return False

        total_cost = self.transaction_costs.calculate_buy_cost(shares, price)

        if total_cost > self.cash:
            logger.warning(f"Insufficient cash to buy {shares} shares of {symbol} at ${price:.2f}")
            return False

        # Update cash
        self.cash -= total_cost

        # Update or create position
        if symbol in self.positions:
            # Average up
            existing_pos = self.positions[symbol]
            total_shares = existing_pos.shares + shares
            total_cost_basis = (existing_pos.shares * existing_pos.avg_entry_price) + total_cost
            avg_price = total_cost_basis / total_shares

            self.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                avg_entry_price=avg_price,
                entry_date=existing_pos.entry_date
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_entry_price=total_cost / shares,
                entry_date=date
            )

        # Record trade
        self.trades.append(Trade(
            date=date,
            symbol=symbol,
            action="BUY",
            shares=shares,
            price=price,
            cost=total_cost,
            regime_mode=regime_mode,
            ai_confidence=ai_confidence,
            ai_rationale=ai_rationale
        ))

        logger.debug(f"BUY {shares} {symbol} @ ${price:.2f} (cost: ${total_cost:.2f})")
        return True

    def execute_sell(
        self,
        symbol: str,
        shares: Optional[int],
        price: float,
        date: datetime,
        regime_mode: str,
        ai_confidence: str,
        ai_rationale: str
    ) -> bool:
        """Execute a sell order with transaction costs."""
        if symbol not in self.positions:
            logger.warning(f"Cannot sell {symbol} - no position")
            return False

        position = self.positions[symbol]

        # If shares is None, sell entire position
        if shares is None:
            shares = position.shares

        if shares > position.shares:
            logger.warning(f"Cannot sell {shares} shares of {symbol} - only have {position.shares}")
            return False

        # Calculate proceeds after transaction costs
        net_proceeds = self.transaction_costs.calculate_sell_proceeds(shares, price)

        # Update cash
        self.cash += net_proceeds

        # Update position
        if shares == position.shares:
            # Close position completely
            del self.positions[symbol]
        else:
            # Reduce position
            position.shares -= shares

        # Record trade
        self.trades.append(Trade(
            date=date,
            symbol=symbol,
            action="SELL",
            shares=shares,
            price=price,
            cost=net_proceeds,
            regime_mode=regime_mode,
            ai_confidence=ai_confidence,
            ai_rationale=ai_rationale
        ))

        logger.debug(f"SELL {shares} {symbol} @ ${price:.2f} (proceeds: ${net_proceeds:.2f})")
        return True

    def record_equity(self, date: datetime, current_prices: Dict[str, float]):
        """Record portfolio equity for this date."""
        equity = self.get_equity(current_prices)
        self.equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "positions_value": equity - self.cash
        })

        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

class PerformanceMetrics:
    """Calculates comprehensive performance statistics."""

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.04) -> float:
        """Calculate annualized Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        # Annualize (sqrt of 252 trading days)
        return sharpe * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[Dict]) -> Tuple[float, datetime, datetime]:
        """Calculate maximum drawdown and its dates."""
        if not equity_curve:
            return 0.0, None, None

        equity_values = [point["equity"] for point in equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        peak_date = equity_curve[0]["date"]
        trough_date = equity_curve[0]["date"]

        for i, point in enumerate(equity_curve):
            equity = point["equity"]
            if equity > peak:
                peak = equity
                peak_date = point["date"]

            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                trough_date = point["date"]

        return max_dd, peak_date, trough_date

    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> Dict:
        """Calculate win rate and average win/loss."""
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }

        # Group trades into round trips (buy + sell pairs)
        buy_trades = {}
        round_trips = []

        for trade in trades:
            if trade.action == "BUY":
                if trade.symbol not in buy_trades:
                    buy_trades[trade.symbol] = []
                buy_trades[trade.symbol].append(trade)
            elif trade.action == "SELL":
                if trade.symbol in buy_trades and buy_trades[trade.symbol]:
                    buy_trade = buy_trades[trade.symbol].pop(0)
                    # Calculate P&L for this round trip
                    pnl = trade.cost - (buy_trade.cost * (trade.shares / buy_trade.shares))
                    pnl_pct = pnl / buy_trade.cost if buy_trade.cost > 0 else 0
                    round_trips.append(pnl_pct)

        if not round_trips:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }

        wins = [r for r in round_trips if r > 0]
        losses = [r for r in round_trips if r < 0]

        return {
            "win_rate": len(wins) / len(round_trips) if round_trips else 0.0,
            "avg_win": np.mean(wins) if wins else 0.0,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "total_trades": len(round_trips),
            "winning_trades": len(wins),
            "losing_trades": len(losses)
        }

    @staticmethod
    def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
        """Calculate Compound Annual Growth Rate."""
        if initial_value <= 0 or years <= 0:
            return 0.0
        return (final_value / initial_value) ** (1 / years) - 1

    @staticmethod
    def calculate_volatility(returns: List[float]) -> float:
        """Calculate annualized volatility."""
        if not returns or len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Main backtesting engine."""

    def __init__(
        self,
        config: Config = None,
        initial_capital: float = 100000.0,
        transaction_costs: TransactionCosts = None,
        use_ai: bool = False  # Set to False to save API costs during backtesting
    ):
        self.config = config or Config()
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs or TransactionCosts()
        self.use_ai = use_ai

        # Initialize components
        self.market_data = MarketData()
        self.portfolio = BacktestPortfolio(initial_capital, self.transaction_costs)

        # Benchmark data
        self.benchmark_equity_curve = []

    def load_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Load historical data for all symbols."""
        logger.info(f"Loading historical data from {start_date.date()} to {end_date.date()}")

        data = {}
        days = (end_date - start_date).days + 60  # Extra buffer for indicators

        for symbol in symbols:
            try:
                df = self.market_data.get_historical_bars(symbol, days)
                # Filter to exact date range
                df = df[df.index >= pd.Timestamp(start_date)]
                df = df[df.index <= pd.Timestamp(end_date)]
                data[symbol] = df
                logger.info(f"Loaded {len(df)} days of data for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")

        return data

    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str] = None
    ) -> Dict:
        """
        Run backtest over specified date range.

        Returns comprehensive performance report.
        """
        if symbols is None:
            symbols = self.config.symbols

        logger.info("=" * 60)
        logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"AI analysis: {'ENABLED' if self.use_ai else 'DISABLED (rules-only)'}")
        logger.info(f"Transaction costs: {self.transaction_costs.get_total_cost_pct():.4%} round-trip")
        logger.info("=" * 60)

        # Load historical data
        historical_data = self.load_historical_data(
            symbols + [self.config.benchmark, self.config.regime_indicator],
            start_date,
            end_date
        )

        # Get trading dates (use SPY as reference)
        if self.config.regime_indicator not in historical_data:
            raise ValueError(f"Missing data for regime indicator {self.config.regime_indicator}")

        trading_dates = historical_data[self.config.regime_indicator].index

        # Initialize components for strategy logic
        filters = StrategyFilters(self.config, self.market_data)
        analyzer = AIAnalyzer(self.config) if self.use_ai else None

        # Simulate day by day
        for current_date in trading_dates:
            logger.debug(f"\n--- Processing {current_date.date()} ---")

            # Get current prices for all symbols
            current_prices = {}
            for symbol in symbols:
                if symbol in historical_data and current_date in historical_data[symbol].index:
                    current_prices[symbol] = float(historical_data[symbol].loc[current_date, 'close'])

            # Skip if we don't have data for this date
            if not current_prices:
                continue

            # Step 1: Check market regime
            regime_mode, spy_return = self._check_regime_at_date(
                historical_data, current_date
            )

            # Step 2: Enforce stop losses
            self._enforce_stop_losses(
                current_date, current_prices, regime_mode
            )

            # Step 3: Check profit-taking
            self._check_profit_taking(
                historical_data, current_date, current_prices, regime_mode, filters
            )

            # Step 4: Process each symbol for potential entry/exit
            for symbol in symbols:
                if symbol not in current_prices:
                    continue

                self._process_symbol(
                    symbol, current_date, historical_data,
                    current_prices, regime_mode, filters, analyzer
                )

            # Record portfolio equity for this date
            self.portfolio.record_equity(current_date, current_prices)

            # Track benchmark (buy and hold QQQ)
            if self.config.benchmark in current_prices:
                benchmark_value = self._calculate_benchmark_value(
                    self.config.benchmark, historical_data, current_date
                )
                self.benchmark_equity_curve.append({
                    "date": current_date,
                    "equity": benchmark_value
                })

        # Calculate performance metrics
        performance = self._calculate_performance(start_date, end_date)

        logger.info("=" * 60)
        logger.info("Backtest complete")
        logger.info("=" * 60)

        return performance

    def _check_regime_at_date(
        self,
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime
    ) -> Tuple[TradingMode, float]:
        """Check market regime at a specific date."""
        spy_symbol = self.config.regime_indicator

        if spy_symbol not in historical_data:
            return TradingMode.OFFENSIVE, 0.0

        spy_data = historical_data[spy_symbol]

        # Get data up to current date
        spy_data_upto_date = spy_data[spy_data.index <= current_date]

        if len(spy_data_upto_date) < self.config.regime_lookback_days + 1:
            return TradingMode.OFFENSIVE, 0.0

        # Calculate return over lookback period
        recent_data = spy_data_upto_date.tail(self.config.regime_lookback_days + 1)
        start_price = float(recent_data['close'].iloc[0])
        end_price = float(recent_data['close'].iloc[-1])
        spy_return = (end_price - start_price) / start_price

        if spy_return < self.config.regime_threshold:
            return TradingMode.DEFENSIVE, spy_return

        return TradingMode.OFFENSIVE, spy_return

    def _enforce_stop_losses(
        self,
        current_date: datetime,
        current_prices: Dict[str, float],
        regime_mode: TradingMode
    ):
        """Enforce stop losses on existing positions."""
        positions_to_close = []

        for symbol, position in self.portfolio.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            unrealized_plpc = position.get_unrealized_plpc(current_price)

            if unrealized_plpc < -self.config.stop_loss_pct:
                logger.debug(
                    f"STOP LOSS: {symbol} at {unrealized_plpc:.2%} loss "
                    f"(threshold: {self.config.stop_loss_pct:.0%})"
                )
                positions_to_close.append(symbol)

        for symbol in positions_to_close:
            self.portfolio.execute_sell(
                symbol=symbol,
                shares=None,  # Sell all
                price=current_prices[symbol],
                date=current_date,
                regime_mode=regime_mode.value,
                ai_confidence="N/A",
                ai_rationale=f"Stop loss triggered"
            )

    def _check_profit_taking(
        self,
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        current_prices: Dict[str, float],
        regime_mode: TradingMode,
        filters: StrategyFilters
    ):
        """Check profit-taking opportunities."""
        for symbol, position in list(self.portfolio.positions.items()):
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            unrealized_plpc = position.get_unrealized_plpc(current_price)

            # Convert position to dict format expected by check_profit_taking
            position_data = {
                "qty": position.shares,
                "avg_entry_price": position.avg_entry_price,
                "unrealized_plpc": unrealized_plpc,
                "unrealized_pl": position.get_unrealized_pl(current_price),
                "market_value": position.get_market_value(current_price)
            }

            # This is a simplified version - in production you'd need to inject
            # market_data that respects the backtest date
            # For now, just check the simple profit-taking rule
            if unrealized_plpc >= 0.15:
                # Scale out 50% at +15% gain
                shares_to_sell = position.shares // 2
                if shares_to_sell > 0:
                    logger.debug(f"PROFIT-TAKING: Selling 50% of {symbol} at +{unrealized_plpc:.1%}")
                    self.portfolio.execute_sell(
                        symbol=symbol,
                        shares=shares_to_sell,
                        price=current_price,
                        date=current_date,
                        regime_mode=regime_mode.value,
                        ai_confidence="N/A",
                        ai_rationale=f"Profit target reached: +{unrealized_plpc:.1%}"
                    )

    def _process_symbol(
        self,
        symbol: str,
        current_date: datetime,
        historical_data: Dict[str, pd.DataFrame],
        current_prices: Dict[str, float],
        regime_mode: TradingMode,
        filters: StrategyFilters,
        analyzer: Optional[AIAnalyzer]
    ):
        """Process a single symbol for trading decisions."""
        current_price = current_prices[symbol]

        # Get historical data up to current date for this symbol
        symbol_data = historical_data[symbol]
        symbol_data_upto_date = symbol_data[symbol_data.index <= current_date]

        if len(symbol_data_upto_date) < 35:  # Need enough data for indicators
            return

        # Calculate relative strength (simplified - uses recent data)
        # In production, this would use data only up to current_date
        stock_return_14d = self._calculate_return_at_date(
            symbol_data_upto_date, 14
        )

        benchmark_data = historical_data.get(self.config.benchmark)
        if benchmark_data is None:
            return

        benchmark_data_upto_date = benchmark_data[benchmark_data.index <= current_date]
        benchmark_return_14d = self._calculate_return_at_date(
            benchmark_data_upto_date, 14
        )

        is_outperforming = stock_return_14d > benchmark_return_14d

        # Check if we should buy
        existing_position = self.portfolio.get_position(symbol)

        # Simple decision logic (without full AI analysis to save costs)
        recommendation = "HOLD"

        if existing_position is None:
            # Consider buying
            if regime_mode == TradingMode.OFFENSIVE and is_outperforming:
                # Additional entry signals check
                entry_signals = self._generate_entry_signals_at_date(
                    symbol_data_upto_date, benchmark_data_upto_date
                )

                if entry_signals["signal_count"] >= 2:  # At least MODERATE strength
                    recommendation = "BUY"
        else:
            # Consider selling (if AI suggests or if underperforming badly)
            unrealized_plpc = existing_position.get_unrealized_plpc(current_price)

            # Sell if severely underperforming
            if unrealized_plpc < -0.05 and not is_outperforming:
                recommendation = "SELL"

        # Execute decision
        if recommendation == "BUY":
            # Calculate position size
            current_equity = self.portfolio.get_equity(current_prices)
            position_value = current_equity * self.config.base_position_pct

            # Apply volatility adjustment (simplified)
            # In production, calculate ATR at this date
            position_value = min(
                position_value,
                current_equity * self.config.max_position_pct,
                self.portfolio.cash * 0.9  # Leave some cash buffer
            )

            shares = int(position_value / current_price)

            if shares > 0:
                self.portfolio.execute_buy(
                    symbol=symbol,
                    shares=shares,
                    price=current_price,
                    date=current_date,
                    regime_mode=regime_mode.value,
                    ai_confidence="MEDIUM",
                    ai_rationale=f"Entry signals detected, outperforming benchmark"
                )

        elif recommendation == "SELL" and existing_position:
            self.portfolio.execute_sell(
                symbol=symbol,
                shares=None,  # Sell all
                price=current_price,
                date=current_date,
                regime_mode=regime_mode.value,
                ai_confidence="MEDIUM",
                ai_rationale="Underperforming benchmark with losses"
            )

    def _calculate_return_at_date(self, data: pd.DataFrame, days: int) -> float:
        """Calculate return over specified days using data up to current date."""
        if len(data) < days + 1:
            return 0.0

        recent = data.tail(days + 1)
        start_price = float(recent['close'].iloc[0])
        end_price = float(recent['close'].iloc[-1])

        return (end_price - start_price) / start_price

    def _generate_entry_signals_at_date(
        self,
        symbol_data: pd.DataFrame,
        benchmark_data: pd.DataFrame
    ) -> Dict:
        """Generate entry signals using historical data up to current date."""
        signals = {
            "momentum_flip": False,
            "ma_crossover": False,
            "rsi_recovery": False,
            "volume_confirmation": False,
            "signal_count": 0,
            "signal_strength": "NONE"
        }

        if len(symbol_data) < 30:
            return signals

        # Calculate indicators
        current_price = float(symbol_data['close'].iloc[-1])
        sma_20 = float(symbol_data['close'].tail(20).mean())

        # RSI calculation
        close = symbol_data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # Volume
        avg_volume = symbol_data['volume'].tail(20).mean()
        current_volume = float(symbol_data['volume'].iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Signal checks
        if current_price > sma_20:
            ret_14d = self._calculate_return_at_date(symbol_data, 14)
            if ret_14d < 0:
                signals["ma_crossover"] = True
                signals["signal_count"] += 1

        if 30 < rsi < 60:
            signals["rsi_recovery"] = True
            signals["signal_count"] += 1

        if volume_ratio > 1.5:
            signals["volume_confirmation"] = True
            signals["signal_count"] += 1

        # Determine strength
        if signals["signal_count"] >= 3:
            signals["signal_strength"] = "STRONG"
        elif signals["signal_count"] >= 2:
            signals["signal_strength"] = "MODERATE"
        elif signals["signal_count"] >= 1:
            signals["signal_strength"] = "WEAK"

        return signals

    def _calculate_benchmark_value(
        self,
        benchmark_symbol: str,
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime
    ) -> float:
        """Calculate buy-and-hold benchmark value."""
        if benchmark_symbol not in historical_data:
            return self.initial_capital

        benchmark_data = historical_data[benchmark_symbol]
        benchmark_data_upto_date = benchmark_data[benchmark_data.index <= current_date]

        if len(benchmark_data_upto_date) < 2:
            return self.initial_capital

        initial_price = float(benchmark_data_upto_date['close'].iloc[0])
        current_price = float(benchmark_data_upto_date['close'].iloc[-1])

        return self.initial_capital * (current_price / initial_price)

    def _calculate_performance(self, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate comprehensive performance metrics."""
        equity_curve = self.portfolio.equity_curve
        trades = self.portfolio.trades
        daily_returns = self.portfolio.daily_returns

        if not equity_curve:
            return {"error": "No data in equity curve"}

        initial_equity = self.initial_capital
        final_equity = equity_curve[-1]["equity"]

        # Calculate years
        years = (end_date - start_date).days / 365.25

        # Calculate metrics
        cagr = PerformanceMetrics.calculate_cagr(initial_equity, final_equity, years)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(daily_returns)
        max_dd, peak_date, trough_date = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        volatility = PerformanceMetrics.calculate_volatility(daily_returns)
        win_stats = PerformanceMetrics.calculate_win_rate(trades)

        # Benchmark comparison
        benchmark_initial = self.initial_capital
        benchmark_final = self.benchmark_equity_curve[-1]["equity"] if self.benchmark_equity_curve else benchmark_initial
        benchmark_cagr = PerformanceMetrics.calculate_cagr(benchmark_initial, benchmark_final, years)

        # Total transaction costs
        total_costs = sum(
            abs(trade.cost - (trade.shares * trade.price))
            for trade in trades
        )

        performance = {
            "period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "years": round(years, 2)
            },
            "returns": {
                "initial_capital": initial_equity,
                "final_equity": round(final_equity, 2),
                "total_return": round((final_equity - initial_equity) / initial_equity, 4),
                "total_return_pct": f"{((final_equity - initial_equity) / initial_equity):.2%}",
                "cagr": round(cagr, 4),
                "cagr_pct": f"{cagr:.2%}"
            },
            "risk_metrics": {
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown": round(max_dd, 4),
                "max_drawdown_pct": f"{max_dd:.2%}",
                "peak_date": peak_date.strftime("%Y-%m-%d") if peak_date else None,
                "trough_date": trough_date.strftime("%Y-%m-%d") if trough_date else None,
                "volatility": round(volatility, 4),
                "volatility_pct": f"{volatility:.2%}"
            },
            "trading_stats": {
                "total_trades": win_stats["total_trades"],
                "winning_trades": win_stats["winning_trades"],
                "losing_trades": win_stats["losing_trades"],
                "win_rate": round(win_stats["win_rate"], 4),
                "win_rate_pct": f"{win_stats['win_rate']:.2%}",
                "avg_win": round(win_stats["avg_win"], 4),
                "avg_win_pct": f"{win_stats['avg_win']:.2%}",
                "avg_loss": round(win_stats["avg_loss"], 4),
                "avg_loss_pct": f"{win_stats['avg_loss']:.2%}",
                "total_executed_orders": len(trades)
            },
            "costs": {
                "total_transaction_costs": round(total_costs, 2),
                "cost_as_pct_of_gains": f"{(total_costs / max(final_equity - initial_equity, 1)):.2%}",
                "avg_cost_per_trade": round(total_costs / len(trades), 2) if trades else 0
            },
            "benchmark": {
                "symbol": self.config.benchmark,
                "initial_value": benchmark_initial,
                "final_value": round(benchmark_final, 2),
                "return": round((benchmark_final - benchmark_initial) / benchmark_initial, 4),
                "return_pct": f"{((benchmark_final - benchmark_initial) / benchmark_initial):.2%}",
                "cagr": round(benchmark_cagr, 4),
                "cagr_pct": f"{benchmark_cagr:.2%}",
                "alpha": round(cagr - benchmark_cagr, 4),
                "alpha_pct": f"{(cagr - benchmark_cagr):.2%}"
            },
            "equity_curve": equity_curve,
            "all_trades": [
                {
                    "date": trade.date.strftime("%Y-%m-%d"),
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "shares": trade.shares,
                    "price": round(trade.price, 2),
                    "cost": round(trade.cost, 2)
                }
                for trade in trades
            ]
        }

        return performance

    def print_performance_report(self, performance: Dict):
        """Print formatted performance report."""
        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE REPORT")
        print("=" * 70)

        print(f"\nPeriod: {performance['period']['start_date']} to {performance['period']['end_date']} ({performance['period']['years']} years)")

        print("\n--- RETURNS ---")
        print(f"Initial Capital:    ${performance['returns']['initial_capital']:>12,.2f}")
        print(f"Final Equity:       ${performance['returns']['final_equity']:>12,.2f}")
        print(f"Total Return:       {performance['returns']['total_return_pct']:>13}")
        print(f"CAGR:               {performance['returns']['cagr_pct']:>13}")

        print("\n--- RISK METRICS ---")
        print(f"Sharpe Ratio:       {performance['risk_metrics']['sharpe_ratio']:>13.2f}")
        print(f"Max Drawdown:       {performance['risk_metrics']['max_drawdown_pct']:>13}")
        print(f"Volatility:         {performance['risk_metrics']['volatility_pct']:>13}")

        print("\n--- TRADING STATS ---")
        print(f"Total Trades:       {performance['trading_stats']['total_trades']:>13}")
        print(f"Win Rate:           {performance['trading_stats']['win_rate_pct']:>13}")
        print(f"Avg Win:            {performance['trading_stats']['avg_win_pct']:>13}")
        print(f"Avg Loss:           {performance['trading_stats']['avg_loss_pct']:>13}")

        print("\n--- COSTS ---")
        print(f"Total Tx Costs:     ${performance['costs']['total_transaction_costs']:>12,.2f}")
        print(f"Cost per Trade:     ${performance['costs']['avg_cost_per_trade']:>12,.2f}")

        print("\n--- BENCHMARK COMPARISON ---")
        print(f"Benchmark ({performance['benchmark']['symbol']}):   {performance['benchmark']['return_pct']:>13} ({performance['benchmark']['cagr_pct']} CAGR)")
        print(f"Strategy:           {performance['returns']['total_return_pct']:>13} ({performance['returns']['cagr_pct']} CAGR)")
        print(f"Alpha:              {performance['benchmark']['alpha_pct']:>13}")

        print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run backtest example."""
    # Initialize backtester
    backtester = Backtester(
        initial_capital=100000.0,
        transaction_costs=TransactionCosts(),
        use_ai=False  # Set to False to save API costs
    )

    # Define backtest period
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)

    # Run backtest
    performance = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date
    )

    # Print report
    backtester.print_performance_report(performance)

    # Save detailed results to JSON
    output_file = "backtest_results.json"
    with open(output_file, 'w') as f:
        json.dump(performance, f, indent=2, default=str)

    logger.info(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
