#!/usr/bin/env python3
"""
Diagnostic tool for analyzing ClaudeTrader backtest performance.
Identifies filter restrictiveness, missed opportunities, and optimization suggestions.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from trader import Config, MarketData, TradingMode, load_symbols
from backtest import Backtester, TransactionCosts

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestDiagnostics:
    """Comprehensive diagnostics for backtest performance."""

    def __init__(self, backtester: Backtester, performance: Dict, historical_data: Dict):
        self.backtester = backtester
        self.performance = performance
        self.historical_data = historical_data
        self.config = backtester.config

        # Analysis results
        self.filter_analysis = {}
        self.missed_opportunities = []
        self.trade_quality = {}

    def run_full_diagnostics(self):
        """Run complete diagnostic analysis."""
        logger.info("=" * 70)
        logger.info("BACKTEST DIAGNOSTICS")
        logger.info("=" * 70)

        # 1. Analyze filter restrictiveness
        self.analyze_filter_restrictiveness()

        # 2. Review trade quality
        self.analyze_trade_quality()

        # 3. Identify missed opportunities
        self.identify_missed_opportunities()

        # 4. Generate optimization recommendations
        recommendations = self.generate_recommendations()

        return {
            "filter_analysis": self.filter_analysis,
            "trade_quality": self.trade_quality,
            "missed_opportunities": self.missed_opportunities,
            "recommendations": recommendations
        }

    def analyze_filter_restrictiveness(self):
        """Analyze how often each filter blocks trading."""
        logger.info("\n--- FILTER RESTRICTIVENESS ANALYSIS ---")

        # Get all trading dates from equity curve
        equity_curve = self.performance.get("equity_curve", [])
        if not equity_curve:
            logger.warning("No equity curve data available")
            return

        total_days = len(equity_curve)

        # Analyze regime filter
        defensive_days = 0
        spy_data = self.historical_data.get(self.config.regime_indicator)

        if spy_data is not None:
            for i in range(len(spy_data)):
                if i < self.config.regime_lookback_days:
                    continue

                recent = spy_data.iloc[i-self.config.regime_lookback_days:i+1]
                spy_return = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]

                if spy_return < self.config.regime_threshold:
                    defensive_days += 1

        defensive_pct = (defensive_days / total_days * 100) if total_days > 0 else 0

        # Analyze relative strength filter
        symbols = self.config.symbols
        benchmark_data = self.historical_data.get(self.config.benchmark)

        underperforming_counts = {symbol: 0 for symbol in symbols}
        total_evaluations = 0

        for symbol in symbols:
            symbol_data = self.historical_data.get(symbol)
            if symbol_data is None or benchmark_data is None:
                continue

            for i in range(len(symbol_data)):
                if i < self.config.relative_strength_days:
                    continue

                total_evaluations += 1

                # Calculate 14-day returns
                stock_recent = symbol_data.iloc[i-self.config.relative_strength_days:i+1]
                stock_return = (stock_recent['close'].iloc[-1] - stock_recent['close'].iloc[0]) / stock_recent['close'].iloc[0]

                bench_recent = benchmark_data.iloc[i-self.config.relative_strength_days:i+1]
                bench_return = (bench_recent['close'].iloc[-1] - bench_recent['close'].iloc[0]) / bench_recent['close'].iloc[0]

                if stock_return <= bench_return:
                    underperforming_counts[symbol] += 1

        self.filter_analysis = {
            "regime_filter": {
                "total_days": total_days,
                "defensive_days": defensive_days,
                "defensive_pct": round(defensive_pct, 2),
                "threshold": self.config.regime_threshold,
                "restrictiveness": "HIGH" if defensive_pct > 40 else "MODERATE" if defensive_pct > 20 else "LOW"
            },
            "relative_strength_filter": {
                "underperforming_by_symbol": {
                    symbol: {
                        "count": count,
                        "pct": round(count / (total_evaluations / len(symbols)) * 100, 2) if total_evaluations > 0 else 0
                    }
                    for symbol, count in underperforming_counts.items()
                },
                "avg_underperformance_pct": round(
                    sum(underperforming_counts.values()) / total_evaluations * 100, 2
                ) if total_evaluations > 0 else 0
            },
            "entry_signal_requirements": {
                "minimum_signals": 2,
                "restrictiveness": "HIGH (requires 2+ simultaneous signals)"
            }
        }

        # Print analysis
        print(f"\n📊 REGIME FILTER:")
        print(f"   Defensive mode: {defensive_days}/{total_days} days ({defensive_pct:.1f}%)")
        print(f"   Restrictiveness: {self.filter_analysis['regime_filter']['restrictiveness']}")
        print(f"   Threshold: {self.config.regime_threshold:.1%} SPY 5-day return")

        print(f"\n📊 RELATIVE STRENGTH FILTER:")
        print(f"   Average underperformance: {self.filter_analysis['relative_strength_filter']['avg_underperformance_pct']:.1f}%")
        for symbol, data in self.filter_analysis['relative_strength_filter']['underperforming_by_symbol'].items():
            print(f"   {symbol}: Underperforming {data['pct']:.1f}% of time")

        print(f"\n📊 ENTRY SIGNALS:")
        print(f"   Minimum required: 2+ signals")
        print(f"   Available signals: Momentum flip, MA crossover, RSI recovery, Volume confirmation")
        print(f"   Restrictiveness: HIGH")

    def analyze_trade_quality(self):
        """Analyze the quality of executed trades."""
        logger.info("\n--- TRADE QUALITY ANALYSIS ---")

        trades = self.performance.get("all_trades", [])
        if not trades:
            logger.warning("No trades to analyze")
            return

        # Group by symbol
        by_symbol = {}
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in by_symbol:
                by_symbol[symbol] = {"buys": [], "sells": []}

            if trade["action"] == "BUY":
                by_symbol[symbol]["buys"].append(trade)
            else:
                by_symbol[symbol]["sells"].append(trade)

        # Calculate P&L for each symbol
        symbol_pnl = {}
        for symbol, data in by_symbol.items():
            buys = data["buys"]
            sells = data["sells"]

            if not buys or not sells:
                continue

            # Simple FIFO P&L calculation
            total_pnl = 0
            total_invested = 0

            for i, sell in enumerate(sells):
                if i < len(buys):
                    buy = buys[i]
                    shares = min(sell["shares"], buy["shares"])
                    pnl = shares * (sell["price"] - buy["price"])
                    pnl_pct = (sell["price"] - buy["price"]) / buy["price"]

                    total_pnl += pnl
                    total_invested += buy["shares"] * buy["price"]

            symbol_pnl[symbol] = {
                "total_pnl": total_pnl,
                "total_invested": total_invested,
                "return_pct": (total_pnl / total_invested * 100) if total_invested > 0 else 0,
                "num_roundtrips": min(len(buys), len(sells))
            }

        # Analyze actual returns vs benchmark during holding periods
        benchmark_data = self.historical_data.get(self.config.benchmark)

        for symbol, data in by_symbol.items():
            if not data["buys"] or not data["sells"]:
                continue

            symbol_data = self.historical_data.get(symbol)
            if symbol_data is None or benchmark_data is None:
                continue

            # Get first buy and last sell dates
            first_buy = pd.to_datetime(data["buys"][0]["date"])
            last_sell = pd.to_datetime(data["sells"][-1]["date"])

            # Calculate stock return during period
            stock_start = symbol_data[symbol_data.index >= first_buy].iloc[0]['close']
            stock_end = symbol_data[symbol_data.index <= last_sell].iloc[-1]['close']
            stock_return = (stock_end - stock_start) / stock_start

            # Calculate benchmark return during same period
            bench_start = benchmark_data[benchmark_data.index >= first_buy].iloc[0]['close']
            bench_end = benchmark_data[benchmark_data.index <= last_sell].iloc[-1]['close']
            bench_return = (bench_end - bench_start) / bench_start

            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = {}

            symbol_pnl[symbol]["hold_period_return"] = stock_return * 100
            symbol_pnl[symbol]["benchmark_return"] = bench_return * 100
            symbol_pnl[symbol]["captured_pct"] = (symbol_pnl[symbol].get("return_pct", 0) / (stock_return * 100)) * 100 if stock_return != 0 else 0

        self.trade_quality = {
            "by_symbol": symbol_pnl,
            "total_roundtrips": sum(data.get("num_roundtrips", 0) for data in symbol_pnl.values()),
            "avg_trades_per_symbol": len(trades) / len(self.config.symbols)
        }

        # Print analysis
        print(f"\n📈 TRADE QUALITY BY SYMBOL:")
        for symbol, data in symbol_pnl.items():
            print(f"\n   {symbol}:")
            print(f"      Roundtrips: {data.get('num_roundtrips', 0)}")
            print(f"      Realized P&L: ${data.get('total_pnl', 0):,.0f} ({data.get('return_pct', 0):.1f}%)")
            if "hold_period_return" in data:
                print(f"      Stock rose: {data['hold_period_return']:.1f}% during holding period")
                print(f"      Benchmark rose: {data['benchmark_return']:.1f}% during same period")
                print(f"      Capture rate: {data['captured_pct']:.1f}% of available gains")

        print(f"\n   Total roundtrips: {self.trade_quality['total_roundtrips']}")
        print(f"   Avg trades per symbol: {self.trade_quality['avg_trades_per_symbol']:.1f}")

    def identify_missed_opportunities(self):
        """Identify periods where stocks rallied but we weren't holding."""
        logger.info("\n--- MISSED OPPORTUNITIES ANALYSIS ---")

        symbols = self.config.symbols
        trades = self.performance.get("all_trades", [])

        # Build position timeline for each symbol
        position_timeline = {symbol: [] for symbol in symbols}

        for trade in trades:
            symbol = trade["symbol"]
            date = pd.to_datetime(trade["date"])

            if trade["action"] == "BUY":
                position_timeline[symbol].append({"start": date, "end": None})
            elif trade["action"] == "SELL":
                # Find the most recent open position and close it
                for pos in reversed(position_timeline[symbol]):
                    if pos["end"] is None:
                        pos["end"] = date
                        break

        # Analyze periods without positions
        missed_opportunities = []

        for symbol in symbols:
            symbol_data = self.historical_data.get(symbol)
            if symbol_data is None:
                continue

            # Find periods we weren't holding
            positions = position_timeline[symbol]

            # Check each month for big moves while not holding
            for i in range(0, len(symbol_data), 20):  # Check every ~20 trading days
                if i + 20 >= len(symbol_data):
                    break

                period_start = symbol_data.index[i]
                period_end = symbol_data.index[min(i + 20, len(symbol_data) - 1)]

                # Was we holding during this period?
                holding = False
                for pos in positions:
                    if pos["start"] <= period_start and (pos["end"] is None or pos["end"] >= period_end):
                        holding = True
                        break

                if not holding:
                    # Calculate return during this period
                    start_price = symbol_data.iloc[i]['close']
                    end_price = symbol_data.iloc[min(i + 20, len(symbol_data) - 1)]['close']
                    period_return = (end_price - start_price) / start_price

                    # If return > 10%, it's a missed opportunity
                    if period_return > 0.10:
                        missed_opportunities.append({
                            "symbol": symbol,
                            "start_date": period_start.strftime("%Y-%m-%d"),
                            "end_date": period_end.strftime("%Y-%m-%d"),
                            "return": round(period_return * 100, 1),
                            "reason": self._diagnose_why_not_holding(symbol, period_start)
                        })

        # Sort by return
        missed_opportunities.sort(key=lambda x: x["return"], reverse=True)
        self.missed_opportunities = missed_opportunities[:10]  # Top 10

        # Print analysis
        print(f"\n🚨 TOP MISSED OPPORTUNITIES (>10% moves while not holding):")
        for i, opp in enumerate(self.missed_opportunities, 1):
            print(f"\n   {i}. {opp['symbol']}: +{opp['return']}%")
            print(f"      Period: {opp['start_date']} to {opp['end_date']}")
            print(f"      Likely reason: {opp['reason']}")

    def _diagnose_why_not_holding(self, symbol: str, date: pd.Timestamp) -> str:
        """Diagnose why we weren't holding a position at a given date."""
        # Check regime
        spy_data = self.historical_data.get(self.config.regime_indicator)
        if spy_data is not None:
            spy_data_upto = spy_data[spy_data.index <= date]
            if len(spy_data_upto) >= self.config.regime_lookback_days + 1:
                recent = spy_data_upto.tail(self.config.regime_lookback_days + 1)
                spy_return = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]

                if spy_return < self.config.regime_threshold:
                    return f"DEFENSIVE mode (SPY {spy_return:.1%} < {self.config.regime_threshold:.1%})"

        # Check relative strength
        symbol_data = self.historical_data.get(symbol)
        benchmark_data = self.historical_data.get(self.config.benchmark)

        if symbol_data is not None and benchmark_data is not None:
            symbol_data_upto = symbol_data[symbol_data.index <= date]
            benchmark_data_upto = benchmark_data[benchmark_data.index <= date]

            if len(symbol_data_upto) >= self.config.relative_strength_days + 1 and len(benchmark_data_upto) >= self.config.relative_strength_days + 1:
                stock_recent = symbol_data_upto.tail(self.config.relative_strength_days + 1)
                stock_return = (stock_recent['close'].iloc[-1] - stock_recent['close'].iloc[0]) / stock_recent['close'].iloc[0]

                bench_recent = benchmark_data_upto.tail(self.config.relative_strength_days + 1)
                bench_return = (bench_recent['close'].iloc[-1] - bench_recent['close'].iloc[0]) / bench_recent['close'].iloc[0]

                if stock_return <= bench_return:
                    return f"Underperforming benchmark ({stock_return:.1%} vs {bench_return:.1%})"

        return "Insufficient entry signals (<2 signals)"

    def generate_recommendations(self) -> List[Dict]:
        """Generate specific recommendations for reducing conservativeness."""
        logger.info("\n--- OPTIMIZATION RECOMMENDATIONS ---")

        recommendations = []

        # Recommendation 1: Regime filter
        regime = self.filter_analysis.get("regime_filter", {})
        if regime.get("defensive_pct", 0) > 30:
            recommendations.append({
                "category": "REGIME FILTER",
                "severity": "HIGH",
                "current": f"{regime['threshold']:.1%} threshold, defensive {regime['defensive_pct']:.0f}% of time",
                "recommendation": "Relax regime threshold from -2% to -3% or -4%",
                "expected_impact": "Reduce defensive periods by 30-50%, enabling more trades",
                "config_change": "regime_threshold: -0.02 → -0.03 or -0.04"
            })

        # Recommendation 2: Relative strength filter
        rs_filter = self.filter_analysis.get("relative_strength_filter", {})
        avg_underperf = rs_filter.get("avg_underperformance_pct", 0)
        if avg_underperf > 40:
            recommendations.append({
                "category": "RELATIVE STRENGTH",
                "severity": "HIGH",
                "current": f"Stocks underperform {avg_underperf:.0f}% of time, blocks all buys",
                "recommendation": "Allow buying if within -5% of benchmark (vs requiring outperformance)",
                "expected_impact": "Increase buy opportunities by 50-100%",
                "config_change": "Add tolerance: is_outperforming = stock_return > benchmark_return - 0.05"
            })

        # Recommendation 3: Entry signals
        total_roundtrips = self.trade_quality.get("total_roundtrips", 0)
        if total_roundtrips < 15:
            recommendations.append({
                "category": "ENTRY SIGNALS",
                "severity": "HIGH",
                "current": "Requires 2+ simultaneous signals, resulting in very few trades",
                "recommendation": "Reduce requirement to 1+ signal (WEAK strength minimum)",
                "expected_impact": "Increase trade frequency by 100-200%",
                "config_change": "In _process_symbol: entry_signals['signal_count'] >= 1 (was >= 2)"
            })

        # Recommendation 4: Position sizing
        recommendations.append({
            "category": "POSITION SIZING",
            "severity": "MEDIUM",
            "current": "10% base position, 15% max",
            "recommendation": "Increase to 12% base, 20% max for high-conviction plays",
            "expected_impact": "Capture more upside from winners",
            "config_change": "base_position_pct: 0.10 → 0.12, max_position_pct: 0.15 → 0.20"
        })

        # Recommendation 5: Profit-taking
        captured_rates = [
            data.get("captured_pct", 0)
            for data in self.trade_quality.get("by_symbol", {}).values()
        ]
        if captured_rates and np.mean([r for r in captured_rates if r > 0]) < 50:
            recommendations.append({
                "category": "PROFIT-TAKING",
                "severity": "MEDIUM",
                "current": "Scaling out at +15% may be too early, capturing <50% of moves",
                "recommendation": "Increase profit target to +20% or +25%",
                "expected_impact": "Capture 20-30% more gains from winning positions",
                "config_change": "In _check_profit_taking: unrealized_plpc >= 0.20 (was >= 0.15)"
            })

        # Recommendation 6: Stop loss
        win_rate = self.performance.get("trading_stats", {}).get("win_rate", 0)
        if win_rate < 0.55:
            recommendations.append({
                "category": "STOP LOSS",
                "severity": "LOW",
                "current": "8% stop loss may be too tight for volatile AI stocks",
                "recommendation": "Widen to 10% or 12% for more breathing room",
                "expected_impact": "Reduce premature stops by 20-30%",
                "config_change": "stop_loss_pct: 0.08 → 0.10 or 0.12"
            })

        # Print recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            severity_emoji = "🔴" if rec["severity"] == "HIGH" else "🟡" if rec["severity"] == "MEDIUM" else "🟢"
            print(f"{i}. {severity_emoji} {rec['category']} ({rec['severity']} priority)")
            print(f"   Current: {rec['current']}")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Expected impact: {rec['expected_impact']}")
            print(f"   Config change: {rec['config_change']}")
            print()

        return recommendations


def main():
    """Run backtest diagnostics."""
    # Load config
    config = Config()

    # Get backtest parameters
    start_date_str = os.getenv('BACKTEST_START_DATE', '2023-01-01')
    end_date_str = os.getenv('BACKTEST_END_DATE', '2025-12-31')

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(
        config=config,
        initial_capital=100000,
        transaction_costs=TransactionCosts(),
        use_ai=False
    )

    performance = backtester.run_backtest(start_date, end_date)

    # Load historical data for analysis
    logger.info("Loading historical data for diagnostics...")
    market_data = MarketData()
    days = (end_date - start_date).days + 60

    historical_data = {}
    for symbol in config.symbols + [config.benchmark, config.regime_indicator]:
        try:
            df = market_data.get_historical_bars(symbol, days)
            # Filter to date range
            start_ts = pd.Timestamp(start_date, tz='UTC')
            end_ts = pd.Timestamp(end_date, tz='UTC')
            df = df[df.index >= start_ts]
            df = df[df.index <= end_ts]
            historical_data[symbol] = df
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")

    # Run diagnostics
    diagnostics = BacktestDiagnostics(backtester, performance, historical_data)
    results = diagnostics.run_full_diagnostics()

    # Save results
    output_file = "backtest_diagnostics.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n✅ Diagnostics complete! Results saved to {output_file}")

    return results


if __name__ == "__main__":
    main()
