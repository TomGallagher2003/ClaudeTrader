#!/usr/bin/env python3
"""
Generate GitHub-flavored markdown summary from JSON outputs.
Used in GitHub Actions to display results in browser.
"""

import json
import sys
from datetime import datetime


def format_backtest_summary(data: dict) -> str:
    """Format backtest results as markdown."""
    md = []

    md.append("# 📊 Backtest Results\n")

    # Period
    period = data.get('period', {})
    md.append(f"**Period:** {period.get('start_date', 'N/A')} to {period.get('end_date', 'N/A')} ({period.get('years', 'N/A')} years)\n")

    # Returns section
    returns = data.get('returns', {})
    md.append("## 💰 Returns\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Initial Capital | ${returns.get('initial_capital', 0):,.2f} |")
    md.append(f"| Final Equity | ${returns.get('final_equity', 0):,.2f} |")
    md.append(f"| Total Return | {returns.get('total_return_pct', 'N/A')} |")
    md.append(f"| CAGR | {returns.get('cagr_pct', 'N/A')} |\n")

    # Risk metrics
    risk = data.get('risk_metrics', {})
    md.append("## 📉 Risk Metrics\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Sharpe Ratio | {risk.get('sharpe_ratio', 'N/A')} |")
    md.append(f"| Max Drawdown | {risk.get('max_drawdown_pct', 'N/A')} |")
    md.append(f"| Volatility | {risk.get('volatility_pct', 'N/A')} |")

    if risk.get('peak_date') and risk.get('trough_date'):
        md.append(f"| Peak Date | {risk.get('peak_date')} |")
        md.append(f"| Trough Date | {risk.get('trough_date')} |\n")
    else:
        md.append("")

    # Trading stats
    stats = data.get('trading_stats', {})
    md.append("## 📈 Trading Statistics\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Trades | {stats.get('total_trades', 0)} |")
    md.append(f"| Winning Trades | {stats.get('winning_trades', 0)} |")
    md.append(f"| Losing Trades | {stats.get('losing_trades', 0)} |")
    md.append(f"| Win Rate | {stats.get('win_rate_pct', 'N/A')} |")
    md.append(f"| Avg Win | {stats.get('avg_win_pct', 'N/A')} |")
    md.append(f"| Avg Loss | {stats.get('avg_loss_pct', 'N/A')} |\n")

    # Costs
    costs = data.get('costs', {})
    md.append("## 💸 Transaction Costs\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Costs | ${costs.get('total_transaction_costs', 0):,.2f} |")
    md.append(f"| Cost per Trade | ${costs.get('avg_cost_per_trade', 0):,.2f} |")
    md.append(f"| Cost as % of Gains | {costs.get('cost_as_pct_of_gains', 'N/A')} |\n")

    # Benchmark comparison
    benchmark = data.get('benchmark', {})
    md.append("## 🎯 Benchmark Comparison\n")
    md.append("| Metric | Strategy | Benchmark (QQQ) |")
    md.append("|--------|----------|-----------------|")
    md.append(f"| Total Return | {returns.get('total_return_pct', 'N/A')} | {benchmark.get('return_pct', 'N/A')} |")
    md.append(f"| CAGR | {returns.get('cagr_pct', 'N/A')} | {benchmark.get('cagr_pct', 'N/A')} |")
    md.append(f"| **Alpha** | **{benchmark.get('alpha_pct', 'N/A')}** | - |\n")

    # Recent trades (last 10)
    all_trades = data.get('all_trades', [])
    if all_trades:
        md.append("## 📝 Recent Trades (Last 10)\n")
        md.append("| Date | Symbol | Action | Shares | Price |")
        md.append("|------|--------|--------|--------|-------|")
        for trade in all_trades[-10:]:
            md.append(
                f"| {trade.get('date', 'N/A')} | "
                f"{trade.get('symbol', 'N/A')} | "
                f"{trade.get('action', 'N/A')} | "
                f"{trade.get('shares', 0):,} | "
                f"${trade.get('price', 0):.2f} |"
            )
        md.append("")

    md.append("---")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}_")

    return "\n".join(md)


def format_trading_summary(data: dict) -> str:
    """Format trading decisions as markdown."""
    md = []

    md.append("# 🤖 Trading Bot Execution Summary\n")

    timestamp = data.get('timestamp', datetime.now().isoformat())
    md.append(f"**Execution Time:** {timestamp}\n")

    # Portfolio summary
    portfolio = data.get('portfolio', {})
    md.append("## 💼 Portfolio Status\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Equity | ${portfolio.get('equity', 0):,.2f} |")
    md.append(f"| Cash | ${portfolio.get('cash', 0):,.2f} |")
    md.append(f"| Buying Power | ${portfolio.get('buying_power', 0):,.2f} |\n")

    # Positions
    positions = data.get('positions', [])
    if positions:
        md.append("## 📊 Current Positions\n")
        md.append("| Symbol | Shares | Entry Price | Current Price | P&L % |")
        md.append("|--------|--------|-------------|---------------|-------|")
        for pos in positions:
            md.append(
                f"| {pos.get('symbol', 'N/A')} | "
                f"{pos.get('qty', 0):,} | "
                f"${pos.get('avg_entry_price', 0):.2f} | "
                f"${pos.get('current_price', 0):.2f} | "
                f"{pos.get('unrealized_plpc', 0):.2%} |"
            )
        md.append("")
    else:
        md.append("## 📊 Current Positions\n")
        md.append("_No positions held_\n")

    # Decisions
    decisions = data.get('decisions', [])
    if decisions:
        md.append("## 🎯 Trading Decisions\n")
        md.append("| Symbol | Decision | Rationale |")
        md.append("|--------|----------|-----------|")
        for decision in decisions:
            rationale = decision.get('ai_rationale', 'N/A')
            # Truncate long rationales
            if len(rationale) > 100:
                rationale = rationale[:97] + "..."
            md.append(
                f"| {decision.get('symbol', 'N/A')} | "
                f"{decision.get('recommendation', 'N/A')} | "
                f"{rationale} |"
            )
        md.append("")

    # Market regime
    regime = data.get('regime', {})
    if regime:
        md.append("## 🌡️ Market Regime\n")
        mode = regime.get('mode', 'UNKNOWN')
        emoji = "🟢" if mode == "OFFENSIVE" else "🔴"
        md.append(f"{emoji} **{mode}**")
        if 'spy_return_5d' in regime:
            md.append(f"- SPY 5-day return: {regime['spy_return_5d']:.2%}")
        md.append("")

    md.append("---")
    md.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}_")

    return "\n".join(md)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_summary.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Detect file type based on content
        if 'returns' in data and 'risk_metrics' in data:
            # Backtest results
            summary = format_backtest_summary(data)
        elif 'decisions' in data or 'portfolio' in data:
            # Trading decisions
            summary = format_trading_summary(data)
        else:
            print(f"# ⚠️ Unknown JSON Format\n")
            print("```json")
            print(json.dumps(data, indent=2))
            print("```")
            return

        print(summary)

    except FileNotFoundError:
        print(f"# ❌ Error\n")
        print(f"File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"# ❌ Error\n")
        print(f"Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"# ❌ Error\n")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
