"""
Unit tests for ClaudeTrader with mocked API calls.
Tests portfolio-aware trading with percentage-based position sizing.
"""

import csv
import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set dummy env vars before importing trader
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_anthropic_key")

import trader
from trader import PortfolioState, TradeDecision


class MockBar:
    """Mock Alpaca bar data."""

    def __init__(self, date: str, o: float, h: float, l: float, c: float, v: int):
        self.timestamp = datetime.strptime(date, "%Y-%m-%d")
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v


def create_mock_bars():
    """Create mock OHLCV data for testing."""
    return [
        MockBar("2026-01-15", 580.00, 582.50, 578.00, 581.00, 50000000),
        MockBar("2026-01-16", 581.00, 585.00, 580.00, 584.50, 55000000),
        MockBar("2026-01-17", 584.50, 586.00, 583.00, 585.00, 48000000),
        MockBar("2026-01-20", 585.00, 588.00, 584.00, 587.50, 52000000),
        MockBar("2026-01-21", 587.50, 590.00, 586.00, 589.00, 60000000),
    ]


def create_mock_portfolio(
    equity: float = 100000,
    cash: float = 50000,
    buying_power: float = 100000,
    position_qty: int = 85,
    position_value: float = 50000,
    position_pct: float = 50.0,
    current_price: float = 588.00,
) -> PortfolioState:
    """Create a mock portfolio state for testing."""
    return PortfolioState(
        equity=Decimal(str(equity)),
        cash=Decimal(str(cash)),
        buying_power=Decimal(str(buying_power)),
        position_qty=position_qty,
        position_value=Decimal(str(position_value)),
        position_pct=Decimal(str(position_pct)),
        current_price=Decimal(str(current_price)),
    )


class TestGetClaudeDecision:
    """Tests for Claude decision parsing with target allocation."""

    def test_buy_decision_increase_allocation(self):
        """Test Claude recommending higher allocation triggers BUY."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 75, "reasoning": "Strong uptrend, increase exposure."}'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.target_pct == Decimal("75")
        assert decision.action == "BUY"
        assert decision.shares_to_trade > 0

    def test_sell_decision_decrease_allocation(self):
        """Test Claude recommending lower allocation triggers SELL."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 25, "reasoning": "Bearish signals, reduce exposure."}'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.target_pct == Decimal("25")
        assert decision.action == "SELL"
        assert decision.shares_to_trade > 0

    def test_hold_decision_similar_allocation(self):
        """Test small allocation difference triggers HOLD."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 50.5, "reasoning": "No clear signal, maintain position."}'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.action == "HOLD"
        assert decision.shares_to_trade == 0

    def test_full_exit_zero_allocation(self):
        """Test 0% target sells all shares."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 0, "reasoning": "Exit completely, market crash imminent."}'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0, position_qty=85)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.target_pct == Decimal("0")
        assert decision.action == "SELL"
        assert decision.shares_to_trade == 85  # Sell all

    def test_invalid_json_defaults_to_hold(self):
        """Test invalid JSON response defaults to HOLD."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I think we should wait and see...")]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        # Should maintain current allocation
        assert decision.target_pct == Decimal("50.0")
        assert decision.action == "HOLD"

    def test_json_in_code_block_extracted(self):
        """Test JSON wrapped in markdown code block is extracted."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='```json\n{"target_allocation_pct": 60, "reasoning": "Moderate bullish."}\n```'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.target_pct == Decimal("60")
        assert decision.action == "BUY"

    def test_allocation_clamped_to_100(self):
        """Test allocation over 100 is clamped."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 150, "reasoning": "Very bullish."}'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.target_pct == Decimal("100")

    def test_negative_allocation_clamped_to_zero(self):
        """Test negative allocation is clamped to 0."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": -10, "reasoning": "Short the market."}'
        )]
        mock_client.messages.create.return_value = mock_response

        portfolio = create_mock_portfolio(position_pct=50.0)
        decision = trader.get_claude_decision(mock_client, "test data", portfolio)

        assert decision.target_pct == Decimal("0")


class TestExecuteTrade:
    """Tests for trade execution with portfolio context."""

    def test_hold_no_action(self):
        """Test HOLD decision takes no action."""
        mock_client = MagicMock()
        portfolio = create_mock_portfolio()
        decision = TradeDecision(
            target_pct=Decimal("50"),
            reasoning="Maintain position",
            action="HOLD",
            shares_to_trade=0,
        )

        result = trader.execute_trade(mock_client, decision, portfolio)
        assert "HOLD" in result
        assert "Maintaining" in result
        mock_client.submit_order.assert_not_called()

    def test_buy_submits_order(self):
        """Test BUY decision submits order with correct quantity."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "buy-order-123"
        mock_client.submit_order.return_value = mock_order

        portfolio = create_mock_portfolio()
        decision = TradeDecision(
            target_pct=Decimal("75"),
            reasoning="Increase exposure",
            action="BUY",
            shares_to_trade=42,
        )

        result = trader.execute_trade(mock_client, decision, portfolio)
        assert "BUY 42 shares" in result
        assert "buy-order-123" in result
        mock_client.submit_order.assert_called_once()

    def test_sell_submits_order(self):
        """Test SELL decision submits order with correct quantity."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "sell-order-456"
        mock_client.submit_order.return_value = mock_order

        portfolio = create_mock_portfolio()
        decision = TradeDecision(
            target_pct=Decimal("25"),
            reasoning="Reduce exposure",
            action="SELL",
            shares_to_trade=42,
        )

        result = trader.execute_trade(mock_client, decision, portfolio)
        assert "SELL 42 shares" in result
        assert "sell-order-456" in result
        mock_client.submit_order.assert_called_once()

    def test_buy_zero_shares_skipped(self):
        """Test BUY with 0 shares is skipped."""
        mock_client = MagicMock()
        portfolio = create_mock_portfolio()
        decision = TradeDecision(
            target_pct=Decimal("55"),
            reasoning="Slight increase",
            action="BUY",
            shares_to_trade=0,
        )

        result = trader.execute_trade(mock_client, decision, portfolio)
        assert "HOLD" in result or "Maintaining" in result
        mock_client.submit_order.assert_not_called()

    def test_sell_zero_shares_skipped(self):
        """Test SELL with 0 shares is skipped."""
        mock_client = MagicMock()
        portfolio = create_mock_portfolio()
        decision = TradeDecision(
            target_pct=Decimal("45"),
            reasoning="Slight decrease",
            action="SELL",
            shares_to_trade=0,
        )

        result = trader.execute_trade(mock_client, decision, portfolio)
        assert "HOLD" in result or "Maintaining" in result
        mock_client.submit_order.assert_not_called()


class TestFetchOHLCVData:
    """Tests for market data fetching."""

    def test_format_output(self):
        """Test OHLCV data is formatted correctly."""
        mock_client = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {"SPY": create_mock_bars()}
        mock_client.get_stock_bars.return_value = mock_bars

        result = trader.fetch_ohlcv_data(mock_client)

        assert "SPY" in result
        assert "Open" in result
        assert "Close" in result
        assert "580.00" in result

    def test_no_data_raises_error(self):
        """Test error when no data returned."""
        mock_client = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {}
        mock_client.get_stock_bars.return_value = mock_bars

        with pytest.raises(ValueError, match="No data returned"):
            trader.fetch_ohlcv_data(mock_client)


class TestLogTrade:
    """Tests for CSV logging with portfolio context."""

    def test_creates_csv_with_header(self, tmp_path):
        """Test CSV is created with correct header."""
        csv_file = tmp_path / "test_trades.csv"
        portfolio = create_mock_portfolio()
        decision = TradeDecision(
            target_pct=Decimal("60"),
            reasoning="Bullish signal",
            action="BUY",
            shares_to_trade=17,
        )

        with patch.object(trader, "CSV_FILE", csv_file):
            trader.log_trade(decision, portfolio, "BUY 17 shares")

        assert csv_file.exists()
        with open(csv_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[0] == [
                "timestamp", "symbol", "action", "shares", "price",
                "prev_alloc_pct", "target_alloc_pct", "equity", "reasoning", "result"
            ]
            assert rows[1][2] == "BUY"
            assert rows[1][3] == "17"

    def test_appends_to_existing_csv(self, tmp_path):
        """Test trades are appended to existing CSV."""
        csv_file = tmp_path / "test_trades.csv"
        portfolio = create_mock_portfolio()

        with patch.object(trader, "CSV_FILE", csv_file):
            decision1 = TradeDecision(Decimal("60"), "First", "BUY", 10)
            trader.log_trade(decision1, portfolio, "First trade")

            decision2 = TradeDecision(Decimal("40"), "Second", "SELL", 5)
            trader.log_trade(decision2, portfolio, "Second trade")

        with open(csv_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # Header + 2 trades


class TestMarketOpen:
    """Tests for market open check."""

    def test_market_open(self):
        """Test market open returns True."""
        mock_client = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_client.get_clock.return_value = mock_clock

        assert trader.check_market_open(mock_client) is True

    def test_market_closed(self):
        """Test market closed returns False."""
        mock_client = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_client.get_clock.return_value = mock_clock

        assert trader.check_market_open(mock_client) is False


class TestGetPortfolioState:
    """Tests for portfolio state fetching."""

    def test_portfolio_with_position(self):
        """Test portfolio state with existing position."""
        mock_trading = MagicMock()
        mock_data = MagicMock()

        # Mock account
        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        mock_account.cash = "50000.00"
        mock_account.buying_power = "100000.00"
        mock_trading.get_account.return_value = mock_account

        # Mock position
        mock_position = MagicMock()
        mock_position.qty = "85"
        mock_position.market_value = "50000.00"
        mock_trading.get_open_position.return_value = mock_position

        # Mock bars for current price
        mock_bars = MagicMock()
        mock_bars.data = {"SPY": [MockBar("2026-01-21", 588, 590, 586, 588, 50000000)]}
        mock_data.get_stock_bars.return_value = mock_bars

        state = trader.get_portfolio_state(mock_trading, mock_data)

        assert state.equity == Decimal("100000.00")
        assert state.position_qty == 85
        assert state.position_pct == Decimal("50.00")

    def test_portfolio_no_position(self):
        """Test portfolio state with no position."""
        mock_trading = MagicMock()
        mock_data = MagicMock()

        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        mock_account.cash = "100000.00"
        mock_account.buying_power = "200000.00"
        mock_trading.get_account.return_value = mock_account

        # No position
        mock_trading.get_open_position.side_effect = Exception("No position")

        mock_bars = MagicMock()
        mock_bars.data = {"SPY": [MockBar("2026-01-21", 588, 590, 586, 588, 50000000)]}
        mock_data.get_stock_bars.return_value = mock_bars

        state = trader.get_portfolio_state(mock_trading, mock_data)

        assert state.position_qty == 0
        assert state.position_value == Decimal("0")
        assert state.position_pct == Decimal("0.00")


class TestDryRun:
    """Full dry run tests with all APIs mocked."""

    @patch("trader.get_alpaca_clients")
    @patch("trader.get_anthropic_client")
    def test_dry_run_buy_flow(self, mock_anthropic, mock_alpaca, tmp_path):
        """Test complete BUY flow with mocked APIs."""
        # Setup mock trading client
        mock_trading = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_trading.get_clock.return_value = mock_clock

        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        mock_account.cash = "50000.00"
        mock_account.buying_power = "100000.00"
        mock_trading.get_account.return_value = mock_account

        mock_position = MagicMock()
        mock_position.qty = "85"
        mock_position.market_value = "50000.00"
        mock_trading.get_open_position.return_value = mock_position

        mock_order = MagicMock()
        mock_order.id = "dry-run-order-789"
        mock_trading.submit_order.return_value = mock_order

        # Setup mock data client
        mock_data = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {"SPY": create_mock_bars()}
        mock_data.get_stock_bars.return_value = mock_bars

        mock_alpaca.return_value = (mock_trading, mock_data)

        # Setup mock Anthropic client
        mock_claude = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 70, "reasoning": "Strong momentum, increase position."}'
        )]
        mock_claude.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_claude

        # Run with temporary CSV
        csv_file = tmp_path / "dry_run_trades.csv"
        with patch.object(trader, "CSV_FILE", csv_file):
            result = trader.main()

        assert result == 0
        assert csv_file.exists()
        mock_trading.submit_order.assert_called_once()

    @patch("trader.get_alpaca_clients")
    @patch("trader.get_anthropic_client")
    def test_dry_run_sell_flow(self, mock_anthropic, mock_alpaca, tmp_path):
        """Test complete SELL flow with mocked APIs."""
        mock_trading = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_trading.get_clock.return_value = mock_clock

        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        mock_account.cash = "20000.00"
        mock_account.buying_power = "40000.00"
        mock_trading.get_account.return_value = mock_account

        mock_position = MagicMock()
        mock_position.qty = "136"
        mock_position.market_value = "80000.00"
        mock_trading.get_open_position.return_value = mock_position

        mock_order = MagicMock()
        mock_order.id = "sell-order-456"
        mock_trading.submit_order.return_value = mock_order

        mock_data = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {"SPY": create_mock_bars()}
        mock_data.get_stock_bars.return_value = mock_bars

        mock_alpaca.return_value = (mock_trading, mock_data)

        mock_claude = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 30, "reasoning": "Bearish divergence, reduce exposure."}'
        )]
        mock_claude.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_claude

        csv_file = tmp_path / "dry_run_trades.csv"
        with patch.object(trader, "CSV_FILE", csv_file):
            result = trader.main()

        assert result == 0
        mock_trading.submit_order.assert_called_once()

    @patch("trader.get_alpaca_clients")
    @patch("trader.get_anthropic_client")
    def test_dry_run_market_closed(self, mock_anthropic, mock_alpaca, tmp_path):
        """Test behavior when market is closed."""
        mock_trading = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_trading.get_clock.return_value = mock_clock

        mock_data = MagicMock()
        mock_alpaca.return_value = (mock_trading, mock_data)

        csv_file = tmp_path / "dry_run_trades.csv"
        with patch.object(trader, "CSV_FILE", csv_file):
            result = trader.main()

        assert result == 0
        mock_data.get_stock_bars.assert_not_called()
        mock_trading.submit_order.assert_not_called()

    @patch("trader.get_alpaca_clients")
    def test_dry_run_api_error_graceful(self, mock_alpaca, tmp_path):
        """Test graceful handling of API errors."""
        mock_alpaca.side_effect = Exception("API connection failed")

        csv_file = tmp_path / "dry_run_trades.csv"
        with patch.object(trader, "CSV_FILE", csv_file):
            result = trader.main()

        assert result == 0
        assert csv_file.exists()


class TestPositionSizingCalculations:
    """Tests for share calculation logic."""

    def test_calculate_shares_to_buy(self):
        """Test correct share calculation for increasing position."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 75, "reasoning": "Increase."}'
        )]
        mock_client.messages.create.return_value = mock_response

        # Portfolio: $100k equity, 50% in SPY ($50k), price $500
        portfolio = create_mock_portfolio(
            equity=100000,
            position_qty=100,
            position_value=50000,
            position_pct=50.0,
            current_price=500.0,
            buying_power=50000,
        )

        decision = trader.get_claude_decision(mock_client, "data", portfolio)

        # Target: 75% = $75k, Current: $50k, Diff: $25k
        # Shares needed: $25k / $500 = 50 shares
        assert decision.action == "BUY"
        assert decision.shares_to_trade == 50

    def test_calculate_shares_to_sell(self):
        """Test correct share calculation for decreasing position."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 25, "reasoning": "Decrease."}'
        )]
        mock_client.messages.create.return_value = mock_response

        # Portfolio: $100k equity, 50% in SPY ($50k), price $500
        portfolio = create_mock_portfolio(
            equity=100000,
            position_qty=100,
            position_value=50000,
            position_pct=50.0,
            current_price=500.0,
        )

        decision = trader.get_claude_decision(mock_client, "data", portfolio)

        # Target: 25% = $25k, Current: $50k, Diff: -$25k
        # Shares to sell: $25k / $500 = 50 shares
        assert decision.action == "SELL"
        assert decision.shares_to_trade == 50

    def test_buying_power_limits_purchase(self):
        """Test that buying power limits share purchase."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 100, "reasoning": "All in."}'
        )]
        mock_client.messages.create.return_value = mock_response

        # Portfolio: $100k equity, 50% in SPY, but only $10k buying power
        portfolio = create_mock_portfolio(
            equity=100000,
            position_qty=100,
            position_value=50000,
            position_pct=50.0,
            current_price=500.0,
            buying_power=10000,  # Only $10k available
        )

        decision = trader.get_claude_decision(mock_client, "data", portfolio)

        # Would need 100 shares ($50k) but only have $10k buying power
        # Max affordable: $10k / $500 = 20 shares
        assert decision.action == "BUY"
        assert decision.shares_to_trade == 20

    def test_cannot_sell_more_than_owned(self):
        """Test that can't sell more shares than owned."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"target_allocation_pct": 0, "reasoning": "Exit all."}'
        )]
        mock_client.messages.create.return_value = mock_response

        # Portfolio: 50 shares owned
        portfolio = create_mock_portfolio(
            equity=100000,
            position_qty=50,
            position_value=25000,
            position_pct=25.0,
            current_price=500.0,
        )

        decision = trader.get_claude_decision(mock_client, "data", portfolio)

        assert decision.action == "SELL"
        assert decision.shares_to_trade == 50  # Can only sell what we own
