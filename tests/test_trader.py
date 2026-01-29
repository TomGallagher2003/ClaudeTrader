"""
Unit tests for ClaudeTrader with mocked API calls.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set dummy env vars before importing trader
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_anthropic_key")

import trader


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


class TestGetClaudeDecision:
    """Tests for Claude decision parsing."""

    def test_buy_decision(self):
        """Test that BUY response is correctly parsed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="BUY: Upward momentum detected.")]
        mock_client.messages.create.return_value = mock_response

        decision = trader.get_claude_decision(mock_client, "test data")
        assert decision == "BUY"

    def test_sell_decision(self):
        """Test that SELL response is correctly parsed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="SELL: Bearish pattern forming.")]
        mock_client.messages.create.return_value = mock_response

        decision = trader.get_claude_decision(mock_client, "test data")
        assert decision == "SELL"

    def test_hold_decision(self):
        """Test that HOLD response is correctly parsed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="HOLD: No clear signal.")]
        mock_client.messages.create.return_value = mock_response

        decision = trader.get_claude_decision(mock_client, "test data")
        assert decision == "HOLD"

    def test_ambiguous_defaults_to_hold(self):
        """Test that ambiguous responses default to HOLD."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I think maybe we should wait...")]
        mock_client.messages.create.return_value = mock_response

        decision = trader.get_claude_decision(mock_client, "test data")
        assert decision == "HOLD"


class TestExecuteTrade:
    """Tests for trade execution logic."""

    def test_hold_no_action(self):
        """Test HOLD decision takes no action."""
        mock_client = MagicMock()
        result = trader.execute_trade(mock_client, "HOLD", 0)
        assert "No action" in result
        mock_client.submit_order.assert_not_called()

    def test_buy_with_no_position(self):
        """Test BUY when no current position."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "test-order-123"
        mock_client.submit_order.return_value = mock_order

        result = trader.execute_trade(mock_client, "BUY", 0)
        assert "BUY ORDER SUBMITTED" in result
        assert "test-order-123" in result
        mock_client.submit_order.assert_called_once()

    def test_buy_with_existing_position_skipped(self):
        """Test BUY is skipped when already holding."""
        mock_client = MagicMock()
        result = trader.execute_trade(mock_client, "BUY", 5)
        assert "SKIP BUY" in result
        assert "Already holding" in result
        mock_client.submit_order.assert_not_called()

    def test_sell_with_position(self):
        """Test SELL when holding position."""
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = "test-sell-456"
        mock_client.submit_order.return_value = mock_order

        result = trader.execute_trade(mock_client, "SELL", 3)
        assert "SELL ORDER SUBMITTED" in result
        assert "test-sell-456" in result
        mock_client.submit_order.assert_called_once()

    def test_sell_with_no_position_skipped(self):
        """Test SELL is skipped when no position."""
        mock_client = MagicMock()
        result = trader.execute_trade(mock_client, "SELL", 0)
        assert "SKIP SELL" in result
        assert "No position" in result
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
        assert "$580.00" in result  # First bar open price

    def test_no_data_raises_error(self):
        """Test error when no data returned."""
        mock_client = MagicMock()
        mock_bars = MagicMock()
        mock_bars.data = {}
        mock_client.get_stock_bars.return_value = mock_bars

        with pytest.raises(ValueError, match="No data returned"):
            trader.fetch_ohlcv_data(mock_client)


class TestLogTrade:
    """Tests for CSV logging."""

    def test_creates_csv_with_header(self, tmp_path):
        """Test CSV is created with header if it doesn't exist."""
        csv_file = tmp_path / "test_trades.csv"

        with patch.object(trader, "CSV_FILE", csv_file):
            trader.log_trade("BUY", "Test result")

        assert csv_file.exists()
        with open(csv_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[0] == ["timestamp", "symbol", "decision", "result"]
            assert rows[1][2] == "BUY"
            assert rows[1][3] == "Test result"

    def test_appends_to_existing_csv(self, tmp_path):
        """Test trades are appended to existing CSV."""
        csv_file = tmp_path / "test_trades.csv"

        with patch.object(trader, "CSV_FILE", csv_file):
            trader.log_trade("BUY", "First trade")
            trader.log_trade("SELL", "Second trade")

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


class TestDryRun:
    """Full dry run tests with all APIs mocked."""

    @patch("trader.get_alpaca_clients")
    @patch("trader.get_anthropic_client")
    def test_dry_run_full_flow(self, mock_anthropic, mock_alpaca, tmp_path):
        """Test complete trading flow with mocked APIs."""
        # Setup mock trading client
        mock_trading = MagicMock()
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_trading.get_clock.return_value = mock_clock
        mock_trading.get_open_position.side_effect = Exception("No position")

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
        mock_response.content = [MagicMock(text="BUY: Strong uptrend detected.")]
        mock_claude.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_claude

        # Run with temporary CSV
        csv_file = tmp_path / "dry_run_trades.csv"
        with patch.object(trader, "CSV_FILE", csv_file):
            result = trader.main()

        # Verify
        assert result == 0
        assert csv_file.exists()
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
        # Should not fetch data or submit orders
        mock_data.get_stock_bars.assert_not_called()
        mock_trading.submit_order.assert_not_called()

    @patch("trader.get_alpaca_clients")
    def test_dry_run_api_error_graceful(self, mock_alpaca, tmp_path):
        """Test graceful handling of API errors."""
        mock_alpaca.side_effect = Exception("API connection failed")

        csv_file = tmp_path / "dry_run_trades.csv"
        with patch.object(trader, "CSV_FILE", csv_file):
            result = trader.main()

        # Should return 0 (not fail the workflow)
        assert result == 0
        # Error should be logged
        assert csv_file.exists()
