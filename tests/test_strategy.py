"""
Unit tests for ClaudeTrader strategy functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/home/user/ClaudeTrader')

from trader import (
    Config,
    TradingMode,
    MarketData,
    StrategyFilters,
    load_symbols,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        symbols=["NVDA", "LLY"],
        benchmark="QQQ",
        regime_indicator="SPY",
        regime_threshold=-0.02,
        regime_lookback_days=5,
        relative_strength_days=14,
        atr_period=30,
        atr_volatility_threshold=0.05,
        volatility_size_multiplier=0.5,
    )


@pytest.fixture
def mock_market_data():
    """Create a mock MarketData instance."""
    return Mock(spec=MarketData)


@pytest.fixture
def strategy_filters(config, mock_market_data):
    """Create StrategyFilters with mocked market data."""
    return StrategyFilters(config, mock_market_data)


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration loading."""

    def test_default_symbols_loaded(self):
        """Test that default symbols are loaded when no file exists."""
        with patch('trader.load_symbols', return_value=["NVDA", "AVGO"]):
            config = Config()
            assert "NVDA" in config.symbols

    def test_config_thresholds(self, config):
        """Test configuration threshold values."""
        assert config.regime_threshold == -0.02
        assert config.atr_volatility_threshold == 0.05
        assert config.volatility_size_multiplier == 0.5

    def test_position_sizing_limits(self, config):
        """Test position sizing configuration."""
        assert config.base_position_pct == 0.10
        assert config.max_position_pct == 0.15
        assert config.min_cash_reserve_pct == 0.10


# =============================================================================
# REGIME DETECTION TESTS
# =============================================================================

class TestRegimeDetection:
    """Tests for SPY-based regime detection."""

    def test_defensive_mode_triggered(self, strategy_filters, mock_market_data):
        """Test that defensive mode is triggered when SPY drops >2%."""
        # SPY down 3% over 5 days
        mock_market_data.calculate_return.return_value = -0.03

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.DEFENSIVE
        mock_market_data.calculate_return.assert_called_once_with("SPY", 5)

    def test_offensive_mode_normal_market(self, strategy_filters, mock_market_data):
        """Test that offensive mode is active in normal market conditions."""
        # SPY up 1% over 5 days
        mock_market_data.calculate_return.return_value = 0.01

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.OFFENSIVE

    def test_offensive_mode_at_threshold(self, strategy_filters, mock_market_data):
        """Test that exactly -2% does not trigger defensive mode."""
        # SPY exactly at threshold
        mock_market_data.calculate_return.return_value = -0.02

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.OFFENSIVE

    def test_defensive_mode_just_below_threshold(self, strategy_filters, mock_market_data):
        """Test that just below -2% triggers defensive mode."""
        # SPY just below threshold
        mock_market_data.calculate_return.return_value = -0.0201

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.DEFENSIVE


# =============================================================================
# RELATIVE STRENGTH TESTS
# =============================================================================

class TestRelativeStrength:
    """Tests for relative strength filter (vs QQQ)."""

    def test_outperforming_stock(self, strategy_filters, mock_market_data):
        """Test detection of outperforming stock."""
        # Stock up 5%, QQQ up 2%
        mock_market_data.calculate_return.side_effect = [0.05, 0.02]

        is_outperforming, stock_ret, bench_ret = strategy_filters.check_relative_strength("NVDA")

        assert is_outperforming is True
        assert stock_ret == 0.05
        assert bench_ret == 0.02

    def test_underperforming_stock(self, strategy_filters, mock_market_data):
        """Test detection of underperforming stock."""
        # Stock up 1%, QQQ up 3%
        mock_market_data.calculate_return.side_effect = [0.01, 0.03]

        is_outperforming, stock_ret, bench_ret = strategy_filters.check_relative_strength("PLTR")

        assert is_outperforming is False
        assert stock_ret == 0.01
        assert bench_ret == 0.03

    def test_equal_performance(self, strategy_filters, mock_market_data):
        """Test behavior when stock equals benchmark."""
        # Equal returns
        mock_market_data.calculate_return.side_effect = [0.02, 0.02]

        is_outperforming, _, _ = strategy_filters.check_relative_strength("MSFT")

        assert is_outperforming is False  # Must outperform, not equal

    def test_negative_outperformance(self, strategy_filters, mock_market_data):
        """Test outperformance when both are negative."""
        # Stock down 1%, QQQ down 3%
        mock_market_data.calculate_return.side_effect = [-0.01, -0.03]

        is_outperforming, stock_ret, bench_ret = strategy_filters.check_relative_strength("LLY")

        assert is_outperforming is True  # -1% > -3%


# =============================================================================
# VOLATILITY SIZING TESTS
# =============================================================================

class TestVolatilitySizing:
    """Tests for ATR-based position sizing."""

    def test_high_volatility_reduces_position(self, strategy_filters, mock_market_data):
        """Test that high ATR reduces position size by 50%."""
        # ATR is 6% of price (above 5% threshold)
        mock_market_data.calculate_atr.return_value = (10.0, 0.06)

        multiplier, atr_pct = strategy_filters.calculate_position_multiplier("NVDA")

        assert multiplier == 0.5
        assert atr_pct == 0.06

    def test_normal_volatility_full_position(self, strategy_filters, mock_market_data):
        """Test that normal ATR allows full position."""
        # ATR is 3% of price (below 5% threshold)
        mock_market_data.calculate_atr.return_value = (5.0, 0.03)

        multiplier, atr_pct = strategy_filters.calculate_position_multiplier("LLY")

        assert multiplier == 1.0
        assert atr_pct == 0.03

    def test_exactly_at_threshold(self, strategy_filters, mock_market_data):
        """Test behavior at exactly 5% ATR threshold."""
        # ATR exactly at 5% threshold
        mock_market_data.calculate_atr.return_value = (8.0, 0.05)

        multiplier, _ = strategy_filters.calculate_position_multiplier("MSFT")

        assert multiplier == 1.0  # At threshold, not above


# =============================================================================
# ATR CALCULATION TESTS
# =============================================================================

class TestATRCalculation:
    """Tests for Average True Range calculation."""

    def test_atr_calculation_basic(self):
        """Test basic ATR calculation with sample data."""
        # Create sample OHLC data
        dates = pd.date_range(end=datetime.now(), periods=31, freq='D')
        data = pd.DataFrame({
            'open': [100] * 31,
            'high': [105] * 31,
            'low': [95] * 31,
            'close': [102] * 31,
        }, index=dates)

        # Calculate TR manually for this data
        # TR = max(high-low, |high-prev_close|, |low-prev_close|)
        # = max(10, 3, 7) = 10
        expected_atr = 10.0

        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        calculated_atr = true_range.rolling(window=30).mean().iloc[-1]

        assert abs(calculated_atr - expected_atr) < 0.01


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestStrategyIntegration:
    """Integration tests for complete strategy flow."""

    def test_buy_blocked_in_defensive_mode(self, config, mock_market_data):
        """Test that BUY is blocked when in defensive mode."""
        filters = StrategyFilters(config, mock_market_data)

        # Setup: Defensive mode (SPY down 3%)
        mock_market_data.calculate_return.return_value = -0.03

        mode, _ = filters.check_regime_mode()

        assert mode == TradingMode.DEFENSIVE
        # In actual execution, BUY would be converted to HOLD

    def test_buy_blocked_when_underperforming(self, config, mock_market_data):
        """Test that BUY is blocked when stock underperforms QQQ."""
        filters = StrategyFilters(config, mock_market_data)

        # Stock underperforms
        mock_market_data.calculate_return.side_effect = [0.01, 0.05]

        is_outperforming, _, _ = filters.check_relative_strength("PLTR")

        assert is_outperforming is False
        # In actual execution, BUY would be converted to HOLD

    def test_full_filter_chain(self, config, mock_market_data):
        """Test complete filter chain for a trade decision."""
        filters = StrategyFilters(config, mock_market_data)

        # Setup: Offensive mode
        mock_market_data.calculate_return.side_effect = [
            0.01,   # SPY 5-day return (offensive mode)
            0.08,   # NVDA 14-day return
            0.03,   # QQQ 14-day return
        ]
        mock_market_data.calculate_atr.return_value = (15.0, 0.07)  # High volatility

        # Check all filters
        mode, _ = filters.check_regime_mode()
        is_outperforming, _, _ = filters.check_relative_strength("NVDA")
        multiplier, _ = filters.calculate_position_multiplier("NVDA")

        assert mode == TradingMode.OFFENSIVE
        assert is_outperforming is True
        assert multiplier == 0.5  # High ATR reduces position


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_return(self, strategy_filters, mock_market_data):
        """Test handling of zero returns."""
        mock_market_data.calculate_return.return_value = 0.0

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.OFFENSIVE

    def test_large_positive_return(self, strategy_filters, mock_market_data):
        """Test handling of large positive returns."""
        mock_market_data.calculate_return.return_value = 0.50  # 50% gain

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.OFFENSIVE

    def test_large_negative_return(self, strategy_filters, mock_market_data):
        """Test handling of large negative returns (crash)."""
        mock_market_data.calculate_return.return_value = -0.20  # 20% crash

        mode, _ = strategy_filters.check_regime_mode()

        assert mode == TradingMode.DEFENSIVE
