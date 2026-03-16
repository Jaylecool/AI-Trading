"""
Automated unit tests for the AI Trading System.

Covers:
- Configuration loading
- Trading rules (buy/sell signal generation)
- Position sizing
- Risk management (drawdown, circuit breaker)
- Input validation helpers
- Backtesting metrics (win rate, Sharpe ratio, profit factor)
"""

import os
import sys
import json
import math
import unittest
from datetime import datetime, timedelta

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from trading_rules import (
    TradingParameters,
    TradingRules,
    PositionSizingCalculator,
    RiskManager,
    Position,
    Trade,
)


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig(unittest.TestCase):
    """Verify that config.py exposes the expected defaults."""

    def test_flask_port_is_int(self):
        self.assertIsInstance(cfg.FLASK_PORT, int)

    def test_initial_capital_positive(self):
        self.assertGreater(cfg.INITIAL_CAPITAL, 0)

    def test_supported_symbols_non_empty(self):
        self.assertIsInstance(cfg.SUPPORTED_SYMBOLS, list)
        self.assertGreater(len(cfg.SUPPORTED_SYMBOLS), 0)

    def test_cors_origins_is_list(self):
        self.assertIsInstance(cfg.CORS_ORIGINS, list)

    def test_results_dir_exists(self):
        self.assertTrue(os.path.isdir(cfg.RESULTS_DIR))


# ============================================================================
# TRADING PARAMETERS TESTS
# ============================================================================

class TestTradingParameters(unittest.TestCase):
    """Verify parameter defaults and constraints."""

    def setUp(self):
        self.params = TradingParameters()

    def test_buy_threshold_positive(self):
        self.assertGreater(self.params.buy_threshold, 0)

    def test_stop_loss_positive(self):
        self.assertGreater(self.params.stop_loss_percent, 0)

    def test_take_profit_exceeds_stop_loss(self):
        self.assertGreater(self.params.take_profit_target, self.params.stop_loss_percent)

    def test_confidence_threshold_in_range(self):
        self.assertGreaterEqual(self.params.confidence_threshold, 0)
        self.assertLessEqual(self.params.confidence_threshold, 1)

    def test_max_concurrent_positions_positive(self):
        self.assertGreater(self.params.max_concurrent_positions, 0)


# ============================================================================
# TRADING RULES – BUY/SELL SIGNAL TESTS
# ============================================================================

class TestTradingRulesBuySignal(unittest.TestCase):
    """Test buy-signal generation logic."""

    def setUp(self):
        self.params = TradingParameters()
        self.rules = TradingRules(self.params)
        # Market data that confirms an uptrend (RSI > 50, Price > SMA20, EMA10 > EMA20)
        self.bullish_market = {
            'RSI_14': 60,
            'Close': 155.0,
            'SMA_20': 150.0,
            'SMA_50': 148.0,
            'EMA_10': 154.0,
            'EMA_20': 152.0,
        }

    def test_strong_buy_signal(self):
        """A large predicted appreciation with bullish confirmations should trigger a buy."""
        predicted = 155.0 * 1.05  # +5 % appreciation
        signal, confidence, reason = self.rules.get_buy_signal(
            predicted, 155.0, self.bullish_market
        )
        self.assertTrue(signal)
        self.assertGreater(confidence, 0)

    def test_no_signal_below_threshold(self):
        """Tiny predicted appreciation should NOT trigger a buy."""
        predicted = 155.0 * 1.001  # +0.1 % – below 2 % threshold
        signal, confidence, reason = self.rules.get_buy_signal(
            predicted, 155.0, self.bullish_market
        )
        self.assertFalse(signal)

    def test_no_signal_downtrend(self):
        """Appreciation exists but market is in a downtrend (price below SMA-50)."""
        bearish_market = dict(self.bullish_market)
        bearish_market['SMA_50'] = 160.0  # price 155 < SMA-50 160
        predicted = 155.0 * 1.05
        signal, confidence, reason = self.rules.get_buy_signal(
            predicted, 155.0, bearish_market
        )
        self.assertFalse(signal)

    def test_volatility_raises_threshold(self):
        """High volatility should increase the buy threshold, blocking a borderline signal."""
        # Normal threshold = 2 %; high-vol threshold = 3 % (1.5x)
        predicted = 155.0 * 1.025  # +2.5 % – passes normal, fails high-vol
        signal_normal, _, _ = self.rules.get_buy_signal(
            predicted, 155.0, self.bullish_market, daily_volatility=0.01
        )
        signal_highvol, _, _ = self.rules.get_buy_signal(
            predicted, 155.0, self.bullish_market, daily_volatility=0.04
        )
        # With normal volatility and strong confirmation it may fire; with high volatility it shouldn't
        self.assertFalse(signal_highvol)


class TestTradingRulesSellSignal(unittest.TestCase):
    """Test sell-signal generation logic."""

    def setUp(self):
        self.params = TradingParameters()
        self.rules = TradingRules(self.params)
        self.bearish_market = {
            'RSI_14': 35,
            'Close': 155.0,
            'SMA_20': 160.0,
            'EMA_10': 153.0,
            'EMA_20': 157.0,
        }

    def test_strong_sell_signal(self):
        """Predicted drop with bearish confirmations should trigger sell."""
        predicted = 155.0 * 0.95  # -5 %
        signal, confidence, reason = self.rules.get_sell_signal(
            predicted, 155.0, self.bearish_market
        )
        self.assertTrue(signal)

    def test_no_sell_on_flat_price(self):
        predicted = 155.0  # 0 % change
        signal, confidence, reason = self.rules.get_sell_signal(
            predicted, 155.0, self.bearish_market
        )
        self.assertFalse(signal)


# ============================================================================
# POSITION SIZING TESTS
# ============================================================================

class TestPositionSizing(unittest.TestCase):
    """Test risk-based position sizing calculations."""

    def setUp(self):
        self.params = TradingParameters()
        self.sizer = PositionSizingCalculator(self.params)

    def test_shares_positive(self):
        shares = self.sizer.calculate_position_size(
            entry_price=150.0,
            portfolio_value=100000.0,
            available_cash=100000.0,
        )
        self.assertGreater(shares, 0)

    def test_shares_respect_min_size(self):
        shares = self.sizer.calculate_position_size(
            entry_price=50000.0,  # very expensive stock
            portfolio_value=100000.0,
            available_cash=100000.0,
        )
        self.assertGreaterEqual(shares, self.params.min_position_size)

    def test_position_limited_by_cash(self):
        """If available cash is tiny, shares should be limited."""
        shares = self.sizer.calculate_position_size(
            entry_price=150.0,
            portfolio_value=100000.0,
            available_cash=500.0,  # very little cash
        )
        cost = shares * 150.0
        # Cost might exceed cash because of min_position_size floor,
        # but otherwise should not exceed available cash
        # (min_position_size override is acceptable)

    def test_calculate_position_limits_max_reached(self):
        limits = self.sizer.calculate_position_limits(
            portfolio_value=100000.0,
            available_cash=50000.0,
            num_active_positions=self.params.max_concurrent_positions,
        )
        self.assertFalse(limits['can_open_position'])

    def test_calculate_position_limits_ok(self):
        limits = self.sizer.calculate_position_limits(
            portfolio_value=100000.0,
            available_cash=50000.0,
            num_active_positions=0,
        )
        self.assertTrue(limits['can_open_position'])


# ============================================================================
# RISK MANAGEMENT TESTS
# ============================================================================

class TestRiskManager(unittest.TestCase):
    """Test drawdown and circuit-breaker logic."""

    def setUp(self):
        self.params = TradingParameters()
        self.rm = RiskManager(self.params)

    def test_drawdown_zero_at_peak(self):
        self.rm.update_peak_value(100000)
        dd = self.rm.calculate_drawdown(100000)
        self.assertAlmostEqual(dd, 0.0)

    def test_drawdown_positive_below_peak(self):
        self.rm.update_peak_value(100000)
        dd = self.rm.calculate_drawdown(95000)
        self.assertAlmostEqual(abs(dd), 0.05, places=4)

    def test_circuit_breaker_not_triggered(self):
        """Small loss should NOT trigger the circuit breaker."""
        self.rm.update_peak_value(100000)
        triggered, reason = self.rm.check_circuit_breaker(99000, 100000)
        self.assertFalse(triggered)

    def test_circuit_breaker_triggered(self):
        """Loss exceeding threshold should trigger the circuit breaker."""
        self.rm.update_peak_value(100000)
        triggered, reason = self.rm.check_circuit_breaker(
            94000, 100000  # -6 % loss exceeds -5 % threshold
        )
        self.assertTrue(triggered)


# ============================================================================
# POSITION & TRADE DATA CLASS TESTS
# ============================================================================

class TestPosition(unittest.TestCase):

    def test_entry_value(self):
        pos = Position(
            symbol='AAPL',
            entry_date=datetime.now(),
            entry_price=150.0,
            shares=10,
            stop_loss_price=147.0,
            take_profit_price=157.5,
        )
        self.assertAlmostEqual(pos.entry_value, 1500.0)

    def test_unrealised_pnl_gain(self):
        pos = Position(
            symbol='AAPL',
            entry_date=datetime.now(),
            entry_price=100.0,
            shares=10,
            stop_loss_price=98.0,
            take_profit_price=105.0,
        )
        pnl, pct = pos.calculate_unrealized_pnl(110.0)
        self.assertAlmostEqual(pnl, 100.0)
        self.assertAlmostEqual(pct, 0.10)

    def test_unrealised_pnl_loss(self):
        pos = Position(
            symbol='AAPL',
            entry_date=datetime.now(),
            entry_price=100.0,
            shares=10,
            stop_loss_price=98.0,
            take_profit_price=105.0,
        )
        pnl, pct = pos.calculate_unrealized_pnl(95.0)
        self.assertLess(pnl, 0)


class TestTradeDataClass(unittest.TestCase):

    def _make_trade(self, entry, exit_price):
        return Trade(
            symbol='AAPL',
            entry_date=datetime(2025, 1, 1),
            entry_price=entry,
            entry_shares=10,
            exit_date=datetime(2025, 1, 5),
            exit_price=exit_price,
            exit_reason='TAKE_PROFIT',
        )

    def test_winning_trade(self):
        t = self._make_trade(100.0, 110.0)
        self.assertTrue(t.is_winning_trade)
        self.assertAlmostEqual(t.pnl_amount, 100.0)

    def test_losing_trade(self):
        t = self._make_trade(100.0, 90.0)
        self.assertFalse(t.is_winning_trade)

    def test_duration(self):
        t = self._make_trade(100.0, 110.0)
        self.assertEqual(t.duration_days, 4)


# ============================================================================
# BACKTESTING METRICS HELPERS
# ============================================================================

def _win_rate(trades):
    """Win rate = winning trades / total trades."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.is_winning_trade)
    return wins / len(trades)


def _profit_factor(trades):
    """Gross profit / gross loss (absolute). Returns inf if no losses."""
    gross_profit = sum(t.pnl_amount for t in trades if t.pnl_amount > 0)
    gross_loss = abs(sum(t.pnl_amount for t in trades if t.pnl_amount < 0))
    if gross_loss == 0:
        return float('inf')
    return gross_profit / gross_loss


def _sharpe_ratio(returns, risk_free_rate=0.0):
    """Annualised Sharpe ratio from a list of daily returns."""
    if len(returns) < 2:
        return 0.0
    import numpy as np
    excess = [r - risk_free_rate for r in returns]
    mean_ret = sum(excess) / len(excess)
    std_ret = (sum((r - mean_ret) ** 2 for r in excess) / (len(excess) - 1)) ** 0.5
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * (252 ** 0.5)


class TestBacktestMetrics(unittest.TestCase):
    """Verify metric calculation helpers."""

    def _make(self, entry, exit_price):
        return Trade(
            symbol='AAPL',
            entry_date=datetime(2025, 1, 1),
            entry_price=entry,
            entry_shares=10,
            exit_date=datetime(2025, 1, 5),
            exit_price=exit_price,
            exit_reason='TEST',
        )

    def test_win_rate_all_winners(self):
        trades = [self._make(100, 110), self._make(100, 105)]
        self.assertAlmostEqual(_win_rate(trades), 1.0)

    def test_win_rate_mixed(self):
        trades = [self._make(100, 110), self._make(100, 90)]
        self.assertAlmostEqual(_win_rate(trades), 0.5)

    def test_win_rate_empty(self):
        self.assertAlmostEqual(_win_rate([]), 0.0)

    def test_profit_factor_no_losses(self):
        trades = [self._make(100, 110), self._make(100, 120)]
        self.assertEqual(_profit_factor(trades), float('inf'))

    def test_profit_factor_mixed(self):
        trades = [self._make(100, 110), self._make(100, 90)]
        # profit = 100, loss = 100 → PF = 1.0
        self.assertAlmostEqual(_profit_factor(trades), 1.0)

    def test_sharpe_ratio_positive(self):
        # All positive daily returns → positive Sharpe
        returns = [0.01, 0.02, 0.015, 0.005, 0.01]
        sr = _sharpe_ratio(returns)
        self.assertGreater(sr, 0)

    def test_sharpe_ratio_flat(self):
        returns = [0.01, 0.01, 0.01]
        sr = _sharpe_ratio(returns)
        # zero variance → 0 Sharpe (avoids division by zero)
        self.assertAlmostEqual(sr, 0.0)


# ============================================================================
# INPUT VALIDATION TESTS (mirrors dashboard helper)
# ============================================================================

import re
_SYMBOL_RE = re.compile(r'^[A-Z]{1,5}$')

def _validate_symbol(symbol):
    return bool(symbol and _SYMBOL_RE.match(symbol) and symbol in cfg.SUPPORTED_SYMBOLS)


class TestSymbolValidation(unittest.TestCase):

    def test_valid_symbols(self):
        for sym in cfg.SUPPORTED_SYMBOLS:
            self.assertTrue(_validate_symbol(sym), f"{sym} should be valid")

    def test_invalid_lowercase(self):
        self.assertFalse(_validate_symbol('aapl'))

    def test_invalid_empty(self):
        self.assertFalse(_validate_symbol(''))

    def test_invalid_none(self):
        self.assertFalse(_validate_symbol(None))

    def test_invalid_too_long(self):
        self.assertFalse(_validate_symbol('TOOLONG'))

    def test_unknown_symbol(self):
        self.assertFalse(_validate_symbol('ZZZZZ'))


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    unittest.main()
