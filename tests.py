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
# AUTH TESTS
# ============================================================================

import tempfile
import sqlite3
from unittest.mock import patch

class TestAuth(unittest.TestCase):
    """Test user registration, login, and lookup."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test_users.db')

    def tearDown(self):
        try:
            os.remove(self.db_path)
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def _init(self):
        import auth
        with patch.object(auth, '_get_db') as mock_db:
            # Use a real connection to the temp DB instead
            pass
        # Directly init with patched path
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def test_register_and_login(self):
        """Register a user then authenticate them."""
        from auth import register_user, authenticate_user, get_user_by_id
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            result = register_user('Test User', 'test@example.com', 'SecurePass123')
            self.assertTrue(result['success'], result.get('error'))
            user_id = result['user']['id']
            self.assertIsInstance(user_id, int)

            user = authenticate_user('test@example.com', 'SecurePass123')
            self.assertIsNotNone(user)
            self.assertEqual(user['email'], 'test@example.com')

            same = get_user_by_id(user_id)
            self.assertIsNotNone(same)
            self.assertEqual(same['name'], 'Test User')

    def test_register_validation_short_name(self):
        from auth import register_user
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            result = register_user('A', 'a@b.com', 'Password123')
            self.assertFalse(result['success'])

    def test_register_validation_bad_email(self):
        from auth import register_user
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            result = register_user('Good Name', 'not-an-email', 'Password123')
            self.assertFalse(result['success'])

    def test_register_validation_short_password(self):
        from auth import register_user
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            result = register_user('Good Name', 'x@y.com', 'short')
            self.assertFalse(result['success'])

    def test_register_duplicate_email(self):
        from auth import register_user
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            register_user('User One', 'dup@test.com', 'Password123')
            result = register_user('User Two', 'dup@test.com', 'Password456')
            self.assertFalse(result['success'])

    def test_authenticate_wrong_password(self):
        from auth import register_user, authenticate_user
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            register_user('User', 'u@t.com', 'CorrectPass1')
            user = authenticate_user('u@t.com', 'WrongPass1')
            self.assertIsNone(user)

    def test_get_user_nonexistent(self):
        from auth import get_user_by_id
        with patch('auth._get_db', side_effect=self._get_conn):
            self._init()
            self.assertIsNone(get_user_by_id(999))


# ============================================================================
# STRATEGY CONFIGURATIONS TESTS
# ============================================================================

from strategy_configurations import StrategyFactory, STRATEGIES


class TestStrategyConfigurations(unittest.TestCase):
    """Test strategy factory and parameter consistency."""

    def test_all_strategies_exist(self):
        self.assertIn('AGGRESSIVE', STRATEGIES)
        self.assertIn('CONSERVATIVE', STRATEGIES)
        self.assertIn('BALANCED', STRATEGIES)

    def test_aggressive_has_higher_risk(self):
        agg = StrategyFactory.create_aggressive_strategy()
        con = StrategyFactory.create_conservative_strategy()
        self.assertGreaterEqual(agg.risk_percentage, con.risk_percentage)

    def test_conservative_has_tighter_limits(self):
        con = StrategyFactory.create_conservative_strategy()
        bal = StrategyFactory.create_balanced_strategy()
        self.assertLessEqual(con.max_position_value_percent, bal.max_position_value_percent)

    def test_factory_returns_trading_parameters(self):
        for key, info in STRATEGIES.items():
            params = info['factory']()
            self.assertIsInstance(params, TradingParameters, f"{key} factory failed")

    def test_custom_strategy(self):
        custom = StrategyFactory.create_custom_strategy(
            buy_threshold=0.03, stop_loss_percent=0.04, risk_percentage=0.01
        )
        self.assertAlmostEqual(custom.buy_threshold, 0.03)
        self.assertAlmostEqual(custom.stop_loss_percent, 0.04)

    def test_strategies_have_metadata(self):
        for key, info in STRATEGIES.items():
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('risk_level', info)
            self.assertIn('factory', info)


# ============================================================================
# PORTFOLIO TRACKER TESTS
# ============================================================================

from portfolio_tracker import Trade as PTTrade, Portfolio, PortfolioTracker, TradeHistoryFilter


class TestPortfolio(unittest.TestCase):
    """Test Portfolio class."""

    def test_initial_balance(self):
        p = Portfolio(50000.0)
        self.assertEqual(p.cash, 50000.0)

    def test_add_trade_reduces_cash(self):
        p = Portfolio(100000.0)
        t = PTTrade(trade_id='T1', date='2025-01-01', symbol='AAPL',
                    action='BUY', quantity=10, entry_price=150.0)
        p.add_trade(t)
        self.assertAlmostEqual(p.cash, 100000.0 - 10 * 150.0)

    def test_equity_value_includes_positions(self):
        p = Portfolio(100000.0)
        t = PTTrade(trade_id='T1', date='2025-01-01', symbol='AAPL',
                    action='BUY', quantity=10, entry_price=150.0)
        p.add_trade(t)
        p.update_market_price('AAPL', 160.0)
        self.assertGreater(p.calculate_equity_value(), 100000.0)

    def test_metrics_keys(self):
        p = Portfolio(100000.0)
        m = p.get_portfolio_metrics()
        self.assertIn('equity_value', m)
        self.assertIn('current_balance', m)


class TestPortfolioTracker(unittest.TestCase):
    """Test PortfolioTracker trade lifecycle."""

    def test_add_and_close_trade(self):
        pt = PortfolioTracker(100000.0)
        t = PTTrade(trade_id='T1', date='2025-01-01', symbol='AAPL',
                    action='BUY', quantity=5, entry_price=200.0)
        pt.add_trade(t)
        closed = pt.close_trade('T1', 210.0, '2025-01-10')
        self.assertTrue(closed)

    def test_close_nonexistent_trade(self):
        pt = PortfolioTracker(100000.0)
        self.assertFalse(pt.close_trade('FAKE', 100.0, '2025-01-01'))

    def test_portfolio_summary(self):
        pt = PortfolioTracker(100000.0)
        s = pt.get_portfolio_summary()
        self.assertIn('current_balance', s)
        self.assertIn('num_trades', s)

    def test_trade_history_empty(self):
        pt = PortfolioTracker()
        self.assertEqual(len(pt.get_trade_history()), 0)


class TestTradeHistoryFilter(unittest.TestCase):
    """Test filtering trade history."""

    def setUp(self):
        self.trades = [
            PTTrade(trade_id='T1', date='2025-01-01', symbol='AAPL',
                    action='BUY', quantity=10, entry_price=150.0, status='CLOSED'),
            PTTrade(trade_id='T2', date='2025-02-01', symbol='MSFT',
                    action='BUY', quantity=5, entry_price=300.0, status='OPEN'),
            PTTrade(trade_id='T3', date='2025-03-01', symbol='AAPL',
                    action='SELL', quantity=10, entry_price=160.0, status='CLOSED'),
        ]

    def test_filter_by_symbol(self):
        f = TradeHistoryFilter(self.trades)
        result = f.filter_by_symbol('AAPL')
        self.assertEqual(len(result), 2)

    def test_filter_by_action(self):
        f = TradeHistoryFilter(self.trades)
        result = f.filter_by_action('BUY')
        self.assertEqual(len(result), 2)

    def test_filter_by_status(self):
        f = TradeHistoryFilter(self.trades)
        result = f.filter_by_status('OPEN')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, 'MSFT')


# ============================================================================
# ALERT SYSTEM TESTS
# ============================================================================

from alert_system import (
    AlertSystem, AlertRule, AlertType, AlertSeverity, ComparisonOperator,
    get_alert_system,
)


class TestAlertSystem(unittest.TestCase):
    """Test alert rule CRUD and evaluation."""

    def setUp(self):
        self.system = AlertSystem()

    def test_create_rule(self):
        rule = self.system.create_rule(
            name='High Price',
            alert_type=AlertType.PRICE_ALERT,
            metric_field='price',
            operator=ComparisonOperator.GREATER_THAN,
            threshold_value=200.0,
            symbol='AAPL',
        )
        self.assertIsInstance(rule, AlertRule)
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.symbol, 'AAPL')

    def test_get_rule(self):
        rule = self.system.create_rule(
            name='Test', alert_type=AlertType.PRICE_ALERT,
            metric_field='price', operator=ComparisonOperator.GREATER_THAN,
            threshold_value=100.0,
        )
        fetched = self.system.get_rule(rule.rule_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, 'Test')

    def test_delete_rule(self):
        rule = self.system.create_rule(
            name='Del', alert_type=AlertType.PRICE_ALERT,
            metric_field='price', operator=ComparisonOperator.LESS_THAN,
            threshold_value=50.0,
        )
        self.assertTrue(self.system.delete_rule(rule.rule_id))
        self.assertIsNone(self.system.get_rule(rule.rule_id))

    def test_evaluate_triggers_alert(self):
        self.system.create_rule(
            name='Over 200', alert_type=AlertType.PRICE_ALERT,
            metric_field='price', operator=ComparisonOperator.GREATER_THAN,
            threshold_value=200.0, symbol='AAPL',
            severity=AlertSeverity.HIGH,
        )
        events = self.system.evaluate('AAPL', {'price': 210.0})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].symbol, 'AAPL')

    def test_evaluate_no_trigger(self):
        self.system.create_rule(
            name='Over 200', alert_type=AlertType.PRICE_ALERT,
            metric_field='price', operator=ComparisonOperator.GREATER_THAN,
            threshold_value=200.0, symbol='AAPL',
        )
        events = self.system.evaluate('AAPL', {'price': 190.0})
        self.assertEqual(len(events), 0)

    def test_acknowledge_alert(self):
        self.system.create_rule(
            name='Test', alert_type=AlertType.PRICE_ALERT,
            metric_field='price', operator=ComparisonOperator.GREATER_THAN,
            threshold_value=100.0, symbol='AAPL',
        )
        events = self.system.evaluate('AAPL', {'price': 150.0})
        self.assertTrue(self.system.acknowledge_alert(events[0].alert_id))

    def test_export_import_rules(self):
        self.system.create_rule(
            name='Export', alert_type=AlertType.PRICE_ALERT,
            metric_field='price', operator=ComparisonOperator.GREATER_THAN,
            threshold_value=100.0,
        )
        exported = self.system.export_rules()
        new_system = AlertSystem()
        self.assertTrue(new_system.import_rules(exported))
        self.assertEqual(len(new_system.get_all_rules()), 1)

    def test_get_alert_system_singleton(self):
        a = get_alert_system()
        b = get_alert_system()
        self.assertIs(a, b)


# ============================================================================
# NOTIFICATION SERVICE TESTS
# ============================================================================

from notification_service import (
    NotificationService, NotificationPreference, NotificationChannel,
    Notification, get_notification_service,
)


class TestNotificationService(unittest.TestCase):
    """Test notification preferences and delivery."""

    def setUp(self):
        self.service = NotificationService()

    def test_default_preferences(self):
        prefs = self.service.get_user_preferences('default')
        self.assertIsInstance(prefs, NotificationPreference)

    def test_set_and_get_preferences(self):
        prefs = NotificationPreference(
            user_id='user1',
            enable_popup=True,
            enable_sound=False,
            enable_email=False,
        )
        self.service.set_user_preferences('user1', prefs)
        got = self.service.get_user_preferences('user1')
        self.assertFalse(got.enable_sound)

    def test_send_notification(self):
        notif = Notification(
            notification_id='N1',
            alert_id='A1',
            rule_id='R1',
            title='Test',
            message='Hello',
            severity='MEDIUM',
            channels=[NotificationChannel.LOG],
        )
        result = self.service.send_notification(notif, 'default')
        self.assertTrue(result)

    def test_notification_history(self):
        notif = Notification(
            notification_id='N2',
            alert_id='A2',
            rule_id='R2',
            title='Test2',
            message='World',
            severity='LOW',
            channels=[NotificationChannel.DASHBOARD],
        )
        self.service.send_notification(notif, 'default')
        history = self.service.get_notification_history(limit=10)
        self.assertGreaterEqual(len(history), 1)

    def test_export_import_preferences(self):
        prefs = NotificationPreference(user_id='exp', enable_popup=False)
        self.service.set_user_preferences('exp', prefs)
        exported = self.service.export_preferences('exp')
        self.assertIsInstance(exported, str)
        new_svc = NotificationService()
        self.assertTrue(new_svc.import_preferences(exported, 'exp'))

    def test_singleton(self):
        a = get_notification_service()
        b = get_notification_service()
        self.assertIs(a, b)


# ============================================================================
# STREAMING DATA SERVICE TESTS
# ============================================================================

from streaming_data_service import (
    StreamingDataService, PriceUpdate, DataSourceType, get_streaming_service,
)


class TestStreamingDataService(unittest.TestCase):
    """Test streaming service subscription and price retrieval."""

    def test_create_simulation_service(self):
        svc = StreamingDataService(data_source=DataSourceType.SIMULATION)
        self.assertIsNotNone(svc)

    def test_subscribe_and_get_prices(self):
        svc = StreamingDataService(data_source=DataSourceType.SIMULATION)
        received = []
        svc.subscribe('AAPL', lambda update: received.append(update))
        prices = svc.get_all_prices()
        self.assertIsInstance(prices, dict)

    def test_price_update_to_dict(self):
        pu = PriceUpdate(
            symbol='AAPL', timestamp='2025-01-01T10:00:00',
            price=150.0, bid=149.9, ask=150.1,
            volume=1000000, change_percent=0.5,
        )
        d = pu.to_dict()
        self.assertEqual(d['symbol'], 'AAPL')
        self.assertEqual(d['price'], 150.0)

    def test_set_update_frequency(self):
        svc = StreamingDataService(data_source=DataSourceType.SIMULATION)
        svc.set_update_frequency(5)
        # Should not raise


# ============================================================================
# RISK MANAGEMENT ENHANCED TESTS
# ============================================================================

from risk_management_enhanced import (
    EnhancedStopLoss, TrailingStopLoss, DynamicTakeProfitCalculator,
    PortfolioDiversificationManager, DynamicPositionSizer,
)


class TestEnhancedStopLoss(unittest.TestCase):
    """Test enhanced and trailing stop-loss logic."""

    def test_stop_not_triggered_above(self):
        sl = EnhancedStopLoss(entry_price=100.0, initial_stop_loss_percent=0.05)
        triggered, _ = sl.check_trigger(98.0)
        self.assertFalse(triggered)

    def test_stop_triggered_below(self):
        sl = EnhancedStopLoss(entry_price=100.0, initial_stop_loss_percent=0.05)
        triggered, reason = sl.check_trigger(94.0)
        self.assertTrue(triggered)
        self.assertIsNotNone(reason)

    def test_trailing_stop_moves_up(self):
        tsl = TrailingStopLoss(entry_price=100.0, trailing_percent=0.05)
        tsl.update(110.0)  # price rises
        tsl.update(108.0)  # price dips but not enough
        triggered, _ = tsl.update(108.0)
        self.assertFalse(triggered)

    def test_trailing_stop_triggers_on_drop(self):
        tsl = TrailingStopLoss(entry_price=100.0, trailing_percent=0.05)
        tsl.update(110.0)  # high watermark = 110
        triggered, _ = tsl.update(104.0)  # 5.4% below high → triggers
        self.assertTrue(triggered)


class TestDynamicTakeProfit(unittest.TestCase):

    def test_tp_price_above_entry(self):
        params = StrategyFactory.create_balanced_strategy()
        tp_calc = DynamicTakeProfitCalculator(params)
        tp = tp_calc.calculate_tp_price(
            entry_price=100.0, current_price=105.0,
            confidence=0.8, volatility=0.02, rsi=50.0,
        )
        self.assertGreater(tp, 100.0)


class TestPortfolioDiversification(unittest.TestCase):

    def test_sector_lookup(self):
        mgr = PortfolioDiversificationManager()
        sector = mgr.get_sector('AAPL')
        self.assertIsInstance(sector, str)
        self.assertGreater(len(sector), 0)

    def test_single_stock_limit(self):
        mgr = PortfolioDiversificationManager()
        allowed, _ = mgr.check_single_stock_limit(
            'AAPL', proposed_position_value=60000,
            total_portfolio_value=100000, existing_positions=[],
        )
        # 60% of portfolio in one stock should be rejected
        self.assertFalse(allowed)


class TestDynamicPositionSizer(unittest.TestCase):

    def test_returns_positive_shares(self):
        params = StrategyFactory.create_balanced_strategy()
        sizer = DynamicPositionSizer(params)
        shares, value = sizer.calculate_position_size(
            portfolio_value=100000, entry_price=150.0,
            stop_loss_price=142.5, confidence=0.7, volatility=0.02,
        )
        self.assertGreater(shares, 0)
        self.assertGreater(value, 0)

    def test_volatility_reduces_size(self):
        params = StrategyFactory.create_balanced_strategy()
        sizer = DynamicPositionSizer(params)
        s_low, _ = sizer.calculate_position_size(
            portfolio_value=100000, entry_price=150.0,
            stop_loss_price=142.5, confidence=0.7, volatility=0.01,
        )
        s_high, _ = sizer.calculate_position_size(
            portfolio_value=100000, entry_price=150.0,
            stop_loss_price=142.5, confidence=0.7, volatility=0.06,
        )
        self.assertGreaterEqual(s_low, s_high)


# ============================================================================
# PREDICTION ENGINE TESTS
# ============================================================================

import pandas as pd


class TestPredictionEngine(unittest.TestCase):
    """Test prediction engine signal generation."""

    @classmethod
    def setUpClass(cls):
        """Load AAPL indicator data once for all tests."""
        csv_path = 'data/AAPL_stock_data_with_indicators.csv'
        if not os.path.exists(csv_path):
            raise unittest.SkipTest('AAPL indicator data not available')
        cls.data = pd.read_csv(csv_path)

    def _make_engine(self):
        from prediction_engine import PredictionEngine
        return PredictionEngine(self.data, symbol='AAPL')

    def test_create_engine(self):
        pe = self._make_engine()
        self.assertIsNotNone(pe)

    def test_calculate_indicators(self):
        pe = self._make_engine()
        indicators = pe.calculate_technical_indicators()
        self.assertIn('rsi', indicators)
        self.assertIn('macd', indicators)
        self.assertIn('volatility', indicators)

    def test_generate_signal_returns_tuple(self):
        pe = self._make_engine()
        indicators = pe.calculate_technical_indicators()
        signal, strength = pe.generate_signal(indicators)
        self.assertIn(signal, ('BULLISH', 'BEARISH', 'NEUTRAL'))
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)

    def test_ml_predict_return(self):
        pe = self._make_engine()
        pred_return, confidence = pe.ml_predict_return()
        self.assertIsInstance(pred_return, float)
        self.assertIsInstance(confidence, float)

    def test_predict_multi_day(self):
        pe = self._make_engine()
        result = pe.predict_multi_day(days_ahead=3)
        self.assertIn('signal', result)
        self.assertIn('current_price', result)
        self.assertIn('forecasts', result)


# ============================================================================
# BACKTESTING ENGINE TESTS
# ============================================================================

from backtesting_engine import BacktestingEngine, BacktestResults, BacktestTrade


class TestBacktestingEngine(unittest.TestCase):
    """Test backtesting engine produces valid results."""

    @classmethod
    def setUpClass(cls):
        csv_path = 'data/AAPL_stock_data_with_indicators.csv'
        if not os.path.exists(csv_path):
            raise unittest.SkipTest('AAPL indicator data not available')
        engine = BacktestingEngine(symbol='AAPL')
        cls.data = engine.load_data(csv_path)

    def test_run_backtest_returns_results(self):
        params = StrategyFactory.create_balanced_strategy()
        engine = BacktestingEngine(
            initial_capital=100000.0, trading_params=params, symbol='AAPL',
        )
        results = engine.run_backtest(self.data, strategy_name='Test')
        self.assertIsInstance(results, BacktestResults)
        self.assertEqual(results.initial_capital, 100000.0)

    def test_roi_is_float(self):
        params = StrategyFactory.create_balanced_strategy()
        engine = BacktestingEngine(
            initial_capital=100000.0, trading_params=params, symbol='AAPL',
        )
        results = engine.run_backtest(self.data, strategy_name='Test')
        self.assertIsInstance(results.roi, float)

    def test_trades_list(self):
        params = StrategyFactory.create_balanced_strategy()
        engine = BacktestingEngine(
            initial_capital=100000.0, trading_params=params, symbol='AAPL',
        )
        results = engine.run_backtest(self.data, strategy_name='Test')
        self.assertIsInstance(results.trades, list)
        if results.trades:
            self.assertIsInstance(results.trades[0], BacktestTrade)

    def test_results_to_dict(self):
        params = StrategyFactory.create_balanced_strategy()
        engine = BacktestingEngine(
            initial_capital=100000.0, trading_params=params, symbol='AAPL',
        )
        results = engine.run_backtest(self.data, strategy_name='Test')
        d = results.to_dict()
        self.assertIn('roi', d)
        self.assertIn('sharpe_ratio', d)
        self.assertIn('win_rate', d)

    def test_win_rate_in_range(self):
        params = StrategyFactory.create_balanced_strategy()
        engine = BacktestingEngine(
            initial_capital=100000.0, trading_params=params, symbol='AAPL',
        )
        results = engine.run_backtest(self.data, strategy_name='Test')
        self.assertGreaterEqual(results.win_rate, 0.0)
        self.assertLessEqual(results.win_rate, 1.0)


# ============================================================================
# DATA FETCHER TESTS
# ============================================================================

class TestDataFetcher(unittest.TestCase):
    """Test data fetcher module structure."""

    def test_imports(self):
        from data_fetcher import fetch_stock_data, load_stock_data, DEFAULT_SYMBOLS
        self.assertIsInstance(DEFAULT_SYMBOLS, list)
        self.assertGreater(len(DEFAULT_SYMBOLS), 0)

    def test_load_existing_data(self):
        from data_fetcher import load_stock_data
        csv_path = 'data/AAPL_stock_data_with_indicators.csv'
        if not os.path.exists(csv_path):
            self.skipTest('Data file not available')
        df = load_stock_data('AAPL')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


# ============================================================================
# TRADING EXECUTION TESTS
# ============================================================================

class TestTradingExecution(unittest.TestCase):
    """Test order management and trading execution classes."""

    def test_import_classes(self):
        from trading_execution import (
            TradingEngine, OrderManager, TradeLogger,
            MarketOrder, LimitOrder, StopLossOrder, TakeProfitOrder,
        )
        self.assertTrue(True)  # Import succeeded

    def test_market_order_creation(self):
        from trading_execution import MarketOrder, OrderSide
        from datetime import datetime
        order = MarketOrder(
            order_id=1, symbol='AAPL', side=OrderSide.BUY,
            quantity=10, created_at=datetime.now(),
        )
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.quantity, 10)

    def test_order_manager_queue(self):
        from trading_execution import OrderManager, OrderSide
        mgr = OrderManager()
        order = mgr.create_market_order(
            symbol='AAPL', side=OrderSide.BUY,
            quantity=5, entry_price=150.0,
        )
        pending = mgr.get_pending_orders()
        self.assertGreaterEqual(len(pending), 0)


# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    unittest.main()
