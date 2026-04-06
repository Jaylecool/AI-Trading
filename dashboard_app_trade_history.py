
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import sys
import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid

# Custom JSON encoder to handle NaN values
class NaNEncoder(json.JSONEncoder):
    def encode(self, o):
        if isinstance(o, float):
            if np.isnan(o):
                return 'null'
            elif np.isinf(o):
                return 'null'
        return super().encode(o)
    
    def iterencode(self, o, _one_shot=False):
        for chunk in super().iterencode(o, _one_shot):
            yield chunk

# Setup logging
logger = logging.getLogger(__name__)

# Import task 5.3 modules
from portfolio_tracker import (
    PortfolioTracker, Trade, TradeHistoryFilter, 
    PortfolioVisualizer
)

# Import task 5.5 modules (streaming and alerts)
from streaming_data_service import (
    get_streaming_service, DataSourceType
)
from alert_system import (
    get_alert_system, AlertType, AlertSeverity, ComparisonOperator
)
from notification_service import (
    get_notification_service, NotificationChannel, NotificationPreference,
    Notification
)

# Import trading rules for auto-trading
from trading_rules import TradingRules, TradingParameters, PositionSizingCalculator
from strategy_configurations import StrategyFactory, STRATEGIES

# Centralised configuration
import config as cfg

# Supported symbols for multi-stock trading
SUPPORTED_SYMBOLS = cfg.SUPPORTED_SYMBOLS

# Flask application setup
app = Flask(__name__)
app.json_encoder = NaNEncoder
app.secret_key = cfg.SECRET_KEY
CORS(app, origins=cfg.CORS_ORIGINS, supports_credentials=True)

# Import auth module
from auth import init_db as init_auth_db, register_user, authenticate_user, get_user_by_id
from functools import wraps
from flask import session, redirect, Response

def login_required(f):
    """Decorator that redirects to /landing when user is not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            # JSON API endpoints return 401; page routes redirect
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect('/landing')
        return f(*args, **kwargs)
    return decorated

import re
_SYMBOL_RE = re.compile(r'^[A-Z]{1,5}$')

def _validate_symbol(symbol: str) -> bool:
    """Return True if *symbol* looks like a valid ticker and is in SUPPORTED_SYMBOLS."""
    return bool(symbol and _SYMBOL_RE.match(symbol) and symbol in SUPPORTED_SYMBOLS)

# Global system instances (shared — not per-user)
streaming_service = None
alert_system = None
notification_service = None
active_clients = {}  # client_id -> connection info

# Trained ML models: per-symbol dict loaded at startup
# { symbol: { 'model_lr', 'model_rf', 'model_gb_clf', 'scaler', 'feature_cols' } }
trained_models = {}  # symbol -> model dict

# ---------- Per-user portfolio registry ----------
# Each user_id maps to a dict with their own portfolio state:
#   { 'tracker': PortfolioTracker,
#     'open_positions': {trade_id: Trade},
#     'auto_trade_log': [dict],
#     'disabled_symbols': set(),
#     'active_strategy': str,
#     'auto_running': bool }
user_portfolios = {}            # user_id (int) -> state dict
_user_portfolios_lock = threading.Lock()

# Legacy global kept for import compatibility only — not used at runtime.
portfolio_tracker = None

# Auto-trading daemon state (shared)
auto_trader_thread = None
auto_trade_last_check = None
symbol_strategies = {}  # Per-symbol best strategy: {symbol: strategy_key}


def _user_state(user_id):
    """Return the per-user state dict, creating a fresh portfolio on first access."""
    uid = int(user_id)
    with _user_portfolios_lock:
        if uid not in user_portfolios:
            _create_user_portfolio(uid)
        return user_portfolios[uid]


def _create_user_portfolio(uid):
    """Initialise a brand-new portfolio for a user."""
    base_dir = os.path.dirname(__file__)
    tracker = PortfolioTracker(initial_balance=cfg.INITIAL_CAPITAL)
    tracker.LIVE_TRADES_FILE = os.path.join('results', f'user_{uid}_trades.json')
    tracker.set_persistence_path(base_dir)

    state = {
        'tracker': tracker,
        'open_positions': {},
        'auto_trade_log': [],
        'disabled_symbols': set(),
        'active_strategy': 'AUTO',
        'auto_running': True,
    }

    # Try to load existing persisted data
    live_path = os.path.join(base_dir, tracker.LIVE_TRADES_FILE)
    if os.path.exists(live_path):
        tracker.load_from_json(live_path)
        _load_user_auto_state(uid, state)
    else:
        # New user – persist initial empty portfolio
        tracker.save_to_json()

    user_portfolios[uid] = state
    print(f"[USER] Portfolio initialised for user {uid} (cash=${tracker.portfolio.cash:,.2f}, trades={len(tracker.trade_history)})")
    return state


def _user_state_file(uid):
    return os.path.join(cfg.RESULTS_DIR, f'user_{uid}_auto_state.json')


def save_user_auto_state(uid):
    """Persist a single user's auto-trade state."""
    state = user_portfolios.get(int(uid))
    if not state:
        return
    try:
        data = {
            'open_positions': {tid: t.to_dict() for tid, t in state['open_positions'].items()},
            'auto_trade_log': state['auto_trade_log'][-200:],
            'disabled_symbols': list(state['disabled_symbols']),
            'active_strategy': state['active_strategy'],
            'symbol_strategies': symbol_strategies,
            'saved_at': datetime.now().isoformat(),
        }
        path = _user_state_file(uid)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[PERSIST] Error saving auto state for user {uid}: {e}")


def _load_user_auto_state(uid, state):
    """Load a single user's auto-trade state from disk."""
    path = _user_state_file(uid)
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        state['auto_trade_log'] = data.get('auto_trade_log', [])
        state['disabled_symbols'] = set(data.get('disabled_symbols', []))
        saved_strat = data.get('active_strategy', 'AUTO')
        if saved_strat in STRATEGIES or saved_strat == 'AUTO':
            state['active_strategy'] = saved_strat
        for tid, td in data.get('open_positions', {}).items():
            trade = Trade(
                trade_id=td['trade_id'], date=td['date'], symbol=td['symbol'],
                action=td['action'], quantity=float(td['quantity']),
                entry_price=float(td['entry_price']),
                stop_loss=float(td['stop_loss']) if td.get('stop_loss') is not None else None,
                take_profit=float(td['take_profit']) if td.get('take_profit') is not None else None,
                exit_date=td.get('exit_date'),
                exit_price=float(td['exit_price']) if td.get('exit_price') is not None else None,
                status=td.get('status', 'OPEN'), user_id=uid,
            )
            state['open_positions'][tid] = trade
        print(f"[PERSIST] User {uid}: restored {len(state['open_positions'])} positions, {len(state['auto_trade_log'])} log entries")
    except Exception as e:
        print(f"[PERSIST] Error loading auto state for user {uid}: {e}")

def auto_select_strategies():
    """Read backtest CSVs to pick the best strategy per symbol based on a composite score."""
    global symbol_strategies
    best = {}
    for symbol in SUPPORTED_SYMBOLS:
        csv_path = os.path.join(cfg.RESULTS_DIR, f'strategy_comparison_{symbol}.csv')
        if not os.path.exists(csv_path):
            best[symbol] = 'BALANCED'  # default fallback
            continue
        try:
            df = pd.read_csv(csv_path)
            top_score = -float('inf')
            top_strategy = 'BALANCED'
            for _, row in df.iterrows():
                roi = float(str(row['ROI']).replace('%', ''))
                win_rate = float(str(row['Win Rate']).replace('%', ''))
                sharpe = float(row['Sharpe Ratio'])
                profit_factor = float(row['Profit Factor'])
                # Composite score: weighted blend of key metrics
                score = (roi * 0.3) + (win_rate * 0.3) + (sharpe * 0.2) + (profit_factor * 0.2)
                if score > top_score:
                    top_score = score
                    top_strategy = row['Strategy'].strip().upper()
            best[symbol] = top_strategy if top_strategy in STRATEGIES else 'BALANCED'
        except Exception as e:
            print(f"[STRATEGY] Error reading {csv_path}: {e}")
            best[symbol] = 'BALANCED'
    symbol_strategies = best
    print(f"[STRATEGY] Auto-selected per-symbol strategies: {symbol_strategies}")
    return best

LIVE_STATE_FILE = os.path.join(cfg.RESULTS_DIR, 'live_auto_state.json')

def save_auto_state():
    """Legacy wrapper — no-op now (per-user state is saved via save_user_auto_state)."""
    pass

def load_auto_state():
    """Legacy wrapper — no-op now."""
    pass

def initialize_portfolio(backtest_file: str = os.path.join(cfg.RESULTS_DIR, 'backtest_results.json')):
    """Initialize shared services (streaming, alerts, notifications).
    Per-user portfolios are created lazily on first access."""
    global streaming_service, alert_system, notification_service

    # Initialize streaming service with YAHOO FINANCE (LIVE DATA)
    streaming_service = get_streaming_service(data_source=DataSourceType.YAHOO_FINANCE)
    for sym in SUPPORTED_SYMBOLS:
        streaming_service.subscribe(sym, lambda update: None)
    streaming_service.set_update_frequency(cfg.STREAM_UPDATE_FREQUENCY)
    streaming_service.start()
    
    # Initialize alert system
    alert_system = get_alert_system()
    
    # Initialize notification service
    notification_service = get_notification_service()

    print("Streaming service initialized and started")
    print("Alert system initialized")
    print("Notification service initialized")

# ============================================================================
# TRADE HISTORY ENDPOINTS
# ============================================================================

@app.route('/api/trades/history', methods=['GET'])
@login_required
def get_trade_history():
    """Get complete trade history"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        # Get parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        sort_by = request.args.get('sort_by', 'date')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Get trade history
        all_trades = tracker.get_trade_history()
        
        # Sort
        reverse = sort_order.lower() == 'desc'
        if sort_by in ['date', 'entry_price', 'quantity']:
            all_trades = sorted(all_trades, 
                              key=lambda t: t.get(sort_by, ''),
                              reverse=reverse)
        
        # Paginate
        total = len(all_trades)
        trades = all_trades[offset:offset + limit]
        
        # Format for display
        formatted_trades = [
            PortfolioVisualizer.format_trade_for_display(
                Trade(**trade)
            ) for trade in trades
        ]
        
        return jsonify({
            'status': 'success',
            'trades': formatted_trades,
            'total': total,
            'limit': limit,
            'offset': offset,
            'returned': len(formatted_trades)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/filtered', methods=['GET'])
@login_required
def get_filtered_trades():
    """Get trades filtered by various criteria"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        # Get filter parameters
        symbol = request.args.get('symbol')
        action = request.args.get('action')  # BUY/SELL
        status = request.args.get('status')  # OPEN/CLOSED
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        min_pnl = request.args.get('min_pnl', type=float)
        max_pnl = request.args.get('max_pnl', type=float)
        
        # Validate inputs
        if symbol and not _validate_symbol(symbol):
            return jsonify({'error': f'Invalid symbol: {symbol}'}), 400
        if action and action.upper() not in ('BUY', 'SELL'):
            return jsonify({'error': f'Invalid action: {action}'}), 400
        if status and status.upper() not in ('OPEN', 'CLOSED'):
            return jsonify({'error': f'Invalid status: {status}'}), 400
        
        # Create filter object
        filter_obj = TradeHistoryFilter(tracker.portfolio.trades)
        
        # Build filters dictionary
        filters = {}
        if symbol:
            filters['symbol'] = symbol
        if action:
            filters['action'] = action
        if status:
            filters['status'] = status
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if min_pnl is not None:
            filters['min_pnl'] = min_pnl
        if max_pnl is not None:
            filters['max_pnl'] = max_pnl
        
        # Apply filters
        filtered_trades = filter_obj.search(**filters)
        
        # Format for display
        formatted_trades = [
            PortfolioVisualizer.format_trade_for_display(trade)
            for trade in filtered_trades
        ]
        
        return jsonify({
            'status': 'success',
            'filters': filters,
            'trades': formatted_trades,
            'count': len(formatted_trades)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/search', methods=['POST'])
@login_required
def search_trades():
    """Search trades with complex criteria (POST)"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        # Get search criteria from body
        criteria = request.get_json()
        
        # Apply filters
        filter_obj = TradeHistoryFilter(tracker.portfolio.trades)
        filtered_trades = filter_obj.search(**criteria)
        
        # Format for display
        formatted_trades = [
            PortfolioVisualizer.format_trade_for_display(trade)
            for trade in filtered_trades
        ]
        
        return jsonify({
            'status': 'success',
            'criteria': criteria,
            'trades': formatted_trades,
            'count': len(formatted_trades)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# PORTFOLIO ENDPOINTS
# ============================================================================

@app.route('/api/portfolio/summary', methods=['GET'])
@login_required
def get_portfolio_summary():
    """Get portfolio summary with all metrics"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        # Update market prices for accurate position valuation
        try:
            import yfinance as yf
            for sym in SUPPORTED_SYMBOLS:
                try:
                    price = yf.Ticker(sym).history(period='1d')['Close'].iloc[-1]
                    tracker.portfolio.update_market_price(sym, float(price))
                except (KeyError, IndexError, ValueError) as exc:
                    logger.debug(f"Could not fetch price for {sym}: {exc}")
        except ImportError:
            logger.warning("yfinance not available – skipping live price update")
        
        metrics = tracker.get_portfolio_summary()
        formatted = PortfolioVisualizer.format_portfolio_summary(metrics)
        
        return jsonify({
            'status': 'success',
            'summary': formatted,
            'raw': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/allocation', methods=['GET'])
@login_required
def get_asset_allocation():
    """Get asset allocation for pie chart"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        allocation = PortfolioVisualizer.get_asset_allocation(
            tracker.portfolio
        )
        
        # Format for chart
        labels = list(allocation.keys())
        values = list(allocation.values())
        
        return jsonify({
            'status': 'success',
            'labels': labels,
            'values': values,
            'type': 'pie'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/statistics', methods=['GET'])
@login_required
def get_trade_statistics():
    """Get trade statistics by symbol"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        stats = PortfolioVisualizer.get_trade_statistics(
            tracker.portfolio.trades
        )
        
        # Format for bar chart
        symbols = list(stats.keys())
        win_rates = [stats[s]['win_rate'] for s in symbols]
        total_pnls = [stats[s]['total_pnl'] for s in symbols]
        trade_counts = [stats[s]['total_trades'] for s in symbols]
        
        return jsonify({
            'status': 'success',
            'symbols': symbols,
            'win_rates': win_rates,
            'total_pnls': total_pnls,
            'trade_counts': trade_counts,
            'detailed': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/equity-curve', methods=['GET'])
def get_equity_curve():
    """Get stock price data for the equity curve with configurable interval"""
    try:
        import yfinance as yf
        
        symbol = request.args.get('symbol', 'AAPL')
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': f'Unsupported symbol: {symbol}'}), 400
        
        # Get interval parameter from query string, default to 1d (daily)
        interval = request.args.get('interval', '1d').lower()
        
        # Define valid intervals and their corresponding periods
        # Each period is optimized to show meaningful recent data for that timeframe
        interval_config = {
            '1m': {'period': '1h', 'interval': '1m'},       # Last 1 hour, 1 minute intervals
            '5m': {'period': '1d', 'interval': '5m'},       # Last 1 day, 5 minute intervals
            '15m': {'period': '1d', 'interval': '15m'},     # Last 1 day, 15 minute intervals
            '1h': {'period': '7d', 'interval': '1h'},       # Last 7 days, hourly
            '1d': {'period': '60d', 'interval': '1d'},      # Last 60 days, daily
            '1wk': {'period': '52wk', 'interval': '1wk'}    # Last 52 weeks, weekly
        }
        
        # Validate interval
        if interval not in interval_config:
            interval = '1d'
        
        config = interval_config[interval]
        
        # Fetch data with specified interval
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=config['period'], interval=config['interval'])
        
        if hist.empty:
            return jsonify({'error': 'No data available'}), 500
        
        # Extract dates and closing prices
        dates = [date.strftime('%Y-%m-%d %H:%M') if config['interval'] in ['1m', '5m', '15m', '1h'] 
                 else date.strftime('%Y-%m-%d') for date in hist.index]
        equity = hist['Close'].tolist()
        
        # Clean NaN values
        equity = [None if pd.isna(x) else float(x) for x in equity]
        
        logger.info(f"Returning {len(dates)} {symbol} price points (interval: {interval}, period: {config['period']})")
        if equity and any(x is not None for x in equity):
            valid_equity = [x for x in equity if x is not None]
            logger.info(f"Price range: ${min(valid_equity):.2f} - ${max(valid_equity):.2f}")
        
        return jsonify({
            'status': 'success',
            'dates': dates,
            'equity': equity,
            'type': 'line',
            'interval': interval
        })
    
    except Exception as e:
        logger.error(f"Error in get_equity_curve: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/live-price', methods=['GET'])
def get_live_price():
    """Get current stock price with recent history"""
    try:
        import yfinance as yf
        
        symbol = request.args.get('symbol', 'AAPL')
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': f'Unsupported symbol: {symbol}'}), 400
        
        ticker = yf.Ticker(symbol)
        
        # Get current and recent prices
        hist = ticker.history(period='1y')
        info = ticker.info if hasattr(ticker, 'info') else {}
        
        if not hist.empty:
            # Current price
            current = hist.iloc[-1]
            current_price = float(current['Close'])
            
            # Previous close
            previous_price = float(hist.iloc[-2]['Close']) if len(hist) > 1 else current_price
            
            # Price change
            change = current_price - previous_price
            change_percent = (change / previous_price * 100) if previous_price != 0 else 0
            
            # High/Low for the day
            day_high = float(current['High'])
            day_low = float(current['Low'])
            
            # 52-week high/low
            high_52w = float(hist['High'].max())
            low_52w = float(hist['Low'].min())
            
            # Volume
            volume = int(current['Volume'])
            
            return jsonify({
                'status': 'success',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'previous_close': previous_price,
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'day_high': round(day_high, 2),
                'day_low': round(day_low, 2),
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
                'volume': volume,
                'currency': 'USD'
            })
        else:
            return jsonify({'error': 'No data available'}), 500
    
    except ImportError:
        return jsonify({'error': 'yfinance not installed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/pnl-distribution', methods=['GET'])
@login_required
def get_pnl_distribution():
    """Get PnL distribution histogram"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        distribution = PortfolioVisualizer.get_pnl_distribution(
            tracker.portfolio.trades
        )
        
        return jsonify({
            'status': 'success',
            'bins': distribution['bins'],
            'count': distribution['count'],
            'type': 'histogram'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/performance', methods=['GET'])
@login_required
def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        metrics = tracker.get_portfolio_summary()
        
        # Calculate additional metrics
        closed_trades = [t for t in tracker.portfolio.trades 
                        if t.is_closed()]
        
        winning_trades = sum(1 for t in closed_trades 
                           if t.pnl and t.pnl > 0)
        losing_trades = sum(1 for t in closed_trades 
                          if t.pnl and t.pnl < 0)
        
        total_pnl = sum(t.pnl for t in closed_trades 
                       if t.pnl)
        
        avg_winner = metrics['average_win']
        avg_loser = metrics['average_loss']
        
        # Risk metrics
        performance = {
            'total_trades': metrics['num_trades'],
            'closed_trades': metrics['num_closed_trades'],
            'open_trades': metrics['num_open_trades'],
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': metrics['win_rate'],
            'total_pnl': metrics['total_pnl'],
            'total_pnl_percent': metrics['total_pnl_percent'],
            'average_winner': metrics['average_win'],
            'average_loser': metrics['average_loss'],
            'profit_factor': metrics['profit_factor'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'expectancy': (winning_trades * avg_winner - 
                         losing_trades * avg_loser) / metrics['num_closed_trades'] 
                        if metrics['num_closed_trades'] > 0 else 0,
        }
        
        return jsonify({
            'status': 'success',
            'performance': performance,
            'comparative': {
                'benchmark_return': 8.5,  # Typical S&P 500 return
                'your_return': performance['total_pnl_percent'],
                'outperformance': performance['total_pnl_percent'] - 8.5,
                'risk_adjusted': performance['sharpe_ratio']
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.route('/api/portfolio/symbols', methods=['GET'])
@login_required
def get_traded_symbols():
    """Get list of symbols traded"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        symbols = set()
        for trade in tracker.portfolio.trades:
            symbols.add(trade.symbol)
        
        return jsonify({
            'status': 'success',
            'symbols': sorted(list(symbols)),
            'count': len(symbols)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/date-range', methods=['GET'])
@login_required
def get_date_range():
    """Get date range of trades"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        if not tracker.portfolio.trades:
            return jsonify({
                'status': 'success',
                'start_date': None,
                'end_date': None,
                'trading_days': 0
            })
        
        dates = [datetime.fromisoformat(t.date) 
                for t in tracker.portfolio.trades]
        
        start_date = min(dates).date().isoformat()
        end_date = max(dates).date().isoformat()
        trading_days = (max(dates) - min(dates)).days
        
        return jsonify({
            'status': 'success',
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': trading_days
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/refresh', methods=['POST'])
@login_required
def refresh_portfolio():
    """Refresh portfolio data from sources"""
    try:
        ctx = _user_state(session['user_id'])
        tracker = ctx['tracker']
        
        return jsonify({
            'status': 'success',
            'message': 'Portfolio refreshed',
            'trades_loaded': len(tracker.trade_history)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# REAL-TIME STREAMING ENDPOINTS (Task 5.5)
# ============================================================================

@app.route('/api/streaming/subscribe', methods=['POST'])
def subscribe_to_streaming():
    """Subscribe to real-time price updates"""
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        client_id = data.get('client_id', str(uuid.uuid4()))
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        # Validate symbols
        invalid = [s for s in symbols if not _validate_symbol(s)]
        if invalid:
            return jsonify({'error': f'Invalid symbols: {invalid}'}), 400
        
        # Register client
        active_clients[client_id] = {
            'symbols': symbols,
            'subscribed_at': datetime.now().isoformat(),
            'last_update': None
        }
        
        # Subscribe to streaming service
        for symbol in symbols:
            streaming_service.subscribe(symbol, lambda update: None)
        
        return jsonify({
            'status': 'subscribed',
            'client_id': client_id,
            'symbols': symbols,
            'message': f'Subscribed to {len(symbols)} symbols'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/streaming/prices', methods=['GET'])
def get_latest_prices():
    """Get latest prices for subscribed symbols"""
    try:
        client_id = request.args.get('client_id')
        symbols = request.args.getlist('symbols')
        
        prices = {}
        
        # Get prices from cache
        if symbols:
            for symbol in symbols:
                price = streaming_service.get_latest_price(symbol)
                if price:
                    prices[symbol] = price.to_dict()
        else:
            # Get all cached prices
            all_prices = streaming_service.get_all_prices()
            prices = {s: p.to_dict() for s, p in all_prices.items()}
        
        return jsonify({
            'status': 'success',
            'prices': prices,
            'timestamp': datetime.now().isoformat(),
            'count': len(prices)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/streaming/status', methods=['GET'])
def get_streaming_status():
    """Get streaming service status"""
    try:
        return jsonify({
            'status': 'active' if streaming_service.is_running else 'inactive',
            'connected_clients': len(active_clients),
            'subscribed_symbols': len(streaming_service.subscribed_symbols),
            'update_frequency': f"{streaming_service.update_frequency}s",
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ALERT RULE ENDPOINTS (Task 5.5)
# ============================================================================

@app.route('/api/alerts/rules', methods=['GET'])
def get_alert_rules():
    """Get all alert rules"""
    try:
        rules = alert_system.get_all_rules()
        rules_data = [r.to_dict() for r in rules]
        
        return jsonify({
            'status': 'success',
            'rules': rules_data,
            'total': len(rules_data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/rules', methods=['POST'])
def create_alert_rule():
    """Create a new alert rule"""
    try:
        data = request.json or {}
        
        # Validate required fields
        if not data.get('name', '').strip():
            return jsonify({'error': 'Alert name is required'}), 400
        if data.get('symbol') and not _validate_symbol(data['symbol']):
            return jsonify({'error': f'Invalid symbol: {data["symbol"]}'}), 400
        
        rule = alert_system.create_rule(
            name=data.get('name', 'Untitled Alert'),
            alert_type=AlertType(data.get('alert_type', 'price_alert')),
            metric_field=data.get('metric_field', 'price'),
            operator=ComparisonOperator(data.get('operator', '<')),
            threshold_value=float(data.get('threshold_value', 0)),
            symbol=data.get('symbol'),
            severity=AlertSeverity[data.get('severity', 'MEDIUM')]
        )
        
        return jsonify({
            'status': 'created',
            'rule': rule.to_dict()
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/rules/<rule_id>', methods=['PUT'])
def update_alert_rule(rule_id):
    """Update an alert rule"""
    try:
        data = request.json or {}
        
        alert_system.update_rule(rule_id, **data)
        rule = alert_system.get_rule(rule_id)
        
        if rule:
            return jsonify({
                'status': 'updated',
                'rule': rule.to_dict()
            })
        else:
            return jsonify({'error': 'Rule not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/rules/<rule_id>', methods=['DELETE'])
def delete_alert_rule(rule_id):
    """Delete an alert rule"""
    try:
        if alert_system.delete_rule(rule_id):
            return jsonify({'status': 'deleted'})
        else:
            return jsonify({'error': 'Rule not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/evaluate', methods=['POST'])
def evaluate_alerts():
    """Evaluate alert rules against data"""
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'TEST')
        metrics = data.get('metrics', {})
        
        # Evaluate rules
        triggered_alerts = alert_system.evaluate(symbol, metrics)
        
        # Send notifications
        for alert in triggered_alerts:
            notif = Notification(
                notification_id=alert.alert_id,
                alert_id=alert.alert_id,
                title=alert.rule_name,
                message=alert.message,
                severity=alert.severity.name,
                channels=[NotificationChannel.POPUP, NotificationChannel.SOUND]
            )
            notification_service.send_notification(notif)
        
        return jsonify({
            'status': 'evaluated',
            'symbol': symbol,
            'alerts_triggered': len(triggered_alerts),
            'alerts': [a.to_dict() for a in triggered_alerts]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/active', methods=['GET'])
def get_active_alerts():
    """Get all active alerts"""
    try:
        active = alert_system.get_active_alerts()
        
        return jsonify({
            'status': 'success',
            'alerts': [a.to_dict() for a in active],
            'total': len(active)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        if alert_system.acknowledge_alert(alert_id):
            return jsonify({'status': 'acknowledged'})
        else:
            return jsonify({'error': 'Alert not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>/dismiss', methods=['POST'])
def dismiss_alert(alert_id):
    """Dismiss an alert"""
    try:
        if alert_system.dismiss_alert(alert_id):
            return jsonify({'status': 'dismissed'})
        else:
            return jsonify({'error': 'Alert not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# NOTIFICATION PREFERENCES ENDPOINTS (Task 5.5)
# ============================================================================

@app.route('/api/notifications/preferences', methods=['GET'])
def get_notification_preferences():
    """Get user notification preferences"""
    try:
        user_id = request.args.get('user_id', 'default')
        prefs = notification_service.get_user_preferences(user_id)
        
        return jsonify(prefs.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/preferences', methods=['POST'])
def update_notification_preferences():
    """Update user notification preferences"""
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'default')
        
        # Extract preference fields
        preference_fields = {k: v for k, v in data.items() 
                           if k.startswith('enable_') or k.startswith('quiet_') 
                           or k in ['sound_volume', 'email_address']}
        
        notification_service.update_preference(user_id, **preference_fields)
        
        prefs = notification_service.get_user_preferences(user_id)
        return jsonify({
            'status': 'updated',
            'preferences': prefs.to_dict()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/history', methods=['GET'])
def get_notification_history():
    """Get notification history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        history = notification_service.get_notification_history(limit=limit)
        
        return jsonify({
            'status': 'success',
            'notifications': [n.to_dict() for n in history],
            'total': len(history)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.route('/landing', methods=['GET'])
def landing_page():
    """Serve the landing / hero page."""
    if 'user_id' in session:
        return redirect('/dashboard')
    path = os.path.join(os.path.dirname(__file__), 'templates', 'landing.html')
    return Response(open(path, 'r', encoding='utf-8').read(), mimetype='text/html; charset=utf-8')

@app.route('/auth', methods=['GET'])
def auth_page():
    """Serve the login / sign-up page."""
    if 'user_id' in session:
        return redirect('/dashboard')
    path = os.path.join(os.path.dirname(__file__), 'templates', 'auth.html')
    return Response(open(path, 'r', encoding='utf-8').read(), mimetype='text/html; charset=utf-8')

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user account."""
    data = request.get_json(silent=True) or {}
    result = register_user(
        data.get('name', ''),
        data.get('email', ''),
        data.get('password', '')
    )
    if result['success']:
        session['user_id'] = result['user']['id']
        session['user_name'] = result['user']['name']
        session['user_email'] = result['user']['email']
    return jsonify(result)

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Authenticate an existing user."""
    data = request.get_json(silent=True) or {}
    user = authenticate_user(data.get('email', ''), data.get('password', ''))
    if user:
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        return jsonify({'success': True, 'user': user})
    return jsonify({'success': False, 'error': 'Invalid email or password'})

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Clear the session and log out."""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/me', methods=['GET'])
def api_me():
    """Return the current logged-in user info."""
    if 'user_id' not in session:
        return jsonify({'authenticated': False}), 401
    return jsonify({
        'authenticated': True,
        'user': {
            'id': session['user_id'],
            'name': session.get('user_name', ''),
            'email': session.get('user_email', '')
        }
    })

# ============================================================================
# DASHBOARD ENDPOINTS (NEW: Dark Mode UI with Live Data)
# ============================================================================

@app.route('/', methods=['GET'])
def dashboard_index():
    """Root route: landing page for guests, dashboard for logged-in users."""
    if 'user_id' not in session:
        return redirect('/landing')
    return redirect('/dashboard')

@app.route('/dashboard-view', methods=['GET'])
@login_required
def dashboard_view():
    """Serve the dashboard HTML (protected)."""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'templates', 'dashboard_trade_history.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='text/html; charset=utf-8')
    return jsonify({'error': 'Dashboard not found'}), 404

def calculate_sma(prices, window=20):
    """Calculate Simple Moving Average"""
    df = pd.DataFrame({'price': prices})
    return df['price'].rolling(window=window).mean().tolist()

def calculate_ema(prices, span=12):
    """Calculate Exponential Moving Average"""
    df = pd.DataFrame({'price': prices})
    return df['price'].ewm(span=span).mean().tolist()


@app.route('/api/symbols', methods=['GET'])
def get_supported_symbols():
    """Return list of supported trading symbols"""
    return jsonify({'symbols': SUPPORTED_SYMBOLS})


def compute_features_from_yfinance(hist_df, symbol='AAPL'):
    """
    Compute the 30 engineered features matching the trained ML model's schema.
    Returns (feature_array, features_df) where feature_array has shape (1, 30)
    in the exact column order from training_report.json, plus the raw DataFrame.
    
    If the symbol has a trained model, we use its feature_cols list.
    Otherwise falls back to the full 30-feature set.
    Requires at least 200+ rows for SMA_200 warmup.
    """
    df = hist_df.copy()
    
    # Use generic column names
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    opn = df['Open'] if 'Open' in df.columns else close
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    # Moving Averages
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    
    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    # ROC 12
    roc12 = close.pct_change(12) * 100
    
    # Bollinger Bands
    bb_std = close.rolling(20).std()
    bb_upper = sma20 + bb_std * 2
    bb_lower = sma20 - bb_std * 2
    
    # ATR (14)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    
    # Volatility (annualized)
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20).std() * np.sqrt(252) * 100
    
    # Volume SMA for ratio
    vol_sma20 = vol.rolling(20).mean()
    
    # --- Build all 30 engineered features ---
    df['RSI_14'] = rsi
    df['MACD_Histogram'] = macd_hist
    df['ROC_12'] = roc12
    df['Volatility_20'] = vol20
    df['Intraday_Range'] = (high - low) / close
    df['Open_Close_Ratio'] = opn / close
    df['High_Close_Ratio'] = high / close
    df['Low_Close_Ratio'] = low / close
    df['Volume_Ratio'] = (vol / vol_sma20).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df['Price_SMA10_Ratio'] = close / sma10
    df['Price_SMA20_Ratio'] = close / sma20
    df['Price_SMA50_Ratio'] = close / sma50
    df['Price_SMA200_Ratio'] = close / sma200
    df['EMA10_EMA20_Cross'] = (ema10 - ema20) / close
    df['BB_Width'] = (bb_upper - bb_lower) / sma20
    bb_range = bb_upper - bb_lower
    df['BB_Position'] = ((close - bb_lower) / bb_range.replace(0, np.nan)).fillna(0.5)
    df['ATR_Pct'] = atr14 / close
    df['Return_1d'] = daily_ret
    df['Return_5d'] = close.pct_change(5)
    df['Return_10d'] = close.pct_change(10)
    df['Return_20d'] = close.pct_change(20)
    df['SMA10_SMA20_Cross'] = (sma10 - sma20) / sma20.replace(0, np.nan)
    df['SMA50_SMA200_Cross'] = (sma50 - sma200) / sma200.replace(0, np.nan)
    df['RSI_Change_5d'] = rsi - rsi.shift(5)
    df['Volume_Change_5d'] = vol.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    df['Price_vs_20d_High'] = close / high_20
    df['Price_vs_20d_Low'] = close / low_20
    vol_60_mean = vol20.rolling(60).mean()
    df['Volatility_Ratio'] = (vol20 / vol_60_mean.replace(0, np.nan)).fillna(1.0)
    df['MACD_Hist_Change'] = macd_hist.diff(5)
    df['Pos_Days_5d'] = close.diff().gt(0).rolling(5).mean()
    
    # Also keep raw columns for market_data dict used by TradingRules
    df['SMA_20'] = sma20
    df['SMA_50'] = sma50
    df['EMA_10'] = ema10
    df['EMA_20'] = ema20
    df['ATR_14'] = atr14
    
    # Determine feature columns from trained model or use default order
    default_feature_cols = [
        'RSI_14', 'MACD_Histogram', 'ROC_12', 'Volatility_20',
        'Intraday_Range', 'Open_Close_Ratio', 'High_Close_Ratio', 'Low_Close_Ratio',
        'Volume_Ratio', 'Price_SMA10_Ratio', 'Price_SMA20_Ratio', 'Price_SMA50_Ratio',
        'Price_SMA200_Ratio', 'EMA10_EMA20_Cross', 'BB_Width', 'BB_Position',
        'ATR_Pct', 'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
        'SMA10_SMA20_Cross', 'SMA50_SMA200_Cross', 'RSI_Change_5d',
        'Volume_Change_5d', 'Price_vs_20d_High', 'Price_vs_20d_Low',
        'Volatility_Ratio', 'MACD_Hist_Change', 'Pos_Days_5d'
    ]
    
    # Use per-symbol feature cols if the model is loaded
    feature_cols = default_feature_cols
    if symbol in trained_models and trained_models[symbol].get('feature_cols'):
        feature_cols = trained_models[symbol]['feature_cols']
    
    # Get the latest complete row
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        for m in missing:
            df[m] = 0.0
    
    latest = df[feature_cols].dropna()
    if latest.empty:
        return None, df
    
    return latest.iloc[-1:].values, df


# ============================================================================
# AUTO-TRADING ENGINE
# ============================================================================

def auto_trade_cycle():
    """Background thread: periodically checks for trade opportunities using ML model + TradingRules.
    Iterates all registered users and processes each user's portfolio independently."""
    global auto_trade_last_check
    from prediction_engine import PredictionEngine

    # Wait for init to complete
    time.sleep(10)

    # Auto-select best strategies per symbol from backtest data
    if not symbol_strategies:
        auto_select_strategies()

    # Build per-symbol params caches
    def _build_params(strategy_key):
        p = STRATEGIES[strategy_key]['factory']()
        p.max_concurrent_positions = len(SUPPORTED_SYMBOLS)
        return p

    def _get_symbol_strategy(sym, user_active_strategy):
        """Return the strategy key to use for this symbol."""
        if user_active_strategy == 'AUTO':
            return symbol_strategies.get(sym, 'BALANCED')
        return user_active_strategy

    print(f"[AUTO-TRADE] Engine started — checking every 60 seconds")

    while True:
        # If no users registered yet, wait
        if not user_portfolios:
            time.sleep(5)
            continue

        try:
            import yfinance as yf

            auto_trade_last_check = datetime.now().isoformat()
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Snapshot the current set of users
            with _user_portfolios_lock:
                users_snapshot = list(user_portfolios.items())

            for uid, ustate in users_snapshot:
              if not ustate.get('auto_running', True):
                  continue

              open_positions = ustate['open_positions']
              auto_trade_log = ustate['auto_trade_log']
              user_tracker = ustate['tracker']
              user_disabled = ustate['disabled_symbols']
              user_active_strategy = ustate['active_strategy']

              # Loop over all supported symbols
              for symbol in SUPPORTED_SYMBOLS:
               try:
                # Skip symbols with auto-buy disabled
                if symbol in user_disabled:
                    continue

                # Determine strategy for this symbol
                sym_strategy = _get_symbol_strategy(symbol, user_active_strategy)
                params = _build_params(sym_strategy)
                trading_rules = TradingRules(params)
                position_sizer = PositionSizingCalculator(params)

                # 1. Fetch 1 year of data for this symbol
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                if hist.empty:
                    print(f"[AUTO-TRADE] No data for {symbol}, skipping")
                    continue

                current_price = float(hist['Close'].iloc[-1])

                # 2. Compute features and get ML prediction (per-symbol)
                forecast_price = None
                features_df = None

                sym_model = trained_models.get(symbol)
                if sym_model is not None:
                    features, features_df = compute_features_from_yfinance(hist, symbol=symbol)
                    if features is not None:
                        try:
                            scaler = sym_model['scaler']
                            X_sc = scaler.transform(features)
                            # Ensemble return prediction
                            ew = sym_model.get('ensemble_weights', {})
                            lr_w = ew.get('lr', 0.5)
                            rf_w = ew.get('rf', 0.5)
                            pred_return = lr_w * sym_model['model_lr'].predict(X_sc)[0] + rf_w * sym_model['model_rf'].predict(X_sc)[0]
                            # Direction classifier check
                            gb_clf = sym_model.get('model_gb_clf')
                            dir_clf = gb_clf if gb_clf is not None else sym_model.get('model_dir_clf')
                            if dir_clf is not None:
                                dir_proba = dir_clf.predict_proba(X_sc)[0]
                                up_proba = dir_proba[1] if len(dir_proba) > 1 else 0.5
                                # Only use ML forecast when direction classifier agrees with return sign
                                if (pred_return > 0 and up_proba > 0.50) or (pred_return < 0 and up_proba < 0.50):
                                    forecast_price = current_price * (1 + pred_return)
                                # else: direction disagrees, fall through to trend fallback
                            else:
                                forecast_price = current_price * (1 + pred_return)
                        except Exception as ml_err:
                            logger.warning(f"[AUTO-TRADE] ML prediction error for {symbol}: {ml_err}")

                if forecast_price is None:
                    # Improved trend-based fallback:
                    # Only project upward when in a confirmed uptrend (SMA20 > SMA50)
                    # and not overbought, with conservative ATR-based projection
                    if features_df is None:
                        _, features_df = compute_features_from_yfinance(hist, symbol=symbol)
                    last_row = features_df.dropna().iloc[-1] if not features_df.dropna().empty else None
                    if last_row is not None:
                        sma20_val = float(last_row.get('SMA_20', current_price))
                        sma50_val = float(last_row.get('SMA_50', current_price))
                        rsi_val = float(last_row.get('RSI_14', 50))
                        atr_val = float(last_row.get('ATR_14', current_price * 0.02))
                        atr_pct = atr_val / current_price
                        
                        if sma20_val > sma50_val and rsi_val < 65:
                            # Uptrend + not overbought: conservative bullish projection
                            forecast_price = current_price * (1 + atr_pct * 0.5)
                        elif sma20_val < sma50_val and rsi_val > 35:
                            # Downtrend + not oversold: bearish projection
                            forecast_price = current_price * (1 - atr_pct * 0.5)
                        else:
                            # Neutral/conflicting: flat projection
                            forecast_price = current_price
                    else:
                        forecast_price = current_price

                # Build features_df if not already set (for market_data extraction)
                if features_df is None:
                    _, features_df = compute_features_from_yfinance(hist, symbol=symbol)

                # Update market price for position valuation
                user_tracker.portfolio.update_market_price(symbol, current_price)

                # 3. Build market_data dict for TradingRules
                last_row = features_df.dropna().iloc[-1]
                market_data = {
                    'RSI_14': float(last_row.get('RSI_14', 50)),
                    'Close': current_price,
                    'SMA_20': float(last_row.get('SMA_20', current_price)),
                    'SMA_50': float(last_row.get('SMA_50', current_price)),
                    'EMA_10': float(last_row.get('EMA_10', current_price)),
                    'EMA_20': float(last_row.get('EMA_20', current_price)),
                }

                # 4. Daily volatility
                recent_returns = hist['Close'].pct_change().dropna().tail(20)
                daily_volatility = float(recent_returns.std()) if len(recent_returns) > 0 else 0.01

                # 4b. Get PredictionEngine signal (BULLISH / BEARISH / NEUTRAL)
                pe_signal = 'NEUTRAL'
                try:
                    pe_df = pd.DataFrame({
                        'Date': hist.index,
                        'price': hist['Close'].values
                    })
                    pe = PredictionEngine(pe_df, symbol=symbol)
                    indicators = pe.calculate_technical_indicators()
                    pe_signal, _ = pe.generate_signal(indicators)
                except Exception as sig_err:
                    logger.warning(f"[AUTO-TRADE] PredictionEngine signal error for {symbol}: {sig_err}")

                print(f"[AUTO-TRADE] {symbol} [{sym_strategy}] | Signal: {pe_signal} | Forecast: ${forecast_price:.2f} vs Current: ${current_price:.2f}")

                # 5. Check open positions for this symbol for stop-loss / take-profit / trailing-stop exits
                positions_to_close = []
                for tid, trade in list(open_positions.items()):
                    if trade.symbol != symbol:
                        continue
                    sl = trade.stop_loss or (trade.entry_price * (1 - params.stop_loss_percent))
                    tp = trade.take_profit or (trade.entry_price * (1 + params.take_profit_target))

                    # Trailing stop: once 2%+ in profit, trail at 1.5% below highest price
                    unrealized_pct = (current_price - trade.entry_price) / trade.entry_price
                    if unrealized_pct >= 0.02:
                        # Track highest price in trade metadata (stored on trade object)
                        highest = getattr(trade, '_highest_price', trade.entry_price)
                        highest = max(highest, current_price)
                        trade._highest_price = highest
                        trailing_sl = highest * (1 - params.trailing_stop_percent)
                        if trailing_sl > sl:
                            sl = trailing_sl  # raise stop to trailing level

                    if current_price <= sl:
                        positions_to_close.append((tid, 'STOP-LOSS', sl))
                    elif current_price >= tp:
                        positions_to_close.append((tid, 'TAKE-PROFIT', tp))

                for tid, reason, trigger_price in positions_to_close:
                    trade = open_positions[tid]
                    user_tracker.close_trade(tid, current_price, datetime.now().isoformat())
                    pnl = (current_price - trade.entry_price) * trade.quantity
                    pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                    del open_positions[tid]

                    log_entry = {
                        'time': now_str,
                        'action': f'SELL ({reason})',
                        'symbol': symbol,
                        'price': round(current_price, 2),
                        'shares': trade.quantity,
                        'pnl': round(pnl, 2),
                        'reason': f'{reason} triggered at ${trigger_price:.2f}'
                    }
                    auto_trade_log.append(log_entry)
                    print(f"[AUTO-TRADE] User {uid}: {reason} SELL {trade.quantity} {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

                if positions_to_close:
                    save_user_auto_state(uid)

                # 6. Check for sell signal on open positions (only when BEARISH + TradingRules agrees)
                symbol_positions = {tid: t for tid, t in open_positions.items() if t.symbol == symbol}
                if symbol_positions and pe_signal == 'BEARISH':
                    sell_signal, sell_conf, sell_reason = trading_rules.get_sell_signal(
                        forecast_price, current_price, market_data, daily_volatility
                    )

                    # NO BYPASS — only trade when TradingRules confirms the signal
                    if sell_signal and sell_conf >= params.confidence_threshold:
                        # Enforce minimum hold period before signal-based exits
                        oldest_tid = next(iter(symbol_positions))
                        trade = open_positions[oldest_tid]
                        try:
                            entry_dt = datetime.fromisoformat(trade.date)
                            days_held = (datetime.now() - entry_dt).days
                        except (ValueError, TypeError):
                            days_held = 999  # if date parsing fails, allow sell
                        
                        if days_held >= params.minimum_hold_days:
                            user_tracker.close_trade(oldest_tid, current_price, datetime.now().isoformat())
                            pnl = (current_price - trade.entry_price) * trade.quantity
                            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                            del open_positions[oldest_tid]

                            log_entry = {
                                'time': now_str,
                                'action': 'SELL (SIGNAL)',
                                'symbol': symbol,
                                'price': round(current_price, 2),
                                'shares': trade.quantity,
                                'pnl': round(pnl, 2),
                                'reason': sell_reason
                            }
                            auto_trade_log.append(log_entry)
                            print(f"[AUTO-TRADE] User {uid}: SIGNAL SELL {trade.quantity} {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | {sell_reason}")
                            save_user_auto_state(uid)
                        else:
                            print(f"[AUTO-TRADE] User {uid}: {symbol} sell signal but hold period not met ({days_held}/{params.minimum_hold_days} days)")

                # 7. Check for buy signal (only when BULLISH AND TradingRules agrees)
                # Per-symbol limit: only 1 open position per symbol at a time
                symbol_open = {tid: t for tid, t in open_positions.items() if t.symbol == symbol}
                has_symbol_position = len(symbol_open) > 0

                # Portfolio circuit breaker: pause trading if portfolio dropped too much
                summary = user_tracker.get_portfolio_summary()
                portfolio_value = summary.get('current_balance', cfg.INITIAL_CAPITAL)
                portfolio_loss_pct = (portfolio_value - cfg.INITIAL_CAPITAL) / cfg.INITIAL_CAPITAL
                circuit_breaker_tripped = portfolio_loss_pct <= params.portfolio_max_loss_percent

                if circuit_breaker_tripped:
                    print(f"[AUTO-TRADE] User {uid}: CIRCUIT BREAKER — portfolio at ${portfolio_value:.0f} ({portfolio_loss_pct:+.1%}), pausing buys")
                elif pe_signal == 'BULLISH' and not has_symbol_position and len(open_positions) < params.max_concurrent_positions:
                    buy_signal, buy_conf, buy_reason = trading_rules.get_buy_signal(
                        forecast_price, current_price, market_data, daily_volatility
                    )

                    # NO BYPASS — only trade when TradingRules confirms the signal
                    if buy_signal and buy_conf >= params.confidence_threshold:
                        available_cash = user_tracker.portfolio.cash

                        shares = position_sizer.calculate_position_size(
                            current_price, portfolio_value, available_cash
                        )

                        if shares > 0 and (shares * current_price) <= available_cash:
                            trade_id = f"AUTO-{uuid.uuid4().hex[:8].upper()}"
                            # ATR-based stop loss: use wider of ATR-based and percentage-based
                            pct_sl = round(current_price * (1 - params.stop_loss_percent), 2)
                            atr_val = float(features_df.dropna()['ATR_14'].iloc[-1]) if features_df is not None and 'ATR_14' in features_df.columns and not features_df.dropna().empty else current_price * params.stop_loss_percent
                            atr_sl = round(current_price - 2 * atr_val, 2)
                            stop_loss = min(pct_sl, atr_sl)  # wider stop = lower price = safer
                            take_profit = round(current_price * (1 + params.take_profit_target), 2)

                            new_trade = Trade(
                                trade_id=trade_id,
                                date=datetime.now().isoformat(),
                                symbol=symbol,
                                action='BUY',
                                quantity=shares,
                                entry_price=current_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                user_id=uid,
                            )
                            user_tracker.add_trade(new_trade)
                            open_positions[trade_id] = new_trade

                            log_entry = {
                                'time': now_str,
                                'action': 'BUY',
                                'symbol': symbol,
                                'price': round(current_price, 2),
                                'shares': shares,
                                'pnl': None,
                                'reason': buy_reason
                            }
                            auto_trade_log.append(log_entry)
                            print(f"[AUTO-TRADE] User {uid}: BUY {shares} {symbol} @ ${current_price:.2f} | SL: ${stop_loss} | TP: ${take_profit} | {buy_reason}")
                            save_user_auto_state(uid)
                        else:
                            print(f"[AUTO-TRADE] User {uid}: {symbol} buy signal but insufficient cash (need ${shares * current_price:.0f}, have ${available_cash:.0f})")
                    else:
                        appreciation = (forecast_price - current_price) / current_price
                        print(f"[AUTO-TRADE] User {uid}: {symbol} no confirmed buy signal | PE: BULLISH | TradingRules: {'YES' if buy_signal else 'NO'} conf={buy_conf:.2f} | Forecast: ${forecast_price:.2f} ({appreciation:+.2%})")
                elif pe_signal == 'BULLISH' and has_symbol_position:
                    print(f"[AUTO-TRADE] User {uid}: {symbol} BULLISH but already has an open position — skipping")
                elif pe_signal == 'NEUTRAL':
                    print(f"[AUTO-TRADE] User {uid}: {symbol} signal is NEUTRAL — no trade action")
                elif len(open_positions) >= params.max_concurrent_positions:
                    print(f"[AUTO-TRADE] User {uid}: Max positions reached ({len(open_positions)}/{params.max_concurrent_positions}), skipping {symbol}")

               except Exception as sym_err:
                print(f"[AUTO-TRADE] User {uid}: Error processing {symbol}: {sym_err}")

              # Keep log trimmed per-user
              if len(auto_trade_log) > 200:
                  ustate['auto_trade_log'] = auto_trade_log[-200:]

        except Exception as e:
            print(f"[AUTO-TRADE] Error in cycle: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(60)


def start_auto_trader():
    """Launch the auto-trading background thread."""
    global auto_trader_thread
    auto_trader_thread = threading.Thread(target=auto_trade_cycle, daemon=True)
    auto_trader_thread.start()
    print("[AUTO-TRADE] Background thread launched")


@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    """Get chart data with technical indicators (SMA, EMA) from live Yahoo Finance"""
    try:
        import yfinance as yf
        
        symbol = request.args.get('symbol', 'AAPL')
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': f'Unsupported symbol: {symbol}'}), 400
        
        # Fetch last 60 trading days of live data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')  # ~60 trading days
        
        if hist.empty:
            raise ValueError("No data from Yahoo Finance")
        
        # Extract prices
        prices = hist['Close'].tolist()
        
        # Calculate moving averages
        sma20 = calculate_sma(prices, window=20)
        ema12 = calculate_ema(prices, span=12)
        
        # Format dates
        dates = [d.strftime('%m/%d') for d in hist.index]
        
        # Clean NaN values for JSON
        def clean_for_json(arr):
            return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in arr]
        
        prices_clean = clean_for_json(prices)
        sma20_clean = clean_for_json(sma20)
        ema12_clean = clean_for_json(ema12)
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        
        return jsonify({
            'dates': dates,
            'prices': prices_clean,
            'sma20': sma20_clean,
            'ema12': ema12_clean,
            'current_price': round(current_price, 2),
            'previous_close': round(previous_close, 2),
            'data_source': 'yahoo_finance'
        })
    
    except Exception as e:
        logger.warning(f"Yahoo Finance chart-data failed ({e}), falling back to CSV")
        # Fallback to CSV
        try:
            return _csv_fallback_chart_data()
        except Exception as fallback_err:
            logger.error(f"CSV chart fallback also failed: {fallback_err}")
            return jsonify({'error': str(e)}), 500


def _csv_fallback_chart_data():
    """Fallback chart data from local CSV when Yahoo Finance is unavailable"""
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'AAPL_stock_data.csv')
    if not os.path.exists(csv_path):
        return jsonify({'dates': [], 'prices': [], 'sma20': [], 'ema12': [],
                        'current_price': None, 'previous_close': None})
    
    df = pd.read_csv(csv_path)
    df = df[df['Price'] != 'Ticker']
    df = df[df['Price'] != 'Date']
    df['Date'] = pd.to_datetime(df['Price'])
    df['ClosePrice'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['ClosePrice'])
    df = df.sort_values('Date')
    recent_df = df.tail(60).copy()
    
    prices = recent_df['ClosePrice'].tolist()
    sma20 = calculate_sma(prices, window=20)
    ema12 = calculate_ema(prices, span=12)
    dates = recent_df['Date'].dt.strftime('%m/%d').tolist()
    
    def clean_for_json(arr):
        return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in arr]
    
    current_price = float(prices[-1]) if prices else None
    previous_close = float(prices[-2]) if len(prices) > 1 else current_price
    
    return jsonify({
        'dates': dates,
        'prices': clean_for_json(prices),
        'sma20': clean_for_json(sma20),
        'ema12': clean_for_json(ema12),
        'current_price': round(current_price, 2) if current_price else None,
        'previous_close': round(previous_close, 2) if previous_close else None,
        'data_source': 'csv'
    })

@app.route('/api/next-day-prediction', methods=['GET'])
def get_next_day_prediction():
    """Get next-day price prediction using trained ML model with live Yahoo Finance data"""
    try:
        import yfinance as yf
        from prediction_engine import PredictionEngine
        
        symbol = request.args.get('symbol', 'AAPL')
        if symbol not in SUPPORTED_SYMBOLS:
            return jsonify({'error': f'Unsupported symbol: {symbol}'}), 400
        
        # Fetch 1 year of data (need 200+ days for SMA_200 warmup)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1y')
        
        if hist.empty:
            raise ValueError("No data returned from Yahoo Finance")
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        
        # --- ML Model Prediction (per-symbol with trained models) ---
        forecast_price = None
        confidence_level = None
        model_name = 'trend-based'
        
        sym_model = trained_models.get(symbol)
        if sym_model is not None:
            features, features_df = compute_features_from_yfinance(hist, symbol=symbol)
            
            if features is not None:
                try:
                    scaler = sym_model['scaler']
                    X_sc = scaler.transform(features)
                    ew = sym_model.get('ensemble_weights', {})
                    lr_w = ew.get('lr', 0.5)
                    rf_w = ew.get('rf', 0.5)
                    pred_return = lr_w * sym_model['model_lr'].predict(X_sc)[0] + rf_w * sym_model['model_rf'].predict(X_sc)[0]
                    forecast_price = current_price * (1 + pred_return)
                    model_name = 'ML Ensemble (LR+RF+GB)'
                    
                    # Confidence from direction classifier
                    gb_clf = sym_model.get('model_gb_clf')
                    dir_clf = gb_clf if gb_clf is not None else sym_model.get('model_dir_clf')
                    if dir_clf is not None:
                        dir_proba = dir_clf.predict_proba(X_sc)[0]
                        up_proba = dir_proba[1] if len(dir_proba) > 1 else 0.5
                        confidence_level = int(max(50, min(85, up_proba * 100)))
                    else:
                        confidence_level = 60
                except Exception as ml_err:
                    logger.warning(f"ML prediction error for {symbol}: {ml_err}")
        
        # --- Fallback: trend-based prediction if model unavailable ---
        if forecast_price is None:
            recent_prices = hist['Close'].tail(10).tolist()
            if len(recent_prices) >= 2:
                trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                forecast_price = current_price * (1 + trend * 0.3)
            else:
                forecast_price = current_price
            
            recent_vol = hist['Close'].tail(20).std() / hist['Close'].tail(20).mean() * 100
            confidence_level = int(max(50, min(85, 75 - recent_vol)))
        
        # --- Signal from PredictionEngine (RSI + MACD + Bollinger Bands) ---
        signal = 'NEUTRAL'
        try:
            pe_df = pd.DataFrame({
                'Date': hist.index,
                'price': hist['Close'].values
            })
            pe = PredictionEngine(pe_df, symbol=symbol)
            indicators = pe.calculate_technical_indicators()
            signal_type, signal_strength = pe.generate_signal(indicators)
            signal = signal_type
        except Exception as sig_err:
            logger.warning(f"Signal generation fallback: {sig_err}")
            if forecast_price > current_price:
                signal = 'BULLISH'
            elif forecast_price < current_price:
                signal = 'BEARISH'
        
        return jsonify({
            'forecast_price': round(forecast_price, 2),
            'current_price': round(current_price, 2),
            'confidence_level': confidence_level,
            'signal': signal,
            'symbol': symbol,
            'model': model_name,
            'data_source': 'yahoo_finance'
        })
    
    except Exception as e:
        logger.error(f"Error in get_next_day_prediction (live): {e}")
        # Final fallback: CSV-based prediction
        try:
            return _csv_fallback_prediction()
        except Exception as fallback_err:
            logger.error(f"CSV fallback also failed: {fallback_err}")
            return jsonify({'error': str(e)}), 500


def _csv_fallback_prediction():
    """Fallback prediction using local CSV data when Yahoo Finance is unavailable"""
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'AAPL_stock_data.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'No data source available'}), 500
    
    df = pd.read_csv(csv_path)
    df = df[df['Price'] != 'Ticker']
    df = df[df['Price'] != 'Date']
    df['Date'] = pd.to_datetime(df['Price'])
    df['ClosePrice'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['ClosePrice'])
    df = df.sort_values('Date')
    
    prices = df['ClosePrice'].tolist()
    current_price = float(prices[-1])
    recent = [float(p) for p in prices[-10:]]
    
    trend = (recent[-1] - recent[0]) / recent[0] if len(recent) >= 2 else 0
    forecast_price = current_price * (1 + trend * 0.3)
    volatility = np.std(recent) / np.mean(recent) * 100 if np.mean(recent) != 0 else 0
    confidence_level = int(max(50, min(85, 75 - volatility)))
    signal = 'BULLISH' if forecast_price > current_price else 'BEARISH' if forecast_price < current_price else 'NEUTRAL'
    
    return jsonify({
        'forecast_price': round(forecast_price, 2),
        'current_price': round(current_price, 2),
        'confidence_level': confidence_level,
        'signal': signal,
        'model': 'trend-fallback',
        'data_source': 'csv'
    })

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

# ============================================================================
# APPLICATION STARTUP
# ============================================================================


# Initialize on app startup
streaming_service, alert_system, notification_service = None, None, None
_app_initialized = False

def init_app():
    """Initialize on app startup"""
    global streaming_service, alert_system, notification_service, _app_initialized
    global trained_models
    # Initialize auth database
    init_auth_db()
    if not _app_initialized:
        _app_initialized = True
        initialize_portfolio()
        
        # Load trained ML models per-symbol from trained_models/{SYMBOL}/ dirs
        model_base = os.path.join(os.path.dirname(__file__), 'trained_models')
        for symbol in SUPPORTED_SYMBOLS:
            sym_dir = os.path.join(model_base, symbol)
            if not os.path.isdir(sym_dir):
                continue
            try:
                artefacts = {}
                for name in ('model_lr', 'model_rf', 'model_dir_clf', 'scaler'):
                    with open(os.path.join(sym_dir, f'{name}.pkl'), 'rb') as f:
                        artefacts[name] = pickle.load(f)
                gb_path = os.path.join(sym_dir, 'model_gb_clf.pkl')
                if os.path.exists(gb_path):
                    with open(gb_path, 'rb') as f:
                        artefacts['model_gb_clf'] = pickle.load(f)
                report_path = os.path.join(sym_dir, 'training_report.json')
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    artefacts['feature_cols'] = report.get('feature_cols', [])
                    artefacts['ensemble_weights'] = report.get('ensemble_weights', {})
                else:
                    artefacts['feature_cols'] = []
                    artefacts['ensemble_weights'] = {}
                trained_models[symbol] = artefacts
                print(f"[OK] ML Model loaded for {symbol}: LR + RF + GB classifier")
            except Exception as e:
                print(f"[WARN] Could not load ML model for {symbol}: {e}")
        
        print(f"[OK] ML Models loaded: {len(trained_models)}/{len(SUPPORTED_SYMBOLS)} symbols")
        
        print("\n" + "="*70)
        print("TASK 5.5: REAL-TIME UPDATES AND ALERTS - INITIALIZED")
        print("="*70)
        print(f"[OK] Streaming Service: {'ACTIVE' if streaming_service.is_running else 'INACTIVE'}")
        print(f"[OK] Alert System: READY ({len(alert_system.rules)} rules)")
        print(f"[OK] Notification Service: READY")
        print(f"[OK] ML Prediction: {len(trained_models)} symbol models loaded")
        print("="*70 + "\n")
        
        # Start auto-trading engine
        start_auto_trader()

        # Pre-compute best strategy per symbol from backtest CSVs
        # so the first page load already has the correct highlights
        auto_select_strategies()


# ============================================================================
# AUTO-TRADE API ENDPOINTS
# ============================================================================

@app.route('/api/auto-trade/status', methods=['GET'])
@login_required
def get_auto_trade_status():
    """Get current auto-trading engine status"""
    ctx = _user_state(session['user_id'])
    open_positions = ctx['open_positions']
    auto_trade_log = ctx['auto_trade_log']
    return jsonify({
        'enabled': ctx['auto_running'],
        'active_strategy': ctx['active_strategy'],
        'symbol_strategies': symbol_strategies,
        'last_check': auto_trade_last_check,
        'open_positions': len(open_positions),
        'total_auto_trades': len(auto_trade_log),
        'positions': [
            {
                'trade_id': tid,
                'entry_price': round(t.entry_price, 2),
                'quantity': t.quantity,
                'stop_loss': round(t.stop_loss, 2) if t.stop_loss else None,
                'take_profit': round(t.take_profit, 2) if t.take_profit else None,
                'date': t.date
            }
            for tid, t in open_positions.items()
        ],
        'recent_actions': auto_trade_log[-10:][::-1]  # last 10, newest first
    })

@app.route('/api/auto-trade/toggle', methods=['POST'])
@login_required
def toggle_auto_trade():
    """Enable or disable auto-trading for the current user"""
    ctx = _user_state(session['user_id'])
    data = request.get_json(silent=True) or {}
    if 'enabled' in data:
        ctx['auto_running'] = bool(data['enabled'])
    else:
        ctx['auto_running'] = not ctx['auto_running']
    
    status = 'enabled' if ctx['auto_running'] else 'disabled'
    print(f"[AUTO-TRADE] User {session['user_id']}: Trading {status} via API")
    save_user_auto_state(session['user_id'])
    return jsonify({
        'enabled': ctx['auto_running'],
        'message': f'Auto-trading {status}'
    })


@app.route('/api/auto-trade/strategy', methods=['GET'])
@login_required
def get_active_strategy():
    """Get the current active trading strategy"""
    ctx = _user_state(session['user_id'])
    return jsonify({
        'active_strategy': ctx['active_strategy'],
        'symbol_strategies': symbol_strategies,
        'available': ['AUTO'] + list(STRATEGIES.keys()),
    })

@app.route('/api/auto-trade/strategy', methods=['POST'])
@login_required
def set_active_strategy():
    """Set the active trading strategy (AUTO or a specific one)"""
    ctx = _user_state(session['user_id'])
    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy', '').upper()
    valid = {'AUTO'} | set(STRATEGIES.keys())
    if strategy not in valid:
        return jsonify({'error': f'Unknown strategy: {strategy}. Choose from: {sorted(valid)}'}), 400
    ctx['active_strategy'] = strategy
    if strategy == 'AUTO' and not symbol_strategies:
        auto_select_strategies()
    save_user_auto_state(session['user_id'])
    print(f"[AUTO-TRADE] User {session['user_id']}: Strategy changed to {ctx['active_strategy']} via API")
    label = 'Auto (best per stock)' if strategy == 'AUTO' else STRATEGIES[strategy]['name']
    return jsonify({
        'active_strategy': ctx['active_strategy'],
        'symbol_strategies': symbol_strategies,
        'message': f'Strategy set to {label}'
    })

@app.route('/api/auto-trade/symbol-strategy', methods=['POST'])
@login_required
def set_symbol_strategy():
    """Set the trading strategy for a specific symbol"""
    global symbol_strategies
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '').upper()
    strategy = data.get('strategy', '').upper()
    if symbol not in SUPPORTED_SYMBOLS:
        return jsonify({'error': f'Unknown symbol: {symbol}'}), 400
    if strategy not in STRATEGIES:
        return jsonify({'error': f'Unknown strategy: {strategy}'}), 400
    symbol_strategies[symbol] = strategy
    save_user_auto_state(session['user_id'])
    print(f"[AUTO-TRADE] {symbol} strategy set to {strategy} via API")
    return jsonify({
        'symbol': symbol,
        'strategy': strategy,
        'symbol_strategies': symbol_strategies
    })

@app.route('/api/auto-trade/symbol-toggle', methods=['POST'])
@login_required
def toggle_symbol_auto_trade():
    """Enable or disable auto-trading for a specific symbol"""
    ctx = _user_state(session['user_id'])
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '').upper()
    if symbol not in SUPPORTED_SYMBOLS:
        return jsonify({'error': f'Unknown symbol: {symbol}'}), 400

    if 'enabled' in data:
        if data['enabled']:
            ctx['disabled_symbols'].discard(symbol)
        else:
            ctx['disabled_symbols'].add(symbol)
    else:
        # toggle
        if symbol in ctx['disabled_symbols']:
            ctx['disabled_symbols'].discard(symbol)
        else:
            ctx['disabled_symbols'].add(symbol)

    enabled = symbol not in ctx['disabled_symbols']
    save_user_auto_state(session['user_id'])
    print(f"[AUTO-TRADE] User {session['user_id']}: {symbol} auto-buy {'enabled' if enabled else 'disabled'} via API")
    return jsonify({
        'symbol': symbol,
        'enabled': enabled,
        'disabled_symbols': list(ctx['disabled_symbols'])
    })


@app.route('/api/auto-trade/symbol-status', methods=['GET'])
@login_required
def get_symbol_auto_trade_status():
    """Get per-symbol auto-trade enabled/disabled status"""
    ctx = _user_state(session['user_id'])
    return jsonify({
        'symbols': {sym: sym not in ctx['disabled_symbols'] for sym in SUPPORTED_SYMBOLS},
        'disabled_symbols': list(ctx['disabled_symbols'])
    })


@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    """Serve the dashboard HTML (protected)."""
    try:
        html_path = os.path.join(os.path.dirname(__file__), 'templates', 'dashboard_trade_history.html')
        return send_file(html_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'portfolio_initialized': _app_initialized
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize systems
    init_app()
    
    # Start Flask app
    print("\n" + "="*70)
    print("TRADE HISTORY & PORTFOLIO API SERVER WITH REAL-TIME ALERTS")
    print("="*70)
    print("\nAvailable Endpoints:\n")
    
    print("Trade History:")
    print("  GET /api/trades/history")
    print("  GET /api/trades/filtered")
    print("  POST /api/trades/search")
    
    print("\nPortfolio:")
    print("  GET /api/portfolio/summary")
    print("  GET /api/portfolio/allocation")
    print("  GET /api/portfolio/statistics")
    print("  GET /api/portfolio/equity-curve")
    print("  GET /api/portfolio/pnl-distribution")
    print("  GET /api/portfolio/performance")
    print("  GET /api/portfolio/symbols")
    print("  GET /api/portfolio/date-range")
    print("  POST /api/portfolio/refresh")
    
    print("\nReal-Time Streaming (Task 5.5):")
    print("  POST /api/streaming/subscribe")
    print("  GET  /api/streaming/prices")
    print("  GET  /api/streaming/status")
    
    print("\nAlert Rules (Task 5.5):")
    print("  GET  /api/alerts/rules")
    print("  POST /api/alerts/rules")
    print("  PUT  /api/alerts/rules/<rule_id>")
    print("  DELETE /api/alerts/rules/<rule_id>")
    print("  POST /api/alerts/evaluate")
    print("  GET  /api/alerts/active")
    print("  POST /api/alerts/<alert_id>/acknowledge")
    print("  POST /api/alerts/<alert_id>/dismiss")
    
    print("\nNotification Preferences (Task 5.5):")
    print("  GET  /api/notifications/preferences")
    print("  POST /api/notifications/preferences")
    print("  GET  /api/notifications/history")
    
    print("\nAuto-Trading Engine:")
    print("  GET  /api/auto-trade/status")
    print("  POST /api/auto-trade/toggle")
    
    print("\nHealth & Status:")
    print("  GET /health")
    
    print("\nDashboard (Task 5.6 - Dark Mode UI):")
    print("  GET  / - Dashboard UI")
    print("  GET  /api/chart-data - AAPL price with SMA(20) and EMA(12)")
    print("  GET  /api/next-day-prediction - Next-day forecast with confidence")
    
    print("\n" + "="*70)
    print(f"Listening on http://{cfg.FLASK_HOST}:{cfg.FLASK_PORT}")

    # --- Public URL via ngrok (optional) ---
    use_ngrok = '--public' in sys.argv
    if use_ngrok and cfg.NGROK_DOMAIN:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(cfg.FLASK_PORT, domain=cfg.NGROK_DOMAIN)
            print(f"\n>>> PUBLIC URL: {public_url}")
            print(">>> Share this link to access from anywhere")
        except Exception as e:
            print(f"\n[ngrok] Could not create tunnel: {e}")
            print("[ngrok] Run without --public for local-only mode")
    elif use_ngrok:
        print("\n[ngrok] Set NGROK_DOMAIN in .env to enable public tunnelling")

    print("="*70 + "\n")
    
    app.run(debug=cfg.FLASK_DEBUG, host=cfg.FLASK_HOST, port=cfg.FLASK_PORT, threaded=True)
