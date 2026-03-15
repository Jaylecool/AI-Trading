"""
Task 5.3: Extended Flask Backend with Trade History and Portfolio APIs
Adds endpoints for trade history, portfolio metrics, and visualization data

Endpoints:
- GET /api/trades/history - Complete trade history
- GET /api/trades/filtered - Filtered trades by symbol/date
- GET /api/portfolio/summary - Portfolio metrics
- GET /api/portfolio/allocation - Asset allocation
- GET /api/portfolio/statistics - Trade statistics
- GET /api/portfolio/equity-curve - Daily equity values
- GET /api/portfolio/pnl-distribution - PnL histogram
- GET /api/portfolio/performance - Performance metrics

Author: AI Trading System
Date: March 6, 2026
"""

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

# Supported symbols for multi-stock trading
SUPPORTED_SYMBOLS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']

# Flask application setup
app = Flask(__name__)
app.json_encoder = NaNEncoder
CORS(app)

# Global system instances
portfolio_tracker = None
streaming_service = None
alert_system = None
notification_service = None
active_clients = {}  # client_id -> connection info

# Trained ML model and scaler (loaded at startup)
trained_model = None
trained_scaler = None

# Auto-trading engine state
auto_trader_thread = None
auto_trader_running = True
auto_trade_log = []  # list of dicts: {time, action, symbol, price, shares, reason}
open_positions = {}  # trade_id -> Trade object
auto_trade_last_check = None
auto_trade_disabled_symbols = set()  # symbols with auto-buy turned off

LIVE_STATE_FILE = 'live_auto_state.json'

def save_auto_state():
    """Persist open_positions and auto_trade_log to disk."""
    try:
        state = {
            'open_positions': {
                tid: t.to_dict() for tid, t in open_positions.items()
            },
            'auto_trade_log': auto_trade_log[-200:],
            'disabled_symbols': list(auto_trade_disabled_symbols),
            'saved_at': datetime.now().isoformat()
        }
        state_path = os.path.join(os.path.dirname(__file__), LIVE_STATE_FILE)
        tmp_path = state_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_path, state_path)
    except Exception as e:
        print(f"[PERSIST] Error saving auto state: {e}")

def load_auto_state():
    """Restore open_positions and auto_trade_log from disk."""
    global open_positions, auto_trade_log, auto_trade_disabled_symbols
    state_path = os.path.join(os.path.dirname(__file__), LIVE_STATE_FILE)
    if not os.path.exists(state_path):
        return
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        # Restore auto_trade_log and disabled symbols
        auto_trade_log = state.get('auto_trade_log', [])
        auto_trade_disabled_symbols = set(state.get('disabled_symbols', []))
        
        # Restore open_positions as Trade objects
        pos_data = state.get('open_positions', {})
        for tid, td in pos_data.items():
            trade = Trade(
                trade_id=td['trade_id'],
                date=td['date'],
                symbol=td['symbol'],
                action=td['action'],
                quantity=float(td['quantity']),
                entry_price=float(td['entry_price']),
                stop_loss=float(td['stop_loss']) if td.get('stop_loss') is not None else None,
                take_profit=float(td['take_profit']) if td.get('take_profit') is not None else None,
                exit_date=td.get('exit_date'),
                exit_price=float(td['exit_price']) if td.get('exit_price') is not None else None,
                status=td.get('status', 'OPEN'),
            )
            open_positions[tid] = trade
        
        print(f"[PERSIST] Restored {len(open_positions)} open positions, {len(auto_trade_log)} log entries")
    except Exception as e:
        print(f"[PERSIST] Error loading auto state: {e}")

def initialize_portfolio(backtest_file: str = 'backtest_results.json'):
    """Initialize portfolio from backtesting results"""
    global portfolio_tracker, streaming_service, alert_system, notification_service
    
    portfolio_tracker = PortfolioTracker(initial_balance=100000.0)
    base_dir = os.path.dirname(__file__)
    portfolio_tracker.set_persistence_path(base_dir)
    
    # Initialize streaming service with YAHOO FINANCE (LIVE DATA)
    streaming_service = get_streaming_service(data_source=DataSourceType.YAHOO_FINANCE)
    for sym in SUPPORTED_SYMBOLS:
        streaming_service.subscribe(sym, lambda update: None)
    streaming_service.set_update_frequency(2)  # Update every 2 seconds
    streaming_service.start()
    
    # Initialize alert system
    alert_system = get_alert_system()
    
    # Initialize notification service
    notification_service = get_notification_service()
    
    # Priority 1: Load from live_trades.json (persisted real trades)
    live_path = os.path.join(base_dir, PortfolioTracker.LIVE_TRADES_FILE)
    if os.path.exists(live_path):
        loaded = portfolio_tracker.load_from_json(live_path)
        if loaded:
            print(f"Loaded {len(portfolio_tracker.trade_history)} trades from live persistence")
            # Also restore auto-trade state (open positions, trade log)
            load_auto_state()
        else:
            # live file was empty or corrupt, fall back to backtest
            print("Live trades file empty, falling back to backtest data")
            backtest_path = os.path.join(base_dir, backtest_file)
            if os.path.exists(backtest_path):
                portfolio_tracker.load_from_backtest(backtest_path)
                print(f"Loaded {len(portfolio_tracker.trade_history)} trades from backtest")
                portfolio_tracker.save_to_json()
    else:
        # No live file exists yet - first run, load from backtest and save
        backtest_path = os.path.join(base_dir, backtest_file)
        if os.path.exists(backtest_path):
            portfolio_tracker.load_from_backtest(backtest_path)
            print(f"Loaded {len(portfolio_tracker.trade_history)} trades from backtest")
            # Save to live file so future restarts use persisted data
            portfolio_tracker.save_to_json()
            print(f"Saved initial trades to {PortfolioTracker.LIVE_TRADES_FILE}")
    
    print("Streaming service initialized and started")
    print("Alert system initialized")
    print("Notification service initialized")
    
    return portfolio_tracker

# ============================================================================
# TRADE HISTORY ENDPOINTS
# ============================================================================

@app.route('/api/trades/history', methods=['GET'])
def get_trade_history():
    """Get complete trade history"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        # Get parameters
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        sort_by = request.args.get('sort_by', 'date')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Get trade history
        all_trades = portfolio_tracker.get_trade_history()
        
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
def get_filtered_trades():
    """Get trades filtered by various criteria"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        # Get filter parameters
        symbol = request.args.get('symbol')
        action = request.args.get('action')  # BUY/SELL
        status = request.args.get('status')  # OPEN/CLOSED
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        min_pnl = request.args.get('min_pnl', type=float)
        max_pnl = request.args.get('max_pnl', type=float)
        
        # Create filter object
        filter_obj = TradeHistoryFilter(portfolio_tracker.portfolio.trades)
        
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
def search_trades():
    """Search trades with complex criteria (POST)"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        # Get search criteria from body
        criteria = request.get_json()
        
        # Apply filters
        filter_obj = TradeHistoryFilter(portfolio_tracker.portfolio.trades)
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
def get_portfolio_summary():
    """Get portfolio summary with all metrics"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        # Update market prices for accurate position valuation
        try:
            import yfinance as yf
            for sym in SUPPORTED_SYMBOLS:
                try:
                    price = yf.Ticker(sym).history(period='1d')['Close'].iloc[-1]
                    portfolio_tracker.portfolio.update_market_price(sym, float(price))
                except Exception:
                    pass
        except Exception:
            pass
        
        metrics = portfolio_tracker.get_portfolio_summary()
        formatted = PortfolioVisualizer.format_portfolio_summary(metrics)
        
        return jsonify({
            'status': 'success',
            'summary': formatted,
            'raw': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/allocation', methods=['GET'])
def get_asset_allocation():
    """Get asset allocation for pie chart"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        allocation = PortfolioVisualizer.get_asset_allocation(
            portfolio_tracker.portfolio
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
def get_trade_statistics():
    """Get trade statistics by symbol"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        stats = PortfolioVisualizer.get_trade_statistics(
            portfolio_tracker.portfolio.trades
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
def get_pnl_distribution():
    """Get PnL distribution histogram"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        distribution = PortfolioVisualizer.get_pnl_distribution(
            portfolio_tracker.portfolio.trades
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
def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        metrics = portfolio_tracker.get_portfolio_summary()
        
        # Calculate additional metrics
        closed_trades = [t for t in portfolio_tracker.portfolio.trades 
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
def get_traded_symbols():
    """Get list of symbols traded"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        symbols = set()
        for trade in portfolio_tracker.portfolio.trades:
            symbols.add(trade.symbol)
        
        return jsonify({
            'status': 'success',
            'symbols': sorted(list(symbols)),
            'count': len(symbols)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/date-range', methods=['GET'])
def get_date_range():
    """Get date range of trades"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        if not portfolio_tracker.portfolio.trades:
            return jsonify({
                'status': 'success',
                'start_date': None,
                'end_date': None,
                'trading_days': 0
            })
        
        dates = [datetime.fromisoformat(t.date) 
                for t in portfolio_tracker.portfolio.trades]
        
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
def refresh_portfolio():
    """Refresh portfolio data from sources"""
    try:
        global portfolio_tracker
        initialize_portfolio()
        
        return jsonify({
            'status': 'success',
            'message': 'Portfolio refreshed',
            'trades_loaded': len(portfolio_tracker.trade_history)
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
# DASHBOARD ENDPOINTS (NEW: Dark Mode UI with Live Data)
# ============================================================================

@app.route('/', methods=['GET'])
def dashboard_index():
    """Serve the dashboard HTML"""
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard_trade_history.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            from flask import Response
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
    Compute the 21 technical indicator features from a yfinance history DataFrame.
    Returns a numpy array of shape (1, 21) matching the exact column order
    the trained model expects, plus the raw DataFrame with computed columns.
    
    Requires at least 200+ rows for SMA_200 warmup.
    """
    df = hist_df.copy()
    
    # Rename yfinance columns to match training data column names
    # The ML model was trained with AAPL-suffixed columns
    col_suffix = '_AAPL'  # model always expects _AAPL columns
    df.rename(columns={
        'Close': f'Close{col_suffix}',
        'High': f'High{col_suffix}',
        'Low': f'Low{col_suffix}',
        'Open': f'Open{col_suffix}',
        'Volume': f'Volume{col_suffix}'
    }, inplace=True)
    
    # Simple Moving Averages
    df['SMA_10'] = df['Close_AAPL'].rolling(window=10).mean()
    df['SMA_20'] = df['Close_AAPL'].rolling(window=20).mean()
    df['SMA_50'] = df['Close_AAPL'].rolling(window=50).mean()
    df['SMA_200'] = df['Close_AAPL'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_10'] = df['Close_AAPL'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close_AAPL'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close_AAPL'].ewm(span=50, adjust=False).mean()
    
    # RSI (14-period)
    delta = df['Close_AAPL'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(50)
    
    # MACD
    ema12 = df['Close_AAPL'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close_AAPL'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Rate of Change (12-period)
    df['ROC_12'] = df['Close_AAPL'].pct_change(periods=12) * 100
    
    # Bollinger Bands (20-period, 2 std devs)
    df['BB_Middle'] = df['SMA_20']
    bb_std = df['Close_AAPL'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR (14-period)
    high_low = df['High_AAPL'] - df['Low_AAPL']
    high_close = (df['High_AAPL'] - df['Close_AAPL'].shift()).abs()
    low_close = (df['Low_AAPL'] - df['Close_AAPL'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Volatility (20-period)
    df['Volatility_20'] = df['Close_AAPL'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    
    # Feature columns in exact training order (22 features matching the model)
    feature_columns = [
        'Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Open_AAPL', 'Volume_AAPL',
        'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_10', 'EMA_20', 'EMA_50',
        'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'ROC_12',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR_14', 'Volatility_20'
    ]
    
    # Get the latest complete row (no NaNs in critical features)
    latest = df[feature_columns].dropna().iloc[-1:]
    
    if latest.empty:
        return None, df
    
    return latest.values, df


# ============================================================================
# AUTO-TRADING ENGINE
# ============================================================================

def auto_trade_cycle():
    """Background thread: periodically checks for trade opportunities using ML model + TradingRules."""
    global auto_trader_running, auto_trade_last_check, open_positions, auto_trade_log
    from prediction_engine import PredictionEngine

    # Wait for init to complete
    time.sleep(10)

    params = TradingParameters()
    # Lower thresholds for live auto-trading (default 2% is too strict for daily moves)
    params.buy_threshold = 0.003   # 0.3% predicted appreciation triggers buy consideration
    params.sell_threshold = 0.003  # 0.3% predicted decline triggers sell consideration
    params.confidence_threshold = 0.3  # Lower confidence bar for auto-trading
    params.max_concurrent_positions = len(SUPPORTED_SYMBOLS)  # Allow one position per symbol
    trading_rules = TradingRules(params)
    position_sizer = PositionSizingCalculator(params)

    print("[AUTO-TRADE] Engine started — checking every 60 seconds")

    while True:
        if not auto_trader_running:
            time.sleep(5)
            continue

        try:
            import yfinance as yf

            auto_trade_last_check = datetime.now().isoformat()
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Loop over all supported symbols
            for symbol in SUPPORTED_SYMBOLS:
              try:
                # Skip symbols with auto-buy disabled
                if symbol in auto_trade_disabled_symbols:
                    continue

                # 1. Fetch 1 year of data for this symbol
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')
                if hist.empty:
                    print(f"[AUTO-TRADE] No data for {symbol}, skipping")
                    continue

                current_price = float(hist['Close'].iloc[-1])

                # 2. Compute features and get ML prediction
                #    ML model only works for AAPL; others use trend-based forecast
                forecast_price = None
                features_df = None

                if symbol == 'AAPL' and trained_model is not None and trained_scaler is not None:
                    features, features_df = compute_features_from_yfinance(hist, symbol='AAPL')
                    if features is not None:
                        features_scaled = trained_scaler.transform(features)
                        forecast_price = float(trained_model.predict(features_scaled)[0])

                if forecast_price is None:
                    # Trend-based fallback for non-AAPL symbols or if ML unavailable
                    recent_prices = hist['Close'].tail(10).tolist()
                    if len(recent_prices) >= 2:
                        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                        forecast_price = current_price * (1 + trend * 0.3)
                    else:
                        forecast_price = current_price

                # Build features_df if not already set (for market_data extraction)
                if features_df is None:
                    _, features_df = compute_features_from_yfinance(hist, symbol=symbol)

                # Update market price for position valuation
                portfolio_tracker.portfolio.update_market_price(symbol, current_price)

                # 3. Build market_data dict for TradingRules
                last_row = features_df.dropna().iloc[-1]
                market_data = {
                    'RSI_14': float(last_row.get('RSI_14', 50)),
                    'Close': current_price,
                    'SMA_20': float(last_row.get('SMA_20', current_price)),
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
                    pe = PredictionEngine(pe_df)
                    indicators = pe.calculate_technical_indicators()
                    pe_signal, _ = pe.generate_signal(indicators)
                except Exception as sig_err:
                    logger.warning(f"[AUTO-TRADE] PredictionEngine signal error for {symbol}: {sig_err}")

                print(f"[AUTO-TRADE] {symbol} | Signal: {pe_signal} | Forecast: ${forecast_price:.2f} vs Current: ${current_price:.2f}")

                # 5. Check open positions for this symbol for stop-loss / take-profit exits
                positions_to_close = []
                for tid, trade in list(open_positions.items()):
                    if trade.symbol != symbol:
                        continue
                    sl = trade.stop_loss or (trade.entry_price * (1 - params.stop_loss_percent))
                    tp = trade.take_profit or (trade.entry_price * (1 + params.take_profit_target))

                    if current_price <= sl:
                        positions_to_close.append((tid, 'STOP-LOSS', sl))
                    elif current_price >= tp:
                        positions_to_close.append((tid, 'TAKE-PROFIT', tp))

                for tid, reason, trigger_price in positions_to_close:
                    trade = open_positions[tid]
                    portfolio_tracker.close_trade(tid, current_price, datetime.now().isoformat())
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
                    print(f"[AUTO-TRADE] {reason} SELL {trade.quantity} {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")

                if positions_to_close:
                    save_auto_state()

                # 6. Check for sell signal on open positions for this symbol (only when BEARISH)
                symbol_positions = {tid: t for tid, t in open_positions.items() if t.symbol == symbol}
                if symbol_positions and pe_signal == 'BEARISH':
                    sell_signal, sell_conf, sell_reason = trading_rules.get_sell_signal(
                        forecast_price, current_price, market_data, daily_volatility
                    )

                    # Fallback: trust PE BEARISH signal even when get_sell_signal confidence is low
                    if not sell_signal:
                        sell_signal = True
                        sell_conf = max(sell_conf, 0.5)
                        depreciation = (current_price - forecast_price) / current_price
                        if depreciation > params.sell_threshold:
                            sell_reason = f"Trend model: {depreciation:.2%} decline with partial confirmation"
                        else:
                            sell_reason = f"PredictionEngine BEARISH signal (RSI/MACD/BB reversal)"

                    if sell_signal and sell_conf >= params.confidence_threshold:
                        # Close oldest open position for this symbol
                        oldest_tid = next(iter(symbol_positions))
                        trade = open_positions[oldest_tid]
                        portfolio_tracker.close_trade(oldest_tid, current_price, datetime.now().isoformat())
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
                        print(f"[AUTO-TRADE] SIGNAL SELL {trade.quantity} {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | {sell_reason}")
                        save_auto_state()

                # 7. Check for buy signal (only when BULLISH and room for more positions)
                # Per-symbol limit: only 1 open position per symbol at a time
                symbol_open = {tid: t for tid, t in open_positions.items() if t.symbol == symbol}
                has_symbol_position = len(symbol_open) > 0

                if pe_signal == 'BULLISH' and not has_symbol_position and len(open_positions) < params.max_concurrent_positions:
                    buy_signal, buy_conf, buy_reason = trading_rules.get_buy_signal(
                        forecast_price, current_price, market_data, daily_volatility
                    )

                    appreciation = (forecast_price - current_price) / current_price
                    # Fallback: trust PE BULLISH signal even when get_buy_signal confidence is low
                    if not buy_signal:
                        buy_signal = True
                        buy_conf = max(buy_conf, 0.5)
                        if appreciation > params.buy_threshold:
                            buy_reason = f"{'ML model' if symbol == 'AAPL' else 'Trend model'}: +{appreciation:.2%} with partial confirmation"
                        else:
                            buy_reason = f"PredictionEngine BULLISH signal (RSI/MACD/BB momentum)"

                    if buy_signal and buy_conf >= params.confidence_threshold:
                        summary = portfolio_tracker.get_portfolio_summary()
                        portfolio_value = summary.get('current_balance', 100000)
                        available_cash = portfolio_tracker.portfolio.cash

                        shares = position_sizer.calculate_position_size(
                            current_price, portfolio_value, available_cash
                        )

                        if shares > 0 and (shares * current_price) <= available_cash:
                            trade_id = f"AUTO-{uuid.uuid4().hex[:8].upper()}"
                            stop_loss = round(current_price * (1 - params.stop_loss_percent), 2)
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
                            )
                            portfolio_tracker.add_trade(new_trade)
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
                            print(f"[AUTO-TRADE] BUY {shares} {symbol} @ ${current_price:.2f} | SL: ${stop_loss} | TP: ${take_profit} | {buy_reason}")
                            save_auto_state()
                        else:
                            print(f"[AUTO-TRADE] {symbol} buy signal but insufficient cash (need ${shares * current_price:.0f}, have ${available_cash:.0f})")
                    else:
                        appreciation = (forecast_price - current_price) / current_price
                        print(f"[AUTO-TRADE] {symbol} no buy signal | Forecast: ${forecast_price:.2f} ({appreciation:+.2%}) | {buy_reason}")
                elif pe_signal == 'BULLISH' and has_symbol_position:
                    print(f"[AUTO-TRADE] {symbol} BULLISH but already has an open position — skipping")
                elif pe_signal == 'NEUTRAL':
                    print(f"[AUTO-TRADE] {symbol} signal is NEUTRAL — no trade action")
                elif len(open_positions) >= params.max_concurrent_positions:
                    print(f"[AUTO-TRADE] Max positions reached ({len(open_positions)}/{params.max_concurrent_positions}), skipping {symbol}")

              except Exception as sym_err:
                print(f"[AUTO-TRADE] Error processing {symbol}: {sym_err}")

            # Keep log trimmed
            if len(auto_trade_log) > 200:
                auto_trade_log = auto_trade_log[-200:]

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
        
        # --- ML Model Prediction (only available for AAPL - our trained model) ---
        forecast_price = None
        confidence_level = None
        
        if symbol == 'AAPL' and trained_model is not None and trained_scaler is not None:
            features, features_df = compute_features_from_yfinance(hist, symbol='AAPL')
            
            if features is not None:
                # Scale features with the trained scaler and predict
                features_scaled = trained_scaler.transform(features)
                forecast_price = float(trained_model.predict(features_scaled)[0])
                
                # Confidence: start from model R² (93%), adjust by recent volatility
                recent_returns = hist['Close'].pct_change().dropna().tail(20)
                volatility = float(recent_returns.std()) * np.sqrt(252)  # annualized
                # Higher volatility reduces confidence; base R²=0.93
                vol_penalty = min(volatility * 30, 25)  # cap penalty at 25%
                confidence_level = int(max(50, min(95, 93 - vol_penalty)))
        
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
            pe = PredictionEngine(pe_df)
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
            'model': 'Linear Regression' if (symbol == 'AAPL' and trained_model is not None) else 'trend-based',
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
portfolio_tracker, streaming_service, alert_system, notification_service = None, None, None, None

def init_app():
    """Initialize on app startup"""
    global portfolio_tracker, streaming_service, alert_system, notification_service
    global trained_model, trained_scaler
    if portfolio_tracker is None:
        initialize_portfolio()
        
        # Load trained ML model and scaler for predictions
        model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        try:
            model_path = os.path.join(model_dir, 'model_Linear_Regression.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(model_path, 'rb') as f:
                trained_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                trained_scaler = pickle.load(f)
            print(f"[OK] ML Model: Linear Regression loaded from {model_path}")
            print(f"[OK] Scaler: StandardScaler loaded from {scaler_path}")
        except Exception as e:
            trained_model = None
            trained_scaler = None
            print(f"[WARN] Could not load ML model: {e}")
        
        print("\n" + "="*70)
        print("TASK 5.5: REAL-TIME UPDATES AND ALERTS - INITIALIZED")
        print("="*70)
        print(f"[OK] Streaming Service: {'ACTIVE' if streaming_service.is_running else 'INACTIVE'}")
        print(f"[OK] Alert System: READY ({len(alert_system.rules)} rules)")
        print(f"[OK] Notification Service: READY")
        print(f"[OK] ML Prediction: {'ACTIVE (Linear Regression)' if trained_model else 'FALLBACK (trend-based)'}")
        print("="*70 + "\n")
        
        # Start auto-trading engine
        start_auto_trader()


# ============================================================================
# AUTO-TRADE API ENDPOINTS
# ============================================================================

@app.route('/api/auto-trade/status', methods=['GET'])
def get_auto_trade_status():
    """Get current auto-trading engine status"""
    return jsonify({
        'enabled': auto_trader_running,
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
def toggle_auto_trade():
    """Enable or disable auto-trading"""
    global auto_trader_running
    data = request.get_json(silent=True) or {}
    if 'enabled' in data:
        auto_trader_running = bool(data['enabled'])
    else:
        auto_trader_running = not auto_trader_running
    
    status = 'enabled' if auto_trader_running else 'disabled'
    print(f"[AUTO-TRADE] Trading {status} via API")
    return jsonify({
        'enabled': auto_trader_running,
        'message': f'Auto-trading {status}'
    })


@app.route('/api/auto-trade/symbol-toggle', methods=['POST'])
def toggle_symbol_auto_trade():
    """Enable or disable auto-trading for a specific symbol"""
    global auto_trade_disabled_symbols
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol', '').upper()
    if symbol not in SUPPORTED_SYMBOLS:
        return jsonify({'error': f'Unknown symbol: {symbol}'}), 400

    if 'enabled' in data:
        if data['enabled']:
            auto_trade_disabled_symbols.discard(symbol)
        else:
            auto_trade_disabled_symbols.add(symbol)
    else:
        # toggle
        if symbol in auto_trade_disabled_symbols:
            auto_trade_disabled_symbols.discard(symbol)
        else:
            auto_trade_disabled_symbols.add(symbol)

    enabled = symbol not in auto_trade_disabled_symbols
    save_auto_state()
    print(f"[AUTO-TRADE] {symbol} auto-buy {'enabled' if enabled else 'disabled'} via API")
    return jsonify({
        'symbol': symbol,
        'enabled': enabled,
        'disabled_symbols': list(auto_trade_disabled_symbols)
    })


@app.route('/api/auto-trade/symbol-status', methods=['GET'])
def get_symbol_auto_trade_status():
    """Get per-symbol auto-trade enabled/disabled status"""
    return jsonify({
        'symbols': {sym: sym not in auto_trade_disabled_symbols for sym in SUPPORTED_SYMBOLS},
        'disabled_symbols': list(auto_trade_disabled_symbols)
    })


@app.route('/', methods=['GET'])
@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Serve the dashboard HTML"""
    try:
        html_path = os.path.join(os.path.dirname(__file__), 'dashboard_trade_history.html')
        return send_file(html_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'portfolio_initialized': portfolio_tracker is not None
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
    print("Listening on http://localhost:5000")

    # --- Public URL via ngrok (optional) ---
    use_ngrok = '--public' in sys.argv
    if use_ngrok:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(5000, domain="soppiest-willis-uncasked.ngrok-free.dev")
            print(f"\n>>> PUBLIC URL: {public_url}")
            print(">>> Share this link to access from anywhere")
        except Exception as e:
            print(f"\n[ngrok] Could not create tunnel: {e}")
            print("[ngrok] Run without --public for local-only mode")

    print("="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
