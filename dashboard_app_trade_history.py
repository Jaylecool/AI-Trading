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
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid

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

# Flask application setup
app = Flask(__name__)
CORS(app)

# Global system instances
portfolio_tracker = None
streaming_service = None
alert_system = None
notification_service = None
active_clients = {}  # client_id -> connection info

def initialize_portfolio(backtest_file: str = 'backtest_results.json'):
    """Initialize portfolio from backtesting results"""
    global portfolio_tracker, streaming_service, alert_system, notification_service
    
    portfolio_tracker = PortfolioTracker(initial_balance=100000.0)
    
    # Initialize streaming service with YAHOO FINANCE (LIVE DATA)
    streaming_service = get_streaming_service(data_source=DataSourceType.YAHOO_FINANCE)
    streaming_service.subscribe('AAPL', lambda update: None)  # Subscribe to AAPL
    streaming_service.set_update_frequency(2)  # Update every 2 seconds
    streaming_service.start()
    
    # Initialize alert system
    alert_system = get_alert_system()
    
    # Initialize notification service
    notification_service = get_notification_service()
    
    # Try to load from backtest results
    backtest_path = os.path.join(os.path.dirname(__file__), backtest_file)
    if os.path.exists(backtest_path):
        portfolio_tracker.load_from_backtest(backtest_path)
        print(f"Loaded {len(portfolio_tracker.trade_history)} trades from backtest")
    
    # Try to load from CSV if available
    csv_path = os.path.join(os.path.dirname(__file__), 'trades.csv')
    if os.path.exists(csv_path):
        portfolio_tracker.load_from_csv(csv_path)
        print(f"Loaded {len(portfolio_tracker.trade_history)} trades from CSV")
    
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
    """Get portfolio equity curve from trades"""
    try:
        if not portfolio_tracker:
            return jsonify({'error': 'Portfolio not initialized'}), 500
        
        # Get the daily equity curve from portfolio
        equity_data = PortfolioVisualizer.get_daily_equity_curve(portfolio_tracker.portfolio)
        
        logger.info(f"Returning {len(equity_data['dates'])} equity curve points")
        if equity_data['equity']:
            logger.info(f"Equity range: ${min(equity_data['equity']):.2f} - ${max(equity_data['equity']):.2f}")
        
        return jsonify({
            'status': 'success',
            'dates': equity_data['dates'],
            'equity': equity_data['equity'],
            'type': 'line'
        })
    
    except Exception as e:
        logger.error(f"Error in get_equity_curve: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/live-price', methods=['GET'])
def get_live_price():
    """Get current AAPL price with recent history"""
    try:
        import yfinance as yf
        
        aapl = yf.Ticker('AAPL')
        
        # Get current and recent prices
        hist = aapl.history(period='1y')
        info = aapl.info if hasattr(aapl, 'info') else {}
        
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
                'symbol': 'AAPL',
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
        with open(dashboard_path, 'r') as f:
            return f.read()
    return jsonify({'error': 'Dashboard not found'}), 404

def calculate_sma(prices, window=20):
    """Calculate Simple Moving Average"""
    df = pd.DataFrame({'price': prices})
    return df['price'].rolling(window=window).mean().tolist()

def calculate_ema(prices, span=12):
    """Calculate Exponential Moving Average"""
    df = pd.DataFrame({'price': prices})
    return df['price'].ewm(span=span).mean().tolist()

@app.route('/api/chart-data', methods=['GET'])
def get_chart_data():
    """Get AAPL chart data with technical indicators (SMA, EMA)"""
    try:
        # Load AAPL data from CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'AAPL_stock_data.csv')
        if not os.path.exists(csv_path):
            # Return mock data if file not found
            return jsonify({
                'dates': [],
                'prices': [],
                'sma20': [],
                'ema12': [],
                'current_price': 179.66,
                'previous_close': 177.21
            })
        
        # Read CSV with proper headers
        df = pd.read_csv(csv_path)
        
        # Skip metadata rows (Ticker and Date label rows)
        df = df[df['Price'] != 'Ticker']  # Remove Ticker header row
        df = df[df['Price'] != 'Date']    # Remove Date label row
        
        # Convert columns to proper types
        df['Date'] = pd.to_datetime(df['Price'])
        df['ClosePrice'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['ClosePrice'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Get recent data (last 60 days for good moving average calculation)
        recent_df = df.tail(60).copy()
        
        # Extract prices as list
        prices = recent_df['ClosePrice'].tolist()
        
        # Calculate moving averages
        sma20 = calculate_sma(prices, window=20)
        ema12 = calculate_ema(prices, span=12)
        
        # Helper function to replace NaN with None for JSON serialization
        def clean_for_json(arr):
            return [None if pd.isna(x) else float(x) if isinstance(x, (int, float, np.number)) else x for x in arr]
        
        # Format dates
        dates = recent_df['Date'].dt.strftime('%m/%d').tolist()
        
        # Clean all arrays of NaN values
        prices = clean_for_json(prices)
        sma20 = clean_for_json(sma20)
        ema12 = clean_for_json(ema12)
        
        # Current price (last price) and previous close
        current_price = float(prices[-1]) if prices[-1] is not None else None
        previous_close = float(prices[-2]) if len(prices) > 1 and prices[-2] is not None else current_price
        
        return jsonify({
            'dates': dates,
            'prices': prices,
            'sma20': sma20,
            'ema12': ema12,
            'current_price': current_price,
            'previous_close': previous_close
        })
    
    except Exception as e:
        logger.error(f"Error in get_chart_data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/next-day-prediction', methods=['GET'])
def get_next_day_prediction():
    """Get next-day price prediction with confidence level"""
    try:
        # Load AAPL data
        csv_path = os.path.join(os.path.dirname(__file__), 'AAPL_stock_data.csv')
        
        if not os.path.exists(csv_path):
            # Return mock prediction if file not found
            return jsonify({
                'forecast_price': 184.15,
                'current_price': 179.66,
                'confidence_level': 76,
                'signal': 'BULLISH'
            })
        
        # Read and process data
        df = pd.read_csv(csv_path)
        
        # Skip metadata rows
        df = df[df['Price'] != 'Ticker']  # Remove Ticker header row
        df = df[df['Price'] != 'Date']    # Remove Date label row
        
        # Parse the CSV structure correctly
        df['Date'] = pd.to_datetime(df['Price'])
        df['ClosePrice'] = pd.to_numeric(df['Close'], errors='coerce')
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['ClosePrice'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Get current (last) price
        prices = df['ClosePrice'].tolist()
        def safe_float(val):
            try:
                if pd.isnull(val) or val is None:
                    return None
                return float(val)
            except Exception:
                return None

        current_price = safe_float(prices[-1])
        
        # Calculate a simple forecast based on recent trend
        recent_prices = [safe_float(p) for p in prices[-10:]]  # Last 10 days
        recent_prices = [p for p in recent_prices if p is not None]
        
        # Calculate trend
        if len(recent_prices) >= 2:
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
        else:
            trend = 0
        
        # Estimate forecast price (simple: current price + trend adjustment)
        trend_factor = 1 + (trend / 100) * 0.3  # Use 30% of trend
        forecast_price = current_price * trend_factor if current_price is not None else None
        
        # Calculate confidence level based on volatility
        if recent_prices:
            volatility = np.std(recent_prices) / np.mean(recent_prices) * 100 if np.mean(recent_prices) != 0 else 0
        else:
            volatility = 0
        confidence_level = max(50, min(95, 75 - volatility))  # 50-95% range
        
        # Determine signal based on forecast vs current
        if forecast_price is not None and current_price is not None:
            price_change = forecast_price - current_price
            if price_change > 0:
                signal = 'BULLISH'
            elif price_change < 0:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
        else:
            signal = 'NEUTRAL'
        
        return jsonify({
            'forecast_price': round(forecast_price, 2) if forecast_price is not None else None,
            'current_price': round(current_price, 2) if current_price is not None else None,
            'confidence_level': int(confidence_level) if confidence_level is not None else None,
            'signal': signal
        })
    
    except Exception as e:
        logger.error(f"Error in get_next_day_prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

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
    if portfolio_tracker is None:
        initialize_portfolio()
        print("\n" + "="*70)
        print("TASK 5.5: REAL-TIME UPDATES AND ALERTS - INITIALIZED")
        print("="*70)
        print(f"✓ Streaming Service: {'ACTIVE' if streaming_service.is_running else 'INACTIVE'}")
        print(f"✓ Alert System: READY ({len(alert_system.rules)} rules)")
        print(f"✓ Notification Service: READY")
        print("="*70 + "\n")

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
    
    print("\nHealth & Status:")
    print("  GET /health")
    
    print("\nDashboard (Task 5.6 - Dark Mode UI):")
    print("  GET  / - Dashboard UI")
    print("  GET  /api/chart-data - AAPL price with SMA(20) and EMA(12)")
    print("  GET  /api/next-day-prediction - Next-day forecast with confidence")
    
    print("\n" + "="*70)
    print("Listening on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='localhost', port=5000, threaded=True)
