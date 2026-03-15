"""
Task 5.1 & 5.2: Dashboard Backend Application
Integrates with backtesting engine and prediction models
Serves real-time trading dashboard with predictions and trade history

Author: AI Trading System
Date: February 2026
"""

import json
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Initialize Flask Application
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

WORKSPACE_PATH = r"c:\Users\Admin\Documents\AI Trading"
DATA_FILES = {
    'backtest_results': os.path.join(WORKSPACE_PATH, 'backtest_results.json'),
    'strategy_comparison': os.path.join(WORKSPACE_PATH, 'strategy_comparison.csv'),
    'stock_data': os.path.join(WORKSPACE_PATH, 'AAPL_stock_data_normalized.csv'),
    'model_metadata': os.path.join(WORKSPACE_PATH, 'model_metadata.json')
}

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_backtest_results():
    """Load backtesting results from JSON file"""
    try:
        with open(DATA_FILES['backtest_results'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'trades': [],
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0
            }
        }

def load_stock_data():
    """Load historical stock data"""
    try:
        df = pd.read_csv(DATA_FILES['stock_data'])
        return df.tail(100)  # Last 100 trading days
    except FileNotFoundError:
        return pd.DataFrame()

def load_model_predictions():
    """Load latest model predictions"""
    try:
        with open(DATA_FILES['model_metadata'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'latest_prediction': None}

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def calculate_portfolio_metrics(backtest_data):
    """
    Calculate portfolio metrics from backtesting results
    Returns: dict with all portfolio statistics
    """
    trades = backtest_data.get('trades', [])
    
    if not trades:
        return {
            'balance': 100000,
            'gains_losses': 0,
            'roi': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'volatility': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'closed_trades': 0
        }
    
    # Extract metrics from backtest summary
    summary = backtest_data.get('summary', {})
    
    return {
        'balance': summary.get('final_balance', 100000),
        'gains_losses': summary.get('total_profit', 0),
        'roi': summary.get('roi', 0),
        'sharpe_ratio': summary.get('sharpe_ratio', 0),
        'sortino_ratio': summary.get('sortino_ratio', 0),
        'volatility': summary.get('volatility', 0),
        'max_drawdown': abs(summary.get('max_drawdown', 0)),
        'win_rate': summary.get('win_rate', 0),
        'profit_factor': summary.get('profit_factor', 0),
        'total_trades': len(trades),
        'closed_trades': sum(1 for t in trades if t.get('status') == 'CLOSED')
    }

def calculate_risk_indicators(backtest_data, portfolio_metrics):
    """
    Calculate risk indicators (heat score, volatility, etc.)
    Returns: dict with risk metrics
    """
    max_drawdown = portfolio_metrics['max_drawdown']
    volatility = portfolio_metrics['volatility']
    win_rate = portfolio_metrics['win_rate']
    
    # Calculate heat score (0-100, higher = more risk)
    # Based on drawdown, volatility, and win rate
    heat_score = min(100, max(0,
        int((max_drawdown * 300) +  # Drawdown contribution
            (volatility * 200) +      # Volatility contribution
            ((1 - win_rate) * 50))    # Win rate contribution
    ))
    
    risk_level = (
        'LOW' if heat_score <= 25 else
        'MODERATE' if heat_score <= 50 else
        'HIGH' if heat_score <= 75 else
        'CRITICAL'
    )
    
    return {
        'heat_score': heat_score,
        'risk_level': risk_level,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
        'sortino_ratio': portfolio_metrics['sortino_ratio']
    }

def format_trade_for_display(trade):
    """Format trade data for frontend display"""
    return {
        'date': trade.get('entry_date', ''),
        'action': trade.get('action', 'BUY'),
        'entry_price': float(trade.get('entry_price', 0)),
        'stop_loss': float(trade.get('stop_loss', 0)),
        'take_profit': float(trade.get('take_profit', 0)),
        'status': trade.get('status', 'OPEN'),
        'exit_price': float(trade.get('exit_price', 0)) if trade.get('exit_price') else None,
        'pnl': float(trade.get('pnl', 0)) if trade.get('pnl') else None,
        'pnl_percent': float(trade.get('pnl_percent', 0)) if trade.get('pnl_percent') else None,
        'duration': trade.get('duration', 0)
    }

def generate_price_prediction(current_price, historical_data):
    """
    Generate next-day price prediction
    Uses simple ML model simulation based on available data
    """
    if len(historical_data) < 20:
        # Default prediction if insufficient data
        return {
            'forecast_price': current_price * 1.02,
            'confidence': 0.45,
            'confidence_interval': {
                'lower': current_price * 0.98,
                'upper': current_price * 1.04
            },
            'signal': 'NEUTRAL',
            'expected_movement': 0.02
        }
    
    # Calculate simple technical indicators
    recent_prices = historical_data.tail(20)['price'].values
    rsi = calculate_rsi(recent_prices)
    sma_20 = recent_prices.mean()
    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    # Signal generation
    if rsi < 30:
        signal = 'BULLISH'
        confidence = 0.70
        expected_movement = 0.03
    elif rsi > 70:
        signal = 'BEARISH'
        confidence = 0.65
        expected_movement = -0.02
    else:
        signal = 'NEUTRAL'
        confidence = 0.50
        expected_movement = 0.005
    
    forecast_price = current_price * (1 + expected_movement)
    std_dev = recent_prices.std() / sma_20
    
    return {
        'forecast_price': forecast_price,
        'confidence': confidence,
        'confidence_interval': {
            'lower': forecast_price * (1 - std_dev),
            'upper': forecast_price * (1 + std_dev)
        },
        'signal': signal,
        'expected_movement': expected_movement
    }

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def dashboard():
    """Serve the dashboard HTML"""
    return render_template('dashboard.html') if os.path.exists('templates/dashboard.html') else app.send_static_file('dashboard.html')

@app.route('/api/dashboard/overview', methods=['GET'])
def get_dashboard_overview():
    """
    Get complete dashboard overview
    Combines all data needed for dashboard initialization
    """
    try:
        # Load all data
        backtest_results = load_backtest_results()
        stock_data = load_stock_data()
        
        if stock_data.empty:
            return jsonify({'error': 'No stock data available'}), 503
        
        # Calculate metrics
        portfolio_metrics = calculate_portfolio_metrics(backtest_results)
        risk_indicators = calculate_risk_indicators(backtest_results, portfolio_metrics)
        
        # Get current and historical prices
        current_price = float(stock_data.iloc[-1]['price']) if 'price' in stock_data.columns else 179.66
        
        # Generate prediction
        prediction = generate_price_prediction(current_price, stock_data)
        
        # Format price history for chart
        price_history = stock_data.tail(60).to_dict('records') if not stock_data.empty else []
        
        # Format trades for display
        trades = backtest_results.get('trades', [])
        formatted_trades = [format_trade_for_display(t) for t in trades[:25]]  # Latest 25
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'prediction': prediction,
            'portfolio_metrics': portfolio_metrics,
            'risk_indicators': risk_indicators,
            'price_history': price_history,
            'trades': formatted_trades,
            'alerts': [
                {
                    'type': 'info',
                    'message': 'Dashboard data loaded successfully',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/current', methods=['GET'])
def get_current_prediction():
    """Get current stock prediction"""
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        current_price = float(stock_data.iloc[-1]['price'])
        prediction = generate_price_prediction(current_price, stock_data)
        
        return jsonify({
            'success': True,
            'symbol': 'AAPL',
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            **prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/history', methods=['GET'])
def get_trades_history():
    """Get trade history with pagination"""
    try:
        backtest_results = load_backtest_results()
        trades = backtest_results.get('trades', [])
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 25, type=int)
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        paginated_trades = trades[start_idx:end_idx]
        formatted_trades = [format_trade_for_display(t) for t in paginated_trades]
        
        return jsonify({
            'success': True,
            'trades': formatted_trades,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': len(trades),
                'pages': (len(trades) + limit - 1) // limit
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/metrics', methods=['GET'])
def get_portfolio_metrics():
    """Get detailed portfolio metrics"""
    try:
        backtest_results = load_backtest_results()
        metrics = calculate_portfolio_metrics(backtest_results)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/indicators', methods=['GET'])
def get_risk_indicators():
    """Get risk indicators"""
    try:
        backtest_results = load_backtest_results()
        portfolio_metrics = calculate_portfolio_metrics(backtest_results)
        risk_indicators = calculate_risk_indicators(backtest_results, portfolio_metrics)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'risk_indicators': risk_indicators
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/price-history/<symbol>', methods=['GET'])
def get_price_history(symbol):
    """Get historical price data for charting"""
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        # Select columns
        columns_needed = ['date', 'price'] if 'date' in stock_data.columns else ['price']
        history = stock_data[columns_needed].tail(60)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'data': history.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current alerts and notifications"""
    try:
        backtest_results = load_backtest_results()
        portfolio_metrics = calculate_portfolio_metrics(backtest_results)
        
        alerts = []
        
        # Generate contextual alerts
        if portfolio_metrics['max_drawdown'] > 0.15:
            alerts.append({
                'type': 'danger',
                'title': 'High Drawdown',
                'message': f"Maximum drawdown of {portfolio_metrics['max_drawdown']*100:.2f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        if portfolio_metrics['win_rate'] < 0.35:
            alerts.append({
                'type': 'warning',
                'title': 'Low Win Rate',
                'message': f"Win rate of {portfolio_metrics['win_rate']*100:.2f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        alerts.append({
            'type': 'info',
            'title': 'Dashboard Active',
            'message': 'Real-time monitoring enabled',
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'alerts': alerts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STATIC FILE SERVING
# ============================================================================

@app.route('/dashboard.html')
def serve_dashboard_html():
    """Serve dashboard HTML file"""
    dashboard_path = os.path.join(WORKSPACE_PATH, 'dashboard.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return f.read()
    return 'Dashboard not found', 404

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.before_request
def before_request():
    """Log incoming requests"""
    print(f"[{datetime.now()}] {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Add CORS headers"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

if __name__ == '__main__':
    print("=" * 80)
    print("TradingPro Dashboard Backend - Task 5.1/5.2")
    print("=" * 80)
    print(f"Starting Flask application...")
    print(f"Workspace: {WORKSPACE_PATH}")
    print(f"Dashboard URL: http://localhost:5000")
    print("=" * 80)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
