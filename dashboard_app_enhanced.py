"""
Task 5.2: Enhanced Dashboard Application
Integrates prediction engine with real-time updates

Author: AI Trading System
Date: March 1, 2026
"""

import json
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

# Import the new prediction engine
from prediction_engine import (
    PredictionEngine, 
    ConfidenceIntervalCalculator,
    SignalVisualizer,
    RealTimeUpdateHandler
)

# Initialize Flask Application
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKSPACE_PATH = r"c:\Users\Admin\Documents\AI Trading"
DATA_FILES = {
    'backtest_results': os.path.join(WORKSPACE_PATH, 'backtest_results.json'),
    'stock_data': os.path.join(WORKSPACE_PATH, 'AAPL_stock_data_normalized.csv'),
    'model_metadata': os.path.join(WORKSPACE_PATH, 'model_metadata.json')
}

# Real-time update handler
update_handler = RealTimeUpdateHandler(update_interval=300)  # 5 minutes

# ============================================================================
# DATA LOADING
# ============================================================================

def load_stock_data():
    """Load historical stock data"""
    try:
        df = pd.read_csv(DATA_FILES['stock_data'])
        # Rename columns for consistency
        if 'Close' in df.columns:
            df['price'] = df['Close']
        elif 'price' not in df.columns:
            df['price'] = df.iloc[:, 1]  # Assume second column is price
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def load_backtest_results():
    """Load backtesting results"""
    try:
        with open(DATA_FILES['backtest_results'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# ============================================================================
# ENHANCED PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/predictions/multi-day', methods=['GET'])
def get_multi_day_predictions():
    """
    Get multi-day price forecasts with confidence intervals
    
    Query Parameters:
    - days: Number of days to forecast (1-10, default 5)
    - confidence: Confidence level (0.90, 0.95, 0.99)
    """
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data available'}), 503
        
        days = request.args.get('days', 5, type=int)
        days = max(1, min(days, 10))  # Constrain 1-10
        confidence_level = request.args.get('confidence', 0.95, type=float)
        
        # Create prediction engine
        engine = PredictionEngine(stock_data, confidence_level=confidence_level)
        predictions = engine.predict_multi_day(days_ahead=days)
        
        return jsonify({
            'success': True,
            'symbol': 'AAPL',
            'timestamp': datetime.now().isoformat(),
            **predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/confidence-intervals', methods=['GET'])
def get_confidence_intervals():
    """
    Get confidence intervals for predictions
    Shows how prediction uncertainty grows over time
    """
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        prices = stock_data['price'].values
        current_price = float(prices[-1])
        
        # Calculate historical volatility
        returns = np.diff(prices[-60:]) / prices[-59:-1]
        volatility = float(np.std(returns))
        
        # Generate intervals for 1-10 days
        intervals = []
        for day in range(1, 11):
            ci = ConfidenceIntervalCalculator.calculate_prediction_bands(
                current_price, volatility, day, confidence_level=0.95
            )
            intervals.append({
                'day': day,
                'forecast_price': ci['forecast'],
                'lower_bound': ci['lower'],
                'upper_bound': ci['upper'],
                'band_width': ci['width'],
                'band_width_percent': ci['width_percent']
            })
        
        return jsonify({
            'success': True,
            'current_price': current_price,
            'volatility': volatility,
            'intervals': intervals,
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/signals', methods=['GET'])
def get_trading_signals():
    """
    Get current trading signals based on technical analysis
    """
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        # Create engine and calculate signals
        engine = PredictionEngine(stock_data)
        indicators = engine.calculate_technical_indicators()
        signal_type, signal_strength = engine.generate_signal(indicators)
        
        # Format for display
        signal_display = SignalVisualizer.format_signal_display(signal_type, signal_strength)
        
        return jsonify({
            'success': True,
            'signal': signal_display,
            'indicators': {
                'rsi': round(indicators['rsi'], 2),
                'macd': round(indicators['macd'], 4),
                'macd_signal': round(indicators['macd_signal'], 4),
                'macd_hist': round(indicators['macd_hist'], 4),
                'bb_upper': round(indicators['bb_upper'], 2),
                'bb_middle': round(indicators['bb_middle'], 2),
                'bb_lower': round(indicators['bb_lower'], 2),
                'volatility': round(indicators['volatility'], 4)
            },
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/chart-data', methods=['GET'])
def get_prediction_chart_data():
    """
    Get data formatted for chart visualization
    Includes historical prices + multi-day forecasts
    """
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        # Prepare historical data
        historical = stock_data.tail(60).copy()
        historical_data = []
        for idx, row in historical.iterrows():
            historical_data.append({
                'date': str(row.get('Date', row.get('date', ''))),
                'price': float(row['price']),
                'type': 'historical'
            })
        
        # Generate forecasts
        engine = PredictionEngine(stock_data)
        predictions = engine.predict_multi_day(days_ahead=5)
        
        # Add SMA and EMA to historical
        prices = stock_data['price'].tail(60).values
        sma20 = pd.Series(prices).rolling(window=20).mean().values
        ema12 = pd.Series(prices).rolling(window=12).mean().values
        
        # Create chart series
        result = {
            'success': True,
            'historical': {
                'dates': [d['date'] for d in historical_data],
                'prices': [d['price'] for d in historical_data],
                'sma20': [float(x) if not np.isnan(x) else None for x in sma20],
                'ema12': [float(x) if not np.isnan(x) else None for x in ema12]
            },
            'forecast': {
                'dates': [f['date'] for f in predictions['forecasts']],
                'prices': [f['forecast_price'] for f in predictions['forecasts']],
                'upper_band': [f['upper_bound'] for f in predictions['forecasts']],
                'lower_band': [f['lower_bound'] for f in predictions['forecasts']],
                'confidence': [f['confidence'] for f in predictions['forecasts']]
            },
            'signal': {
                'type': predictions['signal'],
                'strength': predictions['signal_strength']
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/next-day', methods=['GET'])
def get_next_day_prediction():
    """
    Get next-day price prediction (simplified endpoint for dashboard)
    """
    try:
        stock_data = load_stock_data()
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        current_price = float(stock_data['price'].iloc[-1])
        
        # Generate 1-day forecast
        engine = PredictionEngine(stock_data)
        predictions = engine.predict_multi_day(days_ahead=1)
        
        forecast = predictions['forecasts'][0]
        indicators = predictions['indicators']
        
        movement = ((forecast['forecast_price'] - current_price) / current_price) * 100
        
        return jsonify({
            'success': True,
            'current_price': float(current_price),
            'forecast_price': float(forecast['forecast_price']),
            'lower_bound': float(forecast['lower_bound']),
            'upper_bound': float(forecast['upper_bound']),
            'confidence': float(forecast['confidence']),
            'signal': predictions['signal'],
            'signal_strength': float(predictions['signal_strength']),
            'expected_movement': float(movement),
            'expected_movement_pct': f"{movement:.2f}%",
            'volatility': float(predictions['volatility']),
            'rsi': float(indicators['rsi']),
            'macd': float(indicators['macd']),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/real-time-update', methods=['GET'])
def get_real_time_update():
    """
    Get updates only if sufficient time has elapsed
    Useful for real-time dashboard updates
    """
    try:
        # Check if update should occur
        should_update = update_handler.should_update()
        next_update_in = update_handler.get_next_update_in()
        
        if not should_update:
            return jsonify({
                'success': True,
                'should_update': False,
                'next_update_in_seconds': next_update_in,
                'last_update': update_handler.last_update.isoformat()
            })
        
        # Mark that update occurred
        update_handler.mark_updated()
        
        # Fetch fresh prediction data
        stock_data = load_stock_data()
        backtest_results = load_backtest_results()
        
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        # Generate predictions
        engine = PredictionEngine(stock_data)
        predictions = engine.predict_multi_day(days_ahead=5)
        
        # Portfolio snapshot
        portfolio_summary = backtest_results.get('summary', {})
        
        return jsonify({
            'success': True,
            'should_update': True,
            'update_timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'portfolio_summary': {
                'balance': portfolio_summary.get('final_balance', 0),
                'roi': portfolio_summary.get('roi', 0),
                'sharpe': portfolio_summary.get('sharpe_ratio', 0)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/accuracy-metrics', methods=['GET'])
def get_prediction_accuracy_metrics():
    """
    Get prediction accuracy metrics from backtesting
    Shows how well the model performed
    """
    try:
        backtest_results = load_backtest_results()
        
        if not backtest_results:
            return jsonify({'error': 'No backtesting data'}), 503
        
        summary = backtest_results.get('summary', {})
        
        accuracy_metrics = {
            'total_predictions': len(backtest_results.get('trades', [])),
            'winning_trades': summary.get('winning_trades', 0),
            'losing_trades': summary.get('losing_trades', 0),
            'win_rate': summary.get('win_rate', 0),
            'profit_factor': summary.get('profit_factor', 0),
            'average_win': summary.get('avg_win', 0),
            'average_loss': summary.get('avg_loss', 0),
            'largest_win': summary.get('largest_win', 0),
            'largest_loss': summary.get('largest_loss', 0),
            'sharpe_ratio': summary.get('sharpe_ratio', 0),
            'sortino_ratio': summary.get('sortino_ratio', 0)
        }
        
        return jsonify({
            'success': True,
            'accuracy_metrics': accuracy_metrics,
            'calculated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# LEGACY ENDPOINTS (from Task 5.1)
# ============================================================================

@app.route('/api/dashboard/overview', methods=['GET'])
def get_dashboard_overview():
    """Get complete dashboard overview"""
    try:
        stock_data = load_stock_data()
        backtest_results = load_backtest_results()
        
        if stock_data.empty:
            return jsonify({'error': 'No stock data'}), 503
        
        current_price = float(stock_data['price'].iloc[-1])
        
        # Generate predictions
        engine = PredictionEngine(stock_data)
        predictions = engine.predict_multi_day(days_ahead=5)
        
        summary = backtest_results.get('summary', {})
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'prediction': predictions['forecasts'][0] if predictions['forecasts'] else {},
            'signal': predictions['signal'],
            'signal_strength': predictions['signal_strength'],
            'portfolio_metrics': {
                'balance': summary.get('final_balance', 0),
                'gains_losses': summary.get('total_profit', 0),
                'roi': summary.get('roi', 0),
                'sharpe_ratio': summary.get('sharpe_ratio', 0),
                'max_drawdown': abs(summary.get('max_drawdown', 0))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("TradingPro Dashboard - Task 5.2 Enhanced Prediction API")
    print("=" * 80)
    print(f"Workspace: {WORKSPACE_PATH}")
    print(f"Server URL: http://localhost:5000")
    print("\nNew Endpoints (Task 5.2):")
    print("  GET /api/predictions/multi-day - Multi-day forecasts")
    print("  GET /api/predictions/confidence-intervals - CI bands")
    print("  GET /api/predictions/signals - Trading signals")
    print("  GET /api/predictions/chart-data - Chart visualization data")
    print("  GET /api/predictions/next-day - Next-day forecast")
    print("  GET /api/predictions/real-time-update - Real-time updates")
    print("  GET /api/predictions/accuracy-metrics - Model accuracy")
    print("=" * 80)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
