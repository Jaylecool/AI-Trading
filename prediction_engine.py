"""
Task 5.2: Prediction Visualization Engine
Generates multi-day forecasts with confidence intervals and signals

Author: AI Trading System
Date: March 1, 2026
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """
    Generates price predictions with confidence intervals
    Integrates with backtesting results and technical analysis
    """
    
    def __init__(self, historical_data: pd.DataFrame, confidence_level: float = 0.95):
        """
        Initialize prediction engine
        
        Args:
            historical_data: DataFrame with 'Date' and 'price' columns
            confidence_level: Confidence level for intervals (0.90-0.99)
        """
        self.historical_data = historical_data.sort_values('Date')
        self.confidence_level = confidence_level
        self.prices = historical_data['price'].values
        self.dates = pd.to_datetime(historical_data['Date'])
        
    def calculate_technical_indicators(self) -> Dict:
        """Calculate technical indicators for prediction"""
        prices = self.prices
        
        # RSI
        rsi = self._calculate_rsi(prices, period=14)
        
        # MACD
        ema12 = self._calculate_ema(prices, period=12)
        ema26 = self._calculate_ema(prices, period=26)
        macd = ema12 - ema26
        macd_signal = self._calculate_ema(macd, period=9)
        macd_hist = macd - macd_signal
        
        # Bollinger Bands
        sma20 = self._calculate_sma(prices, period=20)
        std20 = pd.Series(prices).rolling(window=20).std().values
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        
        # Volatility
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        return {
            'rsi': float(rsi[-1]) if len(rsi) > 0 else 50,
            'macd': float(macd[-1]) if len(macd) > 0 else 0,
            'macd_signal': float(macd_signal[-1]) if len(macd_signal) > 0 else 0,
            'macd_hist': float(macd_hist[-1]) if len(macd_hist) > 0 else 0,
            'bb_upper': float(bb_upper[-1]) if len(bb_upper) > 0 else prices[-1],
            'bb_lower': float(bb_lower[-1]) if len(bb_lower) > 0 else prices[-1],
            'bb_middle': float(sma20[-1]) if len(sma20) > 0 else prices[-1],
            'volatility': float(volatility)
        }
    
    def generate_signal(self, indicators: Dict) -> Tuple[str, float]:
        """
        Generate trading signal based on indicators
        
        Returns:
            (signal_type, signal_strength)
            signal_type: 'BULLISH', 'BEARISH', 'NEUTRAL', 'MIXED'
            signal_strength: 0-1 confidence
        """
        signals = []
        strengths = []
        
        # RSI Signal
        rsi = indicators['rsi']
        if rsi < 30:
            signals.append('BULLISH')
            strengths.append((30 - rsi) / 30)  # 0-1
        elif rsi > 70:
            signals.append('BEARISH')
            strengths.append((rsi - 70) / 30)  # 0-1
        else:
            signals.append('NEUTRAL')
            strengths.append((50 - abs(rsi - 50)) / 50)
        
        # MACD Signal
        if indicators['macd_hist'] > 0 and indicators['macd'] > indicators['macd_signal']:
            signals.append('BULLISH')
            strengths.append(0.6)
        elif indicators['macd_hist'] < 0 and indicators['macd'] < indicators['macd_signal']:
            signals.append('BEARISH')
            strengths.append(0.6)
        else:
            signals.append('NEUTRAL')
            strengths.append(0.4)
        
        # Price position in Bollinger Bands
        bb_middle = indicators['bb_middle']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        current_price = self.prices[-1]
        
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        if bb_position > 0.7:
            signals.append('BEARISH')
            strengths.append(bb_position - 0.5)
        elif bb_position < 0.3:
            signals.append('BULLISH')
            strengths.append(0.5 - bb_position)
        else:
            signals.append('NEUTRAL')
            strengths.append(0.5)
        
        # Determine final signal
        bullish_count = signals.count('BULLISH')
        bearish_count = signals.count('BEARISH')
        avg_strength = np.mean(strengths)
        
        if bullish_count > bearish_count:
            final_signal = 'BULLISH'
        elif bearish_count > bullish_count:
            final_signal = 'BEARISH'
        else:
            final_signal = 'NEUTRAL' if avg_strength < 0.6 else 'MIXED'
        
        return final_signal, min(avg_strength, 1.0)
    
    def predict_multi_day(self, days_ahead: int = 5) -> Dict:
        """
        Generate multi-day price forecast with confidence intervals
        
        Args:
            days_ahead: Number of days to forecast (1-10)
        
        Returns:
            Dictionary with forecasts, confidence intervals, signals
        """
        prices = self.prices
        current_price = prices[-1]
        
        # Calculate trend
        recent_prices = prices[-20:]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate volatility
        returns = np.diff(prices[-60:]) / prices[-61:-1]
        volatility = np.std(returns)
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators()
        signal_type, signal_strength = self.generate_signal(indicators)
        
        # Generate forecasts
        forecasts = []
        for day in range(1, days_ahead + 1):
            # Linear trend with random walk
            trend_component = trend * day * 0.3  # Decay trend influence
            volatility_component = volatility * np.sqrt(day)
            
            # Expected price based on trend
            forecast_price = current_price * (1 + trend_component)
            
            # Confidence interval (widens with time)
            z_score = 1.96 if self.confidence_level == 0.95 else 2.576 if self.confidence_level == 0.99 else 1.645
            ci_width = forecast_price * volatility_component * z_score * np.sqrt(day / 5)
            
            # Bootstrap confidence from signal strength
            confidence = signal_strength * (1 - 0.1 * day)  # Decrease confidence over time
            confidence = max(0.3, min(confidence, 0.95))  # Bound between 0.3-0.95
            
            forecast_date = (self.dates.iloc[-1] + timedelta(days=day)).strftime('%Y-%m-%d')
            
            forecasts.append({
                'date': forecast_date,
                'day': day,
                'forecast_price': float(forecast_price),
                'lower_bound': float(max(forecast_price - ci_width, current_price * 0.9)),
                'upper_bound': float(forecast_price + ci_width),
                'confidence': float(confidence),
                'direction': 'UP' if forecast_price > current_price else 'DOWN' if forecast_price < current_price else 'FLAT'
            })
        
        return {
            'current_price': float(current_price),
            'signal': signal_type,
            'signal_strength': float(signal_strength),
            'volatility': float(volatility),
            'trend': float(trend),
            'indicators': indicators,
            'forecasts': forecasts,
            'generated_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def _calculate_sma(prices, period=20):
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values
    
    @staticmethod
    def _calculate_ema(prices, period=12):
        """Calculate Exponential Moving Average"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi

# ============================================================================
# CONFIDENCE INTERVAL CALCULATOR
# ============================================================================

class ConfidenceIntervalCalculator:
    """Calculates and visualizes confidence intervals for predictions"""
    
    @staticmethod
    def calculate_historical_confidence(prices: np.ndarray, window: int = 20) -> float:
        """
        Calculate confidence based on historical accuracy
        
        Args:
            prices: Historical price array
            window: Lookback window for accuracy calculation
        
        Returns:
            Confidence score (0-1)
        """
        if len(prices) < window:
            return 0.5
        
        recent = prices[-window:]
        returns = np.diff(recent) / recent[:-1]
        volatility = np.std(returns)
        
        # Lower volatility = higher confidence
        confidence = max(0.0, 1.0 - (volatility * 3))
        return min(confidence, 1.0)
    
    @staticmethod
    def calculate_prediction_bands(forecast_price: float, volatility: float, 
                                   days_ahead: int, confidence_level: float = 0.95) -> Dict:
        """
        Calculate prediction confidence bands
        
        Returns:
            Dict with upper/lower bounds and band width
        """
        # t-distribution critical value
        z_score = 1.96 if confidence_level == 0.95 else 2.576 if confidence_level == 0.99 else 1.645
        
        # Standard error grows with time
        std_error = forecast_price * volatility * np.sqrt(days_ahead)
        ci_width = z_score * std_error
        
        return {
            'forecast': float(forecast_price),
            'upper': float(forecast_price + ci_width),
            'lower': float(forecast_price - ci_width),
            'width': float(ci_width),
            'width_percent': float((ci_width / forecast_price) * 100)
        }

# ============================================================================
# SIGNAL VISUALIZATION
# ============================================================================

class SignalVisualizer:
    """Handles signal visualization and highlighting"""
    
    SIGNAL_COLORS = {
        'BULLISH': '#00D084',    # Green
        'BEARISH': '#FF3B30',    # Red
        'NEUTRAL': '#FFCC33',    # Yellow
        'MIXED': '#9C27B0'       # Purple
    }
    
    SIGNAL_ICONS = {
        'BULLISH': '▲',
        'BEARISH': '▼',
        'NEUTRAL': '◆',
        'MIXED': '◇'
    }
    
    @staticmethod
    def get_signal_color(signal_type: str) -> str:
        """Get color for signal type"""
        return SignalVisualizer.SIGNAL_COLORS.get(signal_type, '#007AFF')
    
    @staticmethod
    def get_signal_icon(signal_type: str) -> str:
        """Get icon for signal type"""
        return SignalVisualizer.SIGNAL_ICONS.get(signal_type, '●')
    
    @staticmethod
    def format_signal_display(signal_type: str, confidence: float) -> Dict:
        """Format signal for UI display"""
        return {
            'type': signal_type,
            'color': SignalVisualizer.get_signal_color(signal_type),
            'icon': SignalVisualizer.get_signal_icon(signal_type),
            'confidence': float(confidence),
            'confidence_percent': f"{confidence*100:.0f}%",
            'display_text': f"{signal_type} ({confidence*100:.0f}%)"
        }

# ============================================================================
# REAL-TIME UPDATE HANDLER
# ============================================================================

class RealTimeUpdateHandler:
    """Manages real-time prediction updates"""
    
    def __init__(self, update_interval: int = 300):
        """
        Initialize update handler
        
        Args:
            update_interval: Update interval in seconds (default 5 minutes)
        """
        self.update_interval = update_interval
        self.last_update = datetime.now()
        self.update_queue = []
    
    def should_update(self) -> bool:
        """Check if update interval has elapsed"""
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed >= self.update_interval
    
    def mark_updated(self):
        """Mark that an update has occurred"""
        self.last_update = datetime.now()
    
    def get_next_update_in(self) -> int:
        """Get seconds until next update"""
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return max(0, int(self.update_interval - elapsed))
    
    @staticmethod
    def create_update_payload(predictions: Dict, portfolio_metrics: Dict) -> Dict:
        """Create payload for real-time update"""
        return {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'portfolio': portfolio_metrics,
            'signal_update': True
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_historical_forecast_comparison(historical_prices: np.ndarray, 
                                           predictions: Dict) -> Dict:
    """
    Compare predictions with historical data
    Useful for accuracy metrics
    """
    return {
        'current_price': float(historical_prices[-1]),
        'recent_high': float(np.max(historical_prices[-20:])),
        'recent_low': float(np.min(historical_prices[-20:])),
        'predictions': predictions,
        'comparison_generated': datetime.now().isoformat()
    }

def format_prediction_for_display(prediction: Dict) -> Dict:
    """Format prediction data for frontend display"""
    return {
        'date': prediction['date'],
        'forecast_price': float(prediction['forecast_price']),
        'lower_bound': float(prediction['lower_bound']),
        'upper_bound': float(prediction['upper_bound']),
        'confidence': float(prediction['confidence']),
        'direction': prediction['direction'],
        'movement': float((prediction['forecast_price'] - prediction.get('current_price', prediction['forecast_price'])) / (prediction.get('current_price', prediction['forecast_price']) or 1) * 100)
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Prediction Visualization Engine - Task 5.2")
    print("=" * 80)
    
    # Example usage
    example_dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    example_prices = 175 + np.cumsum(np.random.randn(60) * 2) + np.linspace(0, 10, 60)
    
    df = pd.DataFrame({
        'Date': example_dates,
        'price': example_prices
    })
    
    # Create engine
    engine = PredictionEngine(df)
    
    # Generate predictions
    predictions = engine.predict_multi_day(days_ahead=5)
    
    print("\n📊 Multi-Day Price Forecast")
    print("-" * 80)
    print(f"Current Price: ${predictions['current_price']:.2f}")
    print(f"Signal: {predictions['signal']} ({predictions['signal_strength']:.2%})")
    print(f"Volatility: {predictions['volatility']:.2%}")
    print(f"Trend: {predictions['trend']:+.2%}")
    
    print("\n📈 Forecasts:")
    for forecast in predictions['forecasts']:
        print(f"  Day {forecast['day']} ({forecast['date']}): "
              f"${forecast['forecast_price']:.2f} "
              f"(${forecast['lower_bound']:.2f} - ${forecast['upper_bound']:.2f}) "
              f"Confidence: {forecast['confidence']:.1%}")
    
    print("\n✅ Prediction engine ready for Task 5.2 dashboard integration")
