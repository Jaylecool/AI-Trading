"""
Task 5.2: Prediction Visualization Engine  (ML-powered)
Generates multi-day forecasts with confidence intervals and signals
using trained per-stock ML models (Linear Regression + Random Forest ensemble
plus a direction classifier).

Falls back to technical-indicator-only mode when no trained model exists.

Author: AI Trading System
Date: March 14, 2026
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, 'trained_models')

# TFT and NLP imports (optional — system works without them)
try:
    from transformer_predictor import predict_tft as _predict_tft, SEQUENCE_LENGTH as _TFT_SEQ_LEN
    _TFT_AVAILABLE = True
except Exception:
    _TFT_AVAILABLE = False
    _TFT_SEQ_LEN = 30

try:
    from nlp_sentiment_service import get_sentiment_features as _get_sentiment_features
    _SENTIMENT_AVAILABLE = True
except Exception:
    _SENTIMENT_AVAILABLE = False


# ============================================================================
# MODEL CACHE  (load once, reuse)
# ============================================================================

_model_cache: Dict[str, Dict] = {}


def _load_models(symbol: str) -> Optional[Dict]:
    """Load trained artefacts for *symbol*. Returns None if not available."""
    if symbol in _model_cache:
        return _model_cache[symbol]

    model_dir = os.path.join(_MODELS_DIR, symbol)
    if not os.path.isdir(model_dir):
        return None

    try:
        artefacts = {}
        for name in ('model_lr', 'model_rf', 'model_dir_clf', 'scaler'):
            path = os.path.join(model_dir, f'{name}.pkl')
            with open(path, 'rb') as f:
                artefacts[name] = pickle.load(f)
        # Load GradientBoosting classifier if available
        gb_path = os.path.join(model_dir, 'model_gb_clf.pkl')
        if os.path.exists(gb_path):
            with open(gb_path, 'rb') as f:
                artefacts['model_gb_clf'] = pickle.load(f)
        report_path = os.path.join(model_dir, 'training_report.json')
        with open(report_path, 'r') as f:
            artefacts['report'] = json.load(f)
        _model_cache[symbol] = artefacts
        return artefacts
    except (FileNotFoundError, KeyError, pickle.UnpicklingError, json.JSONDecodeError):
        return None

# ============================================================================
# PREDICTION ENGINE  (ML-powered)
# ============================================================================

class PredictionEngine:
    """
    Generates price predictions with confidence intervals.
    Uses trained ML models when available; falls back to technical indicators.
    """

    def __init__(self, historical_data: pd.DataFrame, confidence_level: float = 0.95,
                 symbol: str = 'AAPL'):
        """
        Args:
            historical_data: DataFrame with at least 'Date' and a price column.
                             Accepts either 'price' or 'Close_{symbol}'.
            confidence_level: Confidence level for intervals (0.90–0.99).
            symbol: Stock symbol used to load the correct ML model.
        """
        self.symbol = symbol
        self.historical_data = historical_data.sort_values('Date').reset_index(drop=True)
        self.confidence_level = confidence_level

        # Resolve the close price column
        close_col = f'Close_{symbol}'
        if close_col in historical_data.columns:
            self.prices = historical_data[close_col].values
        elif 'price' in historical_data.columns:
            self.prices = historical_data['price'].values
        elif 'Close' in historical_data.columns:
            self.prices = historical_data['Close'].values
        else:
            raise ValueError("DataFrame must contain 'price', 'Close', or 'Close_{symbol}' column")

        self.dates = pd.to_datetime(historical_data['Date'])

        # Try to load trained ML models for this symbol
        self._models = _load_models(symbol)

        # Try to get live sentiment features for signal enrichment
        self._sentiment: Optional[Dict] = None
        if _SENTIMENT_AVAILABLE:
            try:
                self._sentiment = _get_sentiment_features(symbol)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Technical indicators (kept for confirmation + fallback)
    # ------------------------------------------------------------------
    def calculate_technical_indicators(self) -> Dict:
        """Calculate technical indicators for prediction"""
        prices = self.prices

        rsi = self._calculate_rsi(prices, period=14)
        ema12 = self._calculate_ema(prices, period=12)
        ema26 = self._calculate_ema(prices, period=26)
        macd = ema12 - ema26
        macd_signal = self._calculate_ema(macd, period=9)
        macd_hist = macd - macd_signal

        sma20 = self._calculate_sma(prices, period=20)
        sma50 = self._calculate_sma(prices, period=50)
        std20 = pd.Series(prices).rolling(window=20).std().values
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)

        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)

        return {
            'rsi': float(rsi[-1]) if len(rsi) > 0 else 50,
            'macd': float(macd[-1]) if len(macd) > 0 else 0,
            'macd_signal': float(macd_signal[-1]) if len(macd_signal) > 0 else 0,
            'macd_hist': float(macd_hist[-1]) if len(macd_hist) > 0 else 0,
            'bb_upper': float(bb_upper[-1]) if len(bb_upper) > 0 else prices[-1],
            'bb_lower': float(bb_lower[-1]) if len(bb_lower) > 0 else prices[-1],
            'bb_middle': float(sma20[-1]) if len(sma20) > 0 else prices[-1],
            'sma50': float(sma50[-1]) if not np.isnan(sma50[-1]) else prices[-1],
            'volatility': float(volatility),
        }

    # ------------------------------------------------------------------
    # ML-powered signal generation
    # ------------------------------------------------------------------
    def generate_signal(self, indicators: Dict) -> Tuple[str, float]:
        """
        Generate trading signal.  If ML models are available the signal is
        primarily driven by the ensemble return prediction and direction
        classifier, confirmed / filtered by classic technicals.

        Returns:
            (signal_type, signal_strength)
        """
        # --- ML prediction (primary) ---
        ml_signal, ml_confidence = self._ml_predict_signal(indicators)

        # --- Technical confirmation factors ---
        tech_signals = []
        tech_strengths = []

        # RSI
        rsi = indicators['rsi']
        if rsi < 30:
            tech_signals.append('BULLISH'); tech_strengths.append((30 - rsi) / 30)
        elif rsi > 70:
            tech_signals.append('BEARISH'); tech_strengths.append((rsi - 70) / 30)
        else:
            tech_signals.append('NEUTRAL'); tech_strengths.append(0.3)

        # MACD histogram
        if indicators['macd_hist'] > 0 and indicators['macd'] > indicators['macd_signal']:
            tech_signals.append('BULLISH'); tech_strengths.append(0.6)
        elif indicators['macd_hist'] < 0 and indicators['macd'] < indicators['macd_signal']:
            tech_signals.append('BEARISH'); tech_strengths.append(0.6)
        else:
            tech_signals.append('NEUTRAL'); tech_strengths.append(0.3)

        # Bollinger band position
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        current_price = self.prices[-1]
        bb_range = bb_upper - bb_lower
        bb_pos = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        if bb_pos > 0.7:
            tech_signals.append('BEARISH'); tech_strengths.append(bb_pos - 0.5)
        elif bb_pos < 0.3:
            tech_signals.append('BULLISH'); tech_strengths.append(0.5 - bb_pos)
        else:
            tech_signals.append('NEUTRAL'); tech_strengths.append(0.3)

        # Trend filter: price vs SMA-50  (critical for avoiding downtrend buys)
        sma50 = indicators.get('sma50', current_price)
        in_uptrend = current_price > sma50

        # --- Combine ML + technicals ---
        if ml_signal is not None:
            # ML is primary; technicals confirm
            tech_bullish = tech_signals.count('BULLISH')
            tech_bearish = tech_signals.count('BEARISH')

            if ml_signal == 'BULLISH':
                if not in_uptrend:
                    # Downtrend → degrade to NEUTRAL (avoid catching falling knives)
                    return 'NEUTRAL', 0.3
                confirmation = tech_bullish >= 1  # require 1/3 tech agreement
                if not confirmation:
                    return 'NEUTRAL', 0.35  # insufficient confirmation
                conf = ml_confidence
                return 'BULLISH', min(conf, 1.0)

            elif ml_signal == 'BEARISH':
                if in_uptrend:
                    # Strong uptrend → degrade bearish to NEUTRAL
                    return 'NEUTRAL', 0.3
                confirmation = tech_bearish >= 1  # require 1/3 tech agreement
                if not confirmation:
                    return 'NEUTRAL', 0.35  # insufficient confirmation
                conf = ml_confidence
                return 'BEARISH', min(conf, 1.0)

            else:
                return 'NEUTRAL', 0.35

        # --- Fallback: technicals only ---
        bullish_count = tech_signals.count('BULLISH')
        bearish_count = tech_signals.count('BEARISH')
        avg_str = float(np.mean(tech_strengths))

        # Require 2/3 technical indicators to agree for a signal
        if bullish_count >= 2 and in_uptrend:
            return 'BULLISH', min(avg_str * 0.7, 0.60)
        elif bearish_count >= 2:
            return 'BEARISH', min(avg_str * 0.7, 0.60)
        return 'NEUTRAL', 0.3

    # ------------------------------------------------------------------
    # Internal ML helpers
    # ------------------------------------------------------------------
    def _build_feature_row(self, indicators: Dict) -> Optional[np.ndarray]:
        """Build a single feature row matching the trained model's schema."""
        if self._models is None:
            return None

        feature_cols = self._models['report'].get('feature_cols', [])
        if not feature_cols:
            return None

        df = self.historical_data.copy()
        close_col = f'Close_{self.symbol}'
        high_col = f'High_{self.symbol}'
        low_col = f'Low_{self.symbol}'
        open_col = f'Open_{self.symbol}'
        vol_col = f'Volume_{self.symbol}'

        # If the DF already has indicators, use last row directly
        row = {}
        last = df.iloc[-1]

        # Try columns from df first, then from indicators dict
        for col in feature_cols:
            if col in df.columns:
                row[col] = float(last[col])
            else:
                row[col] = self._derive_feature(col, df, indicators)

        # Build array in correct order
        try:
            arr = np.array([[row[c] for c in feature_cols]], dtype=float)
            # Replace any NaN (e.g. from un-warmed rolling indicators) with 0.0
            arr = np.where(np.isnan(arr), 0.0, arr)
            return arr
        except (KeyError, TypeError):
            return None

    def _derive_feature(self, col: str, df: pd.DataFrame, indicators: Dict) -> float:
        """Compute a single feature value from available data and indicators."""
        prices = self.prices
        current = prices[-1]
        close_col = f'Close_{self.symbol}'
        high_col = f'High_{self.symbol}'
        low_col = f'Low_{self.symbol}'
        open_col = f'Open_{self.symbol}'
        vol_col = f'Volume_{self.symbol}'

        mapping = {
            'RSI_14': indicators.get('rsi', 50),
            'MACD_Histogram': indicators.get('macd_hist', 0),
            'ROC_12': float((current - prices[-13]) / prices[-13] * 100) if len(prices) >= 13 else 0,
            'Volatility_20': indicators.get('volatility', 0.02) * 100,
            'BB_Width': (indicators.get('bb_upper', current) - indicators.get('bb_lower', current))
                        / indicators.get('bb_middle', current) if indicators.get('bb_middle', current) else 0,
            'BB_Position': (current - indicators.get('bb_lower', current))
                           / max(indicators.get('bb_upper', current) - indicators.get('bb_lower', current), 1e-8),
            'ATR_Pct': 0.02,  # fallback
            'Price_SMA10_Ratio': current / indicators.get('bb_middle', current) if indicators.get('bb_middle') else 1.0,
            'Price_SMA20_Ratio': current / indicators.get('bb_middle', current) if indicators.get('bb_middle') else 1.0,
            'Price_SMA50_Ratio': current / indicators.get('sma50', current) if indicators.get('sma50') else 1.0,
            'Price_SMA200_Ratio': 1.0,  # fallback
            'EMA10_EMA20_Cross': 0.0,
        }

        if col in mapping:
            return float(mapping[col])

        # OHLCV ratios
        if col == 'Intraday_Range' and high_col in df.columns and low_col in df.columns:
            return float((df[high_col].iloc[-1] - df[low_col].iloc[-1]) / current)
        if col == 'Open_Close_Ratio' and open_col in df.columns:
            return float(df[open_col].iloc[-1] / current)
        if col == 'High_Close_Ratio' and high_col in df.columns:
            return float(df[high_col].iloc[-1] / current)
        if col == 'Low_Close_Ratio' and low_col in df.columns:
            return float(df[low_col].iloc[-1] / current)
        if col == 'Volume_Ratio' and vol_col in df.columns:
            vol_sma = df[vol_col].rolling(20).mean().iloc[-1]
            return float(df[vol_col].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0

        # Past returns
        if col == 'Return_1d' and len(prices) >= 2:
            return float((prices[-1] - prices[-2]) / prices[-2])
        if col == 'Return_5d' and len(prices) >= 6:
            return float((prices[-1] - prices[-6]) / prices[-6])
        if col == 'Return_10d' and len(prices) >= 11:
            return float((prices[-1] - prices[-11]) / prices[-11])

        return 0.0

    def _ml_predict_signal(self, indicators: Dict) -> Tuple[Optional[str], float]:
        """
        Use ML models to predict signal and confidence.
        Returns (signal_type, confidence) or (None, 0) if unavailable.
        """
        if self._models is None:
            return None, 0.0

        feature_row = self._build_feature_row(indicators)
        if feature_row is None:
            return None, 0.0

        scaler = self._models['scaler']
        lr = self._models['model_lr']
        rf = self._models['model_rf']
        clf = self._models['model_dir_clf']

        try:
            X_sc = scaler.transform(feature_row)
        except (ValueError, TypeError):
            return None, 0.0

        # Ensemble return prediction (use optimized weights from training)
        report = self._models.get('report', {})
        ew = report.get('ensemble_weights', {})
        lr_w = ew.get('lr', 0.5)
        rf_w = ew.get('rf', 0.5)

        pred_lr = lr.predict(X_sc)[0]
        pred_rf = rf.predict(X_sc)[0]
        pred_return_lr_rf = lr_w * pred_lr + rf_w * pred_rf

        # Direction classifier — use GB if available, else RF
        gb_clf = self._models.get('model_gb_clf')
        if gb_clf is not None:
            dir_proba = gb_clf.predict_proba(X_sc)[0]
        else:
            dir_proba = clf.predict_proba(X_sc)[0]
        up_proba_clf = dir_proba[1] if len(dir_proba) > 1 else 0.5

        # --- TFT prediction (blended in at 30% weight when available) ---
        tft_pred_return = 0.0
        tft_dir_prob = 0.5
        tft_weight = 0.0
        if _TFT_AVAILABLE:
            try:
                tft_result = _predict_tft(self.symbol, self.historical_data.tail(_TFT_SEQ_LEN + 20))
                if tft_result['backend'] != 'unavailable':
                    tft_pred_return = tft_result['predicted_return']
                    tft_dir_prob = tft_result['direction_prob']
                    tft_weight = 0.30
            except Exception:
                pass

        # --- Sentiment adjustment ---
        sentiment_bias = 0.0
        if self._sentiment is not None:
            s1d = self._sentiment.get('Sentiment_1d', 0.0) or 0.0
            momentum = self._sentiment.get('Sentiment_Momentum', 0.0) or 0.0
            # Scale: extreme sentiment (|score| > 0.5) biases return prediction by ±0.001
            sentiment_bias = float(s1d) * 0.002 + float(momentum) * 0.001

        # --- Combine: LR+RF ensemble (70%) + TFT (30%) + sentiment bias ---
        lr_rf_weight = 1.0 - tft_weight
        pred_return = (lr_rf_weight * pred_return_lr_rf +
                       tft_weight * tft_pred_return +
                       sentiment_bias)
        up_proba = (lr_rf_weight * up_proba_clf + tft_weight * tft_dir_prob)

        # Combine with calibrated thresholds
        if pred_return > 0.001 and up_proba > 0.52:
            signal = 'BULLISH'
            # Normalised confidence: 0.52 proba → low, 0.70 → high
            confidence = (up_proba - 0.5) * 2.0 * 0.7 + min(abs(pred_return) * 20, 0.3) * 0.3
        elif pred_return < -0.001 and up_proba < 0.48:
            signal = 'BEARISH'
            confidence = (0.5 - up_proba) * 2.0 * 0.7 + min(abs(pred_return) * 20, 0.3) * 0.3
        else:
            signal = 'NEUTRAL'
            confidence = 0.35

        return signal, min(max(confidence, 0.1), 1.0)

    def ml_predict_return(self, indicators: Dict = None) -> Tuple[float, float]:
        """
        Public helper returning (predicted_return, confidence).
        *predicted_return* is the expected next-day return as a fraction,
        e.g. 0.012 means +1.2 %.
        """
        if indicators is None:
            indicators = self.calculate_technical_indicators()

        if self._models is None:
            return 0.0, 0.3

        feature_row = self._build_feature_row(indicators)
        if feature_row is None:
            return 0.0, 0.3

        scaler = self._models['scaler']
        lr = self._models['model_lr']
        rf = self._models['model_rf']
        clf = self._models['model_dir_clf']

        try:
            X_sc = scaler.transform(feature_row)
        except (ValueError, TypeError):
            return 0.0, 0.3

        report = self._models.get('report', {})
        ew = report.get('ensemble_weights', {})
        lr_w = ew.get('lr', 0.5)
        rf_w = ew.get('rf', 0.5)
        pred_return = lr_w * lr.predict(X_sc)[0] + rf_w * rf.predict(X_sc)[0]

        gb_clf = self._models.get('model_gb_clf')
        if gb_clf is not None:
            dir_proba = gb_clf.predict_proba(X_sc)[0]
        else:
            dir_proba = clf.predict_proba(X_sc)[0]
        up_proba = dir_proba[1] if len(dir_proba) > 1 else 0.5

        confidence = max(up_proba, 1 - up_proba)
        return float(pred_return), float(confidence)

    # ------------------------------------------------------------------
    # Multi-day forecast  (now ML-powered)
    # ------------------------------------------------------------------
    
    def predict_multi_day(self, days_ahead: int = 5) -> Dict:
        """
        Generate multi-day price forecast with confidence intervals.

        Uses ML ensemble return prediction for day-1, then propagates
        with decay for subsequent days.
        """
        prices = self.prices
        current_price = prices[-1]

        # Technical indicators
        indicators = self.calculate_technical_indicators()
        signal_type, signal_strength = self.generate_signal(indicators)

        # Volatility
        recent_ret = np.diff(prices[-60:]) / prices[-60:-1] if len(prices) > 60 else np.diff(prices) / prices[:-1]
        volatility = float(np.std(recent_ret)) if len(recent_ret) > 0 else 0.01

        # ML predicted return for day-1
        ml_return, ml_conf = self.ml_predict_return(indicators)

        # 20-day trend (fallback component)
        recent = prices[-20:] if len(prices) >= 20 else prices
        trend = float((recent[-1] - recent[0]) / recent[0])

        forecasts = []
        z = 1.96 if self.confidence_level == 0.95 else (2.576 if self.confidence_level == 0.99 else 1.645)

        for day in range(1, days_ahead + 1):
            # Day-1 uses pure ML; further days blend ML with decaying trend
            if day == 1:
                day_return = ml_return
            else:
                decay = 0.5 ** (day - 1)  # rapid decay
                day_return = ml_return * decay + (trend / 20) * (1 - decay)

            forecast_price = current_price * (1 + day_return * day)
            ci_width = current_price * volatility * z * np.sqrt(day)

            confidence = ml_conf * max(0.4, 1.0 - 0.12 * (day - 1))
            confidence = max(0.25, min(confidence, 0.95))

            forecast_date = (self.dates.iloc[-1] + timedelta(days=day)).strftime('%Y-%m-%d')

            forecasts.append({
                'date': forecast_date,
                'day': day,
                'forecast_price': float(forecast_price),
                'lower_bound': float(max(forecast_price - ci_width, current_price * 0.85)),
                'upper_bound': float(forecast_price + ci_width),
                'confidence': float(confidence),
                'direction': 'UP' if forecast_price > current_price else ('DOWN' if forecast_price < current_price else 'FLAT'),
            })

        return {
            'current_price': float(current_price),
            'signal': signal_type,
            'signal_strength': float(signal_strength),
            'volatility': float(volatility),
            'trend': float(trend),
            'ml_predicted_return': float(ml_return),
            'indicators': indicators,
            'forecasts': forecasts,
            'model_available': self._models is not None,
            'generated_at': datetime.now().isoformat(),
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
    print("Prediction Visualization Engine - Task 5.2  (ML-powered)")
    print("=" * 80)

    # Try real stock data first; fall back to synthetic
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META']
    for sym in symbols:
        data_path = os.path.join(_BASE_DIR, 'data', f'{sym}_stock_data_with_indicators.csv')
        if not os.path.exists(data_path):
            continue
        df = pd.read_csv(data_path)
        if 'Date' not in df.columns:
            continue

        engine = PredictionEngine(df, symbol=sym)
        predictions = engine.predict_multi_day(days_ahead=5)

        print(f"\n{'='*60}")
        print(f"  {sym}  |  Model loaded: {predictions['model_available']}")
        print(f"{'='*60}")
        print(f"  Current Price : ${predictions['current_price']:.2f}")
        print(f"  Signal        : {predictions['signal']} ({predictions['signal_strength']:.2%})")
        print(f"  ML Return     : {predictions.get('ml_predicted_return', 0):+.4f}")
        print(f"  Volatility    : {predictions['volatility']:.2%}")
        print(f"  Trend (20d)   : {predictions['trend']:+.2%}")
        print()
        for fc in predictions['forecasts']:
            print(f"    Day {fc['day']} ({fc['date']}): "
                  f"${fc['forecast_price']:.2f}  "
                  f"[${fc['lower_bound']:.2f} – ${fc['upper_bound']:.2f}]  "
                  f"{fc['direction']}  conf={fc['confidence']:.1%}")
    print("\nDone.")
