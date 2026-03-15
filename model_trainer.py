"""
Per-Stock ML Model Trainer
Trains an ensemble of Linear Regression + Random Forest for price prediction
and a Random Forest classifier for direction (Up/Down) prediction.

Models are saved to  trained_models/{SYMBOL}/
    model_lr.pkl          – Linear Regression (price)
    model_rf.pkl          – Random Forest Regressor (price)
    model_dir_clf.pkl     – Random Forest Classifier (direction)
    scaler.pkl            – StandardScaler fitted on training features
    training_report.json  – Metrics on test set

Usage:
    python model_trainer.py          # trains all 6 default symbols
    python model_trainer.py AAPL     # trains only AAPL
"""

import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.preprocessing import StandardScaler

from data_fetcher import DEFAULT_SYMBOLS, DATA_DIR, load_stock_data, fetch_stock_data

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')

# ---------------------------------------------------------------------------
# Feature / target helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_10', 'EMA_20', 'EMA_50',
    'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'ROC_12',
    'BB_Upper', 'BB_Lower', 'BB_Middle',
    'ATR_14', 'Volatility_20',
]


def _prepare_dataset(df: pd.DataFrame, symbol: str) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series
]:
    """
    Return (features, target_return, target_direction).

    Targets:
      - target_return  = (next_close - close) / close   (percentage daily return)
      - target_dir     = 1 if next_close > close else 0

    Features are normalised *relative to current price* so the model
    is regime-independent (works whether the stock is $50 or $500).
    """
    close_col = f'Close_{symbol}'
    high_col = f'High_{symbol}'
    low_col = f'Low_{symbol}'
    open_col = f'Open_{symbol}'
    vol_col = f'Volume_{symbol}'

    df = df.copy()
    close = df[close_col]

    # --- targets ---
    next_close = close.shift(-1)
    df['Target_Return'] = (next_close - close) / close
    df['Target_Dir'] = (next_close > close).astype(int)

    # --- price-relative features (regime-independent) ---
    df['Intraday_Range'] = (df[high_col] - df[low_col]) / close
    df['Open_Close_Ratio'] = df[open_col] / close
    df['High_Close_Ratio'] = df[high_col] / close
    df['Low_Close_Ratio'] = df[low_col] / close
    df['Volume_SMA20'] = df[vol_col].rolling(20).mean()
    df['Volume_Ratio'] = df[vol_col] / df['Volume_SMA20'].replace(0, np.nan)
    df['Price_SMA10_Ratio'] = close / df['SMA_10']
    df['Price_SMA20_Ratio'] = close / df['SMA_20']
    df['Price_SMA50_Ratio'] = close / df['SMA_50']
    df['Price_SMA200_Ratio'] = close / df['SMA_200']
    df['EMA10_EMA20_Cross'] = (df['EMA_10'] - df['EMA_20']) / close
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['ATR_Pct'] = df['ATR_14'] / close
    # Past 1/5/10 day returns (momentum features)
    df['Return_1d'] = close.pct_change(1)
    df['Return_5d'] = close.pct_change(5)
    df['Return_10d'] = close.pct_change(10)

    feature_cols = [
        # Technical indicators (absolute values still useful)
        'RSI_14', 'MACD_Histogram', 'ROC_12', 'Volatility_20',
        # Price-relative features
        'Intraday_Range', 'Open_Close_Ratio', 'High_Close_Ratio', 'Low_Close_Ratio',
        'Volume_Ratio',
        'Price_SMA10_Ratio', 'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'Price_SMA200_Ratio',
        'EMA10_EMA20_Cross', 'BB_Width', 'BB_Position', 'ATR_Pct',
        'Return_1d', 'Return_5d', 'Return_10d',
    ]

    df = df.dropna(subset=feature_cols + ['Target_Return'])
    X = df[feature_cols]
    y_return = df['Target_Return']
    y_dir = df['Target_Dir']
    return X, y_return, y_dir


def _time_split(X, y_ret, y_dir, train_ratio=0.7, val_ratio=0.15):
    """Chronological split into train / val / test."""
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train_r, y_train_d = X.iloc[:train_end], y_ret.iloc[:train_end], y_dir.iloc[:train_end]
    X_val, y_val_r, y_val_d = X.iloc[train_end:val_end], y_ret.iloc[train_end:val_end], y_dir.iloc[train_end:val_end]
    X_test, y_test_r, y_test_d = X.iloc[val_end:], y_ret.iloc[val_end:], y_dir.iloc[val_end:]
    return (X_train, y_train_r, y_train_d,
            X_val, y_val_r, y_val_d,
            X_test, y_test_r, y_test_d)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models_for_symbol(symbol: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Train LR + RF regressors and RF direction classifier for *symbol*.
    Saves artefacts to trained_models/{symbol}/.
    Returns a metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"  Training models for {symbol}")
    print(f"{'='*60}")

    # Load data if not provided
    if df is None:
        try:
            df = load_stock_data(symbol)
        except FileNotFoundError:
            print(f"  Data not found locally – downloading via yfinance …")
            df = fetch_stock_data(symbol, save=True)

    X, y_ret, y_dir = _prepare_dataset(df, symbol)
    splits = _time_split(X, y_ret, y_dir)
    X_train, y_train_r, y_train_d = splits[0], splits[1], splits[2]
    X_val, y_val_r, y_val_d = splits[3], splits[4], splits[5]
    X_test, y_test_r, y_test_d = splits[6], splits[7], splits[8]

    print(f"  Samples → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # --- 1. Linear Regression (predicts daily return) ---
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train_r)

    # --- 2. Random Forest Regressor (predicts daily return) ---
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, max_features='sqrt',
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train_r)

    # --- 3. Direction Classifier ---
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, max_features='sqrt',
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train_d)

    # --- Evaluate on TEST set ---
    pred_lr = lr.predict(X_test_sc)
    pred_rf = rf.predict(X_test_sc)
    pred_ens = 0.6 * pred_lr + 0.4 * pred_rf  # ensemble return prediction

    pred_dir = clf.predict(X_test_sc)
    dir_proba = clf.predict_proba(X_test_sc)

    # Direction accuracy of the return regression (predicted sign vs actual sign)
    ens_dir_accuracy = float(np.mean(
        (pred_ens > 0).astype(int) == y_test_d.values
    ))

    metrics = {
        'symbol': symbol,
        'test_samples': int(len(X_test)),
        'prediction_target': 'daily_return',
        'lr': {
            'mae_return': float(mean_absolute_error(y_test_r, pred_lr)),
            'direction_accuracy': float(np.mean((pred_lr > 0).astype(int) == y_test_d.values)),
        },
        'rf': {
            'mae_return': float(mean_absolute_error(y_test_r, pred_rf)),
            'direction_accuracy': float(np.mean((pred_rf > 0).astype(int) == y_test_d.values)),
        },
        'ensemble': {
            'mae_return': float(mean_absolute_error(y_test_r, pred_ens)),
            'direction_accuracy': float(ens_dir_accuracy),
        },
        'direction_classifier': {
            'accuracy': float(accuracy_score(y_test_d, pred_dir)),
        },
        'feature_cols': list(X.columns),
    }

    print(f"  Ensemble MAE(return)={metrics['ensemble']['mae_return']:.6f}  "
          f"DirAcc(ens)={ens_dir_accuracy:.2%}  "
          f"DirAcc(clf)={metrics['direction_classifier']['accuracy']:.2%}")

    # --- Save artefacts ---
    model_dir = os.path.join(MODELS_DIR, symbol)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'model_lr.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(model_dir, 'model_rf.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(model_dir, 'model_dir_clf.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(model_dir, 'training_report.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✓ Models saved → {model_dir}")
    return metrics


def train_all_stocks(symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Train and save models for every symbol. Returns {symbol: metrics}."""
    symbols = symbols or DEFAULT_SYMBOLS
    all_metrics = {}
    for sym in symbols:
        try:
            all_metrics[sym] = train_models_for_symbol(sym)
        except Exception as e:
            print(f"  ✗ Failed for {sym}: {e}")
    return all_metrics


def load_trained_models(symbol: str) -> Dict:
    """
    Load saved model artefacts for *symbol*.
    Returns dict with keys: 'lr', 'rf', 'clf', 'scaler', 'feature_cols'.
    """
    model_dir = os.path.join(MODELS_DIR, symbol)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No trained models for {symbol}. Run train_models_for_symbol('{symbol}') first.")

    with open(os.path.join(model_dir, 'model_lr.pkl'), 'rb') as f:
        lr = pickle.load(f)
    with open(os.path.join(model_dir, 'model_rf.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(model_dir, 'model_dir_clf.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'training_report.json'), 'r') as f:
        report = json.load(f)

    return {
        'lr': lr,
        'rf': rf,
        'clf': clf,
        'scaler': scaler,
        'feature_cols': report.get('feature_cols', FEATURE_COLS),
        'report': report,
    }


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    print("=" * 70)
    print("Per-Stock Model Trainer")
    print("=" * 70)
    results = train_all_stocks(targets)

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for sym, m in results.items():
        ens = m['ensemble']
        d = m['direction_classifier']
        print(f"  {sym:6s}  MAE(ret)={ens['mae_return']:.6f}  "
              f"DirAcc(ens)={ens['direction_accuracy']:.2%}  "
              f"DirAcc(clf)={d['accuracy']:.2%}")
    print("Done.")
