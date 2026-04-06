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
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
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
    # Past 1/5/10/20 day returns (momentum features)
    df['Return_1d'] = close.pct_change(1)
    df['Return_5d'] = close.pct_change(5)
    df['Return_10d'] = close.pct_change(10)
    df['Return_20d'] = close.pct_change(20)

    # --- NEW: additional discriminative features ---
    # Momentum crossovers (relative difference between fast/slow MA)
    df['SMA10_SMA20_Cross'] = (df['SMA_10'] - df['SMA_20']) / df['SMA_20'].replace(0, np.nan)
    df['SMA50_SMA200_Cross'] = (df['SMA_50'] - df['SMA_200']) / df['SMA_200'].replace(0, np.nan)
    # RSI momentum (change in RSI over 5 days — momentum of momentum)
    df['RSI_Change_5d'] = df['RSI_14'].diff(5)
    # Volume acceleration
    df['Volume_Change_5d'] = df[vol_col].pct_change(5)
    # Mean reversion signals
    df['Price_vs_20d_High'] = close / close.rolling(20).max()
    df['Price_vs_20d_Low'] = close / close.rolling(20).min()
    # Volatility regime (current volatility vs its own 60-day mean)
    vol_60_mean = df['Volatility_20'].rolling(60).mean()
    df['Volatility_Ratio'] = df['Volatility_20'] / vol_60_mean.replace(0, np.nan)
    # MACD histogram momentum
    df['MACD_Hist_Change'] = df['MACD_Histogram'].diff(5)
    # Return consistency (fraction of positive days in last 5)
    df['Pos_Days_5d'] = close.diff().gt(0).rolling(5).mean()

    feature_cols = [
        # Technical indicators (absolute values still useful)
        'RSI_14', 'MACD_Histogram', 'ROC_12', 'Volatility_20',
        # Price-relative features
        'Intraday_Range', 'Open_Close_Ratio', 'High_Close_Ratio', 'Low_Close_Ratio',
        'Volume_Ratio',
        'Price_SMA10_Ratio', 'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'Price_SMA200_Ratio',
        'EMA10_EMA20_Cross', 'BB_Width', 'BB_Position', 'ATR_Pct',
        'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
        # NEW features
        'SMA10_SMA20_Cross', 'SMA50_SMA200_Cross',
        'RSI_Change_5d', 'Volume_Change_5d',
        'Price_vs_20d_High', 'Price_vs_20d_Low',
        'Volatility_Ratio', 'MACD_Hist_Change', 'Pos_Days_5d',
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
    # Use validation set to select best max_depth
    best_rf_depth = 12
    best_rf_val_acc = 0.0
    for depth in [6, 8, 10, 12]:
        rf_candidate = RandomForestRegressor(
            n_estimators=300, max_depth=depth, max_features='sqrt',
            min_samples_leaf=10, random_state=42, n_jobs=-1,
        )
        rf_candidate.fit(X_train_sc, y_train_r)
        val_pred = rf_candidate.predict(X_val_sc)
        val_dir_acc = float(np.mean((val_pred > 0).astype(int) == y_val_d.values))
        if val_dir_acc > best_rf_val_acc:
            best_rf_val_acc = val_dir_acc
            best_rf_depth = depth
    print(f"  RF best max_depth={best_rf_depth} (val dir acc={best_rf_val_acc:.2%})")

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=best_rf_depth, max_features='sqrt',
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train_r)

    # --- 3. RF Direction Classifier ---
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, max_features='sqrt',
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train_d)

    # --- 4. GradientBoosting Direction Classifier (strongest for tabular data) ---
    gb_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    gb_clf.fit(X_train_sc, y_train_d)

    # --- Optimise ensemble weights on VALIDATION set ---
    val_pred_lr = lr.predict(X_val_sc)
    val_pred_rf = rf.predict(X_val_sc)
    best_w, best_val_dir = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.1):
        ens_val = w * val_pred_lr + (1 - w) * val_pred_rf
        acc = float(np.mean((ens_val > 0).astype(int) == y_val_d.values))
        if acc > best_val_dir:
            best_val_dir = acc
            best_w = round(float(w), 2)
    lr_weight = best_w
    rf_weight = round(1.0 - best_w, 2)
    print(f"  Ensemble weights: LR={lr_weight}, RF={rf_weight} (val dir acc={best_val_dir:.2%})")

    # --- Feature importance (log top features, prune noise) ---
    importances = rf.feature_importances_
    feat_importance = sorted(zip(X.columns, importances), key=lambda x: -x[1])
    print(f"  Top-10 features: {[f'{n}={v:.3f}' for n, v in feat_importance[:10]]}")

    # --- Evaluate on TEST set ---
    pred_lr = lr.predict(X_test_sc)
    pred_rf = rf.predict(X_test_sc)
    pred_ens = lr_weight * pred_lr + rf_weight * pred_rf

    pred_dir = clf.predict(X_test_sc)
    dir_proba = clf.predict_proba(X_test_sc)
    pred_gb_dir = gb_clf.predict(X_test_sc)
    gb_dir_proba = gb_clf.predict_proba(X_test_sc)

    # Direction accuracy of the return regression (predicted sign vs actual sign)
    ens_dir_accuracy = float(np.mean(
        (pred_ens > 0).astype(int) == y_test_d.values
    ))

    metrics = {
        'symbol': symbol,
        'test_samples': int(len(X_test)),
        'prediction_target': 'daily_return',
        'ensemble_weights': {'lr': lr_weight, 'rf': rf_weight},
        'rf_best_max_depth': best_rf_depth,
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
            'precision': float(precision_score(y_test_d, pred_dir, zero_division=0)),
            'recall': float(recall_score(y_test_d, pred_dir, zero_division=0)),
            'f1_score': float(f1_score(y_test_d, pred_dir, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test_d, pred_dir).tolist(),
        },
        'gb_classifier': {
            'accuracy': float(accuracy_score(y_test_d, pred_gb_dir)),
            'precision': float(precision_score(y_test_d, pred_gb_dir, zero_division=0)),
            'recall': float(recall_score(y_test_d, pred_gb_dir, zero_division=0)),
            'f1_score': float(f1_score(y_test_d, pred_gb_dir, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test_d, pred_gb_dir).tolist(),
        },
        'ensemble_direction': {
            'accuracy': float(ens_dir_accuracy),
            'precision': float(precision_score(y_test_d, (pred_ens > 0).astype(int), zero_division=0)),
            'recall': float(recall_score(y_test_d, (pred_ens > 0).astype(int), zero_division=0)),
            'f1_score': float(f1_score(y_test_d, (pred_ens > 0).astype(int), zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test_d, (pred_ens > 0).astype(int)).tolist(),
        },
        'feature_cols': list(X.columns),
        'feature_importance': {n: round(float(v), 4) for n, v in feat_importance},
    }

    print(f"  Ensemble MAE(return)={metrics['ensemble']['mae_return']:.6f}  "
          f"DirAcc(ens)={ens_dir_accuracy:.2%}  "
          f"DirAcc(clf)={metrics['direction_classifier']['accuracy']:.2%}  "
          f"DirAcc(gb)={metrics['gb_classifier']['accuracy']:.2%}")

    # --- Save artefacts ---
    model_dir = os.path.join(MODELS_DIR, symbol)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, 'model_lr.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(model_dir, 'model_rf.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(model_dir, 'model_dir_clf.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, 'model_gb_clf.pkl'), 'wb') as f:
        pickle.dump(gb_clf, f)
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
