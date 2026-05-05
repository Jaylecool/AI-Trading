"""
Transformer Predictor
---------------------
Trains and runs a Temporal Fusion Transformer (TFT) for stock return prediction.

Primary  : pytorch-forecasting TFT — state-of-the-art for tabular time-series.
Fallback : Lightweight custom nn.Transformer block when pytorch-forecasting is
           unavailable (requires torch only).

Both variants output:
  {
    'predicted_return': float,    # predicted next-day return as a decimal
    'direction_prob':  float,     # probability price goes UP (0–1)
    'confidence':      float,     # model-internal confidence (0–1)
    'lower_q':         float,     # 10th-percentile return (TFT only)
    'upper_q':         float,     # 90th-percentile return (TFT only)
    'backend':         str,       # 'tft' | 'transformer_lite' | 'unavailable'
  }

Model artefacts are saved to  trained_models/{SYMBOL}/tft/

Usage:
    python transformer_predictor.py          # trains all default symbols
    python transformer_predictor.py AAPL     # trains only AAPL
"""

import json
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, 'trained_models')

# Feature columns expected in the data (the 34-feature set including sentiment)
SEQUENCE_LENGTH = 30   # number of past timesteps fed to the model
PREDICTION_HORIZON = 1  # days ahead to predict

# The 29 technical features + 5 sentiment features
FEATURE_COLS = [
    'RSI_14', 'MACD_Histogram', 'ROC_12', 'Volatility_20',
    'Intraday_Range', 'Open_Close_Ratio', 'High_Close_Ratio', 'Low_Close_Ratio',
    'Volume_Ratio',
    'Price_SMA10_Ratio', 'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'Price_SMA200_Ratio',
    'EMA10_EMA20_Cross', 'BB_Width', 'BB_Position', 'ATR_Pct',
    'Return_1d', 'Return_5d', 'Return_10d', 'Return_20d',
    'SMA10_SMA20_Cross', 'SMA50_SMA200_Cross',
    'RSI_Change_5d', 'Volume_Change_5d',
    'Price_vs_20d_High', 'Price_vs_20d_Low',
    'Volatility_Ratio', 'MACD_Hist_Change', 'Pos_Days_5d',
    # Sentiment features (added when available)
    'Sentiment_1d', 'Sentiment_3d', 'Sentiment_7d',
    'News_Volume_7d', 'Sentiment_Momentum',
]

# Base features without sentiment (backward-compat)
FEATURE_COLS_BASE = FEATURE_COLS[:30]


# ===========================================================================
# Model cache
# ===========================================================================

_tft_cache: Dict[str, object] = {}


def _model_dir(symbol: str) -> str:
    d = os.path.join(_MODELS_DIR, symbol, 'tft')
    os.makedirs(d, exist_ok=True)
    return d


# ===========================================================================
# Data preparation helpers
# ===========================================================================

def _build_feature_matrix(df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from a stock dataframe that already contains feature columns.

    X shape: (N, SEQUENCE_LENGTH, n_features)
    y shape: (N,)   — next-day return (regression target)
    """
    close_col = f'Close_{symbol}'
    if close_col not in df.columns:
        close_col = 'Close'

    close = df[close_col]
    df = df.copy()
    df['target_return'] = (close.shift(-1) - close) / close

    # Select only available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 10:
        raise ValueError(f"Too few feature columns available ({len(available)})")

    df = df.dropna(subset=available + ['target_return']).reset_index(drop=True)

    feature_array = df[available].values.astype(np.float32)
    target_array = df['target_return'].values.astype(np.float32)

    n_features = len(available)
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(feature_array) - PREDICTION_HORIZON + 1):
        X.append(feature_array[i - SEQUENCE_LENGTH:i])
        y.append(target_array[i + PREDICTION_HORIZON - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _time_split_xy(X: np.ndarray, y: np.ndarray,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15):
    n = len(X)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    return (X[:t], y[:t]), (X[t:v], y[t:v]), (X[v:], y[v:])


# ===========================================================================
# Backend 1: Full TFT via pytorch-forecasting
# ===========================================================================

def _train_tft_full(symbol: str,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    n_features: int) -> Optional[object]:
    """
    Train the pytorch-forecasting TFT.
    Returns the trained Lightning trainer or None on failure.
    """
    try:
        import torch
        try:
            import lightning.pytorch as pl
        except ImportError:
            import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError:
        return None

    print(f"  [TFT-full] Training pytorch-forecasting TFT for {symbol} …")

    # Convert sequence data to flat TimeSeriesDataSet format
    def _to_flat_df(X: np.ndarray, y: np.ndarray, offset: int = 0) -> pd.DataFrame:
        rows = []
        for i, (seq, target) in enumerate(zip(X, y)):
            for t in range(len(seq)):
                row = {'time_idx': offset + i * len(seq) + t,
                       'group': symbol,
                       'target': float(target)}
                for fi, feat_val in enumerate(seq[t]):
                    row[f'feat_{fi}'] = float(feat_val)
                rows.append(row)
        return pd.DataFrame(rows)

    try:
        train_df = _to_flat_df(X_train, y_train, offset=0)
        val_df = _to_flat_df(X_val, y_val, offset=len(train_df))

        time_varying_cols = [f'feat_{i}' for i in range(n_features)]

        train_dataset = TimeSeriesDataSet(
            train_df,
            time_idx='time_idx',
            target='target',
            group_ids=['group'],
            min_encoder_length=SEQUENCE_LENGTH // 2,
            max_encoder_length=SEQUENCE_LENGTH,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_unknown_reals=time_varying_cols,
            add_relative_time_idx=True,
            add_target_scales=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df)

        train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=3e-3,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        )

        trainer = pl.Trainer(
            max_epochs=8,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=0.1,
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Save model
        mdir = _model_dir(symbol)
        tft_path = os.path.join(mdir, 'tft_model.ckpt')
        trainer.save_checkpoint(tft_path)
        print(f"  [TFT-full] Saved → {tft_path}")
        return tft

    except Exception as e:
        print(f"  [TFT-full] Training failed: {e}")
        return None


# ===========================================================================
# Backend 2: Lightweight custom Transformer (torch only)
# ===========================================================================

def _build_lite_transformer(n_features: int, d_model: int = 64,
                             nhead: int = 4, num_layers: int = 2):
    """Build a compact PyTorch Transformer encoder → regression head."""
    import torch
    import torch.nn as nn

    class LiteTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.regressor = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):               # x: (B, T, n_features)
            x = self.input_proj(x)          # (B, T, d_model)
            x = self.encoder(x)             # (B, T, d_model)
            x = x[:, -1, :]                 # last timestep → (B, d_model)
            return self.regressor(x).squeeze(-1)   # (B,)

    return LiteTransformer()


def _train_lite_transformer(symbol: str,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             n_features: int) -> Optional[object]:
    """Train the lightweight Transformer. Returns model or None on failure."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return None

    print(f"  [TFT-lite] Training lightweight Transformer for {symbol} …")

    try:
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_vl = torch.tensor(X_val, dtype=torch.float32)
        y_vl = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                                  batch_size=64, shuffle=True)

        model = _build_lite_transformer(n_features)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(20):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_vl)
                val_loss = criterion(val_pred, y_vl).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}  val_loss={val_loss:.6f}")

        if best_state:
            model.load_state_dict(best_state)

        # Save model
        mdir = _model_dir(symbol)
        model_path = os.path.join(mdir, 'lite_transformer.pt')
        torch.save({'state_dict': model.state_dict(), 'n_features': n_features},
                   model_path)
        print(f"  [TFT-lite] Saved → {model_path}  (val_loss={best_val_loss:.6f})")
        return model

    except Exception as e:
        print(f"  [TFT-lite] Training failed: {e}")
        return None


# ===========================================================================
# Public train interface
# ===========================================================================

def train_tft(symbol: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Train the best available Transformer for *symbol*.
    Tries pytorch-forecasting TFT first; falls back to LiteTransformer.

    Args:
        symbol: Stock ticker.
        df:     DataFrame with features already computed (from data_fetcher).
                If None, loads from disk.

    Returns:
        {'backend': str, 'val_loss': float, 'n_features': int, ...}
    """
    from data_fetcher import load_stock_data, fetch_stock_data

    if df is None:
        try:
            df = load_stock_data(symbol)
        except FileNotFoundError:
            df = fetch_stock_data(symbol, start_date='2023-01-01', save=True)

    # Build feature rows (same pipeline as model_trainer)
    from model_trainer import _prepare_dataset
    X_raw, y_ret, _ = _prepare_dataset(df, symbol)

    # Add sentiment columns if present in the df (they may not be yet)
    sentiment_cols = ['Sentiment_1d', 'Sentiment_3d', 'Sentiment_7d',
                      'News_Volume_7d', 'Sentiment_Momentum']
    for col in sentiment_cols:
        if col in df.columns:
            # align by index
            if col not in X_raw.columns:
                X_raw = X_raw.copy()
                X_raw[col] = df.loc[X_raw.index, col].values

    n_features = X_raw.shape[1]

    # Normalise
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    y_arr = y_ret.values.astype(np.float32)

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled) - PREDICTION_HORIZON + 1):
        X_seq.append(X_scaled[i - SEQUENCE_LENGTH:i])
        y_seq.append(y_arr[i + PREDICTION_HORIZON - 1])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    (X_tr, y_tr), (X_vl, y_vl), _ = _time_split_xy(X_seq, y_seq)
    print(f"\n[TFT] {symbol}: train={len(X_tr)}, val={len(X_vl)}, features={n_features}")

    # Save scaler for inference
    mdir = _model_dir(symbol)
    scaler_path = os.path.join(mdir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'feature_cols': list(X_raw.columns)}, f)

    meta = {'n_features': n_features, 'feature_cols': list(X_raw.columns)}

    # Try full TFT first
    model = _train_tft_full(symbol, X_tr, y_tr, X_vl, y_vl, n_features)
    if model is not None:
        meta['backend'] = 'tft'
        meta_path = os.path.join(mdir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump({**meta, 'trained_at': pd.Timestamp.now().isoformat()}, f)
        _tft_cache[symbol] = {'model': model, 'scaler': scaler, **meta}
        return meta

    # Fallback: LiteTransformer
    model = _train_lite_transformer(symbol, X_tr, y_tr, X_vl, y_vl, n_features)
    if model is not None:
        meta['backend'] = 'transformer_lite'
    else:
        meta['backend'] = 'unavailable'

    meta_path = os.path.join(mdir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump({**meta, 'trained_at': pd.Timestamp.now().isoformat()}, f)
    if model is not None:
        _tft_cache[symbol] = {'model': model, 'scaler': scaler, **meta}
    return meta


# ===========================================================================
# Public inference interface
# ===========================================================================

def _load_tft_artefacts(symbol: str) -> Optional[Dict]:
    """Load trained TFT artefacts for inference. Returns None if not trained."""
    if symbol in _tft_cache:
        return _tft_cache[symbol]

    mdir = _model_dir(symbol)
    meta_path = os.path.join(mdir, 'meta.json')
    scaler_path = os.path.join(mdir, 'scaler.pkl')

    if not os.path.exists(meta_path) or not os.path.exists(scaler_path):
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(scaler_path, 'rb') as f:
            sc_data = pickle.load(f)

        backend = meta.get('backend', 'unavailable')
        model = None

        if backend == 'tft':
            try:
                from pytorch_forecasting import TemporalFusionTransformer
                ckpt = os.path.join(mdir, 'tft_model.ckpt')
                if os.path.exists(ckpt):
                    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
            except Exception:
                pass

        elif backend == 'transformer_lite':
            try:
                import torch
                lite_path = os.path.join(mdir, 'lite_transformer.pt')
                if os.path.exists(lite_path):
                    saved = torch.load(lite_path, map_location='cpu')
                    n_feat = saved['n_features']
                    model = _build_lite_transformer(n_feat)
                    model.load_state_dict(saved['state_dict'])
                    model.eval()
            except Exception:
                pass

        if model is None:
            return None

        artefacts = {
            'model': model,
            'scaler': sc_data['scaler'],
            'feature_cols': sc_data['feature_cols'],
            'backend': backend,
            **meta,
        }
        _tft_cache[symbol] = artefacts
        return artefacts

    except Exception:
        return None


def _unavailable_result() -> Dict:
    return {
        'predicted_return': 0.0,
        'direction_prob': 0.5,
        'confidence': 0.0,
        'lower_q': -0.01,
        'upper_q': 0.01,
        'backend': 'unavailable',
    }


def predict_tft(symbol: str, recent_df: pd.DataFrame) -> Dict:
    """
    Generate a next-day return prediction using the trained Transformer.

    Args:
        symbol:    Stock ticker.
        recent_df: DataFrame containing AT LEAST the last SEQUENCE_LENGTH rows
                   with feature columns (from data_fetcher).

    Returns:
        Dict with predicted_return, direction_prob, confidence, backend, etc.
    """
    artefacts = _load_tft_artefacts(symbol)
    if artefacts is None:
        return _unavailable_result()

    backend = artefacts['backend']
    model = artefacts['model']
    scaler = artefacts['scaler']
    feature_cols = artefacts['feature_cols']

    try:
        available_cols = [c for c in feature_cols if c in recent_df.columns]
        if len(available_cols) < 10:
            return _unavailable_result()

        df_feat = recent_df[available_cols].dropna().tail(SEQUENCE_LENGTH)
        if len(df_feat) < SEQUENCE_LENGTH:
            return _unavailable_result()

        X = scaler.transform(df_feat.values).astype(np.float32)  # (T, n_feat)

        if backend == 'tft':
            # TFT inference
            try:
                import torch
                # Build a minimal prediction dataframe
                pred_df = pd.DataFrame(
                    X, columns=[f'feat_{i}' for i in range(X.shape[1])]
                )
                pred_df['time_idx'] = list(range(len(pred_df)))
                pred_df['group'] = symbol
                pred_df['target'] = 0.0
                raw_pred = model.predict(pred_df, mode='quantiles', return_x=False)
                q10, q50, q90 = float(raw_pred[0][0]), float(raw_pred[0][1]), float(raw_pred[0][2])
                direction_prob = float(np.clip(0.5 + q50 * 50, 0.0, 1.0))
                conf = float(1.0 - min(abs(q90 - q10) * 10, 1.0))
                return {
                    'predicted_return': round(q50, 5),
                    'direction_prob': round(direction_prob, 4),
                    'confidence': round(conf, 4),
                    'lower_q': round(q10, 5),
                    'upper_q': round(q90, 5),
                    'backend': 'tft',
                }
            except Exception:
                pass

        elif backend == 'transformer_lite':
            import torch
            model.eval()
            X_tensor = torch.tensor(X[np.newaxis], dtype=torch.float32)  # (1, T, F)
            with torch.no_grad():
                pred = model(X_tensor).item()
            direction_prob = float(np.clip(0.5 + pred * 50, 0.0, 1.0))
            conf = float(min(abs(pred) * 100, 1.0))
            return {
                'predicted_return': round(pred, 5),
                'direction_prob': round(direction_prob, 4),
                'confidence': round(conf, 4),
                'lower_q': round(pred - 0.005, 5),
                'upper_q': round(pred + 0.005, 5),
                'backend': 'transformer_lite',
            }

    except Exception as e:
        print(f"[TFT] predict_tft error for {symbol}: {e}")

    return _unavailable_result()


# ===========================================================================
# Standalone test / CLI
# ===========================================================================

if __name__ == '__main__':
    import sys
    from data_fetcher import DEFAULT_SYMBOLS, load_stock_data, fetch_stock_data

    symbols = sys.argv[1:] if len(sys.argv) > 1 else ['AAPL']

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  Training TFT for {sym}")
        print(f"{'='*60}")
        try:
            df = load_stock_data(sym)
        except FileNotFoundError:
            df = fetch_stock_data(sym, save=True)

        meta = train_tft(sym, df)
        print(f"\n  Result: backend={meta.get('backend')}, "
              f"features={meta.get('n_features')}")

        # Quick inference test
        result = predict_tft(sym, df.tail(SEQUENCE_LENGTH + 10))
        print(f"  Inference: predicted_return={result['predicted_return']:+.4f}  "
              f"direction_prob={result['direction_prob']:.2%}  "
              f"backend={result['backend']}")
