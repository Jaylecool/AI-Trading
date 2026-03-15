"""
Multi-Stock Data Fetcher with Technical Indicator Engineering
Downloads OHLCV data via yfinance and computes 22 technical indicators
matching the schema in data/AAPL_stock_data_with_indicators.csv

Supported indicators:
  SMA_10, SMA_20, SMA_50, SMA_200, EMA_10, EMA_20, EMA_50,
  RSI_14, MACD, MACD_Signal, MACD_Histogram, ROC_12,
  BB_Upper, BB_Lower, BB_Middle, ATR_14, Volatility_20
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional


DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META']
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def fetch_stock_data(
    symbol: str,
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for *symbol* and compute technical indicators.

    Returns a DataFrame with columns:
        Date, Close_{sym}, High_{sym}, Low_{sym}, Open_{sym}, Volume_{sym},
        SMA_10 … Volatility_20  (22 feature columns total)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'Date',
        'Open': f'Open_{symbol}',
        'High': f'High_{symbol}',
        'Low': f'Low_{symbol}',
        'Close': f'Close_{symbol}',
        'Volume': f'Volume_{symbol}',
    })

    # Keep only OHLCV + Date
    keep_cols = ['Date', f'Close_{symbol}', f'High_{symbol}', f'Low_{symbol}',
                 f'Open_{symbol}', f'Volume_{symbol}']
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    dates = pd.to_datetime(df['Date'])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    df['Date'] = dates
    df = df.sort_values('Date').reset_index(drop=True)

    close = df[f'Close_{symbol}']
    high = df[f'High_{symbol}']
    low = df[f'Low_{symbol}']

    # --- Moving Averages ---
    for period in [10, 20, 50, 200]:
        df[f'SMA_{period}'] = close.rolling(window=period).mean()
    for period in [10, 20, 50]:
        df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()

    # --- RSI (14) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # --- Rate of Change (12) ---
    df['ROC_12'] = close.pct_change(periods=12) * 100

    # --- Bollinger Bands (20, 2σ) ---
    df['BB_Middle'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std

    # --- ATR (14) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    # --- Volatility (20-day annualised std of returns) ---
    daily_ret = close.pct_change()
    df['Volatility_20'] = daily_ret.rolling(window=20).std() * np.sqrt(252) * 100

    # Drop warm-up NaN rows (SMA_200 needs ~200 rows)
    indicator_cols = [c for c in df.columns if c not in keep_cols]
    df = df.dropna(subset=indicator_cols).reset_index(drop=True)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f'{symbol}_stock_data_with_indicators.csv')
        df.to_csv(path, index=False)
        print(f"  ✓ Saved {len(df)} rows → {path}")

    return df


def fetch_all_stocks(
    symbols: Optional[List[str]] = None,
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
) -> dict:
    """Fetch and save data for multiple symbols. Returns {symbol: DataFrame}."""
    symbols = symbols or DEFAULT_SYMBOLS
    results = {}
    for sym in symbols:
        print(f"[DataFetcher] Downloading {sym} …")
        try:
            results[sym] = fetch_stock_data(sym, start_date, end_date, save=True)
        except Exception as e:
            print(f"  ✗ Error fetching {sym}: {e}")
    return results


def load_stock_data(symbol: str) -> pd.DataFrame:
    """Load previously saved indicator CSV for *symbol*."""
    path = os.path.join(DATA_DIR, f'{symbol}_stock_data_with_indicators.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file for {symbol}. Run fetch_stock_data('{symbol}') first."
        )
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("Multi-Stock Data Fetcher")
    print("=" * 70)
    fetch_all_stocks()
    print("\nDone.")
