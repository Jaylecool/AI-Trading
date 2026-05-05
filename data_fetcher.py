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

# NLP / Sentiment enrichment (optional — skipped if service is unavailable)
try:
    from nlp_sentiment_service import get_sentiment_features as _get_sentiment_features
    _SENTIMENT_AVAILABLE = True
except Exception:
    _SENTIMENT_AVAILABLE = False


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

    # --- Sentiment Enrichment (NLP) ---
    df = enrich_with_sentiment(df, symbol)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f'{symbol}_stock_data_with_indicators.csv')
        df.to_csv(path, index=False)
        print(f"  ✓ Saved {len(df)} rows → {path}")

    return df


def enrich_with_sentiment(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Append five sentiment-based feature columns to *df* using cached news data.

    Columns added (all floats, forward-filled where gaps exist):
      Sentiment_1d       — 1-day recency-weighted compound sentiment score (-1..+1)
      Sentiment_3d       — 3-day score
      Sentiment_7d       — 7-day score
      News_Volume_7d     — count of articles in the last 7 days
      Sentiment_Momentum — Sentiment_1d minus Sentiment_7d (short vs long shift)

    When news data is unavailable the columns default to 0.0 (neutral).
    This function is intentionally fast: it uses the pre-cached news JSON and
    never makes network calls at inference time.
    """
    SENTIMENT_COLS = ['Sentiment_1d', 'Sentiment_3d', 'Sentiment_7d',
                      'News_Volume_7d', 'Sentiment_Momentum']

    if not _SENTIMENT_AVAILABLE:
        for col in SENTIMENT_COLS:
            df[col] = 0.0
        return df

    try:
        # Refresh news cache (incremental; skips if already fresh)
        from news_data_fetcher import refresh_news, load_cached_news, get_recent_articles
        from nlp_sentiment_service import score_news_batch, score_articles_inplace
        import math

        # Only fetch if we're enriching a reasonably recent dataset
        refresh_news(symbol)

        # We need per-row sentiment based on the date of each row.
        # For historical rows: use articles published up to (and including) that date.
        # We compute this efficiently by grouping articles by date window.
        all_articles = load_cached_news(symbol)
        # Score all unscored articles once
        from nlp_sentiment_service import score_articles_inplace
        all_articles = score_articles_inplace(symbol, all_articles)

        # Build {date_str -> compound_score} for each article
        # Then for each row, aggregate scores for the 1/3/7 day windows.
        # For efficiency, only compute per-date when df is large; otherwise
        # use the current-time windows (acceptable for backfill).
        n_rows = len(df)

        if n_rows <= 500:  # Small dataset — use current-time rolling windows
            from nlp_sentiment_service import get_sentiment_features
            features = get_sentiment_features(symbol)
            for col in SENTIMENT_COLS:
                df[col] = features.get(col, 0.0)
        else:
            # Large historical dataset — compute per-row windows
            # This is intentionally approximate: for historical rows we use
            # articles available AT THAT TIME (published <= row_date).
            dates = pd.to_datetime(df['Date'])

            s1d_arr = np.zeros(n_rows, dtype=float)
            s3d_arr = np.zeros(n_rows, dtype=float)
            s7d_arr = np.zeros(n_rows, dtype=float)
            vol7d_arr = np.zeros(n_rows, dtype=float)

            # Pre-parse article dates (only once)
            parsed_articles = []
            for art in all_articles:
                pub_str = art.get('published_at', '')
                try:
                    pub_str_clean = pub_str.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(pub_str_clean)
                    if dt.tzinfo is not None:
                        from datetime import timezone
                        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    parsed_articles.append((dt, art.get('sentiment_score', 0.0) or 0.0))
                except (ValueError, TypeError):
                    continue

            for i, row_date in enumerate(dates):
                row_dt = row_date.to_pydatetime().replace(tzinfo=None)
                scores_1d, w_1d = [], []
                scores_3d, w_3d = [], []
                scores_7d, w_7d = [], []

                for art_dt, compound in parsed_articles:
                    if art_dt > row_dt:
                        continue  # future news — skip
                    age_hours = (row_dt - art_dt).total_seconds() / 3600
                    if age_hours <= 24:
                        w = math.exp(-age_hours / 24)
                        scores_1d.append(compound * w); w_1d.append(w)
                    if age_hours <= 72:
                        w = math.exp(-age_hours / 72)
                        scores_3d.append(compound * w); w_3d.append(w)
                    if age_hours <= 168:
                        w = math.exp(-age_hours / 168)
                        scores_7d.append(compound * w); w_7d.append(w)
                        vol7d_arr[i] += 1

                s1d_arr[i] = sum(scores_1d) / sum(w_1d) if w_1d else 0.0
                s3d_arr[i] = sum(scores_3d) / sum(w_3d) if w_3d else 0.0
                s7d_arr[i] = sum(scores_7d) / sum(w_7d) if w_7d else 0.0

            df['Sentiment_1d'] = np.round(s1d_arr, 4)
            df['Sentiment_3d'] = np.round(s3d_arr, 4)
            df['Sentiment_7d'] = np.round(s7d_arr, 4)
            df['News_Volume_7d'] = vol7d_arr.astype(float)
            df['Sentiment_Momentum'] = np.round(s1d_arr - s7d_arr, 4)

        print(f"  ✓ Sentiment enrichment complete for {symbol} ")
    except Exception as e:
        print(f"  [sentiment] Enrichment skipped for {symbol}: {e}")
        for col in SENTIMENT_COLS:
            if col not in df.columns:
                df[col] = 0.0

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


def refresh_stock_data(symbol: str) -> pd.DataFrame:
    """
    Incrementally update an existing data file with new rows since the last
    saved date, then recompute all indicators on the full combined dataset.

    If no file exists yet, falls back to a full fetch from 2020-01-01.
    """
    path = os.path.join(DATA_DIR, f'{symbol}_stock_data_with_indicators.csv')

    if not os.path.exists(path):
        print(f"  [refresh] No existing file for {symbol} — running full fetch.")
        return fetch_stock_data(symbol, start_date='2020-01-01', save=True)

    existing = pd.read_csv(path)
    existing['Date'] = pd.to_datetime(existing['Date'])
    last_date = existing['Date'].max()
    today = datetime.now().date()

    if last_date.date() >= today:
        print(f"  [refresh] {symbol} already up to date ({last_date.date()}).")
        return existing

    # Fetch only the missing window (overlap by 1 day to avoid gaps)
    fetch_from = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    fetch_to = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"  [refresh] {symbol}: fetching {fetch_from} → {today} …")

    ticker = yf.Ticker(symbol)
    new_raw = ticker.history(start=fetch_from, end=fetch_to, auto_adjust=True)

    if new_raw.empty:
        print(f"  [refresh] {symbol}: no new data available yet.")
        return existing

    new_raw = new_raw.reset_index()
    new_raw = new_raw.rename(columns={
        'Date': 'Date',
        'Open': f'Open_{symbol}',
        'High': f'High_{symbol}',
        'Low': f'Low_{symbol}',
        'Close': f'Close_{symbol}',
        'Volume': f'Volume_{symbol}',
    })
    keep_cols = ['Date', f'Close_{symbol}', f'High_{symbol}', f'Low_{symbol}',
                 f'Open_{symbol}', f'Volume_{symbol}']
    new_raw = new_raw[[c for c in keep_cols if c in new_raw.columns]].copy()
    dates = pd.to_datetime(new_raw['Date'])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    new_raw['Date'] = dates

    # Merge existing OHLCV with new rows, drop indicator columns first
    ohlcv_cols = ['Date', f'Close_{symbol}', f'High_{symbol}',
                  f'Low_{symbol}', f'Open_{symbol}', f'Volume_{symbol}']
    existing_ohlcv = existing[[c for c in ohlcv_cols if c in existing.columns]].copy()
    combined = pd.concat([existing_ohlcv, new_raw], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Recompute all indicators on the full combined dataset
    close = combined[f'Close_{symbol}']
    high = combined[f'High_{symbol}']
    low = combined[f'Low_{symbol}']

    for period in [10, 20, 50, 200]:
        combined[f'SMA_{period}'] = close.rolling(window=period).mean()
    for period in [10, 20, 50]:
        combined[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    combined['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    combined['MACD'] = ema12 - ema26
    combined['MACD_Signal'] = combined['MACD'].ewm(span=9, adjust=False).mean()
    combined['MACD_Histogram'] = combined['MACD'] - combined['MACD_Signal']

    combined['ROC_12'] = close.pct_change(periods=12) * 100

    combined['BB_Middle'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    combined['BB_Upper'] = combined['BB_Middle'] + 2 * bb_std
    combined['BB_Lower'] = combined['BB_Middle'] - 2 * bb_std

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    combined['ATR_14'] = tr.rolling(window=14).mean()

    daily_ret = close.pct_change()
    combined['Volatility_20'] = daily_ret.rolling(window=20).std() * np.sqrt(252) * 100

    indicator_cols = [c for c in combined.columns if c not in ohlcv_cols]
    combined = combined.dropna(subset=indicator_cols).reset_index(drop=True)

    combined.to_csv(path, index=False)
    new_rows = len(combined) - len(existing)
    print(f"  ✓ {symbol}: +{new_rows} new rows → {len(combined)} total (up to {combined['Date'].max().date()})")
    return combined


def refresh_all_stocks(symbols: Optional[List[str]] = None) -> dict:
    """Refresh data for all symbols. Returns {symbol: DataFrame}."""
    symbols = symbols or DEFAULT_SYMBOLS
    results = {}
    print(f"[DataRefresh] Refreshing {len(symbols)} stocks …")
    for sym in symbols:
        try:
            results[sym] = refresh_stock_data(sym)
        except Exception as e:
            print(f"  ✗ Error refreshing {sym}: {e}")
    print("[DataRefresh] Done.")
    return results


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    print("=" * 70)
    print("Multi-Stock Data Fetcher")
    print("=" * 70)
    if '--refresh' in sys.argv:
        refresh_all_stocks()
    else:
        fetch_all_stocks()
    print("\nDone.")
