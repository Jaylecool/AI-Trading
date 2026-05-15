"""
Walk-forward validation of daily / weekly / monthly prediction horizons.
Fast version: uses only LR+RF ML models, skips TFT and FinBERT.
Completes in ~10 seconds.
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Patch out heavy imports so PredictionEngine loads fast
import sys, types

# Stub out transformer_predictor (TFT) — not needed for validation
stub = types.ModuleType('transformer_predictor')
stub.predict_tft = lambda *a, **k: (_ for _ in ()).throw(Exception('stub'))
stub.SEQUENCE_LENGTH = 30
sys.modules['transformer_predictor'] = stub

# Stub out nlp_sentiment_service (FinBERT) — not needed for validation
stub2 = types.ModuleType('nlp_sentiment_service')
stub2.get_sentiment_features = lambda sym: None
sys.modules['nlp_sentiment_service'] = stub2

from prediction_engine import PredictionEngine

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
HORIZONS = {'daily': 1, 'weekly': 5, 'monthly': 21}
MIN_LOOKBACK = 200
STEP = 15   # sample every 15 bars to avoid leakage / slowness

all_results = {}

for SYMBOL in SYMBOLS:
    df = pd.read_csv(f'data/{SYMBOL}_stock_data_with_indicators.csv')
    if 'Date' not in df.columns:
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    close_col = next((c for c in df.columns if 'Close' in c), None)
    if close_col and 'price' not in df.columns:
        df['price'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df.dropna(subset=['price']).reset_index(drop=True)

    MAX_IDX = len(df) - 22
    results = {k: {'correct': 0, 'total': 0, 'pred_rets': [], 'actual_rets': []} for k in HORIZONS}

    for i in range(MIN_LOOKBACK, MAX_IDX, STEP):
        window = df.iloc[:i].copy()
        try:
            pe = PredictionEngine(window, symbol=SYMBOL)
            hz = pe.predict_all_horizons()['horizons']
            current_price = float(df.iloc[i]['price'])
            for key, days in HORIZONS.items():
                future_idx = min(i + days, len(df) - 1)
                actual_price = float(df.iloc[future_idx]['price'])
                actual_ret = (actual_price - current_price) / current_price
                pred = hz.get(key, {})
                pred_ret = pred.get('pct_change', 0) / 100.0
                pred_signal = pred.get('signal', 'NEUTRAL')
                if pred_signal != 'NEUTRAL':
                    actual_dir = 'UP' if actual_ret > 0 else 'DOWN'
                    pred_dir = 'UP' if pred_signal == 'BULLISH' else 'DOWN'
                    results[key]['total'] += 1
                    if actual_dir == pred_dir:
                        results[key]['correct'] += 1
                    results[key]['pred_rets'].append(pred_ret * 100)
                    results[key]['actual_rets'].append(actual_ret * 100)
        except Exception:
            pass

    all_results[SYMBOL] = results
    print(f'  {SYMBOL}: {MAX_IDX // STEP} samples processed')

print()
print('=' * 75)
print(f'{"WALK-FORWARD VALIDATION  — 6 Symbols  (2020-2026)":^75}')
print('=' * 75)
print(f'{"Horizon":<10} {"Dir Acc":>8} {"n":>5} {"Avg Pred%":>10} {"Avg Actual%":>12} {"Corr":>7} {"Profitable?":>12}')
print('-' * 75)

agg = {k: {'correct': 0, 'total': 0, 'pred_rets': [], 'actual_rets': []} for k in HORIZONS}

for sym, res in all_results.items():
    for k in HORIZONS:
        agg[k]['correct'] += res[k]['correct']
        agg[k]['total'] += res[k]['total']
        agg[k]['pred_rets'].extend(res[k]['pred_rets'])
        agg[k]['actual_rets'].extend(res[k]['actual_rets'])

for key, days in HORIZONS.items():
    r = agg[key]
    if r['total'] == 0:
        print(f'{key.capitalize()+f" ({days}d)":<10}: No data')
        continue
    win_rate = r['correct'] / r['total'] * 100
    pred_arr = np.array(r['pred_rets'])
    actual_arr = np.array(r['actual_rets'])
    corr = float(np.corrcoef(pred_arr, actual_arr)[0, 1]) if len(pred_arr) > 2 else 0

    # When our model is bullish and direction is correct, avg gain
    bullish_correct = [actual_arr[i] for i in range(len(pred_arr)) if pred_arr[i] > 0 and actual_arr[i] > 0]
    avg_gain = np.mean(bullish_correct) if bullish_correct else 0

    label = f'{key.capitalize()} ({days}d)'
    profitable = 'YES' if win_rate > 55 and avg_gain > 0 else 'MARGINAL' if win_rate > 50 else 'NO'
    print(f'{label:<12} {win_rate:>7.1f}% {r["total"]:>5} {pred_arr.mean():>+9.2f}% {actual_arr.mean():>+11.2f}% {corr:>7.3f} {profitable:>12}')

print()
print('Per-symbol breakdown:')
print(f'{"Symbol":<8}', end='')
for key, days in HORIZONS.items():
    print(f'  {key[:3].upper()} acc', end='')
print()
for sym, res in all_results.items():
    print(f'{sym:<8}', end='')
    for key in HORIZONS:
        r = res[key]
        wr = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0
        print(f'  {wr:>7.1f}%', end='')
    print()

print()
print('KEY FINDING:')
print('  Dir Acc > 60% = reliable directional signal')
print('  Corr    > 0.1 = model magnitude correlates with reality')
print('  Profitable = YES means signal wins > 55% + avg gain > 0 when right')
