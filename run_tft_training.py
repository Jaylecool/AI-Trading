"""
One-shot TFT training for all symbols.
Run: .venv\Scripts\python.exe run_tft_training.py
"""
import sys
import time
from data_fetcher import fetch_stock_data
from transformer_predictor import train_tft

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

results = {}
for sym in SYMBOLS:
    print(f"\n{'='*55}")
    print(f"  Training TFT: {sym}")
    print(f"{'='*55}")
    t0 = time.time()
    try:
        df = fetch_stock_data(sym, start_date='2023-01-01', save=True)
        meta = train_tft(sym, df)
        elapsed = time.time() - t0
        results[sym] = {'status': 'OK', 'backend': meta.get('backend'), 'elapsed': f'{elapsed:.1f}s'}
        print(f"  [{sym}] Done — backend={meta.get('backend')}  ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        results[sym] = {'status': 'ERROR', 'error': str(e), 'elapsed': f'{elapsed:.1f}s'}
        print(f"  [{sym}] FAILED: {e}", file=sys.stderr)

print(f"\n{'='*55}")
print("  TRAINING SUMMARY")
print(f"{'='*55}")
for sym, r in results.items():
    status = r['status']
    backend = r.get('backend', '')
    elapsed = r['elapsed']
    err = r.get('error', '')
    if status == 'OK':
        print(f"  {sym:<6}  ✓  {backend:<20}  {elapsed}")
    else:
        print(f"  {sym:<6}  ✗  {err[:40]}")
print()
