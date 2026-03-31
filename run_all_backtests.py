"""Run backtests for all stocks and print a summary profit table."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest_runner import BacktestRunner
from strategy_configurations import STRATEGIES

SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
INITIAL_CAPITAL = 100_000.0

all_results = {}

for symbol in SYMBOLS:
    data_path = f'data/{symbol}_stock_data_with_indicators.csv'
    if not os.path.exists(data_path):
        print(f"[SKIP] No data for {symbol}")
        continue
    
    runner = BacktestRunner(symbol=symbol, data_filepath=data_path)
    try:
        runner.load_data()
        runner.run_all_strategies(initial_capital=INITIAL_CAPITAL)
        all_results[symbol] = runner.results
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")

# ---- Print summary table ----
print("\n" + "=" * 120)
print("BACKTEST RESULTS — PERCENTAGE PROFIT BY STOCK & STRATEGY")
print("=" * 120)

strategies = list(STRATEGIES.keys())

# Header
header = f"{'Symbol':<8}"
for s in strategies:
    header += f" | {'ROI %':>10} {'Win%':>7} {'Trades':>7} {'Sharpe':>7} {'MaxDD%':>7}  "
print(f"\n{'':8}", end="")
for s in strategies:
    label = f"--- {s} ---"
    print(f" | {label:^42}", end="")
print()

header = f"{'Symbol':<8}"
for s in strategies:
    header += f" | {'ROI%':>8}  {'Win%':>6}  {'#Trd':>5}  {'Shrp':>6}  {'MxDD%':>6} "
print(header)
print("-" * (8 + len(strategies) * 45))

for symbol in SYMBOLS:
    if symbol not in all_results:
        continue
    row = f"{symbol:<8}"
    for s in strategies:
        r = all_results[symbol].get(s)
        if r:
            row += f" | {r.roi*100:>+7.2f}%  {r.win_rate*100:>5.1f}%  {r.total_trades:>5d}  {r.sharpe_ratio:>6.2f}  {r.max_drawdown*100:>5.2f}% "
        else:
            row += f" | {'N/A':>8}  {'N/A':>6}  {'N/A':>5}  {'N/A':>6}  {'N/A':>6} "
    print(row)

print("-" * (8 + len(strategies) * 45))

# Averages
print(f"\n{'AVERAGE':<8}", end="")
for s in strategies:
    rois = [all_results[sym][s].roi * 100 for sym in SYMBOLS if sym in all_results and s in all_results[sym]]
    wins = [all_results[sym][s].win_rate * 100 for sym in SYMBOLS if sym in all_results and s in all_results[sym]]
    sharps = [all_results[sym][s].sharpe_ratio for sym in SYMBOLS if sym in all_results and s in all_results[sym]]
    if rois:
        print(f" | {sum(rois)/len(rois):>+7.2f}%  {sum(wins)/len(wins):>5.1f}%  {'':>5}  {sum(sharps)/len(sharps):>6.2f}  {'':>6} ", end="")
    else:
        print(f" | {'N/A':>42}", end="")
print()

# Simple ROI-only table
print("\n\n" + "=" * 70)
print("SIMPLE PROFIT TABLE — ROI% PER STOCK")
print("=" * 70)
print(f"\n{'Symbol':<8} | {'AGGRESSIVE':>12} | {'CONSERVATIVE':>14} | {'BALANCED':>12}")
print("-" * 58)
for symbol in SYMBOLS:
    if symbol not in all_results:
        continue
    parts = [f"{symbol:<8}"]
    for s in strategies:
        r = all_results[symbol].get(s)
        if r:
            parts.append(f"{r.roi*100:>+10.2f}%")
        else:
            parts.append(f"{'N/A':>11}")
    print(f"{parts[0]} | {parts[1]:>12} | {parts[2]:>14} | {parts[3]:>12}")

print("-" * 58)
# Average row
parts = [f"{'AVERAGE':<8}"]
for s in strategies:
    rois = [all_results[sym][s].roi * 100 for sym in SYMBOLS if sym in all_results and s in all_results[sym]]
    if rois:
        parts.append(f"{sum(rois)/len(rois):>+10.2f}%")
    else:
        parts.append(f"{'N/A':>11}")
print(f"{parts[0]} | {parts[1]:>12} | {parts[2]:>14} | {parts[3]:>12}")
print("=" * 70)
