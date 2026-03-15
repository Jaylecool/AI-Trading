"""
Task 3.3: Deep Learning Visualizations

Purpose: Generate comprehensive charts for LSTM and GRU training and evaluation
Outputs: 6+ PNG visualization files at 300 DPI

Timeline: Feb 19-25, 2026
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n" + "="*80)
print("TASK 3.3: Deep Learning Visualizations")
print("="*80)

# ============================================================================
# SETUP & LOAD DATA
# ============================================================================
print("\n[SETUP] Loading training history and data...")

# Load training history
with open('deep_learning_training_history.json', 'r') as f:
    history = json.load(f)

# Load data and prepare for visualization
full_data = pd.read_csv('AAPL_stock_data_with_indicators.csv', index_col='Date')
val_data = pd.read_csv('AAPL_stock_data_val.csv', index_col='Date')
feature_cols = [col for col in full_data.columns if col not in ['Close_AAPL', 'Target_Close_Price']]

# Prepare validation data
val_data['Target_Close_Price'] = val_data['Close_AAPL'].shift(-1)
val_data = val_data[:-1]
X_val = val_data[feature_cols].values
y_val = val_data['Target_Close_Price'].values

# Scale
with open('trained_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_val_scaled = scaler.transform(X_val)

# Create sequences
def create_sequences(data, targets, timesteps=30):
    X_seq, y_seq = [], []
    for i in range(len(data) - timesteps):
        X_seq.append(data[i:i+timesteps])
        y_seq.append(targets[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, timesteps=30)

# Load models
lstm_model = load_model('trained_models/model_LSTM.keras')
gru_model = load_model('trained_models/model_GRU.keras')

# Get predictions
lstm_pred = lstm_model.predict(X_val_seq, verbose=0).flatten()
gru_pred = gru_model.predict(X_val_seq, verbose=0).flatten()

print(f"  ✓ Loaded training history")
print(f"  ✓ Loaded LSTM and GRU models")
print(f"  ✓ Generated predictions on {len(y_val_seq)} validation samples")

# ============================================================================
# VISUALIZATION 1: TRAINING & VALIDATION LOSS CURVES
# ============================================================================
print("\n[VIZ 1] Training & Validation Loss Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# LSTM Loss
axes[0].plot(history['LSTM']['training_loss'], label='Training Loss', linewidth=2, color='#1f77b4')
axes[0].plot(history['LSTM']['validation_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
axes[0].set_title('LSTM Model Training History', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, len(history['LSTM']['training_loss'])-1)

# GRU Loss
axes[1].plot(history['GRU']['training_loss'], label='Training Loss', linewidth=2, color='#2ca02c')
axes[1].plot(history['GRU']['validation_loss'], label='Validation Loss', linewidth=2, color='#d62728')
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
axes[1].set_title('GRU Model Training History', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, len(history['GRU']['training_loss'])-1)

plt.tight_layout()
plt.savefig('13_training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 13_training_validation_loss.png")

# ============================================================================
# VISUALIZATION 2: LSTM VS GRU LOSS COMPARISON
# ============================================================================
print("\n[VIZ 2] LSTM vs GRU Loss Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Get max length for alignment
lstm_len = len(history['LSTM']['training_loss'])
gru_len = len(history['GRU']['training_loss'])

# Training Loss Comparison
axes[0].plot(range(1, lstm_len + 1), history['LSTM']['training_loss'], marker='o', label='LSTM', linewidth=2, markersize=4)
axes[0].plot(range(1, gru_len + 1), history['GRU']['training_loss'], marker='s', label='GRU', linewidth=2, markersize=4)
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Training Loss (MSE)', fontsize=11, fontweight='bold')
axes[0].set_title('Training Loss: LSTM vs GRU', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Validation Loss Comparison
axes[1].plot(range(1, lstm_len + 1), history['LSTM']['validation_loss'], marker='o', label='LSTM', linewidth=2, markersize=4)
axes[1].plot(range(1, gru_len + 1), history['GRU']['validation_loss'], marker='s', label='GRU', linewidth=2, markersize=4)
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Validation Loss (MSE)', fontsize=11, fontweight='bold')
axes[1].set_title('Validation Loss: LSTM vs GRU', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('14_lstm_vs_gru_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 14_lstm_vs_gru_loss.png")

# ============================================================================
# VISUALIZATION 3: MAE CURVES
# ============================================================================
print("\n[VIZ 3] MAE Training Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# LSTM MAE
axes[0].plot(history['LSTM']['training_mae'], label='Training MAE', linewidth=2, color='#1f77b4')
axes[0].plot(history['LSTM']['validation_mae'], label='Validation MAE', linewidth=2, color='#ff7f0e')
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('MAE ($)', fontsize=11, fontweight='bold')
axes[0].set_title('LSTM Model MAE History', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# GRU MAE
axes[1].plot(history['GRU']['training_mae'], label='Training MAE', linewidth=2, color='#2ca02c')
axes[1].plot(history['GRU']['validation_mae'], label='Validation MAE', linewidth=2, color='#d62728')
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('MAE ($)', fontsize=11, fontweight='bold')
axes[1].set_title('GRU Model MAE History', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_mae_training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 15_mae_training_curves.png")

# ============================================================================
# VISUALIZATION 4: ACTUAL VS PREDICTED
# ============================================================================
print("\n[VIZ 4] Actual vs Predicted Prices...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# LSTM
axes[0].scatter(y_val_seq, lstm_pred, alpha=0.5, s=20, color='#1f77b4')
axes[0].plot([y_val_seq.min(), y_val_seq.max()], [y_val_seq.min(), y_val_seq.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Close Price ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted Close Price ($)', fontsize=11, fontweight='bold')
axes[0].set_title(f"LSTM: Actual vs Predicted (R²={history['LSTM']['final_metrics']['r2']:.4f})", 
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# GRU
axes[1].scatter(y_val_seq, gru_pred, alpha=0.5, s=20, color='#2ca02c')
axes[1].plot([y_val_seq.min(), y_val_seq.max()], [y_val_seq.min(), y_val_seq.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Close Price ($)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Predicted Close Price ($)', fontsize=11, fontweight='bold')
axes[1].set_title(f"GRU: Actual vs Predicted (R²={history['GRU']['final_metrics']['r2']:.4f})", 
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('16_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 16_actual_vs_predicted.png")

# ============================================================================
# VISUALIZATION 5: PREDICTION ERRORS & DISTRIBUTION
# ============================================================================
print("\n[VIZ 5] Prediction Errors...")

lstm_errors = y_val_seq - lstm_pred
gru_errors = y_val_seq - gru_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# LSTM Error Distribution
axes[0, 0].hist(lstm_errors, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(lstm_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${lstm_errors.mean():.2f}')
axes[0, 0].set_xlabel('Error ($)', fontsize=10, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
axes[0, 0].set_title(f'LSTM Error Distribution (Std: ${lstm_errors.std():.2f})', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# GRU Error Distribution
axes[0, 1].hist(gru_errors, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(gru_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${gru_errors.mean():.2f}')
axes[0, 1].set_xlabel('Error ($)', fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=10, fontweight='bold')
axes[0, 1].set_title(f'GRU Error Distribution (Std: ${gru_errors.std():.2f})', fontsize=11, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# LSTM Residuals over time
axes[1, 0].plot(lstm_errors, label='LSTM Error', color='#1f77b4', linewidth=1, alpha=0.7)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].fill_between(range(len(lstm_errors)), lstm_errors.std(), -lstm_errors.std(), alpha=0.1, color='#1f77b4')
axes[1, 0].set_xlabel('Sample Index', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Error ($)', fontsize=10, fontweight='bold')
axes[1, 0].set_title('LSTM Residuals Over Time', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# GRU Residuals over time
axes[1, 1].plot(gru_errors, label='GRU Error', color='#2ca02c', linewidth=1, alpha=0.7)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].fill_between(range(len(gru_errors)), gru_errors.std(), -gru_errors.std(), alpha=0.1, color='#2ca02c')
axes[1, 1].set_xlabel('Sample Index', fontsize=10, fontweight='bold')
axes[1, 1].set_ylabel('Error ($)', fontsize=10, fontweight='bold')
axes[1, 1].set_title('GRU Residuals Over Time', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('17_prediction_errors.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 17_prediction_errors.png")

# ============================================================================
# VISUALIZATION 6: TIME SERIES PREDICTIONS
# ============================================================================
print("\n[VIZ 6] Time Series Predictions...")

# Use last 100 samples for clarity
plot_range = slice(-100, None)
time_idx = np.arange(len(y_val_seq))[plot_range]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=300)

# LSTM
axes[0].plot(time_idx, y_val_seq[plot_range], 'o-', label='Actual', linewidth=2, markersize=4, color='black')
axes[0].plot(time_idx, lstm_pred[plot_range], 's--', label='Predicted', linewidth=2, markersize=4, color='#1f77b4', alpha=0.8)
axes[0].fill_between(time_idx, y_val_seq[plot_range], lstm_pred[plot_range], alpha=0.2, color='#1f77b4')
axes[0].set_ylabel('Close Price ($)', fontsize=10, fontweight='bold')
axes[0].set_title('LSTM: Last 100 Days Prediction', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# GRU
axes[1].plot(time_idx, y_val_seq[plot_range], 'o-', label='Actual', linewidth=2, markersize=4, color='black')
axes[1].plot(time_idx, gru_pred[plot_range], 's--', label='Predicted', linewidth=2, markersize=4, color='#2ca02c', alpha=0.8)
axes[1].fill_between(time_idx, y_val_seq[plot_range], gru_pred[plot_range], alpha=0.2, color='#2ca02c')
axes[1].set_xlabel('Sample Index', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Close Price ($)', fontsize=10, fontweight='bold')
axes[1].set_title('GRU: Last 100 Days Prediction', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('18_timeseries_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 18_timeseries_predictions.png")

# ============================================================================
# VISUALIZATION 7: PERFORMANCE METRICS COMPARISON
# ============================================================================
print("\n[VIZ 7] Performance Metrics Comparison...")

lstm_metrics = history['LSTM']['final_metrics']
gru_metrics = history['GRU']['final_metrics']

metrics_names = ['R²', 'RMSE', 'MAE', 'MAPE']
lstm_vals = [lstm_metrics['r2'], lstm_metrics['rmse'], lstm_metrics['mae'], lstm_metrics['mape']]
gru_vals = [gru_metrics['r2'], gru_metrics['rmse'], gru_metrics['mae'], gru_metrics['mape']]
baseline_vals = [0.9316, 2.32, 1.74, None]  # Linear Regression

fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)

# R² Score
ax = axes[0, 0]
models = ['LSTM', 'GRU', 'Baseline\n(Linear Reg)']
values = [lstm_metrics['r2'], gru_metrics['r2'], 0.9316]
colors = ['#1f77b4', '#2ca02c', '#d62728']
bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{values[i]:.4f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax.set_title('R² Score Comparison', fontsize=11, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# RMSE
ax = axes[0, 1]
values = [lstm_metrics['rmse'], gru_metrics['rmse'], 2.32]
bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'${values[i]:.2f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('RMSE ($)', fontsize=10, fontweight='bold')
ax.set_title('Root Mean Squared Error', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# MAE
ax = axes[1, 0]
values = [lstm_metrics['mae'], gru_metrics['mae'], 1.74]
bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'${values[i]:.2f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('MAE ($)', fontsize=10, fontweight='bold')
ax.set_title('Mean Absolute Error', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# MAPE
ax = axes[1, 1]
values = [lstm_metrics['mape'], gru_metrics['mape']]
models_mape = ['LSTM', 'GRU']
colors_mape = colors[:2]
bars = ax.bar(models_mape, values, color=colors_mape, alpha=0.8, edgecolor='black', linewidth=1.5)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{values[i]:.2f}%', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylabel('MAPE (%)', fontsize=10, fontweight='bold')
ax.set_title('Mean Absolute Percentage Error', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('19_performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 19_performance_metrics_comparison.png")

# ============================================================================
# VISUALIZATION 8: SUMMARY SCORECARD
# ============================================================================
print("\n[VIZ 8] Summary Scorecard...")

fig = plt.figure(figsize=(14, 8), dpi=300)
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('TASK 3.3: LSTM vs GRU - Performance Summary Scorecard', 
             fontsize=14, fontweight='bold', y=0.98)

# Determine winner
lstm_better = lstm_metrics['r2'] > gru_metrics['r2']
winner = 'LSTM' if lstm_better else 'GRU'

# Overall Winner
ax_winner = fig.add_subplot(gs[0, :])
ax_winner.axis('off')
winner_text = f"🏆 WINNER: {winner} Model"
ax_winner.text(0.5, 0.5, winner_text, ha='center', va='center', fontsize=20, fontweight='bold',
               transform=ax_winner.transAxes, bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

# LSTM Card
ax_lstm = fig.add_subplot(gs[1, 0])
ax_lstm.axis('off')
lstm_text = f"""
LSTM MODEL
{'='*35}
R² Score:        {lstm_metrics['r2']:.4f}
RMSE:            ${lstm_metrics['rmse']:.2f}
MAE:             ${lstm_metrics['mae']:.2f}
MAPE:            {lstm_metrics['mape']:.2f}%

Training Time:   {history['LSTM']['training_time_seconds']:.2f}s
Epochs Trained:  {history['LSTM']['epochs']}
Final Val Loss:  {history['LSTM']['validation_loss'][-1]:.6f}

Status: {'✓ WINNER' if lstm_better else '✗ Second'}
"""
ax_lstm.text(0.05, 0.95, lstm_text, ha='left', va='top', fontsize=10, family='monospace',
             transform=ax_lstm.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# GRU Card
ax_gru = fig.add_subplot(gs[1, 1])
ax_gru.axis('off')
gru_text = f"""
GRU MODEL
{'='*35}
R² Score:        {gru_metrics['r2']:.4f}
RMSE:            ${gru_metrics['rmse']:.2f}
MAE:             ${gru_metrics['mae']:.2f}
MAPE:            {gru_metrics['mape']:.2f}%

Training Time:   {history['GRU']['training_time_seconds']:.2f}s
Epochs Trained:  {history['GRU']['epochs']}
Final Val Loss:  {history['GRU']['validation_loss'][-1]:.6f}

Status: {'✓ WINNER' if not lstm_better else '✗ Second'}
"""
ax_gru.text(0.05, 0.95, gru_text, ha='left', va='top', fontsize=10, family='monospace',
            transform=ax_gru.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Comparison vs Baseline
ax_baseline = fig.add_subplot(gs[2, :])
ax_baseline.axis('off')
baseline_status_lstm = '✓ BEATS' if lstm_metrics['r2'] > 0.9316 else '✗ Below'
baseline_status_gru = '✓ BEATS' if gru_metrics['r2'] > 0.9316 else '✗ Below'
baseline_text = f"""
BASELINE COMPARISON (Linear Regression: R²=0.9316)
{'='*90}
LSTM vs Baseline:  R² {lstm_metrics['r2']:.4f} vs 0.9316  ({lstm_metrics['r2'] - 0.9316:+.4f})  {baseline_status_lstm} baseline
GRU vs Baseline:   R² {gru_metrics['r2']:.4f} vs 0.9316  ({gru_metrics['r2'] - 0.9316:+.4f})  {baseline_status_gru} baseline

Recommendation:  Use {('LSTM' if lstm_metrics['r2'] > gru_metrics['r2'] and lstm_metrics['r2'] > 0.9316 else ('GRU' if gru_metrics['r2'] > 0.9316 else 'Linear Regression'))} for production
"""
ax_baseline.text(0.05, 0.5, baseline_text, ha='left', va='center', fontsize=10, family='monospace',
                 transform=ax_baseline.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('20_summary_scorecard.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 20_summary_scorecard.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ VISUALIZATION SUMMARY")
print("="*80)
print(f"\n✓ Generated 8 comprehensive PNG files (13-20)")
print(f"✓ All visualizations at 300 DPI for publication quality")
print(f"\nFiles Created:")
print(f"  13_training_validation_loss.png - Loss curves for LSTM and GRU")
print(f"  14_lstm_vs_gru_loss.png - Direct comparison of loss trajectories")
print(f"  15_mae_training_curves.png - MAE convergence over epochs")
print(f"  16_actual_vs_predicted.png - Scatter plots with perfect prediction line")
print(f"  17_prediction_errors.png - Error distributions and residuals")
print(f"  18_timeseries_predictions.png - Last 100 days predictions")
print(f"  19_performance_metrics_comparison.png - R², RMSE, MAE, MAPE bars")
print(f"  20_summary_scorecard.png - Complete performance scorecard")

print("\n" + "="*80)
print("✅ TASK 3.3 VISUALIZATIONS COMPLETE")
print("="*80 + "\n")
