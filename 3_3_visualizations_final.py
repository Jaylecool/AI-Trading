"""
Task 3.3: Deep Learning Visualizations - Final Version

Purpose: Generate visualizations for LSTM and GRU models trained with corrected normalization
Outputs: 8 PNG files showing training history and performance
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
print("TASK 3.3: Deep Learning Visualizations - Final Version")
print("="*80)

# Load training history
with open('deep_learning_training_history.json', 'r') as f:
    history = json.load(f)

# Load data and prepare sequences
full_data = pd.read_csv('AAPL_stock_data_with_indicators.csv', index_col='Date')
val_data = pd.read_csv('AAPL_stock_data_val.csv', index_col='Date')
feature_cols = [col for col in full_data.columns if col not in ['Close_AAPL', 'Target_Close_Price']]

# Prepare validation data
val_data['Target_Close_Price'] = val_data['Close_AAPL'].shift(-1)
val_data = val_data[:-1]
X_val = val_data[feature_cols].values
y_val_original = val_data['Target_Close_Price'].values

# Load scalers
with open('trained_models/feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('trained_models/target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

X_val_scaled = feature_scaler.transform(X_val)
y_val_scaled = target_scaler.transform(y_val_original.reshape(-1, 1)).flatten()

# Create sequences
def create_sequences(data, targets, timesteps=30):
    X_seq, y_seq = [], []
    for i in range(len(data) - timesteps):
        X_seq.append(data[i:i+timesteps])
        y_seq.append(targets[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 30
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, timesteps)
y_val_original_subset = y_val_original[timesteps:]

# Load models
lstm_model = load_model('trained_models/model_LSTM.keras')
gru_model = load_model('trained_models/model_GRU.keras')

# Get predictions
lstm_pred_scaled = lstm_model.predict(X_val_seq, verbose=0).flatten()
lstm_pred_original = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()

gru_pred_scaled = gru_model.predict(X_val_seq, verbose=0).flatten()
gru_pred_original = target_scaler.inverse_transform(gru_pred_scaled.reshape(-1, 1)).flatten()

print("\n[SETUP] Data loaded and prepared")
print(f"  ✓ Validation sequences: {X_val_seq.shape}")
print(f"  ✓ LSTM and GRU models loaded")
print(f"  ✓ Predictions generated on {len(y_val_original_subset)} validation samples")

# ============================================================================
# VIZ 1: Training & Validation Loss Curves
# ============================================================================
print("\n[VIZ 1] Training & Validation Loss Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

axes[0].plot(history['LSTM']['training_loss'], label='Training Loss', linewidth=2, color='#1f77b4')
axes[0].plot(history['LSTM']['validation_loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
axes[0].set_title('LSTM: Training & Validation Loss', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['GRU']['training_loss'], label='Training Loss', linewidth=2, color='#2ca02c')
axes[1].plot(history['GRU']['validation_loss'], label='Validation Loss', linewidth=2, color='#d62728')
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
axes[1].set_title('GRU: Training & Validation Loss', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('13_training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 13_training_validation_loss.png")

# ============================================================================
# VIZ 2: LSTM vs GRU Loss Comparison
# ============================================================================
print("\n[VIZ 2] LSTM vs GRU Loss Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

lstm_len = len(history['LSTM']['training_loss'])
gru_len = len(history['GRU']['training_loss'])

axes[0].plot(range(1, lstm_len + 1), history['LSTM']['training_loss'], marker='o', label='LSTM', linewidth=2, markersize=4)
axes[0].plot(range(1, gru_len + 1), history['GRU']['training_loss'], marker='s', label='GRU', linewidth=2, markersize=4)
axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Training Loss (MSE)', fontsize=11, fontweight='bold')
axes[0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, lstm_len + 1), history['LSTM']['validation_loss'], marker='o', label='LSTM', linewidth=2, markersize=4)
axes[1].plot(range(1, gru_len + 1), history['GRU']['validation_loss'], marker='s', label='GRU', linewidth=2, markersize=4)
axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Validation Loss (MSE)', fontsize=11, fontweight='bold')
axes[1].set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('14_lstm_vs_gru_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 14_lstm_vs_gru_loss.png")

# ============================================================================
# VIZ 3: Actual vs Predicted
# ============================================================================
print("\n[VIZ 3] Actual vs Predicted Prices...")

lstm_r2 = r2_score(y_val_original_subset, lstm_pred_original)
gru_r2 = r2_score(y_val_original_subset, gru_pred_original)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

axes[0].scatter(y_val_original_subset, lstm_pred_original, alpha=0.5, s=20, color='#1f77b4')
axes[0].plot([y_val_original_subset.min(), y_val_original_subset.max()], 
             [y_val_original_subset.min(), y_val_original_subset.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Close Price ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Predicted Close Price ($)', fontsize=11, fontweight='bold')
axes[0].set_title(f"LSTM: Actual vs Predicted (R²={lstm_r2:.4f})", fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_val_original_subset, gru_pred_original, alpha=0.5, s=20, color='#2ca02c')
axes[1].plot([y_val_original_subset.min(), y_val_original_subset.max()], 
             [y_val_original_subset.min(), y_val_original_subset.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Close Price ($)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Predicted Close Price ($)', fontsize=11, fontweight='bold')
axes[1].set_title(f"GRU: Actual vs Predicted (R²={gru_r2:.4f})", fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('16_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 16_actual_vs_predicted.png")

# ============================================================================
# VIZ 4: Prediction Errors
# ============================================================================
print("\n[VIZ 4] Prediction Errors...")

lstm_errors = y_val_original_subset - lstm_pred_original
gru_errors = y_val_original_subset - gru_pred_original

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
# VIZ 5: Time Series Predictions (Last 100 Days)
# ============================================================================
print("\n[VIZ 5] Time Series Predictions...")

plot_range = slice(-100, None)
time_idx = np.arange(len(y_val_original_subset))[plot_range]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=300)

axes[0].plot(time_idx, y_val_original_subset[plot_range], 'o-', label='Actual', linewidth=2, markersize=4, color='black')
axes[0].plot(time_idx, lstm_pred_original[plot_range], 's--', label='Predicted', linewidth=2, markersize=4, color='#1f77b4', alpha=0.8)
axes[0].fill_between(time_idx, y_val_original_subset[plot_range], lstm_pred_original[plot_range], alpha=0.2, color='#1f77b4')
axes[0].set_ylabel('Close Price ($)', fontsize=10, fontweight='bold')
axes[0].set_title('LSTM: Last 100 Days Prediction', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_idx, y_val_original_subset[plot_range], 'o-', label='Actual', linewidth=2, markersize=4, color='black')
axes[1].plot(time_idx, gru_pred_original[plot_range], 's--', label='Predicted', linewidth=2, markersize=4, color='#2ca02c', alpha=0.8)
axes[1].fill_between(time_idx, y_val_original_subset[plot_range], gru_pred_original[plot_range], alpha=0.2, color='#2ca02c')
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
# VIZ 6: Performance Metrics Comparison
# ============================================================================
print("\n[VIZ 6] Performance Metrics Comparison...")

lstm_metrics = history['LSTM']['final_metrics']
gru_metrics = history['GRU']['final_metrics']

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
# VIZ 7: Summary Scorecard
# ============================================================================
print("\n[VIZ 7] Summary Scorecard...")

fig = plt.figure(figsize=(14, 8), dpi=300)
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

fig.suptitle('TASK 3.3: LSTM vs GRU - Performance Summary', 
             fontsize=14, fontweight='bold', y=0.98)

lstm_better = lstm_metrics['r2'] > gru_metrics['r2']
winner = 'LSTM' if lstm_better else 'GRU'

# Title
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
winner_text = f"Best Model: {winner} (R²={max(lstm_metrics['r2'], gru_metrics['r2']):.4f})"
ax_title.text(0.5, 0.5, winner_text, ha='center', va='center', fontsize=18, fontweight='bold',
              transform=ax_title.transAxes, bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7))

# LSTM Card
ax_lstm = fig.add_subplot(gs[1, 0])
ax_lstm.axis('off')
lstm_text = f"""
LSTM MODEL
R² Score:       {lstm_metrics['r2']:.4f}
RMSE:           ${lstm_metrics['rmse']:.2f}
MAE:            ${lstm_metrics['mae']:.2f}
MAPE:           {lstm_metrics['mape']:.2f}%

Epochs:         {history['LSTM']['epochs']}
Time:           {history['LSTM']['training_time_seconds']:.2f}s
Parameters:     34,977
"""
ax_lstm.text(0.05, 0.95, lstm_text, ha='left', va='top', fontsize=10, family='monospace',
             transform=ax_lstm.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# GRU Card
ax_gru = fig.add_subplot(gs[1, 1])
ax_gru.axis('off')
gru_text = f"""
GRU MODEL
R² Score:       {gru_metrics['r2']:.4f}
RMSE:           ${gru_metrics['rmse']:.2f}
MAE:            ${gru_metrics['mae']:.2f}
MAPE:           {gru_metrics['mape']:.2f}%

Epochs:         {history['GRU']['epochs']}
Time:           {history['GRU']['training_time_seconds']:.2f}s
Parameters:     26,657
"""
ax_gru.text(0.05, 0.95, gru_text, ha='left', va='top', fontsize=10, family='monospace',
            transform=ax_gru.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Comparison
ax_baseline = fig.add_subplot(gs[2, :])
ax_baseline.axis('off')
baseline_text = f"""
BASELINE COMPARISON (Linear Regression: R²=0.9316, RMSE=$2.32, MAE=$1.74)

LSTM vs Baseline:  R² = {lstm_metrics['r2']:.4f} vs 0.9316 (difference: {lstm_metrics['r2'] - 0.9316:+.4f})  Below baseline
GRU vs Baseline:   R² = {gru_metrics['r2']:.4f} vs 0.9316 (difference: {gru_metrics['r2'] - 0.9316:+.4f})  Below baseline

RECOMMENDATION: Linear Regression preferred for production (simpler, more accurate)
Deep learning models provide insights into sequential patterns but need optimization
"""
ax_baseline.text(0.05, 0.5, baseline_text, ha='left', va='center', fontsize=10, family='monospace',
                 transform=ax_baseline.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('20_summary_scorecard.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 20_summary_scorecard.png")

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"\n✓ Generated 7 comprehensive PNG files (13-20)")
print(f"✓ All visualizations at 300 DPI")
print(f"\nFiles: 13_training_validation_loss.png through 20_summary_scorecard.png")
print("\n" + "="*80 + "\n")
