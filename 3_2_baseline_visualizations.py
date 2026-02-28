# =====================================================
# TASK 3.2: BASELINE MODEL VISUALIZATIONS
# Performance Comparison and Analysis
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("=" * 80)
print("GENERATING VISUALIZATIONS FOR TASK 3.2")
print("=" * 80)

# =====================================================
# LOAD RESULTS AND DATA
# =====================================================
print("\nLoading results and data...")

# Load comparison CSV
comparison_df = pd.read_csv('baseline_models_comparison.csv')
print(f"✓ Comparison data loaded: {len(comparison_df)} models")

# Load JSON results
with open('baseline_models_results.json', 'r') as f:
    json_results = json.load(f)
print(f"✓ JSON results loaded")

# Load data for predictions
full_data = pd.read_csv("AAPL_stock_data_with_indicators.csv", index_col=0, parse_dates=True)
val_data = pd.read_csv("AAPL_stock_data_val.csv", index_col=0, parse_dates=True)
scaler = pickle.load(open('trained_models/scaler.pkl', 'rb'))

# Create target
target_series = full_data['Close_AAPL'].shift(-1)[:-1]
y_val = target_series.loc[val_data.index]

# Get features
features = [col for col in full_data.columns]
X_val = full_data.loc[val_data.index, features]
X_val_scaled = scaler.transform(X_val)

# Get predictions
models_dict = {
    'Linear Regression': pickle.load(open('trained_models/model_Linear_Regression.pkl', 'rb')),
    'Random Forest': pickle.load(open('trained_models/model_Random_Forest.pkl', 'rb')),
    'SVR': pickle.load(open('trained_models/model_SVR.pkl', 'rb'))
}

# =====================================================
# FIGURE 1: PERFORMANCE METRICS COMPARISON
# =====================================================
print("\nGenerating Figure 1: Performance Metrics Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Baseline Model Performance Comparison (Validation Set)', fontsize=16, fontweight='bold')

# R² Score
ax = axes[0, 0]
colors = ['#2ecc71' if x > 0.7 else '#f39c12' if x > 0.3 else '#e74c3c' for x in comparison_df['R² Score']]
bars = ax.bar(comparison_df['Model'], comparison_df['R² Score'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax.set_title('R² Score (Higher is Better)', fontweight='bold', fontsize=12)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylim([min(comparison_df['R² Score']) - 0.1, 1.0])
for bar, val in zip(bars, comparison_df['R² Score']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# RMSE
ax = axes[0, 1]
bars = ax.bar(comparison_df['Model'], comparison_df['RMSE ($)'], color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('RMSE ($)', fontweight='bold', fontsize=12)
ax.set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold', fontsize=12)
for bar, val in zip(bars, comparison_df['RMSE ($)']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# MAE
ax = axes[1, 0]
bars = ax.bar(comparison_df['Model'], comparison_df['MAE ($)'], color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('MAE ($)', fontweight='bold', fontsize=12)
ax.set_title('Mean Absolute Error (Lower is Better)', fontweight='bold', fontsize=12)
for bar, val in zip(bars, comparison_df['MAE ($)']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Training Time
ax = axes[1, 1]
bars = ax.bar(comparison_df['Model'], comparison_df['Training Time (s)'], color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('Time (seconds)', fontweight='bold', fontsize=12)
ax.set_title('Training Time (Lower is Better)', fontweight='bold', fontsize=12)
for bar, val in zip(bars, comparison_df['Training Time (s)']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig('07_baseline_performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_baseline_performance_metrics.png")
plt.close()

# =====================================================
# FIGURE 2: ACTUAL VS PREDICTED
# =====================================================
print("Generating Figure 2: Actual vs Predicted...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted Close Prices (Validation Set)', fontsize=14, fontweight='bold')

model_names = list(models_dict.keys())
colors_models = ['#2ecc71', '#3498db', '#e74c3c']

for idx, (model_name, color) in enumerate(zip(model_names, colors_models)):
    ax = axes[idx]
    model = models_dict[model_name]
    y_pred = model.predict(X_val_scaled)
    
    ax.scatter(y_val, y_pred, alpha=0.6, s=50, color=color, edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    ax.set_xlabel('Actual Price ($)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Predicted Price ($)', fontweight='bold', fontsize=11)
    ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE=${rmse:.2f}, MAE=${mae:.2f}', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_actual_vs_predicted.png")
plt.close()

# =====================================================
# FIGURE 3: RESIDUALS ANALYSIS
# =====================================================
print("Generating Figure 3: Residuals Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Residuals Analysis (Actual - Predicted)', fontsize=14, fontweight='bold')

for idx, (model_name, color) in enumerate(zip(model_names, colors_models)):
    model = models_dict[model_name]
    y_pred = model.predict(X_val_scaled)
    residuals = y_val.values - y_pred
    
    # Residuals over time
    ax = axes[0, idx]
    ax.plot(range(len(residuals)), residuals, color=color, linewidth=2, alpha=0.7)
    ax.scatter(range(len(residuals)), residuals, color=color, s=30, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.fill_between(range(len(residuals)), residuals, alpha=0.2, color=color)
    ax.set_xlabel('Validation Sample', fontweight='bold', fontsize=10)
    ax.set_ylabel('Residual ($)', fontweight='bold', fontsize=10)
    ax.set_title(f'{model_name} - Residuals Over Time', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Residuals distribution
    ax = axes[1, idx]
    ax.hist(residuals, bins=20, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${residuals.mean():.2f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Residual ($)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=10)
    ax.set_title(f'{model_name} - Residuals Distribution', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('09_residuals_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_residuals_analysis.png")
plt.close()

# =====================================================
# FIGURE 4: TIME SERIES PREDICTIONS
# =====================================================
print("Generating Figure 4: Time Series Predictions...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Validation Set: Actual vs Predicted Prices Over Time', fontsize=14, fontweight='bold')

model_colors = ['#2ecc71', '#3498db', '#e74c3c']

for idx, (model_name, color) in enumerate(zip(model_names, model_colors)):
    ax = axes[idx]
    model = models_dict[model_name]
    y_pred = model.predict(X_val_scaled)
    
    # Plot actual
    ax.plot(val_data.index, y_val.values, color='black', linewidth=2.5, label='Actual Price', zorder=3)
    
    # Plot predicted
    ax.plot(val_data.index, y_pred, color=color, linewidth=2, linestyle='--', label='Predicted Price', zorder=2)
    
    # Fill between
    ax.fill_between(val_data.index, y_val.values, y_pred, alpha=0.2, color=color)
    
    # Metrics
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    ax.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
    ax.set_title(f'{model_name} (R²={r2:.4f}, RMSE=${rmse:.2f})', fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Date', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('10_timeseries_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 10_timeseries_predictions.png")
plt.close()

# =====================================================
# FIGURE 5: ERROR DISTRIBUTION
# =====================================================
print("Generating Figure 5: Error Distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Absolute Error Distribution by Model', fontsize=14, fontweight='bold')

for idx, (model_name, color) in enumerate(zip(model_names, model_colors)):
    ax = axes[idx]
    model = models_dict[model_name]
    y_pred = model.predict(X_val_scaled)
    abs_errors = np.abs(y_val.values - y_pred)
    
    ax.hist(abs_errors, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    ax.axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${abs_errors.mean():.2f}')
    ax.axvline(np.median(abs_errors), color='green', linestyle='--', linewidth=2, label=f'Median: ${np.median(abs_errors):.2f}')
    
    ax.set_xlabel('Absolute Error ($)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title(f'{model_name}\nMax Error: ${abs_errors.max():.2f}', fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('11_error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 11_error_distribution.png")
plt.close()

# =====================================================
# FIGURE 6: SUMMARY SCORECARD
# =====================================================
print("Generating Figure 6: Summary Scorecard...")

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Baseline Models Summary Scorecard', fontsize=16, fontweight='bold')

# Overall ranking (top row spanning all columns)
ax_rank = fig.add_subplot(gs[0, :])
ax_rank.axis('off')

ranking_text = f"""
BASELINE MODEL EVALUATION RESULTS - VALIDATION SET

🥇 BEST OVERALL: {comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']}
   R² Score: {comparison_df['R² Score'].max():.6f}  |  RMSE: ${comparison_df['RMSE ($)'].min():.2f}  |  MAE: ${comparison_df['MAE ($)'].min():.2f}

🥈 SECOND PLACE: {comparison_df.loc[comparison_df['R² Score'].values.argsort()[-2] if len(comparison_df) > 1 else 0, 'Model']}
   R² Score: {comparison_df['R² Score'].values[np.argsort(comparison_df['R² Score'].values)[-2] if len(comparison_df) > 1 else 0]:.6f}

⏱️  FASTEST: {comparison_df.loc[comparison_df['Training Time (s)'].idxmin(), 'Model']}
   Training Time: {comparison_df['Training Time (s)'].min():.3f} seconds
"""

ax_rank.text(0.05, 0.5, ranking_text, fontsize=12, family='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Detailed scores for each model (bottom row)
models_list = comparison_df['Model'].tolist()
colors_score = ['#2ecc71', '#f39c12', '#e74c3c']

for model_idx in range(len(models_list)):
    ax = fig.add_subplot(gs[1, model_idx])
    ax.axis('off')
    
    model_name = models_list[model_idx]
    r2 = comparison_df.loc[model_idx, 'R² Score']
    rmse = comparison_df.loc[model_idx, 'RMSE ($)']
    mae = comparison_df.loc[model_idx, 'MAE ($)']
    
    details_text = f"{model_name}\n\n" \
                   f"R²: {r2:.4f}\n\n" \
                   f"RMSE: ${rmse:.2f}\n\n" \
                   f"MAE: ${mae:.2f}"
    
    ax.text(0.5, 0.5, details_text, fontsize=11, fontweight='bold',
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor=colors_score[model_idx], alpha=0.6, pad=1))

plt.savefig('12_summary_scorecard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 12_summary_scorecard.png")
plt.close()

# =====================================================
# COMPLETION
# =====================================================
print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)

print("\n✅ Generated Visualizations:")
print("   1. 07_baseline_performance_metrics.png - R², RMSE, MAE, Training Time")
print("   2. 08_actual_vs_predicted.png - Scatter plots for each model")
print("   3. 09_residuals_analysis.png - Time series and distribution of residuals")
print("   4. 10_timeseries_predictions.png - Predictions over validation period")
print("   5. 11_error_distribution.png - Absolute error distributions")
print("   6. 12_summary_scorecard.png - Quick reference scorecard")

print("\n" + "=" * 80)
