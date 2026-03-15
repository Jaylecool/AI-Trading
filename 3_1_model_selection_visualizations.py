# =====================================================
# VISUALIZATION: TARGET VARIABLES & FEATURES
# Model Selection - Data Analysis
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

print("=" * 70)
print("GENERATING VISUALIZATIONS FOR TASK 3.1")
print("=" * 70)

# Load data
print("\nLoading data...")
data = pd.read_csv("AAPL_stock_data_with_indicators.csv", index_col=0, parse_dates=True)

# Get Close price
close_col = None
for col in data.columns:
    if 'close' in col.lower():
        close_col = col
        break

# Create targets
target_close = data[close_col].shift(-1)
price_change = data[close_col].diff()
directional_change = (price_change.shift(-1) > 0).astype(int)
pct_change = data[close_col].pct_change() * 100
target_return = pct_change.shift(-1)

# Clean data
data_clean = data.dropna()
data_clean['Target_Close'] = target_close[data_clean.index]
data_clean['Target_Direction'] = directional_change[data_clean.index]
data_clean['Target_Return'] = target_return[data_clean.index]
data_clean = data_clean.dropna()

print(f"Data shape: {data_clean.shape}")

# =====================================================
# FIGURE 1: TARGET VARIABLES OVERVIEW
# =====================================================
print("\nGenerating Figure 1: Target Variables Overview...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Target Variables for Model Selection - Task 3.1', fontsize=16, fontweight='bold', y=1.00)

# Plot 1: Close Price Over Time
ax = axes[0, 0]
ax.plot(data_clean.index, data_clean['Target_Close'], color='#1f77b4', linewidth=2, label='Target Close Price')
ax.fill_between(data_clean.index, data_clean['Target_Close'].min(), data_clean['Target_Close'], alpha=0.2)
ax.set_title('Target 1: Next Day Close Price (Regression)', fontweight='bold', fontsize=12)
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Close Price Distribution
ax = axes[0, 1]
ax.hist(data_clean['Target_Close'], bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
ax.axvline(data_clean['Target_Close'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${data_clean["Target_Close"].mean():.2f}')
ax.axvline(data_clean['Target_Close'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${data_clean["Target_Close"].median():.2f}')
ax.set_title('Distribution of Next Day Close Price', fontweight='bold', fontsize=12)
ax.set_ylabel('Frequency')
ax.set_xlabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Directional Movement
ax = axes[1, 0]
direction_counts = data_clean['Target_Direction'].value_counts().sort_index()
colors = ['#d62728', '#2ca02c']  # Red for Down, Green for Up
bars = ax.bar(['Down (0)', 'Up (1)'], [direction_counts[0], direction_counts[1]], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_title('Target 2: Directional Movement (Classification)', fontweight='bold', fontsize=12)
ax.set_ylabel('Count')
ax.set_ylabel('Frequency')

# Add percentage labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = 100 * height / len(data_clean)
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Daily Return Distribution
ax = axes[1, 1]
ax.hist(data_clean['Target_Return'], bins=50, color='#ff7f0e', alpha=0.7, edgecolor='black')
ax.axvline(data_clean['Target_Return'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data_clean["Target_Return"].mean():.3f}%')
ax.axvline(data_clean['Target_Return'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data_clean["Target_Return"].median():.3f}%')
ax.set_title('Target 3: Daily Return % (Regression)', fontweight='bold', fontsize=12)
ax.set_ylabel('Frequency')
ax.set_xlabel('Return (%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_target_variables_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_target_variables_overview.png")
plt.close()

# =====================================================
# FIGURE 2: TECHNICAL INDICATORS DISTRIBUTION
# =====================================================
print("Generating Figure 2: Technical Indicators Distribution...")

technical_features = [col for col in data_clean.columns 
                     if col not in ['Open_AAPL', 'High_AAPL', 'Low_AAPL', 'Close_AAPL', 'Volume_AAPL',
                                   'BB_Lower', 'Target_Close', 'Target_Direction', 'Target_Return']]

# Select key indicators for visualization
key_indicators = ['SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR_14', 'Volatility_20']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Technical Indicators Overview (30-Day Window)', fontsize=16, fontweight='bold')

for idx, indicator in enumerate(key_indicators):
    ax = axes[idx // 4, idx % 4]
    
    if indicator in data_clean.columns:
        data_clean[indicator].plot(ax=ax, color='#1f77b4', linewidth=1.5)
        ax.fill_between(range(len(data_clean)), data_clean[indicator].min(), data_clean[indicator], alpha=0.2)
        ax.set_title(indicator, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_technical_indicators_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_technical_indicators_timeseries.png")
plt.close()

# =====================================================
# FIGURE 3: FEATURE CORRELATION HEATMAP
# =====================================================
print("Generating Figure 3: Feature Correlation Heatmap...")

# Select features for correlation
correlation_features = ['Close_AAPL', 'Volume_AAPL', 'SMA_20', 'RSI_14', 'MACD', 
                        'BB_Upper', 'ATR_14', 'Volatility_20', 'Target_Close', 'Target_Return']

correlation_matrix = data_clean[correlation_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix - Task 3.1', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('03_feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_feature_correlation_heatmap.png")
plt.close()

# =====================================================
# FIGURE 4: PRICE & INDICATORS TOGETHER
# =====================================================
print("Generating Figure 4: Price with Technical Indicators...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Price with SMA
ax = axes[0]
ax.plot(data_clean.index, data_clean['Close_AAPL'], label='Close Price', linewidth=2, color='black')
ax.plot(data_clean.index, data_clean['SMA_20'], label='SMA_20', linewidth=2, alpha=0.7)
ax.plot(data_clean.index, data_clean['SMA_50'], label='SMA_50', linewidth=2, alpha=0.7)
ax.fill_between(data_clean.index, data_clean['Close_AAPL'].min(), data_clean['Close_AAPL'], alpha=0.1)
ax.set_title('Price with Moving Averages (Trend Indicators)', fontweight='bold', fontsize=12)
ax.set_ylabel('Price ($)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: RSI
ax = axes[1]
ax.plot(data_clean.index, data_clean['RSI_14'], label='RSI_14', linewidth=2, color='#ff7f0e')
ax.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
ax.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
ax.fill_between(data_clean.index, 30, 70, alpha=0.1, color='gray')
ax.set_title('RSI_14 (Momentum Indicator)', fontweight='bold', fontsize=12)
ax.set_ylabel('RSI Value')
ax.set_ylim(0, 100)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 3: Bollinger Bands
ax = axes[2]
ax.plot(data_clean.index, data_clean['Close_AAPL'], label='Close Price', linewidth=2, color='black')
ax.plot(data_clean.index, data_clean['BB_Upper'], label='BB Upper', linewidth=1.5, linestyle='--', alpha=0.7)
ax.plot(data_clean.index, data_clean['BB_Middle'], label='BB Middle (SMA_20)', linewidth=1.5, alpha=0.7)
ax.plot(data_clean.index, data_clean['BB_Lower'], label='BB Lower', linewidth=1.5, linestyle='--', alpha=0.7)
ax.fill_between(data_clean.index, data_clean['BB_Upper'], data_clean['BB_Lower'], alpha=0.1)
ax.set_title('Bollinger Bands (Volatility Indicator)', fontweight='bold', fontsize=12)
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

fig.suptitle('Technical Indicators for Model Selection', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('04_price_with_indicators.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_price_with_indicators.png")
plt.close()

# =====================================================
# FIGURE 5: MODEL SELECTION SUMMARY INFOGRAPHIC
# =====================================================
print("Generating Figure 5: Model Selection Summary...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('Task 3.1: Model Selection Summary', fontsize=18, fontweight='bold', y=0.98)

# Left column: Dataset Info
ax_data = fig.add_subplot(gs[0:2, 0])
ax_data.axis('off')
dataset_info = f"""
DATASET SUMMARY

Total Samples: 1,058
Date Range: Oct 2020 - Dec 2024
Features: 23
  • Price Data: 7
  • Technical: 16
  
Target Variables: 3
  • Close Price (Regression)
  • Direction (Classification)
  • Return % (Regression)
"""
ax_data.text(0.1, 0.5, dataset_info, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Middle column: Baseline Models
ax_baseline = fig.add_subplot(gs[0, 1:])
ax_baseline.axis('off')
baseline_text = """BASELINE MODELS (Traditional ML)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Linear Regression     ⭐ Low Complexity
2. Random Forest         ⭐⭐ Medium Complexity
3. Support Vector Reg.   ⭐⭐⭐ High Complexity"""
ax_baseline.text(0.05, 0.5, baseline_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

# Right column: Advanced Models
ax_advanced = fig.add_subplot(gs[1, 1:])
ax_advanced.axis('off')
advanced_text = """ADVANCED MODELS (Deep Learning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. LSTM (RNN)            ⭐⭐⭐⭐⭐ Very High
5. GRU (RNN)             ⭐⭐⭐⭐ High
Sequence Length: 30 days"""
ax_advanced.text(0.05, 0.5, advanced_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Bottom row: Features breakdown
ax_features = fig.add_subplot(gs[2, :])
ax_features.axis('off')
features_text = """INPUT FEATURES: 23 Total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Price Data (7):     Close, Open, High, Low, Volume, BB_Upper, BB_Lower
Trend (7):          SMA_10, SMA_20, SMA_50, SMA_200, EMA_10, EMA_20, EMA_50
Momentum (5):       RSI_14, MACD, MACD_Signal, MACD_Histogram, ROC_12
Volatility (4):     BB_Upper, BB_Lower, BB_Middle, ATR_14, Volatility_20"""
ax_features.text(0.05, 0.5, features_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

plt.savefig('05_model_selection_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_model_selection_summary.png")
plt.close()

# =====================================================
# FIGURE 6: BASELINE MODEL PERFORMANCE
# =====================================================
print("Generating Figure 6: Baseline Model Performance Comparison...")

models = ['Linear\nRegression', 'Random\nForest', 'Support Vector\nRegression']
r2_scores = [1.0, -0.0784, -2.5867]
rmse_values = [0.00, 26.95, 49.15]
mae_values = [0.00, 20.67, 38.98]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Baseline Models Performance Comparison (Test Set)', fontsize=14, fontweight='bold')

# R² Score
ax = axes[0]
colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in r2_scores]
bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('R² Score (Higher is Better)', fontweight='bold')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
           f'{val:.4f}', ha='center', va='center', fontweight='bold', fontsize=10, color='white')

# RMSE
ax = axes[1]
bars = ax.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('RMSE ($)', fontweight='bold')
ax.set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, rmse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# MAE
ax = axes[2]
bars = ax.bar(models, mae_values, color=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('MAE ($)', fontweight='bold')
ax.set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('06_baseline_model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_baseline_model_performance.png")
plt.close()

# =====================================================
# COMPLETION SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 70)

print("\n✅ Generated Visualizations:")
print("   1. 01_target_variables_overview.png")
print("   2. 02_technical_indicators_timeseries.png")
print("   3. 03_feature_correlation_heatmap.png")
print("   4. 04_price_with_indicators.png")
print("   5. 05_model_selection_summary.png")
print("   6. 06_baseline_model_performance.png")

print("\n" + "=" * 70)
