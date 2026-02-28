# =====================================================
# TASK 3.1: SELECTING ML MODELS
# Model Selection for Time Series Stock Price Forecasting
# =====================================================
# Objective: Identify baseline and advanced models for forecasting
# Timeline: Feb 10 - Feb 13, 2026
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# STEP 1: LOAD DATA WITH TECHNICAL INDICATORS
# =====================================================
print("=" * 70)
print("TASK 3.1: MODEL SELECTION FOR TIME SERIES FORECASTING")
print("=" * 70)
print("\nSTEP 1: LOADING DATA WITH TECHNICAL INDICATORS")
print("-" * 70)

try:
    # Load the feature-rich dataset
    data = pd.read_csv("AAPL_stock_data_with_indicators.csv", index_col=0, parse_dates=True)
    print(f"✓ Data loaded successfully!")
    print(f"  Shape: {data.shape}")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    print(f"  Total features: {len(data.columns)}")
    
except FileNotFoundError:
    print("✗ Error: AAPL_stock_data_with_indicators.csv not found!")
    print("  Please run test.py first to generate the feature-rich dataset.")
    exit()

# =====================================================
# STEP 2: DEFINE TARGET VARIABLES
# =====================================================
print("\n" + "=" * 70)
print("STEP 2: DEFINING TARGET VARIABLES")
print("-" * 70)

# Get Close price column (handle different naming conventions)
close_col = None
for col in data.columns:
    if 'close' in col.lower():
        close_col = col
        break

if close_col is None:
    print("✗ Error: Close price column not found!")
    exit()

print(f"\nPrice column identified: {close_col}")

# =====================================================
# TARGET 1: NEXT DAY CLOSE PRICE (REGRESSION)
# =====================================================
print("\n📊 TARGET VARIABLE 1: NEXT DAY CLOSE PRICE (Regression)")
print("-" * 70)
target_close = data[close_col].shift(-1)  # Next day's close
data['Target_Close_Price'] = target_close

# Remove last row (has NaN target)
data_with_targets = data.dropna()

print(f"  Type: Continuous (Regression)")
print(f"  Prediction Horizon: 1 day ahead")
print(f"  Value Range: ${data_with_targets['Target_Close_Price'].min():.2f} - ${data_with_targets['Target_Close_Price'].max():.2f}")
print(f"  Mean: ${data_with_targets['Target_Close_Price'].mean():.2f}")
print(f"  Std Dev: ${data_with_targets['Target_Close_Price'].std():.2f}")

# =====================================================
# TARGET 2: DIRECTIONAL MOVEMENT (CLASSIFICATION)
# =====================================================
print("\n📊 TARGET VARIABLE 2: DIRECTIONAL MOVEMENT (Classification)")
print("-" * 70)
price_change = data[close_col].diff()
directional_change = (price_change.shift(-1) > 0).astype(int)  # 1 = Up, 0 = Down
data['Target_Direction'] = directional_change

data_with_targets['Target_Direction'] = data['Target_Direction'].iloc[:-1]

up_count = (data_with_targets['Target_Direction'] == 1).sum()
down_count = (data_with_targets['Target_Direction'] == 0).sum()

print(f"  Type: Binary Classification (Up/Down)")
print(f"  Prediction Horizon: 1 day ahead")
print(f"  Class Distribution:")
print(f"    - Up (1):   {up_count:4d} samples ({100*up_count/len(data_with_targets):.1f}%)")
print(f"    - Down (0): {down_count:4d} samples ({100*down_count/len(data_with_targets):.1f}%)")

# =====================================================
# TARGET 3: PRICE RETURN (PERCENTAGE CHANGE)
# =====================================================
print("\n📊 TARGET VARIABLE 3: DAILY RETURN % (Regression)")
print("-" * 70)
pct_change = data[close_col].pct_change() * 100
target_return = pct_change.shift(-1)
data['Target_Return_Pct'] = target_return

data_with_targets['Target_Return_Pct'] = data['Target_Return_Pct'].iloc[:-1]

print(f"  Type: Continuous (Regression)")
print(f"  Prediction Horizon: 1 day ahead")
print(f"  Value Range: {data_with_targets['Target_Return_Pct'].min():.2f}% - {data_with_targets['Target_Return_Pct'].max():.2f}%")
print(f"  Mean: {data_with_targets['Target_Return_Pct'].mean():.2f}%")
print(f"  Std Dev: {data_with_targets['Target_Return_Pct'].std():.2f}%")

# =====================================================
# STEP 3: DEFINE INPUT FEATURES
# =====================================================
print("\n" + "=" * 70)
print("STEP 3: DEFINING INPUT FEATURES")
print("-" * 70)

# Price columns (original OHLCV)
price_features = [col for col in data.columns 
                 if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]

# Technical indicators (already calculated)
technical_features = [col for col in data.columns 
                     if col not in price_features and not col.startswith('Target_')]

all_features = price_features + technical_features

print(f"\n✓ PRICE FEATURES ({len(price_features)}):")
for i, feat in enumerate(price_features, 1):
    print(f"  {i:2d}. {feat}")

print(f"\n✓ TECHNICAL INDICATORS ({len(technical_features)}):")
print(f"\n  Trend Indicators:")
trend_indicators = ['SMA', 'EMA']
trend_cols = [col for col in technical_features if any(x in col for x in trend_indicators)]
for i, col in enumerate(trend_cols, 1):
    print(f"    {i}. {col}")

print(f"\n  Momentum Indicators:")
momentum_indicators = ['RSI', 'MACD', 'ROC']
momentum_cols = [col for col in technical_features if any(x in col for x in momentum_indicators)]
for i, col in enumerate(momentum_cols, 1):
    print(f"    {i}. {col}")

print(f"\n  Volatility Indicators:")
volatility_indicators = ['BB_', 'ATR', 'Volatility']
volatility_cols = [col for col in technical_features if any(x in col for x in volatility_indicators)]
for i, col in enumerate(volatility_cols, 1):
    print(f"    {i}. {col}")

print(f"\n  Total Input Features: {len(all_features)}")

# =====================================================
# STEP 4: PREPARE FEATURE MATRIX AND TARGETS
# =====================================================
print("\n" + "=" * 70)
print("STEP 4: PREPARING FEATURE MATRIX AND TARGET VARIABLES")
print("-" * 70)

# Use only complete rows
X = data_with_targets[all_features].copy()
y_close = data_with_targets['Target_Close_Price'].copy()
y_direction = data_with_targets['Target_Direction'].copy()
y_return = data_with_targets['Target_Return_Pct'].copy()

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Target (Close Price) shape: {y_close.shape}")
print(f"✓ Target (Direction) shape: {y_direction.shape}")
print(f"✓ Target (Return %) shape: {y_return.shape}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print(f"\n✓ Features scaled using StandardScaler")
print(f"  Mean (should be ~0): {X_scaled.mean().mean():.6f}")
print(f"  Std (should be ~1): {X_scaled.std().mean():.6f}")

# =====================================================
# STEP 5: BASELINE MODELS (Simple & Interpretable)
# =====================================================
print("\n" + "=" * 70)
print("STEP 5: BASELINE MODELS (Simple & Interpretable)")
print("-" * 70)

baseline_models = {}

# MODEL 1: LINEAR REGRESSION
print("\n🔹 MODEL 1: LINEAR REGRESSION")
print("  Description: Simple linear relationship between features and target")
print("  Use Case: Baseline for quick interpretation and speed")
print("  Pros: Fast, interpretable, low memory usage")
print("  Cons: Limited to linear patterns, poor for complex relationships")
print("  Best For: Initial benchmarking, feature importance analysis")
baseline_models['Linear_Regression'] = {
    'estimator': LinearRegression(),
    'name': 'Linear Regression',
    'description': 'Simple linear regression model',
    'complexity': 'Low',
    'interpretability': 'Very High',
    'computational_cost': 'Low',
    'hyperparameters': {}
}

# MODEL 2: RANDOM FOREST
print("\n🔹 MODEL 2: RANDOM FOREST")
print("  Description: Ensemble of decision trees using bootstrap sampling")
print("  Use Case: Captures non-linear relationships and feature interactions")
print("  Pros: Non-linear capability, feature importance, robust to outliers")
print("  Cons: Slower than linear, less interpretable, may overfit")
print("  Best For: General-purpose regression with non-linear patterns")
baseline_models['Random_Forest'] = {
    'estimator': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'name': 'Random Forest',
    'description': 'Ensemble of decision trees',
    'complexity': 'Medium',
    'interpretability': 'Medium',
    'computational_cost': 'Medium',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
}

# MODEL 3: SUPPORT VECTOR REGRESSION (SVR)
print("\n🔹 MODEL 3: SUPPORT VECTOR REGRESSION (SVR)")
print("  Description: Maps data to higher dimension using kernel trick")
print("  Use Case: Non-linear regression via kernel functions")
print("  Pros: Effective in high dimensions, kernel flexibility")
print("  Cons: Slower training/prediction, sensitive to feature scaling")
print("  Best For: Complex non-linear relationships, small-medium datasets")
baseline_models['SVR'] = {
    'estimator': SVR(
        kernel='rbf',
        C=100,
        epsilon=0.1,
        gamma='scale'
    ),
    'name': 'Support Vector Regression',
    'description': 'SVM regression with RBF kernel',
    'complexity': 'Medium-High',
    'interpretability': 'Low',
    'computational_cost': 'High',
    'hyperparameters': {
        'kernel': 'rbf',
        'C': 100,
        'epsilon': 0.1,
        'gamma': 'scale'
    }
}

print(f"\n✓ {len(baseline_models)} Baseline Models Configured")

# =====================================================
# STEP 6: ADVANCED MODELS (Deep Learning)
# =====================================================
print("\n" + "=" * 70)
print("STEP 6: ADVANCED MODELS (Deep Learning - Sequential Architecture)")
print("-" * 70)

advanced_models = {}

# MODEL 4: LSTM (LONG SHORT-TERM MEMORY)
print("\n🔹 MODEL 4: LSTM (LONG SHORT-TERM MEMORY)")
print("  Description: Recurrent Neural Network with memory cells")
print("  Use Case: Time-series forecasting with long-term dependencies")
print("  Pros: Captures temporal patterns, handles long sequences")
print("  Cons: Complex hyperparameter tuning, slow training, prone to overfitting")
print("  Best For: Sequential time-series with long-range dependencies")
print("  Architecture:")
print("    - Input: Time series sequences (lookback window)")
print("    - Hidden Layers: 2-3 LSTM layers with dropout")
print("    - Output: Continuous price predictions")
advanced_models['LSTM'] = {
    'name': 'LSTM (RNN)',
    'description': 'Long Short-Term Memory Neural Network',
    'type': 'RNN',
    'complexity': 'High',
    'interpretability': 'Very Low',
    'computational_cost': 'Very High',
    'architecture': {
        'layers': [
            {'type': 'LSTM', 'units': 64, 'return_sequences': True, 'dropout': 0.2},
            {'type': 'LSTM', 'units': 32, 'return_sequences': False, 'dropout': 0.2},
            {'type': 'Dense', 'units': 16, 'activation': 'relu'},
            {'type': 'Dense', 'units': 1, 'activation': 'linear'}
        ],
        'sequence_length': 30,  # 30 days lookback
        'optimizer': 'Adam',
        'loss': 'mse',
        'batch_size': 32,
        'epochs': 50
    },
    'hyperparameters': {
        'sequence_length': 30,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'dropout': 0.2
    }
}

# MODEL 5: GRU (GATED RECURRENT UNIT)
print("\n🔹 MODEL 5: GRU (GATED RECURRENT UNIT)")
print("  Description: Simplified RNN with gating mechanism")
print("  Use Case: Time-series forecasting (faster LSTM alternative)")
print("  Pros: Faster than LSTM, fewer parameters, good for long sequences")
print("  Cons: May lose fine-grained temporal patterns vs LSTM")
print("  Best For: Time-series with limited data/compute resources")
print("  Architecture:")
print("    - Input: Time series sequences (lookback window)")
print("    - Hidden Layers: 2 GRU layers with dropout")
print("    - Output: Continuous price predictions")
advanced_models['GRU'] = {
    'name': 'GRU (RNN)',
    'description': 'Gated Recurrent Unit Neural Network',
    'type': 'RNN',
    'complexity': 'High',
    'interpretability': 'Very Low',
    'computational_cost': 'High',
    'architecture': {
        'layers': [
            {'type': 'GRU', 'units': 64, 'return_sequences': True, 'dropout': 0.2},
            {'type': 'GRU', 'units': 32, 'return_sequences': False, 'dropout': 0.2},
            {'type': 'Dense', 'units': 16, 'activation': 'relu'},
            {'type': 'Dense', 'units': 1, 'activation': 'linear'}
        ],
        'sequence_length': 30,  # 30 days lookback
        'optimizer': 'Adam',
        'loss': 'mse',
        'batch_size': 32,
        'epochs': 50
    },
    'hyperparameters': {
        'sequence_length': 30,
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'dropout': 0.2
    }
}

print(f"\n✓ {len(advanced_models)} Advanced Models Configured")

# =====================================================
# STEP 7: QUICK BASELINE TRAINING & EVALUATION
# =====================================================
print("\n" + "=" * 70)
print("STEP 7: QUICK BASELINE TRAINING & EVALUATION")
print("-" * 70)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_close, test_size=0.2, shuffle=False
)

print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Train baseline models
baseline_results = {}

for model_key, model_info in baseline_models.items():
    print(f"\n  Training {model_info['name']}...")
    
    try:
        model = model_info['estimator']
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        baseline_results[model_key] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"    ✓ R² Score: {r2:.4f}")
        print(f"    ✓ RMSE: ${rmse:.2f}")
        print(f"    ✓ MAE: ${mae:.2f}")
        
    except Exception as e:
        print(f"    ✗ Error training model: {str(e)}")

# =====================================================
# STEP 8: MODEL COMPARISON & SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("STEP 8: MODEL SELECTION SUMMARY")
print("-" * 70)

print("\n📋 BASELINE MODELS PERFORMANCE COMPARISON:")
print("-" * 70)

comparison_df = pd.DataFrame({
    'Model': [baseline_results[k]['model'].__class__.__name__ for k in baseline_results.keys()],
    'R² Score': [baseline_results[k]['r2'] for k in baseline_results.keys()],
    'RMSE ($)': [baseline_results[k]['rmse'] for k in baseline_results.keys()],
    'MAE ($)': [baseline_results[k]['mae'] for k in baseline_results.keys()]
})

print(comparison_df.to_string(index=False))

best_baseline = max(baseline_results, key=lambda x: baseline_results[x]['r2'])
print(f"\n✓ Best Baseline Model: {baseline_models[best_baseline]['name']}")
print(f"  R² Score: {baseline_results[best_baseline]['r2']:.4f}")

# =====================================================
# STEP 9: SAVE MODEL SELECTION SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("STEP 9: SAVING MODEL SELECTION SUMMARY")
print("-" * 70)

# Create comprehensive model selection report
report = {
    'Dataset': 'AAPL Stock Data (2020-2025)',
    'Total_Samples': len(data_with_targets),
    'Features': len(all_features),
    'Price_Features': len(price_features),
    'Technical_Indicators': len(technical_features),
    'Target_Variables': 3,
    'Targets': ['Close Price (Regression)', 'Direction (Classification)', 'Return % (Regression)'],
}

# Save report
with open('model_selection_report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MODEL SELECTION SUMMARY - TASK 3.1\n")
    f.write("Time Series Stock Price Forecasting\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("DATASET OVERVIEW:\n")
    f.write("-" * 70 + "\n")
    for key, value in report.items():
        if isinstance(value, list):
            f.write(f"{key}:\n")
            for item in value:
                f.write(f"  - {item}\n")
        else:
            f.write(f"{key}: {value}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("TARGET VARIABLES:\n")
    f.write("-" * 70 + "\n")
    f.write("1. Next Day Close Price (Regression)\n")
    f.write("2. Directional Movement Up/Down (Classification)\n")
    f.write("3. Daily Return % (Regression)\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("BASELINE MODELS (Traditional ML):\n")
    f.write("-" * 70 + "\n")
    for model_key, model_info in baseline_models.items():
        f.write(f"\n{model_info['name']}:\n")
        f.write(f"  Description: {model_info['description']}\n")
        f.write(f"  Complexity: {model_info['complexity']}\n")
        f.write(f"  Interpretability: {model_info['interpretability']}\n")
        f.write(f"  Computational Cost: {model_info['computational_cost']}\n")
        if model_key in baseline_results:
            f.write(f"  Performance (Test Set):\n")
            f.write(f"    - R² Score: {baseline_results[model_key]['r2']:.4f}\n")
            f.write(f"    - RMSE: ${baseline_results[model_key]['rmse']:.2f}\n")
            f.write(f"    - MAE: ${baseline_results[model_key]['mae']:.2f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("ADVANCED MODELS (Deep Learning):\n")
    f.write("-" * 70 + "\n")
    for model_key, model_info in advanced_models.items():
        f.write(f"\n{model_info['name']}:\n")
        f.write(f"  Description: {model_info['description']}\n")
        f.write(f"  Type: {model_info['type']}\n")
        f.write(f"  Complexity: {model_info['complexity']}\n")
        f.write(f"  Interpretability: {model_info['interpretability']}\n")
        f.write(f"  Computational Cost: {model_info['computational_cost']}\n")
        f.write(f"  Sequence Length: {model_info['hyperparameters']['sequence_length']} days\n")
        f.write(f"  Architecture:\n")
        for i, layer in enumerate(model_info['architecture']['layers'], 1):
            f.write(f"    Layer {i}: {layer}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("INPUT FEATURES:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Features: {len(all_features)}\n\n")
    f.write("Price Data:\n")
    for feat in price_features:
        f.write(f"  - {feat}\n")
    f.write("\nTechnical Indicators:\n")
    for feat in technical_features:
        f.write(f"  - {feat}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("NEXT STEPS (Task 3.2):\n")
    f.write("-" * 70 + "\n")
    f.write("1. Implement and train baseline models\n")
    f.write("2. Implement and train deep learning models (LSTM, GRU)\n")
    f.write("3. Hyperparameter tuning and cross-validation\n")
    f.write("4. Model evaluation and comparison\n")
    f.write("5. Feature importance analysis\n")
    f.write("6. Select best model for production\n")

print("✓ Model selection report saved to: model_selection_report.txt")

# Save model metadata to JSON
import json

model_metadata = {
    'dataset': {
        'name': 'AAPL Stock Data',
        'samples': int(len(data_with_targets)),
        'date_range': {
            'start': str(data_with_targets.index.min()),
            'end': str(data_with_targets.index.max())
        }
    },
    'features': {
        'total': len(all_features),
        'price_features': price_features,
        'technical_indicators': technical_features
    },
    'targets': {
        'close_price': {
            'type': 'Regression',
            'description': 'Next day close price prediction',
            'min': float(y_close.min()),
            'max': float(y_close.max()),
            'mean': float(y_close.mean()),
            'std': float(y_close.std())
        },
        'direction': {
            'type': 'Classification',
            'description': 'Up/Down directional movement',
            'classes': ['Down (0)', 'Up (1)'],
            'class_distribution': {
                'up': int(up_count),
                'down': int(down_count)
            }
        },
        'return_pct': {
            'type': 'Regression',
            'description': 'Daily return percentage',
            'min': float(y_return.min()),
            'max': float(y_return.max()),
            'mean': float(y_return.mean()),
            'std': float(y_return.std())
        }
    },
    'baseline_models': {
        k: {
            'name': v['name'],
            'description': v['description'],
            'complexity': v['complexity'],
            'interpretability': v['interpretability'],
            'hyperparameters': v['hyperparameters']
        } for k, v in baseline_models.items()
    },
    'advanced_models': {
        k: {
            'name': v['name'],
            'description': v['description'],
            'type': v['type'],
            'complexity': v['complexity'],
            'interpretability': v['interpretability'],
            'hyperparameters': v['hyperparameters']
        } for k, v in advanced_models.items()
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✓ Model metadata saved to: model_metadata.json")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("TASK 3.1 COMPLETE: MODEL SELECTION SUMMARY")
print("=" * 70)

print("\n✅ DELIVERABLES:")
print(f"   1. ✓ Target Variables Defined: 3 (Close Price, Direction, Return %)")
print(f"   2. ✓ Baseline Models Selected: {len(baseline_models)} models")
print(f"   3. ✓ Advanced Models Selected: {len(advanced_models)} models")
print(f"   4. ✓ Input Features Prepared: {len(all_features)} features")
print(f"   5. ✓ Model Selection Report Generated")
print(f"   6. ✓ Model Metadata Saved")

print("\n📊 BASELINE MODEL PERFORMANCE (on test set):")
for model_key, results in baseline_results.items():
    print(f"   {baseline_models[model_key]['name']:25s} - R²: {results['r2']:.4f}, RMSE: ${results['rmse']:.2f}")

print("\n🚀 ADVANCED MODELS READY FOR IMPLEMENTATION:")
for model_key, model_info in advanced_models.items():
    print(f"   - {model_info['name']} (Sequence Length: {model_info['hyperparameters']['sequence_length']} days)")

print("\n📅 TIMELINE:")
print("   Feb 10-13: ✓ Model Selection (COMPLETED)")
print("   Feb 14-20: ○ Model Training & Evaluation (NEXT)")

print("\n" + "=" * 70)
print("Files Generated:")
print("  - model_selection_report.txt")
print("  - model_metadata.json")
print("=" * 70)
