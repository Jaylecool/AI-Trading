# =====================================================
# TASK 3.2: TRAINING BASELINE MODELS
# Establishing Performance Benchmarks for Stock Forecasting
# =====================================================
# Timeline: Feb 12-18, 2026
# Objective: Train Linear Regression, Random Forest, SVR
#            Evaluate on validation set and establish benchmarks
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# STEP 1: LOAD TRAINING AND VALIDATION DATASETS
# =====================================================
print("=" * 80)
print("TASK 3.2: TRAINING BASELINE MODELS")
print("Establishing Performance Benchmarks for Stock Price Forecasting")
print("=" * 80)
print("\nSTEP 1: LOADING TRAINING AND VALIDATION DATASETS")
print("-" * 80)

try:
    # Load full data with indicators
    full_data = pd.read_csv("AAPL_stock_data_with_indicators.csv", index_col=0, parse_dates=True)
    print(f"✓ Full data with indicators loaded")
    print(f"  Shape: {full_data.shape}")
    print(f"  Date range: {full_data.index.min()} to {full_data.index.max()}")
    
    # Load train/val split info to use indices
    train_data = pd.read_csv("AAPL_stock_data_train.csv", index_col=0, parse_dates=True)
    val_data = pd.read_csv("AAPL_stock_data_val.csv", index_col=0, parse_dates=True)
    
    print(f"\n✓ Training set info loaded")
    print(f"  Shape: {train_data.shape}")
    print(f"  Date range: {train_data.index.min()} to {train_data.index.max()}")
    
    print(f"\n✓ Validation set info loaded")
    print(f"  Shape: {val_data.shape}")
    print(f"  Date range: {val_data.index.min()} to {val_data.index.max()}")
    
except FileNotFoundError as e:
    print(f"✗ Error: {str(e)}")
    print("  Please ensure AAPL_stock_data_with_indicators.csv exists")
    exit()

print("\n" + "=" * 80)
print("STEP 2: SPLITTING INTO FEATURES (X) AND TARGET (Y)")
print("-" * 80)

# Get close price column
close_col = None
for col in full_data.columns:
    if 'close' in col.lower():
        close_col = col
        break

if close_col is None:
    print("✗ Error: Close price column not found!")
    exit()

print(f"✓ Target column identified: {close_col}")

# Create target: next day close price
target_series = full_data[close_col].shift(-1)  # Next day's close
target_series = target_series[:-1]  # Remove last NaN

# Create feature matrix (all columns except target)
feature_columns = [col for col in full_data.columns]

# Align indices for train/val split
train_indices = train_data.index
val_indices = val_data.index

# Create X and y for training set
X_train = full_data.loc[train_indices, feature_columns].copy()
y_train = target_series.loc[train_indices].copy()

# Create X and y for validation set
X_val = full_data.loc[val_indices, feature_columns].copy()
y_val = target_series.loc[val_indices].copy()

print(f"✓ Feature columns identified: {len(feature_columns)} features")
print(f"\n✓ Training features shape: {X_train.shape}")
print(f"✓ Training target shape: {y_train.shape}")
print(f"✓ Validation features shape: {X_val.shape}")
print(f"✓ Validation target shape: {y_val.shape}")

# Check for missing values
missing_train = X_train.isnull().sum().sum()
missing_val = X_val.isnull().sum().sum()

print(f"\n✓ Missing values in training features: {missing_train}")
print(f"✓ Missing values in validation features: {missing_val}")

if missing_train > 0 or missing_val > 0:
    print("  ⚠ Warning: Missing values detected. Handling...")
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_val.mean())

# =====================================================
# STEP 3: FEATURE SCALING
# =====================================================
print("\n" + "=" * 80)
print("STEP 3: FEATURE SCALING (STANDARDIZATION)")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"✓ Features scaled using StandardScaler")
print(f"  Training set - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
print(f"  Validation set - Mean: {X_val_scaled.mean():.6f}, Std: {X_val_scaled.std():.6f}")

# Convert back to DataFrame for better handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns, index=X_val.index)

# =====================================================
# STEP 4: INITIALIZE BASELINE MODELS
# =====================================================
print("\n" + "=" * 80)
print("STEP 4: INITIALIZING BASELINE MODELS")
print("-" * 80)

# Model 1: Linear Regression
print("\n🔹 MODEL 1: LINEAR REGRESSION")
print("   Parameters: Default (no hyperparameter tuning needed)")
lr_model = LinearRegression()
print("   ✓ Initialized")

# Model 2: Random Forest
print("\n🔹 MODEL 2: RANDOM FOREST REGRESSOR")
print("   Parameters:")
print("      - n_estimators: 200 (increased from 100)")
print("      - max_depth: 15")
print("      - min_samples_split: 5")
print("      - min_samples_leaf: 2")
print("      - max_features: 'sqrt'")
print("      - random_state: 42")
print("      - n_jobs: -1 (parallel processing)")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
print("   ✓ Initialized")

# Model 3: Support Vector Regression
print("\n🔹 MODEL 3: SUPPORT VECTOR REGRESSION (SVR)")
print("   Parameters:")
print("      - kernel: 'rbf' (Radial Basis Function)")
print("      - C: 100 (Regularization strength)")
print("      - epsilon: 0.1 (Margin of tolerance)")
print("      - gamma: 'scale' (Kernel coefficient)")
svr_model = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,
    gamma='scale'
)
print("   ✓ Initialized")

# Store models in dictionary
models = {
    'Linear_Regression': {
        'model': lr_model,
        'name': 'Linear Regression',
        'hyperparameters': {}
    },
    'Random_Forest': {
        'model': rf_model,
        'name': 'Random Forest Regressor',
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }
    },
    'SVR': {
        'model': svr_model,
        'name': 'Support Vector Regression',
        'hyperparameters': {
            'kernel': 'rbf',
            'C': 100,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
    }
}

print(f"\n✓ All {len(models)} baseline models initialized and ready for training")

# =====================================================
# STEP 5: TRAIN BASELINE MODELS
# =====================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING BASELINE MODELS")
print("-" * 80)

trained_models = {}
training_times = {}

for model_key, model_info in models.items():
    print(f"\n⏱️  Training {model_info['name']}...")
    
    import time
    start_time = time.time()
    
    try:
        model = model_info['model']
        model.fit(X_train_scaled, y_train)
        
        end_time = time.time()
        training_time = end_time - start_time
        training_times[model_key] = training_time
        
        trained_models[model_key] = model
        
        print(f"   ✓ Training completed in {training_time:.2f} seconds")
        print(f"   ✓ Model trained on {len(X_train_scaled)} samples")
        
    except Exception as e:
        print(f"   ✗ Error training model: {str(e)}")

print(f"\n✓ Successfully trained {len(trained_models)}/{len(models)} models")

# =====================================================
# STEP 6: EVALUATE ON VALIDATION SET
# =====================================================
print("\n" + "=" * 80)
print("STEP 6: EVALUATING MODELS ON VALIDATION SET")
print("-" * 80)

evaluation_results = {}

for model_key, model in trained_models.items():
    model_name = models[model_key]['name']
    print(f"\n📊 Evaluating {model_name}")
    print("-" * 80)
    
    # Predictions on validation set
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    mape = mean_absolute_percentage_error(y_val, y_val_pred)
    
    # Additional metrics
    residuals = y_val - y_val_pred
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    max_error = np.abs(residuals).max()
    
    evaluation_results[model_key] = {
        'model_name': model_name,
        'predictions': y_val_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'max_error': max_error,
        'residuals': residuals,
        'training_time': training_times[model_key]
    }
    
    # Print metrics
    print(f"   Metrics:")
    print(f"     R² Score:           {r2:.6f}")
    print(f"     RMSE (Root MSE):    ${rmse:.2f}")
    print(f"     MAE (Mean Abs Error): ${mae:.2f}")
    print(f"     MAPE (%):            {mape:.4f}%")
    print(f"     Mean Residual:       ${mean_residual:.2f}")
    print(f"     Std Residual:        ${std_residual:.2f}")
    print(f"     Max Error:           ${max_error:.2f}")
    print(f"     Training Time:       {training_times[model_key]:.2f}s")
    
    # Directional accuracy (for classification perspective)
    actual_direction = np.diff(np.sign(np.diff(y_val.values)))
    pred_direction = np.diff(np.sign(np.diff(y_val_pred)))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    print(f"     Directional Acc (%): {directional_accuracy:.2f}%")

# =====================================================
# STEP 7: MODEL COMPARISON AND RANKING
# =====================================================
print("\n" + "=" * 80)
print("STEP 7: MODEL COMPARISON AND RANKING")
print("-" * 80)

# Create comparison DataFrame
comparison_data = {
    'Model': [evaluation_results[k]['model_name'] for k in evaluation_results.keys()],
    'R² Score': [evaluation_results[k]['r2'] for k in evaluation_results.keys()],
    'RMSE ($)': [evaluation_results[k]['rmse'] for k in evaluation_results.keys()],
    'MAE ($)': [evaluation_results[k]['mae'] for k in evaluation_results.keys()],
    'MAPE (%)': [evaluation_results[k]['mape'] for k in evaluation_results.keys()],
    'Training Time (s)': [evaluation_results[k]['training_time'] for k in evaluation_results.keys()]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n📋 PERFORMANCE COMPARISON TABLE:")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# Ranking
print("\n🏆 MODEL RANKINGS:")
print("-" * 80)

# Best R² score
best_r2_idx = comparison_df['R² Score'].idxmax()
print(f"\n1. Best R² Score:")
print(f"   {comparison_df.loc[best_r2_idx, 'Model']}: {comparison_df.loc[best_r2_idx, 'R² Score']:.6f}")

# Best RMSE
best_rmse_idx = comparison_df['RMSE ($)'].idxmin()
print(f"\n2. Lowest RMSE:")
print(f"   {comparison_df.loc[best_rmse_idx, 'Model']}: ${comparison_df.loc[best_rmse_idx, 'RMSE ($)']:.2f}")

# Best MAE
best_mae_idx = comparison_df['MAE ($)'].idxmin()
print(f"\n3. Lowest MAE:")
print(f"   {comparison_df.loc[best_mae_idx, 'Model']}: ${comparison_df.loc[best_mae_idx, 'MAE ($)']:.2f}")

# Fastest training
fastest_idx = comparison_df['Training Time (s)'].idxmin()
print(f"\n4. Fastest Training:")
print(f"   {comparison_df.loc[fastest_idx, 'Model']}: {comparison_df.loc[fastest_idx, 'Training Time (s)']:.2f}s")

# =====================================================
# STEP 8: SAVE TRAINED MODELS
# =====================================================
print("\n" + "=" * 80)
print("STEP 8: SAVING TRAINED MODELS")
print("-" * 80)

# Create models directory if it doesn't exist
import os
os.makedirs('trained_models', exist_ok=True)

for model_key, model in trained_models.items():
    model_name = models[model_key]['name']
    filepath = f'trained_models/model_{model_key}.pkl'
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved: {model_name} -> {filepath}")
    except Exception as e:
        print(f"✗ Error saving {model_name}: {str(e)}")

# Save scaler
try:
    with open('trained_models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: StandardScaler -> trained_models/scaler.pkl")
except Exception as e:
    print(f"✗ Error saving scaler: {str(e)}")

# =====================================================
# STEP 9: GENERATE DETAILED REPORT
# =====================================================
print("\n" + "=" * 80)
print("STEP 9: GENERATING DETAILED REPORT")
print("-" * 80)

report_text = f"""
{'='*80}
TASK 3.2: BASELINE MODEL TRAINING REPORT
Stock Price Forecasting - AAPL Stock
{'='*80}

EXECUTION DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'-'*80}
DATASET SUMMARY:
{'-'*80}
Training Set:
  - Samples: {len(X_train)}
  - Date Range: {train_data.index.min()} to {train_data.index.max()}
  - Features: {X_train.shape[1]}

Validation Set:
  - Samples: {len(X_val)}
  - Date Range: {val_data.index.min()} to {val_data.index.max()}
  - Features: {X_val.shape[1]}

Target Variable:
  - Column: {close_col}
  - Type: Continuous (Regression)
  - Training Range: ${y_train.min():.2f} - ${y_train.max():.2f}
  - Validation Range: ${y_val.min():.2f} - ${y_val.max():.2f}
  - Training Mean: ${y_train.mean():.2f}
  - Validation Mean: ${y_val.mean():.2f}

{'-'*80}
FEATURE SCALING:
{'-'*80}
Method: StandardScaler
- Training Set Mean: {X_train_scaled.values.mean():.6f}
- Training Set Std: {X_train_scaled.values.std():.6f}
- Validation Set Mean: {X_val_scaled.values.mean():.6f}
- Validation Set Std: {X_val_scaled.values.std():.6f}

{'-'*80}
MODEL CONFIGURATIONS:
{'-'*80}

1. LINEAR REGRESSION
   - Type: Linear regression model
   - Hyperparameters: Default
   - Training Time: {training_times.get('Linear_Regression', 'N/A'):.2f}s
   
2. RANDOM FOREST REGRESSOR
   - Type: Ensemble of decision trees
   - Hyperparameters:
     * n_estimators: 200
     * max_depth: 15
     * min_samples_split: 5
     * min_samples_leaf: 2
     * max_features: sqrt
   - Training Time: {training_times.get('Random_Forest', 'N/A'):.2f}s

3. SUPPORT VECTOR REGRESSION (SVR)
   - Type: Support vector machine (regression)
   - Hyperparameters:
     * kernel: rbf
     * C: 100
     * epsilon: 0.1
     * gamma: scale
   - Training Time: {training_times.get('SVR', 'N/A'):.2f}s

{'-'*80}
VALIDATION SET PERFORMANCE METRICS:
{'-'*80}
"""

for model_key in evaluation_results.keys():
    results = evaluation_results[model_key]
    report_text += f"""
{results['model_name'].upper()}:
  Regression Metrics:
    - R² Score:           {results['r2']:.6f}
    - RMSE (Root MSE):    ${results['rmse']:.2f}
    - MAE (Mean Abs Error): ${results['mae']:.2f}
    - MAPE (%):            {results['mape']:.4f}%
  
  Error Analysis:
    - Mean Residual:       ${results['mean_residual']:.2f}
    - Std Residual:        ${results['std_residual']:.2f}
    - Max Error:           ${results['max_error']:.2f}
    - Training Time:       {results['training_time']:.2f}s

"""

report_text += f"""
{'-'*80}
COMPARATIVE SUMMARY:
{'-'*80}
{comparison_df.to_string(index=False)}

{'-'*80}
KEY FINDINGS:
{'-'*80}
1. Best Overall Performance: {comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']}
   - R² Score: {comparison_df['R² Score'].max():.6f}
   
2. Most Accurate Predictions: {comparison_df.loc[comparison_df['MAE ($)'].idxmin(), 'Model']}
   - MAE: ${comparison_df['MAE ($)'].min():.2f}
   
3. Fastest Training: {comparison_df.loc[comparison_df['Training Time (s)'].idxmin(), 'Model']}
   - Time: {comparison_df['Training Time (s)'].min():.2f}s

4. Best for Production:
   - Linear Regression: Fastest, most interpretable
   - Random Forest: Balanced performance and speed
   - SVR: Best for complex patterns (if R² permits)

{'-'*80}
NEXT STEPS (Task 3.3 & Beyond):
{'-'*80}
1. Deep Learning Models (LSTM, GRU) implementation
2. Hyperparameter tuning via GridSearchCV/RandomSearchCV
3. Cross-validation for robustness assessment
4. Feature importance analysis
5. Model ensemble methods
6. Final model selection and deployment

{'='*80}
REPORT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

# Save report
with open('baseline_models_training_report.txt', 'w') as f:
    f.write(report_text)

print("✓ Report saved: baseline_models_training_report.txt")

# =====================================================
# STEP 10: SAVE EVALUATION RESULTS AS JSON
# =====================================================
print("\n" + "=" * 80)
print("STEP 10: SAVING EVALUATION RESULTS")
print("-" * 80)

# Prepare JSON-serializable results
json_results = {
    'execution_date': datetime.now().isoformat(),
    'dataset': {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'total_features': X_train.shape[1],
        'train_date_range': {
            'start': str(train_data.index.min()),
            'end': str(train_data.index.max())
        },
        'val_date_range': {
            'start': str(val_data.index.min()),
            'end': str(val_data.index.max())
        }
    },
    'target_statistics': {
        'training': {
            'min': float(y_train.min()),
            'max': float(y_train.max()),
            'mean': float(y_train.mean()),
            'std': float(y_train.std())
        },
        'validation': {
            'min': float(y_val.min()),
            'max': float(y_val.max()),
            'mean': float(y_val.mean()),
            'std': float(y_val.std())
        }
    },
    'models': {}
}

for model_key in evaluation_results.keys():
    results = evaluation_results[model_key]
    json_results['models'][model_key] = {
        'model_name': results['model_name'],
        'hyperparameters': models[model_key]['hyperparameters'],
        'metrics': {
            'r2_score': float(results['r2']),
            'rmse': float(results['rmse']),
            'mae': float(results['mae']),
            'mape': float(results['mape']),
            'mean_residual': float(results['mean_residual']),
            'std_residual': float(results['std_residual']),
            'max_error': float(results['max_error'])
        },
        'training_time_seconds': float(results['training_time'])
    }

with open('baseline_models_results.json', 'w') as f:
    json.dump(json_results, f, indent=2)

print("✓ Results saved: baseline_models_results.json")

# Save comparison to CSV
comparison_df.to_csv('baseline_models_comparison.csv', index=False)
print("✓ Comparison saved: baseline_models_comparison.csv")

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n" + "=" * 80)
print("TASK 3.2 COMPLETE: BASELINE MODEL TRAINING")
print("=" * 80)

print("\n✅ DELIVERABLES:")
print("   1. ✓ Trained 3 baseline models")
print("   2. ✓ Evaluated on validation set")
print("   3. ✓ Generated performance metrics")
print("   4. ✓ Saved trained models (pkl files)")
print("   5. ✓ Generated detailed report")
print("   6. ✓ Saved JSON results")

print("\n📊 MODEL PERFORMANCE SUMMARY:")
print("-" * 80)
for idx, row in comparison_df.iterrows():
    print(f"   {idx+1}. {row['Model']:25s} | R²: {row['R² Score']:8.6f} | RMSE: ${row['RMSE ($)']:8.2f} | MAE: ${row['MAE ($)']:7.2f}")

print("\n📁 FILES GENERATED:")
print("   ✓ trained_models/model_Linear_Regression.pkl")
print("   ✓ trained_models/model_Random_Forest.pkl")
print("   ✓ trained_models/model_SVR.pkl")
print("   ✓ trained_models/scaler.pkl")
print("   ✓ baseline_models_training_report.txt")
print("   ✓ baseline_models_results.json")
print("   ✓ baseline_models_comparison.csv")

print("\n📅 TIMELINE:")
print("   Feb 12-18: ✓ Task 3.2 - Baseline Model Training (COMPLETED)")
print("   Feb 19-25: ○ Task 3.3 - Deep Learning Models (NEXT)")

print("\n" + "=" * 80)
print("Ready for next phase: Deep Learning model implementation")
print("=" * 80)
