"""
Task 3.3: Training Advanced Models (LSTM and GRU)

Purpose: Train LSTM and GRU neural networks for sequential stock price prediction
Output: Trained models, training history, performance metrics, comparison reports

Timeline: Feb 19-25, 2026
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "="*80)
print("TASK 3.3: Training Advanced Models (LSTM and GRU)")
print("="*80)

# ============================================================================
# STEP 1: Load Data and Create Sequences
# ============================================================================
print("\n[STEP 1] Loading data and creating sequences...")
start_time = time.time()

# Load full data
full_data = pd.read_csv('AAPL_stock_data_with_indicators.csv', index_col='Date')
print(f"  ✓ Loaded {len(full_data)} total records")

# Load train/val split info
train_data = pd.read_csv('AAPL_stock_data_train.csv', index_col='Date')
val_data = pd.read_csv('AAPL_stock_data_val.csv', index_col='Date')

print(f"  ✓ Training set: {len(train_data)} samples")
print(f"  ✓ Validation set: {len(val_data)} samples")

# Get feature columns (22 features)
feature_cols = [col for col in full_data.columns if col not in ['Close_AAPL', 'Target_Close_Price']]
target_col = 'Close_AAPL'

print(f"  ✓ Features: {len(feature_cols)} ({', '.join(feature_cols[:5])}...)")

# Create target column (next-day close price)
full_data['Target_Close_Price'] = full_data[target_col].shift(-1)
train_data['Target_Close_Price'] = train_data[target_col].shift(-1)
val_data['Target_Close_Price'] = val_data[target_col].shift(-1)

# Remove last row (NaN target)
train_data = train_data[:-1]
val_data = val_data[:-1]

# Get X and y
X_train = train_data[feature_cols].values
y_train = train_data['Target_Close_Price'].values
X_val = val_data[feature_cols].values
y_val = val_data['Target_Close_Price'].values

print(f"\n  Training shape before scaling: X={X_train.shape}, y={y_train.shape}")
print(f"  Validation shape before scaling: X={X_val.shape}, y={y_val.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"  ✓ Scaled features (mean={X_train_scaled.mean():.2f}, std={X_train_scaled.std():.2f})")

# ============================================================================
# STEP 2: Create 3D Sequences for LSTM/GRU
# ============================================================================
print("\n[STEP 2] Creating 3D sequences for LSTM/GRU...")

def create_sequences(data, targets, timesteps=30):
    """
    Reshape data into 3D sequences: (samples, timesteps, features)
    Each sample contains 30 days of historical data to predict the next day
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(data) - timesteps):
        X_seq.append(data[i:i+timesteps])
        y_seq.append(targets[i+timesteps])
    
    return np.array(X_seq), np.array(y_seq)

timesteps = 30
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, timesteps)

print(f"  ✓ Training sequences: {X_train_seq.shape} (samples={X_train_seq.shape[0]}, timesteps={X_train_seq.shape[1]}, features={X_train_seq.shape[2]})")
print(f"  ✓ Validation sequences: {X_val_seq.shape}")
print(f"  ✓ Timesteps (lookback window): {timesteps} days")

# ============================================================================
# STEP 3: Build LSTM Model
# ============================================================================
print("\n[STEP 3] Building LSTM model...")

lstm_model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, 
         input_shape=(timesteps, X_train_seq.shape[2]), name='LSTM_Layer1'),
    Dropout(0.2, name='Dropout1'),
    LSTM(32, activation='relu', name='LSTM_Layer2'),
    Dropout(0.2, name='Dropout2'),
    Dense(16, activation='relu', name='Dense1'),
    Dense(1, name='Output')
], name='LSTM_Model')

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("  ✓ LSTM Architecture:")
lstm_model.summary()

# ============================================================================
# STEP 4: Build GRU Model
# ============================================================================
print("\n[STEP 4] Building GRU model...")

gru_model = Sequential([
    GRU(64, activation='relu', return_sequences=True,
        input_shape=(timesteps, X_train_seq.shape[2]), name='GRU_Layer1'),
    Dropout(0.2, name='Dropout1'),
    GRU(32, activation='relu', name='GRU_Layer2'),
    Dropout(0.2, name='Dropout2'),
    Dense(16, activation='relu', name='Dense1'),
    Dense(1, name='Output')
], name='GRU_Model')

gru_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("  ✓ GRU Architecture:")
gru_model.summary()

# ============================================================================
# STEP 5: Train LSTM Model
# ============================================================================
print("\n[STEP 5] Training LSTM model (20 epochs, batch_size=32)...")

lstm_start = time.time()
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
lstm_training_time = time.time() - lstm_start

print(f"\n  ✓ LSTM Training completed in {lstm_training_time:.2f} seconds")
print(f"  ✓ Training loss: {lstm_history.history['loss'][-1]:.6f}")
print(f"  ✓ Validation loss: {lstm_history.history['val_loss'][-1]:.6f}")

# ============================================================================
# STEP 6: Train GRU Model
# ============================================================================
print("\n[STEP 6] Training GRU model (20 epochs, batch_size=32)...")

gru_start = time.time()
gru_history = gru_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
gru_training_time = time.time() - gru_start

print(f"\n  ✓ GRU Training completed in {gru_training_time:.2f} seconds")
print(f"  ✓ Training loss: {gru_history.history['loss'][-1]:.6f}")
print(f"  ✓ Validation loss: {gru_history.history['val_loss'][-1]:.6f}")

# ============================================================================
# STEP 7: Evaluate Models on Validation Set
# ============================================================================
print("\n[STEP 7] Evaluating models on validation set...")

# LSTM predictions
lstm_pred_val = lstm_model.predict(X_val_seq, verbose=0)
lstm_rmse = np.sqrt(mean_squared_error(y_val_seq, lstm_pred_val))
lstm_mae = mean_absolute_error(y_val_seq, lstm_pred_val)
lstm_r2 = r2_score(y_val_seq, lstm_pred_val)
lstm_mape = np.mean(np.abs((y_val_seq - lstm_pred_val.flatten()) / y_val_seq)) * 100

print(f"\n  LSTM Results:")
print(f"    • RMSE: ${lstm_rmse:.2f}")
print(f"    • MAE: ${lstm_mae:.2f}")
print(f"    • R²: {lstm_r2:.4f}")
print(f"    • MAPE: {lstm_mape:.2f}%")

# GRU predictions
gru_pred_val = gru_model.predict(X_val_seq, verbose=0)
gru_rmse = np.sqrt(mean_squared_error(y_val_seq, gru_pred_val))
gru_mae = mean_absolute_error(y_val_seq, gru_pred_val)
gru_r2 = r2_score(y_val_seq, gru_pred_val)
gru_mape = np.mean(np.abs((y_val_seq - gru_pred_val.flatten()) / y_val_seq)) * 100

print(f"\n  GRU Results:")
print(f"    • RMSE: ${gru_rmse:.2f}")
print(f"    • MAE: ${gru_mae:.2f}")
print(f"    • R²: {gru_r2:.4f}")
print(f"    • MAPE: {gru_mape:.2f}%")

# Compare with baseline (Linear Regression: R²=0.9316)
baseline_r2 = 0.9316
print(f"\n  Baseline Comparison (Linear Regression R²={baseline_r2}):")
print(f"    • LSTM vs Baseline: R² difference = {lstm_r2 - baseline_r2:+.4f}")
print(f"    • GRU vs Baseline: R² difference = {gru_r2 - baseline_r2:+.4f}")

# ============================================================================
# STEP 8: Save Models
# ============================================================================
print("\n[STEP 8] Saving trained models...")

# Create trained_models directory if it doesn't exist
import os
os.makedirs('trained_models', exist_ok=True)

# Save models
lstm_model.save('trained_models/model_LSTM.keras')
gru_model.save('trained_models/model_GRU.keras')
print(f"  ✓ Saved LSTM model to trained_models/model_LSTM.keras")
print(f"  ✓ Saved GRU model to trained_models/model_GRU.keras")

# Save scaler for future inference
with open('trained_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved scaler to trained_models/scaler.pkl")

# ============================================================================
# STEP 9: Export Training History
# ============================================================================
print("\n[STEP 9] Exporting training history and logs...")

history_dict = {
    'LSTM': {
        'epochs': len(lstm_history.history['loss']),
        'training_loss': [float(x) for x in lstm_history.history['loss']],
        'validation_loss': [float(x) for x in lstm_history.history['val_loss']],
        'training_mae': [float(x) for x in lstm_history.history['mae']],
        'validation_mae': [float(x) for x in lstm_history.history['val_mae']],
        'training_time_seconds': lstm_training_time,
        'final_metrics': {
            'rmse': float(lstm_rmse),
            'mae': float(lstm_mae),
            'r2': float(lstm_r2),
            'mape': float(lstm_mape)
        }
    },
    'GRU': {
        'epochs': len(gru_history.history['loss']),
        'training_loss': [float(x) for x in gru_history.history['loss']],
        'validation_loss': [float(x) for x in gru_history.history['val_loss']],
        'training_mae': [float(x) for x in gru_history.history['mae']],
        'validation_mae': [float(x) for x in gru_history.history['val_mae']],
        'training_time_seconds': gru_training_time,
        'final_metrics': {
            'rmse': float(gru_rmse),
            'mae': float(gru_mae),
            'r2': float(gru_r2),
            'mape': float(gru_mape)
        }
    },
    'Metadata': {
        'timesteps': timesteps,
        'features': len(feature_cols),
        'training_samples': X_train_seq.shape[0],
        'validation_samples': X_val_seq.shape[0],
        'batch_size': 32,
        'optimizer': 'Adam (lr=0.001)',
        'loss_function': 'MSE'
    }
}

with open('deep_learning_training_history.json', 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"  ✓ Exported training history to deep_learning_training_history.json")

# ============================================================================
# STEP 10: Generate Training Report
# ============================================================================
print("\n[STEP 10] Generating training report...")

report = f"""
{'='*80}
TASK 3.3: TRAINING ADVANCED MODELS (LSTM AND GRU) - COMPLETE REPORT
{'='*80}

Project: AI Trading System - AAPL Stock Price Prediction
Date: February 19-25, 2026
Task: Train LSTM and GRU neural networks for sequence-to-scalar prediction

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

✓ LSTM Model: Successfully trained
✓ GRU Model: Successfully trained
✓ Performance: Both models evaluated against baseline (Linear Regression R²=0.9316)
✓ Best Model: {('LSTM' if lstm_r2 > gru_r2 else 'GRU')} with R²={max(lstm_r2, gru_r2):.4f}

{'='*80}
DATA CONFIGURATION
{'='*80}

Source: AAPL_stock_data_with_indicators.csv
Features: 22 engineered features (price, trend, momentum, volatility)
Target: Next-day close price
Training Samples: {X_train_seq.shape[0]} (after sequence creation)
Validation Samples: {X_val_seq.shape[0]} (after sequence creation)
Timesteps (Lookback Window): {timesteps} days
Sequence Shape: (samples, timesteps={timesteps}, features={X_train_seq.shape[2]})

Feature Scaling:
  - Method: StandardScaler (fitted on training data)
  - Mean: {X_train_scaled.mean():.6f}
  - Std Dev: {X_train_scaled.std():.6f}

{'='*80}
LSTM MODEL ARCHITECTURE
{'='*80}

Sequential Model: LSTM_Model
Layer Configuration:
  1. LSTM Layer 1: 64 units, activation='relu', return_sequences=True
  2. Dropout Layer 1: rate=0.2
  3. LSTM Layer 2: 32 units, activation='relu'
  4. Dropout Layer 2: rate=0.2
  5. Dense Layer 1: 16 units, activation='relu'
  6. Output Layer: 1 unit (regression)

Compiler Settings:
  - Optimizer: Adam (learning rate=0.001)
  - Loss Function: Mean Squared Error (MSE)
  - Metrics: Mean Absolute Error (MAE)

Training Configuration:
  - Epochs: 20
  - Batch Size: 32
  - Early Stopping: Yes (patience=5 on val_loss)
  - Training Time: {lstm_training_time:.2f} seconds

{'='*80}
GRU MODEL ARCHITECTURE
{'='*80}

Sequential Model: GRU_Model
Layer Configuration:
  1. GRU Layer 1: 64 units, activation='relu', return_sequences=True
  2. Dropout Layer 1: rate=0.2
  3. GRU Layer 2: 32 units, activation='relu'
  4. Dropout Layer 2: rate=0.2
  5. Dense Layer 1: 16 units, activation='relu'
  6. Output Layer: 1 unit (regression)

Compiler Settings:
  - Optimizer: Adam (learning rate=0.001)
  - Loss Function: Mean Squared Error (MSE)
  - Metrics: Mean Absolute Error (MAE)

Training Configuration:
  - Epochs: 20
  - Batch Size: 32
  - Early Stopping: Yes (patience=5 on val_loss)
  - Training Time: {gru_training_time:.2f} seconds

{'='*80}
VALIDATION SET PERFORMANCE
{'='*80}

LSTM METRICS:
  Root Mean Squared Error (RMSE): ${lstm_rmse:.2f}
  Mean Absolute Error (MAE): ${lstm_mae:.2f}
  R² Score: {lstm_r2:.4f}
  Mean Absolute Percentage Error (MAPE): {lstm_mape:.2f}%

  Final Training Loss: {lstm_history.history['loss'][-1]:.6f}
  Final Validation Loss: {lstm_history.history['val_loss'][-1]:.6f}
  Improvement (Val): {((lstm_history.history['val_loss'][0] - lstm_history.history['val_loss'][-1]) / lstm_history.history['val_loss'][0] * 100):.2f}%

GRU METRICS:
  Root Mean Squared Error (RMSE): ${gru_rmse:.2f}
  Mean Absolute Error (MAE): ${gru_mae:.2f}
  R² Score: {gru_r2:.4f}
  Mean Absolute Percentage Error (MAPE): {gru_mape:.2f}%

  Final Training Loss: {gru_history.history['loss'][-1]:.6f}
  Final Validation Loss: {gru_history.history['val_loss'][-1]:.6f}
  Improvement (Val): {((gru_history.history['val_loss'][0] - gru_history.history['val_loss'][-1]) / gru_history.history['val_loss'][0] * 100):.2f}%

{'='*80}
COMPARISON WITH BASELINE
{'='*80}

Baseline Model (Task 3.2): Linear Regression
  R² Score: {baseline_r2:.4f}
  RMSE: $2.32
  MAE: $1.74

LSTM vs Baseline:
  R² Difference: {lstm_r2 - baseline_r2:+.4f}
  RMSE Difference: ${lstm_rmse - 2.32:+.2f}
  Performance: {'✓ BEATS BASELINE' if lstm_r2 > baseline_r2 else '✗ Below baseline'}

GRU vs Baseline:
  R² Difference: {gru_r2 - baseline_r2:+.4f}
  RMSE Difference: ${gru_rmse - 2.32:+.2f}
  Performance: {'✓ BEATS BASELINE' if gru_r2 > baseline_r2 else '✗ Below baseline'}

{'='*80}
TRAINING EFFICIENCY COMPARISON
{'='*80}

Model           Training Time    Epochs   Samples/Sec
LSTM            {lstm_training_time:6.2f}s        {len(lstm_history.history['loss']):2d}      {X_train_seq.shape[0] / lstm_training_time:>10.1f}
GRU             {gru_training_time:6.2f}s        {len(gru_history.history['loss']):2d}      {X_train_seq.shape[0] / gru_training_time:>10.1f}

GRU vs LSTM:
  Time Difference: {gru_training_time - lstm_training_time:+.2f}s ({'faster' if gru_training_time < lstm_training_time else 'slower'})
  Percentage: {((gru_training_time - lstm_training_time) / lstm_training_time * 100):+.1f}%

{'='*80}
KEY FINDINGS
{'='*80}

1. LSTM Performance:
   - Achieved R² = {lstm_r2:.4f} ({('✓ BEATS' if lstm_r2 > baseline_r2 else '✗ Below')} baseline of 0.9316)
   - Prediction error: ${lstm_mae:.2f} per day (MAE)
   - Strong convergence with stable validation loss

2. GRU Performance:
   - Achieved R² = {gru_r2:.4f} ({('✓ BEATS' if gru_r2 > baseline_r2 else '✗ Below')} baseline of 0.9316)
   - Prediction error: ${gru_mae:.2f} per day (MAE)
   - Training efficiency: {('✓ Faster' if gru_training_time < lstm_training_time else '✗ Slower')} than LSTM

3. Model Comparison:
   - Best Model: {('LSTM' if lstm_r2 > gru_r2 else 'GRU')} (R²={max(lstm_r2, gru_r2):.4f})
   - Margin: {abs(lstm_r2 - gru_r2):.4f} (absolute difference)
   - Winner Reason: {'Better accuracy' if (lstm_r2 > gru_r2 and lstm_r2 > baseline_r2) else 'Computational efficiency' if gru_training_time < lstm_training_time else 'Comparable performance'}

4. Baseline Comparison:
   - Linear Regression remains strong (R²=0.9316)
   - Deep learning models: {('Surpass' if max(lstm_r2, gru_r2) > baseline_r2 else 'Do not surpass')} baseline
   - Complexity-benefit tradeoff: {'Justified improvement' if max(lstm_r2, gru_r2) > baseline_r2 else 'Simple model preferred'}

5. Early Stopping Impact:
   - LSTM stopped at epoch {len(lstm_history.history['loss'])} (target: 20)
   - GRU stopped at epoch {len(gru_history.history['loss'])} (target: 20)
   - Prevented overfitting on {'training' if len(lstm_history.history['loss']) < 20 else 'validation'} set

{'='*80}
SAVED ARTIFACTS
{'='*80}

Models:
  ✓ trained_models/model_LSTM.keras
  ✓ trained_models/model_GRU.keras
  ✓ trained_models/scaler.pkl

History & Logs:
  ✓ deep_learning_training_history.json
  ✓ deep_learning_models_report.txt (this file)

Next Steps:
  ✓ Generate visualizations (loss curves, predictions)
  ✓ Test on held-out test set
  ✓ Final model selection based on all metrics
  ✓ Production deployment preparation

{'='*80}
RECOMMENDATIONS
{'='*80}

1. Model Selection for Inference:
   - Recommended: {('LSTM' if lstm_r2 > gru_r2 else 'GRU')} (best R²={max(lstm_r2, gru_r2):.4f})
   - Alternative: Linear Regression (simpler, nearly equal R²=0.9316)

2. Hyperparameter Optimization:
   - Increase LSTM/GRU units if accuracy needed
   - Experiment with different timesteps (15, 45, 60)
   - Try ensemble: average LSTM + GRU predictions

3. For Production Deployment:
   - Quantize models for mobile/edge devices
   - Implement real-time retraining pipeline
   - Monitor prediction drift over time

4. Future Experiments:
   - Attention mechanisms (Transformer architecture)
   - Bidirectional LSTM/GRU
   - Multi-step forecasting (predict next 5-7 days)
   - Multivariate forecasting (volume, other assets)

{'='*80}
CONCLUSION
{'='*80}

Task 3.3 successfully completed. Both LSTM and GRU models trained and evaluated
on the AAPL validation set. {('LSTM' if lstm_r2 > gru_r2 else 'GRU')} achieved the best performance with R²={max(lstm_r2, gru_r2):.4f}.

{'✓ DEEP LEARNING MODELS BEAT BASELINE' if max(lstm_r2, gru_r2) > baseline_r2 else '✗ Baseline (Linear Regression) still optimal'}

Next Phase: Task 3.4 - Test Set Evaluation and Final Model Selection

{'='*80}
"""

with open('deep_learning_models_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✓ Generated training report: deep_learning_models_report.txt")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ DELIVERABLES SUMMARY")
print("="*80)
print(f"\n1. ✓ Trained 2 neural networks (LSTM + GRU)")
print(f"2. ✓ Evaluated on validation set ({X_val_seq.shape[0]} samples)")
print(f"3. ✓ Generated performance metrics (RMSE, MAE, R², MAPE)")
print(f"4. ✓ Saved trained models (.keras files)")
print(f"5. ✓ Exported training history (JSON format)")
print(f"6. ✓ Generated comprehensive report\n")

print("="*80)
print("QUICK RESULTS")
print("="*80)
print(f"\nLSTM:  R²={lstm_r2:.4f}, RMSE=${lstm_rmse:.2f}, MAE=${lstm_mae:.2f} ({lstm_training_time:.2f}s)")
print(f"GRU:   R²={gru_r2:.4f}, RMSE=${gru_rmse:.2f}, MAE=${gru_mae:.2f} ({gru_training_time:.2f}s)")
print(f"\nBaseline (Linear Regression): R²=0.9316")
print(f"Best Model: {('LSTM' if lstm_r2 > gru_r2 else 'GRU')} with R²={max(lstm_r2, gru_r2):.4f}")
print(f"Status: {'✓ BEATS BASELINE' if max(lstm_r2, gru_r2) > baseline_r2 else '✗ Below baseline'}")

print("\n" + "="*80)
print("Next: Run 3_3_deep_learning_visualizations.py for charts")
print("="*80 + "\n")
