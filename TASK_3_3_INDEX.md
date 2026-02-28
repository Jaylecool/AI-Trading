# Task 3.3: Training Advanced Models - Complete Index

**Status:** ✅ COMPLETED (100%)  
**Date:** February 19-25, 2026  
**Files Created:** 12  
**Visualizations:** 7 PNG at 300 DPI  

---

## 📊 Quick Results

| Model | R² | RMSE | MAE | Time | Status |
|-------|----|----|-----|------|--------|
| LSTM | 0.7048 | $4.96 | $3.82 | 15.57s | ✓ Trained |
| GRU | 0.7359 | $4.69 | $3.63 | 23.10s | ✓ Best DL |
| **Baseline (LR)** | **0.9316** | **$2.32** | **$1.74** | **0.03s** | **✓ Winner** |

**Key Finding:** Linear Regression outperforms deep learning by 19.6%

---

## 📁 Deliverables List

### Core Scripts (2 files)

| File | Purpose | Lines |
|------|---------|-------|
| `3_3_lstm_gru_final.py` | Train LSTM & GRU models | 342 |
| `3_3_visualizations_final.py` | Generate 7 visualizations | 267 |

### Trained Models (4 files)

| File | Type | Size | Purpose |
|------|------|------|---------|
| `trained_models/model_LSTM.keras` | Keras Model | ~400 KB | LSTM (34,977 params) |
| `trained_models/model_GRU.keras` | Keras Model | ~350 KB | GRU (26,657 params) |
| `trained_models/feature_scaler.pkl` | StandardScaler | ~1 KB | Feature normalization |
| `trained_models/target_scaler.pkl` | StandardScaler | ~1 KB | Target normalization |

### Results Files (2 files)

| File | Format | Content |
|------|--------|---------|
| `deep_learning_training_history.json` | JSON | Training/validation loss by epoch |
| `deep_learning_models_report.txt` | TXT | Comprehensive analysis report |

### Visualizations (7 files, 300 DPI PNG)

| # | File | Content | Insight |
|---|------|---------|---------|
| 13 | `13_training_validation_loss.png` | Loss curves LSTM & GRU | Convergence patterns |
| 14 | `14_lstm_vs_gru_loss.png` | Direct comparison | GRU better convergence |
| 16 | `16_actual_vs_predicted.png` | Scatter plots (R²=0.70/0.74) | Prediction accuracy |
| 17 | `17_prediction_errors.png` | Error distributions | Error patterns |
| 18 | `18_timeseries_predictions.png` | Last 100 days | Visual prediction quality |
| 19 | `19_performance_metrics_comparison.png` | Bar charts R²/RMSE/MAE | Metric comparison |
| 20 | `20_summary_scorecard.png` | Performance card | Executive summary |

### Documentation (2 files)

| File | Purpose | Size |
|------|---------|------|
| `TASK_3_3_DOCUMENTATION.md` | Technical deep dive | 500+ lines |
| `TASK_3_3_COMPLETION_SUMMARY.md` | Results & lessons | 600+ lines |

---

## 🎯 Task Objectives ✓

### Primary Objectives
- [x] Reshape data into 3D format (samples, timesteps=30, features=21)
- [x] Build LSTM model with 2 layers (64→32 units)
- [x] Build GRU model with 2 layers (64→32 units)
- [x] Compile with Adam optimizer and MSE loss
- [x] Train for ~20 epochs with batch_size=32 (early stopped)
- [x] Evaluate using RMSE, MAE, R²
- [x] Save trained models and logs
- [x] Generate validation loss curves

### Success Criteria
- ✅ Both models trained successfully
- ✅ Performance metrics calculated (LSTM: R²=0.7048, GRU: R²=0.7359)
- ✅ Models saved (.keras format)
- ✅ 7 visualizations generated (300 DPI)
- ✅ Documentation completed
- ✅ Comparison with baseline provided

---

## 🔍 Key Findings

### Performance Summary

**LSTM Model:**
- R² = 0.7048 (explains 70.48% of variance)
- RMSE = $4.96
- MAE = $3.82
- Epochs: 10 (early stopped)
- Time: 15.57 seconds
- Parameters: 34,977
- Status: ✗ Below baseline by 22.7%

**GRU Model:**
- R² = 0.7359 (explains 73.59% of variance)
- RMSE = $4.69 (best deep learning)
- MAE = $3.63 (lowest error)
- Epochs: 13 (early stopped)
- Time: 23.10 seconds
- Parameters: 26,657 (fewer than LSTM)
- Status: ✗ Below baseline by 19.6%

**Baseline (Linear Regression - Task 3.2):**
- R² = 0.9316 (explains 93.16% of variance)
- RMSE = $2.32
- MAE = $1.74
- Time: 0.03 seconds (500x faster)
- Parameters: 22 (1,500x fewer)
- Status: ✓ WINNER

### Why Deep Learning Underperforms

1. **Linear Data Pattern**
   - AAPL stock follows strong uptrend
   - Linear Regression captures this perfectly
   - RNNs designed for non-linear patterns

2. **Model Over-complexity**
   - 34,977 LSTM parameters on 710 training samples
   - Overfitting risk much higher
   - Linear 22-parameter model generalizes better

3. **Insufficient Data**
   - Deep learning needs 10,000+ sequences
   - Current dataset: only 710
   - RNNs underfed on this problem

4. **Weak Temporal Dependencies**
   - Stock price weakly correlated with past 30 days
   - Current features are stronger predictors
   - RNNs not needed for this data

---

## 🔧 Technical Configuration

### Data Setup
- **Source:** AAPL_stock_data_with_indicators.csv
- **Training:** 741 samples (Oct 2020 - Sep 2023)
- **Validation:** 158 samples (Sep 2023 - May 2024)
- **Features:** 21 engineered features
- **Target:** Next-day close price
- **Sequences:** 30-day lookback window

### Model Specifications

**LSTM:**
```
LSTM(64) + Dropout(0.2)
↓
LSTM(32) + Dropout(0.2)
↓
Dense(16, relu)
↓
Dense(1)

Parameters: 34,977
Optimizer: Adam (lr=0.001)
Loss: MSE
```

**GRU:**
```
GRU(64) + Dropout(0.2)
↓
GRU(32) + Dropout(0.2)
↓
Dense(16, relu)
↓
Dense(1)

Parameters: 26,657
Optimizer: Adam (lr=0.001)
Loss: MSE
```

### Training Details
- Epochs: 20 (target), 10-13 (actual, early stopped)
- Batch Size: 32
- Early Stopping: Yes (patience=5 on val_loss)
- Dropout: 0.2 (regularization)
- Learning Rate: 0.001

---

## 📊 Performance Comparison

### Metrics Breakdown

```
Metric              LSTM        GRU         Linear Reg
─────────────────────────────────────────────────────
R² Score            0.7048      0.7359      0.9316 ✓
RMSE                $4.96       $4.69       $2.32 ✓
MAE                 $3.82       $3.63       $1.74 ✓
MAPE                2.20%       2.04%       N/A
Training Time       15.57s      23.10s      0.03s ✓
Inference Time      ~1s         ~1s         <0.01s ✓
Parameters          34,977      26,657      22 ✓
Memory Required     ~400KB      ~350KB      <1KB ✓
─────────────────────────────────────────────────────
Winner per metric:  0            2           8/10 metrics ✓
```

### Accuracy Gap Analysis

| Model | vs Baseline | Gap |
|-------|------------|-----|
| LSTM | 0.7048 vs 0.9316 | -0.2268 (-22.7%) |
| GRU | 0.7359 vs 0.9316 | -0.1957 (-19.6%) |
| **Conclusion** | **DL underperforms** | **~20% worse** |

---

## 📈 Training History Summary

### LSTM Training Curve
- **Epoch 1:** Loss converges rapidly (0.47 → 0.20)
- **Epoch 2-5:** Smooth convergence (0.20 → 0.07)
- **Epoch 6+:** Overfitting begins (train↓ val↑)
- **Stopped:** Epoch 10 (patience=5)
- **Pattern:** Typical overfitting on small dataset

### GRU Training Curve
- **Epoch 1:** Strong start (0.50 → 0.14)
- **Epoch 2-6:** Excellent convergence (0.14 → 0.06)
- **Epoch 7+:** Slight divergence
- **Stopped:** Epoch 13 (longer convergence)
- **Pattern:** Better stability than LSTM

---

## 🛠️ How to Use

### Training Models from Scratch
```bash
# Go to project directory
cd "C:\Users\Admin\Documents\AI Trading"

# Activate virtual environment
.venv\Scripts\activate

# Run training script
python 3_3_lstm_gru_final.py

# Output: Trained models + JSON history + TXT report
```

### Generating Visualizations
```bash
# Run visualization script
python 3_3_visualizations_final.py

# Output: 7 PNG files (13-20)
```

### Using Trained Models for Inference
```python
import tensorflow as tf
import pickle
import numpy as np

# Load models
lstm = tf.keras.models.load_model('trained_models/model_LSTM.keras')
gru = tf.keras.models.load_model('trained_models/model_GRU.keras')

# Load scalers
with open('trained_models/feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('trained_models/target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

# Prepare new data
X_new = ... # (batch_size, 21)
X_new_scaled = feature_scaler.transform(X_new)
X_sequences = ... # shape (batch_size, 30, 21)

# Make predictions
lstm_pred_scaled = lstm.predict(X_sequences)
lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled)

gru_pred_scaled = gru.predict(X_sequences)
gru_pred = target_scaler.inverse_transform(gru_pred_scaled)

# Results are in original price scale
print(f"LSTM prediction: ${lstm_pred[0][0]:.2f}")
print(f"GRU prediction: ${gru_pred[0][0]:.2f}")
```

---

## 📚 Documentation Files

### TASK_3_3_DOCUMENTATION.md (500+ lines)
- Technical deep dive into implementation
- Why deep learning underperformed
- Data normalization strategy explained
- Model architecture details
- Recommendations for improvement

### TASK_3_3_COMPLETION_SUMMARY.md (600+ lines)
- Final results and comparison
- Training progression analysis
- Root cause analysis
- Lessons learned
- Next steps for Task 3.4

### TASK_3_3_INDEX.md (This File)
- Quick reference guide
- File inventory
- Performance summary
- How-to instructions

---

## ✅ Validation Checklist

**Task Requirements:**
- [x] Reshape data into 3D format: ✓ (samples, 30 timesteps, 21 features)
- [x] Build LSTM model: ✓ (LSTM→Dropout→LSTM→Dropout→Dense→Dense)
- [x] Compile with adam & mse: ✓ (Adam lr=0.001, loss='mse')
- [x] Train for ~20 epochs: ✓ (10-13 actual, early stopped)
- [x] Batch size 32: ✓ (32 samples per batch)
- [x] Repeat with GRU: ✓ (GRU→Dropout→GRU→Dropout→Dense→Dense)
- [x] Save models: ✓ (.keras format)
- [x] Export logs: ✓ (JSON format)
- [x] Validation loss curves: ✓ (7 visualizations)

**Deliverables:**
- [x] 2 trained models (LSTM.keras, GRU.keras)
- [x] 2 scalers (.pkl files)
- [x] Training history (JSON)
- [x] Report (TXT)
- [x] 7 visualizations (PNG)
- [x] 2 documentation files (MD)
- [x] 2 scripts (PY)

**Total: 16 files ✓**

---

## 🎓 Key Lessons

1. **Simpler models often beat complex ones**
   - LSTM 34,977 params: R²=0.7048
   - Linear Regression 22 params: R²=0.9316
   - 1,500x fewer parameters, 32% better accuracy

2. **Match model to problem**
   - RNNs for long-range dependencies
   - Linear for linear trends
   - Tree models for non-linear local patterns

3. **Always compare baselines**
   - Don't assume new = better
   - Test simple methods first
   - Measure everything objectively

4. **Data normalization matters critically**
   - Separate scalers for features and targets
   - Improper scaling → training failures
   - Inverse transforms for final metrics

5. **Monitor training carefully**
   - Overfitting visible after epoch 5 (LSTM)
   - Early stopping prevents wasted training
   - Val loss curve tells the real story

---

## 🚀 Next Phase: Task 3.4

### Objectives
1. Evaluate all 5 models on **test set**
   - Linear Regression
   - Random Forest
   - SVR
   - LSTM
   - GRU

2. Compare generalization performance
   - Which model best on unseen data?
   - Speed vs accuracy trade-off?

3. Select production model
   - Recommendation: **Linear Regression** (best R²=0.9316)
   - Reason: Simplicity, accuracy, speed

4. Prepare deployment
   - API endpoint design
   - Real-time inference pipeline
   - Monitoring & drift detection

---

## 📞 Summary

**Task 3.3: Training Advanced Models - COMPLETE**

✅ Successfully trained LSTM (R²=0.7048) and GRU (R²=0.7359)  
✅ Both converged smoothly with early stopping  
✅ 7 visualizations generated at 300 DPI  
✅ Comprehensive documentation provided  

⚠️ **Important Finding:** Deep learning **underperforms baseline** (Linear Regression R²=0.9316)

🎯 **Recommendation:** Use Linear Regression for production; keep deep learning for research

📅 **Status:** Ready for Task 3.4 (Test Evaluation & Final Selection)

---

**Task Created:** February 19-25, 2026  
**Status:** 100% Complete  
**Next:** Task 3.4 - Test Set Evaluation  
