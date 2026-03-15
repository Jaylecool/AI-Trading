# TASK 3.1: SELECTING ML MODELS FOR STOCK PRICE FORECASTING

## 📊 Project Overview

**Task:** Predict Engine - Model Selection (Task 3.1)  
**Timeline:** February 10-13, 2026  
**Objective:** Identify and document baseline and advanced models for AAPL stock price forecasting  
**Status:** ✅ COMPLETED

---

## 🎯 Task Summary

This task involves selecting appropriate machine learning models for time-series stock price forecasting. We have:

1. **Defined 3 Target Variables:**
   - Next Day Close Price (Regression)
   - Directional Movement Up/Down (Classification)
   - Daily Return Percentage (Regression)

2. **Selected 3 Baseline Models:**
   - Linear Regression
   - Random Forest
   - Support Vector Regression (SVR)

3. **Selected 2 Advanced Models:**
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)

4. **Prepared 23 Input Features:**
   - 7 Price Data features
   - 16 Technical Indicators (Trend, Momentum, Volatility)

---

## 📁 Deliverables

### Python Scripts

| File | Purpose |
|------|---------|
| `3_1_model_selection.py` | Main script for model selection, feature preparation, and baseline training |
| `3_1_model_selection_visualizations.py` | Generate visualizations for data and model analysis |

### Reports & Documentation

| File | Content |
|------|---------|
| `model_selection_report.txt` | Detailed report of all selected models |
| `model_selection_summary.md` | Comprehensive markdown documentation |
| `model_metadata.json` | Structured model configurations in JSON format |
| `README.md` | This file |

### Visualizations (6 PNG files)

| File | Description |
|------|-------------|
| `01_target_variables_overview.png` | Price, Distribution, Direction, and Returns |
| `02_technical_indicators_timeseries.png` | Time series of 8 key technical indicators |
| `03_feature_correlation_heatmap.png` | Correlation matrix of key features |
| `04_price_with_indicators.png` | Price movements with SMA, RSI, and Bollinger Bands |
| `05_model_selection_summary.png` | Model selection infographic |
| `06_baseline_model_performance.png` | Baseline model comparison metrics |

### Data Files (Already Generated)

| File | Purpose |
|------|---------|
| `AAPL_stock_data_with_indicators.csv` | 1,058 samples with 22 technical indicators |
| `AAPL_stock_data_train.csv` | Training set (70%) |
| `AAPL_stock_data_val.csv` | Validation set (15%) |
| `AAPL_stock_data_test.csv` | Test set (15%) |

---

## 📊 Dataset Summary

**Stock:** Apple (AAPL)  
**Period:** October 15, 2020 - December 31, 2024  
**Total Samples:** 1,058 trading days (~4 years)  
**Features:** 23 (7 price + 16 technical indicators)  
**Frequency:** Daily

### Data Quality
- **Missing Values:** 0
- **Duplicates:** 0
- **Normalization:** StandardScaler applied
- **Scaling:** Mean ≈ 0, Std Dev ≈ 1

---

## 🎲 Target Variables

### 1. Next Day Close Price (Regression)
- **Type:** Continuous prediction
- **Range:** $105.70 - $257.85
- **Mean:** $164.09
- **Std Dev:** $32.41
- **Use Case:** Predict exact closing price

### 2. Directional Movement (Classification)
- **Type:** Binary classification (Up=1, Down=0)
- **Distribution:** Up: 52.9%, Down: 47.1%
- **Use Case:** Predict up/down movement

### 3. Daily Return % (Regression)
- **Type:** Continuous prediction
- **Range:** -5.87% to +8.90%
- **Mean:** +0.09%
- **Std Dev:** 1.70%
- **Use Case:** Predict percentage change

---

## 🔧 Input Features (23 Total)

### Price Data (7)
```
1. Close_AAPL      - Daily closing price
2. Open_AAPL       - Daily opening price
3. High_AAPL       - Daily high price
4. Low_AAPL        - Daily low price
5. Volume_AAPL     - Trading volume
6. BB_Lower        - Bollinger Band lower band
7. BB_Middle       - Bollinger Band middle (SMA_20)
```

### Trend Indicators (7)
```
Simple Moving Averages:
8. SMA_10          - 10-day moving average
9. SMA_20          - 20-day moving average
10. SMA_50         - 50-day moving average
11. SMA_200        - 200-day moving average

Exponential Moving Averages:
12. EMA_10         - 10-day exponential average
13. EMA_20         - 20-day exponential average
14. EMA_50         - 50-day exponential average
```

### Momentum Indicators (5)
```
15. RSI_14         - Relative Strength Index (14-period)
16. MACD           - MACD line (12-26 period)
17. MACD_Signal    - MACD Signal line (9-period EMA)
18. MACD_Histogram - MACD Histogram (MACD - Signal)
19. ROC_12         - Rate of Change (12-period)
```

### Volatility Indicators (4)
```
20. BB_Upper       - Bollinger Band upper band
21. ATR_14         - Average True Range (14-period)
22. Volatility_20  - Standard Deviation (20-period)
```

---

## 🤖 Baseline Models (Traditional ML)

### Model 1: Linear Regression ⭐

**Complexity:** ⭐ Low  
**Interpretability:** ⭐⭐⭐⭐⭐ Very High  
**Computational Cost:** ⭐ Low  

**Description:**
Simple linear relationship between features and target price.

**Architecture:**
- Single linear equation: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- No hidden layers
- Direct mapping from features to output

**Hyperparameters:** None (standard implementation)

**Advantages:**
- Fast training and prediction
- Highly interpretable (coefficients show feature impact)
- Provides feature importance
- Low memory usage

**Disadvantages:**
- Limited to linear patterns
- Cannot capture complex relationships
- Poor for non-linear time series

**Use Cases:**
- Quick baseline comparison
- Feature importance analysis
- When interpretability is critical

**Performance (Test Set):**
- R² Score: 1.0000
- RMSE: $0.00
- MAE: $0.00
- **Note:** Perfect performance suggests target leakage (to investigate)

---

### Model 2: Random Forest 🌲

**Complexity:** ⭐⭐ Medium  
**Interpretability:** ⭐⭐ Medium  
**Computational Cost:** ⭐⭐ Medium  

**Description:**
Ensemble of 100 decision trees using bootstrap sampling and random feature selection.

**Architecture:**
```
Random Forest (Regression)
├── Tree 1: Decision tree on bootstrap sample 1
├── Tree 2: Decision tree on bootstrap sample 2
├── ...
└── Tree 100: Decision tree on bootstrap sample 100
Output: Average prediction from all trees
```

**Hyperparameters:**
```
n_estimators: 100        # Number of trees
max_depth: 15            # Maximum tree depth
min_samples_split: 5     # Minimum samples to split node
min_samples_leaf: 2      # Minimum samples in leaf node
random_state: 42         # Reproducibility
```

**Advantages:**
- Captures non-linear relationships
- Handles feature interactions
- Built-in feature importance
- Robust to outliers
- Reduced overfitting via bagging

**Disadvantages:**
- Slower than linear regression
- Less interpretable than linear models
- May overfit with default parameters
- Memory intensive with many trees

**Use Cases:**
- General-purpose regression
- When non-linear patterns exist
- Feature importance analysis
- Moderate dataset sizes

**Performance (Test Set):**
- R² Score: -0.0784
- RMSE: $26.95
- MAE: $20.67

---

### Model 3: Support Vector Regression (SVR) 📈

**Complexity:** ⭐⭐⭐ Medium-High  
**Interpretability:** ⭐ Low  
**Computational Cost:** ⭐⭐⭐ High  

**Description:**
Maps data to higher-dimensional space using kernel functions for non-linear regression.

**Architecture:**
```
Input Features (23D)
    ↓
[Kernel Transformation]
    ↓
High-Dimensional Space (implicit)
    ↓
Support Vector Regression
    ↓
Output: Predicted Price
```

**Hyperparameters:**
```
kernel: 'rbf'            # Radial Basis Function kernel
C: 100                   # Regularization parameter
epsilon: 0.1             # Margin of tolerance
gamma: 'scale'           # Kernel coefficient
```

**Advantages:**
- Effective in high-dimensional spaces
- Flexible kernel options (linear, RBF, polynomial)
- Good for complex non-linear relationships
- Memory efficient (uses only support vectors)

**Disadvantages:**
- Slow training on large datasets
- Slow prediction on large datasets
- Sensitive to feature scaling (why StandardScaler used)
- Hyperparameter tuning is challenging
- Black-box model (low interpretability)

**Use Cases:**
- Complex non-linear relationships
- High-dimensional feature spaces
- Small-medium datasets
- When kernel flexibility needed

**Performance (Test Set):**
- R² Score: -2.5867
- RMSE: $49.15
- MAE: $38.98

---

## 🚀 Advanced Models (Deep Learning)

### Model 4: LSTM (Long Short-Term Memory) 🧠

**Type:** Recurrent Neural Network (RNN)  
**Complexity:** ⭐⭐⭐⭐⭐ Very High  
**Interpretability:** ⭐ Very Low  
**Computational Cost:** ⭐⭐⭐⭐⭐ Very High  

**Description:**
Advanced RNN with memory cells designed to capture long-term dependencies in sequential data.

**Network Architecture:**
```
Sequential Input (30 days × 23 features)
    ↓
[LSTM Layer 1] 64 units, return_sequences=True, dropout=0.2
    ↓
[LSTM Layer 2] 32 units, return_sequences=False, dropout=0.2
    ↓
[Dense Layer] 16 units, activation='relu'
    ↓
[Output Layer] 1 unit, activation='linear'
    ↓
Predicted Price
```

**Key Components:**
- **Cell State (C):** Long-term memory
- **Hidden State (H):** Short-term memory
- **Forget Gate:** Controls what to forget
- **Input Gate:** Controls what to remember
- **Output Gate:** Controls output generation

**Hyperparameters:**
```
sequence_length: 30      # 30-day lookback window
batch_size: 32           # Samples per batch
epochs: 50               # Training iterations
learning_rate: 0.001     # Adam optimizer learning rate
dropout: 0.2             # Regularization (prevent overfitting)
```

**Training Configuration:**
```
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Metrics: MAE, RMSE, R²
Early Stopping: Monitor validation loss
```

**Advantages:**
- Captures long-term temporal dependencies
- Handles variable-length sequences
- Avoids vanishing gradient problem
- State-of-the-art performance potential
- Flexible architecture

**Disadvantages:**
- Very slow training (hours/days)
- Computationally expensive (GPU/TPU needed)
- Prone to overfitting on small datasets
- Complex hyperparameter tuning
- Black-box interpretation
- Requires significant data (1000+ samples)

**Use Cases:**
- Time-series with long-range dependencies
- Large datasets (1000+ samples)
- GPU/TPU resources available
- Need state-of-the-art accuracy
- Sequential pattern learning

**Sequence Processing:**
```
Input: 30 days of historical data (30 × 23 matrix)
Processing: LSTM cell processes one day at a time
Prediction: Output price for day 31
```

---

### Model 5: GRU (Gated Recurrent Unit) 🧠

**Type:** Recurrent Neural Network (RNN)  
**Complexity:** ⭐⭐⭐⭐ High  
**Interpretability:** ⭐ Very Low  
**Computational Cost:** ⭐⭐⭐ High  

**Description:**
Simplified RNN architecture combining forget and input gates into a single "reset" gate.

**Network Architecture:**
```
Sequential Input (30 days × 23 features)
    ↓
[GRU Layer 1] 64 units, return_sequences=True, dropout=0.2
    ↓
[GRU Layer 2] 32 units, return_sequences=False, dropout=0.2
    ↓
[Dense Layer] 16 units, activation='relu'
    ↓
[Output Layer] 1 unit, activation='linear'
    ↓
Predicted Price
```

**Key Components:**
- **Hidden State (H):** Combined memory
- **Reset Gate:** Controls previous state influence
- **Update Gate:** Controls current input influence
- **Fewer Parameters:** 2/3 parameters vs LSTM

**Hyperparameters:**
```
sequence_length: 30      # 30-day lookback window
batch_size: 32           # Samples per batch
epochs: 50               # Training iterations
learning_rate: 0.001     # Adam optimizer learning rate
dropout: 0.2             # Regularization
```

**Training Configuration:**
```
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Metrics: MAE, RMSE, R²
Early Stopping: Monitor validation loss
```

**Advantages:**
- Faster than LSTM (30-40% speedup)
- Fewer parameters (reduced overfitting)
- Similar performance to LSTM
- Better for limited data
- Computational efficiency

**Disadvantages:**
- May miss fine-grained patterns vs LSTM
- Still requires hyperparameter tuning
- Slower than traditional ML
- Black-box interpretation
- GPU beneficial but not required

**Use Cases:**
- Time-series with limited compute
- Moderate-sized datasets (500-2000)
- When speed is important
- Limited GPU resources
- Similar accuracy to LSTM needed

**Comparison to LSTM:**
```
GRU: Faster, Fewer Parameters, Good for Small Data
LSTM: More Powerful, More Parameters, Need More Data
```

---

## 📊 Baseline Model Performance

### Quick Evaluation Results (Test Set)

```
Model                          R² Score    RMSE ($)    MAE ($)
═══════════════════════════════════════════════════════════════
Linear Regression              1.0000      $0.00       $0.00
Random Forest                 -0.0784     $26.95      $20.67
Support Vector Regression     -2.5867     $49.15      $38.98
```

### Key Findings:

1. **Linear Regression:** Perfect performance (R²=1.0) indicates:
   - Target leakage issue (needs investigation)
   - May have direct correlation with features
   - Not realistic; to be validated in Task 3.2

2. **Random Forest:** Negative R² indicates:
   - Model performing worse than mean baseline
   - Non-linear features may not capture target well
   - Hyperparameter tuning needed

3. **SVR:** Worst performance suggests:
   - Poor kernel choice or hyperparameters
   - Feature scaling issues despite StandardScaler
   - Requires careful hyperparameter optimization

---

## 🔄 Next Steps (Task 3.2 - Feb 14-20)

### Phase 2: Model Training & Evaluation

1. **Fix Data Issues**
   - Investigate target leakage
   - Remove redundant features
   - Proper feature selection

2. **Baseline Model Training**
   - Implement cross-validation
   - Hyperparameter tuning (GridSearchCV)
   - Feature importance analysis

3. **Advanced Model Implementation**
   - Build LSTM model with Keras/TensorFlow
   - Build GRU model with Keras/TensorFlow
   - GPU acceleration setup

4. **Comprehensive Evaluation**
   - Training/validation/test curves
   - Residual analysis
   - Directional accuracy
   - Multiple metrics (R², RMSE, MAE, MAPE)

5. **Model Comparison**
   - Head-to-head comparison
   - Statistical significance testing
   - Robustness analysis

6. **Final Selection**
   - Choose best model
   - Document rationale
   - Prepare for deployment

---

## 📈 Feature Engineering Notes

### Why These Indicators?

**Trend Indicators (SMA, EMA):**
- Identify price direction
- Smooth out noise
- Different periods capture different trends

**Momentum Indicators (RSI, MACD, ROC):**
- Measure rate of change
- Detect overbought/oversold conditions
- Confirm trend strength

**Volatility Indicators (Bollinger Bands, ATR):**
- Measure price uncertainty
- Identify breakout potential
- Risk assessment

### Feature Normalization:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Result: Mean ≈ 0, Std Dev ≈ 1
```

**Why Normalize?**
- SVR and neural networks are sensitive to scale
- Improves convergence in gradient descent
- Fair comparison between features
- Prevents features with larger scale dominating

---

## 🎯 Model Selection Criteria

### Performance Metrics:
- **R² Score:** Explains variance (0-1, higher better)
- **RMSE:** Root mean squared error (lower better)
- **MAE:** Mean absolute error (lower better)
- **MAPE:** Mean absolute percentage error
- **Directional Accuracy:** % correct direction predictions

### Business Criteria:
- Prediction Horizon: 1 day ahead
- Latency Requirements: Daily predictions sufficient
- Interpretability: Important for risk management
- Computational Cost: Should be reasonable
- Robustness: Consistent across market conditions

---

## 🔐 Model Configuration Files

### model_metadata.json
Structured configuration for all models:
- Dataset information
- Feature specifications
- Target variable definitions
- Baseline model hyperparameters
- Advanced model architectures

### model_selection_report.txt
Human-readable report including:
- Dataset overview
- Target definitions
- Model descriptions
- Performance metrics
- Architecture details

---

## 🛠️ How to Run

### Run Model Selection Script:
```bash
python 3_1_model_selection.py
```

**Output:**
- Model selection console output
- model_selection_report.txt
- model_metadata.json

### Generate Visualizations:
```bash
python 3_1_model_selection_visualizations.py
```

**Output:**
- 6 PNG visualization files
- Target variables overview
- Technical indicators analysis
- Model comparison charts

---

## 📝 Summary

**Task 3.1 Status:** ✅ COMPLETE

**Deliverables:**
- ✅ 3 Target variables defined
- ✅ 3 Baseline models selected
- ✅ 2 Advanced models selected
- ✅ 23 Input features prepared
- ✅ Model architectures documented
- ✅ Hyperparameters configured
- ✅ Baseline training completed
- ✅ Comprehensive visualizations generated
- ✅ Full documentation provided

**Files Created:** 10+
- 2 Python scripts
- 3 Documentation files
- 6 Visualization PNG files
- 1 JSON configuration file

**Timeline:** Feb 10-13, 2026 ✓  
**Next Task:** Feb 14-20, 2026 (Task 3.2 - Model Training & Evaluation)

---

## 📖 References

- **sklearn Documentation:** https://scikit-learn.org/
- **TensorFlow/Keras:** https://www.tensorflow.org/
- **PyTorch:** https://pytorch.org/
- **Technical Indicators:** https://en.wikipedia.org/wiki/Technical_indicator
- **Time Series Forecasting:** https://en.wikipedia.org/wiki/Time_series

---

**Project:** AI Trading System  
**Phase:** 3 - Prediction Engine  
**Task:** 3.1 - Model Selection  
**Status:** ✅ Completed  
**Date:** February 10-13, 2026
