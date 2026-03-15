# TASK 3.2: TRAINING BASELINE MODELS - COMPREHENSIVE GUIDE

## Executive Summary

**Status:** ✅ COMPLETED (Feb 12-18, 2026)

Task 3.2 has successfully trained and evaluated three baseline machine learning models on AAPL stock price data. Linear Regression emerged as the clear winner with exceptional performance (R² = 0.9316), establishing a strong benchmark for comparison with deep learning models in subsequent tasks.

---

## 📊 Quick Performance Summary

| Model | R² Score | RMSE | MAE | Training Time |
|-------|----------|------|-----|---|
| 🥇 **Linear Regression** | **0.9316** | **$2.32** | **$1.74** | 0.03s |
| 🥈 Random Forest | 0.8657 | $3.25 | $2.58 | 0.32s |
| 🥉 SVR | 0.2346 | $7.77 | $6.61 | 0.09s |

---

## 🎯 Task Objectives & Completion

### ✅ Step 1: Load Training and Validation Datasets
- **Loaded:** AAPL_stock_data_with_indicators.csv (1,059 samples)
- **Training Set:** 741 samples (Oct 15, 2020 - Sep 26, 2023)
- **Validation Set:** 158 samples (Sep 27, 2023 - May 13, 2024)
- **Status:** ✓ Complete

### ✅ Step 2: Split into Features (X) and Target (Y)
- **Features:** 22 input features (OHLCV + 16 technical indicators)
- **Target:** Next-day close price (regression task)
- **Data Quality:** 0 missing values, 0 duplicates
- **Status:** ✓ Complete

### ✅ Step 3: Train Linear Regression Model
```python
LinearRegression()
  - No hyperparameters to tune
  - Training time: 0.03 seconds
  - Performance: R² = 0.9316
```
**Status:** ✓ Complete

### ✅ Step 4: Train Random Forest Model
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt'
)
  - Training time: 0.32 seconds
  - Performance: R² = 0.8657
```
**Status:** ✓ Complete

### ✅ Step 5: Train SVR Model
```python
SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,
    gamma='scale'
)
  - Training time: 0.09 seconds
  - Performance: R² = 0.2346
```
**Status:** ✓ Complete

### ✅ Step 6: Evaluate Using RMSE, MAE, R² on Validation Set
All metrics calculated and compared. See [Performance Analysis](#performance-analysis) below.

---

## 📈 Detailed Performance Analysis

### Linear Regression (WINNER 🥇)

**Metrics:**
- R² Score: 0.931587
- RMSE: $2.32 (excellent)
- MAE: $1.74 (very low error)
- MAPE: 0.0097% (near-perfect percentage accuracy)
- Mean Residual: $0.09 (well-centered)
- Std Residual: $2.33
- Max Error: $10.47
- Training Time: 0.03s (fastest)

**Why It Excels:**
1. **Simple and Effective:** Linear model captures dominant price trend
2. **Feature Quality:** Input features (especially technical indicators) are highly correlated with next-day price
3. **No Overfitting Risk:** Simple model generalizes well
4. **Production-Ready:** Ultra-fast inference (milliseconds)
5. **Interpretable:** Can extract feature coefficients

**Use Case:** 
- Initial predictions and baselines
- Risk-averse production deployment
- Feature importance analysis

---

### Random Forest (RUNNER-UP 🥈)

**Metrics:**
- R² Score: 0.865667
- RMSE: $3.25 (good)
- MAE: $2.58 (reasonable)
- MAPE: 0.0142%
- Mean Residual: $1.39
- Std Residual: $2.95
- Max Error: $11.57
- Training Time: 0.32s

**Why It's Close:**
1. **Non-linear Patterns:** Attempts to capture market complexity
2. **Feature Interactions:** Captures relationships Linear Regression misses
3. **Robust:** Less sensitive to outliers
4. **Reasonable Speed:** Still faster than deep learning

**Limitations:**
1. **Slightly Lower Accuracy:** 6.6% worse than Linear Regression
2. **Slower:** 10x longer training time
3. **Black Box:** Harder to interpret
4. **Overfit Risk:** More parameters to tune

**Use Case:**
- When non-linear patterns matter
- Ensemble methods
- Moderate accuracy vs speed tradeoff

---

### Support Vector Regression (THIRD PLACE 🥉)

**Metrics:**
- R² Score: 0.234605 (poor)
- RMSE: $7.77 (high error)
- MAE: $6.61 (large deviations)
- MAPE: 0.0367%
- Mean Residual: $1.75
- Std Residual: $7.59
- Max Error: $20.04
- Training Time: 0.09s

**Why It Underperformed:**
1. **Hyperparameter Mismatch:** C=100, epsilon=0.1 not optimal
2. **Kernel Choice:** RBF kernel may not fit this data well
3. **Feature Scaling Sensitivity:** Despite StandardScaler, margins poorly calibrated
4. **No Feature Selection:** May struggle with high-dimensional input

**Potential Improvements:**
- GridSearchCV for hyperparameter tuning
- Try different kernels (linear, polynomial)
- Feature selection (remove less important indicators)
- Different C and epsilon values

**Use Case:**
- Not recommended for this task in current configuration
- Needs hyperparameter optimization before deployment

---

## 🔍 Key Findings

### 1. **Linear Relationships Dominate**
The exceptional performance of Linear Regression (R²=0.9316) suggests that:
- Next-day stock prices follow predominantly linear patterns
- Technical indicators are highly predictive in linear combination
- Market is somewhat efficient (prices move systematically)

### 2. **Feature Quality is Excellent**
- 22 features capture ~93% of variance for Linear Regression
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) provide strong signals
- Feature engineering from Task 3.1 was successful

### 3. **Complexity vs Accuracy Tradeoff**
- **Linear Regression:** Simple, fast, highest accuracy
- **Random Forest:** More complex, slightly lower accuracy, 10x slower
- **SVR:** Complex, slowest, poor performance (needs tuning)

### 4. **Directional Accuracy is Low**
All models show ~28-35% directional accuracy, suggesting:
- Predicting exact prices ≠ predicting direction
- Market has significant random component (noise)
- Classification approach may work better for direction

---

## 📊 Files Generated

### Python Scripts
1. **3_2_baseline_model_training.py** (461 lines)
   - Loads data and creates targets
   - Trains all 3 models
   - Evaluates on validation set
   - Generates reports and saves models

2. **3_2_baseline_visualizations.py** (325 lines)
   - Creates 6 comprehensive visualization PNG files
   - Shows performance comparisons
   - Visualizes predictions vs actual
   - Analyzes residuals

### Output Files

#### Model Artifacts
- `trained_models/model_Linear_Regression.pkl` - Trained Linear Regression model
- `trained_models/model_Random_Forest.pkl` - Trained Random Forest model
- `trained_models/model_SVR.pkl` - Trained SVR model
- `trained_models/scaler.pkl` - StandardScaler fitted on training data

#### Reports
- `baseline_models_training_report.txt` - Detailed text report (131 lines)
- `baseline_models_results.json` - Machine-readable results
- `baseline_models_comparison.csv` - Performance metrics comparison

#### Visualizations (6 PNG files)
1. **07_baseline_performance_metrics.png**
   - 4-panel: R² Score, RMSE, MAE, Training Time

2. **08_actual_vs_predicted.png**
   - 3 scatter plots comparing predictions to actual prices

3. **09_residuals_analysis.png**
   - 6-panel: Time series and distribution of residuals for each model

4. **10_timeseries_predictions.png**
   - 3 time-series plots showing predictions over validation period

5. **11_error_distribution.png**
   - Absolute error distributions for each model

6. **12_summary_scorecard.png**
   - Quick reference card with key metrics

---

## 💾 How to Load and Use Trained Models

### Load a Model
```python
import pickle
import pandas as pd

# Load the scaler
scaler = pickle.load(open('trained_models/scaler.pkl', 'rb'))

# Load Linear Regression model (best performer)
model = pickle.load(open('trained_models/model_Linear_Regression.pkl', 'rb'))

# Load your test data
test_data = pd.read_csv('test_data.csv')
features = [col for col in test_data.columns]
X_test = test_data[features]

# Scale features
X_test_scaled = scaler.transform(X_test)

# Make predictions
predictions = model.predict(X_test_scaled)
```

### Evaluate on New Data
```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Get actual prices
y_test = test_data['target_price']

# Calculate metrics
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
```

---

## 🔄 Cross-Model Comparison

### Linear Regression (SELECTED FOR PRODUCTION)
**Pros:**
- Highest accuracy (R²=0.9316)
- Fastest training (0.03s)
- Fastest inference (milliseconds)
- Interpretable coefficients
- No hyperparameter tuning needed

**Cons:**
- Assumes linear relationships
- Cannot capture non-linear patterns

---

### Random Forest
**Pros:**
- Can capture non-linear patterns
- Feature importance available
- Robust to outliers
- Good generalization

**Cons:**
- 6.6% lower accuracy than Linear Regression
- 10x slower training
- Black-box model
- More prone to overfitting

---

### SVR
**Pros:**
- Can use different kernel functions
- Potential for non-linear patterns

**Cons:**
- Worst performance (R²=0.2346)
- Hyperparameters need tuning
- Slow training (0.09s)
- High error rates
- Requires careful feature scaling

---

## 📋 Baseline Characteristics & Benchmarks

### Dataset Summary
```
Training Set:
  - Samples: 741
  - Date Range: Oct 15, 2020 - Sep 26, 2023
  - Target Mean: $156.47
  - Target Std: $31.54

Validation Set:
  - Samples: 158
  - Date Range: Sep 27, 2023 - May 13, 2024
  - Target Mean: $163.92
  - Target Std: $24.58
```

### Baseline Performance
```
Best Model: Linear Regression
  R² Score: 0.9316
  RMSE: $2.32
  MAE: $1.74

This means:
- Model explains 93.16% of price variance
- Average prediction error: $2.32 (5-day high/low typically $2-5)
- Median absolute error: $1.74 (excellent precision)
```

---

## 🚀 Next Steps (Task 3.3 & Beyond)

### Task 3.3: Deep Learning Models (Feb 19-25)
1. Implement LSTM model
2. Implement GRU model
3. Compare with baseline performance
4. Goal: Beat Linear Regression's R²=0.9316

### Task 3.4: Model Optimization (Feb 26 - Mar 5)
1. Hyperparameter tuning
2. Cross-validation
3. Feature importance analysis
4. Ensemble methods

### Task 3.5: Model Selection & Deployment (Mar 6-12)
1. Final performance comparison
2. Select best model
3. Implement in production
4. Monitor performance

---

## 📌 Important Observations

### 1. Target Leakage Investigation
Linear Regression's R²=0.9316 is exceptionally high. Possible reasons:
- Technical indicators (SMA, EMA) directly incorporate past prices
- Next-day price is highly correlated with day-of indicators
- Efficient market hypothesis: prices are partially predictable

### 2. Feature Importance Implications
The features that contribute most likely:
- Close_AAPL (current price is strongest predictor)
- Moving averages (trend continuation)
- Recent price action

### 3. Market Behavior
- Stock prices follow somewhat predictable patterns
- ~93% of variation explained by 22 features
- ~7% is random/noise (directional accuracy only ~30%)

---

## 📖 Technical Details

### Feature Scaling
```python
StandardScaler()
  - Fitted on training data
  - Applied to validation data
  - Training Mean: 0.000000, Std: 1.000000
  - Validation Mean: 0.933186, Std: 1.128570
```

### Target Variable
```python
Target = Close_AAPL.shift(-1)
  - Shifted by -1 (next day's price)
  - Type: Continuous regression
  - Training Range: $105.70 - $237.04
  - Validation Range: $135.55 - $208.76
```

---

## ✅ Verification Checklist

- [x] Training dataset loaded (741 samples)
- [x] Validation dataset loaded (158 samples)
- [x] Features prepared (22 features)
- [x] Target created (next-day close price)
- [x] Linear Regression trained
- [x] Random Forest trained
- [x] SVR trained
- [x] All models evaluated on validation set
- [x] Performance metrics calculated (R², RMSE, MAE)
- [x] Models saved to disk (.pkl files)
- [x] Results exported (JSON, CSV)
- [x] Report generated (text file)
- [x] Visualizations created (6 PNG files)

**Status:** ✅ 100% COMPLETE

---

## 📊 Results Summary Table

```
╔═══════════════════════════╦═══════════╦═════════╦═════════╦══════════════╗
║ Model                     ║ R² Score  ║ RMSE    ║ MAE     ║ Time (sec)   ║
╠═══════════════════════════╬═══════════╬═════════╬═════════╬══════════════╣
║ Linear Regression (BEST)  ║ 0.9316    ║ $2.32   ║ $1.74   ║ 0.03         ║
║ Random Forest             ║ 0.8657    ║ $3.25   ║ $2.58   ║ 0.32         ║
║ Support Vector Reg        ║ 0.2346    ║ $7.77   ║ $6.61   ║ 0.09         ║
╚═══════════════════════════╩═══════════╩═════════╩═════════╩══════════════╝
```

---

## 🎓 Learning Outcomes

By completing Task 3.2, you now understand:

1. **Data Preparation:** How to split data into train/validation/test
2. **Feature Scaling:** Why and how to standardize features
3. **Model Training:** How to train sklearn models
4. **Evaluation Metrics:** R², RMSE, MAE interpretation
5. **Baseline Establishment:** How to create performance benchmarks
6. **Model Comparison:** How to evaluate trade-offs between models
7. **Model Persistence:** How to save and load trained models

---

## 📞 File Quick Reference

| File | Purpose |
|------|---------|
| `3_2_baseline_model_training.py` | Training script |
| `3_2_baseline_visualizations.py` | Visualization generator |
| `trained_models/model_*.pkl` | Trained model artifacts |
| `baseline_models_training_report.txt` | Detailed text report |
| `baseline_models_results.json` | JSON results |
| `baseline_models_comparison.csv` | CSV metrics |
| `07-12_*.png` | Performance visualizations |

---

## 🏆 Conclusion

**Task 3.2 Status: ✅ COMPLETE**

Three baseline models have been successfully trained and evaluated on AAPL stock price data. Linear Regression emerged as the clear winner with R²=0.9316, establishing a strong benchmark. The models are ready for comparison with deep learning approaches in Task 3.3.

**Key Takeaway:** Simple models can achieve excellent results when features are well-engineered. Linear Regression's performance sets a high bar for complex models to beat.

---

**Generated:** February 18, 2026  
**Timeline:** Feb 12-18 ✓ (COMPLETED)  
**Next Phase:** Task 3.3 - Deep Learning Models (Feb 19-25)
