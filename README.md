# AI-Driven Automated Stock Trading System Using Predictive Market Analysis

A full-stack AI trading system that uses machine learning to predict stock price movements and automatically execute trades. Built as a final-year dissertation project.

## Features

- **ML Ensemble Predictions** — Per-symbol models (Linear Regression + Random Forest + Gradient Boosting) trained on 30 engineered features to forecast next-day returns
- **Automated Trading Engine** — Background thread evaluates signals every 60 seconds with dual confirmation (PredictionEngine + TradingRules) before executing trades
- **Per-Symbol Strategy Selection** — Auto-selects the best strategy (Aggressive / Balanced / Conservative) for each stock based on backtest results
- **Risk Management** — ATR-based adaptive stop losses, trailing stops, portfolio circuit breaker, minimum hold periods, position sizing limits
- **Real-Time Dashboard** — Dark-mode web UI with live price streaming, interactive Plotly charts, portfolio metrics, trade history, and alert configuration
- **Backtesting Engine** — Simulates strategies on historical data with full metrics (ROI, Sharpe, drawdown, win rate, profit factor)
- **User Authentication** — SQLite-backed registration/login with session management and per-user portfolio isolation
- **Alert & Notification System** — Configurable price/prediction/risk alerts with multi-channel delivery

## Supported Stocks

AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA

## Project Structure

```
├── dashboard_app_trade_history.py   # Flask backend (47 API routes + auto-trader)
├── prediction_engine.py             # ML signal generation with technical analysis
├── trading_rules.py                 # Buy/sell signal logic + position sizing
├── strategy_configurations.py       # Aggressive / Conservative / Balanced strategies
├── risk_management_enhanced.py      # Trailing stops, dynamic TP, portfolio risk
├── trading_execution.py             # Order management and trade execution
├── backtesting_engine.py            # Historical simulation engine
├── backtest_runner.py               # Multi-symbol, multi-strategy backtest runner
├── model_trainer.py                 # ML model training pipeline
├── data_fetcher.py                  # Yahoo Finance data download + indicator calc
├── portfolio_tracker.py             # Portfolio state and trade history management
├── streaming_data_service.py        # Real-time price streaming service
├── alert_system.py                  # Configurable alert rules engine
├── notification_service.py          # Multi-channel notifications (email, popup, etc.)
├── auth.py                          # User registration/login (SQLite + Werkzeug)
├── config.py                        # Centralised configuration (.env support)
├── tests.py                         # Unit test suite (44 tests)
├── templates/
│   ├── landing.html                 # Landing page
│   ├── auth.html                    # Login / Register page
│   └── dashboard_trade_history.html # Main dashboard UI
├── trained_models/                  # Per-symbol ML model artifacts
│   ├── AAPL/ MSFT/ GOOGL/ AMZN/ TSLA/ META/
│   │   ├── model_lr.pkl             # Linear Regression
│   │   ├── model_rf.pkl             # Random Forest Regressor
│   │   ├── model_gb_clf.pkl         # Gradient Boosting Classifier
│   │   ├── model_dir_clf.pkl        # Direction Classifier
│   │   ├── scaler.pkl               # StandardScaler
│   │   └── training_report.json     # Training metrics
├── data/                            # Historical price CSVs
├── results/                         # Backtest results and strategy comparisons
├── models/                          # ML research scripts (LSTM, evaluation, etc.)
├── visualizations/                  # Generated charts and plots
└── docs/                            # Task documentation (Tasks 3–5)
```

## Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone <repo-url>
cd "AI Trading"
pip install -r requirements.txt
```

### Run the Dashboard

```bash
python dashboard_app_trade_history.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Run Backtests

```bash
# All stocks
python backtest_runner.py

# Specific stocks
python backtest_runner.py AAPL TSLA META
```

### Run Tests

```bash
python -m unittest tests -v
```

### Retrain Models

```bash
python model_trainer.py
```

### Fetch Fresh Data

```bash
python data_fetcher.py
```

## Configuration

All settings can be overridden via a `.env` file in the project root:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | 5000 | Server port |
| `INITIAL_CAPITAL` | 100000 | Starting portfolio balance ($) |
| `SUPPORTED_SYMBOLS` | AAPL,NVDA,MSFT,GOOGL,AMZN,TSLA,META | Comma-separated stock tickers |
| `SECRET_KEY` | auto-generated | Flask session secret |
| `SMTP_SERVER` | *(empty)* | Email server for notifications |

## Backtest Results

All 6 stocks are profitable across all 3 strategies:

| Stock | Aggressive | Conservative | Balanced |
|-------|-----------|-------------|---------|
| AAPL  | +5.96%    | +3.61%      | **+6.43%** |
| MSFT  | +2.30%    | +0.57%      | **+2.79%** |
| GOOGL | **+9.92%** | +5.08%     | +8.48%  |
| AMZN  | **+3.40%** | +1.18%     | +2.56%  |
| TSLA  | +20.97%   | +13.41%     | **+21.13%** |
| META  | **+14.32%** | +9.20%    | +14.14% |

Win rates: 65–96%. Sharpe ratios: 5.2–19.6. Max drawdowns: 0.5–3.1%.

## Architecture

```
Browser ──► Flask (dashboard_app_trade_history.py)
              │
              ├── Auth (auth.py + SQLite)
              ├── Streaming (streaming_data_service.py + Yahoo Finance)
              ├── Alerts (alert_system.py + notification_service.py)
              │
              └── Auto-Trade Engine (background thread, 60s cycle)
                    │
                    ├── PredictionEngine ──► per-symbol ML models
                    ├── TradingRules ──► buy/sell signal confirmation
                    ├── StrategyConfigurations ──► per-symbol best strategy
                    └── RiskManagement ──► stops, sizing, circuit breaker
```

## Technology Stack

- **Backend:** Python, Flask, SQLite
- **ML:** scikit-learn (LR, RF, GB), StandardScaler, 30 engineered features
- **Data:** Yahoo Finance (yfinance), pandas, numpy
- **Frontend:** HTML/CSS/JS, Plotly.js, dark-mode responsive UI
- **Testing:** unittest (44 tests)