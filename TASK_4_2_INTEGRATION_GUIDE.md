# Task 4.2: Integration Guide

## How to Connect Model Predictions to Automated Trading

This guide shows how to integrate your AI trading model with the automated execution system.

---

## Step 1: Model Output Format

Your model should output predictions in this format:

```python
class ModelPrediction:
    def __init__(self, timestamp, symbol, current_price, predicted_price, 
                 market_data, confidence):
        self.timestamp = timestamp          # datetime object
        self.symbol = symbol                # "AAPL" or similar
        self.current_price = current_price  # Current market price
        self.predicted_price = predicted_price  # Model's predicted price
        self.market_data = market_data      # Technical indicators
        self.confidence = confidence        # 0.0 to 1.0
```

---

## Step 2: Market Data Requirements

Your model must provide these technical indicators:

```python
market_data = {
    'Close': 200.50,              # Current ticker price
    'RSI_14': 55.0,               # RSI(14) - Range 0-100
    'SMA_20': 199.50,             # 20-day Simple Moving Average
    'EMA_10': 200.25,             # 10-day Exponential Moving Average
    'EMA_20': 199.75,             # 20-day Exponential Moving Average
    'Volatility_20': 1.5          # 20-day volatility percentage
}
```

### Calculating Technical Indicators

```python
import talib
import pandas as pd

def calculate_market_data(prices_df):
    """Calculate all required indicators"""
    
    close = prices_df['Close'].values
    
    return {
        'Close': close[-1],
        'RSI_14': talib.RSI(close, timeperiod=14)[-1],
        'SMA_20': talib.SMA(close, timeperiod=20)[-1],
        'EMA_10': talib.EMA(close, timeperiod=10)[-1],
        'EMA_20': talib.EMA(close, timeperiod=20)[-1],
        'Volatility_20': pd.Series(close).pct_change().rolling(20).std().iloc[-1] * 100
    }
```

---

## Step 3: Integration Example

### Basic Integration (Backtesting)

```python
from trading_execution import TradingEngine
from trading_rules import TradingParameters
import pandas as pd

# Load historical data
data = pd.read_csv('AAPL_historical.csv', index_col='Date', parse_dates=True)

# Initialize trading engine
params = TradingParameters()
engine = TradingEngine(params, initial_capital=100000.0)

# Simulation loop
for date, row in data.iterrows():
    # 1. Calculate market data
    market_data = {
        'Close': row['Close'],
        'RSI_14': row['RSI'],
        'SMA_20': row['SMA_20'],
        'EMA_10': row['EMA_10'],
        'EMA_20': row['EMA_20'],
        'Volatility_20': row['Volatility']
    }
    
    # 2. Get model prediction
    predicted_price = model.predict(market_data)[0]
    
    # 3. Process prediction and execute trades
    signal, executed = engine.process_prediction(
        current_price=row['Close'],
        predicted_price=predicted_price,
        market_data=market_data,
        current_date=date
    )
    
    # 4. Track performance (optional)
    if signal:
        status = engine.get_portfolio_status()
        print(f"{date}: {signal} executed | P&L: ${status['total_pnl']:.2f}")

# 5. Save results
engine.trade_logger.save_logs(output_dir='./backtest_results')

# 6. Get final statistics
final_stats = engine.get_portfolio_status()
summary = engine.trade_logger.get_summary()

print("\nBacktest Results:")
print(f"Final Portfolio Value: ${final_stats['portfolio_value']:,.2f}")
print(f"Total P&L: ${final_stats['total_pnl']:,.2f}")
print(f"Return: {final_stats['return_percent']:.2%}")
print(f"Total Trades: {summary['total_trades']}")
print(f"Win Rate: {summary['win_rate']:.1f}%")
```

---

### Real-Time Integration (Live Trading)

```python
import yfinance as yf
import time
from datetime import datetime, timedelta
from trading_execution import TradingEngine
from trading_rules import TradingParameters

class LiveTrader:
    def __init__(self, model, initial_capital=100000.0):
        self.model = model
        self.params = TradingParameters()
        self.engine = TradingEngine(self.params, initial_capital=initial_capital)
        self.symbol = 'AAPL'
        self.last_check = None
        
    def get_current_data(self):
        """Fetch current market data from yfinance"""
        ticker = yf.Ticker(self.symbol)
        
        # Get recent history for indicators
        hist = ticker.history(period='1mo')
        current = hist.iloc[-1]
        
        import talib
        close_prices = hist['Close'].values
        
        return {
            'Close': current['Close'],
            'RSI_14': talib.RSI(close_prices, timeperiod=14)[-1],
            'SMA_20': talib.SMA(close_prices, timeperiod=20)[-1],
            'EMA_10': talib.EMA(close_prices, timeperiod=10)[-1],
            'EMA_20': talib.EMA(close_prices, timeperiod=20)[-1],
            'Volatility_20': hist['Close'].pct_change().std() * 100
        }
    
    def check_signals(self):
        """Check for trading signals every minute"""
        now = datetime.now()
        
        # Only check during market hours (9:30 AM - 4:00 PM ET)
        if not (9.5 <= now.hour < 16):
            return
        
        # Check at least 1 minute since last check
        if self.last_check and (now - self.last_check).seconds < 60:
            return
        
        try:
            # Get current market data
            market_data = self.get_current_data()
            current_price = market_data['Close']
            
            # Get model prediction
            predicted_price = self.model.predict(market_data)[0]
            
            # Process prediction
            signal, executed = self.engine.process_prediction(
                current_price=current_price,
                predicted_price=predicted_price,
                market_data=market_data,
                current_date=now
            )
            
            # Log execution
            if signal:
                status = self.engine.get_portfolio_status()
                print(f"[{now}] {signal} | Price: ${current_price:.2f} | "
                      f"P&L: ${status['total_pnl']:.2f}")
            
            self.last_check = now
            
        except Exception as e:
            print(f"Error checking signals: {e}")
    
    def run_live(self):
        """Run trading loop (execute in background)"""
        while True:
            try:
                self.check_signals()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\nStopping live trader...")
                self.save_session()
                break
    
    def save_session(self):
        """Save trading session logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.engine.trade_logger.save_logs(f"./trades_{timestamp}")
        print(f"Session saved to ./trades_{timestamp}")

# Usage:
if __name__ == "__main__":
    # Load your model
    from my_trading_model import TradingModel
    model = TradingModel()
    model.load('trained_model.pkl')
    
    # Initialize and run
    trader = LiveTrader(model, initial_capital=100000.0)
    trader.run_live()  # Run continuously in background
```

---

## Step 4: Model Requirements

Your prediction model should:

```python
class MyTradingModel:
    def __init__(self):
        # Your model initialization
        pass
    
    def predict(self, market_data):
        """
        Input:  market_data dict with Close, RSI_14, SMA_20, EMA_10, EMA_20, Volatility_20
        Output: single float value representing predicted next price
        
        Example:
            Input:  {'Close': 200.50, 'RSI_14': 55, ...}
            Output: 202.30  # Predicted price will be $202.30
        """
        
        # Extract features from market_data
        features = [
            market_data['Close'],
            market_data['RSI_14'],
            market_data['SMA_20'],
            market_data['EMA_10'],
            market_data['EMA_20'],
            market_data['Volatility_20']
        ]
        
        # Make prediction
        predicted_price = self.model.predict([features])[0]
        
        return predicted_price
```

---

## Step 5: Backtesting Framework

```python
from trading_execution import TradingEngine
from trading_rules import TradingParameters
import pandas as pd
import numpy as np

class BacktestFramework:
    def __init__(self, model, data_path, initial_capital=100000.0):
        self.model = model
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        self.initial_capital = initial_capital
        self.params = TradingParameters()
        self.results = []
    
    def run_backtest(self):
        """Run complete backtest"""
        engine = TradingEngine(self.params, self.initial_capital)
        
        for date, row in self.data.iterrows():
            market_data = {
                'Close': row['Close'],
                'RSI_14': row['RSI'],
                'SMA_20': row['SMA_20'],
                'EMA_10': row['EMA_10'],
                'EMA_20': row['EMA_20'],
                'Volatility_20': row['Volatility']
            }
            
            predicted = self.model.predict(market_data)[0]
            signal, executed = engine.process_prediction(
                current_price=row['Close'],
                predicted_price=predicted,
                market_data=market_data,
                current_date=date
            )
            
            status = engine.get_portfolio_status()
            self.results.append({
                'date': date,
                'signal': signal,
                'executed': executed,
                'portfolio_value': status['portfolio_value'],
                'pnl': status['total_pnl'],
                'positions': status['num_open_positions']
            })
        
        return engine, pd.DataFrame(self.results)
    
    def get_metrics(self, results_df):
        """Calculate performance metrics"""
        final_portfolio = results_df['portfolio_value'].iloc[-1]
        return {
            'total_return': (final_portfolio - self.initial_capital) / self.initial_capital,
            'max_drawdown': (results_df['portfolio_value'].min() - self.initial_capital) / self.initial_capital,
            'total_pnl': final_portfolio - self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe(results_df['pnl'].diff()),
            'max_portfolio_value': results_df['portfolio_value'].max()
        }
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio (simplified)"""
        excess_returns = returns - (risk_free_rate / 252)
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# Usage:
backtester = BacktestFramework(model, 'AAPL_data.csv')
engine, results = backtester.run_backtest()
metrics = backtester.get_metrics(results)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

---

## Step 6: Configuration Adjustments

Customize trading parameters before running:

```python
# Create custom parameters
params = TradingParameters()

# Adjust thresholds based on your model's performance
params.buy_threshold = 0.03              # 3% instead of 2%
params.take_profit_percent = 0.035       # 3.5% instead of 2.5%
params.stop_loss_percent = -0.02         # -2% instead of -1.5%

# Adjust risk management
params.risk_per_trade = 0.025            # 2.5% of capital per trade
params.max_concurrent_positions = 5      # Allow 5 positions

# Create engine with custom parameters
engine = TradingEngine(params, initial_capital=100000.0)
```

---

## Step 7: Performance Monitoring

Monitor key metrics during trading:

```python
def monitor_performance(engine, interval_minutes=60):
    """Print performance metrics periodically"""
    status = engine.get_portfolio_status()
    summary = engine.trade_logger.get_summary()
    
    print("\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
    print(f"Total P&L: ${status['total_pnl']:,.2f} ({status['return_percent']:.2%})")
    print(f"Drawdown: {-status.get('drawdown', 0):.2%}")
    print(f"\nPositions: {status['num_open_positions']} open")
    print(f"Buy Signals: {status['buy_signals']}")
    print(f"Sell Signals: {status['sell_signals']}")
    print(f"\nWin Rate: {summary.get('win_rate', 0):.1f}%")
    print(f"Avg Trade P&L: ${summary.get('avg_pnl', 0):.2f}")
    print(f"Largest Win: ${summary.get('largest_pnl', 0):.2f}")
    print(f"Largest Loss: ${summary.get('smallest_pnl', 0):.2f}")
    print("="*60 + "\n")

# Call periodically
monitor_performance(engine)
```

---

## Step 8: Risk Controls

Always implement these safety measures:

```python
class RiskControls:
    def __init__(self, engine):
        self.engine = engine
        self.max_daily_loss = 1000.0      # Stop trading if loss > $1000
        self.max_positions = 3             # Already in TradingParameters
        self.alert_threshold = 0.9         # Alert at 90% of max loss
    
    def check_risk_limits(self):
        """Check if risk limits are exceeded"""
        status = self.engine.get_portfolio_status()
        pnl = status['total_pnl']
        
        if pnl < -self.max_daily_loss:
            print(f"CRITICAL: Daily loss limit reached: ${pnl:.2f}")
            # Close all positions
            self._close_all_positions()
            return False
        
        elif pnl < -(self.max_daily_loss * self.alert_threshold):
            print(f"WARNING: Daily loss approaching limit: ${pnl:.2f}")
            # Reduce position size or stop new trades
            return True
        
        return True
    
    def _close_all_positions(self):
        """Emergency close all positions"""
        # Implementation depends on your broker
        pass

# Use in trading loop:
risk_ctrl = RiskControls(engine)
if not risk_ctrl.check_risk_limits():
    print("Trading stopped due to risk limit")
    break
```

---

## Step 9: Troubleshooting

### No Signals Generated
- Check market_data values are numeric
- Verify predictions are reasonable (within ±10% of current price)
- Ensure indicators are calculated correctly
- Check trading threshold settings

### Orders Not Executing
- Verify sufficient cash available
- Check position limit not reached
- Ensure current_price > 0
- Review order status in logs

### Memory Issues with Live Trading
- Save logs periodically: `engine.trade_logger.save_logs()`
- Clear old position history
- Implement session rotation (restart daily)

```python
# Periodic cleanup
if daily_trades > 100:
    engine.trade_logger.save_logs(f"./daily_logs/{date}")
    engine = TradingEngine(params, initial_capital)  # Fresh engine
```

---

## Step 10: Testing Checklist

Before deploying to production:

- [ ] Backtest on ≥ 6 months historical data
- [ ] Verify Sharpe ratio ≥ 1.0
- [ ] Check max drawdown ≤ 20%
- [ ] Confirm win rate ≥ 40%
- [ ] Test with paper trading for 2+ weeks
- [ ] Verify order logging is working
- [ ] Test stop-loss trigger with live market data
- [ ] Confirm circuit breaker activates correctly
- [ ] Set up email/SMS alerts for trades
- [ ] Document all parameter settings

---

## Example: Complete Integration Code

```python
# ============================================================================
# COMPLETE INTEGRATION EXAMPLE
# ============================================================================

import pandas as pd
from datetime import datetime
from trading_execution import TradingEngine
from trading_rules import TradingParameters
from my_model import MyTradingModel

# 1. LOAD MODEL
print("Loading model...")
model = MyTradingModel()
model.load('my_trained_model.pkl')

# 2. LOAD DATA
print("Loading data...")
data = pd.read_csv('AAPL_2024.csv', index_col='Date', parse_dates=True)

# 3. INITIALIZE ENGINE
print("Initializing trading engine...")
params = TradingParameters()
engine = TradingEngine(params, initial_capital=100000.0)

# 4. CREATE RESULTS TRACKER
trades = []
daily_pnl = []

# 5. SIMULATION LOOP
print("Running simulation...")
for date, row in data.iterrows():
    # Prepare market data
    market_data = {
        'Close': row['Close'],
        'RSI_14': row['RSI'],
        'SMA_20': row['SMA_20'],
        'EMA_10': row['EMA_10'],
        'EMA_20': row['EMA_20'],
        'Volatility_20': row['Volatility']
    }
    
    # Get prediction from model
    predicted_price = model.predict(market_data)[0]
    
    # Execute trading logic
    signal, executed = engine.process_prediction(
        current_price=row['Close'],
        predicted_price=predicted_price,
        market_data=market_data,
        current_date=date
    )
    
    # Track results
    status = engine.get_portfolio_status()
    daily_pnl.append(status['total_pnl'])
    
    if signal:
        trades.append({
            'date': date,
            'signal': signal,
            'price': row['Close'],
            'pnl': status['total_pnl']
        })

# 6. SAVE LOGS
print("\nSaving logs...")
engine.trade_logger.save_logs('./backtest_results')

# 7. PRINT RESULTS
print("\n" + "="*60)
print("BACKTEST COMPLETE")
print("="*60)

final_status = engine.get_portfolio_status()
summary = engine.trade_logger.get_summary()

print(f"Initial Capital:    ${params.initial_capital:,.2f}")
print(f"Final Portfolio:    ${final_status['portfolio_value']:,.2f}")
print(f"Total P&L:          ${final_status['total_pnl']:,.2f}")
print(f"Return:             {final_status['return_percent']:.2%}")
print(f"\nTotal Trades:       {summary['total_trades']}")
print(f"Winning Trades:     {summary['winning_trades']}")
print(f"Win Rate:           {summary['win_rate']:.1f}%")
print(f"Avg Trade P&L:      ${summary['avg_pnl']:.2f}")
print(f"Max Drawdown:       {final_status.get('max_drawdown', 0):.2%}")
print("="*60)

# 8. SAVE SUMMARY
import json
with open('backtest_summary.json', 'w') as f:
    json.dump({
        'final_portfolio': final_status['portfolio_value'],
        'total_pnl': final_status['total_pnl'],
        'return_percent': final_status['return_percent'],
        'total_trades': summary['total_trades'],
        'win_rate': summary['win_rate']
    }, f, indent=2)

print("\nResults saved to backtest_summary.json")
```

---

## Questions & Support

For issues integrating with your model:
1. Check market_data keys match exactly
2. Verify prediction output is a single float
3. Ensure date/time handling is consistent
4. Check logs in `execution_summary.json`

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Status:** Production Ready
