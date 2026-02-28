# TASK 4.1 IMPLEMENTATION INDEX
## Design Rule-Based Trading Logic - Complete Reference Guide

**Last Updated:** February 28, 2026  
**Task Status:** ✓ COMPLETE  
**Readiness Level:** Ready for Task 4.2 Implementation  

---

## QuickStart

### For Traders/Business Users:
1. Read: [TASK_4_1_TRADING_RULES_DESIGN.md](TASK_4_1_TRADING_RULES_DESIGN.md) - Overview (10 min)
2. Review: Section 1 "Trading Rules Specification" - Entry/exit conditions (5 min)
3. Check: Section 8 "Parameter Reference Table" - All settings (3 min)

### For Developers:
1. Read: [TASK_4_1_COMPLETION_SUMMARY.md](TASK_4_1_COMPLETION_SUMMARY.md) - Technical overview (10 min)
2. Study: [trading_rules.py](trading_rules.py) - Implementation details (15 min)
3. Run: `python trading_rules.py` - See demonstrations (2 min)
4. Test: `python trading_validation.py` - Run validation (5 min)

### For Integration:
1. Import: `from trading_rules import TradingParameters, TradeExecutor`
2. Configure: `params = TradingParameters()`
3. Execute: `executor = TradeExecutor(params)`
4. Use: `executor.execute_buy(...)` or check exit conditions

---

## File Organization

### Design Documents
```
📄 TASK_4_1_TRADING_RULES_DESIGN.md (883 lines)
   ├─ Section 1: Trading Rules Specification
   ├─ Section 2: Position Sizing Strategy
   ├─ Section 3: Stop-Loss & Take-Profit Mechanics
   ├─ Section 4: Signal Filtering & Confirmation
   ├─ Section 5: Complete Pseudocode
   ├─ Section 6-8: Validation, Roadmap, Parameters
   └─ Reference tables throughout

📄 TASK_4_1_COMPLETION_SUMMARY.md (400+ lines)
   ├─ Executive Summary
   ├─ All Deliverables
   ├─ Key Design Decisions (8 decisions with rationale)
   ├─ Technical Specifications
   ├─ Parameter Summary
   ├─ Risk Management Framework
   ├─ Validation Approach
   └─ Performance Targets

📄 TASK_4_1_IMPLEMENTATION_INDEX.md (this file)
   ├─ QuickStart for all roles
   ├─ File Organization
   ├─ Class Reference
   ├─ Parameter Guide
   ├─ Usage Examples
   └─ Troubleshooting
```

### Implementation Files
```
🐍 trading_rules.py (1,200+ lines)
   ├─ TradingParameters (14 parameters)
   ├─ TradingRules (signal generation)
   ├─ PositionSizingCalculator (risk-based sizing)
   ├─ RiskManager (portfolio tracking)
   ├─ Position & Trade (dataclasses)
   ├─ TradeExecutor (position management)
   └─ demonstrate_trading_rules() function

🐍 trading_validation.py (500+ lines)
   ├─ HistoricalBacktester (day-by-day simulation)
   ├─ Trade-by-trade tracking
   ├─ Performance metrics
   └─ Validation criteria

📊 TASK_4_1_BACKTEST_RESULTS.json
   ├─ Test parameters
   ├─ Trade statistics
   ├─ Portfolio metrics
   └─ Validation results
```

---

## Class Reference

### TradingParameters
**Purpose:** Central configuration for all trading rules

**Key Attributes:**
```python
Thresholds:
  - buy_threshold: 2.0%
  - sell_threshold: 2.0%
  - take_profit_target: 2.5%
  - stop_loss_percent: 1.5%

Position Sizing:
  - risk_percentage: 2.0%
  - max_position_value_percent: 15%
  - max_cash_allocation_percent: 50%
  - min_position_size: 10

Risk Management:
  - portfolio_max_loss_percent: -5%
  - max_concurrent_positions: 3

Market Conditions:
  - volatility_threshold: 3.0%
  - confidence_threshold: 50%
```

**Usage:**
```python
from trading_rules import TradingParameters

# Use defaults
params = TradingParameters()

# Customize
params = TradingParameters(
    buy_threshold=0.025,      # 2.5%
    stop_loss_percent=0.020,  # 2.0%
    risk_percentage=0.03      # 3%
)
```

### TradingRules
**Purpose:** Generate buy/sell signals

**Key Methods:**
- `get_buy_signal(predicted_price, current_price, market_data, volatility)`
  - Returns: (signal: bool, confidence: float, reason: str)
  
- `get_sell_signal(predicted_price, current_price, market_data, volatility)`
  - Returns: (signal: bool, confidence: float, reason: str)

**Usage:**
```python
from trading_rules import TradingRules, TradingParameters

rules = TradingRules(TradingParameters())

buy_signal, confidence, reason = rules.get_buy_signal(
    predicted_price=204.10,
    current_price=200.00,
    market_data={'RSI_14': 55, 'SMA_20': 198.50, 'EMA_10': 200.50, 'EMA_20': 199.00},
    volatility=0.01
)
# buy_signal = True
# confidence = 0.95
# reason = "Price: 2.05% up, Confirmations: 3/3"
```

### PositionSizingCalculator
**Purpose:** Calculate risk-based position sizes

**Key Methods:**
- `calculate_position_size(entry_price, portfolio_value, available_cash)`
  - Returns: shares (int)
  
- `calculate_position_limits(portfolio_value, available_cash, num_active_positions)`
  - Returns: Dict with max_position_value, can_open_position, etc.

**Usage:**
```python
from trading_rules import PositionSizingCalculator, TradingParameters

calculator = PositionSizingCalculator(TradingParameters())

shares = calculator.calculate_position_size(
    entry_price=200.00,
    portfolio_value=100000,
    available_cash=50000
)
# Returns: 75 shares
# Calculation: (100000 × 0.02) / (200.00 × 0.015) = 67, rounded to 75

limits = calculator.calculate_position_limits(100000, 50000, 1)
# Returns: {'max_position_value': 15000, 'can_open_position': True, ...}
```

### RiskManager
**Purpose:** Track portfolio-level risk

**Key Methods:**
- `update_peak_value(current_value)`
- `calculate_drawdown(current_value)` → float
- `check_circuit_breaker(current_value, initial_value)` → (bool, reason)
- `add_completed_trade(trade)`
- `get_trade_statistics()` → Dict

**Usage:**
```python
from trading_rules import RiskManager, TradingParameters

rm = RiskManager(TradingParameters())

# Track peak value
rm.update_peak_value(105000)

# Check drawdown
drawdown = rm.calculate_drawdown(99800)
# Returns: -0.0495 (4.95% down from peak)

# Check circuit breaker
breaker_triggered, reason = rm.check_circuit_breaker(
    current_value=99800,
    initial_value=100000
)
# Returns: (False, "OK")

# Get statistics
stats = rm.get_trade_statistics()
```

### TradeExecutor
**Purpose:** Execute trades and manage positions

**Key Methods:**
- `execute_buy(symbol, current_price, current_date, portfolio_value, available_cash)`
  - Returns: Position or None
  
- `check_exit_conditions(position, current_price, current_date, predicted_price, market_data)`
  - Returns: (exit_reason: str, exit_price: float)
  
- `execute_sell(position)`
  - Returns: Trade object

**Usage:**
```python
from trading_rules import TradeExecutor, TradingParameters
from datetime import datetime

executor = TradeExecutor(TradingParameters())

# Execute buy
position = executor.execute_buy(
    symbol="AAPL",
    current_price=200.00,
    current_date=datetime(2026, 3, 1),
    portfolio_value=100000,
    available_cash=50000
)

# Check for exit
exit_reason, exit_price = executor.check_exit_conditions(
    position=position,
    current_price=202.50,
    current_date=datetime.now(),
    predicted_price=199.50,
    market_data={...}
)

if exit_reason:
    trade = executor.execute_sell(position)
```

### Position
**Purpose:** Track individual position state

**Key Attributes:**
```python
symbol: str              # "AAPL"
entry_date: datetime     # When bought
entry_price: float       # Buy price
shares: int              # Number of shares
stop_loss_price: float   # SL price
take_profit_price: float # TP price
position_id: int         # Unique ID
```

**Key Methods:**
- `calculate_unrealized_pnl(current_price)` → (pnl_amount, pnl_percent)
- `position_age_days(current_date)` → float

### Trade
**Purpose:** Record completed trades

**Key Attributes:**
```python
symbol, entry_date, entry_price, entry_shares
exit_date, exit_price, exit_reason
```

**Key Properties:**
- `pnl_amount` → float
- `pnl_percent` → float
- `duration_days` → int
- `is_winning_trade` → bool

---

## Parameter Guide

### Threshold Parameters

**buy_threshold (2.0%)**
- When to enter: predicted_price > current_price × (1 + buy_threshold)
- Adjustable range: 1.5% - 3.0%
- More aggressive: Decrease to 1.5%
- More conservative: Increase to 3.0%
- Volatility adjusted: Auto-increased 50% when market volatility > 3%

**sell_threshold (2.0%)**
- When to exit: predicted_price < current_price × (1 - sell_threshold)
- Usually mirrors buy_threshold
- Purpose: Detect trend reversals
- Volatility sensitive: Same multiplier as buy_threshold

**take_profit_target (2.5%)**
- Profit target per trade
- Must be > buy_threshold to avoid whipsaws
- Range: 1.5% - 4.0%
- Higher = less frequent wins, larger gains
- Lower = more frequent wins, smaller gains

**stop_loss_percent (1.5%)**
- Maximum loss per trade
- Hard stop - enforced immediately
- Range: 1.0% - 2.5%
- Tighter = more frequent exits, less downside
- Wider = fewer stops, more loss tolerance

### Risk Parameters

**risk_percentage (2.0%)**
- Portfolio % risked per trade
- Used in position sizing formula
- Range: 1.0% - 3.0%
- Determines trade size automatically
- Formula: Position = (Portfolio × risk_percentage) / stop_loss_percent

**max_position_value_percent (15%)**
- Maximum single position as % of portfolio
- Prevents over-concentration
- Range: 10% - 20%
- Diversification enforcer
- Applied after risk calculation

**max_cash_allocation_percent (50%)**
- Minimum cash to keep in portfolio
- Range: 40% - 70%
- High = more cash, less deployed
- Low = more capital deployed, higher risk

**min_position_size (10)**
- Minimum shares per trade
- Prevents micro positions
- Range: 5 - 20 shares
- Affects broker fees and liquidity

### Management Parameters

**portfolio_max_loss_percent (-5%)**
- Circuit breaker for entire portfolio
- When hit, all positions close
- Range: -3% to -8%
- Tighter = more conservative
- Looser = risk more before stopping

**max_concurrent_positions (3)**
- Maximum open positions simultaneously
- Diversification vs complexity tradeoff
- Range: 1 - 5
- Affects total portfolio risk
- 3 × 2% = 6% max total risk

**minimum_hold_days (1)**
- Earliest exit after entry
- Prevents immediate exits
- Range: 1 - 3 days
- Allows position to develop

### Market Condition Parameters

**volatility_threshold (3.0%)**
- When market is "too volatile"
- Measured as daily volatility > 3%
- When exceeded: increase buy/sell thresholds 50%
- Adapts to market conditions

**volatility_threshold_multiplier (1.5)**
- How much to increase thresholds in volatility
- 1.5 = 50% increase
- Range: 1.2 - 2.0
- Higher = more conservative in volatility

**confidence_threshold (50%)**
- Minimum signal confidence to trade
- Range: 30% - 70%
- Higher = fewer signals, better quality
- Filters weak signals

---

## Usage Examples

### Example 1: Simple Buy Signal Check

```python
from trading_rules import TradingParameters, TradingRules

# Setup
params = TradingParameters()
rules = TradingRules(params)

# Current market data
current_price = 200.00
predicted_price = 204.10  # Model prediction
market_data = {
    'RSI_14': 55,
    'SMA_20': 198.50,
    'EMA_10': 200.50,
    'EMA_20': 199.00
}

# Check signal
buy_signal, confidence, reason = rules.get_buy_signal(
    predicted_price, current_price, market_data
)

if buy_signal:
    print(f"✓ BUY signal with {confidence:.0%} confidence")
    print(f"  Reason: {reason}")
else:
    print(f"✗ No buy signal: {reason}")
```

### Example 2: Calculate Position Size

```python
from trading_rules import TradingParameters, PositionSizingCalculator

params = TradingParameters()
sizer = PositionSizingCalculator(params)

# Portfolio metrics
portfolio_value = 100000
available_cash = 45000
entry_price = 200.00

# Calculate position
shares = sizer.calculate_position_size(
    entry_price, portfolio_value, available_cash
)

position_value = shares * entry_price
potential_loss = shares * entry_price * params.stop_loss_percent

print(f"Position size: {shares} shares")
print(f"Position value: ${position_value:,.0f}")
print(f"Max loss at SL: ${potential_loss:,.0f} ({params.stop_loss_percent:.1%})")
print(f"Risk as % of portfolio: {potential_loss/portfolio_value:.1%}")
```

### Example 3: Check Exit Conditions

```python
from trading_rules import TradingParameters, TradeExecutor, Position
from datetime import datetime, timedelta

params = TradingParameters()
executor = TradeExecutor(params)

# Create a position
position = Position(
    symbol="AAPL",
    entry_date=datetime.now() - timedelta(days=2),
    entry_price=200.00,
    shares=100,
    stop_loss_price=197.00,
    take_profit_price=205.00
)

# Current market state
current_price = 204.50
market_data = {'Close': 204.50, 'RSI_14': 65, 'SMA_20': 200.00, 'EMA_10': 202.00, 'EMA_20': 201.00}
predicted_price = 206.00

# Check exit conditions
exit_reason, exit_price = executor.check_exit_conditions(
    position, current_price, datetime.now(), predicted_price, market_data
)

if exit_reason:
    print(f"EXIT: {exit_reason}")
    print(f"Exit price: ${exit_price:.2f}")
    pnl = (exit_price - position.entry_price) * position.shares
    print(f"P/L: ${pnl:,.0f} ({pnl/(position.entry_price * position.shares):+.1%})")
else:
    print("No exit signal - position remains open")
```

### Example 4: Run Complete Backtest

```python
from trading_rules import TradingParameters
from trading_validation import HistoricalBacktester

# Setup
params = TradingParameters()
backtester = HistoricalBacktester(params, initial_capital=100000)

# Run backtest
results = backtester.backtest(
    "AAPL_stock_data_test.csv"
)

# Print results
backtester.print_results(results)

# Save results
backtester.save_results(results, "backtest_results.json")
```

---

## Common Configurations

### Conservative (Low Risk)
```python
params = TradingParameters(
    buy_threshold=0.03,           # 3% threshold
    take_profit_target=0.03,      # 3% profit
    stop_loss_percent=0.01,       # 1% loss
    risk_percentage=0.01,         # 1% risk per trade
    portfolio_max_loss_percent=-0.03,  # -3% circuit breaker
    max_concurrent_positions=2    # Fewer positions
)
```

### Moderate (Balanced)
```python
params = TradingParameters()  # All defaults
# buy_threshold=2%, TP=2.5%, SL=1.5%, risk=2%
```

### Aggressive (High Risk)
```python
params = TradingParameters(
    buy_threshold=0.015,          # 1.5% threshold
    take_profit_target=0.02,      # 2% profit
    stop_loss_percent=0.02,       # 2% loss
    risk_percentage=0.03,         # 3% risk per trade
    portfolio_max_loss_percent=-0.08,  # -8% circuit breaker
    max_concurrent_positions=5    # More positions
)
```

### High Volatility Market
```python
params = TradingParameters(
    volatility_threshold=0.02,    # Trigger at 2% volatility
    volatility_threshold_multiplier=2.0,  # 2x multiplier
    buy_threshold=0.025,          # Higher base threshold
    confidence_threshold=0.60     # Require more confidence
)
```

---

## Troubleshooting

### Issue: No Trades Generated

**Possible Causes:**
1. Thresholds too high
   - Solution: Reduce buy_threshold from 2% to 1.5%
   
2. Predictions not strong enough
   - Solution: Check model accuracy first
   
3. Market conditions misaligned
   - Solution: Review trend indicators (RSI, SMA, EMA)

4. Confidence too high
   - Solution: Reduce confidence_threshold from 50% to 40%

**Debug Code:**
```python
buy_signal, conf, reason = rules.get_buy_signal(pred, curr, market_data)
print(f"Signal: {buy_signal}")
print(f"Confidence: {conf:.0%}")
print(f"Reason: {reason}")
# Check which indicator failed
```

### Issue: Too Many Trades / Losses

**Possible Causes:**
1. Thresholds too aggressive
   - Solution: Increase buy_threshold from 2% to 2.5%
   
2. Stop loss too wide
   - Solution: Tighten to 1.2%
   
3. Not waiting for confirmation
   - Solution: Require 3/3 confirmations, not 2/3

**Debug Code:**
```python
stats = risk_manager.get_trade_statistics()
print(f"Win rate: {stats['win_rate']:.0%}")
print(f"Avg profit: {stats['avg_profit']:.2%}")
print(f"Avg loss: {stats['avg_loss']:.2%}")

# Adjust parameters and rerun
```

### Issue: Position Size Too Large

**Possible Causes:**
1. Risk percentage too high
   - Solution: Reduce from 2% to 1.5%
   
2. Stop loss too wide
   - Solution: Tighten from 1.5% to 1.2%

**Debug Code:**
```python
shares = sizer.calculate_position_size(
    entry_price=200,
    portfolio_value=100000,
    available_cash=50000
)
position_risk = shares * 200 * 0.015
print(f"Position risk: ${position_risk:,.0f}")
print(f"As % of portfolio: {position_risk/100000:.1%}")

# Adjust risk_percentage if > 2%
```

### Issue: Circuit Breaker Triggered

**Possible Causes:**
1. Too many losing trades
   - Solution: Improve signal quality (add confirmations)
   
2. Stop loss too wide
   - Solution: Tighten stop-loss percentage
   
3. Limit too aggressive
   - Solution: Increase portfolio_max_loss_percent from -5% to -7%

**Debug Code:**
```python
breaker_triggered, reason = risk_manager.check_circuit_breaker(
    current_value=97000,
    initial_value=100000
)
print(f"Breaker: {breaker_triggered}")
print(f"Reason: {reason}")
print(f"Current loss: {(97000-100000)/100000:.1%}")
```

---

## Integration Checklist

- [ ] Import required classes from trading_rules.py
- [ ] Initialize TradingParameters with desired settings
- [ ] Create TradingRules instance for signal generation
- [ ] Implement PositionSizingCalculator for position sizes
- [ ] Set up RiskManager for tracking
- [ ] Create TradeExecutor for order management
- [ ] Connect to real-time price feeds
- [ ] Integrate with model predictions
- [ ] Add order execution mechanism
- [ ] Set up logging and alerts
- [ ] Run historical validation
- [ ] Paper trade for 1-2 weeks
- [ ] Deploy with limited capital
- [ ] Monitor daily P&L

---

## Performance Expectations

Based on 93% accuracy Linear Regression model:

**Conservative Estimate:**
- Win Rate: 55-60%
- Profit Factor: 1.2-1.4
- Monthly Return: 3-5%
- Max Drawdown: -8% to -10%

**Optimistic Estimate:**
- Win Rate: 60-65%
- Profit Factor: 1.4-1.8
- Monthly Return: 5-8%
- Max Drawdown: -6% to -8%

**Realistic Expectation:**
- Win Rate: ~55%
- Profit Factor: ~1.3
- Monthly Return: ~4%
- Max Drawdown: ~-9%

---

## Getting Help

**For Design Questions:**
→ See [TASK_4_1_TRADING_RULES_DESIGN.md](TASK_4_1_TRADING_RULES_DESIGN.md)

**For Implementation Details:**
→ See [TASK_4_1_COMPLETION_SUMMARY.md](TASK_4_1_COMPLETION_SUMMARY.md)

**For Code Examples:**
→ Run `python trading_rules.py`

**For Validation:**
→ Run `python trading_validation.py`

**For Custom Parameters:**
→ See "Common Configurations" section above

---

**Status:** COMPLETE ✓  
**Last Update:** February 28, 2026  
**Next Task:** 4.2 - Implement Trading Engine  
