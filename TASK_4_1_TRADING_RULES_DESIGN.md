# TASK 4.1: DESIGN RULE-BASED TRADING LOGIC
## AI Trading System - AAPL Stock Trading Rules Design

**Document Version:** 1.0  
**Date:** February 28, 2026  
**Task:** 4.1 - Design Rule-Based Trading Logic  
**Time Period:** Feb 28 – Mar 4  
**Status:** DESIGN PHASE  

---

## Executive Summary

This document outlines the comprehensive rule-based trading logic for the AI Trading System. The system uses price predictions from the Linear Regression model (93.16% accuracy) to generate buy/sell signals, with risk management features including stop-loss, take-profit thresholds, and intelligent position sizing.

### Key Components:
- **Buy/Sell Rules:** Clear conditions for market entry and exit
- **Risk Management:** Stop-loss and take-profit mechanisms
- **Position Sizing:** Risk-based allocation strategy
- **Validation:** Historical backtesting requirements

---

## 1. TRADING RULES SPECIFICATION

### 1.1 Market Entry Conditions (BUY Signal)

**Primary Rule: Price Appreciation Expectation**

```
IF predicted_price > current_price * (1 + BUY_THRESHOLD)
    AND recent_trend == UPTREND
    AND risk_score < RISK_LIMIT
    AND portfolio_available_cash >= minimum_allocation
THEN
    Generate BUY signal
END IF
```

**Parameters:**
- **BUY_THRESHOLD:** 2.0% (default)
  - Interpretation: Buy when model predicts ≥2% daily price increase
  - Rationale: Model has 0.97% MAPE error; 2% threshold filters noise
  - Conservative approach: Uses only strong signals
  
- **UPTREND Validation:** 
  - RSI (14) > 50 OR
  - Current_Price > SMA(20) OR
  - EMA(10) > EMA(20)
  - Rationale: Confirms momentum in predicted direction
  
**Example Scenario:**
```
- Current AAPL Price: $200.00
- Predicted Next-Day Price: $204.10 (model prediction)
- BUY Threshold = 2%
- Required Price = $200.00 × 1.02 = $204.00
- Since $204.10 > $204.00 ✓ AND Uptrend confirmed ✓
→ GENERATE BUY SIGNAL
```

### 1.2 Market Exit Conditions (SELL Signals)

#### Exit Type 1: Take-Profit (TP)
```
IF current_price > entry_price * (1 + TAKE_PROFIT_TARGET)
    AND position_age > MINIMUM_HOLD_DAYS
THEN
    Generate SELL signal (PROFIT_TARGET)
END IF
```

**Parameters:**
- **TAKE_PROFIT_TARGET:** 2.5% (default)
  - Rationale: Lock profits when model's expected gain is achieved
  - Models predict ~1-2% daily changes; 2.5% TP captures strong moves
  
- **MINIMUM_HOLD_DAYS:** 1 day
  - Allows at least one prediction cycle
  - Prevents immediate exits

#### Exit Type 2: Stop-Loss (SL)
```
IF current_price < entry_price * (1 - STOP_LOSS_PERCENT)
    OR portfolio_loss_percentage > PORTFOLIO_MAX_LOSS
THEN
    Generate SELL signal (STOP_LOSS)
    Priority: Immediate execution
END IF
```

**Parameters:**
- **STOP_LOSS_PERCENT:** 1.5% (default)
  - Risk Tolerance: Lose max 1.5% per trade
  - Ratio: TP/SL = 2.5%/1.5% = 1.67:1 (favorable risk-reward)
  
- **PORTFOLIO_MAX_LOSS:** -5.0% (default)
  - Entire portfolio protection; circuit breaker
  - Stop all trading if portfolio down 5% from start
  - Prevents cascading losses

#### Exit Type 3: Signal Reversal (Trend Reversal)
```
IF predicted_price < current_price * (1 - SELL_THRESHOLD)
    AND recent_downtrend confirmed
    AND position_duration > HOLD_MINIMUM_BARS
THEN
    Generate SELL signal (REVERSAL)
END IF
```

**Parameters:**
- **SELL_THRESHOLD:** 2.0% (default)
  - Mirror of BUY_THRESHOLD for symmetry
  - Sell when model predicts ≥2% daily decrease
  
- **DOWNTREND Validation:**
  - RSI (14) < 50 OR
  - Current_Price < SMA(20) OR
  - EMA(10) < EMA(20)

#### Exit Priority Order:
1. **Stop-Loss** (highest priority - protect capital)
2. **Portfolio Max Loss** (circuit breaker)
3. **Take-Profit** (lock gains)
4. **Signal Reversal** (react to prediction change)

**Example Exit Scenario:**
```
Entry: Bought at $200.00
Day 1: Price → $201.50 (TP not hit: <$205.00)
Day 2: Price → $205.20 (TP HIT: ✓) 
  Gain = 2.6% > 2.5% target
  SELL executed at market → $205.20
  Profit: $5.20 per share
```

---

## 2. POSITION SIZING STRATEGY

### 2.1 Risk-Based Position Sizing

**Core Formula:**
```
position_size = (portfolio_value × RISK_PERCENTAGE) / (entry_price × SL_PERCENT)

Where:
  - portfolio_value: Total capital available
  - RISK_PERCENTAGE: % of portfolio risked per trade
  - entry_price: Purchase price
  - SL_PERCENT: Stop-loss threshold (1.5%)
```

**Example Calculation:**
```
Portfolio Value: $100,000
Risk Per Trade: 2% (RISK_PERCENTAGE)
Entry Price: $200.00
Stop Loss: 1.5%

Position Size = ($100,000 × 0.02) / ($200.00 × 0.015)
              = $2,000 / $3.00
              = 666 shares

Max Risk per Trade = 666 × $200 × 0.015 = $1,999.80 ≈ 2% of portfolio
```

### 2.2 Position Sizing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **RISK_PERCENTAGE** | 2% | Max % of portfolio risked per trade |
| **MAX_POSITION_VALUE** | 15% | Single position ≤ 15% of portfolio |
| **MAX_CASH_ALLOCATION** | 50% | Keep ≥50% in cash for flexibility |
| **MIN_POSITION_SIZE** | 10 shares | Lower bound on shares per trade |

### 2.3 Multi-Position Management

```
Total Portfolio Risk = SUM(position_risk_per_trade)

IF total_risk > PORTFOLIO_RISK_LIMIT (5%)
THEN
    Reduce position size for new entries
    or skip signal until risk reduces
END IF

Max Concurrent Positions: 3 (diversification)
```

---

## 3. STOP-LOSS & TAKE-PROFIT MECHANICS

### 3.1 Stop-Loss Implementation

**Dynamic Stop-Loss (Optional Enhancement):**
```
IF position_age > 3 days AND unrealized_gain > 1%
THEN
    trailing_stop = current_price × (1 - TRAILING_STOP_PERCENT)
    stop_loss_price = MAX(initial_stop_loss_price, trailing_stop)
    // Protects gains while allowing upside
END IF
```

**Parameters:**
- **Initial Stop-Loss:** 1.5% below entry
- **Trailing Stop:** 0.75% (optional, only if position profitable)
- **Recalculation Frequency:** Daily

### 3.2 Take-Profit Levels (Multi-Level Strategy)

**Two-Tier Take-Profit (Optional):**
```
Position Size: 100 shares (example)

Level 1: 50 shares at +2.0% (take quick profit)
Level 2: 50 shares at +2.5% (hold for full target)
Remainder: Exit on stop-loss or reversal signal
```

### 3.3 Execution Rules

**Order Execution Priority:**
1. Market order on SL breach (no slippage penalty)
2. Limit order on TP (0.05% buffer below target)
3. Market order on signal reversal (captures reversal early)

---

## 4. SIGNAL FILTERING & CONFIRMATION

### 4.1 Noise Filtering

**Volatility Filter:**
```
IF daily_volatility > VOLATILITY_THRESHOLD (3%)
THEN
    Increase required threshold:
    BUY_THRESHOLD = 3.0% (instead of 2%)
    SELL_THRESHOLD = 3.0% (instead of 2%)
    // Larger moves needed when market is volatile
END IF
```

### 4.2 Trend Confirmation

**Multiple Timeframe Confirmation:**
```
IF prediction_signal == BUY
    AND SMA(10) > SMA(20) > SMA(50)  // All moving averages aligned
    AND RSI(14) > 40 AND RSI(14) < 80  // Momentum zone
    AND volume > average_volume * 1.2  // Volume support
THEN
    Confidence = HIGH
    Position_Size = STANDARD
ELSE IF prediction_signal == BUY (weak confirmation)
    Position_Size = REDUCED (75% of standard)
    Confidence = MEDIUM
END IF
```

---

## 5. PSEUDOCODE: COMPLETE TRADING LOGIC FLOW

```
FUNCTION ExecuteTradingDay(current_date, market_data, model):
    
    // Step 1: Update portfolio and risk metrics
    UpdatePortfolioMetrics()
    CalculateTotalPortfolioRisk()
    
    // Step 2: Get model prediction
    predicted_price = model.Predict(market_data)
    confidence = CalculateConfidence(predicted_price)
    
    // Step 3: Check existing positions for exit signals
    FOR EACH position IN active_positions:
        
        current_price = market_data.Close
        unrealized_gain = (current_price - position.entry_price) / position.entry_price
        position_age = current_date - position.entry_date
        
        // Check Exit Conditions (Priority Order)
        exit_signal = NONE
        
        // Priority 1: Stop Loss
        IF current_price <= position.stop_loss_price:
            exit_signal = STOP_LOSS
            exit_price = position.stop_loss_price
            
        // Priority 2: Portfolio Circuit Breaker
        ELSE IF portfolio_loss > -5%:
            exit_signal = PORTFOLIO_LOSS
            exit_price = current_price (market order)
            
        // Priority 3: Take-Profit
        ELSE IF unrealized_gain >= TAKE_PROFIT_TARGET 
                 AND position_age >= MINIMUM_HOLD_DAYS:
            exit_signal = TAKE_PROFIT
            exit_price = position.take_profit_price
            
        // Priority 4: Reversal Signal
        ELSE IF predicted_price < current_price * (1 - SELL_THRESHOLD)
                 AND position_age >= MINIMUM_HOLD_DAYS
                 AND NOT_DOWNTREND_CONFIRMED:
            exit_signal = REVERSAL
            exit_price = current_price (market order)
        
        // Execute Exit
        IF exit_signal != NONE:
            ClosePosition(position, exit_signal, exit_price)
            LogTrade(position.entry_date, position.entry_price,
                    current_date, exit_price, exit_signal)
        END IF
    END FOR
    
    // Step 4: Check for new entry signals
    IF available_cash > minimum_investment
            AND total_portfolio_risk < PORTFOLIO_RISK_LIMIT
            AND num_active_positions < MAX_POSITIONS:
        
        price_appreciation = (predicted_price - current_price) / current_price
        
        // Check BUY conditions
        IF price_appreciation > BUY_THRESHOLD
                AND CheckUptrendConfirmation(market_data)
                AND confidence > CONFIDENCE_THRESHOLD:
            
            position_size = CalculatePositionSize(available_cash)
            entry_price = current_price
            stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
            take_profit = entry_price * (1 + TAKE_PROFIT_TARGET)
            
            // Execute BUY
            ExecuteBuyOrder(position_size, entry_price, 
                           stop_loss, take_profit)
            LogSignal(current_date, BUY, entry_price, confidence)
            
        // Check SELL conditions (short selling or exit logic)
        ELSE IF price_appreciation < -SELL_THRESHOLD
                 AND CheckDowntrendConfirmation(market_data)
                 AND confidence > CONFIDENCE_THRESHOLD:
            
            // Log signal but don't short (conservative approach)
            LogSignal(current_date, SELL, current_price, confidence)
        
        END IF
    END IF
    
    // Step 5: Log daily metrics
    LogDailyMetrics(current_date, portfolio_value, cash, 
                    num_positions, unrealized_gain)
    
END FUNCTION


FUNCTION CalculatePositionSize(available_cash):
    
    risk_amount = portfolio_value × RISK_PERCENTAGE
    position_size_dollars = risk_amount / STOP_LOSS_PERCENT
    
    // Constraints
    max_allocation = portfolio_value × MAX_POSITION_VALUE
    position_size_dollars = MIN(position_size_dollars, max_allocation)
    
    shares = FLOOR(position_size_dollars / current_price)
    shares = MAX(shares, MIN_POSITION_SIZE)
    
    RETURN shares
    
END FUNCTION


FUNCTION CheckUptrendConfirmation(market_data):
    
    rsi = market_data.RSI_14
    current_price = market_data.Close
    sma_20 = market_data.SMA_20
    ema_10 = market_data.EMA_10
    ema_20 = market_data.EMA_20
    
    confirmations = 0
    IF rsi > 50:
        confirmations += 1
    IF current_price > sma_20:
        confirmations += 1
    IF ema_10 > ema_20:
        confirmations += 1
    
    RETURN confirmations >= 2  // Need at least 2 confirmations
    
END FUNCTION
```

---

## 6. VALIDATION AGAINST HISTORICAL SCENARIOS

### 6.1 Scenario Testing Approach

**Test Set:** Historical AAPL data (May 2024 - Present, 160 trading days)

**Metrics to Calculate:**
1. **Win Rate:** % of profitable trades
2. **Profit Factor:** Sum(wins) / Sum(losses)
3. **Max Drawdown:** Largest peak-to-trough decline
4. **Sharpe Ratio:** Risk-adjusted returns
5. **Total Return:** % return over test period
6. **Avg Trade Duration:** Days in position
7. **Average P/L per trade:** Mean profit/loss

### 6.2 Test Scenarios

**Scenario 1: Normal Market (May-Jun 2024)**
- Markets stable, moderate gains
- Expected: Moderate win rate (55-65%), small gains

**Scenario 2: Strong Uptrend (Jun-Jul 2024)**
- Stock rising consistently (AAPL +5% in Jun-Jul)
- Expected: High win rate (65-75%), larger gains

**Scenario 3: Volatility (July 2024)**
- Price swings, uncertain direction
- Expected: Lower win rate (45-55%), increased SL hits

**Scenario 4: Downtrend Protection**
- Market declining
- Expected: System exits on reversal signals, limits losses

### 6.3 Acceptance Criteria

| Metric | Minimum | Target | Description |
|--------|---------|--------|-------------|
| **Win Rate** | 45% | 55%+ | Profitable trades > losses |
| **Profit Factor** | 0.8 | 1.2+ | Gains ≥ losses |
| **Max Drawdown** | -15% | -8% | Portfolio decline limit |
| **Sharpe Ratio** | 0.5 | 1.0+ | Risk-adjusted return |
| **Avg Winning Trade** | 0.5% | 2%+ | Per-trade profit |
| **Avg Losing Trade** | -1.5% | -1.2% | Per-trade loss limit |
| **Trade Duration** | N/A | 1-3 days | Optimal holding period |

### 6.4 Risk-Adjusted Validation

```
Criteria for PASSING validation:

IF win_rate >= 45%
    AND profit_factor >= 0.8
    AND max_drawdown <= -15%
    AND sharpe_ratio >= 0.5
    AND avg_win > abs(avg_loss)
THEN
    System qualifies for LIVE trading with position sizing
ELSE
    Adjust parameters and re-validate
END IF
```

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Core Trading Logic (Task 4.2)
- [ ] Implement TradingRules class
- [ ] Implement PositionSizingCalculator
- [ ] Implement RiskManager
- [ ] Create basic backtesting framework

### Phase 2: Validation & Backtesting (Task 4.3)
- [ ] Historical scenario testing
- [ ] Performance metrics calculation
- [ ] Parameter sensitivity analysis
- [ ] Optimization of thresholds

### Phase 3: Integration & Testing (Task 4.4)
- [ ] Integration with model predictions
- [ ] Mock trading simulation
- [ ] Edge case handling
- [ ] Error handling & logging

### Phase 4: Deployment Preparation (Task 4.5)
- [ ] Paper trading validation
- [ ] Live trading with limits
- [ ] Monitoring & alerting system
- [ ] Documentation & handoff

---

## 8. PARAMETER REFERENCE TABLE

| Parameter | Value | Range | Description |
|-----------|-------|-------|-------------|
| **BUY_THRESHOLD** | 2.0% | 1.5%-3.0% | Min price appreciation for buy |
| **SELL_THRESHOLD** | 2.0% | 1.5%-3.0% | Min price decline for sell |
| **TAKE_PROFIT_TARGET** | 2.5% | 1.5%-4.0% | Profit target per trade |
| **STOP_LOSS_PERCENT** | 1.5% | 1.0%-2.5% | Max loss per trade |
| **RISK_PERCENTAGE** | 2.0% | 1.0%-3.0% | % portfolio risked/trade |
| **PORTFOLIO_MAX_LOSS** | -5.0% | -3.0% to -8.0% | Circuit breaker |
| **MAX_POSITION_VALUE** | 15% | 10%-20% | Single position size |
| **MAX_CASH_ALLOCATION** | 50% | 40%-70% | Min cash reserve |
| **MIN_POSITION_SIZE** | 10 | 5-20 | Min shares per trade |
| **MINIMUM_HOLD_DAYS** | 1 | 1-3 | Earliest exit day |
| **MAX_POSITIONS** | 3 | 1-5 | Concurrent positions |
| **VOLATILITY_THRESHOLD** | 3.0% | 2.0%-4.0% | Increases thresholds |
| **TRAILING_STOP_PERCENT** | 0.75% | 0.5%-1.0% | Dynamic SL |

---

## 9. RISK MANAGEMENT SUMMARY

**Tier 1: Per-Trade Risk**
- Stop-loss at 1.5% loss limit
- Risk-based position sizing
- Profit target at 2.5% gain

**Tier 2: Portfolio Risk**
- Max 2% risk per trade
- Single position ≤15% of portfolio
- Max 3 concurrent positions

**Tier 3: Overall Portfolio Risk**
- Circuit breaker at -5% portfolio loss
- Minimum 50% cash reserve
- Mandatory exit on max loss

**Tier 4: Market Risk**
- Volatility filter (3%+) adjusts thresholds
- Trend confirmation required
- Multi-timeframe validation

---

## 10. NEXT STEPS

✅ Task 4.1: Design complete
→ Task 4.2: Implement trading logic module
→ Task 4.3: Validate with historical backtesting
→ Task 4.4: Integration testing
→ Task 4.5: Deployment & monitoring

---

**Document Status:** COMPLETE - READY FOR IMPLEMENTATION
**Review Date:** March 4, 2026
**Approved for:** Task 4.2 Implementation
