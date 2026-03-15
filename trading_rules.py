"""
Task 4.1: Rule-Based Trading Logic Implementation
Implements the trading rules designed in TASK_4_1_TRADING_RULES_DESIGN.md

Key Classes:
- TradingRules: Core trading signal generation
- PositionSizingCalculator: Risk-based position sizing
- RiskManager: Portfolio-level risk tracking
- Position: Individual position tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd


# ============================================================================
# DATA CLASSES FOR TRADING STATE
# ============================================================================

@dataclass
class TradingParameters:
    """All trading rule parameters in one place"""
    
    # Entry/Exit Thresholds
    buy_threshold: float = 0.02  # 2%
    sell_threshold: float = 0.02  # 2%
    take_profit_target: float = 0.04  # 4% (wider for profitable swings)
    stop_loss_percent: float = 0.03  # 3% (wider to avoid noise stop-outs)
    
    # Position Sizing
    risk_percentage: float = 0.02  # 2% per trade
    max_position_value_percent: float = 0.15  # 15% of portfolio
    max_cash_allocation_percent: float = 0.60  # Allocate up to 60%
    min_position_size: int = 10  # Minimum shares
    
    # Portfolio Risk
    portfolio_max_loss_percent: float = -0.05  # -5% circuit breaker
    max_concurrent_positions: int = 5  # Support multi-stock portfolio
    
    # Position Management
    minimum_hold_days: int = 2  # Hold ≥2 days to let winners run
    trailing_stop_percent: float = 0.02  # 2% trailing stop
    
    # Market Conditions
    volatility_threshold: float = 0.03  # 3%
    volatility_threshold_multiplier: float = 1.5  # Increase thresholds 50%
    
    # Confidence Requirements
    confidence_threshold: float = 0.52  # 52% min confidence for trade
    
    def __str__(self) -> str:
        """Pretty print all parameters"""
        params_str = "TRADING PARAMETERS:\n"
        params_str += "=" * 50 + "\n"
        for key, value in self.__dict__.items():
            if isinstance(value, float):
                if value < 1:
                    params_str += f"  {key:.<35} {value:>8.2%}\n"
                else:
                    params_str += f"  {key:.<35} {value:>8.2f}\n"
            else:
                params_str += f"  {key:.<35} {value:>8}\n"
        return params_str


@dataclass
class Position:
    """Represents a single open position"""
    
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss_price: float
    take_profit_price: float
    
    # Tracking
    position_id: int = field(default_factory=lambda: np.random.randint(100000, 999999))
    trailing_stop_active: bool = field(default=False)
    current_trailing_stop: float = field(default=0.0)
    
    @property
    def entry_value(self) -> float:
        """Total entry cost"""
        return self.entry_price * self.shares
    
    @property
    def position_age_days(self, current_date: datetime = None) -> float:
        """Days since entry"""
        ref_date = current_date or datetime.now()
        return (ref_date - self.entry_date).days
    
    def calculate_unrealized_pnl(self, current_price: float) -> Tuple[float, float]:
        """Return (PnL amount, PnL percent)"""
        pnl = (current_price - self.entry_price) * self.shares
        pnl_percent = (current_price - self.entry_price) / self.entry_price
        return pnl, pnl_percent
    
    def __str__(self) -> str:
        return (f"Position({self.symbol}, {self.shares} shares @ "
                f"${self.entry_price:.2f}, SL: ${self.stop_loss_price:.2f}, "
                f"TP: ${self.take_profit_price:.2f})")


@dataclass
class Trade:
    """Represents a completed trade"""
    
    symbol: str
    entry_date: datetime
    entry_price: float
    entry_shares: int
    exit_date: datetime
    exit_price: float
    exit_reason: str  # BUY, SELL, STOP_LOSS, TAKE_PROFIT, REVERSAL, etc.
    
    @property
    def duration_days(self) -> int:
        """Days held"""
        return (self.exit_date - self.entry_date).days
    
    @property
    def pnl_amount(self) -> float:
        """Absolute P/L"""
        return (self.exit_price - self.entry_price) * self.entry_shares
    
    @property
    def pnl_percent(self) -> float:
        """Percentage P/L"""
        return (self.exit_price - self.entry_price) / self.entry_price
    
    @property
    def is_winning_trade(self) -> bool:
        """Won or lost"""
        return self.pnl_percent > 0
    
    def __str__(self) -> str:
        return (f"Trade({self.symbol}: "
                f"${self.entry_price:.2f}→${self.exit_price:.2f} "
                f"[{self.pnl_percent:+.2%}] {self.exit_reason})")


# ============================================================================
# CORE TRADING LOGIC CLASSES
# ============================================================================

class TradingRules:
    """
    Core trading rule logic generation and signal creation
    
    Implements:
    - Buy/Sell signal generation
    - Trend confirmation
    - Signal filtering
    """
    
    def __init__(self, params: TradingParameters):
        self.params = params
    
    def get_buy_signal(
        self,
        predicted_price: float,
        current_price: float,
        market_data: Dict,
        daily_volatility: float = 0.01
    ) -> Tuple[bool, float, str]:
        """
        Generate BUY signal
        
        Returns:
            (signal_generated, confidence, reason)
        """
        
        # Calculate price appreciation
        price_appreciation = (predicted_price - current_price) / current_price
        
        # Adjust thresholds based on volatility
        buy_threshold = self.params.buy_threshold
        if daily_volatility > self.params.volatility_threshold:
            buy_threshold *= self.params.volatility_threshold_multiplier
        
        # Check primary condition
        if price_appreciation <= buy_threshold:
            return False, 0.0, f"Price appreciation {price_appreciation:.2%} < threshold {buy_threshold:.2%}"
        
        # SMA-50 trend filter: only buy when price is above SMA-50 (uptrend)
        sma50 = market_data.get('SMA_50')
        close = market_data.get('Close', current_price)
        if sma50 is not None and close < sma50:
            return False, 0.0, f"Price ${close:.2f} below SMA-50 ${sma50:.2f} — downtrend filter"
        
        # Check uptrend confirmation
        confirmations, confirmation_details = self._check_uptrend_confirmation(market_data)
        
        if confirmations < 2:
            return False, 0.5, f"Insufficient uptrend confirmations: {confirmations}"
        
        # Calculate confidence score (0-1)
        confidence = min(1.0, (price_appreciation / buy_threshold) * 0.7 + (confirmations / 3) * 0.3)
        
        # Check confidence threshold
        if confidence < self.params.confidence_threshold:
            return False, confidence, f"Confidence {confidence:.2%} < threshold"
        
        reason = f"Price: {price_appreciation:.2%} up, Confirmations: {confirmations}/3"
        return True, confidence, reason
    
    def get_sell_signal(
        self,
        predicted_price: float,
        current_price: float,
        market_data: Dict,
        daily_volatility: float = 0.01
    ) -> Tuple[bool, float, str]:
        """
        Generate SELL signal (reversal signal)
        
        Returns:
            (signal_generated, confidence, reason)
        """
        
        # Calculate price depreciation
        price_depreciation = (current_price - predicted_price) / current_price
        
        # Adjust thresholds based on volatility
        sell_threshold = self.params.sell_threshold
        if daily_volatility > self.params.volatility_threshold:
            sell_threshold *= self.params.volatility_threshold_multiplier
        
        # Check primary condition
        if price_depreciation <= sell_threshold:
            return False, 0.0, f"Price depreciation {price_depreciation:.2%} < threshold {sell_threshold:.2%}"
        
        # Check downtrend confirmation
        confirmations, confirmation_details = self._check_downtrend_confirmation(market_data)
        
        if confirmations < 2:
            return False, 0.5, f"Insufficient downtrend confirmations: {confirmations}"
        
        # Calculate confidence score (0-1)
        confidence = min(1.0, (price_depreciation / sell_threshold) * 0.7 + (confirmations / 3) * 0.3)
        
        # Check confidence threshold
        if confidence < self.params.confidence_threshold:
            return False, confidence, f"Confidence {confidence:.2%} < threshold"
        
        reason = f"Price: {price_depreciation:.2%} down, Confirmations: {confirmations}/3"
        return True, confidence, reason
    
    def _check_uptrend_confirmation(self, market_data: Dict) -> Tuple[int, str]:
        """
        Check multiple uptrend indicators
        Returns: (count of confirmations, details string)
        """
        confirmations = 0
        details = []
        
        # RSI Check: > 50
        if 'RSI_14' in market_data:
            if market_data['RSI_14'] > 50:
                confirmations += 1
                details.append(f"RSI={market_data['RSI_14']:.1f}>50")
            else:
                details.append(f"RSI={market_data['RSI_14']:.1f}≤50")
        
        # Price vs SMA(20)
        if 'Close' in market_data and 'SMA_20' in market_data:
            if market_data['Close'] > market_data['SMA_20']:
                confirmations += 1
                details.append(f"Price>SMA20")
            else:
                details.append(f"Price≤SMA20")
        
        # EMA(10) vs EMA(20)
        if 'EMA_10' in market_data and 'EMA_20' in market_data:
            if market_data['EMA_10'] > market_data['EMA_20']:
                confirmations += 1
                details.append(f"EMA10>EMA20")
            else:
                details.append(f"EMA10≤EMA20")
        
        return confirmations, ", ".join(details)
    
    def _check_downtrend_confirmation(self, market_data: Dict) -> Tuple[int, str]:
        """
        Check multiple downtrend indicators
        Returns: (count of confirmations, details string)
        """
        confirmations = 0
        details = []
        
        # RSI Check: < 50
        if 'RSI_14' in market_data:
            if market_data['RSI_14'] < 50:
                confirmations += 1
                details.append(f"RSI={market_data['RSI_14']:.1f}<50")
            else:
                details.append(f"RSI={market_data['RSI_14']:.1f}≥50")
        
        # Price vs SMA(20)
        if 'Close' in market_data and 'SMA_20' in market_data:
            if market_data['Close'] < market_data['SMA_20']:
                confirmations += 1
                details.append(f"Price<SMA20")
            else:
                details.append(f"Price≥SMA20")
        
        # EMA(10) vs EMA(20)
        if 'EMA_10' in market_data and 'EMA_20' in market_data:
            if market_data['EMA_10'] < market_data['EMA_20']:
                confirmations += 1
                details.append(f"EMA10<EMA20")
            else:
                details.append(f"EMA10≤EMA20")
        
        return confirmations, ", ".join(details)


class PositionSizingCalculator:
    """
    Risk-based position sizing
    
    Calculates:
    - Position size in shares based on portfolio risk
    - Validates position constraints
    - Applies position limits
    """
    
    def __init__(self, params: TradingParameters):
        self.params = params
    
    def calculate_position_size(
        self,
        entry_price: float,
        portfolio_value: float,
        available_cash: float
    ) -> int:
        """
        Calculate position size in shares using risk-based approach
        
        Formula:
            max_risk = portfolio_value × RISK_PERCENTAGE
            position_value = max_risk / STOP_LOSS_PERCENT
            shares = position_value / entry_price
        
        Args:
            entry_price: Entry price per share
            portfolio_value: Total portfolio value
            available_cash: Available cash to allocate
        
        Returns:
            Number of shares to buy
        """
        
        # Calculate based on risk percentage
        max_risk = portfolio_value * self.params.risk_percentage
        
        # Maximum position value based on risk tolerance
        position_value = max_risk / self.params.stop_loss_percent
        
        # Apply position size constraints
        max_position_value = portfolio_value * self.params.max_position_value_percent
        position_value = min(position_value, max_position_value)
        
        # Apply cash constraint
        position_value = min(position_value, available_cash)
        
        # Calculate shares
        shares = int(position_value / entry_price)
        
        # Apply minimum shares requirement
        shares = max(shares, self.params.min_position_size)
        
        return shares
    
    def calculate_position_limits(
        self,
        portfolio_value: float,
        available_cash: float,
        num_active_positions: int
    ) -> Dict[str, float]:
        """
        Calculate current portfolio position limits
        
        Returns:
            {
                'max_position_value': max single position value,
                'current_total_risk': total portfolio risk,
                'remaining_risk_budget': remaining risk capacity,
                'can_open_position': bool,
                'reason': explanation
            }
        """
        
        # Position count check
        if num_active_positions >= self.params.max_concurrent_positions:
            return {
                'max_position_value': 0,
                'current_total_risk': 0,
                'remaining_risk_budget': 0,
                'can_open_position': False,
                'reason': f'Max positions ({self.params.max_concurrent_positions}) reached'
            }
        
        # Cash check
        min_cash_required = portfolio_value * (1 - self.params.max_cash_allocation_percent)
        if available_cash < min_cash_required:
            return {
                'max_position_value': 0,
                'current_total_risk': 0,
                'remaining_risk_budget': 0,
                'can_open_position': False,
                'reason': f'Insufficient cash (min: ${min_cash_required:.2f})'
            }
        
        max_position_value = portfolio_value * self.params.max_position_value_percent
        max_risk = portfolio_value * self.params.risk_percentage
        
        return {
            'max_position_value': max_position_value,
            'max_risk_per_trade': max_risk,
            'remaining_cash': available_cash,
            'can_open_position': True,
            'reason': 'OK'
        }


class RiskManager:
    """
    Portfolio-level risk tracking and management
    
    Tracks:
    - Overall portfolio risk
    - Drawdown from peak
    - Circuit breaker conditions
    - Trade statistics
    """
    
    def __init__(self, params: TradingParameters):
        self.params = params
        self.peak_portfolio_value = 0.0
        self.completed_trades: List[Trade] = []
    
    def update_peak_value(self, current_value: float):
        """Update peak portfolio value for drawdown calculation"""
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
    
    def calculate_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_portfolio_value == 0:
            return 0.0
        return (current_value - self.peak_portfolio_value) / self.peak_portfolio_value
    
    def check_circuit_breaker(self, current_value: float, initial_value: float) -> Tuple[bool, str]:
        """
        Check if circuit breaker (max portfolio loss) is triggered
        
        Returns:
            (breaker_triggered, reason)
        """
        portfolio_loss = (current_value - initial_value) / initial_value
        
        if portfolio_loss < self.params.portfolio_max_loss_percent:
            return True, f"Portfolio loss {portfolio_loss:.2%} < limit {self.params.portfolio_max_loss_percent:.2%}"
        
        return False, "OK"
    
    def add_completed_trade(self, trade: Trade):
        """Record completed trade for statistics"""
        self.completed_trades.append(trade)
    
    def get_trade_statistics(self) -> Dict:
        """Calculate statistics from all completed trades"""
        
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration_days': 0
            }
        
        winning_trades = [t for t in self.completed_trades if t.is_winning_trade]
        losing_trades = [t for t in self.completed_trades if not t.is_winning_trade]
        
        total_wins = sum(t.pnl_amount for t in winning_trades)
        total_losses = sum(abs(t.pnl_amount) for t in losing_trades)
        
        win_rates = [t.pnl_percent for t in winning_trades] if winning_trades else [0]
        loss_rates = [t.pnl_percent for t in losing_trades] if losing_trades else [0]
        
        durations = [t.duration_days for t in self.completed_trades]
        
        return {
            'total_trades': len(self.completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.completed_trades) if self.completed_trades else 0,
            'avg_profit': np.mean(win_rates) if winning_trades else 0.0,
            'avg_loss': np.mean(loss_rates) if losing_trades else 0.0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0.0,
            'largest_win': max(win_rates) if winning_trades else 0.0,
            'largest_loss': min(loss_rates) if losing_trades else 0.0,
            'avg_duration_days': np.mean(durations) if durations else 0
        }
    
    def print_trade_statistics(self):
        """Pretty print trade statistics"""
        stats = self.get_trade_statistics()
        
        print("\n" + "="*60)
        print("TRADE STATISTICS")
        print("="*60)
        print(f"Total Trades:        {stats['total_trades']}")
        print(f"Winning Trades:      {stats['winning_trades']}")
        print(f"Losing Trades:       {stats['losing_trades']}")
        print(f"Win Rate:            {stats['win_rate']:>6.1%}")
        print(f"Profit Factor:       {stats['profit_factor']:>6.2f}")
        print(f"Avg Winning Trade:   {stats['avg_profit']:>6.2%}")
        print(f"Avg Losing Trade:    {stats['avg_loss']:>6.2%}")
        print(f"Largest Win:         {stats['largest_win']:>6.2%}")
        print(f"Largest Loss:        {stats['largest_loss']:>6.2%}")
        print(f"Avg Trade Duration:  {stats['avg_duration_days']:>6.1f} days")
        print("="*60 + "\n")


class TradeExecutor:
    """
    Execute trading signals and manage positions
    
    Handles:
    - Buy signal execution
    - Exit signal evaluation (SL, TP, Reversal)
    - Trade state management
    """
    
    def __init__(self, params: TradingParameters):
        self.params = params
        self.active_positions: List[Position] = []
        self.risk_manager = RiskManager(params)
        self.position_sizer = PositionSizingCalculator(params)
        self.trading_rules = TradingRules(params)
    
    def check_exit_conditions(
        self,
        position: Position,
        current_price: float,
        current_date: datetime,
        predicted_price: float,
        market_data: Dict
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Check if position should be exited
        
        Priority Order:
        1. Stop Loss (highest priority)
        2. Portfolio Circuit Breaker (handled outside)
        3. Take Profit
        4. Reversal Signal
        
        Returns:
            (exit_reason, exit_price) or (None, None)
        """
        
        # Calculate position metrics
        days_held = (current_date - position.entry_date).days
        unrealized_gain_pct = (current_price - position.entry_price) / position.entry_price
        
        # Exit Priority 1: Stop Loss
        if current_price <= position.stop_loss_price:
            return "STOP_LOSS", position.stop_loss_price
        
        # Exit Priority 2: Take Profit
        if unrealized_gain_pct >= self.params.take_profit_target and days_held >= self.params.minimum_hold_days:
            return "TAKE_PROFIT", position.take_profit_price
        
        # Exit Priority 3: Reversal Signal
        if days_held >= self.params.minimum_hold_days:
            sell_signal, confidence, reason = self.trading_rules.get_sell_signal(
                predicted_price, current_price, market_data
            )
            if sell_signal:
                return "REVERSAL", current_price
        
        return None, None
    
    def execute_buy(
        self,
        symbol: str,
        current_price: float,
        current_date: datetime,
        portfolio_value: float,
        available_cash: float
    ) -> Optional[Position]:
        """
        Execute buy signal
        
        Returns:
            Position object if successful, None otherwise
        """
        
        # Calculate position size
        shares = self.position_sizer.calculate_position_size(
            current_price, portfolio_value, available_cash
        )
        
        # Check if we have enough cash
        if shares * current_price > available_cash:
            return None
        
        # Calculate stop-loss and take-profit
        stop_loss = current_price * (1 - self.params.stop_loss_percent)
        take_profit = current_price * (1 + self.params.take_profit_target)
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_date=current_date,
            entry_price=current_price,
            shares=shares,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit
        )
        
        self.active_positions.append(position)
        return position
    
    def execute_sell(self, position: Position) -> Trade:
        """
        Execute sell of position
        
        Returns:
            Completed Trade object
        """
        # Remove from active positions
        if position in self.active_positions:
            self.active_positions.remove(position)
        
        return position  # Will be converted to Trade elsewhere
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return {
            'num_open_positions': len(self.active_positions),
            'positions': self.active_positions.copy()
        }


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_trading_rules():
    """Demonstrate the trading rules with example scenarios"""
    
    print("\n" + "="*80)
    print("TRADING RULES DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize parameters
    params = TradingParameters()
    print(params)
    
    # Initialize rules
    rules = TradingRules(params)
    position_sizer = PositionSizingCalculator(params)
    risk_manager = RiskManager(params)
    
    # Example Scenario 1: Normal Market Buy Signal
    print("\n[SCENARIO 1] Normal Market - BUY Signal Test")
    print("-" * 60)
    
    current_price_1 = 200.00
    predicted_price_1 = 204.10  # 2.05% gain
    market_data_1 = {
        'Close': 200.00,
        'RSI_14': 55,
        'SMA_20': 198.50,
        'EMA_10': 200.50,
        'EMA_20': 199.00
    }
    
    buy_signal, confidence, reason = rules.get_buy_signal(
        predicted_price_1, current_price_1, market_data_1
    )
    
    print(f"Current Price:        ${current_price_1:.2f}")
    print(f"Predicted Price:      ${predicted_price_1:.2f}")
    print(f"Expected Gain:         {(predicted_price_1 - current_price_1)/current_price_1:>6.2%}")
    print(f"BUY Signal:            {buy_signal}")
    print(f"Confidence:            {confidence:>6.1%}")
    print(f"Reason:                {reason}")
    
    if buy_signal:
        # Calculate position size
        portfolio_value = 100000
        available_cash = 50000
        shares = position_sizer.calculate_position_size(
            current_price_1, portfolio_value, available_cash
        )
        print(f"\nPosition Sizing:")
        print(f"  Portfolio Value:     ${portfolio_value:,.2f}")
        print(f"  Available Cash:      ${available_cash:,.2f}")
        print(f"  Shares to Buy:       {shares}")
        print(f"  Position Value:      ${shares * current_price_1:,.2f}")
        print(f"  Stop Loss Price:     ${current_price_1 * (1 - params.stop_loss_percent):.2f}")
        print(f"  Take Profit Price:   ${current_price_1 * (1 + params.take_profit_target):.2f}")
    
    # Example Scenario 2: Sell Signal During Downtrend
    print("\n\n[SCENARIO 2] Downtrend - SELL Signal Test")
    print("-" * 60)
    
    current_price_2 = 200.00
    predicted_price_2 = 195.90  # 2.05% loss
    market_data_2 = {
        'Close': 200.00,
        'RSI_14': 35,
        'SMA_20': 201.50,
        'EMA_10': 199.50,
        'EMA_20': 201.00
    }
    
    sell_signal, confidence, reason = rules.get_sell_signal(
        predicted_price_2, current_price_2, market_data_2
    )
    
    print(f"Current Price:        ${current_price_2:.2f}")
    print(f"Predicted Price:      ${predicted_price_2:.2f}")
    print(f"Expected Loss:         {(predicted_price_2 - current_price_2)/current_price_2:>6.2%}")
    print(f"SELL Signal:           {sell_signal}")
    print(f"Confidence:            {confidence:>6.1%}")
    print(f"Reason:                {reason}")
    
    # Example Scenario 3: Risk Management
    print("\n\n[SCENARIO 3] Risk Management - Drawdown & Circuit Breaker")
    print("-" * 60)
    
    initial_value = 100000
    peak_value = 105000
    current_value = 99800
    
    risk_manager.update_peak_value(peak_value)
    drawdown = risk_manager.calculate_drawdown(current_value)
    breaker_triggered, breaker_reason = risk_manager.check_circuit_breaker(current_value, initial_value)
    
    print(f"Initial Portfolio:    ${initial_value:,.2f}")
    print(f"Peak Portfolio:        ${peak_value:,.2f}")
    print(f"Current Portfolio:     ${current_value:,.2f}")
    print(f"Drawdown from Peak:    {drawdown:>6.2%}")
    print(f"Loss from Start:       {(current_value - initial_value)/initial_value:>6.2%}")
    print(f"Circuit Breaker:       {breaker_triggered}")
    print(f"Breaker Reason:        {breaker_reason}")
    
    # Example Scenario 4: Position Exit Evaluation
    print("\n\n[SCENARIO 4] Position Management - Exit Conditions")
    print("-" * 60)
    
    from datetime import datetime, timedelta
    
    entry_date = datetime.now() - timedelta(days=2)
    position = Position(
        symbol="AAPL",
        entry_date=entry_date,
        entry_price=200.00,
        shares=100,
        stop_loss_price=197.00,
        take_profit_price=205.00
    )
    
    print(f"Position:              {position}")
    print(f"Entry Date:            {position.entry_date.strftime('%Y-%m-%d')}")
    print(f"Days Held:             {(datetime.now() - position.entry_date).days}")
    
    # Test exit conditions
    print("\nExit Scenarios:")
    
    # Scenario 4a: Take Profit
    current_price_4a = 205.50
    print(f"\n  4a. Current Price: ${current_price_4a:.2f} (TP should trigger)")
    unreal_pnl, unreal_pct = position.calculate_unrealized_pnl(current_price_4a)
    print(f"      Unrealized P/L:  ${unreal_pnl:,.2f} ({unreal_pct:+.2%})")
    
    # Scenario 4b: Stop Loss
    current_price_4b = 196.50
    print(f"\n  4b. Current Price: ${current_price_4b:.2f} (SL should trigger)")
    unreal_pnl, unreal_pct = position.calculate_unrealized_pnl(current_price_4b)
    print(f"      Unrealized P/L:  ${unreal_pnl:,.2f} ({unreal_pct:+.2%})")
    
    # Scenario 4c: No exit
    current_price_4c = 202.50
    print(f"\n  4c. Current Price: ${current_price_4c:.2f} (No exit)")
    unreal_pnl, unreal_pct = position.calculate_unrealized_pnl(current_price_4c)
    print(f"      Unrealized P/L:  ${unreal_pnl:,.2f} ({unreal_pct:+.2%})")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    demonstrate_trading_rules()
