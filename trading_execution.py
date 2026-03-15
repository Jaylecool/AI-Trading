"""
Task 4.2: Trading Execution & Order Management
Implements automated buy/sell execution with order management and logging

Core Components:
- Order classes (MarketOrder, LimitOrder)
- OrderManager for order lifecycle
- TradeLogger for comprehensive logging
- TradingEngine for automated execution
- Prediction integration for automatic triggering
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import json
import os
from pathlib import Path
import pandas as pd

# Import from Task 4.1
from trading_rules import (
    TradingParameters, TradingRules, PositionSizingCalculator,
    RiskManager, Position, Trade, TradeExecutor
)


# ============================================================================
# ENUMS FOR ORDER MANAGEMENT
# ============================================================================

class OrderType(Enum):
    """Types of orders"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderSide(Enum):
    """Buy or Sell"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeSignal(Enum):
    """Type of trade signal"""
    BUY_SIGNAL = "BUY_SIGNAL"
    SELL_SIGNAL = "SELL_SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    REVERSAL = "REVERSAL"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


# ============================================================================
# ORDER CLASSES
# ============================================================================

@dataclass
class Order:
    """Base order class"""
    
    order_id: int
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    created_at: datetime
    
    # Order details
    entry_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution details
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_filled_price: float = 0.0
    filled_at: Optional[datetime] = None
    
    # Tracking
    position_id: Optional[int] = None
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.filled_quantity == self.quantity
    
    @property
    def fill_percentage(self) -> float:
        """Percentage of order filled"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0
    
    def fill_order(self, fill_quantity: int, fill_price: float):
        """Fill part or all of order"""
        self.filled_quantity += fill_quantity
        
        # Update average filled price
        if self.fill_percentage > 0:
            total_filled_value = (
                (self.filled_quantity - fill_quantity) * self.average_filled_price +
                fill_quantity * fill_price
            )
            self.average_filled_price = total_filled_value / self.filled_quantity
        else:
            self.average_filled_price = fill_price
        
        # Update status
        if self.is_filled:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
        
        return self.is_filled
    
    def cancel_order(self):
        """Cancel order"""
        self.status = OrderStatus.CANCELLED
    
    def __str__(self) -> str:
        return (f"Order({self.order_id}, {self.symbol} {self.side.value} "
                f"{self.quantity} @ {self.order_type.value}, "
                f"Status: {self.status.value})")


@dataclass
class MarketOrder(Order):
    """Market order - execute at current market price"""
    
    order_type: OrderType = field(default=OrderType.MARKET, init=False)
    
    def execute(self, current_price: float) -> bool:
        """Execute market order at current price"""
        self.entry_price = current_price
        self.fill_order(self.quantity, current_price)
        return self.is_filled


@dataclass
class LimitOrder(Order):
    """Limit order - execute at specified price or better"""
    
    order_type: OrderType = field(default=OrderType.LIMIT, init=False)
    
    def execute(self, current_price: float) -> bool:
        """Attempt to execute limit order if price conditions met"""
        if self.limit_price is None:
            raise ValueError("Limit price must be set for limit orders")
        
        # For BUY: only fill if current < limit
        # For SELL: only fill if current > limit
        can_fill = (
            (self.side == OrderSide.BUY and current_price <= self.limit_price) or
            (self.side == OrderSide.SELL and current_price >= self.limit_price)
        )
        
        if can_fill:
            self.entry_price = current_price
            self.fill_order(self.quantity, current_price)
            return self.is_filled
        
        return False


@dataclass
class StopLossOrder(Order):
    """Stop-loss order - triggers when price falls below stop price"""
    
    order_type: OrderType = field(default=OrderType.STOP_LOSS, init=False)
    side: OrderSide = field(default=OrderSide.SELL, init=False)
    triggered: bool = False
    
    def check_trigger(self, current_price: float) -> bool:
        """Check if stop price is breached"""
        if self.stop_price is None:
            return False
        
        if not self.triggered and current_price <= self.stop_price:
            self.triggered = True
            return True
        
        return self.triggered
    
    def execute(self, current_price: float) -> bool:
        """Execute stop-loss at market price"""
        if self.check_trigger(current_price):
            self.entry_price = current_price
            self.fill_order(self.quantity, current_price)
            return self.is_filled
        
        return False


@dataclass
class TakeProfitOrder(Order):
    """Take-profit order - triggers when price rises above target price"""
    
    order_type: OrderType = field(default=OrderType.TAKE_PROFIT, init=False)
    side: OrderSide = field(default=OrderSide.SELL, init=False)
    triggered: bool = False
    
    def check_trigger(self, current_price: float) -> bool:
        """Check if profit target is reached"""
        if self.stop_price is None:
            return False
        
        if not self.triggered and current_price >= self.stop_price:
            self.triggered = True
            return True
        
        return self.triggered
    
    def execute(self, current_price: float) -> bool:
        """Execute take-profit at market price"""
        if self.check_trigger(current_price):
            self.entry_price = current_price
            self.fill_order(self.quantity, current_price)
            return self.is_filled
        
        return False


# ============================================================================
# ORDER MANAGER
# ============================================================================

class OrderManager:
    """
    Manages all orders for the trading system
    
    Responsibilities:
    - Create orders
    - Track order lifecycle
    - Execute orders when conditions met
    - Maintain order history
    """
    
    def __init__(self):
        self.orders: List[Order] = []
        self.order_counter = 1000
        self.order_history: List[Order] = []
    
    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        entry_price: float,
        position_id: Optional[int] = None
    ) -> Order:
        """Create and register a market order"""
        
        order = MarketOrder(
            order_id=self.order_counter,
            symbol=symbol,
            side=side,
            quantity=quantity,
            created_at=datetime.now(),
            entry_price=entry_price,
            position_id=position_id
        )
        
        self.order_counter += 1
        self.orders.append(order)
        
        return order
    
    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        position_id: Optional[int] = None
    ) -> Order:
        """Create and register a limit order"""
        
        order = LimitOrder(
            order_id=self.order_counter,
            symbol=symbol,
            side=side,
            quantity=quantity,
            created_at=datetime.now(),
            limit_price=limit_price,
            position_id=position_id
        )
        
        self.order_counter += 1
        self.orders.append(order)
        
        return order
    
    def create_stop_loss_order(
        self,
        symbol: str,
        quantity: int,
        stop_price: float,
        position_id: Optional[int] = None
    ) -> Order:
        """Create stop-loss order"""
        
        order = StopLossOrder(
            order_id=self.order_counter,
            symbol=symbol,
            quantity=quantity,
            created_at=datetime.now(),
            stop_price=stop_price,
            position_id=position_id
        )
        
        self.order_counter += 1
        self.orders.append(order)
        
        return order
    
    def create_take_profit_order(
        self,
        symbol: str,
        quantity: int,
        target_price: float,
        position_id: Optional[int] = None
    ) -> Order:
        """Create take-profit order"""
        
        order = TakeProfitOrder(
            order_id=self.order_counter,
            symbol=symbol,
            quantity=quantity,
            created_at=datetime.now(),
            stop_price=target_price,
            position_id=position_id
        )
        
        self.order_counter += 1
        self.orders.append(order)
        
        return order
    
    def process_orders(self, current_price: float, symbol: str = "AAPL") -> List[Order]:
        """Process all active orders and return filled orders"""
        
        filled_orders = []
        
        for order in self.orders:
            if order.symbol != symbol or order.status == OrderStatus.FILLED:
                continue
            
            # Execute order based on type
            if isinstance(order, MarketOrder):
                order.execute(current_price)
            elif isinstance(order, LimitOrder):
                order.execute(current_price)
            elif isinstance(order, StopLossOrder):
                order.execute(current_price)
            elif isinstance(order, TakeProfitOrder):
                order.execute(current_price)
            
            if order.is_filled:
                filled_orders.append(order)
                self.order_history.append(order)
        
        return filled_orders
    
    def get_pending_orders(self, symbol: str = "AAPL") -> List[Order]:
        """Get all pending orders for symbol"""
        return [o for o in self.orders 
                if o.symbol == symbol and o.status == OrderStatus.PENDING]
    
    def get_active_orders(self, symbol: str = "AAPL") -> List[Order]:
        """Get all active orders for symbol"""
        return [o for o in self.orders 
                if o.symbol == symbol and o.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]]
    
    def cancel_order(self, order_id: int):
        """Cancel an order"""
        for order in self.orders:
            if order.order_id == order_id:
                order.cancel_order()
                return True
        return False


# ============================================================================
# TRADE LOGGER
# ============================================================================

class TradeLogger:
    """
    Comprehensive logging system for all trading activity
    
    Logs:
    - Buy/sell orders
    - Order fills
    - Trade entries/exits
    - P&L results
    - Signals generated
    """
    
    def __init__(self, log_dir: str = "."):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.trades_log = []
        self.signals_log = []
        self.orders_log = []
        
        # Setup log files
        self.trades_file = self.log_dir / "trading_activity.csv"
        self.signals_file = self.log_dir / "trade_signals.csv"
        self.orders_file = self.log_dir / "orders.csv"
        self.summary_file = self.log_dir / "trading_summary.json"
    
    def log_signal(
        self,
        timestamp: datetime,
        signal_type: TradeSignal,
        symbol: str,
        current_price: float,
        predicted_price: float,
        confidence: float,
        market_data: Dict
    ):
        """Log a trade signal"""
        
        signal_record = {
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Signal': signal_type.value,
            'Symbol': symbol,
            'Current_Price': current_price,
            'Predicted_Price': predicted_price,
            'Expected_Move': (predicted_price - current_price) / current_price,
            'Confidence': confidence,
            'RSI_14': market_data.get('RSI_14', None),
            'SMA_20': market_data.get('SMA_20', None),
            'EMA_10': market_data.get('EMA_10', None),
            'EMA_20': market_data.get('EMA_20', None)
        }
        
        self.signals_log.append(signal_record)
        return signal_record
    
    def log_order(
        self,
        timestamp: datetime,
        order: Order,
        reason: str = ""
    ):
        """Log an order creation or execution"""
        
        order_record = {
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Order_ID': order.order_id,
            'Symbol': order.symbol,
            'Type': order.order_type.value,
            'Side': order.side.value,
            'Quantity': order.quantity,
            'Entry_Price': order.entry_price,
            'Limit_Price': order.limit_price,
            'Stop_Price': order.stop_price,
            'Status': order.status.value,
            'Filled': order.filled_quantity,
            'Avg_Fill_Price': order.average_filled_price,
            'Position_ID': order.position_id,
            'Reason': reason
        }
        
        self.orders_log.append(order_record)
        return order_record
    
    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,  # BUY or SELL
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_percent: Optional[float] = None,
        position_id: Optional[int] = None,
        reason: str = ""
    ):
        """Log a trade execution"""
        
        trade_record = {
            'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Symbol': symbol,
            'Action': action,
            'Quantity': quantity,
            'Price': price,
            'Stop_Loss': stop_loss,
            'Take_Profit': take_profit,
            'P&L': pnl,
            'P&L_Percent': pnl_percent,
            'Position_ID': position_id,
            'Reason': reason
        }
        
        self.trades_log.append(trade_record)
        return trade_record
    
    def save_logs(self):
        """Save all logs to files"""
        
        # Save trades log
        if self.trades_log:
            df_trades = pd.DataFrame(self.trades_log)
            df_trades.to_csv(self.trades_file, index=False)
        
        # Save signals log
        if self.signals_log:
            df_signals = pd.DataFrame(self.signals_log)
            df_signals.to_csv(self.signals_file, index=False)
        
        # Save orders log
        if self.orders_log:
            df_orders = pd.DataFrame(self.orders_log)
            df_orders.to_csv(self.orders_file, index=False)
    
    def get_summary(self) -> Dict:
        """Generate summary statistics from logs"""
        
        if not self.trades_log:
            return {
                'total_trades': 0,
                'buy_count': 0,
                'sell_count': 0,
                'total_pnl': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0
            }
        
        df = pd.DataFrame(self.trades_log)
        
        buy_trades = df[df['Action'] == 'BUY']
        sell_trades = df[df['Action'] == 'SELL']
        
        winning = df[df['P&L'] > 0] if 'P&L' in df.columns else pd.DataFrame()
        losing = df[df['P&L'] < 0] if 'P&L' in df.columns else pd.DataFrame()
        
        summary = {
            'total_trades': len(df),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'total_pnl': df['P&L'].sum() if 'P&L' in df.columns else 0,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(df) if len(df) > 0 else 0.0,
            'avg_pnl': df['P&L'].mean() if 'P&L' in df.columns else 0,
            'total_signals': len(self.signals_log),
            'total_orders': len(self.orders_log)
        }
        
        return summary
    
    def print_summary(self):
        """Pretty print summary to console"""
        
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("TRADING ACTIVITY SUMMARY")
        print("="*70)
        print(f"Total Trades:          {summary['total_trades']}")
        print(f"Buys:                  {summary['buy_count']}")
        print(f"Sells:                 {summary['sell_count']}")
        print(f"Winning Trades:        {summary['winning_trades']}")
        print(f"Losing Trades:         {summary['losing_trades']}")
        print(f"Win Rate:              {summary['win_rate']:.1%}")
        print(f"Total P&L:             ${summary['total_pnl']:>10,.2f}")
        print(f"Avg P&L per Trade:     ${summary['avg_pnl']:>10,.2f}")
        print(f"Total Signals:         {summary['total_signals']}")
        print(f"Total Orders:          {summary['total_orders']}")
        print("="*70 + "\n")


# ============================================================================
# TRADING ENGINE - MAIN EXECUTION
# ============================================================================

class TradingEngine:
    """
    Main trading engine that integrates:
    - Trading rules from Task 4.1
    - Order management
    - Trade logging
    - Automated execution on prediction signals
    """
    
    def __init__(
        self,
        params: TradingParameters,
        initial_capital: float = 100000,
        log_dir: str = ".",
        symbol: str = "AAPL"
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.symbol = symbol
        
        # Components
        self.trading_rules = TradingRules(params)
        self.position_sizer = PositionSizingCalculator(params)
        self.risk_manager = RiskManager(params)
        self.trade_executor = TradeExecutor(params)
        self.order_manager = OrderManager()
        self.trade_logger = TradeLogger(log_dir)
        
        # State tracking
        self.portfolio_value = initial_capital
        self.peak_portfolio_value = initial_capital
        self.current_price = 0.0
        self.daily_pnl = 0.0
        self.session_start_time = datetime.now()
        
        # Statistics
        self.buy_signals_received = 0
        self.sell_signals_received = 0
        self.buy_orders_executed = 0
        self.sell_orders_executed = 0
    
    def process_prediction(
        self,
        current_price: float,
        predicted_price: float,
        market_data: Dict,
        current_date: datetime = None
    ) -> Tuple[Optional[str], bool]:
        """
        Process model prediction and generate trading signals
        
        Args:
            current_price: Current market price
            predicted_price: Model's price prediction
            market_data: Market indicators (RSI, SMA, EMA, etc.)
            current_date: Current date/time
        
        Returns:
            (signal_type: str or None, order_executed: bool)
        """
        
        if current_date is None:
            current_date = datetime.now()
        
        self.current_price = current_price
        
        # Calculate portfolio value
        self.portfolio_value = self.current_capital
        for pos in self.trade_executor.active_positions:
            pos_value = pos.shares * current_price
            self.portfolio_value += pos_value
        
        # Update peak for drawdown
        self.risk_manager.update_peak_value(self.portfolio_value)
        
        # Check circuit breaker
        breaker_triggered, breaker_reason = self.risk_manager.check_circuit_breaker(
            self.portfolio_value, self.initial_capital
        )
        
        if breaker_triggered:
            self._close_all_positions(current_date, current_price, "CIRCUIT_BREAKER")
            self.trade_logger.log_signal(
                current_date,
                TradeSignal.CIRCUIT_BREAKER,
                self.symbol,
                current_price,
                predicted_price,
                1.0,
                market_data
            )
            return "CIRCUIT_BREAKER", False
        
        # Check open positions for exits first
        self._check_position_exits(current_date, current_price, predicted_price, market_data)
        
        # Calculate volatility
        daily_volatility = market_data.get('Volatility_20', 0.01) / 100
        
        # Check for BUY signal
        buy_signal, confidence, reason = self.trading_rules.get_buy_signal(
            predicted_price, current_price, market_data, daily_volatility
        )
        
        if buy_signal:
            self.buy_signals_received += 1
            executed = self._execute_buy_signal(
                current_date, current_price, confidence, reason
            )
            self.trade_logger.log_signal(
                current_date,
                TradeSignal.BUY_SIGNAL,
                self.symbol,
                current_price,
                predicted_price,
                confidence,
                market_data
            )
            return "BUY", executed
        
        # Check for SELL signal
        sell_signal, confidence, reason = self.trading_rules.get_sell_signal(
            predicted_price, current_price, market_data, daily_volatility
        )
        
        if sell_signal:
            self.sell_signals_received += 1
            self.trade_logger.log_signal(
                current_date,
                TradeSignal.SELL_SIGNAL,
                self.symbol,
                current_price,
                predicted_price,
                confidence,
                market_data
            )
            return "SELL", False  # Don't auto-close on sell signal alone
        
        return None, False
    
    def _execute_buy_signal(
        self,
        current_date: datetime,
        current_price: float,
        confidence: float,
        reason: str
    ) -> bool:
        """Execute BUY signal"""
        
        # Check position limits
        if len(self.trade_executor.active_positions) >= self.params.max_concurrent_positions:
            self.trade_logger.log_signal(
                current_date,
                TradeSignal.BUY_SIGNAL,
                self.symbol,
                current_price,
                current_price,
                confidence,
                {}
            )
            return False
        
        if self.current_capital <= 0:
            return False
        
        # Calculate position size
        shares = self.position_sizer.calculate_position_size(
            current_price, self.portfolio_value, self.current_capital
        )
        
        if shares < self.params.min_position_size:
            return False
        
        # Create buy order
        buy_order = self.order_manager.create_market_order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            quantity=shares,
            entry_price=current_price
        )
        
        self.trade_logger.log_order(
            current_date, buy_order,
            reason=f"BUY signal (confidence: {confidence:.0%})"
        )
        
        # Execute order immediately (market order)
        buy_order.execute(current_price)
        
        # Deduct cost from capital
        position_cost = shares * current_price
        self.current_capital -= position_cost
        
        self.trade_logger.log_trade(
            current_date,
            symbol=self.symbol,
            action="BUY",
            quantity=shares,
            price=current_price,
            position_id=buy_order.order_id,
            reason=reason
        )
        
        # Create position
        position = Position(
            symbol=self.symbol,
            entry_date=current_date,
            entry_price=current_price,
            shares=shares,
            stop_loss_price=current_price * (1 - self.params.stop_loss_percent),
            take_profit_price=current_price * (1 + self.params.take_profit_target)
        )
        position.position_id = buy_order.order_id
        
        self.trade_executor.active_positions.append(position)
        
        # Create stop-loss and take-profit orders
        sl_order = self.order_manager.create_stop_loss_order(
            symbol=self.symbol,
            quantity=shares,
            stop_price=position.stop_loss_price,
            position_id=buy_order.order_id
        )
        
        tp_order = self.order_manager.create_take_profit_order(
            symbol=self.symbol,
            quantity=shares,
            target_price=position.take_profit_price,
            position_id=buy_order.order_id
        )
        
        self.trade_logger.log_order(
            current_date, sl_order,
            reason="Stop-loss order"
        )
        
        self.trade_logger.log_order(
            current_date, tp_order,
            reason="Take-profit order"
        )
        
        self.buy_orders_executed += 1
        
        return True
    
    def _check_position_exits(
        self,
        current_date: datetime,
        current_price: float,
        predicted_price: float,
        market_data: Dict
    ):
        """Check and execute exits for all open positions"""
        
        positions_to_close = []
        
        for position in self.trade_executor.active_positions:
            
            # Check exit conditions
            exit_reason, exit_price = self.trade_executor.check_exit_conditions(
                position,
                current_price,
                current_date,
                predicted_price,
                market_data
            )
            
            if exit_reason:
                positions_to_close.append((position, exit_reason, exit_price))
        
        # Close positions
        for position, exit_reason, exit_price in positions_to_close:
            self._close_position(position, current_date, exit_price, exit_reason)
    
    def _close_position(
        self,
        position: Position,
        current_date: datetime,
        exit_price: float,
        exit_reason: str
    ):
        """Close a position"""
        
        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.shares
        pnl_percent = (exit_price - position.entry_price) / position.entry_price
        
        # Create sell order
        sell_order = self.order_manager.create_market_order(
            symbol=self.symbol,
            side=OrderSide.SELL,
            quantity=position.shares,
            entry_price=exit_price,
            position_id=position.position_id
        )
        
        sell_order.execute(exit_price)
        
        # Update capital
        self.current_capital += exit_price * position.shares
        self.daily_pnl += pnl
        
        # Log trade
        self.trade_logger.log_trade(
            current_date,
            symbol=self.symbol,
            action="SELL",
            quantity=position.shares,
            price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            position_id=position.position_id,
            reason=exit_reason
        )
        
        self.trade_logger.log_order(
            current_date, sell_order,
            reason=f"Close position: {exit_reason}"
        )
        
        # Remove position
        if position in self.trade_executor.active_positions:
            self.trade_executor.active_positions.remove(position)
        
        self.sell_orders_executed += 1
    
    def _close_all_positions(
        self,
        current_date: datetime,
        current_price: float,
        reason: str
    ):
        """Close all open positions"""
        
        positions = self.trade_executor.active_positions.copy()
        
        for position in positions:
            self._close_position(position, current_date, current_price, reason)
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        
        open_positions_value = sum(
            pos.shares * self.current_price 
            for pos in self.trade_executor.active_positions
        )
        
        total_value = self.current_capital + open_positions_value
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': total_value,
            'cash': self.current_capital,
            'open_positions_value': open_positions_value,
            'num_open_positions': len(self.trade_executor.active_positions),
            'daily_pnl': self.daily_pnl,
            'total_pnl': total_value - self.initial_capital,
            'return_percent': (total_value - self.initial_capital) / self.initial_capital,
            'buy_signals': self.buy_signals_received,
            'sell_signals': self.sell_signals_received,
            'buy_orders_executed': self.buy_orders_executed,
            'sell_orders_executed': self.sell_orders_executed
        }
    
    def print_status(self):
        """Print portfolio status"""
        
        status = self.get_portfolio_status()
        
        print("\n" + "="*70)
        print("TRADING ENGINE STATUS")
        print("="*70)
        print(f"Portfolio Value:       ${status['portfolio_value']:>12,.2f}")
        print(f"Cash:                  ${status['cash']:>12,.2f}")
        print(f"Open Positions Value:  ${status['open_positions_value']:>12,.2f}")
        print(f"Open Positions:        {status['num_open_positions']:>12}")
        print(f"Daily P&L:             ${status['daily_pnl']:>12,.2f}")
        print(f"Total P&L:             ${status['total_pnl']:>12,.2f}")
        print(f"Return:                {status['return_percent']:>12.2%}")
        print(f"\nBuy Signals:           {status['buy_signals']:>12}")
        print(f"Sell Signals:          {status['sell_signals']:>12}")
        print(f"Buy Orders Executed:   {status['buy_orders_executed']:>12}")
        print(f"Sell Orders Executed:  {status['sell_orders_executed']:>12}")
        print("="*70 + "\n")


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

def demonstrate_trading_execution():
    """Demonstrate trading execution with sample data"""
    
    print("\n" + "="*70)
    print("TASK 4.2 TRADING EXECUTION DEMONSTRATION")
    print("="*70 + "\n")
    
    # Setup
    params = TradingParameters()
    engine = TradingEngine(params, initial_capital=100000, log_dir=".")
    
    print("Trading Engine Initialized")
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Buy Threshold: {params.buy_threshold:.1%}")
    print(f"Stop Loss: {params.stop_loss_percent:.1%}")
    print(f"Take Profit: {params.take_profit_target:.1%}\n")
    
    # Simulate trading day
    print("SIMULATED TRADING DAY:")
    print("-" * 70)
    
    # Market scenario: Uptrend
    scenarios = [
        {
            'name': 'Morning (Uptrend)',
            'current_price': 200.00,
            'predicted_price': 204.10,  # 2.05% up
            'rsi': 55,
            'sma20': 198.50,
            'ema10': 200.50,
            'ema20': 199.00,
            'volatility': 1.0
        },
        {
            'name': 'Midday (Peak)',
            'current_price': 204.50,
            'predicted_price': 205.00,  # Slight gain
            'rsi': 65,
            'sma20': 200.00,
            'ema10': 202.00,
            'ema20': 201.50,
            'volatility': 1.2
        },
        {
            'name': 'Afternoon (Reversal)',
            'current_price': 202.00,
            'predicted_price': 198.50,  # -1.7% predicted
            'rsi': 45,
            'sma20': 201.50,
            'ema10': 200.00,
            'ema20': 201.00,
            'volatility': 1.5
        },
        {
            'name': 'Close (Stabilize)',
            'current_price': 201.50,
            'predicted_price': 201.00,  # Consolidation
            'rsi': 50,
            'sma20': 201.20,
            'ema10': 201.00,
            'ema20': 201.10,
            'volatility': 1.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Current Price: ${scenario['current_price']:.2f}")
        print(f"  Predicted Price: ${scenario['predicted_price']:.2f}")
        
        market_data = {
            'Close': scenario['current_price'],
            'RSI_14': scenario['rsi'],
            'SMA_20': scenario['sma20'],
            'EMA_10': scenario['ema10'],
            'EMA_20': scenario['ema20'],
            'Volatility_20': scenario['volatility']
        }
        
        # Process prediction
        signal, executed = engine.process_prediction(
            scenario['current_price'],
            scenario['predicted_price'],
            market_data,
            datetime.now()
        )
        
        if signal:
            print(f"  ✓ Signal: {signal}")
            if executed:
                print(f"  ✓ Order Executed")
        else:
            print(f"  - No signal")
        
        engine.print_status()
    
    # Print summary
    engine.trade_logger.print_summary()
    
    # Save logs
    engine.trade_logger.save_logs()
    print("Logs saved to trading_activity.csv, trade_signals.csv, orders.csv")


if __name__ == "__main__":
    demonstrate_trading_execution()
