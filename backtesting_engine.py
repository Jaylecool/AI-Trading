"""
Task 4.4: Backtesting Engine for Trading Strategies
Comprehensive backtesting system that simulates trading on historical data
and calculates performance metrics (ROI, Sharpe ratio, max drawdown, win rate, profit factor)

Key Features:
- Historical data loading and validation
- Trading simulation with realistic constraints
- Performance metrics calculation
- Trade-level tracking and analysis
- Portfolio value tracking over time
- Risk-adjusted return metrics
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import json
from collections import defaultdict

from trading_rules import TradingParameters, Position, RiskManager
from risk_management_enhanced import (
    EnhancedStopLoss, TrailingStopLoss, DynamicTakeProfitCalculator,
    PortfolioDiversificationManager, DynamicPositionSizer, EnhancedRiskMonitor
)


@dataclass
class TradingSignal:
    """Trading signal data"""
    buy: bool
    sell: bool
    confidence: float
    reason: str


# ============================================================================
# BACKTESTING DATA STRUCTURES
# ============================================================================

@dataclass
class BacktestTrade:
    """Record of a completed trade"""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: int
    entry_reason: str
    exit_reason: str
    pnl: float
    pnl_percent: float
    position_duration_days: int
    
    @property
    def roi_percent(self) -> float:
        """Return on investment percent"""
        return self.pnl_percent * 100
    
    @property
    def is_winning_trade(self) -> bool:
        """Whether trade was profitable"""
        return self.pnl > 0
    
    @property
    def is_losing_trade(self) -> bool:
        """Whether trade was losing"""
        return self.pnl < 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'entry_price': float(self.entry_price),
            'exit_date': self.exit_date.strftime('%Y-%m-%d'),
            'exit_price': float(self.exit_price),
            'shares': int(self.shares),
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
            'pnl': float(self.pnl),
            'pnl_percent': float(self.pnl_percent),
            'roi_percent': float(self.roi_percent),
            'position_duration_days': int(self.position_duration_days),
            'is_winning_trade': self.is_winning_trade,
            'type': 'LONG'
        }


@dataclass
class PortfolioSnapshot:
    """Daily portfolio state"""
    date: datetime
    total_value: float
    cash: float
    positions_value: float
    num_open_positions: int
    daily_return: float
    cumulative_return: float
    max_drawdown: float
    portfolio_heat_score: float


@dataclass
class BacktestResults:
    """Complete backtest results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Performance Metrics
    roi: float  # Total Return on Investment
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown: float  # Peak to trough
    win_rate: float  # Percentage of winning trades
    profit_factor: float  # Gross profit / Gross loss
    
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_hold_days: float
    
    # Risk Metrics
    annual_volatility: float
    calmar_ratio: float  # Return / Max Drawdown
    sortino_ratio: float  # Risk-adjusted (downside only)
    
    # Additional
    trades: List[BacktestTrade] = field(default_factory=list)
    portfolio_history: List[PortfolioSnapshot] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': float(self.initial_capital),
            'final_capital': float(self.final_capital),
            'roi': float(self.roi),
            'sharpe_ratio': float(self.sharpe_ratio),
            'max_drawdown': float(self.max_drawdown),
            'win_rate': float(self.win_rate),
            'profit_factor': float(self.profit_factor),
            'total_trades': int(self.total_trades),
            'winning_trades': int(self.winning_trades),
            'losing_trades': int(self.losing_trades),
            'avg_win': float(self.avg_win),
            'avg_loss': float(self.avg_loss),
            'avg_hold_days': float(self.avg_hold_days),
            'annual_volatility': float(self.annual_volatility),
            'calmar_ratio': float(self.calmar_ratio),
            'sortino_ratio': float(self.sortino_ratio),
            'trades': [t.to_dict() for t in self.trades],
            'final_portfolio_heat_score': float(self.portfolio_history[-1].portfolio_heat_score if self.portfolio_history else 0.0)
        }
    
    def __str__(self) -> str:
        """Pretty print results"""
        output = f"\n{'='*70}\n"
        output += f"BACKTEST RESULTS: {self.strategy_name}\n"
        output += f"{'='*70}\n"
        output += f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"
        output += f"\nPERFORMANCE METRICS:\n"
        output += f"{'-'*70}\n"
        output += f"  Initial Capital................ ${self.initial_capital:>15,.2f}\n"
        output += f"  Final Capital.................. ${self.final_capital:>15,.2f}\n"
        output += f"  Total Return (ROI)............ {self.roi:>15.2%}\n"
        output += f"  Sharpe Ratio................... {self.sharpe_ratio:>15.4f}\n"
        output += f"  Max Drawdown................... {self.max_drawdown:>15.2%}\n"
        output += f"  Calmar Ratio (Return/DD)....... {self.calmar_ratio:>15.4f}\n"
        output += f"  Sortino Ratio (Downside Risk).. {self.sortino_ratio:>15.4f}\n"
        output += f"  Annual Volatility.............. {self.annual_volatility:>15.2%}\n"
        
        output += f"\nTRADE STATISTICS:\n"
        output += f"{'-'*70}\n"
        output += f"  Total Trades................... {self.total_trades:>15,}\n"
        output += f"  Winning Trades................. {self.winning_trades:>15,}\n"
        output += f"  Losing Trades.................. {self.losing_trades:>15,}\n"
        output += f"  Win Rate....................... {self.win_rate:>15.2%}\n"
        output += f"  Average Winning Trade........... {self.avg_win:>15.2%}\n"
        output += f"  Average Losing Trade........... {self.avg_loss:>15.2%}\n"
        output += f"  Profit Factor (Gross P/L)...... {self.profit_factor:>15.4f}\n"
        output += f"  Average Hold Time.............. {self.avg_hold_days:>15.1f} days\n"
        
        output += f"\n{'='*70}\n"
        return output


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestingEngine:
    """Main backtesting engine for strategy evaluation"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        trading_params: Optional[TradingParameters] = None,
        use_risk_management: bool = True
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting portfolio value
            trading_params: Trading parameters (uses defaults if None)
            use_risk_management: Whether to apply risk management
        """
        self.initial_capital = initial_capital
        self.trading_params = trading_params or TradingParameters()
        self.use_risk_management = use_risk_management
        
        # Initialize components
        self.diversification_mgr = PortfolioDiversificationManager()
        self.position_sizer = DynamicPositionSizer(self.trading_params)
        self.risk_monitor = EnhancedRiskMonitor(self.trading_params)
        self.tp_calculator = DynamicTakeProfitCalculator(self.trading_params)
        
        # State
        self.current_capital = initial_capital
        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.daily_returns: List[float] = []
        
        # Stop-loss tracking
        self.stop_losses: Dict[str, EnhancedStopLoss] = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate historical data"""
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Validate required columns
        required_cols = ['Date', 'Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Volume_AAPL']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_name: str = "Test Strategy"
    ) -> BacktestResults:
        """
        Run complete backtest on historical data
        
        Args:
            data: Historical price data
            strategy_name: Name for this backtest
        
        Returns:
            BacktestResults with all metrics and trades
        """
        print(f"\n[BACKTEST] Starting backtest: {strategy_name}")
        print(f"[BACKTEST] Period: {data['Date'].min().date()} to {data['Date'].max().date()}")
        print(f"[BACKTEST] Initial Capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self.reset_backtest()
        
        # Process each day
        for idx, row in data.iterrows():
            current_date = row['Date']
            current_price = row['Close_AAPL']
            high_price = row['High_AAPL']
            low_price = row['Low_AAPL']
            volume = row['Volume_AAPL']
            
            # Extract indicators if available
            rsi = row.get('RSI_14', 50.0)
            volatility = row.get('Volatility_20', 0.02)
            
            # Update open positions (check for exits)
            self._process_exits(current_date, current_price, low_price, high_price)
            
            # Generate trading signals
            signals = self._generate_signals(
                current_price=current_price,
                rsi=rsi,
                volatility=volatility,
                volume=volume,
                prev_close=current_price
            )
            
            # Process entry signals
            if signals.buy and self._can_open_position():
                self._process_buy_signal(
                    current_date, current_price, signals, volatility, rsi
                )
            
            elif signals.sell and 'AAPL' in self.open_positions:
                self._process_sell_signal(
                    current_date, current_price, signals, 'SELL_SIGNAL'
                )
            
            # Record portfolio snapshot
            self._record_portfolio_snapshot(current_date, current_price)
        
        # Close any remaining positions at end of period
        if data.shape[0] > 0:
            final_date = data['Date'].iloc[-1]
            final_price = data['Close_AAPL'].iloc[-1]
            for symbol in list(self.open_positions.keys()):
                self._close_position(symbol, final_date, final_price, 'END_OF_PERIOD')
        
        # Calculate results
        results = self._calculate_results(
            strategy_name,
            data['Date'].min(),
            data['Date'].max()
        )
        
        print(f"[BACKTEST] Completed. Trades: {results.total_trades}, "
              f"ROI: {results.roi:.2%}, Sharpe: {results.sharpe_ratio:.4f}")
        
        return results
    
    def _generate_signals(
        self,
        current_price: float,
        rsi: float,
        volatility: float,
        volume: float,
        prev_close: Optional[float] = None
    ) -> TradingSignal:
        """
        Generate trading signals based on technical indicators
        
        Buy Signal: RSI < 30 (oversold) or RSI 40-50 with rising price
        Sell Signal: RSI > 70 (overbought) or RSI rising above 70
        """
        buy_signal = False
        sell_signal = False
        confidence = 0.0
        reason = ""
        
        # Adjust thresholds for volatility
        volatility_factor = volatility / 0.02  # 2% is baseline
        
        # Buy signals (RSI oversold + low volatility, or momentum)
        if rsi < (30 * volatility_factor):  # Deeper oversold in high vol
            buy_signal = True
            confidence = (30 - rsi) / 30  # 0-1 based on how oversold
            reason = f"Oversold RSI: {rsi:.1f}"
        elif rsi < 50 and 40 < rsi < 55:
            buy_signal = True
            confidence = 0.4
            reason = f"Momentum RSI: {rsi:.1f}"
        
        # Sell signals (RSI overbought)
        if rsi > (70 * volatility_factor) or rsi > 80:
            sell_signal = True
            confidence = (rsi - 70) / 30  # 0-1 based on how overbought
            reason = f"Overbought RSI: {rsi:.1f}"
        
        # Add risk management constraints based on volatility
        if volatility > self.trading_params.volatility_threshold:
            # High volatility: be more conservative
            buy_signal = buy_signal and (rsi < 30)  # Only very oversold
            confidence *= (1 - (volatility - self.trading_params.volatility_threshold) / 0.1)
        
        # Ensure confidence is in valid range
        confidence = min(max(confidence, 0.0), 1.0)
        
        return TradingSignal(
            buy=buy_signal,
            sell=sell_signal,
            confidence=confidence,
            reason=reason
        )
    
    def _can_open_position(self) -> bool:
        """Check if new position can be opened"""
        # Check max concurrent positions
        if len(self.open_positions) >= self.trading_params.max_concurrent_positions:
            return False
        
        # Check if we have cash available
        if self.current_capital < self.initial_capital * 0.1:  # Keep 10% cash minimum
            return False
        
        return True
    
    def _process_buy_signal(
        self,
        date: datetime,
        price: float,
        signals: TradingSignal,
        volatility: float,
        rsi: float
    ) -> None:
        """Process buy signal and open position"""
        symbol = 'AAPL'
        
        # Calculate stop-loss and take-profit first
        stop_loss = price * (1 - self.trading_params.stop_loss_percent)
        take_profit = self.tp_calculator.calculate_tp_price(
            entry_price=price,
            current_price=price,
            confidence=signals.confidence,
            volatility=volatility,
            rsi=rsi
        )
        
        # Calculate position size
        shares, position_value = self.position_sizer.calculate_position_size(
            portfolio_value=self.current_capital,
            entry_price=price,
            stop_loss_price=stop_loss,
            confidence=signals.confidence,
            volatility=volatility,
            existing_positions=list(self.open_positions.values())
        )
        
        if shares < self.trading_params.min_position_size:
            return
        
        # Check diversification
        if self.use_risk_management:
            allowed, reason = self.diversification_mgr.check_single_stock_limit(
                symbol=symbol,
                proposed_position_value=position_value,
                total_portfolio_value=self.current_capital + position_value,
                existing_positions=list(self.open_positions.values())
            )
            if not allowed:
                return
        
        # Don't use more than 50% on one trade
        if position_value > self.current_capital * 0.5:
            return
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            trailing_stop_active=True
        )
        
        # Store stop-loss
        self.stop_losses[symbol] = EnhancedStopLoss(
            entry_price=price,
            initial_stop_loss_percent=self.trading_params.stop_loss_percent
        )
        
        # Update state
        self.open_positions[symbol] = position
        self.current_capital -= position_value
    
    def _process_exits(
        self,
        date: datetime,
        current_price: float,
        low_price: float,
        high_price: float
    ) -> None:
        """Check for and process exit conditions"""
        for symbol in list(self.open_positions.keys()):
            position = self.open_positions[symbol]
            
            # Check stop-loss (use low price for execution)
            if symbol in self.stop_losses:
                sl = self.stop_losses[symbol]
                triggered, reason = sl.check_trigger(low_price)
                if triggered:
                    self._close_position(symbol, date, low_price, 'STOP_LOSS')
                    continue
            
            # Check take-profit (use high price for execution)
            if position.take_profit_price > 0:
                if high_price >= position.take_profit_price:
                    self._close_position(symbol, date, position.take_profit_price, 'TAKE_PROFIT')
                    continue
            
            # Check time-based stop (exceed max hold time)
            max_hold = 20  # 20 days max hold
            if (date - position.entry_date).days > max_hold:
                self._close_position(symbol, date, current_price, 'TIME_STOP')
    
    def _process_sell_signal(
        self,
        date: datetime,
        price: float,
        signals: Any,
        reason: str
    ) -> None:
        """Process sell signal"""
        self._close_position('AAPL', date, price, reason)
    
    def _close_position(
        self,
        symbol: str,
        date: datetime,
        exit_price: float,
        exit_reason: str
    ) -> None:
        """Close a position and record the trade"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        
        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.shares
        pnl_percent = (exit_price - position.entry_price) / position.entry_price
        hold_days = (date - position.entry_date).days
        
        # Record trade
        trade = BacktestTrade(
            symbol=symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=date,
            exit_price=exit_price,
            shares=position.shares,
            entry_reason='BUY_SIGNAL',
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_percent=pnl_percent,
            position_duration_days=hold_days
        )
        
        self.closed_trades.append(trade)
        self.daily_returns.append(pnl_percent)
        
        # Update capital
        self.current_capital += position.shares * exit_price
        
        # Clean up
        del self.open_positions[symbol]
        if symbol in self.stop_losses:
            del self.stop_losses[symbol]
    
    def _record_portfolio_snapshot(self, date: datetime, current_price: float) -> None:
        """Record daily portfolio state"""
        positions_value = sum(
            pos.shares * current_price for pos in self.open_positions.values()
        )
        total_value = self.current_capital + positions_value
        
        # Calculate returns
        daily_return = (total_value - self.initial_capital) / self.initial_capital if len(self.portfolio_history) == 0 else \
                      (total_value - self.portfolio_history[-1].total_value) / self.portfolio_history[-1].total_value
        
        cumulative_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate max drawdown so far
        max_val = self.initial_capital
        max_dd = 0.0
        for snapshot in self.portfolio_history:
            if snapshot.total_value > max_val:
                max_val = snapshot.total_value
            dd = (max_val - snapshot.total_value) / max_val if max_val > 0 else 0
            max_dd = max(max_dd, dd)
        
        # Apply latest drawdown
        max_val = max(max_val, total_value)
        current_dd = (max_val - total_value) / max_val if max_val > 0 else 0
        max_dd = max(max_dd, current_dd)
        
        # Calculate portfolio heat
        # Calculate portfolio heat (simplified since risk_monitor has different signature)
        portfolio_heat = 0.0
        if self.use_risk_management and self.open_positions:
            # Simple portfolio heat calculation: average unrealized loss
            for pos in self.open_positions.values():
                unrealized_pnl_percent = (current_price - pos.entry_price) / pos.entry_price
                if unrealized_pnl_percent < 0:
                    portfolio_heat += abs(unrealized_pnl_percent) * 100
            portfolio_heat = min(100, portfolio_heat / max(1, len(self.open_positions)))
        
        snapshot = PortfolioSnapshot(
            date=date,
            total_value=total_value,
            cash=self.current_capital,
            positions_value=positions_value,
            num_open_positions=len(self.open_positions),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            max_drawdown=max_dd,
            portfolio_heat_score=portfolio_heat
        )
        
        self.portfolio_history.append(snapshot)
    
    def _calculate_results(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResults:
        """Calculate all performance metrics"""
        
        # Basic metrics
        final_capital = self.current_capital + sum(
            pos.shares * 0 for pos in self.open_positions.values()  # Positions closed at 0
        )
        
        roi = (final_capital - self.initial_capital) / self.initial_capital
        total_trades = len(self.closed_trades)
        
        # Trade statistics
        if total_trades > 0:
            winning_trades = sum(1 for t in self.closed_trades if t.is_winning_trade)
            losing_trades = sum(1 for t in self.closed_trades if t.is_losing_trade)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            wins = [t.pnl_percent for t in self.closed_trades if t.is_winning_trade]
            losses = [t.pnl_percent for t in self.closed_trades if t.is_losing_trade]
            
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            
            gross_profit = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            avg_hold = np.mean([t.position_duration_days for t in self.closed_trades])
        else:
            winning_trades = losing_trades = 0
            win_rate = 0
            avg_win = avg_loss = 0
            profit_factor = 0
            avg_hold = 0
        
        # Risk metrics
        daily_returns = np.array(self.daily_returns)
        if len(daily_returns) > 0:
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            
            # Sharpe Ratio (annualized, assuming 0% risk-free rate)
            if annual_volatility > 0:
                annual_return = (daily_returns.mean() * 252)
                sharpe_ratio = annual_return / annual_volatility
            else:
                sharpe_ratio = 0.0
            
            # Max Drawdown
            max_drawdown = 0.0
            peak = self.initial_capital
            for snapshot in self.portfolio_history:
                if snapshot.total_value > peak:
                    peak = snapshot.total_value
                dd = (peak - snapshot.total_value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)
            
            # Calmar Ratio
            calmar_ratio = roi / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino Ratio (downside deviation only)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = sharpe_ratio  # If no down days, use Sharpe
        else:
            annual_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
            sortino_ratio = 0
        
        # Create results
        results = BacktestResults(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            roi=roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_hold_days=avg_hold,
            annual_volatility=annual_volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            trades=self.closed_trades.copy(),
            portfolio_history=self.portfolio_history.copy()
        )
        
        return results
    
    def reset_backtest(self) -> None:
        """Reset state for new backtest"""
        self.current_capital = self.initial_capital
        self.open_positions = {}
        self.closed_trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.stop_losses = {}
