"""
Task 5.3: Trade History and Portfolio Management
Portfolio tracker, trade history management, and portfolio metrics calculation

Classes:
- Trade: Represents individual trades
- Portfolio: Manages portfolio state and calculations
- PortfolioTracker: Tracks all portfolio activity
- TradeHistoryFilter: Filters and searches trades
- PortfolioVisualizer: Formats portfolio data for visualization

Author: AI Trading System
Date: March 6, 2026
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class TradeAction(Enum):
    """Trade action types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeStatus(Enum):
    """Trade status types"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


@dataclass
class Trade:
    """Represents an individual trade"""
    trade_id: str
    date: str
    symbol: str
    action: str  # BUY or SELL
    quantity: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    status: str = "OPEN"
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    def is_closed(self):
        """Check if trade is closed"""
        return self.status == "CLOSED" and self.exit_price is not None
    
    def calculate_pnl(self):
        """Calculate PnL if trade is closed"""
        if not self.is_closed():
            return None, None
        
        if self.action == "BUY":
            pnl = (self.exit_price - self.entry_price) * self.quantity
            pnl_percent = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SELL
            pnl = (self.entry_price - self.exit_price) * self.quantity
            pnl_percent = ((self.entry_price - self.exit_price) / self.exit_price) * 100
        
        self.pnl = pnl
        self.pnl_percent = pnl_percent
        return pnl, pnl_percent
    
    def get_risk_reward_ratio(self):
        """Calculate risk/reward ratio"""
        if self.action != "BUY" or not self.stop_loss or not self.take_profit:
            return None
        
        risk = self.entry_price - self.stop_loss
        reward = self.take_profit - self.entry_price
        
        if risk <= 0:
            return None
        
        return reward / risk


class Portfolio:
    """Manages portfolio state and metrics"""
    
    def __init__(self, initial_balance: float = 100000.0):
        """Initialize portfolio"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}  # symbol -> quantity
        self.trades = []  # History of all trades
        self.equity_value = initial_balance
        self.market_prices = {}  # Current market prices for positions
        
    def add_trade(self, trade: Trade):
        """Add a trade to portfolio"""
        self.trades.append(trade)
        
        if trade.action == "BUY":
            # Deduct cash and add position
            cost = trade.quantity * trade.entry_price
            self.cash -= cost
            self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) + trade.quantity
        
        elif trade.action == "SELL":
            # Add cash and reduce position
            proceeds = trade.quantity * trade.entry_price
            self.cash += proceeds
            self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) - trade.quantity
            
            # Remove if no shares left
            if self.positions.get(trade.symbol, 0) <= 0:
                self.positions.pop(trade.symbol, None)
    
    def update_market_price(self, symbol: str, price: float):
        """Update current market price"""
        self.market_prices[symbol] = price
    
    def calculate_position_value(self):
        """Calculate total value of open positions"""
        position_value = 0
        for symbol, quantity in self.positions.items():
            price = self.market_prices.get(symbol, 0)
            position_value += quantity * price
        return position_value
    
    def calculate_equity_value(self):
        """Calculate total portfolio equity value"""
        self.equity_value = self.cash + self.calculate_position_value()
        return self.equity_value
    
    def calculate_total_gain_loss(self):
        """Calculate total gain/loss from closed trades"""
        total_pnl = 0
        for trade in self.trades:
            if trade.is_closed():
                pnl, _ = trade.calculate_pnl()
                if pnl:
                    total_pnl += pnl
        return total_pnl
    
    def calculate_total_gain_loss_percent(self):
        """Calculate total gain/loss as percentage"""
        total_pnl = self.calculate_total_gain_loss()
        initial = self.initial_balance
        return (total_pnl / initial) * 100 if initial > 0 else 0
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02):
        """Calculate Sharpe ratio"""
        # Get daily returns from trades
        if len(self.trades) < 2:
            return 0
        
        daily_returns = []
        sorted_trades = sorted(self.trades, key=lambda t: t.date)
        
        for i, trade in enumerate(sorted_trades):
            if trade.is_closed():
                pnl, pnl_pct = trade.calculate_pnl()
                daily_returns.append(pnl_pct / 100)
        
        if len(daily_returns) < 2:
            return 0
        
        # Calculate Sharpe ratio
        returns_array = np.array(daily_returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        equity_curve = [self.initial_balance]
        cumulative_pnl = 0
        
        for trade in sorted(self.trades, key=lambda t: t.date):
            if trade.is_closed():
                pnl, _ = trade.calculate_pnl()
                cumulative_pnl += pnl if pnl else 0
                equity_curve.append(self.initial_balance + cumulative_pnl)
        
        if not equity_curve:
            return 0
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return max_drawdown * 100 if max_drawdown != 0 else 0
    
    def get_portfolio_metrics(self):
        """Get comprehensive portfolio metrics"""
        self.calculate_equity_value()
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.cash,
            'position_value': self.calculate_position_value(),
            'equity_value': self.equity_value,
            'total_pnl': self.calculate_total_gain_loss(),
            'total_pnl_percent': self.calculate_total_gain_loss_percent(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'num_trades': len(self.trades),
            'num_closed_trades': sum(1 for t in self.trades if t.is_closed()),
            'num_open_trades': sum(1 for t in self.trades if not t.is_closed()),
            'win_rate': self._calculate_win_rate(),
            'average_win': self._calculate_average_win(),
            'average_loss': self._calculate_average_loss(),
            'profit_factor': self._calculate_profit_factor(),
        }
    
    def _calculate_win_rate(self):
        """Calculate percentage of winning trades"""
        closed_trades = [t for t in self.trades if t.is_closed()]
        if not closed_trades:
            return 0
        
        winners = sum(1 for t in closed_trades if t.pnl and t.pnl > 0)
        return (winners / len(closed_trades)) * 100
    
    def _calculate_average_win(self):
        """Calculate average win size"""
        winners = [t for t in self.trades if t.is_closed() and t.pnl and t.pnl > 0]
        if not winners:
            return 0
        return sum(t.pnl for t in winners) / len(winners)
    
    def _calculate_average_loss(self):
        """Calculate average loss size"""
        losers = [t for t in self.trades if t.is_closed() and t.pnl and t.pnl < 0]
        if not losers:
            return 0
        return sum(abs(t.pnl) for t in losers) / len(losers)
    
    def _calculate_profit_factor(self):
        """Calculate profit factor (gross profit / gross loss)"""
        total_wins = sum(t.pnl for t in self.trades if t.is_closed() and t.pnl and t.pnl > 0)
        total_losses = sum(abs(t.pnl) for t in self.trades if t.is_closed() and t.pnl and t.pnl < 0)
        
        if total_losses == 0:
            return 0 if total_wins == 0 else float('inf')
        
        return total_wins / total_losses


class PortfolioTracker:
    """Tracks portfolio activity and history"""
    
    def __init__(self, initial_balance: float = 100000.0):
        """Initialize tracker"""
        self.portfolio = Portfolio(initial_balance)
        self.trade_history = []
        self.performance_history = []
        
    def load_from_backtest(self, backtest_results_path: str):
        """Load trades from backtesting results"""
        try:
            with open(backtest_results_path, 'r') as f:
                backtest_data = json.load(f)
            
            # Extract trades from backtest results (check different possible structures)
            trades_list = None
            
            # Structure 1: backtest_data['trades']
            if 'trades' in backtest_data:
                trades_list = backtest_data['trades']
            # Structure 2: backtest_data['strategies'][strategy_name]['trades']
            elif 'strategies' in backtest_data:
                # Get the first strategy's trades
                for strategy_name, strategy_data in backtest_data['strategies'].items():
                    if 'trades' in strategy_data:
                        trades_list = strategy_data['trades']
                        break
            
            if trades_list:
                # Load trades without executing them (they're already closed from backtest)
                for trade_data in trades_list:
                    pnl = float(trade_data.get('pnl', 0))
                    
                    trade = Trade(
                        trade_id=f"TRADE_{len(self.portfolio.trades)}",
                        date=trade_data.get('entry_date', datetime.now().isoformat()),
                        symbol=trade_data.get('symbol', 'AAPL'),
                        action='BUY' if trade_data.get('type') == 'LONG' else 'SELL',
                        quantity=float(trade_data.get('shares', 0)),
                        entry_price=float(trade_data.get('entry_price', 0)),
                        stop_loss=None,
                        take_profit=None,
                        exit_date=trade_data.get('exit_date'),
                        exit_price=float(trade_data.get('exit_price', 0)) if trade_data.get('exit_price') else None,
                        status='CLOSED',
                        pnl=pnl,
                        pnl_percent=float(trade_data.get('pnl_percent', 0))
                    )
                    # Add directly to trades list without executing through portfolio.add_trade()
                    self.portfolio.trades.append(trade)
                    self.trade_history.append(trade)
                
                # Fix portfolio state: add all P&L to cash, set positions to empty for backtest data
                total_pnl = sum(t.pnl for t in self.portfolio.trades if t.pnl)
                self.portfolio.cash = self.portfolio.initial_balance + total_pnl
                self.portfolio.positions = {}  # No open positions from backtest
                self.portfolio.equity_value = self.portfolio.cash
            
            return len(self.trade_history)
        except Exception as e:
            print(f"Error loading backtest data: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def load_from_csv(self, csv_path: str):
        """Load trades from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                trade = Trade(
                    trade_id=row.get('trade_id', f"TRADE_{len(self.portfolio.trades)}"),
                    date=row.get('date', datetime.now().isoformat()),
                    symbol=row.get('symbol', 'AAPL'),
                    action=row.get('action', 'BUY'),
                    quantity=float(row.get('quantity', 0)),
                    entry_price=float(row.get('entry_price', 0)),
                    stop_loss=float(row.get('stop_loss')) if pd.notna(row.get('stop_loss')) else None,
                    take_profit=float(row.get('take_profit')) if pd.notna(row.get('take_profit')) else None,
                    exit_date=row.get('exit_date'),
                    exit_price=float(row.get('exit_price')) if pd.notna(row.get('exit_price')) else None,
                    status=row.get('status', 'OPEN'),
                )
                self.portfolio.add_trade(trade)
                self.trade_history.append(trade)
            
            return len(self.trade_history)
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return 0
    
    def add_trade(self, trade: Trade):
        """Add a new trade"""
        self.portfolio.add_trade(trade)
        self.trade_history.append(trade)
    
    def close_trade(self, trade_id: str, exit_price: float, exit_date: Optional[str] = None):
        """Close an open trade"""
        for trade in self.portfolio.trades:
            if trade.trade_id == trade_id and trade.status == "OPEN":
                trade.exit_price = exit_price
                trade.exit_date = exit_date or datetime.now().isoformat()
                trade.status = "CLOSED"
                trade.calculate_pnl()
                return True
        return False
    
    def get_portfolio_summary(self):
        """Get portfolio summary"""
        return self.portfolio.get_portfolio_metrics()
    
    def get_trade_history(self):
        """Get trade history as list of dicts"""
        return [trade.to_dict() for trade in self.trade_history]


class TradeHistoryFilter:
    """Filter and search trades"""
    
    def __init__(self, trades: List[Trade]):
        """Initialize with trades list"""
        self.trades = trades
    
    def filter_by_date_range(self, start_date: str, end_date: str) -> List[Trade]:
        """Filter trades by date range"""
        filtered = []
        for trade in self.trades:
            trade_date = datetime.fromisoformat(trade.date)
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            if start <= trade_date <= end:
                filtered.append(trade)
        return filtered
    
    def filter_by_symbol(self, symbol: str) -> List[Trade]:
        """Filter trades by symbol"""
        return [t for t in self.trades if t.symbol == symbol]
    
    def filter_by_action(self, action: str) -> List[Trade]:
        """Filter trades by action (BUY/SELL)"""
        return [t for t in self.trades if t.action == action]
    
    def filter_by_status(self, status: str) -> List[Trade]:
        """Filter trades by status (OPEN/CLOSED)"""
        return [t for t in self.trades if t.status == status]
    
    def filter_by_symbol_and_date(self, symbol: str, start_date: str, end_date: str) -> List[Trade]:
        """Filter by both symbol and date range"""
        by_symbol = self.filter_by_symbol(symbol)
        temp_filter = TradeHistoryFilter(by_symbol)
        return temp_filter.filter_by_date_range(start_date, end_date)
    
    def search(self, **filters) -> List[Trade]:
        """Search with multiple filters"""
        result = self.trades
        
        if 'symbol' in filters:
            result = [t for t in result if t.symbol == filters['symbol']]
        
        if 'action' in filters:
            result = [t for t in result if t.action == filters['action']]
        
        if 'status' in filters:
            result = [t for t in result if t.status == filters['status']]
        
        if 'start_date' in filters and 'end_date' in filters:
            start = datetime.fromisoformat(filters['start_date'])
            end = datetime.fromisoformat(filters['end_date'])
            result = [t for t in result if start <= datetime.fromisoformat(t.date) <= end]
        
        if 'min_pnl' in filters:
            result = [t for t in result if t.pnl and t.pnl >= filters['min_pnl']]
        
        if 'max_pnl' in filters:
            result = [t for t in result if t.pnl and t.pnl <= filters['max_pnl']]
        
        return result


class PortfolioVisualizer:
    """Format portfolio data for visualization"""
    
    @staticmethod
    def format_trade_for_display(trade: Trade) -> Dict:
        """Format trade for table display"""
        return {
            'id': trade.trade_id,
            'date': trade.date[:10],  # YYYY-MM-DD
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': f"{trade.quantity:.2f}",
            'entry_price': f"${trade.entry_price:.2f}",
            'stop_loss': f"${trade.stop_loss:.2f}" if trade.stop_loss else "N/A",
            'take_profit': f"${trade.take_profit:.2f}" if trade.take_profit else "N/A",
            'status': trade.status,
            'exit_date': trade.exit_date[:10] if trade.exit_date else "Open",
            'exit_price': f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
            'pnl': f"${trade.pnl:.2f}" if trade.pnl else "N/A",
            'pnl_percent': f"{trade.pnl_percent:.2f}%" if trade.pnl_percent else "N/A",
        }
    
    @staticmethod
    def format_portfolio_summary(metrics: Dict) -> Dict:
        """Format portfolio metrics for display"""
        return {
            'initial_balance': f"${metrics['initial_balance']:,.2f}",
            'current_balance': f"${metrics['current_balance']:,.2f}",
            'position_value': f"${metrics['position_value']:,.2f}",
            'equity_value': f"${metrics['equity_value']:,.2f}",
            'total_pnl': f"${metrics['total_pnl']:,.2f}",
            'total_pnl_percent': f"{metrics['total_pnl_percent']:.2f}%",
            'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
            'max_drawdown': f"{metrics['max_drawdown']:.2f}%",
            'num_trades': metrics['num_trades'],
            'num_closed_trades': metrics['num_closed_trades'],
            'num_open_trades': metrics['num_open_trades'],
            'win_rate': f"{metrics['win_rate']:.1f}%",
            'average_win': f"${metrics['average_win']:,.2f}",
            'average_loss': f"${metrics['average_loss']:,.2f}",
            'profit_factor': f"{metrics['profit_factor']:.2f}",
        }
    
    @staticmethod
    def get_asset_allocation(portfolio: Portfolio) -> Dict:
        """Get asset allocation for pie chart"""
        allocation = {}
        total_value = portfolio.calculate_equity_value()
        
        # Add cash
        if total_value > 0:
            allocation['Cash'] = (portfolio.cash / total_value) * 100
        
        # Add positions
        for symbol, quantity in portfolio.positions.items():
            price = portfolio.market_prices.get(symbol, 0)
            position_value = quantity * price
            if total_value > 0:
                allocation[symbol] = (position_value / total_value) * 100
        
        return allocation
    
    @staticmethod
    def get_trade_statistics(trades: List[Trade]) -> Dict:
        """Get trade statistics for bar chart"""
        by_symbol = {}
        
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            
            by_symbol[trade.symbol]['total_trades'] += 1
            
            if trade.is_closed():
                if trade.pnl and trade.pnl > 0:
                    by_symbol[trade.symbol]['winning_trades'] += 1
                elif trade.pnl and trade.pnl < 0:
                    by_symbol[trade.symbol]['losing_trades'] += 1
                
                if trade.pnl:
                    by_symbol[trade.symbol]['total_pnl'] += trade.pnl
        
        # Calculate win rates
        for symbol in by_symbol:
            total = by_symbol[symbol]['total_trades']
            if total > 0:
                by_symbol[symbol]['win_rate'] = (by_symbol[symbol]['winning_trades'] / total) * 100
        
        return by_symbol
    
    @staticmethod
    def get_daily_equity_curve(portfolio: Portfolio) -> Dict:
        """Get daily equity curve for line chart - from first trade to today"""
        equity_curve = []
        dates = []
        cumulative_pnl = 0
        
        # Get trades sorted by exit date (chronologically)
        sorted_trades = sorted(portfolio.trades, key=lambda t: t.exit_date if t.exit_date else t.date)
        
        if not sorted_trades:
            # No trades - show initial balance to today
            equity_curve = [portfolio.initial_balance, portfolio.initial_balance]
            dates = [(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), 
                    datetime.now().strftime('%Y-%m-%d')]
            return {'dates': dates, 'equity': equity_curve}
        
        # Add starting point (first trade date)
        first_date = sorted_trades[0].date[:10] if sorted_trades[0].date else datetime.now().strftime('%Y-%m-%d')
        equity_curve.append(portfolio.initial_balance)
        dates.append(first_date)
        
        # Process each closed trade chronologically
        last_date = first_date
        for trade in sorted_trades:
            if trade.is_closed():
                pnl, _ = trade.calculate_pnl()
                cumulative_pnl += pnl if pnl else 0
                equity_curve.append(portfolio.initial_balance + cumulative_pnl)
                trade_date = trade.exit_date[:10] if trade.exit_date else trade.date[:10]
                dates.append(trade_date)
                last_date = trade_date
        
        # Extend to today with current balance (LIVE extension)
        last_equity = equity_curve[-1] if equity_curve else portfolio.initial_balance
        last_date_obj = datetime.strptime(last_date, '%Y-%m-%d')
        today = datetime.now()
        
        # Add intermediate points for visual continuity if gap > 30 days
        if (today - last_date_obj).days > 30:
            # Add a mid-point
            mid_date = last_date_obj + (today - last_date_obj) / 2
            equity_curve.append(last_equity)
            dates.append(mid_date.strftime('%Y-%m-%d'))
        
        # Add today's balance
        if last_date != today.strftime('%Y-%m-%d'):
            equity_curve.append(portfolio.cash)  # Current balance (which includes all historical P&L)
            dates.append(today.strftime('%Y-%m-%d'))
        
        return {
            'dates': dates,
            'equity': equity_curve,
        }
    
    @staticmethod
    def get_pnl_distribution(trades: List[Trade]) -> Dict:
        """Get PnL distribution for histogram"""
        pnl_values = []
        
        for trade in trades:
            if trade.is_closed() and trade.pnl:
                pnl_values.append(trade.pnl)
        
        if not pnl_values:
            return {'bins': [], 'count': []}
        
        # Create histogram data
        hist, bin_edges = np.histogram(pnl_values, bins=10)
        
        return {
            'bins': [f"${edge:.0f}" for edge in bin_edges[:-1]],
            'count': hist.tolist(),
        }
