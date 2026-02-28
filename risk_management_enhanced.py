"""
Task 4.3: Enhanced Risk Management Features
Implements advanced risk management: stop-loss, take-profit, diversification, 
and dynamic position sizing

Key Features:
- Enhanced stop-loss with trailing stops
- Dynamic take-profit targeting
- Portfolio diversification rules (stock/sector limits)
- Confidence-based position sizing
- Volatility-adjusted position sizing
- Comprehensive risk metrics
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict

from trading_rules import (
    TradingParameters, Position, RiskManager
)


# ============================================================================
# ENHANCED RISK MANAGEMENT DATA STRUCTURES
# ============================================================================

@dataclass
class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "LOW"              # < 1% drawdown
    MODERATE = "MODERATE"    # 1-3% drawdown
    HIGH = "HIGH"            # 3-5% drawdown
    CRITICAL = "CRITICAL"    # > 5% drawdown


@dataclass
class ExitSignal:
    """Result of exit condition check"""
    should_exit: bool
    reason: str  # "STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", "REVERSAL", "TIME_STOP"
    exit_price: float
    pnl: float
    pnl_percent: float


@dataclass
class PositionRiskMetrics:
    """Risk metrics for a single position"""
    position_id: int
    symbol: str
    entry_price: float
    current_price: float
    shares: int
    entry_date: datetime
    
    # Risk tracking
    unrealized_pnl: float
    unrealized_pnl_percent: float
    distance_to_sl: float  # percentage
    distance_to_tp: float  # percentage
    position_size_percent: float  # % of portfolio
    
    # Risk level
    risk_level: RiskLevel
    heat_score: float  # 0-100, how risky is this position
    

@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    total_portfolio_value: float
    cash_balance: float
    cash_percent: float
    
    num_positions: int
    total_exposure_percent: float
    
    # Sector/Stock exposure
    stock_maximum_exposure: float  # % of portfolio
    sector_maximum_exposure: float  # % of portfolio
    
    # Overall risk
    portfolio_drawdown: float
    portfolio_risk_level: RiskLevel
    portfolio_heat_score: float  # Average heat across positions
    
    # Loss limits
    daily_loss: float
    daily_loss_percent: float
    max_loss_breached: bool


# ============================================================================
# ENHANCED STOP-LOSS SYSTEM
# ============================================================================

class EnhancedStopLoss:
    """Advanced stop-loss management with multiple trigger types"""
    
    def __init__(self, entry_price: float, initial_stop_loss_percent: float):
        self.entry_price = entry_price
        self.initial_stop_loss_percent = initial_stop_loss_percent
        self.stop_price = entry_price * (1 - initial_stop_loss_percent)
        
        # Tracking
        self.highest_price = entry_price
        self.trigger_history: List[Dict] = []
        self.triggered = False
        self.trigger_price = None
        self.trigger_reason = None
    
    def check_trigger(self, current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if stop-loss is triggered
        
        Returns:
            (triggered, reason)
        """
        if self.triggered:
            return True, self.trigger_reason
        
        loss_percent = (current_price - self.entry_price) / self.entry_price
        
        if current_price < self.stop_price:
            self.triggered = True
            self.trigger_price = current_price
            self.trigger_reason = f"STOP_LOSS at {loss_percent:.2%}"
            self.trigger_history.append({
                'timestamp': datetime.now(),
                'trigger_price': current_price,
                'reason': self.trigger_reason
            })
            return True, self.trigger_reason
        
        return False, None
    
    def update_stop_price(self, new_stop_price: float):
        """Only allow stop-loss to move UP (toward entry), not down"""
        if new_stop_price > self.stop_price:
            self.stop_price = new_stop_price
    
    def calculate_distance_to_trigger(self, current_price: float) -> float:
        """Distance in percent to stop-loss trigger"""
        if current_price <= self.stop_price:
            return 0.0
        return (current_price - self.stop_price) / self.stop_price


class TrailingStopLoss:
    """Trailing stop-loss that follows price up but not down"""
    
    def __init__(self, entry_price: float, trailing_percent: float):
        self.entry_price = entry_price
        self.trailing_percent = trailing_percent
        
        self.highest_price = entry_price
        self.trailing_stop_price = entry_price * (1 - trailing_percent)
        
        self.triggered = False
        self.trigger_price = None
    
    def update(self, current_price: float) -> Tuple[bool, Optional[float]]:
        """
        Update trailing stop with current price
        
        Returns:
            (triggered, new_stop_price)
        """
        if self.triggered:
            return True, self.trigger_price
        
        # Update highest price
        if current_price > self.highest_price:
            self.highest_price = current_price
            # Move stop-loss up
            self.trailing_stop_price = current_price * (1 - self.trailing_percent)
        
        # Check if triggered
        if current_price < self.trailing_stop_price:
            self.triggered = True
            self.trigger_price = current_price
            return True, self.trigger_price
        
        return False, self.trailing_stop_price
    
    def get_distance_to_trigger(self, current_price: float) -> float:
        """Distance to trigger in percent"""
        if current_price <= self.trailing_stop_price:
            return 0.0
        return (current_price - self.trailing_stop_price) / self.trailing_stop_price


# ============================================================================
# DYNAMIC TAKE-PROFIT SYSTEM
# ============================================================================

class DynamicTakeProfitCalculator:
    """Calculate dynamic take-profit based on multiple factors"""
    
    def __init__(self, params: TradingParameters):
        self.params = params
        self.base_tp_percent = params.take_profit_target
    
    def calculate_tp_price(
        self,
        entry_price: float,
        current_price: float,
        confidence: float,
        volatility: float,
        rsi: float
    ) -> float:
        """
        Calculate dynamic take-profit price based on:
        - Base target (2.5%)
        - Confidence level (higher confidence = higher target)
        - Volatility (higher volatility = higher target for compensation)
        - RSI level (overbought = tighter target)
        
        Returns:
            target price in dollars
        """
        
        tp_percent = self.base_tp_percent
        
        # Confidence adjustment (0.5 - 1.5x)
        confidence_multiplier = 0.5 + (confidence * 1.0)
        tp_percent *= confidence_multiplier
        
        # Volatility adjustment (+10% to +50%)
        vol_adjustment = min(0.50, volatility * 10)  # 1% vol = +10%, max +50%
        tp_percent += vol_adjustment
        
        # RSI adjustment (if overbought, tighten target)
        if rsi > 70:
            tp_percent *= 0.8  # Reduce by 20%
        elif rsi < 40:
            tp_percent *= 1.1  # Increase by 10%
        
        # Calculate target price
        target_price = entry_price * (1 + tp_percent)
        
        return target_price
    
    def is_tp_reached(
        self,
        current_price: float,
        tp_target: float,
        entry_side: str = "BUY"
    ) -> bool:
        """Check if take-profit target is reached"""
        if entry_side == "BUY":
            return current_price >= tp_target
        else:  # SELL
            return current_price <= tp_target


# ============================================================================
# PORTFOLIO DIVERSIFICATION RULES
# ============================================================================

class PortfolioDiversificationManager:
    """Enforce portfolio diversification constraints"""
    
    def __init__(self):
        self.params = TradingParameters()
        
        # Constraints
        self.max_single_stock_exposure = 0.25  # Max 25% in one stock
        self.max_sector_exposure = 0.40        # Max 40% in one sector
        self.max_correlation_threshold = 0.85  # Max correlation between positions
        
        # Stock to sector mapping (simplified)
        self.stock_sector_map = {
            "AAPL": "TECHNOLOGY",
            "MSFT": "TECHNOLOGY",
            "GOOGL": "TECHNOLOGY",
            "TSLA": "AUTOMOTIVE",
            "F": "AUTOMOTIVE",
            "GM": "AUTOMOTIVE",
            "JPM": "FINANCE",
            "BAC": "FINANCE",
            "GS": "FINANCE",
            "JNJ": "HEALTHCARE",
            "PFE": "HEALTHCARE",
            "UNH": "HEALTHCARE",
        }
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for stock symbol"""
        return self.stock_sector_map.get(symbol, "OTHER")
    
    def check_single_stock_limit(
        self,
        symbol: str,
        proposed_position_value: float,
        total_portfolio_value: float,
        existing_positions: List[Position]
    ) -> Tuple[bool, str]:
        """
        Check if adding position would violate single-stock limit
        
        Returns:
            (allowed, reason)
        """
        proposed_exposure = proposed_position_value / total_portfolio_value
        
        # Check existing exposure to this stock
        existing_exposure = 0.0
        for pos in existing_positions:
            if pos.symbol == symbol:
                existing_exposure += pos.entry_value / total_portfolio_value
        
        total_exposure = proposed_exposure + existing_exposure
        
        if total_exposure > self.max_single_stock_exposure:
            reason = (f"Stock {symbol} exposure {total_exposure:.1%} exceeds limit "
                     f"{self.max_single_stock_exposure:.1%}")
            return False, reason
        
        return True, f"Stock {symbol} exposure OK at {total_exposure:.1%}"
    
    def check_sector_limit(
        self,
        symbol: str,
        proposed_position_value: float,
        total_portfolio_value: float,
        existing_positions: List[Position]
    ) -> Tuple[bool, str]:
        """
        Check if adding position would violate sector limit
        
        Returns:
            (allowed, reason)
        """
        
        sector = self.get_sector(symbol)
        proposed_exposure = proposed_position_value / total_portfolio_value
        
        # Calculate existing sector exposure
        sector_exposure = 0.0
        for pos in existing_positions:
            pos_sector = self.get_sector(pos.symbol)
            if pos_sector == sector:
                sector_exposure += pos.entry_value / total_portfolio_value
        
        total_sector_exposure = proposed_exposure + sector_exposure
        
        if total_sector_exposure > self.max_sector_exposure:
            reason = (f"Sector {sector} exposure {total_sector_exposure:.1%} exceeds limit "
                     f"{self.max_sector_exposure:.1%}")
            return False, reason
        
        return True, f"Sector {sector} exposure OK at {total_sector_exposure:.1%}"
    
    def get_portfolio_exposure(
        self,
        positions: List[Position],
        total_value: float
    ) -> Dict[str, float]:
        """Get exposure by stock and sector"""
        
        exposure = {
            'by_stock': defaultdict(float),
            'by_sector': defaultdict(float),
        }
        
        for pos in positions:
            stock_exposure = pos.entry_value / total_value
            sector = self.get_sector(pos.symbol)
            
            exposure['by_stock'][pos.symbol] += stock_exposure
            exposure['by_sector'][sector] += stock_exposure
        
        return exposure


# ============================================================================
# DYNAMIC POSITION SIZING
# ============================================================================

class DynamicPositionSizer:
    """Calculate position size dynamically based on risk factors"""
    
    def __init__(self, params: TradingParameters):
        self.params = params
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        confidence: float,
        volatility: float,
        existing_positions: List[Position] = None
    ) -> Tuple[int, float]:
        """
        Calculate optimal position size based on:
        - Risk management (max loss per trade)
        - Confidence level
        - Volatility
        - Existing positions
        
        Returns:
            (shares, position_value)
        """
        
        if existing_positions is None:
            existing_positions = []
        
        # Calculate risk per trade
        risk_amount = portfolio_value * self.params.risk_percentage
        
        # Distance to stop-loss
        distance_to_sl = abs(entry_price - stop_loss_price)
        
        # Base position size
        base_shares = int(risk_amount / distance_to_sl)
        
        # Adjust for confidence (0.5x to 1.5x)
        confidence_multiplier = 0.5 + (confidence * 1.0)
        
        # Adjust for volatility (reduce in high volatility)
        volatility_multiplier = max(0.5, 1.0 - (volatility * 2))
        
        # Apply multipliers
        adjusted_shares = int(base_shares * confidence_multiplier * volatility_multiplier)
        
        # Minimum position size
        adjusted_shares = max(adjusted_shares, self.params.min_position_size)
        
        # Apply portfolio constraints
        position_value = adjusted_shares * entry_price
        max_position_value = portfolio_value * self.params.max_position_value_percent
        
        if position_value > max_position_value:
            adjusted_shares = int(max_position_value / entry_price)
        
        # Final validation
        position_value = adjusted_shares * entry_price
        
        return adjusted_shares, position_value
    
    def calculate_volatility_adjusted_size(
        self,
        base_size: int,
        volatility: float,
        threshold: float = 0.03
    ) -> int:
        """
        Reduce position size in high volatility environments
        
        Args:
            base_size: Original position size
            volatility: Current volatility (as decimal, 0.01 = 1%)
            threshold: Volatility threshold (default 3%)
        
        Returns:
            Adjusted position size
        """
        
        if volatility <= threshold:
            return base_size
        
        # Above threshold: reduce size
        excess_volatility = volatility - threshold
        reduction_factor = max(0.5, 1.0 - (excess_volatility * 5))
        
        return int(base_size * reduction_factor)


# ============================================================================
# ENHANCED RISK MONITORING
# ============================================================================

class EnhancedRiskMonitor:
    """Monitor and report on portfolio risk in real-time"""
    
    def __init__(self, params: TradingParameters):
        self.params = params
        self.diversification_mgr = PortfolioDiversificationManager()
    
    def calculate_position_heat_score(
        self,
        position: Position,
        current_price: float,
        portfolio_value: float
    ) -> Tuple[float, RiskLevel]:
        """
        Calculate "heat score" (0-100) indicating risk level of position
        
        Factors:
        - Unrealized loss magnitude
        - Distance to stop-loss
        - Position size relative to portfolio
        
        Returns:
            (heat_score 0-100, risk_level)
        """
        
        unrealized_pnl_percent = (current_price - position.entry_price) / position.entry_price
        position_size_percent = (position.entry_price * position.shares) / portfolio_value
        
        # Loss magnitude score (0-30 points)
        loss_score = 0
        if unrealized_pnl_percent < 0:
            loss_magnitude = abs(unrealized_pnl_percent)
            loss_score = min(30, loss_magnitude * 100)  # 2% loss = 2 points, etc
        
        # Position size score (0-40 points)
        size_score = position_size_percent * 100
        
        # Distance to SL score (0-30 points)
        distance_to_sl = (current_price - position.stop_loss_price) / position.stop_loss_price
        if distance_to_sl < 0.05:  # Within 5% of SL
            sl_score = 30
        elif distance_to_sl < 0.10:  # Within 10% of SL
            sl_score = 20
        else:
            sl_score = 5
        
        heat_score = loss_score + size_score + sl_score
        
        # Determine risk level
        if heat_score < 25:
            risk_level = RiskLevel.LOW
        elif heat_score < 50:
            risk_level = RiskLevel.MODERATE
        elif heat_score < 75:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return min(100, heat_score), risk_level
    
    def calculate_portfolio_metrics(
        self,
        positions: List[Position],
        cash: float,
        current_prices: Dict[str, float],
        peak_portfolio_value: float,
        initial_capital: float
    ) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        total_position_value = sum(
            pos.entry_price * pos.shares for pos in positions
        )
        portfolio_value = cash + total_position_value
        
        # Exposure calculations
        cash_percent = cash / portfolio_value if portfolio_value > 0 else 0
        total_exposure_percent = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Get diversification exposure
        exposure = self.diversification_mgr.get_portfolio_exposure(positions, portfolio_value)
        max_stock_exposure = max(exposure['by_stock'].values()) if exposure['by_stock'] else 0
        max_sector_exposure = max(exposure['by_sector'].values()) if exposure['by_sector'] else 0
        
        # Risk calculations
        portfolio_drawdown = (portfolio_value - peak_portfolio_value) / peak_portfolio_value if peak_portfolio_value > 0 else 0
        daily_loss = portfolio_value - initial_capital
        daily_loss_percent = daily_loss / initial_capital if initial_capital > 0 else 0
        
        # Portfolio risk level
        if portfolio_drawdown > -0.01:
            portfolio_risk_level = RiskLevel.LOW
        elif portfolio_drawdown > -0.03:
            portfolio_risk_level = RiskLevel.MODERATE
        elif portfolio_drawdown > -0.05:
            portfolio_risk_level = RiskLevel.HIGH
        else:
            portfolio_risk_level = RiskLevel.CRITICAL
        
        # Portfolio heat score (average of position heat scores)
        heat_scores = []
        for pos in positions:
            score, _ = self.calculate_position_heat_score(pos, current_prices.get(pos.symbol, pos.entry_price), portfolio_value)
            heat_scores.append(score)
        portfolio_heat = np.mean(heat_scores) if heat_scores else 0
        
        # Check circuit breaker
        max_loss_breached = daily_loss_percent < self.params.portfolio_max_loss_percent
        
        return PortfolioRiskMetrics(
            total_portfolio_value=portfolio_value,
            cash_balance=cash,
            cash_percent=cash_percent,
            num_positions=len(positions),
            total_exposure_percent=total_exposure_percent,
            stock_maximum_exposure=max_stock_exposure,
            sector_maximum_exposure=max_sector_exposure,
            portfolio_drawdown=portfolio_drawdown,
            portfolio_risk_level=portfolio_risk_level,
            portfolio_heat_score=portfolio_heat,
            daily_loss=daily_loss,
            daily_loss_percent=daily_loss_percent,
            max_loss_breached=max_loss_breached
        )
    
    def print_risk_report(self, metrics: PortfolioRiskMetrics):
        """Print comprehensive risk report"""
        
        print("\n" + "="*80)
        print("ENHANCED RISK MANAGEMENT REPORT")
        print("="*80)
        
        print("\nPORTFOLIO OVERVIEW:")
        print(f"  Portfolio Value:        ${metrics.total_portfolio_value:>12,.2f}")
        print(f"  Cash Balance:           ${metrics.cash_balance:>12,.2f} ({metrics.cash_percent:>6.1%})")
        print(f"  Total Exposure:         {metrics.total_exposure_percent:>13.1%}")
        print(f"  Open Positions:         {metrics.num_positions:>13}")
        
        print("\nEXPOSURE LIMITS:")
        print(f"  Max Stock Exposure:     {metrics.stock_maximum_exposure:>13.1%} (limit: 25%)")
        print(f"  Max Sector Exposure:    {metrics.sector_maximum_exposure:>13.1%} (limit: 40%)")
        
        print("\nRISK METRICS:")
        print(f"  Drawdown:               {metrics.portfolio_drawdown:>13.2%}")
        print(f"  Risk Level:             {metrics.portfolio_risk_level.value:>13}")
        print(f"  Portfolio Heat Score:   {metrics.portfolio_heat_score:>13.1f}/100")
        print(f"  Daily P&L:              ${metrics.daily_loss:>12,.2f} ({metrics.daily_loss_percent:>6.2%})")
        
        if metrics.max_loss_breached:
            print(f"\n  [ALERT] CIRCUIT BREAKER TRIGGERED - Max loss exceeded!")
        
        print("="*80 + "\n")


# ============================================================================
# RISK LEVEL HEAT MAP VISUALIZATION
# ============================================================================

class RiskHeatMap:
    """Generate risk heat maps for portfolio visualization"""
    
    @staticmethod
    def generate_position_status(metrics: PortfolioRiskMetrics) -> str:
        """Generate ASCII heat map of current positions"""
        
        heat_level = metrics.portfolio_heat_score
        
        # Create heat bar (0-100)
        bar_length = 50
        filled = int((heat_level / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Color coding (represented by brackets)
        if heat_level < 25:
            color = "[LOW]"
        elif heat_level < 50:
            color = "[MOD]"
        elif heat_level < 75:
            color = "[HIGH]"
        else:
            color = "[CRIT]"
        
        return f"{color} {bar} {heat_level:.1f}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def categorize_volatility(volatility: float) -> str:
    """Categorize volatility level"""
    if volatility < 0.01:
        return "VERY_LOW"
    elif volatility < 0.02:
        return "LOW"
    elif volatility < 0.03:
        return "NORMAL"
    elif volatility < 0.05:
        return "HIGH"
    else:
        return "VERY_HIGH"


def calculate_position_correlation(
    price_series_1: List[float],
    price_series_2: List[float]
) -> float:
    """Calculate correlation between two price series"""
    returns_1 = [price_series_1[i] / price_series_1[i-1] - 1 for i in range(1, len(price_series_1))]
    returns_2 = [price_series_2[i] / price_series_2[i-1] - 1 for i in range(1, len(price_series_2))]
    
    if len(returns_1) < 2:
        return 0.0
    
    correlation = np.corrcoef(returns_1, returns_2)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def estimate_position_survival_probability(
    current_price: float,
    entry_price: float,
    stop_loss_price: float,
    target_price: float,
    daily_volatility: float,
    days_to_evaluate: int = 5
) -> float:
    """
    Estimate probability that position survives to target without hitting SL
    (simplified model using random walk)
    
    Returns:
        Probability 0.0-1.0
    """
    
    distance_to_sl = abs(current_price - stop_loss_price)
    distance_to_tp = abs(target_price - current_price)
    
    # Expected moves per day
    expected_daily_move = current_price * daily_volatility
    
    # Simple approximation: chance to reach target before SL
    if distance_to_sl == 0:
        return 0.0
    if distance_to_tp == 0:
        return 1.0
    
    probability = distance_to_tp / (distance_to_tp + distance_to_sl)
    
    return min(1.0, probability)
