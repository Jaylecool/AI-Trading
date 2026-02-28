"""
Task 4.4: Strategy Configurations
Defines multiple trading strategies with different risk/reward profiles:
- Aggressive: High risk, frequent trading, larger positions
- Conservative: Low risk, selective trading, smaller positions
- Balanced: Medium risk, regular trading, medium positions

Each strategy adjusts key trading parameters to achieve its objectives.
"""

from trading_rules import TradingParameters


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

class StrategyFactory:
    """Factory for creating different trading strategy configurations"""
    
    @staticmethod
    def create_aggressive_strategy() -> TradingParameters:
        """
        Aggressive Strategy: Maximum returns focus
        
        Characteristics:
        - Lower thresholds: More frequent entries
        - Larger position sizes: 3% risk per trade
        - Smaller stops: 1% stop-loss for quick exits
        - Smaller TP targets: 3% take-profit for frequent wins
        - Higher max concurrent positions: 5 positions
        - Lower confidence threshold: 0.4
        
        Use Case: Active traders with high risk tolerance
        """
        return TradingParameters(
            # Entry/Exit Thresholds - MORE AGGRESSIVE
            buy_threshold=0.01,  # 1% (vs default 2%)
            sell_threshold=0.01,  # 1% (vs default 2%)
            take_profit_target=0.03,  # 3% (vs default 2.5%)
            stop_loss_percent=0.01,  # 1% (vs default 1.5%)
            
            # Position Sizing - LARGER POSITIONS
            risk_percentage=0.03,  # 3% per trade (vs default 2%)
            max_position_value_percent=0.25,  # 25% per position (vs default 15%)
            max_cash_allocation_percent=0.30,  # Only keep 30% cash (vs default 50%)
            min_position_size=10,
            
            # Portfolio Risk - HIGHER TOLERANCE
            portfolio_max_loss_percent=-0.08,  # -8% circuit breaker (vs default -5%)
            max_concurrent_positions=5,  # More positions (vs default 3)
            
            # Position Management
            minimum_hold_days=0,  # No minimum hold (vs default 1)
            trailing_stop_percent=0.005,  # 0.5% trailing stop
            
            # Market Conditions - LESS SENSITIVE
            volatility_threshold=0.05,  # 5% (vs default 3%)
            volatility_threshold_multiplier=1.0,  # Same thresholds (vs default 1.5x)
            
            # Confidence Requirements - LOWER THRESHOLD
            confidence_threshold=0.40,  # 40% (vs default 50%)
        )
    
    @staticmethod
    def create_conservative_strategy() -> TradingParameters:
        """
        Conservative Strategy: Capital preservation focus
        
        Characteristics:
        - Higher thresholds: Fewer but higher-quality entries
        - Smaller position sizes: 1% risk per trade
        - Larger stops: 2% stop-loss for tolerance
        - Larger TP targets: 2% take-profit to lock wins
        - Lower max concurrent positions: 2 positions
        - Higher confidence threshold: 0.65
        
        Use Case: Risk-averse investors, retirees seeking steady returns
        """
        return TradingParameters(
            # Entry/Exit Thresholds - MORE SELECTIVE
            buy_threshold=0.03,  # 3% (vs default 2%)
            sell_threshold=0.03,  # 3% (vs default 2%)
            take_profit_target=0.02,  # 2% (vs default 2.5%)
            stop_loss_percent=0.02,  # 2% (vs default 1.5%)
            
            # Position Sizing - SMALLER POSITIONS
            risk_percentage=0.01,  # 1% per trade (vs default 2%)
            max_position_value_percent=0.08,  # 8% per position (vs default 15%)
            max_cash_allocation_percent=0.70,  # Keep 70% cash (vs default 50%)
            min_position_size=10,
            
            # Portfolio Risk - LOWER TOLERANCE
            portfolio_max_loss_percent=-0.02,  # -2% circuit breaker (vs default -5%)
            max_concurrent_positions=2,  # Fewer positions (vs default 3)
            
            # Position Management
            minimum_hold_days=3,  # Longer hold (vs default 1)
            trailing_stop_percent=0.01,  # 1% trailing stop
            
            # Market Conditions - MORE SENSITIVE
            volatility_threshold=0.02,  # 2% (vs default 3%)
            volatility_threshold_multiplier=2.0,  # Wider thresholds (vs default 1.5x)
            
            # Confidence Requirements - HIGHER THRESHOLD
            confidence_threshold=0.65,  # 65% (vs default 50%)
        )
    
    @staticmethod
    def create_balanced_strategy() -> TradingParameters:
        """
        Balanced Strategy: Moderate risk/reward balance
        
        Characteristics:
        - Default thresholds: Standard signal generation
        - Standard position sizes: 2% risk per trade (default)
        - Standard TP targets: 2.5% take-profit
        - Standard stops: 1.5% stop-loss
        - Moderate positions: 3 concurrent (default)
        - Standard confidence: 50% (default)
        
        Use Case: Most traders, balanced risk/reward seekers
        """
        return TradingParameters(
            # Entry/Exit Thresholds - DEFAULT
            buy_threshold=0.02,  # 2% (default)
            sell_threshold=0.02,  # 2% (default)
            take_profit_target=0.025,  # 2.5% (default)
            stop_loss_percent=0.015,  # 1.5% (default)
            
            # Position Sizing - STANDARD
            risk_percentage=0.02,  # 2% per trade (default)
            max_position_value_percent=0.15,  # 15% per position (default)
            max_cash_allocation_percent=0.50,  # Keep 50% cash (default)
            min_position_size=10,
            
            # Portfolio Risk - STANDARD
            portfolio_max_loss_percent=-0.05,  # -5% circuit breaker (default)
            max_concurrent_positions=3,  # Standard (default)
            
            # Position Management
            minimum_hold_days=1,  # Standard (default)
            trailing_stop_percent=0.0075,  # 0.75% trailing stop
            
            # Market Conditions - STANDARD
            volatility_threshold=0.03,  # 3% (default)
            volatility_threshold_multiplier=1.5,  # Standard (default)
            
            # Confidence Requirements - STANDARD
            confidence_threshold=0.50,  # 50% (default)
        )
    
    @staticmethod
    def create_custom_strategy(
        buy_threshold: float = 0.02,
        sell_threshold: float = 0.02,
        stop_loss_percent: float = 0.015,
        take_profit_target: float = 0.025,
        risk_percentage: float = 0.02,
        max_position_value_percent: float = 0.15,
        max_concurrent_positions: int = 3,
        confidence_threshold: float = 0.50,
        **kwargs
    ) -> TradingParameters:
        """
        Create custom trading strategy with specified parameters
        
        Args:
            buy_threshold: Buy signal threshold (as decimal, e.g., 0.02 for 2%)
            sell_threshold: Sell signal threshold
            stop_loss_percent: Stop-loss percentage
            take_profit_target: Take-profit target percentage
            risk_percentage: Risk per trade as % of portfolio
            max_position_value_percent: Max position size as % of portfolio
            max_concurrent_positions: Maximum concurrent open positions
            confidence_threshold: Minimum confidence for trade (0.0-1.0)
            **kwargs: Additional parameters for TradingParameters
        
        Returns:
            TradingParameters with custom settings
        """
        return TradingParameters(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            stop_loss_percent=stop_loss_percent,
            take_profit_target=take_profit_target,
            risk_percentage=risk_percentage,
            max_position_value_percent=max_position_value_percent,
            max_concurrent_positions=max_concurrent_positions,
            confidence_threshold=confidence_threshold,
            **kwargs
        )


# ============================================================================
# STRATEGY PROFILES FOR COMPARISON
# ============================================================================

STRATEGIES = {
    'AGGRESSIVE': {
        'name': 'Aggressive Strategy',
        'description': 'High-frequency trading, larger positions, tight stops.',
        'risk_level': 'HIGH',
        'factory': StrategyFactory.create_aggressive_strategy,
        'expected_characteristics': {
            'trade_frequency': 'Very High',
            'avg_position_size': 'Large (25% max)',
            'win_rate': 'Medium (45-55%)',
            'sharpe_ratio': 'Medium',
            'max_drawdown': 'High (5-10%)',
            'best_for': 'Active traders, high risk tolerance'
        }
    },
    'CONSERVATIVE': {
        'name': 'Conservative Strategy',
        'description': 'Low-frequency trading, small positions, wide stops.',
        'risk_level': 'LOW',
        'factory': StrategyFactory.create_conservative_strategy,
        'expected_characteristics': {
            'trade_frequency': 'Low',
            'avg_position_size': 'Small (8% max)',
            'win_rate': 'High (55-70%)',
            'sharpe_ratio': 'High',
            'max_drawdown': 'Low (1-3%)',
            'best_for': 'Risk-averse, capital preservation'
        }
    },
    'BALANCED': {
        'name': 'Balanced Strategy',
        'description': 'Moderate trading, standard positions, balanced risk.',
        'risk_level': 'MEDIUM',
        'factory': StrategyFactory.create_balanced_strategy,
        'expected_characteristics': {
            'trade_frequency': 'Medium',
            'avg_position_size': 'Medium (15% max)',
            'win_rate': 'Medium-High (50-60%)',
            'sharpe_ratio': 'Balanced',
            'max_drawdown': 'Moderate (3-5%)',
            'best_for': 'Most traders, balanced approach'
        }
    }
}


def print_strategy_comparison() -> None:
    """Print comparison of all available strategies"""
    print("\n" + "="*100)
    print("AVAILABLE TRADING STRATEGIES")
    print("="*100 + "\n")
    
    for strategy_key, strategy_info in STRATEGIES.items():
        print(f"STRATEGY: {strategy_info['name'].upper()}")
        print(f"Type: {strategy_key}")
        print(f"Risk Level: {strategy_info['risk_level']}")
        print(f"Description: {strategy_info['description']}")
        print(f"\nExpected Characteristics:")
        for char, value in strategy_info['expected_characteristics'].items():
            print(f"  - {char:.<30} {value}")
        print("\n" + "-"*100 + "\n")


def print_strategy_parameters(strategy_name: str) -> None:
    """Print detailed parameters for a specific strategy"""
    if strategy_name not in STRATEGIES:
        print(f"Unknown strategy: {strategy_name}")
        return
    
    params = STRATEGIES[strategy_name]['factory']()
    print(f"\n{STRATEGIES[strategy_name]['name'].upper()} PARAMETERS")
    print(params)
