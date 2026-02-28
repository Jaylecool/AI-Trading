"""
Task 4.3: Risk Management Testing - Volatile Market Scenarios
Tests all risk management features under challenging market conditions

Test Scenarios:
1. Extreme volatility spike
2. Rapid drawdown
3. Complete sector crash
4. Correlation spike between positions
5. Long tail event (flash crash)
6. Volatility clustering
7. Mean reversion trap
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sys

from trading_rules import TradingParameters, Position
from risk_management_enhanced import (
    EnhancedStopLoss, TrailingStopLoss, DynamicTakeProfitCalculator,
    PortfolioDiversificationManager, DynamicPositionSizer, EnhancedRiskMonitor,
    RiskHeatMap, categorize_volatility, estimate_position_survival_probability
)


# ============================================================================
# VOLATILE MARKET SCENARIO GENERATOR
# ============================================================================

class VolatileMarketScenario:
    """Base class for market scenarios"""
    
    def __init__(self, name: str, initial_price: float = 100.0, days: int = 20):
        self.name = name
        self.initial_price = initial_price
        self.days = days
        self.prices = []
        self.volatility = []
        self.rsi = []
        self.timestamps = []
        
    def generate_prices(self) -> List[float]:
        """Override in subclasses to generate scenario prices"""
        raise NotImplementedError
    
    def generate_volatility(self) -> List[float]:
        """Calculate rolling volatility"""
        returns = [self.prices[i] / self.prices[i-1] - 1 for i in range(1, len(self.prices))]
        window = 5
        vol = [np.std(returns[max(0, i-window):i+1]) for i in range(len(returns))]
        return [vol[0]] + vol  # Pad first value
    
    def generate_rsi(self, period: int = 14) -> List[float]:
        """Calculate RSI"""
        deltas = [self.prices[i] - self.prices[i-1] for i in range(1, len(self.prices))]
        
        rsi_values = []
        for i in range(len(deltas)):
            if i < period:
                rsi_values.append(50)  # Default during warm-up period
            else:
                gains = [d for d in deltas[i-period:i] if d > 0]
                losses = [abs(d) for d in deltas[i-period:i] if d < 0]
                
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                
                if avg_loss == 0:
                    rsi = 100 if avg_gain > 0 else 50
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                rsi_values.append(rsi)
        
        return [50] + rsi_values


class ExtremeVolatilityScenario(VolatileMarketScenario):
    """Sudden spike in volatility - 5% moves"""
    
    def generate_prices(self) -> List[float]:
        np.random.seed(42)
        prices = [self.initial_price]
        
        for day in range(self.days):
            if day < 5:
                # Normal volatility (1%)
                change = np.random.normal(0, 0.01)
            elif day < 15:
                # Extreme volatility (5%)
                change = np.random.normal(0, 0.05)
            else:
                # Return to normal
                change = np.random.normal(0, 0.01)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices


class RapidDrawdownScenario(VolatileMarketScenario):
    """Market crash - 10% drop over 5 days"""
    
    def generate_prices(self) -> List[float]:
        prices = [self.initial_price]
        
        for day in range(self.days):
            if day < 5:
                # Crash: 2% per day = 10% total
                change = -0.02
            elif day < 10:
                # Stabilization
                change = np.random.normal(0.001, 0.02)
            else:
                # Recovery attempt
                change = np.random.normal(0.005, 0.015)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices


class SectorCrashScenario(VolatileMarketScenario):
    """Sector-specific crash - affects multiple positions"""
    
    def generate_prices(self) -> List[float]:
        np.random.seed(42)
        prices = [self.initial_price]
        
        for day in range(self.days):
            if day < 7:
                # Sector crash: -2% to -4% per day
                change = np.random.uniform(-0.04, -0.02)
            elif day < 12:
                # Volatility swings
                change = np.random.normal(0, 0.05)
            else:
                # Stabilization
                change = np.random.normal(-0.001, 0.01)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices


class FlashCrashScenario(VolatileMarketScenario):
    """Black swan event - sudden 20% drop and recovery"""
    
    def generate_prices(self) -> List[float]:
        np.random.seed(42)
        prices = [self.initial_price]
        
        for day in range(self.days):
            if day == 10:
                # Flash crash: -20% in one day
                change = -0.20
            elif day == 11:
                # Partial recovery: +15%
                change = 0.15
            elif day > 11:
                # Normal recovery
                change = np.random.normal(0.001, 0.02)
            else:
                # Normal market
                change = np.random.normal(0, 0.01)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices


class VolatilityClusteringScenario(VolatileMarketScenario):
    """High volatility persists for period (volatility clustering)"""
    
    def generate_prices(self) -> List[float]:
        np.random.seed(42)
        prices = [self.initial_price]
        
        volatility_state = "normal"  # normal, high, extreme
        
        for day in range(self.days):
            # Volatility clustering: states persist
            if day == 5:
                volatility_state = "high"
            elif day == 10:
                volatility_state = "extreme"
            elif day == 15:
                volatility_state = "high"
            elif day == 18:
                volatility_state = "normal"
            
            if volatility_state == "normal":
                change = np.random.normal(0, 0.01)
            elif volatility_state == "high":
                change = np.random.normal(0, 0.03)
            else:  # extreme
                change = np.random.normal(0, 0.08)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices


class MeanReversionTrapScenario(VolatileMarketScenario):
    """False recovery leading to deeper crash"""
    
    def generate_prices(self) -> List[float]:
        prices = [self.initial_price]
        
        for day in range(self.days):
            if day < 5:
                # Initial drop: -3%
                change = -0.03
            elif day < 8:
                # False recovery: +2%
                change = 0.02
            elif day < 15:
                # Deeper crash: -5%
                change = -0.05
            else:
                # Attempted recovery
                change = np.random.normal(0.01, 0.02)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices


# ============================================================================
# RISK MANAGEMENT TEST SUITE
# ============================================================================

class RiskManagementTester:
    """Test risk management features under various scenarios"""
    
    def __init__(self):
        self.params = TradingParameters()
        self.diversification_mgr = PortfolioDiversificationManager()
        self.position_sizer = DynamicPositionSizer(self.params)
        self.tp_calculator = DynamicTakeProfitCalculator(self.params)
        self.risk_monitor = EnhancedRiskMonitor(self.params)
        self.results = []
    
    def test_stop_loss_trigger(self, scenario: VolatileMarketScenario) -> Dict:
        """Test stop-loss functionality"""
        
        scenario.prices = scenario.generate_prices()
        scenario.volatility = scenario.generate_volatility()
        scenario.rsi = scenario.generate_rsi()
        
        # Entry at price[5]
        entry_price = scenario.prices[5]
        entry_idx = 5
        
        # Create stop-loss
        sl = EnhancedStopLoss(entry_price, self.params.stop_loss_percent)
        
        stop_loss_price = sl.stop_price
        triggered_day = None
        triggered_price = None
        
        results = {
            'scenario': scenario.name,
            'test': 'STOP_LOSS_TRIGGER',
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'stop_loss_percent': self.params.stop_loss_percent,
            'triggered': False,
            'triggered_day': None,
            'triggered_price': None,
            'max_profit': 0,
            'max_drawdown': 0,
            'price_path': []
        }
        
        max_profit = 0
        max_loss = 0
        
        for day in range(entry_idx, len(scenario.prices)):
            price = scenario.prices[day]
            
            # Track P&L
            pnl_percent = (price - entry_price) / entry_price
            max_profit = max(max_profit, pnl_percent)
            max_loss = min(max_loss, pnl_percent)
            
            # Track price path
            results['price_path'].append({
                'day': day,
                'price': price,
                'pnl_percent': pnl_percent,
                'distance_to_sl': (price - stop_loss_price) / stop_loss_price if stop_loss_price > 0 else 0
            })
            
            # Check SL trigger
            triggered, reason = sl.check_trigger(price)
            if triggered:
                results['triggered'] = True
                results['triggered_day'] = day - entry_idx
                results['triggered_price'] = price
                results['exit_pnl'] = pnl_percent
                break
        
        results['max_profit'] = max_profit
        results['max_drawdown'] = max_loss
        
        return results
    
    def test_trailing_stop(self, scenario: VolatileMarketScenario) -> Dict:
        """Test trailing stop-loss"""
        
        scenario.prices = scenario.generate_prices()
        
        entry_price = scenario.prices[5]
        entry_idx = 5
        trailing_percent = 0.03  # 3% trailing
        
        trailing_sl = TrailingStopLoss(entry_price, trailing_percent)
        
        results = {
            'scenario': scenario.name,
            'test': 'TRAILING_STOP',
            'entry_price': entry_price,
            'trailing_percent': trailing_percent,
            'max_price_reached': entry_price,
            'triggered': False,
            'triggered_day': None,
            'triggered_price': None,
            'max_profit': 0,
            'price_path': []
        }
        
        max_profit = 0
        
        for day in range(entry_idx, len(scenario.prices)):
            price = scenario.prices[day]
            
            pnl_percent = (price - entry_price) / entry_price
            max_profit = max(max_profit, pnl_percent)
            
            if price > results['max_price_reached']:
                results['max_price_reached'] = price
            
            triggered, new_stop = trailing_sl.update(price)
            
            results['price_path'].append({
                'day': day,
                'price': price,
                'trailing_stop': new_stop,
                'distance_to_trailing_sl': (price - new_stop) / new_stop if new_stop > 0 else 0
            })
            
            if triggered:
                results['triggered'] = True
                results['triggered_day'] = day - entry_idx
                results['triggered_price'] = price
                exit_pnl = (price - entry_price) / entry_price
                results['exit_pnl'] = exit_pnl
                break
        
        results['max_profit'] = max_profit
        
        return results
    
    def test_dynamic_position_sizing(self, scenario: VolatileMarketScenario) -> Dict:
        """Test position sizing adjusts with volatility and confidence"""
        
        scenario.prices = scenario.generate_prices()
        scenario.volatility = scenario.generate_volatility()
        
        portfolio_value = 100000
        
        results = {
            'scenario': scenario.name,
            'test': 'DYNAMIC_POSITION_SIZING',
            'portfolio_value': portfolio_value,
            'sizing_analysis': []
        }
        
        for day in range(5, min(len(scenario.prices), 15)):
            current_price = scenario.prices[day]
            volatility = scenario.volatility[day]
            confidence = 0.7 + (np.random.random() * 0.2)  # 0.7-0.9
            
            stop_loss_price = current_price * (1 - self.params.stop_loss_percent)
            
            shares, position_value = self.position_sizer.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                confidence=confidence,
                volatility=volatility
            )
            
            # Volatility-adjusted size
            adjusted_shares = self.position_sizer.calculate_volatility_adjusted_size(
                shares, volatility
            )
            
            results['sizing_analysis'].append({
                'day': day,
                'price': current_price,
                'volatility': volatility,
                'volatility_category': categorize_volatility(volatility),
                'confidence': confidence,
                'base_shares': shares,
                'adjusted_shares': adjusted_shares,
                'size_reduction_pct': (1 - adjusted_shares / shares) * 100 if shares > 0 else 0,
                'position_value': adjusted_shares * current_price,
                'position_pct': (adjusted_shares * current_price) / portfolio_value
            })
        
        return results
    
    def test_diversification_limits(self) -> Dict:
        """Test portfolio diversification constraints"""
        
        portfolio_value = 100000
        
        # Create test positions
        positions = [
            Position(symbol="AAPL", entry_date=datetime.now(), entry_price=150, shares=100,
                    stop_loss_price=147.75, take_profit_price=153.75),
            Position(symbol="MSFT", entry_date=datetime.now(), entry_price=330, shares=50,
                    stop_loss_price=324.55, take_profit_price=338.25),
            Position(symbol="GOOGL", entry_date=datetime.now(), entry_price=140, shares=60,
                    stop_loss_price=137.90, take_profit_price=143.50),
        ]
        
        results = {
            'scenario': 'DIVERSIFICATION_TEST',
            'test': 'DIVERSIFICATION_LIMITS',
            'portfolio_value': portfolio_value,
            'existing_positions': [],
            'new_position_tests': []
        }
        
        # Analyze existing positions
        for pos in positions:
            exposure = (pos.entry_price * pos.shares) / portfolio_value
            sector = self.diversification_mgr.get_sector(pos.symbol)
            results['existing_positions'].append({
                'symbol': pos.symbol,
                'sector': sector,
                'exposure': exposure,
                'position_value': pos.entry_price * pos.shares
            })
        
        # Test new position additions
        test_symbols = ["TSLA", "JPM", "JNJ", "META"]
        for symbol in test_symbols:
            new_position_value = 15000
            
            stock_allowed, stock_reason = self.diversification_mgr.check_single_stock_limit(
                symbol, new_position_value, portfolio_value, positions
            )
            
            sector_allowed, sector_reason = self.diversification_mgr.check_sector_limit(
                symbol, new_position_value, portfolio_value, positions
            )
            
            results['new_position_tests'].append({
                'symbol': symbol,
                'proposed_value': new_position_value,
                'proposed_pct': new_position_value / portfolio_value,
                'stock_limit_check': {
                    'allowed': stock_allowed,
                    'reason': stock_reason
                },
                'sector_limit_check': {
                    'allowed': sector_allowed,
                    'reason': sector_reason
                },
                'approved': stock_allowed and sector_allowed
            })
        
        return results
    
    def test_portfolio_risk_monitoring(self, scenario: VolatileMarketScenario) -> Dict:
        """Test real-time portfolio risk monitoring"""
        
        scenario.prices = scenario.generate_prices()
        scenario.volatility = scenario.generate_volatility()
        
        # Create sample positions
        positions = [
            Position(symbol="AAPL", entry_date=datetime.now(), entry_price=scenario.prices[0],
                    shares=50, stop_loss_price=scenario.prices[0] * 0.985,
                    take_profit_price=scenario.prices[0] * 1.025),
            Position(symbol="MSFT", entry_date=datetime.now(), entry_price=scenario.prices[0] * 1.1,
                    shares=40, stop_loss_price=scenario.prices[0] * 1.085,
                    take_profit_price=scenario.prices[0] * 1.135),
        ]
        
        peak_value = 100000
        initial_capital = 100000
        cash = 75000
        
        results = {
            'scenario': scenario.name,
            'test': 'PORTFOLIO_RISK_MONITORING',
            'monitoring_history': []
        }
        
        for day in range(0, min(len(scenario.prices), 20)):
            # Update position prices (all move together in this scenario)
            price_change_factor = scenario.prices[day] / scenario.prices[0]
            
            current_prices = {}
            for pos in positions:
                current_prices[pos.symbol] = pos.entry_price * price_change_factor
            
            # Calculate portfolio value
            position_value = sum(pos.shares * current_prices[pos.symbol] for pos in positions)
            portfolio_value = cash + position_value
            
            # Calculate metrics
            metrics = self.risk_monitor.calculate_portfolio_metrics(
                positions=positions,
                cash=cash,
                current_prices=current_prices,
                peak_portfolio_value=max(peak_value, portfolio_value),
                initial_capital=initial_capital
            )
            
            results['monitoring_history'].append({
                'day': day,
                'portfolio_value': metrics.total_portfolio_value,
                'drawdown': metrics.portfolio_drawdown,
                'risk_level': metrics.portfolio_risk_level.value,
                'heat_score': metrics.portfolio_heat_score,
                'cash_percent': metrics.cash_percent,
                'max_loss_breached': metrics.max_loss_breached,
                'volatility': scenario.volatility[day]
            })
        
        return results
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test scenarios"""
        
        scenarios = [
            ExtremeVolatilityScenario("Extreme Volatility Spike", days=20),
            RapidDrawdownScenario("Rapid Market Drawdown", days=20),
            SectorCrashScenario("Sector Crash", days=20),
            FlashCrashScenario("Flash Crash Event", days=25),
            VolatilityClusteringScenario("Volatility Clustering", days=20),
            MeanReversionTrapScenario("Mean Reversion Trap", days=20),
        ]
        
        all_results = []
        
        print("\n" + "="*80)
        print("TASK 4.3: RISK MANAGEMENT TEST SUITE")
        print("Testing Risk Management Under Volatile Market Conditions")
        print("="*80 + "\n")
        
        for scenario in scenarios:
            print(f"Testing: {scenario.name}")
            print("-" * 80)
            
            # Test 1: Stop-loss
            sl_result = self.test_stop_loss_trigger(scenario)
            all_results.append(sl_result)
            print(f"  Stop-Loss: {'TRIGGERED' if sl_result['triggered'] else 'NOT TRIGGERED'}", end="")
            if sl_result['triggered']:
                print(f" at day {sl_result['triggered_day']}, P&L: {sl_result.get('exit_pnl', 0):.2%}")
            else:
                print(f" (Max profit: {sl_result['max_profit']:.2%})")
            
            # Test 2: Trailing stop
            ts_result = self.test_trailing_stop(scenario)
            all_results.append(ts_result)
            print(f"  Trailing Stop: {'TRIGGERED' if ts_result['triggered'] else 'STILL ACTIVE'}", end="")
            if ts_result['triggered']:
                print(f" at day {ts_result['triggered_day']}, P&L: {ts_result.get('exit_pnl', 0):.2%}")
            else:
                print(f" (Max profit: {ts_result['max_profit']:.2%})")
            
            # Test 3: Position sizing
            ps_result = self.test_dynamic_position_sizing(scenario)
            all_results.append(ps_result)
            sizing_data = ps_result['sizing_analysis']
            if sizing_data:
                avg_reduction = np.mean([s['size_reduction_pct'] for s in sizing_data])
                print(f"  Position Sizing: Avg volatility reduction: {avg_reduction:.1f}%")
            
            print()
        
        # Test 4: Diversification (single test)
        div_result = self.test_diversification_limits()
        all_results.append(div_result)
        print("DIVERSIFICATION TEST:")
        print("-" * 80)
        approved = sum(1 for t in div_result['new_position_tests'] if t['approved'])
        print(f"  Positions approved: {approved}/{len(div_result['new_position_tests'])}")
        print()
        
        # Test 5: Portfolio monitoring (sample scenario)
        monitor_result = self.test_portfolio_risk_monitoring(scenarios[0])
        all_results.append(monitor_result)
        print("PORTFOLIO RISK MONITORING:")
        print("-" * 80)
        if monitor_result['monitoring_history']:
            final_heat = monitor_result['monitoring_history'][-1]['heat_score']
            print(f"  Final heat score: {final_heat:.1f}/100")
        print()
        
        return all_results


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    """Run comprehensive risk management tests"""
    
    tester = RiskManagementTester()
    results = tester.run_all_tests()
    
    # Save results
    output_file = 'risk_management_test_results.json'
    
    # Convert datetime objects for JSON serialization
    def serialize_result(result):
        if isinstance(result, dict):
            return {k: serialize_result(v) for k, v in result.items()}
        elif isinstance(result, list):
            return [serialize_result(item) for item in result]
        elif isinstance(result, datetime):
            return result.isoformat()
        else:
            return result
    
    serializable_results = serialize_result(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("TEST EXECUTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print(f"Total tests run: {len(results)}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
