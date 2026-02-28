"""
Task 4.4: Backtest Runner
Executes backtests across multiple strategies on historical data,
generates performance results, and compares strategies.

This script:
1. Loads historical data
2. Runs backtest for each strategy
3. Compares results
4. Identifies strengths and weaknesses
5. Generates comprehensive report
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import numpy as np

from backtesting_engine import BacktestingEngine, BacktestResults
from strategy_configurations import STRATEGIES, print_strategy_comparison, print_strategy_parameters


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

class BacktestRunner:
    """Runs backtests and manages multiple strategy evaluations"""
    
    def __init__(self, data_filepath: str = 'AAPL_stock_data_with_indicators.csv'):
        """
        Initialize backtest runner
        
        Args:
            data_filepath: Path to historical data CSV
        """
        self.data_filepath = data_filepath
        self.data = None
        self.results: Dict[str, BacktestResults] = {}
        self.execution_timestamp = datetime.now()
    
    def load_data(self) -> None:
        """Load and validate historical data"""
        print(f"\n[LOAD] Loading historical data from: {self.data_filepath}")
        engine = BacktestingEngine()
        self.data = engine.load_data(self.data_filepath)
        print(f"[LOAD] Loaded {len(self.data)} days of data")
        print(f"[LOAD] Date range: {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
    
    def run_all_strategies(self, initial_capital: float = 100000.0) -> None:
        """
        Run backtest for all available strategies
        
        Args:
            initial_capital: Starting portfolio value for each strategy
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTESTS FOR ALL STRATEGIES")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"{'='*80}\n")
        
        for strategy_key, strategy_info in STRATEGIES.items():
            print(f"\n[STRATEGY] Running {strategy_info['name']} ({strategy_key})")
            print(f"[STRATEGY] Risk Level: {strategy_info['risk_level']}")
            
            # Get strategy parameters
            strategy_params = strategy_info['factory']()
            
            # Create and run engine
            engine = BacktestingEngine(
                initial_capital=initial_capital,
                trading_params=strategy_params,
                use_risk_management=True
            )
            
            # Run backtest
            results = engine.run_backtest(
                self.data,
                strategy_name=strategy_info['name']
            )
            
            self.results[strategy_key] = results
            
            # Print summary
            print(results)
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all strategy results
        
        Returns:
            DataFrame with comparison metrics
        """
        print(f"\n{'='*100}")
        print(f"STRATEGY COMPARISON")
        print(f"{'='*100}\n")
        
        comparison_data = []
        
        for strategy_key, results in self.results.items():
            comparison_data.append({
                'Strategy': strategy_key,
                'ROI': results.roi,
                'Sharpe Ratio': results.sharpe_ratio,
                'Max Drawdown': results.max_drawdown,
                'Win Rate': results.win_rate,
                'Profit Factor': results.profit_factor,
                'Total Trades': results.total_trades,
                'Winning Trades': results.winning_trades,
                'Avg Win %': results.avg_win,
                'Avg Loss %': results.avg_loss,
                'Annual Volatility': results.annual_volatility,
                'Calmar Ratio': results.calmar_ratio,
                'Sortino Ratio': results.sortino_ratio,
                'Final Capital': results.final_capital,
                'Avg Hold Days': results.avg_hold_days
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Print formatted comparison
        print(df.to_string(index=False))
        print()
        
        return df
    
    def identify_best_strategy(self) -> Dict[str, str]:
        """
        Identify best strategy for different metrics
        
        Returns:
            Dictionary mapping metrics to best strategy
        """
        print(f"\n{'='*100}")
        print(f"BEST PERFORMING STRATEGIES BY METRIC")
        print(f"{'='*100}\n")
        
        if not self.results:
            print("No results to compare")
            return {}
        
        metrics_to_evaluate = [
            ('ROI', 'roi', True),  # (display_name, field_name, is_higher_better)
            ('Sharpe Ratio', 'sharpe_ratio', True),
            ('Max Drawdown', 'max_drawdown', False),  # Lower is better
            ('Win Rate', 'win_rate', True),
            ('Profit Factor', 'profit_factor', True),
            ('Calmar Ratio', 'calmar_ratio', True),
            ('Sortino Ratio', 'sortino_ratio', True),
            ('Annual Volatility', 'annual_volatility', False),  # Lower is better
        ]
        
        best_strategies = {}
        
        for display_name, field_name, higher_is_better in metrics_to_evaluate:
            values = {k: getattr(v, field_name) for k, v in self.results.items()}
            
            if higher_is_better:
                best_key = max(values, key=values.get)
                best_value = values[best_key]
            else:
                best_key = min(values, key=values.get)
                best_value = values[best_key]
            
            best_strategies[display_name] = best_key
            
            print(f"  {display_name:.<30} {best_key:>15} ({best_value:>10.4f})")
            for strategy_key, value in values.items():
                if strategy_key != best_key:
                    print(f"    {strategy_key:.<28} {value:>15.4f}")
            print()
        
        return best_strategies
    
    def analyze_strategy_strengths_weaknesses(self) -> Dict[str, Dict[str, str]]:
        """
        Analyze strengths and weaknesses of each strategy
        
        Returns:
            Dictionary with analysis for each strategy
        """
        print(f"\n{'='*100}")
        print(f"STRATEGY ANALYSIS: STRENGTHS AND WEAKNESSES")
        print(f"{'='*100}\n")
        
        analysis = {}
        
        for strategy_key, results in self.results.items():
            strategy_info = STRATEGIES.get(strategy_key, {})
            
            analysis[strategy_key] = {
                'strengths': [],
                'weaknesses': [],
                'improvements': []
            }
            
            print(f"\n{strategy_key} STRATEGY: {strategy_info.get('name', 'Unknown')}")
            print("-" * 80)
            
            # Analyze ROI
            if results.roi > 0.10:
                analysis[strategy_key]['strengths'].append(f"Strong ROI: {results.roi:.2%}")
            elif results.roi < -0.05:
                analysis[strategy_key]['weaknesses'].append(f"Negative ROI: {results.roi:.2%}")
            
            # Analyze Sharpe Ratio
            if results.sharpe_ratio > 1.0:
                analysis[strategy_key]['strengths'].append(f"Excellent Sharpe Ratio: {results.sharpe_ratio:.4f}")
            elif results.sharpe_ratio < 0.5:
                analysis[strategy_key]['weaknesses'].append(f"Poor risk-adjusted returns: {results.sharpe_ratio:.4f}")
            
            # Analyze Max Drawdown
            if results.max_drawdown < 0.05:
                analysis[strategy_key]['strengths'].append(f"Controlled drawdown: {results.max_drawdown:.2%}")
            elif results.max_drawdown > 0.15:
                analysis[strategy_key]['weaknesses'].append(f"High drawdown: {results.max_drawdown:.2%}")
            
            # Analyze Win Rate
            if results.win_rate > 0.60:
                analysis[strategy_key]['strengths'].append(f"High win rate: {results.win_rate:.2%}")
            elif results.win_rate < 0.40:
                analysis[strategy_key]['weaknesses'].append(f"Low win rate: {results.win_rate:.2%}")
            
            # Analyze Trade Frequency
            trading_days = (results.end_date - results.start_date).days
            trades_per_year = (results.total_trades / trading_days * 365) if trading_days > 0 else 0
            
            if trades_per_year > 50:
                analysis[strategy_key]['strengths'].append(f"High trade frequency: {trades_per_year:.0f} trades/year")
                analysis[strategy_key]['weaknesses'].append("High transaction costs from frequent trading")
            elif trades_per_year < 5:
                analysis[strategy_key]['weaknesses'].append(f"Low trade frequency: {trades_per_year:.0f} trades/year")
            
            # Analyze Profit Factor
            if results.profit_factor > 2.0:
                analysis[strategy_key]['strengths'].append(f"Strong profit factor: {results.profit_factor:.2f}")
            elif results.profit_factor < 1.0:
                analysis[strategy_key]['weaknesses'].append(f"Profit factor < 1.0: {results.profit_factor:.2f}")
            
            # Suggestions for improvement
            if results.max_drawdown > 0.10:
                analysis[strategy_key]['improvements'].append(
                    "Reduce position sizes or tighten stop-loss levels to decrease drawdown"
                )
            
            if results.win_rate < 0.50:
                analysis[strategy_key]['improvements'].append(
                    "Improve entry signal quality or raise confidence threshold"
                )
            
            if results.total_trades < 5:
                analysis[strategy_key]['improvements'].append(
                    "Increase trade frequency by loosening entry thresholds"
                )
            
            # Print analysis
            if analysis[strategy_key]['strengths']:
                print(f"\n  STRENGTHS:")
                for strength in analysis[strategy_key]['strengths']:
                    print(f"    ✓ {strength}")
            
            if analysis[strategy_key]['weaknesses']:
                print(f"\n  WEAKNESSES:")
                for weakness in analysis[strategy_key]['weaknesses']:
                    print(f"    ✗ {weakness}")
            
            if analysis[strategy_key]['improvements']:
                print(f"\n  AREAS FOR IMPROVEMENT:")
                for improvement in analysis[strategy_key]['improvements']:
                    print(f"    → {improvement}")
        
        return analysis
    
    def save_results(self, output_filename: str = 'backtest_results.json') -> None:
        """
        Save all results to JSON file
        
        Args:
            output_filename: Output filename for JSON results
        """
        print(f"\n[SAVE] Saving results to {output_filename}")
        
        output_data = {
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'data_file': self.data_filepath,
            'data_summary': {
                'start_date': self.data['Date'].min().isoformat(),
                'end_date': self.data['Date'].max().isoformat(),
                'total_bars': len(self.data)
            },
            'strategies': {}
        }
        
        for strategy_key, results in self.results.items():
            output_data['strategies'][strategy_key] = results.to_dict()
        
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[SAVE] Results saved successfully")
    
    def save_comparison_csv(self, output_filename: str = 'strategy_comparison.csv') -> None:
        """
        Save strategy comparison to CSV
        
        Args:
            output_filename: Output filename for CSV
        """
        print(f"[SAVE] Saving comparison to {output_filename}")
        
        comparison_data = []
        
        for strategy_key, results in self.results.items():
            comparison_data.append({
                'Strategy': strategy_key,
                'Name': results.strategy_name,
                'ROI': f"{results.roi:.2%}",
                'Sharpe Ratio': f"{results.sharpe_ratio:.4f}",
                'Max Drawdown': f"{results.max_drawdown:.2%}",
                'Win Rate': f"{results.win_rate:.2%}",
                'Profit Factor': f"{results.profit_factor:.2f}",
                'Total Trades': results.total_trades,
                'Final Capital': f"${results.final_capital:,.2f}",
                'Annual Volatility': f"{results.annual_volatility:.2%}",
                'Calmar Ratio': f"{results.calmar_ratio:.4f}",
                'Sortino Ratio': f"{results.sortino_ratio:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_filename, index=False)
        print(f"[SAVE] Comparison saved successfully")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute comprehensive backtesting analysis"""
    
    # Show available strategies
    print_strategy_comparison()
    
    # Initialize runner
    runner = BacktestRunner(data_filepath='AAPL_stock_data_with_indicators.csv')
    
    # Load data
    runner.load_data()
    
    # Run all strategies
    runner.run_all_strategies(initial_capital=100000.0)
    
    # Compare results
    comparison_df = runner.compare_strategies()
    
    # Identify best strategies
    best_strategies = runner.identify_best_strategy()
    
    # Analyze strengths and weaknesses
    analysis = runner.analyze_strategy_strengths_weaknesses()
    
    # Save results
    runner.save_results('backtest_results.json')
    runner.save_comparison_csv('strategy_comparison.csv')
    
    # Print final summary
    print(f"\n{'='*100}")
    print(f"BACKTEST EXECUTION COMPLETED")
    print(f"{'='*100}")
    print(f"\nResults saved to:")
    print(f"  - backtest_results.json (detailed results)")
    print(f"  - strategy_comparison.csv (comparison table)")
    print(f"\nExecution completed at {runner.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
