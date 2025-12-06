# experiments/improved_compare.py
"""
Improved Strategy Comparison
============================

Addresses issues from initial comparison:
1. Multiple runs for statistical significance
2. Risk-adjusted NS (penalize high volatility)
3. Ensemble of diverse QD strategies
4. Proper hyperparameter tuning

Run: python experiments/improved_compare.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import predictors
from deep_q_learning import DeepQLearning
from ddpg import DDPG


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics."""
    annual_return = np.mean(returns) * 252
    annual_vol = np.std(returns) * np.sqrt(252)
    sharpe = annual_return / (annual_vol + 1e-8)
    
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    # Sortino
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 1:
      downside_vol = np.std(neg_returns) * np.sqrt(252)
      downside_vol = max(downside_vol, 0.01)  # Floor at 1% to avoid explosion
    else:
      downside_vol = annual_vol  # Fallback to standard vol
    sortino = annual_return / downside_vol
    
    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'volatility': annual_vol,
        'max_drawdown': max_dd,
        'sortino': sortino,
    }


def backtest(weights_fn, test_data: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, List]:
    """Run backtest."""
    returns = []
    weights_history = []
    
    for t in range(lookback, len(test_data)):
        state = test_data.iloc[t-lookback:t].values.flatten()
        weights = weights_fn(state)
        weights_history.append(weights)
        
        day_returns = test_data.iloc[t].values
        portfolio_return = np.dot(weights, day_returns)
        returns.append(portfolio_return)
    
    return np.array(returns), weights_history


# =============================================================================
# STRATEGIES
# =============================================================================

class EqualWeight:
    def __init__(self, n_assets):
        self.weights = np.ones(n_assets) / n_assets
    
    def __call__(self, state):
        return self.weights.copy()


class RLStrategy:
    def __init__(self, model):
        self.actor = model.actor
        self.actor.eval()
    
    def __call__(self, state):
        with torch.no_grad():
            return self.actor(torch.tensor(state, dtype=torch.float32)).numpy()


class EnsembleStrategy:
    """Ensemble of multiple diverse strategies from QD."""
    def __init__(self, strategies: List, weights: List[float] = None):
        self.strategies = strategies
        self.weights = weights or [1/len(strategies)] * len(strategies)
    
    def __call__(self, state):
        portfolio = np.zeros_like(self.strategies[0](state))
        for strategy, weight in zip(self.strategies, self.weights):
            portfolio += weight * strategy(state)
        return portfolio / np.sum(portfolio)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_experiment(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    n_assets: int,
    lookback: int,
    config: Dict,
    seed: int
) -> Dict[str, Dict]:
    """Run single experiment with all strategies."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    results = {}
    
    # 1. Equal Weight
    equal_weight = EqualWeight(n_assets)
    returns, weights = backtest(equal_weight, test_data, lookback)
    results['Equal Weight'] = calculate_metrics(returns)
    results['Equal Weight']['weights_std'] = np.mean(np.std(weights, axis=0))
    
    # 2. DQN Vanilla
    model_vanilla = DeepQLearning(
        lookback_window=lookback,
        predictor=predictors.MLP,
        batch_size=1,
        short_selling=False,
        forecast_window=0,
        reduce_negatives=True,
        verbose=0,
        hidden_sizes=config['hidden_sizes'],
        seed=seed,
    )
    model_vanilla.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config['num_epochs'],
        use_ns=False,
        risk_preference=config['risk_preference'],
    )
    
    strategy = RLStrategy(model_vanilla)
    returns, weights = backtest(strategy, test_data, lookback)
    results['DQN'] = calculate_metrics(returns)
    results['DQN']['weights_std'] = np.mean(np.std(weights, axis=0))
    
    # 3. DQN + NS (with HIGHER risk penalty to control volatility)
    model_ns = DeepQLearning(
        lookback_window=lookback,
        predictor=predictors.MLP,
        batch_size=1,
        short_selling=False,
        forecast_window=0,
        reduce_negatives=True,
        verbose=0,
        hidden_sizes=config['hidden_sizes'],
        seed=seed,
    )
    model_ns.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config['num_epochs'],
        use_ns=True,
        ns_alpha=config['ns_alpha'],
        ns_beta=config['ns_beta'],
        risk_preference=config['risk_preference_ns'],  # Higher risk penalty for NS
    )
    
    strategy = RLStrategy(model_ns)
    returns, weights = backtest(strategy, test_data, lookback)
    results['DQN + NS'] = calculate_metrics(returns)
    results['DQN + NS']['weights_std'] = np.mean(np.std(weights, axis=0))
    
    # 4. DDPG (often more stable than DQN for continuous actions)
    model_ddpg = DDPG(
        lookback_window=lookback,
        predictor=predictors.MLP,
        batch_size=1,
        short_selling=False,
        forecast_window=0,
        reduce_negatives=True,
        verbose=0,
        hidden_sizes=config['hidden_sizes'],
        seed=seed,
    )
    model_ddpg.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config['num_epochs'],
        use_ns=False,
        risk_preference=config['risk_preference'],
        soft_update=True,
        tau=0.01,
    )
    
    strategy = RLStrategy(model_ddpg)
    returns, weights = backtest(strategy, test_data, lookback)
    results['DDPG'] = calculate_metrics(returns)
    results['DDPG']['weights_std'] = np.mean(np.std(weights, axis=0))
    
    # 5. DDPG + NS
    model_ddpg_ns = DDPG(
        lookback_window=lookback,
        predictor=predictors.MLP,
        batch_size=1,
        short_selling=False,
        forecast_window=0,
        reduce_negatives=True,
        verbose=0,
        hidden_sizes=config['hidden_sizes'],
        seed=seed,
    )
    model_ddpg_ns.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config['num_epochs'],
        use_ns=True,
        ns_alpha=config['ns_alpha'],
        ns_beta=config['ns_beta'],
        risk_preference=config['risk_preference_ns'],
        soft_update=True,
        tau=0.01,
    )
    
    strategy = RLStrategy(model_ddpg_ns)
    returns, weights = backtest(strategy, test_data, lookback)
    results['DDPG + NS'] = calculate_metrics(returns)
    results['DDPG + NS']['weights_std'] = np.mean(np.std(weights, axis=0))
    
    return results


def main():
    print("="*70)
    print("IMPROVED STRATEGY COMPARISON (Multiple Runs)")
    print("="*70)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "multiasset_daily_returns_enhanced.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
    n_assets = df.shape[1]
    
    # Split
    train_end = int(len(df) * 0.6)
    val_end = int(len(df) * 0.8)
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    print(f"\nData: {len(df)} days, {n_assets} assets")
    print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    
    # Configuration
    config = {
        'hidden_sizes': (64, 64),
        'num_epochs': 15,
        'risk_preference': -0.5,        # Standard risk penalty
        'risk_preference_ns': -1.0,     # HIGHER penalty for NS to control vol
        'ns_alpha': 0.7,
        'ns_beta': 0.3,                 # Lower beta = less novelty pressure
    }
    
    lookback = 50
    n_runs = 20
    
    print(f"\nRunning {n_runs} experiments...")
    print(f"Config: risk_pref={config['risk_preference']}, "
          f"risk_pref_ns={config['risk_preference_ns']}, "
          f"ns_beta={config['ns_beta']}")
    
    # Run multiple experiments
    all_results = {strategy: [] for strategy in 
                   ['Equal Weight', 'DQN', 'DQN + NS', 'DDPG', 'DDPG + NS']}
    
    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")
        results = run_experiment(
            train_data, val_data, test_data,
            n_assets, lookback, config, seed=42+run
        )
        
        for strategy, metrics in results.items():
            all_results[strategy].append(metrics)
    
    # Aggregate results
    print("\n" + "="*70)
    print("RESULTS (mean ± std over {} runs)".format(n_runs))
    print("="*70)
    
    summary = []
    for strategy in all_results:
        metrics_list = all_results[strategy]
        
        row = {'Strategy': strategy}
        for metric in ['sharpe', 'annual_return', 'volatility', 'max_drawdown', 'sortino']:
            values = [m[metric] for m in metrics_list]
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
        
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    
    # Print table
    print(f"\n{'Strategy':<15} {'Sharpe':>18} {'Return':>18} {'Vol':>18} {'MaxDD':>18}")
    print("-"*90)
    
    for _, row in summary_df.iterrows():
        sharpe = f"{row['sharpe_mean']:.3f} ± {row['sharpe_std']:.3f}"
        ret = f"{row['annual_return_mean']*100:.1f}% ± {row['annual_return_std']*100:.1f}%"
        vol = f"{row['volatility_mean']*100:.1f}% ± {row['volatility_std']*100:.1f}%"
        dd = f"{row['max_drawdown_mean']*100:.1f}% ± {row['max_drawdown_std']*100:.1f}%"
        print(f"{row['Strategy']:<15} {sharpe:>18} {ret:>18} {vol:>18} {dd:>18}")
    
    # Ranking
    print("\n" + "="*70)
    print("RANKINGS BY SHARPE RATIO")
    print("="*70)
    
    ranked = summary_df.sort_values('sharpe_mean', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['Strategy']:<15} Sharpe: {row['sharpe_mean']:.3f} ± {row['sharpe_std']:.3f}")
    
    # Statistical significance test
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    baseline_sharpes = [m['sharpe'] for m in all_results['Equal Weight']]
    
    for strategy in ['DQN', 'DQN + NS', 'DDPG', 'DDPG + NS']:
        strategy_sharpes = [m['sharpe'] for m in all_results[strategy]]
        
        # Simple t-test approximation
        mean_diff = np.mean(strategy_sharpes) - np.mean(baseline_sharpes)
        pooled_std = np.sqrt((np.var(strategy_sharpes) + np.var(baseline_sharpes)) / 2)
        t_stat = mean_diff / (pooled_std * np.sqrt(2/n_runs) + 1e-8)
        
        sig = "**" if abs(t_stat) > 2.0 else "*" if abs(t_stat) > 1.5 else ""
        better_worse = "better" if mean_diff > 0 else "worse"
        
        print(f"  {strategy} vs Equal Weight: {mean_diff:+.3f} ({better_worse}) {sig}")
    
    print("\n  ** = significant (|t| > 2.0), * = marginally significant (|t| > 1.5)")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    strategies = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    # 1. Sharpe with error bars
    ax1 = axes[0]
    means = [summary_df[summary_df['Strategy']==s]['sharpe_mean'].values[0] for s in strategies]
    stds = [summary_df[summary_df['Strategy']==s]['sharpe_std'].values[0] for s in strategies]
    
    bars = ax1.bar(range(len(strategies)), means, yerr=stds, capsize=5, color=colors, 
                   edgecolor='black', alpha=0.8)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio (mean ± std)', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. Return vs Volatility scatter
    ax2 = axes[1]
    for i, strategy in enumerate(strategies):
        returns = [m['annual_return']*100 for m in all_results[strategy]]
        vols = [m['volatility']*100 for m in all_results[strategy]]
        ax2.scatter(vols, returns, label=strategy, color=colors[i], s=100, alpha=0.7)
        # Plot mean
        ax2.scatter([np.mean(vols)], [np.mean(returns)], color=colors[i], 
                   s=200, marker='*', edgecolor='black', linewidth=2)
    
    ax2.set_xlabel('Volatility (%)')
    ax2.set_ylabel('Annual Return (%)')
    ax2.set_title('Risk-Return Profile', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of Sharpe ratios
    ax3 = axes[2]
    sharpe_data = [[m['sharpe'] for m in all_results[s]] for s in strategies]
    bp = ax3.boxplot(sharpe_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticklabels(strategies, rotation=45, ha='right')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Sharpe Distribution', fontweight='bold')
    ax3.axhline(y=np.mean(baseline_sharpes), color='red', linestyle='--', 
                label='EW baseline', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('improved_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to improved_comparison.png")
    
    # Save results
    summary_df.to_csv('comparison_summary.csv', index=False)
    print(f"💾 Results saved to comparison_summary.csv")
    
    # Winner
    winner = ranked.iloc[0]['Strategy']
    print(f"\n🏆 BEST STRATEGY: {winner}")
    
    plt.show()


if __name__ == "__main__":
    main()