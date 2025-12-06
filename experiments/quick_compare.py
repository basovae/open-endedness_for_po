# experiments/quick_compare.py
"""
Quick Comparison Script with Visualization
==========================================

Faster version for quick testing. Compares:
- Equal Weight baseline
- DQN vanilla
- DQN + Novelty Search

Run: python experiments/quick_compare.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List

import predictors
from deep_q_learning import DeepQLearning


def calculate_cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """Calculate cumulative returns."""
    return np.cumprod(1 + returns) - 1


def calculate_sharpe(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe ratio."""
    return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    return np.min(drawdowns)


def run_backtest(weights_fn, test_data: pd.DataFrame, lookback: int) -> Dict:
    """Run backtest and return metrics."""
    returns = []
    weights_history = []
    
    for t in range(lookback, len(test_data)):
        state = test_data.iloc[t-lookback:t].values.flatten()
        weights = weights_fn(state)
        weights_history.append(weights)
        
        day_returns = test_data.iloc[t].values
        portfolio_return = np.dot(weights, day_returns)
        returns.append(portfolio_return)
    
    returns = np.array(returns)
    
    return {
        'returns': returns,
        'cumulative': calculate_cumulative_returns(returns),
        'sharpe': calculate_sharpe(returns),
        'annual_return': np.mean(returns) * 252,
        'volatility': np.std(returns) * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(returns),
        'weights_history': weights_history,
    }


def main():
    print("="*60)
    print("QUICK STRATEGY COMPARISON")
    print("="*60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "multiasset_daily_returns.csv")
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
    
    lookback = 50
    results = {}
    
    # -------------------------------------------------------------------------
    # 1. Equal Weight Baseline
    # -------------------------------------------------------------------------
    print("\n[1/3] Equal Weight baseline...")
    equal_weights = np.ones(n_assets) / n_assets
    results['Equal Weight'] = run_backtest(
        lambda s: equal_weights.copy(), 
        test_data, 
        lookback
    )
    
    # -------------------------------------------------------------------------
    # 2. DQN Vanilla
    # -------------------------------------------------------------------------
    print("[2/3] Training DQN (vanilla)...")
    model_vanilla = DeepQLearning(
        lookback_window=lookback,
        predictor=predictors.MLP,
        batch_size=1,
        short_selling=False,
        forecast_window=0,
        reduce_negatives=True,
        verbose=0,
        hidden_sizes=(64, 64),
    )
    
    model_vanilla.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=10,
        use_ns=False,
    )
    
    model_vanilla.actor.eval()
    def vanilla_predict(state):
        with torch.no_grad():
            return model_vanilla.actor(torch.tensor(state, dtype=torch.float32)).numpy()
    
    results['DQN (vanilla)'] = run_backtest(vanilla_predict, test_data, lookback)
    
    # -------------------------------------------------------------------------
    # 3. DQN + Novelty Search
    # -------------------------------------------------------------------------
    print("[3/3] Training DQN + Novelty Search...")
    model_ns = DeepQLearning(
        lookback_window=lookback,
        predictor=predictors.MLP,
        batch_size=1,
        short_selling=False,
        forecast_window=0,
        reduce_negatives=True,
        verbose=0,
        hidden_sizes=(64, 64),
    )
    
    model_ns.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=10,
        use_ns=True,
        ns_alpha=1.0,
        ns_beta=0.5,
    )
    
    model_ns.actor.eval()
    def ns_predict(state):
        with torch.no_grad():
            return model_ns.actor(torch.tensor(state, dtype=torch.float32)).numpy()
    
    results['DQN + NS'] = run_backtest(ns_predict, test_data, lookback)
    
    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n{'Strategy':<20} {'Sharpe':>10} {'Return':>10} {'Vol':>10} {'MaxDD':>10}")
    print("-"*60)
    
    for name, r in results.items():
        print(f"{name:<20} {r['sharpe']:>10.3f} {r['annual_return']*100:>9.2f}% "
              f"{r['volatility']*100:>9.2f}% {r['max_drawdown']*100:>9.2f}%")
    
    # -------------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    for name, r in results.items():
        ax1.plot(r['cumulative'] * 100, label=name, linewidth=2)
    ax1.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. Sharpe Comparison
    ax2 = axes[0, 1]
    names = list(results.keys())
    sharpes = [results[n]['sharpe'] for n in names]
    colors = ['#2ecc71' if s == max(sharpes) else '#3498db' for s in sharpes]
    bars = ax2.bar(names, sharpes, color=colors, edgecolor='black')
    ax2.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', fontsize=10)
    
    # 3. Rolling Sharpe (60-day)
    ax3 = axes[1, 0]
    window = 60
    for name, r in results.items():
        returns = r['returns']
        rolling_sharpe = pd.Series(returns).rolling(window).apply(
            lambda x: np.mean(x) / (np.std(x) + 1e-8) * np.sqrt(252)
        )
        ax3.plot(rolling_sharpe, label=name, linewidth=1.5, alpha=0.8)
    ax3.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Weight Distribution (final weights for RL strategies)
    ax4 = axes[1, 1]
    asset_names = list(df.columns)
    x = np.arange(len(asset_names))
    width = 0.25
    
    for i, (name, r) in enumerate(results.items()):
        if 'weights_history' in r and len(r['weights_history']) > 0:
            avg_weights = np.mean(r['weights_history'], axis=0)
            ax4.bar(x + i*width, avg_weights * 100, width, label=name, alpha=0.8)
    
    ax4.set_title('Average Portfolio Weights', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Asset')
    ax4.set_ylabel('Weight (%)')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(asset_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Plot saved to strategy_comparison.png")
    
    # -------------------------------------------------------------------------
    # WINNER
    # -------------------------------------------------------------------------
    best_strategy = max(results.keys(), key=lambda k: results[k]['sharpe'])
    print(f"\n🏆 WINNER: {best_strategy} (Sharpe: {results[best_strategy]['sharpe']:.3f})")
    
    plt.show()


if __name__ == "__main__":
    main()