# experiments/compare_all_strategies.py
"""
Comprehensive Comparison: Baselines vs Novelty Search vs MAP-Elites
====================================================================

This script runs a rigorous comparison of:
1. Classical baselines (Equal Weight, Min Variance, etc.)
2. RL without QD (vanilla DDPG/DQN)
3. RL + Novelty Search
4. MAP-Elites (population-based QD)

Metrics compared:
- Financial: Sharpe ratio, annual return, max drawdown, volatility
- Diversity: Weight entropy, turnover, strategy variance
- Robustness: Performance across different market regimes

Run from project root: python experiments/compare_all_strategies.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Project imports
import predictors
from deep_q_learning import DeepQLearning
from ddpg import DDPG


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for comparison experiment."""
    # Data
    data_path: str = "multiasset_daily_returns.csv"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # RL hyperparameters
    lookback_window: int = 50
    hidden_sizes: Tuple[int, int] = (64, 64)
    num_epochs: int = 20
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    
    # NS hyperparameters
    ns_alpha: float = 1.0
    ns_beta: float = 0.5
    
    # MAP-Elites hyperparameters
    me_iterations: int = 500
    me_mutation_sigma: float = 0.2
    me_grid_dims: Tuple[int, int] = (10, 10)
    
    # Evaluation
    n_runs: int = 3  # Number of runs for statistical significance
    random_seed: int = 42


# =============================================================================
# BASELINE STRATEGIES
# =============================================================================

class EqualWeightStrategy:
    """Equal weight (1/N) portfolio."""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
    
    def fit(self, train_data: pd.DataFrame):
        pass  # No fitting needed
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.weights.copy()


class MinVarianceStrategy:
    """Minimum variance portfolio using sample covariance."""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.weights = None
    
    def fit(self, train_data: pd.DataFrame):
        cov = train_data.cov().values
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(self.n_assets)
            self.weights = inv_cov @ ones / (ones @ inv_cov @ ones)
            # Ensure non-negative (long-only)
            self.weights = np.clip(self.weights, 0, None)
            self.weights /= self.weights.sum()
        except np.linalg.LinAlgError:
            self.weights = np.ones(self.n_assets) / self.n_assets
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.weights.copy()


class MomentumStrategy:
    """Simple momentum strategy - overweight recent winners."""
    
    def __init__(self, n_assets: int, lookback: int = 20):
        self.n_assets = n_assets
        self.lookback = lookback
    
    def fit(self, train_data: pd.DataFrame):
        pass
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        # state is flattened (lookback * n_assets)
        returns = state.reshape(-1, self.n_assets)
        # Use cumulative returns over lookback period
        cum_returns = (1 + returns).prod(axis=0) - 1
        # Convert to weights (softmax-like)
        weights = np.exp(cum_returns * 10)  # Scale factor
        weights = np.clip(weights, 0, None)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(self.n_assets) / self.n_assets
        return weights


class InverseVolatilityStrategy:
    """Risk parity approximation - weight inversely to volatility."""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.weights = None
    
    def fit(self, train_data: pd.DataFrame):
        vols = train_data.std().values
        inv_vols = 1.0 / (vols + 1e-8)
        self.weights = inv_vols / inv_vols.sum()
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.weights.copy()


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(returns: np.ndarray, weights_history: List[np.ndarray]) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    
    # Financial metrics
    annual_return = np.mean(returns) * 252
    annual_vol = np.std(returns) * np.sqrt(252)
    sharpe = annual_return / (annual_vol + 1e-8)
    
    # Drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 1e-8
    sortino = annual_return / downside_vol
    
    # Calmar ratio
    calmar = annual_return / (abs(max_drawdown) + 1e-8)
    
    # Diversity metrics
    weights_array = np.array(weights_history)
    
    # Weight entropy (higher = more diversified)
    avg_weights = np.mean(weights_array, axis=0)
    entropy = -np.sum(avg_weights * np.log(avg_weights + 1e-8))
    
    # Average HHI (concentration)
    hhi = np.mean([np.sum(w**2) for w in weights_array])
    
    # Turnover (how much trading)
    if len(weights_array) > 1:
        turnover = np.mean([np.sum(np.abs(weights_array[i] - weights_array[i-1])) 
                          for i in range(1, len(weights_array))])
    else:
        turnover = 0.0
    
    # Strategy variance (how much weights change)
    weight_std = np.mean(np.std(weights_array, axis=0))
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'weight_entropy': entropy,
        'avg_hhi': hhi,
        'avg_turnover': turnover,
        'weight_std': weight_std,
    }


def evaluate_strategy(strategy, test_data: pd.DataFrame, lookback: int) -> Dict[str, float]:
    """Evaluate a strategy on test data."""
    
    returns = []
    weights_history = []
    
    for t in range(lookback, len(test_data)):
        # Get state
        state = test_data.iloc[t-lookback:t].values.flatten()
        
        # Get weights
        weights = strategy.predict(state)
        weights_history.append(weights)
        
        # Calculate return
        day_returns = test_data.iloc[t].values
        portfolio_return = np.dot(weights, day_returns)
        returns.append(portfolio_return)
    
    return calculate_metrics(np.array(returns), weights_history)


# =============================================================================
# RL STRATEGY WRAPPER
# =============================================================================

class RLStrategyWrapper:
    """Wrapper to evaluate trained RL models."""
    
    def __init__(self, model, n_assets: int):
        self.model = model
        self.n_assets = n_assets
        self.actor = model.actor
        self.actor.eval()
    
    def fit(self, train_data: pd.DataFrame):
        pass  # Already trained
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            weights = self.actor(state_tensor).numpy()
        return weights


# =============================================================================
# MAP-ELITES STRATEGY
# =============================================================================

def run_map_elites_experiment(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: ExperimentConfig
) -> Tuple[object, Dict]:
    """Run MAP-Elites and return best policy + archive stats."""
    
    from qd.map_elites import MAPElitesArchive
    from qd.me_trainer import MAPElitesTrainer
    from qd.bd_presets import bd_for_map_elites
    
    n_assets = train_data.shape[1]
    lookback = config.lookback_window
    input_size = lookback * n_assets
    
    # Policy factory
    def policy_factory():
        return nn.Sequential(
            nn.Linear(input_size, config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_sizes[1], n_assets),
            nn.Softmax(dim=-1)
        )
    
    # Evaluator
    def evaluator(policy):
        policy.eval()
        returns = []
        weights_history = []
        
        # Use validation data for fitness evaluation
        data = val_data
        
        with torch.no_grad():
            for t in range(lookback, len(data)):
                state = torch.tensor(
                    data.iloc[t-lookback:t].values.flatten(), 
                    dtype=torch.float32
                )
                weights = policy(state).numpy()
                weights_history.append(weights)
                
                day_return = np.dot(weights, data.iloc[t].values)
                returns.append(day_return)
        
        returns = np.array(returns)
        
        # Fitness = Sharpe ratio
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Behavior descriptor
        traj = {
            "weights_traj": np.array(weights_history),
            "returns": returns
        }
        bd = bd_for_map_elites(traj)
        
        return sharpe, bd
    
    # Create archive and trainer
    archive = MAPElitesArchive(
        dims=config.me_grid_dims,
        bd_bounds=((0.1, 1.0), (0.0, 0.05))  # HHI and volatility
    )
    
    trainer = MAPElitesTrainer(
        policy_factory=policy_factory,
        archive=archive,
        evaluator=evaluator,
        mutation_sigma=config.me_mutation_sigma
    )
    
    # Run MAP-Elites
    print(f"    Running MAP-Elites for {config.me_iterations} iterations...")
    trainer.initialize(n_random=50)
    
    for i in range(config.me_iterations):
        trainer.step()
        if (i + 1) % 100 == 0:
            print(f"      Iter {i+1}: coverage={archive.coverage():.1%}, "
                  f"QD-score={archive.qd_score():.2f}")
    
    # Get best policy
    best_policy = trainer.get_best_policy()
    
    # Archive statistics
    archive_stats = {
        'coverage': archive.coverage(),
        'qd_score': archive.qd_score(),
        'n_elites': len(archive),
        'max_fitness': archive.max_fitness()
    }
    
    return best_policy, archive_stats


class MAPElitesWrapper:
    """Wrapper for MAP-Elites best policy."""
    
    def __init__(self, policy, n_assets: int):
        self.policy = policy
        self.n_assets = n_assets
        if policy is not None:
            self.policy.eval()
    
    def fit(self, train_data: pd.DataFrame):
        pass
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        if self.policy is None:
            return np.ones(self.n_assets) / self.n_assets
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            weights = self.policy(state_tensor).numpy()
        return weights


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(config: ExperimentConfig) -> pd.DataFrame:
    """Run full comparison of all strategies."""
    
    print("="*70)
    print("STRATEGY COMPARISON: Baselines vs Novelty Search vs MAP-Elites")
    print("="*70)
    
    # Load data
    print(f"\n[1] Loading data from {config.data_path}...")
    df = pd.read_csv(config.data_path, index_col=0, parse_dates=True).sort_index()
    n_assets = df.shape[1]
    
    # Split data
    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))
    
    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]
    
    print(f"    Train: {len(train_data)} days")
    print(f"    Val:   {len(val_data)} days")
    print(f"    Test:  {len(test_data)} days")
    print(f"    Assets: {list(df.columns)}")
    
    all_results = []
    
    # -------------------------------------------------------------------------
    # BASELINES
    # -------------------------------------------------------------------------
    print(f"\n[2] Evaluating baseline strategies...")
    
    baselines = {
        'Equal Weight': EqualWeightStrategy(n_assets),
        'Min Variance': MinVarianceStrategy(n_assets),
        'Inverse Vol': InverseVolatilityStrategy(n_assets),
        'Momentum': MomentumStrategy(n_assets, lookback=20),
    }
    
    for name, strategy in baselines.items():
        print(f"    {name}...")
        strategy.fit(train_data)
        metrics = evaluate_strategy(strategy, test_data, config.lookback_window)
        metrics['strategy'] = name
        metrics['category'] = 'Baseline'
        all_results.append(metrics)
    
    # -------------------------------------------------------------------------
    # RL WITHOUT QD (multiple runs for variance)
    # -------------------------------------------------------------------------
    print(f"\n[3] Training RL without QD ({config.n_runs} runs)...")
    
    rl_vanilla_results = []
    for run in range(config.n_runs):
        print(f"    Run {run+1}/{config.n_runs}...")
        
        model = DeepQLearning(
            lookback_window=config.lookback_window,
            predictor=predictors.MLP,
            batch_size=1,
            short_selling=False,
            forecast_window=0,
            reduce_negatives=True,
            verbose=0,
            hidden_sizes=config.hidden_sizes,
            seed=config.random_seed + run,
        )
        
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            num_epochs=config.num_epochs,
            use_ns=False,  # No NS
        )
        
        wrapper = RLStrategyWrapper(model, n_assets)
        metrics = evaluate_strategy(wrapper, test_data, config.lookback_window)
        rl_vanilla_results.append(metrics)
    
    # Average results
    avg_metrics = {}
    for key in rl_vanilla_results[0].keys():
        values = [r[key] for r in rl_vanilla_results]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    avg_metrics['strategy'] = 'DQN (vanilla)'
    avg_metrics['category'] = 'RL'
    all_results.append(avg_metrics)
    
    # -------------------------------------------------------------------------
    # RL WITH NOVELTY SEARCH (multiple runs)
    # -------------------------------------------------------------------------
    print(f"\n[4] Training RL + Novelty Search ({config.n_runs} runs)...")
    
    rl_ns_results = []
    for run in range(config.n_runs):
        print(f"    Run {run+1}/{config.n_runs}...")
        
        model = DeepQLearning(
            lookback_window=config.lookback_window,
            predictor=predictors.MLP,
            batch_size=1,
            short_selling=False,
            forecast_window=0,
            reduce_negatives=True,
            verbose=0,
            hidden_sizes=config.hidden_sizes,
            seed=config.random_seed + run,
        )
        
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            num_epochs=config.num_epochs,
            use_ns=True,
            ns_alpha=config.ns_alpha,
            ns_beta=config.ns_beta,
        )
        
        wrapper = RLStrategyWrapper(model, n_assets)
        metrics = evaluate_strategy(wrapper, test_data, config.lookback_window)
        rl_ns_results.append(metrics)
    
    # Average results
    avg_metrics = {}
    for key in rl_ns_results[0].keys():
        values = [r[key] for r in rl_ns_results]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    avg_metrics['strategy'] = 'DQN + NS'
    avg_metrics['category'] = 'QD'
    all_results.append(avg_metrics)
    
    # -------------------------------------------------------------------------
    # MAP-ELITES
    # -------------------------------------------------------------------------
    print(f"\n[5] Running MAP-Elites...")
    
    me_results = []
    for run in range(config.n_runs):
        print(f"    Run {run+1}/{config.n_runs}...")
        
        torch.manual_seed(config.random_seed + run)
        np.random.seed(config.random_seed + run)
        
        best_policy, archive_stats = run_map_elites_experiment(
            train_data, val_data, config
        )
        
        wrapper = MAPElitesWrapper(best_policy, n_assets)
        metrics = evaluate_strategy(wrapper, test_data, config.lookback_window)
        metrics.update({f'me_{k}': v for k, v in archive_stats.items()})
        me_results.append(metrics)
    
    # Average results
    avg_metrics = {}
    for key in me_results[0].keys():
        values = [r[key] for r in me_results]
        avg_metrics[key] = np.mean(values)
        if not key.startswith('me_'):
            avg_metrics[f'{key}_std'] = np.std(values)
    avg_metrics['strategy'] = 'MAP-Elites'
    avg_metrics['category'] = 'QD'
    all_results.append(avg_metrics)
    
    # -------------------------------------------------------------------------
    # COMPILE RESULTS
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    
    return results_df


def print_results(results_df: pd.DataFrame):
    """Print formatted comparison results."""
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Key metrics
    key_metrics = ['sharpe_ratio', 'annual_return', 'max_drawdown', 'sortino_ratio']
    
    print("\n📊 FINANCIAL PERFORMANCE")
    print("-"*70)
    
    for metric in key_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        for _, row in results_df.iterrows():
            val = row[metric]
            std_key = f'{metric}_std'
            if std_key in row and pd.notna(row[std_key]):
                print(f"  {row['strategy']:20s}: {val:8.4f} ± {row[std_key]:.4f}")
            else:
                print(f"  {row['strategy']:20s}: {val:8.4f}")
    
    print("\n\n📈 DIVERSITY METRICS")
    print("-"*70)
    
    diversity_metrics = ['weight_entropy', 'avg_hhi', 'avg_turnover', 'weight_std']
    
    for metric in diversity_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        for _, row in results_df.iterrows():
            val = row[metric]
            print(f"  {row['strategy']:20s}: {val:8.4f}")
    
    # Rankings
    print("\n\n🏆 RANKINGS (by Sharpe Ratio)")
    print("-"*70)
    
    ranked = results_df.sort_values('sharpe_ratio', ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['strategy']:20s} - Sharpe: {row['sharpe_ratio']:.4f}")
    
    # Winner
    best = ranked.iloc[0]
    print(f"\n✅ BEST STRATEGY: {best['strategy']}")
    print(f"   Sharpe Ratio: {best['sharpe_ratio']:.4f}")
    print(f"   Annual Return: {best['annual_return']*100:.2f}%")
    print(f"   Max Drawdown: {best['max_drawdown']*100:.2f}%")


def save_results(results_df: pd.DataFrame, filename: str = "comparison_results.csv"):
    """Save results to CSV."""
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Results saved to {filename}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config = ExperimentConfig(
        num_epochs=15,          # Reduce for faster testing
        n_runs=3,               # 3 runs for statistical significance
        me_iterations=300,      # MAP-Elites iterations
    )
    
    results_df = run_comparison(config)
    print_results(results_df)
    save_results(results_df)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)