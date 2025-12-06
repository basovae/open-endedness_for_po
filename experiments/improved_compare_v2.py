"""
IMPROVED STRATEGY COMPARISON (v2)
=================================
This version includes all suggested improvements with clear markers.

Changes from original:
- [CHANGE 1] Asset selection to reduce from 336 → configurable N_ASSETS
- [CHANGE 2] Increased runs from 5 → 20 with proper statistical tests
- [CHANGE 3] Tuned hyperparameters (lower learning rates, higher patience)
- [CHANGE 4] Added training diagnostics
- [CHANGE 5] Fixed NS normalization
- [CHANGE 6] Incremental testing (start small, scale up)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Your existing imports
from ddpg import DDPG
from deep_q_learning import DeepQLearning
from predictors import NeuralNetwork
from utility_functions import calculate_test_performance

# ============================================================================
# [CHANGE 1] ASSET SELECTION - Reduce dimensionality
# ============================================================================

def select_top_assets(
    returns_df: pd.DataFrame, 
    n_assets: int = 30, 
    method: str = 'sharpe'
) -> pd.DataFrame:
    """
    ### [NEW FUNCTION] ###
    Select top N assets by criterion to reduce dimensionality.
    
    Args:
        returns_df: Full returns DataFrame with all assets
        n_assets: Number of assets to select (default 30, was 336)
        method: Selection criterion ('sharpe', 'variance', 'random')
    
    Returns:
        Filtered DataFrame with only top N assets
    """
    if n_assets >= len(returns_df.columns):
        return returns_df
    
    if method == 'sharpe':
        # Select assets with highest Sharpe ratio
        sharpe = returns_df.mean() / (returns_df.std() + 1e-8)
        top_assets = sharpe.nlargest(n_assets).index.tolist()
    
    elif method == 'variance':
        # Select assets with highest variance (more signal)
        variance = returns_df.var()
        top_assets = variance.nlargest(n_assets).index.tolist()
    
    elif method == 'low_correlation':
        # Select assets with low pairwise correlation (more diversification)
        corr_matrix = returns_df.corr().abs()
        selected = [returns_df.columns[0]]  # Start with first
        
        while len(selected) < n_assets:
            remaining = [c for c in returns_df.columns if c not in selected]
            if not remaining:
                break
            # Find asset with lowest average correlation to selected
            avg_corr = corr_matrix.loc[remaining, selected].mean(axis=1)
            best = avg_corr.idxmin()
            selected.append(best)
        top_assets = selected
    
    elif method == 'random':
        top_assets = np.random.choice(
            returns_df.columns, size=n_assets, replace=False
        ).tolist()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"[ASSET SELECTION] Selected {len(top_assets)} assets using '{method}' method")
    return returns_df[top_assets]


# ============================================================================
# [CHANGE 3] IMPROVED HYPERPARAMETERS
# ============================================================================

@dataclass
class ImprovedConfig:
    """
    ### [NEW CLASS] ###
    Centralized configuration with tuned hyperparameters.
    
    Key changes from defaults:
    - actor_lr: 0.05 → 0.005 (10x lower for stability with large input)
    - critic_lr: 0.01 → 0.005 (2x lower)
    - patience: 2 → 10 (5x higher - don't stop too early)
    - num_epochs: 100 → 300 (3x higher)
    - ns_beta: 0.3 → 0.7 (higher novelty weight)
    """
    # === Asset Selection ===
    n_assets: int = 30                    # [CHANGED] Was 336 (full universe)
    asset_selection_method: str = 'sharpe'
    
    # === Training ===
    lookback_window: int = 50
    batch_size: int = 1
    num_epochs: int = 300                 # [CHANGED] Was 100
    
    # === Learning Rates (REDUCED for stability) ===
    actor_lr: float = 0.005               # [CHANGED] Was 0.05
    critic_lr: float = 0.005              # [CHANGED] Was 0.01
    
    # === Early Stopping (MORE PATIENT) ===
    patience: int = 10                    # [CHANGED] Was 2
    min_delta: float = 1e-6               # [CHANGED] Was 0
    early_stopping: bool = True
    
    # === Risk & Reward ===
    risk_preference: float = -0.5
    gamma: float = 1.0
    
    # === Regularization ===
    l1_lambda: float = 0.001              # [CHANGED] Was 0
    l2_lambda: float = 0.001              # [CHANGED] Was 0
    weight_decay: float = 0.001           # [CHANGED] Was 0
    
    # === Target Networks ===
    soft_update: bool = True
    tau: float = 0.005
    
    # === Novelty Search (REBALANCED) ===
    ns_alpha: float = 0.5                 # [CHANGED] Was 1.0 (reduce task weight)
    ns_beta: float = 0.7                  # [CHANGED] Was 0.3 (increase novelty weight)
    
    # === DQN Specific ===
    num_action_samples: int = 20          # [CHANGED] Was 10
    
    # === Experiment ===
    n_runs: int = 20                      # [CHANGED] Was 5
    verbose: int = 0


# ============================================================================
# [CHANGE 4] TRAINING DIAGNOSTICS
# ============================================================================

class TrainingDiagnostics:
    """
    ### [NEW CLASS] ###
    Track and visualize training metrics to diagnose convergence issues.
    """
    def __init__(self):
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.rewards: List[float] = []
        self.q_values: List[float] = []
        self.portfolio_entropy: List[float] = []
        self.portfolio_concentration: List[float] = []  # HHI
        
    def log_step(
        self, 
        actor_loss: float, 
        critic_loss: float, 
        reward: float, 
        weights: np.ndarray,
        q_value: float = 0.0
    ):
        """Log metrics for one training step."""
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.rewards.append(reward)
        self.q_values.append(q_value)
        
        # Portfolio entropy (higher = more diversified)
        weights = np.array(weights).flatten()
        weights = np.clip(weights, 1e-10, 1)
        weights = weights / weights.sum()  # Normalize
        entropy = -np.sum(weights * np.log(weights))
        self.portfolio_entropy.append(entropy)
        
        # HHI concentration (lower = more diversified)
        hhi = np.sum(weights ** 2)
        self.portfolio_concentration.append(hhi)
    
    def plot(self, save_path: str = 'training_diagnostics.png'):
        """Generate diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Smooth function
        def smooth(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Actor loss
        axes[0, 0].plot(smooth(self.actor_losses))
        axes[0, 0].set_title('Actor Loss (smoothed)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        
        # Critic loss
        axes[0, 1].plot(smooth(self.critic_losses))
        axes[0, 1].set_title('Critic Loss / TD Error (smoothed)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        
        # Rewards
        axes[0, 2].plot(smooth(self.rewards))
        axes[0, 2].set_title('Reward (smoothed)')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Reward')
        
        # Q-values (check for explosion)
        if any(self.q_values):
            axes[1, 0].plot(smooth(self.q_values))
            axes[1, 0].set_title('Q-Values (check for explosion)')
            axes[1, 0].set_xlabel('Step')
            # Add warning line if Q-values explode
            if max(self.q_values) > 100:
                axes[1, 0].axhline(y=100, color='r', linestyle='--', label='Warning threshold')
                axes[1, 0].legend()
        
        # Portfolio entropy
        n_assets = len(self.portfolio_entropy)
        if n_assets > 0:
            axes[1, 1].plot(smooth(self.portfolio_entropy))
            # Max entropy for equal weight
            max_entropy = np.log(30)  # Assuming 30 assets
            axes[1, 1].axhline(y=max_entropy, color='r', linestyle='--', 
                              label=f'Equal weight (max={max_entropy:.2f})')
            axes[1, 1].set_title('Portfolio Entropy (higher=more diverse)')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].legend()
        
        # HHI concentration
        axes[1, 2].plot(smooth(self.portfolio_concentration))
        axes[1, 2].axhline(y=1/30, color='r', linestyle='--', 
                          label=f'Equal weight (min={1/30:.3f})')
        axes[1, 2].set_title('HHI Concentration (lower=more diverse)')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[DIAGNOSTICS] Saved to {save_path}")
        
    def summary(self) -> Dict:
        """Return summary statistics."""
        return {
            'final_actor_loss': self.actor_losses[-1] if self.actor_losses else None,
            'final_critic_loss': self.critic_losses[-1] if self.critic_losses else None,
            'avg_reward': np.mean(self.rewards) if self.rewards else None,
            'avg_entropy': np.mean(self.portfolio_entropy) if self.portfolio_entropy else None,
            'avg_hhi': np.mean(self.portfolio_concentration) if self.portfolio_concentration else None,
            'q_value_max': max(self.q_values) if self.q_values else None,
            'converged': self._check_convergence(),
        }
    
    def _check_convergence(self) -> bool:
        """Check if training appears to have converged."""
        if len(self.critic_losses) < 100:
            return False
        # Compare first and last 10% of losses
        n = len(self.critic_losses)
        first_10 = np.mean(self.critic_losses[:n//10])
        last_10 = np.mean(self.critic_losses[-n//10:])
        # Converged if loss decreased by at least 20%
        return last_10 < first_10 * 0.8


# ============================================================================
# [CHANGE 2] IMPROVED STATISTICAL ANALYSIS
# ============================================================================

def run_statistical_analysis(
    results: Dict[str, List[float]], 
    baseline: str = 'Equal Weight',
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    ### [NEW FUNCTION] ###
    Perform proper statistical analysis with confidence intervals.
    
    Args:
        results: Dict mapping strategy name to list of Sharpe ratios
        baseline: Name of baseline strategy for comparison
        confidence: Confidence level for intervals
    
    Returns:
        DataFrame with statistics
    """
    analysis = []
    
    baseline_sharpes = results.get(baseline, [])
    
    for strategy, sharpes in results.items():
        n = len(sharpes)
        mean = np.mean(sharpes)
        std = np.std(sharpes, ddof=1)
        se = stats.sem(sharpes)
        
        # Confidence interval
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        
        # Compare to baseline
        if strategy != baseline and len(baseline_sharpes) == n:
            # Paired t-test (same seeds used)
            t_stat, p_val = stats.ttest_rel(sharpes, baseline_sharpes)
            effect_size = (mean - np.mean(baseline_sharpes)) / np.sqrt(
                (std**2 + np.std(baseline_sharpes, ddof=1)**2) / 2
            )  # Cohen's d
        else:
            t_stat, p_val, effect_size = None, None, None
        
        analysis.append({
            'Strategy': strategy,
            'Mean Sharpe': mean,
            'Std': std,
            'SE': se,
            f'CI {int(confidence*100)}% Low': ci[0],
            f'CI {int(confidence*100)}% High': ci[1],
            'vs Baseline t-stat': t_stat,
            'vs Baseline p-value': p_val,
            'Effect Size (d)': effect_size,
            'Significant (p<0.05)': p_val < 0.05 if p_val else None,
        })
    
    df = pd.DataFrame(analysis)
    return df.sort_values('Mean Sharpe', ascending=False)


# ============================================================================
# [CHANGE 5] IMPROVED NS WRAPPER WITH NORMALIZATION
# ============================================================================

class NSWrapperNormalized:
    """
    ### [NEW CLASS] ###
    Novelty Search wrapper with proper reward normalization.
    
    The original issue: task returns might be ~0.01 while novelty is 0-1,
    making the blending ratio meaningless.
    
    Solution: Normalize both to zero-mean, unit-variance before blending.
    """
    def __init__(
        self,
        bd_fn,
        alpha: float = 0.5,
        beta: float = 0.5,
        archive = None,
        warmup_episodes: int = 10,
    ):
        from qd.novelty_archive import NoveltyArchive
        
        self.bd_fn = bd_fn
        self.alpha = alpha
        self.beta = beta
        self.archive = archive or NoveltyArchive()
        self.warmup_episodes = warmup_episodes
        
        # Running statistics for normalization
        self.task_returns: List[float] = []
        self.novelties: List[float] = []
        
        # Episode buffers
        self.reset_episode_buffers()
    
    def reset_episode_buffers(self):
        self.buf = {
            "weights_traj": [],
            "returns": [],
            "actions": [],
            "states": []
        }
    
    def on_step(self, state, action, reward_task: float, info: Dict):
        """Record step data for behavior descriptor computation."""
        if "weights" in info:
            self.buf["weights_traj"].append(np.asarray(info["weights"], dtype=float))
        if "return_t" in info:
            self.buf["returns"].append(float(info["return_t"]))
        self.buf["actions"].append(np.asarray(action).ravel())
        if state is not None:
            self.buf["states"].append(np.asarray(state).ravel())
        return reward_task
    
    def on_episode_end(self, episode_task_return: float) -> float:
        """
        Compute blended reward at episode end.
        
        ### [KEY CHANGE] ###
        Now normalizes both task return and novelty before blending.
        """
        # Build behavior descriptor
        traj = {
            k: (np.vstack(v) if len(v) > 0 and isinstance(v[0], (list, np.ndarray)) 
                else np.array(v))
            for k, v in self.buf.items()
        }
        
        desc = self.bd_fn(traj)
        nov = self.archive.novelty(desc)
        self.archive.maybe_add(desc)
        
        # Track for normalization
        self.task_returns.append(episode_task_return)
        self.novelties.append(nov)
        
        # [KEY CHANGE] Normalize both signals after warmup
        if len(self.task_returns) > self.warmup_episodes:
            # Z-score normalization
            task_mean = np.mean(self.task_returns)
            task_std = np.std(self.task_returns) + 1e-8
            task_norm = (episode_task_return - task_mean) / task_std
            
            nov_mean = np.mean(self.novelties)
            nov_std = np.std(self.novelties) + 1e-8
            nov_norm = (nov - nov_mean) / nov_std
            
            # Blend normalized signals
            blended = self.alpha * task_norm + self.beta * nov_norm
        else:
            # During warmup, use raw values
            blended = self.alpha * episode_task_return + self.beta * nov
        
        self.reset_episode_buffers()
        return blended


# ============================================================================
# [CHANGE 6] INCREMENTAL TESTING
# ============================================================================

def run_incremental_tests(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: ImprovedConfig,
) -> Dict:
    """
    ### [NEW FUNCTION] ###
    Test at increasing scales to find where RL starts to struggle.
    
    Returns early if RL can't beat equal weight at smaller scale.
    """
    scales = [10, 20, 30, 50, 100, 200, 336]
    scales = [s for s in scales if s <= len(train_data.columns)]
    
    results = {}
    
    for n_assets in scales:
        print(f"\n{'='*60}")
        print(f"[INCREMENTAL TEST] n_assets = {n_assets}")
        print('='*60)
        
        # Select assets
        train_subset = select_top_assets(train_data, n_assets, config.asset_selection_method)
        val_subset = val_data[train_subset.columns]
        test_subset = test_data[train_subset.columns]
        
        # Equal weight baseline
        ew_weights = np.ones(n_assets) / n_assets
        _, ew_sharpe = calculate_test_performance(test_subset, ew_weights)
        
        # Train DDPG (single run for speed)
        model = DDPG(
            lookback_window=config.lookback_window,
            predictor=NeuralNetwork,
            batch_size=config.batch_size,
            verbose=0,
            hidden_sizes=[64, 32],
        )
        
        model.train(
            train_data=train_subset,
            val_data=val_subset,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            num_epochs=config.num_epochs,
            patience=config.patience,
            risk_preference=config.risk_preference,
            soft_update=config.soft_update,
            tau=config.tau,
        )
        
        (_, ddpg_sharpe), _ = model.evaluate(test_subset, dpo=True)
        
        results[n_assets] = {
            'ew_sharpe': ew_sharpe,
            'ddpg_sharpe': ddpg_sharpe,
            'ddpg_beats_ew': ddpg_sharpe > ew_sharpe,
        }
        
        print(f"  Equal Weight Sharpe: {ew_sharpe:.3f}")
        print(f"  DDPG Sharpe:         {ddpg_sharpe:.3f}")
        print(f"  DDPG beats EW:       {ddpg_sharpe > ew_sharpe}")
        
        # Early stopping: if DDPG can't beat EW at this scale, don't scale up
        if not results[n_assets]['ddpg_beats_ew']:
            print(f"\n[EARLY STOP] DDPG underperforms at {n_assets} assets.")
            print("Fix at this scale before scaling up.")
            break
    
    return results


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: ImprovedConfig,
    strategy: str,
    seed: int,
    diagnostics: Optional[TrainingDiagnostics] = None,
) -> Dict:
    """
    Run a single experiment for one strategy.
    
    Args:
        train_data: Training data (already asset-filtered)
        val_data: Validation data
        test_data: Test data
        config: Hyperparameter configuration
        strategy: One of 'EW', 'DQN', 'DQN+NS', 'DDPG', 'DDPG+NS'
        seed: Random seed
        diagnostics: Optional diagnostics tracker
    
    Returns:
        Dict with performance metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_assets = len(train_data.columns)
    
    if strategy == 'Equal Weight':
        weights = np.ones(n_assets) / n_assets
        profit, sharpe = calculate_test_performance(test_data, weights)
        volatility = (test_data * weights).sum(axis=1).std() * np.sqrt(252)
        max_dd = _calculate_max_drawdown(test_data, weights)
        
        return {
            'strategy': strategy,
            'seed': seed,
            'sharpe': sharpe,
            'return': profit,
            'volatility': volatility,
            'max_drawdown': max_dd,
        }
    
    # RL strategies
    use_ns = '+NS' in strategy
    is_ddpg = 'DDPG' in strategy
    
    if is_ddpg:
        model = DDPG(
            lookback_window=config.lookback_window,
            predictor=NeuralNetwork,
            batch_size=config.batch_size,
            verbose=config.verbose,
            seed=seed,
            hidden_sizes=[64, 32],
        )
        
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            num_epochs=config.num_epochs,
            patience=config.patience,
            min_delta=config.min_delta,
            risk_preference=config.risk_preference,
            gamma=config.gamma,
            l1_lambda=config.l1_lambda,
            l2_lambda=config.l2_lambda,
            weight_decay=config.weight_decay,
            soft_update=config.soft_update,
            tau=config.tau,
            use_ns=use_ns,
            ns_alpha=config.ns_alpha,
            ns_beta=config.ns_beta,
        )
        
    else:  # DQN
        model = DeepQLearning(
            lookback_window=config.lookback_window,
            predictor=NeuralNetwork,
            batch_size=config.batch_size,
            verbose=config.verbose,
            seed=seed,
            hidden_sizes=[64, 32],
        )
        
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            num_epochs=config.num_epochs,
            patience=config.patience,
            min_delta=config.min_delta,
            risk_preference=config.risk_preference,
            gamma=config.gamma,
            soft_update=config.soft_update,
            tau=config.tau,
            num_action_samples=config.num_action_samples,
            use_ns=use_ns,
            ns_alpha=config.ns_alpha,
            ns_beta=config.ns_beta,
        )
    
    # Evaluate
    (spo_profit, spo_sharpe), (dpo_profit, dpo_sharpe) = model.evaluate(test_data, dpo=True)
    
    # Use DPO (dynamic) results
    return {
        'strategy': strategy,
        'seed': seed,
        'sharpe': dpo_sharpe,
        'return': dpo_profit,
        'volatility': np.nan,  # Would need to track from model
        'max_drawdown': np.nan,
    }


def _calculate_max_drawdown(returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """Calculate maximum drawdown for a portfolio."""
    portfolio_returns = (returns_df * weights).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return drawdowns.min()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main experiment runner with all improvements.
    """
    print("="*70)
    print(" IMPROVED STRATEGY COMPARISON (v2)")
    print(" With: Asset Selection, Better Stats, Tuned Hyperparameters")
    print("="*70)
    
    # Load data (adjust path as needed)
    # data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
    
    # For demonstration, create synthetic data
    print("\n[DATA] Loading data...")
    np.random.seed(42)
    n_days = 4458
    n_assets_full = 336
    
    dates = pd.date_range('2008-01-01', periods=n_days, freq='B')
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets_full) * 0.02,
        index=dates,
        columns=[f'ASSET_{i}' for i in range(n_assets_full)]
    )
    
    # Split data
    train_end = int(n_days * 0.6)
    val_end = int(n_days * 0.8)
    
    train_data = returns.iloc[:train_end]
    val_data = returns.iloc[train_end:val_end]
    test_data = returns.iloc[val_end:]
    
    print(f"  Train: {len(train_data)} days")
    print(f"  Val:   {len(val_data)} days")
    print(f"  Test:  {len(test_data)} days ({test_data.index[0]} to {test_data.index[-1]})")
    
    # Configuration
    config = ImprovedConfig()
    
    # [CHANGE 1] Select subset of assets
    print(f"\n[ASSET SELECTION] Reducing from {n_assets_full} to {config.n_assets} assets...")
    train_data = select_top_assets(train_data, config.n_assets, config.asset_selection_method)
    val_data = val_data[train_data.columns]
    test_data = test_data[train_data.columns]
    
    # Strategies to test
    strategies = ['Equal Weight', 'DQN', 'DQN+NS', 'DDPG', 'DDPG+NS']
    
    # [CHANGE 2] Run experiments
    print(f"\n[EXPERIMENT] Running {config.n_runs} experiments per strategy...")
    
    all_results = {s: [] for s in strategies}
    
    for run in range(config.n_runs):
        print(f"\nRun {run+1}/{config.n_runs}...")
        
        for strategy in strategies:
            result = run_single_experiment(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                config=config,
                strategy=strategy,
                seed=run,
            )
            all_results[strategy].append(result['sharpe'])
    
    # [CHANGE 2] Statistical analysis
    print("\n" + "="*70)
    print(" STATISTICAL ANALYSIS")
    print("="*70)
    
    stats_df = run_statistical_analysis(
        {s: sharpes for s, sharpes in all_results.items()},
        baseline='Equal Weight',
        confidence=0.95
    )
    
    print(stats_df.to_string(index=False))
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    for strategy in strategies:
        sharpes = all_results[strategy]
        mean = np.mean(sharpes)
        std = np.std(sharpes)
        print(f"{strategy:20s}: {mean:8.3f} ± {std:6.3f}")
    
    # Best strategy
    best = max(strategies, key=lambda s: np.mean(all_results[s]))
    print(f"\n🏆 BEST STRATEGY: {best}")
    
    # Save results
    results_df = pd.DataFrame({
        'Strategy': strategies,
        'Mean Sharpe': [np.mean(all_results[s]) for s in strategies],
        'Std Sharpe': [np.std(all_results[s]) for s in strategies],
    })
    results_df.to_csv('comparison_results_v2.csv', index=False)
    print("\n💾 Results saved to comparison_results_v2.csv")
    
    return all_results


if __name__ == '__main__':
    results = main()