# qd/wrappers.py
# ==============================================================================
# IMPROVED VERSION with normalization fix
# ==============================================================================
"""
Novelty Search wrappers for RL training loops.

### CHANGES FROM ORIGINAL ###
1. [CHANGE 5] Added NSWrapperNormalized with proper reward normalization
2. Added warmup period before normalization kicks in
3. Added diagnostic logging
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List
from .novelty_archive import NoveltyArchive


class NSWrapper:
    """
    Original Novelty Search wrapper (unchanged for backward compatibility).
    
    Adds Novelty Search to an existing RL training loop by:
      1) collecting per-episode trajectories,
      2) computing a behavior descriptor (callable you pass),
      3) computing novelty against an archive,
      4) blending task reward with novelty: R' = alpha * R_task + beta * novelty.
    """
    def __init__(
        self,
        bd_fn: Callable[[Dict[str, Any]], np.ndarray],
        alpha: float = 1.0,        # weight for task reward
        beta: float = 1.0,         # weight for novelty
        archive: Optional[NoveltyArchive] = None,
    ):
        self.bd_fn = bd_fn
        self.alpha = alpha
        self.beta = beta
        self.archive = archive or NoveltyArchive()
        # rolling buffers for a single episode
        self.reset_episode_buffers()

    def reset_episode_buffers(self):
        self.buf = {
            "weights_traj": [],     # list of np.ndarray (n_assets,)
            "returns": [],          # list of floats (r_t)
            "actions": [],          # optional
            "states": []            # optional
        }

    def on_step(self, state, action, reward_task: float, info: Dict[str, Any]):
        # you can push more depending on your trainer
        if "weights" in info:
            self.buf["weights_traj"].append(np.asarray(info["weights"], dtype=float))
        if "return_t" in info:
            self.buf["returns"].append(float(info["return_t"]))
        self.buf["actions"].append(np.asarray(action).ravel())
        self.buf["states"].append(np.asarray(state).ravel())
        # step reward remains task reward; novelty is applied at episode end
        return reward_task

    def on_episode_end(self, episode_task_return: float) -> float:
        # build behavior descriptor
        traj = {k: (np.vstack(v) if len(v)>0 and isinstance(v[0], (list, np.ndarray)) else np.array(v))
                for k, v in self.buf.items()}
        desc = self.bd_fn(traj)
        nov = self.archive.novelty(desc)
        self.archive.maybe_add(desc)
        # blend rewards at the episode level
        blended = self.alpha * float(episode_task_return) + self.beta * float(nov)
        # clear buffers for next episode
        self.reset_episode_buffers()
        return blended
    
    def __len__(self):
        return len(self.archive)


# ==============================================================================
# [CHANGE 5] NEW: Normalized NS Wrapper
# ==============================================================================

class NSWrapperNormalized(NSWrapper):
    """
    ### [NEW CLASS] ###
    Novelty Search wrapper with proper reward normalization.
    
    PROBLEM WITH ORIGINAL:
    - Task returns might be ~0.01 (daily returns)
    - Novelty scores are 0-1
    - With alpha=1.0, beta=0.3:
      - Blended = 1.0 * 0.01 + 0.3 * 0.5 = 0.16
      - Novelty dominates even with lower beta!
    
    SOLUTION:
    - Normalize both task return and novelty to z-scores
    - Then blend: R' = alpha * z(task) + beta * z(novelty)
    - This makes alpha/beta ratios meaningful
    
    Args:
        bd_fn: Function to compute behavior descriptor from trajectory
        alpha: Weight for normalized task reward (default 0.5)
        beta: Weight for normalized novelty (default 0.5)
        archive: Optional pre-existing novelty archive
        warmup_episodes: Episodes before normalization kicks in (default 10)
        use_running_stats: Whether to use running mean/std vs. full history
    """
    
    def __init__(
        self,
        bd_fn: Callable[[Dict[str, Any]], np.ndarray],
        alpha: float = 0.5,                    # [CHANGED] Was 1.0
        beta: float = 0.5,                     # [CHANGED] Was 1.0  
        archive: Optional[NoveltyArchive] = None,
        warmup_episodes: int = 10,             # [NEW] Warmup before normalization
        use_running_stats: bool = True,        # [NEW] Use EMA for efficiency
        ema_decay: float = 0.99,               # [NEW] EMA decay factor
    ):
        super().__init__(bd_fn, alpha, beta, archive)
        
        self.warmup_episodes = warmup_episodes
        self.use_running_stats = use_running_stats
        self.ema_decay = ema_decay
        
        # Statistics tracking
        self.task_returns_history: List[float] = []
        self.novelties_history: List[float] = []
        
        # Running statistics (EMA)
        self.task_mean = 0.0
        self.task_var = 1.0
        self.nov_mean = 0.5  # Novelty starts around 0.5
        self.nov_var = 0.25
        
        # Episode counter
        self.episode_count = 0
        
        # Diagnostic tracking
        self.diagnostics = {
            'raw_task_returns': [],
            'raw_novelties': [],
            'normalized_task': [],
            'normalized_novelty': [],
            'blended_rewards': [],
        }
    
    def on_episode_end(self, episode_task_return: float) -> float:
        """
        Compute blended reward with normalization.
        
        ### [KEY CHANGE FROM ORIGINAL] ###
        """
        # Build behavior descriptor (same as parent)
        traj = {
            k: (np.vstack(v) if len(v) > 0 and isinstance(v[0], (list, np.ndarray)) 
                else np.array(v))
            for k, v in self.buf.items()
        }
        
        desc = self.bd_fn(traj)
        nov = self.archive.novelty(desc)
        self.archive.maybe_add(desc)
        
        # Track history
        self.task_returns_history.append(episode_task_return)
        self.novelties_history.append(nov)
        self.episode_count += 1
        
        # Store raw values for diagnostics
        self.diagnostics['raw_task_returns'].append(episode_task_return)
        self.diagnostics['raw_novelties'].append(nov)
        
        # === NORMALIZATION LOGIC ===
        if self.episode_count <= self.warmup_episodes:
            # During warmup: use raw values (no normalization)
            task_norm = episode_task_return
            nov_norm = nov
            blended = self.alpha * task_norm + self.beta * nov_norm
            
        else:
            if self.use_running_stats:
                # Update running statistics with EMA
                self._update_running_stats(episode_task_return, nov)
                
                # Z-score normalize using running stats
                task_std = np.sqrt(self.task_var) + 1e-8
                nov_std = np.sqrt(self.nov_var) + 1e-8
                
                task_norm = (episode_task_return - self.task_mean) / task_std
                nov_norm = (nov - self.nov_mean) / nov_std
                
            else:
                # Use full history for normalization
                task_mean = np.mean(self.task_returns_history)
                task_std = np.std(self.task_returns_history) + 1e-8
                nov_mean = np.mean(self.novelties_history)
                nov_std = np.std(self.novelties_history) + 1e-8
                
                task_norm = (episode_task_return - task_mean) / task_std
                nov_norm = (nov - nov_mean) / nov_std
            
            # Blend normalized signals
            blended = self.alpha * task_norm + self.beta * nov_norm
        
        # Store normalized values for diagnostics
        self.diagnostics['normalized_task'].append(task_norm)
        self.diagnostics['normalized_novelty'].append(nov_norm)
        self.diagnostics['blended_rewards'].append(blended)
        
        # Clear buffers for next episode
        self.reset_episode_buffers()
        
        return blended
    
    def _update_running_stats(self, task_return: float, novelty: float):
        """Update running mean and variance using EMA."""
        # Task return stats
        delta_task = task_return - self.task_mean
        self.task_mean += (1 - self.ema_decay) * delta_task
        self.task_var = self.ema_decay * self.task_var + (1 - self.ema_decay) * delta_task**2
        
        # Novelty stats
        delta_nov = novelty - self.nov_mean
        self.nov_mean += (1 - self.ema_decay) * delta_nov
        self.nov_var = self.ema_decay * self.nov_var + (1 - self.ema_decay) * delta_nov**2
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about NS behavior."""
        n = len(self.diagnostics['raw_task_returns'])
        if n == 0:
            return {'episodes': 0}
        
        return {
            'episodes': n,
            'task_return_mean': np.mean(self.diagnostics['raw_task_returns']),
            'task_return_std': np.std(self.diagnostics['raw_task_returns']),
            'novelty_mean': np.mean(self.diagnostics['raw_novelties']),
            'novelty_std': np.std(self.diagnostics['raw_novelties']),
            'normalized_task_mean': np.mean(self.diagnostics['normalized_task'][-100:]),
            'normalized_novelty_mean': np.mean(self.diagnostics['normalized_novelty'][-100:]),
            'blended_mean': np.mean(self.diagnostics['blended_rewards'][-100:]),
            'archive_size': len(self.archive),
        }
    
    def plot_diagnostics(self, save_path: str = 'ns_diagnostics.png'):
        """Plot NS diagnostic charts."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = range(len(self.diagnostics['raw_task_returns']))
        
        # Raw values
        ax = axes[0, 0]
        ax.plot(episodes, self.diagnostics['raw_task_returns'], alpha=0.7, label='Task Return')
        ax.set_title('Raw Task Returns')
        ax.set_xlabel('Episode')
        ax.axvline(x=self.warmup_episodes, color='r', linestyle='--', label='Warmup end')
        ax.legend()
        
        ax = axes[0, 1]
        ax.plot(episodes, self.diagnostics['raw_novelties'], alpha=0.7, label='Novelty', color='orange')
        ax.set_title('Raw Novelty Scores')
        ax.set_xlabel('Episode')
        ax.axvline(x=self.warmup_episodes, color='r', linestyle='--', label='Warmup end')
        ax.legend()
        
        # Normalized values
        ax = axes[1, 0]
        ax.plot(episodes, self.diagnostics['normalized_task'], alpha=0.7, label='Norm Task')
        ax.plot(episodes, self.diagnostics['normalized_novelty'], alpha=0.7, label='Norm Novelty')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title('Normalized Values (should be comparable scale)')
        ax.set_xlabel('Episode')
        ax.legend()
        
        # Blended
        ax = axes[1, 1]
        ax.plot(episodes, self.diagnostics['blended_rewards'], alpha=0.7, label='Blended', color='green')
        ax.set_title(f'Blended Reward (α={self.alpha}, β={self.beta})')
        ax.set_xlabel('Episode')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[NS DIAGNOSTICS] Saved to {save_path}")


# ==============================================================================
# Factory function for easy switching
# ==============================================================================

def create_ns_wrapper(
    bd_fn: Callable[[Dict[str, Any]], np.ndarray],
    alpha: float = 0.5,
    beta: float = 0.5,
    normalized: bool = True,    # [NEW] Default to normalized version
    **kwargs
) -> NSWrapper:
    """
    ### [NEW FUNCTION] ###
    Factory function to create appropriate NS wrapper.
    
    Args:
        bd_fn: Behavior descriptor function
        alpha: Task reward weight
        beta: Novelty weight
        normalized: If True, use normalized wrapper (recommended)
        **kwargs: Additional arguments for wrapper
    
    Returns:
        NSWrapper or NSWrapperNormalized
    """
    if normalized:
        return NSWrapperNormalized(bd_fn, alpha, beta, **kwargs)
    else:
        return NSWrapper(bd_fn, alpha, beta, **kwargs)