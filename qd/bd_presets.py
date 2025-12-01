# qd/bd_presets.py
import numpy as np
from .novelty_metrics import bd_weights_hist, bd_returns_shape, concat

def bd_for_map_elites(traj) -> np.ndarray:
    """
    Returns 2D descriptor:
      [0]: Portfolio concentration (Herfindahl index of mean weights)
      [1]: Realized volatility of portfolio returns
    """
    weights = np.asarray(traj["weights_traj"], dtype=float)
    returns = np.asarray(traj["returns"], dtype=float)
    
    # Concentration: HHI of average weights (0 = diversified, 1 = concentrated)
    mean_weights = np.mean(weights, axis=0)
    hhi = np.sum(mean_weights ** 2)
    
    # Volatility
    vol = np.std(returns) if len(returns) > 1 else 0.0
    
    return np.array([hhi, vol], dtype=float)