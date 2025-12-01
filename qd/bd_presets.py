# qd/bd_presets.py
import numpy as np
from .novelty_metrics import bd_weights_hist, bd_returns_shape, concat


def bd_weights_plus_returns(traj) -> np.ndarray:
    """
    Combined behavior descriptor using weights distribution + return shape.
    
    traj: dict filled by NSWrapper:
      - traj["weights_traj"]: (T, n_assets) or list of weight arrays
      - traj["returns"]: (T,) or list of floats
    
    Returns: normalized concatenation of weight histogram BD and returns shape BD
    """
    # Handle case where weights_traj might be missing or empty
    if "weights_traj" not in traj or len(traj.get("weights_traj", [])) == 0:
        # Fallback: use returns only
        returns = np.asarray(traj.get("returns", []), dtype=float)
        if len(returns) == 0:
            return np.zeros(10)  # Return zero vector if no data
        return bd_returns_shape(returns, segments=10)
    
    W = np.asarray(traj["weights_traj"], dtype=float)
    R = np.asarray(traj.get("returns", []), dtype=float)
    
    # Ensure W is 2D
    if W.ndim == 1:
        W = W.reshape(1, -1)
    
    # Handle empty returns
    if len(R) == 0:
        R = np.zeros(W.shape[0])
    
    return concat(bd_weights_hist(W, bins=5), bd_returns_shape(R, segments=10))


def bd_for_map_elites(traj) -> np.ndarray:
    """
    Returns 2D descriptor for MAP-Elites grid:
      [0]: Portfolio concentration (Herfindahl index of mean weights)
      [1]: Realized volatility of portfolio returns
    
    These two dimensions define the behavioral space:
      - HHI: 1/n (fully diversified) to 1.0 (fully concentrated)
      - Volatility: 0 (no variance) to unbounded (high risk)
    """
    weights = np.asarray(traj["weights_traj"], dtype=float)
    returns = np.asarray(traj["returns"], dtype=float)
    
    # Ensure weights is 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Handle empty inputs
    if weights.size == 0:
        return np.array([0.2, 0.0], dtype=float)  # Default: diversified, zero vol
    
    # Concentration: HHI of average weights (0 = diversified, 1 = concentrated)
    mean_weights = np.mean(weights, axis=0)
    # Normalize to ensure sum to 1 (in case of numerical issues)
    mean_weights = mean_weights / (np.sum(mean_weights) + 1e-8)
    hhi = np.sum(mean_weights ** 2)
    
    # Volatility
    vol = np.std(returns) if len(returns) > 1 else 0.0
    
    return np.array([hhi, vol], dtype=float)


def bd_with_market_context(traj) -> np.ndarray:
    """
    Extended behavior descriptor that includes market context.
    
    Returns concatenation of:
      - Weight histogram BD
      - Returns shape BD  
      - Market volatility (scalar)
      - Market trend direction (scalar)
    
    This helps distinguish agents that behave similarly but in different
    market conditions.
    """
    weights = np.asarray(traj.get("weights_traj", []), dtype=float)
    returns = np.asarray(traj.get("returns", []), dtype=float)
    
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    if weights.size == 0 or len(returns) == 0:
        return np.zeros(27)  # 25 (weights) + 10 (returns) + 2 (context) normalized
    
    # Base BDs
    weights_bd = bd_weights_hist(weights, bins=5)
    returns_bd = bd_returns_shape(returns, segments=10)
    
    # Market context features
    market_vol = np.std(returns)
    market_trend = np.sign(np.mean(returns))  # -1, 0, or 1
    
    context = np.array([market_vol, market_trend], dtype=float)
    
    return concat(weights_bd, returns_bd, context)