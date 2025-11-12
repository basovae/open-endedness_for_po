# qd/bd_presets.py
import numpy as np
from .novelty_metrics import bd_weights_hist, bd_returns_shape, concat

def bd_weights_plus_returns(traj) -> np.ndarray:
    """
    traj: dict filled by NSWrapper:
      - traj["weights_traj"]: (T, n_assets)
      - traj["returns"]: (T,)
    """
    if "weights_traj" not in traj or traj["weights_traj"].size == 0:
        # fallback: use returns only
        return bd_returns_shape(np.asarray(traj.get("returns", []), dtype=float))
    W = np.asarray(traj["weights_traj"], dtype=float)
    R = np.asarray(traj.get("returns", []), dtype=float)
    return concat(bd_weights_hist(W, bins=5), bd_returns_shape(R, segments=10))
