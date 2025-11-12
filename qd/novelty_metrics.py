# qd/novelty_metrics.py
import numpy as np

def bd_weights_hist(weights_traj: np.ndarray, bins: int = 5) -> np.ndarray:
    """
    Behavior descriptor from the distribution of portfolio weights over an episode.
    weights_traj: shape (T, n_assets) with weights at each step, rows sum≈1.
    Returns: fixed-length vector (n_assets * bins).
    """
    T, n_assets = weights_traj.shape
    edges = np.linspace(0.0, 1.0, bins + 1)
    hists = []
    for a in range(n_assets):
        hist, _ = np.histogram(weights_traj[:, a], bins=edges, density=True)
        hists.append(hist)
    bd = np.concatenate(hists)
    # L2 normalize for scale stability
    norm = np.linalg.norm(bd) + 1e-8
    return bd / norm

def bd_returns_shape(episode_returns: np.ndarray, segments: int = 10) -> np.ndarray:
    """
    Descriptor from the shape of cumulative returns across the episode.
    Returns: normalized segment means of cumret deltas.
    """
    cum = np.cumprod(1.0 + episode_returns)  # start at 1
    # piecewise-aggregate approximation
    cut = np.linspace(0, len(cum), segments + 1, dtype=int)
    segs = []
    for i in range(segments):
        s, e = cut[i], cut[i+1]
        if e <= s: segs.append(0.0); continue
        segs.append(float(np.mean(np.diff(cum[s:e])) if e - s > 1 else 0.0))
    segs = np.array(segs, dtype=float)
    norm = np.linalg.norm(segs) + 1e-8
    return segs / norm

def concat(*vecs: np.ndarray) -> np.ndarray:
    v = np.concatenate([np.asarray(x, dtype=float).ravel() for x in vecs])
    n = np.linalg.norm(v) + 1e-8
    return v / n
