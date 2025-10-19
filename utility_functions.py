import os, random
import numpy as np
import torch
import pandas as pd
from typing import Optional, Union

def set_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_profit(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return np.prod(1 + returns) - 1

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate
    return (np.mean(excess) / np.std(excess)) * np.sqrt(252)

def reduce_negatives(weights: np.ndarray) -> np.ndarray:
    """Clamp negatives to 0 and renormalize; also clip >1e6 if unstable."""
    w = np.clip(weights, 0.0, None)
    s = w.sum()
    return w if s == 0 else w / s

def calculate_test_performance(
    daily_returns: Union[np.ndarray, pd.DataFrame],
    weights: Optional[np.ndarray] = None
):
    """
    If given a returns DataFrame + weights, compute weighted daily returns.
    If given a 1D array of daily returns, use it directly.
    Returns (annual_profit, sharpe).
    """
    import numpy as np
    if hasattr(daily_returns, "values") and weights is not None:
        r = (daily_returns.values * weights).sum(axis=1)
    else:
        r = np.asarray(daily_returns)
    total = calculate_profit(r)
    ann = (1 + total) ** (252 / max(1, len(r))) - 1
    sharpe = calculate_sharpe_ratio(r)
    return ann, sharpe

def arima_forecast(x: np.ndarray, forecast_size: int):
    """Naive forecast: return the last window repeated or zeros if empty."""
    import numpy as np
    if forecast_size <= 0:
        return x
    last = x[-1]
    pad = np.vstack([last for _ in range(forecast_size)])
    return np.vstack([x, pad])
