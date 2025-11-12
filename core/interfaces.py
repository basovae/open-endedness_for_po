# core/interfaces.py
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np

@dataclass
class FitConfig:
    lookback: int                    # how many past days form the "state"
    transaction_cost_bps: float = 0  # you can use this later in eval
    seed: int = 42
    extra: Dict[str, Any] = None     # method-specific stuff if needed


class BaseStrategy:
    """
    All portfolio strategies must implement:
    - fit(train_df, cfg): learn parameters from training returns
    - predict_weights(hist_window): given last `lookback` days of returns,
                                    output weights for NEXT day.
    Returns must be a numpy array of shape (n_assets,)
    that sums to 1 (long-only, fully invested).
    """

    def fit(self, train_df: pd.DataFrame, cfg: FitConfig) -> None:
        raise NotImplementedError

    def predict_weights(self, hist_window: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        # optional for RL agents; here just a no-op
        pass
