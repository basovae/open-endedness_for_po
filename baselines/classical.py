# baselines/classical.py
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from core.interfaces import BaseStrategy, FitConfig


class EqualWeightStrategy(BaseStrategy):
    """
    Simple 1/N portfolio, rebalanced every step.
    This is a must-have baseline in almost every paper.
    """
    def fit(self, train_df: pd.DataFrame, cfg: FitConfig) -> None:
        self.n_assets = train_df.shape[1]
        # nothing to learn

    def predict_weights(self, hist_window: pd.DataFrame) -> np.ndarray:
        w = np.ones(self.n_assets, dtype=float)
        w = w / w.sum()
        return w


class MeanVarianceStrategy(BaseStrategy):
    """
    Mean-Variance / Markowitz-style.
    You can run it in different modes:
      - 'min_var': minimize variance subject to sum(w)=1
      - 'max_sharpe': maximize Sharpe using sample mean as expected return
    Assumes long-only-ish through post-processing (clip negatives to 0 and renormalize).
    """
    def __init__(self, mode: str = "min_var", ridge: float = 1e-4):
        self.mode = mode
        self.ridge = ridge
        self.w_opt = None

    def fit(self, train_df: pd.DataFrame, cfg: FitConfig) -> None:
        # train_df: rows = time, cols = assets, values = returns
        R = train_df.values  # shape (T, N)
        mu = R.mean(axis=0)  # expected returns (sample mean)
        cov = np.cov(R, rowvar=False)  # N x N
        # add small ridge to make it invertible / stable
        cov = cov + self.ridge * np.eye(cov.shape[0])
        inv_cov = pinv(cov)
        ones = np.ones_like(mu)

        if self.mode == "min_var":
            # classic minimum-variance: argmin w^T cov w  s.t. 1^T w = 1
            # closed form: w ~ inv_cov * 1
            raw_w = inv_cov @ ones
        elif self.mode == "max_sharpe":
            # approximate tangency: w ~ inv_cov * mu
            # (this ignores risk-free, so it's "max return per unit variance" proxy)
            raw_w = inv_cov @ mu
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # long-only cleanup: clip negatives, renormalize
        raw_w = np.clip(raw_w, 0, None)
        if raw_w.sum() == 0:
            # fallback if everything was clipped to 0
            raw_w = ones.copy()
        self.w_opt = raw_w / raw_w.sum()

    def predict_weights(self, hist_window: pd.DataFrame) -> np.ndarray:
        # weights are static over time after training
        return self.w_opt


class RiskParityStrategy(BaseStrategy):
    """
    Very common baseline: 'risk parity'.
    Simplest version here is inverse-volatility weighting.

    More advanced: equal risk contribution, solved by optimization.
    We start with inverse vol = 1/sigma, which is widely used and accepted in papers.
    """
    def fit(self, train_df: pd.DataFrame, cfg: FitConfig) -> None:
        # estimate each asset's volatility
        vol = train_df.std(axis=0).values  # shape (N,)
        inv_vol = 1.0 / np.where(vol == 0, 1e-8, vol)
        w = np.clip(inv_vol, 0, None)
        self.w_opt = w / w.sum()

    def predict_weights(self, hist_window: pd.DataFrame) -> np.ndarray:
        return self.w_opt
