# baselines/ml_predictive.py
import numpy as np
import pandas as pd
import torch
from core.interfaces import BaseStrategy, FitConfig
from predictors import MLP  # ✅ your existing one

class MLPBaseline(BaseStrategy):
    def __init__(self, input_size: int, n_assets: int, hidden_size: int = 128, epochs: int = 20, lr: float = 1e-3):
        self.model = MLP(input_size=input_size, output_size=n_assets, hidden_dim=hidden_size)
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cpu")
        self.lookback = None
        self.n_assets = n_assets

    def fit(self, train_df: pd.DataFrame, cfg: FitConfig):
        self.lookback = cfg.lookback
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        # Build X, y from rolling windows of returns
        X, y = [], []
        data = train_df.values
        for t in range(self.lookback, len(data) - 1):
            X.append(data[t - self.lookback:t].flatten())  # flatten lookback × assets
            y.append(data[t + 1])                         # next-day returns
        X = torch.tensor(np.stack(X), dtype=torch.float32)
        y = torch.tensor(np.stack(y), dtype=torch.float32)

        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

    def predict_weights(self, hist_window: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(hist_window.values.flatten(), dtype=torch.float32)[None, :]
        with torch.no_grad():
            preds = self.model(x).numpy().flatten()
        preds = np.clip(preds, 0, None)
        if preds.sum() == 0:
            preds = np.ones_like(preds)
        return preds / preds.sum()
