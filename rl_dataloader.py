import torch
import pandas as pd

class RLDataLoader:
    '''Loads custom PyTorch DataLoaders for Reinforcement Learning (RL) applications.'''
    def __init__(
        self,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        shuffle: bool = False,
    ):
        self.data_train = data_train
        self.data_val = data_val
        self.shuffle = shuffle

    class RLDataset(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame, window_size: int, forecast_size: int):
            self.data = df.values
            self.window_size = window_size
            self.forecast_size = forecast_size

        def __len__(self) -> int:
            return len(self.data) // (self.window_size + self.forecast_size) - 1

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            train_start = idx*self.window_size
            train_end = train_start + self.window_size
            test_start = train_end
            test_end = train_end + self.window_size + self.forecast_size
            
            input_data = self.data[train_start:train_end]
            # If you use arima_forecast, import it here
            input_window = torch.tensor(
                input_data,
                dtype=torch.float32,
            )
            target_window = torch.tensor(
                self.data[test_start:test_end],
                dtype=torch.float32,
            )
            return input_window, target_window

    def __call__(
        self,
        batch_size: int,
        window_size: int,
        forecast_size: int = 0,
    ):
        train_dataset = self.RLDataset(
            self.data_train,
            window_size=window_size,
            forecast_size=forecast_size,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
        )
        
        val_data = pd.concat((
            self.data_train[-window_size:],
            self.data_val
        ))
        val_dataset = self.RLDataset(
            val_data,
            window_size=window_size,
            forecast_size=forecast_size,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
        )
        
        return train_loader, val_loader
    
    @property
    def number_of_assets(self):
        return self.data_train.shape[1]