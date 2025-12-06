import torch
import torch.nn as nn
import pandas as pd
from utility_functions import reduce_negatives, calculate_test_performance, arima_forecast
import numpy as np
from tqdm import tqdm as progress_bar



class RLEvaluator:
    '''Class to evaluate a PyTorch Actor model using SPO and DPO strategies.

    Args:
        actor (nn.Module or QNN): Trained Actor network, can be written in torch
            or qiskit.
        train_data (pd.DataFrame)): Train data for first step of evaluation.
        test_data (pd.DataFrame): Test data for evaluation and results computation.
    '''
    def __init__(
        self,
        actor: nn.Module,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        forecast_size: int,
        reduce_negatives: bool = True,
    ):
        self.actor = actor
        self.train_data = train_data
        self.test_data = test_data
        self.forecast_size = forecast_size
        self.reduce_negatives = reduce_negatives
        self.tickers = list(self.test_data.columns)
        self.window_size = len(self.train_data) 
        self.window_size = int(actor.input_size / len(self.tickers)) - forecast_size


    def evaluate_spo(self, verbose: int = 0) -> tuple:
        '''Evaluate using Static Portfolio Optimization (SPO).

        Args:
            verbose (int): Verbosity level for printing details.

        Returns:
            tuple: Total profit and Sharpe ratio from SPO strategy.
        '''
        self.actor.eval()  # putting actor in evaluation mode  

        # Use the last training batch to set initial portfolio allocation
        input_data = arima_forecast(
            self.train_data.tail(self.window_size).values,
            self.forecast_size)

        with torch.no_grad():   
            state = torch.tensor(input_data, dtype=torch.float32).flatten()
            portfolio_allocation = self.actor(state).numpy().squeeze()
        if self.reduce_negatives:
            portfolio_allocation = reduce_negatives(portfolio_allocation)

        # Calculate performance on test set
        avg_profit_pa, sharpe = calculate_test_performance(self.test_data,
                                                           portfolio_allocation)

        if verbose > 0:
            print('\nPortfolio Allocation (SPO):')
            for i in range(len(self.tickers)):
                print(f'{self.tickers[i]:<10} {(portfolio_allocation[i]*100):.2f} %')
            print(f'\nProfit p.a. (SPO): {avg_profit_pa*100:.4f} %')
            print(f'Sharpe Ratio (SPO): {sharpe:.4f}\n')

        return avg_profit_pa, sharpe

    def evaluate_dpo(self, interval: int, verbose: int = 0) -> tuple:
        '''Evaluate using Dynamic Portfolio Optimization (DPO).

        Args:
            verbose (int): Verbosity level for printing details.

        Returns:
            tuple: Total profit and Sharpe ratio from DPO strategy.

        Raises:
            ValueError if interval is shorter than actor input window.
        '''
        if interval < self.window_size:
            raise ValueError('DPO interval must be larger than actor input window size.')

        # Merge last batch of training data with the test data to ensure continuity
        test_data = pd.concat((self.train_data.tail(self.window_size), self.test_data))

        self.actor.eval()  # putting actor in evaluation mode

        # setting number of intervals for the dynamic optimization loop
        # adding +1 to ensure entire dataset is covered
        num_intervals = len(self.test_data) // interval + 1

        # empty array for daily portfolio returns of all intervals
        all_daily_portfolio_returns = np.array([])

        if verbose > 0:
            print(f'Performing dynamic portfolio optimization over {num_intervals} intervals...\n')
            wrapper = progress_bar
        else:
            wrapper = lambda x: x

        for i in wrapper(range(num_intervals)):

            # initialize rolling train and test data as consequtive chunks of window_size
            train_start_idx = i * interval
            test_start_idx = train_start_idx + interval
            test_end_idx = test_start_idx + interval

            rolling_train_data = test_data[train_start_idx:test_start_idx]
            rolling_test_data = test_data[test_start_idx:test_end_idx]

            # get portfolio allocation for the given interval based on rolling train dataset
            with torch.no_grad():
                input_data = torch.tensor(
                    arima_forecast(
                        rolling_train_data.tail(self.window_size).values,
                        self.forecast_size,
                    ), dtype=torch.float32)
                portfolio_allocation = self.actor(input_data.flatten()).numpy().squeeze()
            if self.reduce_negatives:
                portfolio_allocation = reduce_negatives(portfolio_allocation)

            if verbose > 1:
                print(f'\nPeriod {i+1} Portfolio Allocations:')
                for i in range(len(self.tickers)):
                    print(f'{self.tickers[i]:<10} {(portfolio_allocation[i]*100):.2f} %')
 
            # Get daily returns based on rolling test dataset
            daily_portfolio_returns = np.sum(rolling_test_data.values * portfolio_allocation, axis=1)
            all_daily_portfolio_returns = np.append(all_daily_portfolio_returns, daily_portfolio_returns)

        # Calculate profit and sharpe ratio
        profit, sharpe_ratio = calculate_test_performance(all_daily_portfolio_returns)

        # Summarize overall performance
        if verbose > 0:
            print(f'\nProfit p.a. (DPO): {profit * 100:.4f} %')
            print(f'Sharpe Ratio (DPO): {sharpe_ratio:.4f}\n')

        return profit, sharpe_ratio