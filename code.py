import os
import sys
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import predictors

from copy import deepcopy

from utilities.data_processing import RLDataLoader
from utilities.metrics import RLEvaluator
from utilities.model_training import EarlyStopper, ReplayBuffer, set_seeds


def progress_bar(
    iterable,
    length=None,
    bar_size=30,
    prefix='',
    suffix='',
    fill='█',
    print_end='\n',
):
    '''A custom progress bar function that overwrites itself in the terminal.

    Args:
        iterable (iterable): The iterable to loop over.
        length (int, optional): Length of the progress bar (in characters).
            Defaults to None (i.e. will be inferred from iterable).
        bar_size (int, optional): Length of the displayed bar. Defaults to 30 elements.
        prefix (str, optional): Prefix string to display before the progress bar.
        suffix (str, optional): Suffix string to display after the progress bar.
        fill (str, optional): Character to fill the progress bar with. Defaults to '█'.
        print_end (str, optional): End character (e.g., '\r' to overwrite,
            '\n' for new line). Defaults to '\n'.
        
    Yields:
        The elements from the provided iterable, one at a time.
    '''
    if length is None:
        length = len(iterable)

    def print_bar(progress, elapsed_time):
        percent = 100 * (progress / float(length))
        filled_length = int(bar_size * progress // length)
        bar = fill * filled_length + '-' * (bar_size - filled_length)
        sys.stdout.flush()
        sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix} [Elapsed: {format_timespan(elapsed_time)}]')

    print_bar(0, 0)

    start_time = time.time()

    for i, item in enumerate(iterable, 1):
        yield item
        elapsed = time.time() - start_time
        print_bar(i, elapsed)

    sys.stdout.write(print_end)


def calculate_test_performance(data: np.array, weights: list = None) -> tuple:
    '''
    Calculate the average yearly profit and Sharpe ratio for a given dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing asset or portfolio returns.
            If asset returns, each column represents an asset, each row
            represents a daily return, and weights must be specified. If
            portfolio returns, no weights and a 1-d array must be passed.
        weights (list, optional): A list of weights for the assets. If provided,
            the function calculates the weighted portfolio return. Defaults to
            None, i.e. data is provided weighted.

    Returns:
        tuple: A tuple containing:
            - avg_profit_pa (float): The average yearly profit calculated based
                on daily returns.
            - sharpe_ratio (float): The Sharpe ratio of the portfolio or assets.

    Raises:
        ValueError: If weights are provided but do not match the number of columns
            in `data`.

    Notes:
        - Assumes 252 trading days in a year for annualizing the profit.
    '''
    if weights is not None:
        # Sum the weighted returns across assets to get portfolio return per day
        data = np.sum(data.values * weights, axis=1)

    # Calculate average yearly profit
    avg_profit_pa = ((1 + np.mean(data)) ** 252) - 1
    
    # Calculate the Sharpe ratio
    sharpe_ratio = sharpe_ratio_series(pd.Series(data))

    return avg_profit_pa, sharpe_ratio


class RLDataLoader:
    '''Loads custom PyTorch DataLoaders for Reinforcement Learning (RL) applications.

    Args:
        data_train (pd.DataFrame): DataFrame with train dataset in standard
            tabular format.
        data_test (pd.DataFrame): DataFrame with validation dataset.
        shuffle (bool, optional): set to True to have the data reshuffled at
            every epoch. Defaults to False.

    Returns:
        tuple: A tuple containing two DataLoader objects:
            - train_loader (DataLoader): DataLoader for the training dataset with
                pairs of sequential states of length `window_size`.
            - val_loader (DataLoader): DataLoader for the val dataset, which
                includes the last `window_size` days of training data
                concatenated with the val data.

    Notes:
        - The training dataset contains sequences derived solely from the
            training data.
        - The val dataset includes the last `window_size` days of training data
            to ensure continuity for the first sample.
    '''
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
        '''Custom PyTorch Dataset for Reinforcement Learning (RL) applications.

        This dataset generates samples consisting of pairs of sequential states: 
        - The `current state` represented by a window of historical data.
        - The `next state` represented by the subsequent window of data. 

        Each entry in the dataset corresponds to two consecutive windows of size `window_size`. 

        Args:
            df (pd.DataFrame): Input DataFrame with each column being a feature
                and each row being an instance or timepoint.
            window_size (int): Number of consecutive data points to consider for each state window.
        '''
        def __init__(self, df: pd.DataFrame, window_size: int, forecast_size: int):
            self.data = df.values
            self.window_size = window_size
            self.forecast_size = forecast_size

        def __len__(self) -> int:
            # -1 to skip last window which isn't full length
            return len(self.data) // (self.window_size + self.forecast_size) - 1

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            train_start = idx*self.window_size
            train_end = train_start + self.window_size
            test_start = train_end
            test_end = train_end + self.window_size + self.forecast_size
            
            input_data = self.data[train_start:train_end]
            if self.forecast_size > 0:
                input_data = arima_forecast(input_data, self.forecast_size)
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
        '''Returns DataLoader objects for the training and validation datasets.
        
        Args:
            batch_size (int): Number of samples per batch.
            window_size (int): Number of consecutive data points to consider for
                each state window.
            forecast_size (int, optional): Number of days to forecast using ARIMA.
                Defaults to 0.
        '''
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
        )) # add last window from the train data to ensure continuity
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
        '''Returns the number of financial assets in the dataset.'''
        return self.data_train.shape[1]


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
        self.window_size = int(actor.input_size / len(tickers)) - forecast_size
        self.forecast_size = forecast_size
        self.reduce_negatives = reduce_negatives

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
            for i in range(len(tickers)):
                print(f'{tickers[i]:<10} {(portfolio_allocation[i]*100):.2f} %')
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
                for i in range(len(tickers)):
                    print(f'{tickers[i]:<10} {(portfolio_allocation[i]*100):.2f} %')
 
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


class DDPG:
    '''Meta-class for Deep Determinitic Policy Gradient Reinforcement Learning.

    Args:
        lookback_window (int): The size of the lookback window for input data.
        predictor (predictors): The predictor class to use for the model.
        batch_size (int, optional): The number of samples per batch. Defaults
            to 1.
        short_selling (bool, optional): Whether to allow short selling, i.e.
            negative portfolio weights in the model. Defaults to False.
        forecast_window (int, optional): The size of the forecast window for
            input data. Defaults to 0.
        reduce_negatives (bool, optional): Whether to clamp negative portfolio
            weights to -100 %. Defaults to False.
        verbose (int, optional): The verbosity level for logging and outputs.
            Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        **kwargs: Keyword arguments to be passed to the predictor at init.
    '''
    def __init__(
        self,
        lookback_window: int,
        predictor: predictors,
        batch_size: int = 1,
        short_selling: bool = False,
        forecast_window: int = 0,
        reduce_negatives: bool = False,
        verbose: int = 1,
        seed: int = 42,
        **kwargs,
    ):
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        self.predictor = predictor
        self.predictor_kwargs = kwargs
        self.short_selling = short_selling
        self.forecast_window = forecast_window
        self.reduce_negatives = reduce_negatives
        self.verbose = verbose
        self.seed = seed

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        actor_lr: float = 0.05,
        critic_lr: float = 0.01,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = False,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        gamma: float = 1.0,
        num_epochs: int = 50,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 0,
    ):
        '''Trains the DDPG model.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            actor_lr (float): Learning rate for the actor model.
            critic_lr (float): Learning rate for the critic model.
            optimizer (torch.optim): Optimizer class for updating model weights.
            l1_lambda (float): L1 regularization parameter for the actor model.
            l2_lambda (float): L2 regularization parameter for the actor model.
            weight_decay (float): Regularization parameter for weight decay.
            soft_update (bool): Whether to use soft updates for target networks.
            tau (float): Soft update factor for target networks.
            risk_preference (float): Risk preference factor for the reward function.
            gamma (float): Discount factor for future rewards.
            num_epochs (int): Number of training epochs.
            patience (int): Early stopping patience.
            min_delta (float, optional): Minimum change in loss for early stopping.

        Returns:
            Actor_Critic_RL_Model: The trained Actor-Critic RL model instance.
        '''
        self.val_data = val_data
        dataloader = RLDataLoader(train_data, val_data, shuffle=False)

        # set data-related hyperparameters
        self.number_of_assets = dataloader.number_of_assets
        number_of_datapoints = self.lookback_window + self.forecast_window
        self.input_size = self.number_of_assets * number_of_datapoints
        self.output_size = self.number_of_assets

        # build dataloaders
        train_loader, val_loader = dataloader(
            batch_size=self.batch_size,
            window_size=self.lookback_window,
            forecast_size=self.forecast_window,
        )

        # initialize models
        if self.short_selling:
            activation = lambda x: x / torch.sum(x, dim=-1, keepdim=True)
        else:
            activation = nn.Softmax(dim=-1)
        self.actor = self.predictor(
            input_size=self.input_size,
            output_size=self.output_size,
            output_activation=activation,
            **self.predictor_kwargs,
            seed=self.seed,
        )
        critic = self.predictor(
            input_size=self.input_size + self.output_size,
            output_size=1,
            **self.predictor_kwargs,
            seed=self.seed,
        )

        # run training loop
        trainer = DDPGTrainer(
            number_of_assets=self.number_of_assets,
            actor=self.actor,
            critic=critic,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            optimizer=optimizer,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            weight_decay=weight_decay,
            soft_update=soft_update,
            tau=tau,
            risk_preference=risk_preference,
            gamma=gamma,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=self.verbose,
            num_epochs=num_epochs,
        )

        return self

    def evaluate(self, test_data: pd.DataFrame, dpo: bool = True) -> tuple:
        '''Evaluates the DDPG model.

        Args:
            test_data (pd.DataFrame): Test dataset.
            dpo (bool, optional): Whether to evaluate the Dynamic Portfolio
                Optimization (DPO) strategy. Defaults to True.

        Returns:
            tuple: ((SPO profit, SPO sharpe ratio),
                    (DPO profit, DPO sharpe ratio))
                    if dpo is True, else (SPO profit, SPO sharpe ratio)
        '''
        evaluator = RLEvaluator(
            actor=self.actor,
            train_data=self.val_data,
            test_data=test_data,
            forecast_size=self.forecast_window,
            reduce_negatives=self.reduce_negatives,
        )
        spo_results = evaluator.evaluate_spo(verbose=self.verbose)
        if dpo:
            dpo_results = evaluator.evaluate_dpo(interval=self.lookback_window,
                                                verbose=self.verbose)
            return spo_results, dpo_results
        else:
            return spo_results


class DDPGTrainer:
    '''Facilitates the training of a DDPG pipeline for financial portfolio
    optimization, focusing on optimizing the actor network (decision-making)
    and the critic network (evaluation) using provided optimizers, loss
    functions, and hyperparameters. It supports early stopping to prevent
    overfitting based on validation performance.

    Args:
        number_of_assets (int): The number of assets in the portfolio, used for
            constructing the actor and critic models appropriately.
        actor (nn.Module): The actor network responsible for making portfolio
            allocation decisions.
        critic (nn.Module): The critic network responsible for evaluating the
            actor's decisions.
        optimizer (torch.optim, optional): Optimizer class (e.g., torch.optim.Adam
            or torch.optim.SGD) used to optimize both actor and critic networks.
            Defaults to torch.optim.Adam.
        weight_decay (float, optional): L2 regularization factor applied during
            optimization. Defaults to 0 (no weight decay).
        l1_lambda (float, optional): L1 regularization factor applied to the
            actor network. Defaults to 0 (no L1 regularization).
        l2_lambda (float, optional): L2 regularization factor applied to the
            actor network. Defaults to 0 (no L2 regularization).
        soft_update (bool, optional): Whether to use soft updates with target
            networks.
        tau (float, optional): Soft update factor for target networks. Defaults
            to 0.005.
        risk_preference (float, optional): Risk preference factor for the reward
            function. Negative value results in volatility lowering the reward.
            Defaults to -0.5.
        gamma (float, optional): Discount factor for future rewards in
            reinforcement learning. Defaults to 1.0.
        actor_lr (float, optional): Learning rate for the actor optimizer.
            Defaults to 0.05.
        critic_lr (float, optional): Learning rate for the critic optimizer.
            Defaults to 0.01.
        early_stopping (bool, optional): Whether to use early stopping based on
            validation performance. Defaults to True.
        patience (int, optional): Number of epochs to wait for improvement in
            validation performance before stopping training. Used for early
            stopping. Defaults to 2.
        min_delta (float, optional): Minimum change in validation performance to
            be considered as an improvement. Defaults to 0.
    '''
    def __init__(
        self,
        number_of_assets: int,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 0.05,
        critic_lr: float = 0.01,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = False,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        gamma: float = 1.0,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 0,
    ):
        self.number_of_assets = number_of_assets
        self.actor = actor
        self.critic = critic
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.soft_update = soft_update
        self.tau = tau
        self.risk_preference = risk_preference
        self.gamma = gamma
        self.early_stopper = EarlyStopper(patience, min_delta) if early_stopping else None

        self.actor_optimizer = optimizer(
            actor.parameters(),
            lr=actor_lr,
            # weight_decay=weight_decay,  # weight decay optional for actor
        )
        self.critic_optimizer = optimizer(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
        )
    
        if soft_update:
            # Initialize update factor and target networks
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            # Synchronize target networks with main networks
            self._soft_update(self.target_actor, self.actor, tau=1.0)
            self._soft_update(self.target_critic, self.critic, tau=1.0)

    def _soft_update(self, target, source, tau):
        '''Soft-update target network parameters.'''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        noise: float = 0.2,
        verbose: int = 1,
    ):
        '''Training loop for the DDPG pipeline.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader providing
                training data in batches.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader
                providing validation data in batches. Used for early stopping
                and performance monitoring. Defaults to None.
            num_epochs (int, optional): Number of training epochs to run.
                Defaults to 100.
            noise (float, optional): Standard deviation of Gaussian noise added
                to the actor's portfolio allocation. Defaults to 0.2.
            verbose (int, optional): Verbosity level for printing training
                details, can be 0, 1, or 2. Defaults to 1.
        '''
        replay_buffer = ReplayBuffer()
        
        for epoch in range(num_epochs):
            total_actor_loss = 0
            total_critic_loss = 0

            for state, next_state in train_loader:

                # Compute current portfolio allocation and Q-value
                portfolio_allocation = self.actor(state.flatten())
                exploration_noise = torch.normal(0, noise, portfolio_allocation.shape)
                noisy_portfolio_allocation = portfolio_allocation + exploration_noise

                # Set target value = average profit + risk preference * volatility
                avg_profit = torch.mean(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1)
                ).detach().cpu()
                volatility = torch.std(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1),
                    correction=0, # maximum likelihood estimation
                ).detach().cpu()
                reward = avg_profit + self.risk_preference * volatility

                # Store transition in replay buffer
                replay_buffer.push((
                    state.detach(),
                    noisy_portfolio_allocation.detach(),
                    reward.detach(),
                    next_state.detach()))

                # Sample transition from replay buffer
                transition = replay_buffer.sample(1)
                state = transition[0][0]
                noisy_portfolio_allocation = transition[0][1]
                reward = transition[0][2]
                next_state = transition[0][3]

                portfolio_allocation = self.actor(state.flatten())

                # Use target networks for next state action and Q-value if soft
                # updates are enabled, else use regular ones
                if self.soft_update:
                    next_portfolio_allocation = self.target_actor(next_state.flatten())
                    next_q_value = self.target_critic(
                        torch.cat((next_state.flatten(),
                                next_portfolio_allocation.flatten()))
                    )
                else:
                    next_portfolio_allocation = self.actor(next_state.flatten())
                    next_q_value = self.critic(
                        torch.cat((next_state.flatten(),
                                   next_portfolio_allocation.flatten()))
                    )

                # Calculate target Q-value according to update function
                target_q_value = reward + self.gamma * next_q_value

                # Critic loss and backpropagation
                q_value = self.critic(
                    torch.cat((state.flatten(),
                               noisy_portfolio_allocation.flatten()))
                )
                critic_loss = (target_q_value - q_value).pow(2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # Actor evaluation
                critic_input = torch.cat(
                    (state.flatten(), portfolio_allocation.flatten()))
                actor_loss = -self.critic(critic_input)

                # Add L1/L2 regularization to actor loss
                l1_actor = sum(weight.abs().sum() for weight in self.actor.parameters())
                l2_actor = sum(weight.pow(2).sum() for weight in self.actor.parameters())
                actor_loss += self.l1_lambda * l1_actor + self.l2_lambda * l2_actor

                # Actor backpropagation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

            # Average losses
            avg_actor_loss = total_actor_loss / len(train_loader)
            avg_critic_loss = total_critic_loss / len(train_loader)

            # Early stopping
            if self.early_stopper:
                with torch.no_grad():
                    val_critic_loss = 0
                    for state, next_state in val_loader:
                        portfolio_allocation = self.actor(state.flatten())
                        q_value = self.critic(
                            torch.cat((state.flatten(), portfolio_allocation.flatten()))
                        )

                        if self.soft_update:
                            next_portfolio_allocation = self.target_actor(next_state.flatten())
                            next_q_value = self.target_critic(
                                torch.cat((next_state.flatten(), next_portfolio_allocation.flatten()))
                            )
                        else:
                            next_portfolio_allocation = self.actor(next_state.flatten())
                            next_q_value = self.critic(
                                torch.cat((next_state.flatten(), next_portfolio_allocation.flatten()))
                            )

                        avg_profit = torch.mean(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                                    dim=-1)
                        ).detach().cpu()
                        volatility = torch.std(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                                    dim=-1),
                            correction=0, # maximum likelihood estimation
                        ).detach().cpu()
                        reward = avg_profit + self.risk_preference * volatility

                        target_q_value = reward + self.gamma * next_q_value
                        val_critic_loss += (target_q_value - q_value).pow(2).item()

                    avg_val_critic_loss = val_critic_loss / len(val_loader)

                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}, Val Critic Loss: {avg_val_critic_loss:.10f}')

                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                        break
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}')

            # Synchronize target networks
            if self.soft_update:
                self._soft_update(self.target_actor, self.actor, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)


class DeepQLearning:
    '''Meta-class for the Deep Q-Learning model.

    Args:
        lookback_window (int): The size of the lookback window for input data.
        predictor (predictors): The predictor class to use for the model.
        batch_size (int, optional): The number of samples per batch. Defeualts
            to 1.
        short_selling (bool, optional): Whether to allow short selling, i.e.
            negative portfolio weights in the model. Defaults to False.
        forecast_window (int, optional): The size of the forecast window for
            input data. Defaults to 0.
        reduce_negatives (bool, optional): Whether to clamp negative portfolio
            weights to -100 %. Defaults to False.
        verbose (int, optional): The verbosity level for logging and outputs.
            Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        **kwargs: Keyword arguments to be passed to the predictor at init.
    '''
    def __init__(
        self,
        lookback_window: int,
        predictor: predictors,
        batch_size: int = 1,
        short_selling: bool = False,
        forecast_window: int = 0,
        reduce_negatives: bool = False,
        verbose: int = 1,
        seed: int = 42,
        **kwargs,
    ):
        self.lookback_window = lookback_window
        self.batch_size = batch_size
        self.predictor = predictor
        self.predictor_kwargs = kwargs
        self.short_selling = short_selling
        self.forecast_window = forecast_window
        self.reduce_negatives = reduce_negatives
        self.verbose = verbose
        self.seed = seed

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        optimizer: torch.optim = torch.optim.Adam,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        weight_decay: float = 0,
        soft_update: bool = True,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        gamma: float = 0.99,
        num_epochs: int = 50,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 0,
        num_action_samples: int = 10,
    ):
        '''Trains the Deep Q-Learning model.

        Args:
            train_data (pd.DataFrame): Training dataset.
            val_data (pd.DataFrame): Validation dataset.
            actor_lr (float): Learning rate for the actor model.
            critic_lr (float): Learning rate for the critic model.
            optimizer (torch.optim): Optimizer class for updating model weights.
            l1_lambda (float): L1 regularization parameter for the actor model.
            l2_lambda (float): L2 regularization parameter for the actor model.
            weight_decay (float): Regularization parameter for weight decay.
            soft_update (bool): Whether to use soft updates for target networks.
            tau (float): Soft update factor for target networks.
            risk_preference (float): Risk preference factor for the reward function.
            gamma (float): Discount factor for future rewards.
            num_epochs (int): Number of training epochs.
            early_stopping (bool): Whether to use early stopping based on
                validation loss.
            patience (int): Early stopping patience.
            min_delta (float): Minimum change in loss for early stopping.
            num_action_samples (int): Number of actions to sample for
                determining the best next Q-value (Q_max).

        Returns:
            DeepQLearningModel: The trained Q-Learning model instance.
        '''
        self.val_data = val_data
        dataloader = RLDataLoader(train_data, val_data, shuffle=False)

        # Set data-related hyperparameters
        self.number_of_assets = dataloader.number_of_assets
        number_of_datapoints = self.lookback_window + self.forecast_window
        self.input_size = self.number_of_assets * number_of_datapoints
        self.output_size = self.number_of_assets

        # Build dataloaders
        train_loader, val_loader = dataloader(
            batch_size=self.batch_size,
            window_size=self.lookback_window,
            forecast_size=self.forecast_window,
        )

        # Initialize models
        if self.short_selling:
            activation = lambda x: x / torch.sum(x, dim=-1, keepdim=True)
        else:
            activation = nn.Softmax(dim=-1)
        self.actor = self.predictor(
            input_size=self.input_size,
            output_size=self.output_size,
            output_activation=activation,
            **self.predictor_kwargs,
            seed=self.seed,
        )
        critic = self.predictor(
            input_size=self.input_size + self.output_size,
            output_size=1,
            **self.predictor_kwargs,
            seed=self.seed,
        )

        # Initialize trainer
        trainer = DeepQLearningTrainer(
            number_of_assets=self.number_of_assets,
            actor=self.actor,
            critic=critic,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            optimizer=optimizer,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            soft_update=soft_update,
            tau=tau,
            risk_preference=risk_preference,
            weight_decay=weight_decay,
            gamma=gamma,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            num_action_samples=num_action_samples,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            verbose=self.verbose,
            num_epochs=num_epochs,
        )

        return self
    
    def evaluate(self, test_data: pd.DataFrame, dpo: bool = True) -> tuple:
        '''Evaluates the Deep Q-Learning model.

        Args:
            test_data (pd.DataFrame): Test dataset.
            dpo (bool, optional): Whether to evaluate the Dynamic Portfolio
                Optimization (DPO) strategy. Defaults to True.

        Returns:
            tuple: ((SPO profit, SPO sharpe ratio),
                    (DPO profit, DPO sharpe ratio))
                    if dpo is True, else (SPO profit, SPO sharpe ratio)
        '''
        evaluator = RLEvaluator(
            actor=self.actor,
            train_data=self.val_data,
            test_data=test_data,
            forecast_size=self.forecast_window,
            reduce_negatives=self.reduce_negatives,
        )
        spo_results = evaluator.evaluate_spo(verbose=self.verbose)
        if dpo:
            dpo_results = evaluator.evaluate_dpo(interval=self.lookback_window,
                                                verbose=self.verbose)
            return spo_results, dpo_results
        else:
            return spo_results


class DeepQLearningTrainer:
    '''Facilitates the training of a Deep Q-Learning model for financial portfolio
    optimization, focusing on optimizing the critic network (evaluation) using
    a target network and properly computing the maximum Q-value for updates.

    Args:
        number_of_assets (int): The number of assets in the portfolio, used for
            constructing the actor and critic models appropriately.
        actor (nn.Module): The actor network responsible for making portfolio
            allocation decisions.
        critic (nn.Module): The critic network responsible for evaluating the
            actor's decisions.
        optimizer (torch.optim, optional): Optimizer class (e.g., torch.optim.Adam).
            Defaults to torch.optim.Adam.
        gamma (float, optional): Discount factor for future rewards in
            reinforcement learning. Defaults to 0.99.
        actor_lr (float, optional): Learning rate for the actor optimizer.
            Defaults to 0.001.
        critic_lr (float, optional): Learning rate for the critic optimizer.
            Defaults to 0.001.
        weight_decay (float, optional): L2 regularization for critic network.
            Defaults to 0.
        l1_lambda (float, optional): L1 regularization factor applied to the
            actor network. Defaults to 0.
        l2_lambda (float, optional): L2 regularization factor applied to the
            actor network. Defaults to 0.
        soft_update (bool, optional): Whether to use soft updates with target
            networks.
        tau (float, optional): Soft update factor for target networks. Defaults
            to 0.005.
        risk_preference (float, optional): Risk preference factor for the reward
            function. Negative value results in volatility lowering the reward.
            Defaults to -0.5.
        early_stopping (bool, optional): Whether to use early stopping based on
            validation performance. Defaults to True.
        patience (int, optional): Number of epochs to wait for improvement in
            validation performance before stopping training. Used for early
            stopping. Defaults to 2.
        min_delta (float, optional): Minimum change in validation performance to
            be considered as an improvement. Defaults to 0.
        num_action_samples (int, optional): Number of actions to sample for
            determining the best next Q-value (Q_max).
    '''

    def __init__(
        self,
        number_of_assets: int,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 0.001,
        critic_lr: float = 0.001,
        optimizer: torch.optim = torch.optim.Adam,
        gamma: float = 0.99,
        l1_lambda: float = 0,
        l2_lambda: float = 0,
        soft_update: bool = True,
        tau: float = 0.005,
        risk_preference: float = -0.5,
        weight_decay: float = 0,
        early_stopping: bool = True,
        patience: int = 2,
        min_delta: float = 0,
        num_action_samples: int = 10,
    ):
        self.number_of_assets = number_of_assets
        self.actor = actor
        self.critic = critic
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.soft_update = soft_update
        self.tau = tau
        self.risk_preference = risk_preference
        self.gamma = gamma
        self.early_stopper = EarlyStopper(patience, min_delta) if early_stopping else None
        self.num_action_samples = num_action_samples

        # Optimizers
        self.actor_optimizer = optimizer(
            actor.parameters(),
            lr=actor_lr,
            # weight_decay=weight_decay,  # weight decay optional for actor
        )
        self.critic_optimizer = optimizer(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay
        )

        if soft_update:
            # Initialize update factor and target critic
            self.target_critic = deepcopy(critic)
            # Synchronize target critic with main critic network
            self._soft_update(self.target_critic, self.critic, tau=1.0)

    def _soft_update(self, target, source, tau):
        '''Soft-update target network parameters.'''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        noise: float = 0.2,
        verbose: int = 1,
    ):
        '''Training loop for Deep Q-Learning.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader providing
                training data in batches.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader
                providing validation data in batches. Used for early stopping
                and performance monitoring. Defaults to None.
            num_epochs (int, optional): Number of training epochs to run.
                Defaults to 100.
            noise (float, optional): Standard deviation of Gaussian noise added
                to the actor's portfolio allocation. Defaults to 0.2.
            verbose (int, optional): Verbosity level for printing training
                details, can be 0, 1, or 2. Defaults to 1.
        '''        
        replay_buffer = ReplayBuffer()

        for epoch in range(num_epochs):
            total_actor_loss = 0
            total_critic_loss = 0

            for state, next_state in train_loader:

                # Compute current portfolio allocation and Q-value
                portfolio_allocation = self.actor(state.flatten())
                exploration_noise = torch.normal(0, noise, portfolio_allocation.shape)
                noisy_portfolio_allocation = portfolio_allocation + exploration_noise

                # Set target value = average profit + risk preference * volatility
                avg_profit = torch.mean(
                    torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                              dim=-1)
                ).detach().cpu()
                volatility = torch.std(
                    torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                              dim=-1),
                    correction=0, # maximum likelihood estimation
                ).detach().cpu()
                reward = avg_profit + self.risk_preference * volatility

                # Store transition in replay buffer
                replay_buffer.push((
                    state.detach(),
                    noisy_portfolio_allocation.detach(),
                    reward.detach(),
                    next_state.detach()))
                
                # Sample transition from replay buffer
                transition = replay_buffer.sample(1)
                state = transition[0][0]
                noisy_portfolio_allocation = transition[0][1]
                reward = transition[0][2]
                next_state = transition[0][3]

                portfolio_allocation = self.actor(state.flatten())

                # Sample multiple actions uniformly between 0 and 1 and compute
                # Q-values to find the best next Q-value
                sampled_actions = torch.rand((self.num_action_samples, self.actor.output_size))
                if self.soft_update:
                    q_values_for_sampled_actions = [
                        self.target_critic(
                            torch.cat((next_state.flatten(), action.flatten()))
                        ) for action in sampled_actions
                    ]
                else:
                    q_values_for_sampled_actions = [
                        self.critic(
                            torch.cat((next_state.flatten(), action.flatten()))
                        ) for action in sampled_actions
                    ]
                max_q_value = torch.max(torch.stack(q_values_for_sampled_actions))

                # Calculate target Q-value according to update function
                target_q_value = reward + self.gamma * max_q_value

                # Critic loss and backpropagation
                q_value = self.critic(
                    torch.cat((state.flatten(),
                               portfolio_allocation.flatten()))
                )
                critic_loss = (target_q_value - q_value).pow(2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # Actor loss and backpropagation
                critic_input = torch.cat(
                    (state.flatten(), portfolio_allocation.flatten()))
                actor_loss = -self.critic(critic_input)

                # Add L1/L2 regularization to actor loss
                l1_actor = sum(weight.abs().sum() for weight in self.actor.parameters())
                l2_actor = sum(weight.pow(2).sum() for weight in self.actor.parameters())
                actor_loss += self.l1_lambda * l1_actor + self.l2_lambda * l2_actor

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

            # Average losses
            avg_actor_loss = total_actor_loss / len(train_loader)
            avg_critic_loss = total_critic_loss / len(train_loader)

            # Early stopping
            if self.early_stopper:
                with torch.no_grad():
                    val_critic_loss = 0
                    for state, next_state in val_loader:
                        portfolio_allocation = self.actor(state.flatten())
                        q_value = self.critic(
                            torch.cat((state.flatten(), portfolio_allocation.flatten()))
                        )

                        # Sample actions uniformly between 0 and 1
                        sampled_actions = torch.rand((self.num_action_samples,
                                                      self.actor.output_size))
                        if self.soft_update:
                            q_values_for_sampled_actions = [
                                self.target_critic(
                                    torch.cat((next_state.flatten(), action.flatten()))
                                ) for action in sampled_actions
                            ]
                        else:
                            q_values_for_sampled_actions = [
                                self.critic(
                                    torch.cat((next_state.flatten(), action.flatten()))
                                ) for action in sampled_actions
                            ]
                        max_q_value = torch.max(torch.stack(q_values_for_sampled_actions))

                        avg_profit = torch.mean(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                                    dim=-1)
                        ).detach().cpu()
                        volatility = torch.std(
                            torch.sum(state.view(-1, self.number_of_assets) * portfolio_allocation,
                                    dim=-1),
                            correction=0, # maximum likelihood estimation
                        ).detach().cpu()
                        reward = avg_profit + self.risk_preference * volatility

                        target_q_value = reward + self.gamma * max_q_value
                        val_critic_loss += (target_q_value - q_value).pow(2).item()

                    avg_val_critic_loss = val_critic_loss / len(val_loader)

                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}, Val Critic Loss: {avg_val_critic_loss:.10f}')

                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                        break
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}')

            # Synchronize target networks
            if self.soft_update:
                self._soft_update(self.target_critic, self.critic, self.tau)


def set_seeds(seed: int):
    '''Set seeds for reproducibility.'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class EarlyStopper:
    '''Class to monitor validation loss during training and stop the training process 
    early if no improvement is observed for a specified number of epochs (patience).

    Args:
        patience (int): Number of consecutive epochs with no improvement after which
            training should be stopped. Default is 1.
        min_delta (float): Minimum change in the validation loss to be considered as 
            an improvement. Default is 0.
    '''
    def __init__(
        self,
        patience: int = 1,
        min_delta: float = 0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def early_stop(self, loss, verbose: bool = True):
        '''Check if training should be stopped early based on the validation loss.
        
        Args:
            loss (float): Current validation loss.
            verbose (bool): Verbosity level. Prints early stopping message if True,
                prints nothing if False. Defaults to True.
        '''
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss >= (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if verbose:
                    print(f'Early stopping triggered.')
                return True
        return False


class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: tuple):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class NeuralNetwork(nn.Module):
    '''Base class for a vanilla neural network / multilayer perceptron.

    Args:
        input_size (int): Size of the input layer.
        hidden_sizes (list of int): List of sizes for the hidden layers.
        output_size (int): Size of the output layer.
        hidden_activation: Activation function for the hidden layers. Can be
            any function R -> R. Defaults to ReLU.
        output_activation: Activation function for the output layer. Can be any
            function R -> R. Defaults to None, i.e. linear activation.
        seed (int): Random seed for reproducibility. Defaults to None, i.e.
            non-deterministic behaviour.
    '''
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        hidden_activation: callable = torch.relu,
        output_activation: callable = None,
        seed: int = None,
    ):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if seed:
            set_seeds(seed)

        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the network.'''

        for layer in self.layers:
            x = self.hidden_activation(layer(x))
        if self.output_activation:
            return self.output_activation(self.output_layer(x))
        else:
            return self.output_layer(x)

