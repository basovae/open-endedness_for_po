from rl_evaluator import RLEvaluator
from rl_dataloader import RLDataLoader
from ddpg_trainer import DDPGTrainer
import pandas as pd
import torch
import torch.nn as nn
import predictors


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
        use_ns: bool = False,
        ns_alpha: float = 1.0,
        ns_beta: float = 0.5,
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
        #number_of_datapoints = self.lookback_window + self.forecast_window
        #self.input_size = self.number_of_assets * number_of_datapoints
        self.input_size = self.number_of_assets * self.lookback_window
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
            # NS
            use_ns=use_ns,
            ns_alpha=ns_alpha,
            ns_beta=ns_beta,
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