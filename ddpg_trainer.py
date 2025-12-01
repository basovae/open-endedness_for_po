from math import tau
from statistics import correlation
from rl_evaluator import RLEvaluator
from rl_dataloader import RLDataLoader
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
from early_stopper import EarlyStopper
from replay_buffer import ReplayBuffer
from qd.wrappers import NSWrapper # Novelty Search
from qd.bd_presets import bd_weights_plus_returns # Novelty Search

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
        use_ns: bool = False,
        ns_alpha: float = 1.0,
        ns_beta: float = 0.5,
    ):

        self.number_of_assets = number_of_assets      # Number of stocks in portfolio
        self.actor = actor                            # Policy network (decision maker)
        self.critic = critic                          # Value network (evaluator)
    
        # Regularization parameters
        self.l1_lambda = l1_lambda                    # L1 penalty for sparsity
        self.l2_lambda = l2_lambda                    # L2 penalty for weight decay
    
        # Target network parameters
        self.soft_update = soft_update                # Use target networks?
        self.tau = tau                                # Soft update rate (typically 0.001-0.01)
    
        # Reward function parameters
        self.risk_preference = risk_preference        # Negative = risk-averse (penalize volatility)
        self.gamma = gamma                            # Discount factor for future rewards
    
        # Early stopping to prevent overfitting
        self.early_stopper = EarlyStopper(patience, min_delta) if early_stopping else None

        # optimizers for Actor and Critic
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
            # Create frozen copies of networks for stable training
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            # Synchronize target networks with main networks
            # Initialize target networks to match main networks (tau=1.0 means complete copy)
            self._soft_update(self.target_actor, self.actor, tau=1.0)
            self._soft_update(self.target_critic, self.critic, tau=1.0)
        
        ## -- NS --
        self.use_ns = use_ns
        self.ns = NSWrapper(bd_fn=bd_weights_plus_returns,
                    alpha=ns_alpha,
                    beta=ns_beta) if use_ns else None


    def _soft_update(self, target, source, tau):
        '''Soft-update target network parameters.'''
        '''Polyak averaging: slowly blend target toward source'''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            # target = (1-τ)×target + τ×source
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
            episode_task_return = 0.0           # <-- track per-epoch reward for NS
            
            if self.use_ns:
                self.ns.reset_episode_buffers()  # <-- NS: start new episode buffers

            # batch loop starts here
            for state, next_state in train_loader:
                # state: Current market window (e.g., last 10 days of returns)
                # state: Current market window (tensor of shape [batch_size, window_size, num_assets])
                # next_state: Next market window (shifted by 1 day)


                # ========== ACTION SELECTION ==========
                # Actor predicts portfolio allocation based on current state
                # Compute current portfolio allocation and Q-value
                portfolio_allocation = self.actor(state.flatten())
                # Add exploration noise (encourage trying different allocations)
                exploration_noise = torch.normal(0, noise, portfolio_allocation.shape)
                noisy_portfolio_allocation = portfolio_allocation + exploration_noise



                # ========== REWARD COMPUTATION ==========
                # Calculate average profit from the portfolio allocation
                # Set target value = average profit + risk preference * volatility
                avg_profit = torch.mean(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1)
                ).detach().cpu()
                # Calculate volatility (risk) of the portfolio
                volatility = torch.std(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1),
                    correction=0, # maximum likelihood estimation
                ).detach().cpu()
                # Reward = Profit - Risk Penalty
                reward = avg_profit + self.risk_preference * volatility
                # avg_profit: Mean return of the portfolio over the window
                # volatility: Standard deviation of returns (risk measure)
                # risk_preference < 0: Penalize high volatility (risk-averse investor)
                # Example: If risk_preference = -0.5, a portfolio with 10% return and 2% volatility gets 
                # reward = 0.10 - 0.5×0.02 = 0.09

               
                # ---- NS step hook ----
                if self.use_ns:
                    info = {
                        "weights": noisy_portfolio_allocation.detach().cpu().numpy().ravel(),
                        "return_t": float(reward)
                    }
                    # state/action can be None; wrapper only needs weights/return_t
                    self.ns.on_step(state=None,
                                    action=noisy_portfolio_allocation.detach().cpu().numpy().ravel(),
                                    reward_task=float(reward),
                                    info=info)
                # accumulate episode score for the epoch
                episode_task_return += float(reward)






                # ========== EXPERIENCE REPLAY ==========
                # Store the transition (experience) in replay buffer for later learning
                replay_buffer.push((
                    state.detach(),
                    noisy_portfolio_allocation.detach(),
                    reward.detach(),
                    next_state.detach()))

                # Consecutive experiences are highly correlated (tomorrow's prices depend on today's)
                # Random sampling breaks correlation → more stable learning
                # Re-use past experiences → sample efficiency
                # Sample transition from replay buffer
                transition = replay_buffer.sample(1)
                state = transition[0][0]
                noisy_portfolio_allocation = transition[0][1]
                reward = transition[0][2]
                next_state = transition[0][3]






                # ========== CRITIC UPDATE - PART 1: COMPUTE TARGET ==========
                # Get the clean action (without noise) for the current state
                portfolio_allocation = self.actor(state.flatten())


                # Use target networks for next state action and Q-value if soft
                # updates are enabled, else use regular ones
                if self.soft_update:
                    # Use TARGET networks for stable Q-value estimation
                    next_portfolio_allocation = self.target_actor(next_state.flatten())
                    next_q_value = self.target_critic(
                        torch.cat((next_state.flatten(),
                                next_portfolio_allocation.flatten()))
                    )
                else:
                    # Use MAIN networks (less stable but simpler)
                    next_portfolio_allocation = self.actor(next_state.flatten())
                    next_q_value = self.critic(
                        torch.cat((next_state.flatten(),
                                   next_portfolio_allocation.flatten()))
                    )

                # Calculate target Q-value according to update function
                # Intuition: 
                # "The value of today's action equals today's reward plus the discounted value of tomorrow's best action"
                # Bellman equation: Q(s,a) = r + γ × Q(s',a')
                # Q(s,a): Expected total reward for taking action a in state s
                # reward: Immediate reward received
                # gamma × Q(s',a'): Discounted future reward from next state
                target_q_value = reward + self.gamma * next_q_value






                # ========== CRITIC UPDATE - PART 2: MINIMIZE TD ERROR ==========
                # Compute current Q-value estimate using the NOISY action
                # Critic predicts Q-value for the state-action pair
                q_value = self.critic(
                    torch.cat((state.flatten(),
                               noisy_portfolio_allocation.flatten()))
                )

                # Temporal Difference (TD) Error: difference between target and prediction
                critic_loss = (target_q_value - q_value).pow(2)

                # Backpropagation: Update Critic to minimize TD error
                # Minimize squared error → Critic learns to estimate rewards accurately
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True) # retain_graph=True needed for Actor update
                self.critic_optimizer.step()






                # ========== ACTOR UPDATE: MAXIMIZE Q-VALUE ==========
                # Goal: Actor should choose actions that the Critic thinks are valuable
                # Loss: -Q(s, a) means maximizing Q-value
                # Gradient flow: ∂(-Q)/∂θ_actor = -∂Q/∂a × ∂a/∂θ_actor
                # Change Actor parameters to produce actions that increase Q-value
                # This is the "deterministic policy gradient"
               
                # Compute Q-value using CLEAN action (no noise)
                critic_input = torch.cat(
                    (state.flatten(), portfolio_allocation.flatten()))
                #actor_loss = -self.critic(critic_input) # Negative because we want to MAXIMIZE Q

                novelty_bonus = self.ns.archive.novelty(current_behavior_descriptor)
                actor_loss = -self.critic(critic_input) - self.beta * novelty_bonus

                # Add L1/L2 regularization penalties to actor loss
                # L1 penalty: Encourages sparse portfolios (few non-zero weights)                
                l1_actor = sum(weight.abs().sum() for weight in self.actor.parameters())
                # L2 penalty: Prevents extreme weight values
                l2_actor = sum(weight.pow(2).sum() for weight in self.actor.parameters())
                actor_loss += self.l1_lambda * l1_actor + self.l2_lambda * l2_actor

                # Actor Backpropagation: Update Actor to choose actions that maximize Q-values
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

            # ← BATCH LOOP ENDS HERE

            
            

            # ========== AFTER ALL BATCHES ARE PROCESSED ==========
            # Why at the end of each epoch? Target networks should change slowly (stability)
            # Updating after every batch would be too frequent

            # First, accumulate losses from all batches
            avg_actor_loss = total_actor_loss / len(train_loader)
            avg_critic_loss = total_critic_loss / len(train_loader)


            # Then do validation (if early stopping is enabled)
            # ========== VALIDATION PHASE ==========
            if self.early_stopper:
                with torch.no_grad():  # No gradient computation during validation
                    val_critic_loss = 0

                    for state, next_state in val_loader:
                        # Compute validation loss same way as training
                        portfolio_allocation = self.actor(state.flatten())
                        q_value = self.critic(
                            torch.cat((state.flatten(), portfolio_allocation.flatten()))
                        )

                        # Use target networks if enabled
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

                        # Calculate reward on validation data
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
                        # Accumulate validation TD error
                        val_critic_loss += (target_q_value - q_value).pow(2).item()

                    avg_val_critic_loss = val_critic_loss / len(val_loader)

                if verbose > 0: 
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}, Val Critic Loss: {avg_val_critic_loss:.10f}')

                # ---- NS episode-end hook ----
                if self.use_ns:
                    blended = self.ns.on_episode_end(episode_task_return)
                    if verbose > 0:
                        print(f"NS blended score (epoch {epoch+1}): {blended:.6f} | "
                              f"task_return: {episode_task_return:.6f}")

                # Check if we should stop training
                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                        break
                if self.soft_update:
                    self._soft_update(self.target_critic, self.critic, self.tau)
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Actor Loss: {avg_actor_loss:.10f}, Critic Loss: {avg_critic_loss:.10f}')

            # ========== TARGET NETWORK UPDATE ==========
            # Synchronize target networks 
            if self.soft_update:
                self._soft_update(self.target_actor, self.actor, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)
