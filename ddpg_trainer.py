from replay_buffer import ReplayBuffer

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
