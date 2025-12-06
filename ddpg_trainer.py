# ddpg_trainer.py
# ==============================================================================
# IMPROVED VERSION with diagnostics and better defaults
# ==============================================================================
"""
DDPG Trainer for portfolio optimization.

### CHANGES FROM ORIGINAL ###
1. [CHANGE 3] Adjusted default hyperparameters for large input spaces
2. [CHANGE 4] Added training diagnostics tracking
3. [CHANGE 5] Use normalized NS wrapper by default
4. Added gradient clipping for stability
5. Added learning rate scheduling option
"""

import numpy as np
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
from typing import Optional, Dict, List, Any

# [CHANGE 5] Import both wrapper versions
from qd.wrappers import NSWrapper
from qd.bd_presets import bd_weights_plus_returns

# Try to import normalized version (may not exist yet)
try:
    from qd.wrappers import NSWrapperNormalized, create_ns_wrapper
    HAS_NORMALIZED_NS = True
except ImportError:
    HAS_NORMALIZED_NS = False


# ==============================================================================
# [CHANGE 4] Training Diagnostics Class
# ==============================================================================

class TrainingDiagnostics:
    """
    ### [NEW CLASS] ###
    Track training metrics for debugging convergence issues.
    """
    def __init__(self):
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.rewards: List[float] = []
        self.q_values: List[float] = []
        self.portfolio_weights: List[np.ndarray] = []
        self.gradients: Dict[str, List[float]] = {'actor': [], 'critic': []}
        
    def log(
        self,
        actor_loss: float,
        critic_loss: float,
        reward: float,
        q_value: float = 0.0,
        weights: Optional[np.ndarray] = None,
        actor_grad_norm: float = 0.0,
        critic_grad_norm: float = 0.0,
    ):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.rewards.append(reward)
        self.q_values.append(q_value)
        if weights is not None:
            self.portfolio_weights.append(weights.copy())
        self.gradients['actor'].append(actor_grad_norm)
        self.gradients['critic'].append(critic_grad_norm)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.actor_losses:
            return {}
        
        # Check for convergence
        n = len(self.critic_losses)
        if n > 100:
            first_10_pct = np.mean(self.critic_losses[:n//10])
            last_10_pct = np.mean(self.critic_losses[-n//10:])
            converged = last_10_pct < first_10_pct * 0.8
        else:
            converged = None
        
        # Check for exploding Q-values
        q_exploding = max(self.q_values) > 1000 if self.q_values else False
        
        # Portfolio diversity (entropy)
        if self.portfolio_weights:
            last_weights = self.portfolio_weights[-1]
            last_weights = np.clip(last_weights, 1e-10, 1)
            last_weights = last_weights / last_weights.sum()
            entropy = -np.sum(last_weights * np.log(last_weights))
        else:
            entropy = None
        
        return {
            'final_actor_loss': self.actor_losses[-1],
            'final_critic_loss': self.critic_losses[-1],
            'avg_reward_last_100': np.mean(self.rewards[-100:]),
            'q_value_max': max(self.q_values) if self.q_values else None,
            'q_exploding': q_exploding,
            'converged': converged,
            'portfolio_entropy': entropy,
            'total_steps': len(self.actor_losses),
        }
    
    def plot(self, save_path: str = 'ddpg_diagnostics.png'):
        """Generate diagnostic plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        def smooth(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Actor loss
        axes[0, 0].plot(smooth(self.actor_losses))
        axes[0, 0].set_title('Actor Loss (smoothed)')
        axes[0, 0].set_xlabel('Step')
        
        # Critic loss
        axes[0, 1].plot(smooth(self.critic_losses))
        axes[0, 1].set_title('Critic Loss (smoothed)')
        axes[0, 1].set_xlabel('Step')
        
        # Rewards
        axes[0, 2].plot(smooth(self.rewards))
        axes[0, 2].set_title('Reward (smoothed)')
        axes[0, 2].set_xlabel('Step')
        
        # Q-values
        axes[1, 0].plot(smooth(self.q_values))
        axes[1, 0].set_title('Q-Values')
        if max(self.q_values) > 100:
            axes[1, 0].axhline(y=100, color='r', linestyle='--', label='Warning')
            axes[1, 0].legend()
        
        # Gradient norms
        axes[1, 1].plot(smooth(self.gradients['actor']), label='Actor')
        axes[1, 1].plot(smooth(self.gradients['critic']), label='Critic')
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].legend()
        
        # Portfolio concentration over time
        if self.portfolio_weights:
            hhis = []
            for w in self.portfolio_weights[::max(1, len(self.portfolio_weights)//1000)]:
                w = np.clip(w, 0, 1)
                w = w / (w.sum() + 1e-8)
                hhis.append(np.sum(w**2))
            axes[1, 2].plot(hhis)
            axes[1, 2].set_title('Portfolio HHI (lower=more diverse)')
            n_assets = len(self.portfolio_weights[0])
            axes[1, 2].axhline(y=1/n_assets, color='r', linestyle='--', label='Equal weight')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[DIAGNOSTICS] Saved to {save_path}")


# ==============================================================================
# IMPROVED DDPG TRAINER
# ==============================================================================

class DDPGTrainer:
    '''
    Facilitates the training of a DDPG pipeline for portfolio optimization.
    
    ### CHANGES FROM ORIGINAL ###
    - [CHANGE 3] Lower default learning rates for stability
    - [CHANGE 3] Higher default patience
    - [CHANGE 4] Added diagnostics tracking
    - [CHANGE 5] Use normalized NS wrapper
    - [NEW] Added gradient clipping
    - [NEW] Added learning rate scheduling
    '''
    
    def __init__(
        self,
        number_of_assets: int,
        actor: nn.Module,
        critic: nn.Module,
        # === [CHANGE 3] LEARNING RATES - REDUCED ===
        actor_lr: float = 0.005,           # [CHANGED] Was 0.05
        critic_lr: float = 0.005,          # [CHANGED] Was 0.01
        optimizer: torch.optim = torch.optim.Adam,
        # === REGULARIZATION ===
        l1_lambda: float = 0.001,          # [CHANGED] Was 0
        l2_lambda: float = 0.001,          # [CHANGED] Was 0
        weight_decay: float = 0.001,       # [CHANGED] Was 0
        # === TARGET NETWORKS ===
        soft_update: bool = True,          # [CHANGED] Was False
        tau: float = 0.005,
        # === REWARD ===
        risk_preference: float = -0.5,
        gamma: float = 1.0,
        # === [CHANGE 3] EARLY STOPPING - MORE PATIENT ===
        early_stopping: bool = True,
        patience: int = 10,                # [CHANGED] Was 2
        min_delta: float = 1e-6,           # [CHANGED] Was 0
        # === [CHANGE 5] NOVELTY SEARCH - REBALANCED ===
        use_ns: bool = False,
        ns_alpha: float = 0.5,             # [CHANGED] Was 1.0
        ns_beta: float = 0.5,              # [CHANGED] Was 0.5
        use_normalized_ns: bool = True,    # [NEW] Use normalized wrapper
        # === [NEW] STABILITY OPTIONS ===
        gradient_clip: float = 1.0,        # [NEW] Gradient clipping
        use_lr_scheduler: bool = False,    # [NEW] Learning rate decay
        # === [CHANGE 4] DIAGNOSTICS ===
        track_diagnostics: bool = True,    # [NEW] Track training metrics
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
        self.gradient_clip = gradient_clip
        
        # Early stopping
        self.early_stopper = EarlyStopper(patience, min_delta) if early_stopping else None

        # Optimizers
        self.actor_optimizer = optimizer(
            actor.parameters(),
            lr=actor_lr,
        )
        self.critic_optimizer = optimizer(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
        )
        
        # [NEW] Learning rate schedulers
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
                self.actor_optimizer, step_size=50, gamma=0.9
            )
            self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
                self.critic_optimizer, step_size=50, gamma=0.9
            )
    
        # Target networks
        if soft_update:
            self.target_actor = deepcopy(actor)
            self.target_critic = deepcopy(critic)
            self._soft_update(self.target_actor, self.actor, tau=1.0)
            self._soft_update(self.target_critic, self.critic, tau=1.0)
        
        # === [CHANGE 5] Novelty Search with normalization ===
        self.use_ns = use_ns
        if use_ns:
            if use_normalized_ns and HAS_NORMALIZED_NS:
                # [NEW] Use normalized wrapper
                self.ns = NSWrapperNormalized(
                    bd_fn=bd_weights_plus_returns,
                    alpha=ns_alpha,
                    beta=ns_beta,
                    warmup_episodes=10,
                )
                print("[NS] Using NORMALIZED wrapper (recommended)")
            else:
                # Fallback to original
                self.ns = NSWrapper(
                    bd_fn=bd_weights_plus_returns,
                    alpha=ns_alpha,
                    beta=ns_beta,
                )
                print("[NS] Using original wrapper")
        else:
            self.ns = None
        
        # === [CHANGE 4] Diagnostics ===
        self.track_diagnostics = track_diagnostics
        self.diagnostics = TrainingDiagnostics() if track_diagnostics else None

    def _soft_update(self, target, source, tau):
        '''Soft-update target network parameters.'''
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * source_param.data)
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm for a model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 300,             # [CHANGED] Was 100
        noise: float = 0.2,
        verbose: int = 1,
    ):
        '''
        Training loop for DDPG.
        
        ### CHANGES FROM ORIGINAL ###
        - [CHANGE 3] More epochs by default
        - [CHANGE 4] Diagnostics logging
        - [NEW] Gradient clipping
        '''
        replay_buffer = ReplayBuffer()

        print(
            "n_assets:", self.number_of_assets,
            "actor_input_size:", self.actor.input_size,
            "inferred_lookback:", self.actor.input_size // self.number_of_assets
        )

        for epoch in range(num_epochs):
            total_actor_loss = 0
            total_critic_loss = 0
            episode_task_return = 0.0
            step_count = 0
            
            # [CHANGE 5] Reset NS buffers at epoch start
            if self.use_ns:
                self.ns.reset_episode_buffers()

            for state, next_state in train_loader:
                step_count += 1
                
                # === ACTION SELECTION ===
                portfolio_allocation = self.actor(state.flatten())
                exploration_noise = torch.normal(0, noise, portfolio_allocation.shape)
                noisy_portfolio_allocation = portfolio_allocation + exploration_noise

                # === REWARD COMPUTATION ===
                avg_profit = torch.mean(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1)
                ).detach().cpu()
                volatility = torch.std(
                    torch.sum(state.view(-1, self.number_of_assets) * noisy_portfolio_allocation,
                              dim=-1),
                    correction=0,
                ).detach().cpu()
                reward = avg_profit + self.risk_preference * volatility
                
                episode_task_return += float(reward)

                # === [CHANGE 5] NS step tracking ===
                if self.use_ns:
                    self.ns.on_step(
                        state=state.numpy(),
                        action=noisy_portfolio_allocation.detach().numpy(),
                        reward_task=float(reward),
                        info={
                            "weights": noisy_portfolio_allocation.detach().numpy(),
                            "return_t": float(avg_profit),
                        }
                    )

                # === REPLAY BUFFER ===
                replay_buffer.push((
                    state.detach(),
                    noisy_portfolio_allocation.detach(),
                    reward.detach(),
                    next_state.detach()
                ))
                
                if len(replay_buffer) < 1:
                    continue
                    
                transition = replay_buffer.sample(1)
                state = transition[0][0]
                noisy_portfolio_allocation = transition[0][1]
                reward = transition[0][2]
                next_state = transition[0][3]

                portfolio_allocation = self.actor(state.flatten())

                # === CRITIC UPDATE ===
                q_value = self.critic(
                    torch.cat((state.flatten(), noisy_portfolio_allocation.flatten()))
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

                target_q_value = reward + self.gamma * next_q_value
                critic_loss = (target_q_value - q_value).pow(2)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                
                # [NEW] Gradient clipping for stability
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
                
                critic_grad_norm = self._compute_gradient_norm(self.critic)
                self.critic_optimizer.step()

                # === ACTOR UPDATE ===
                q_value_for_actor = self.critic(
                    torch.cat((state.flatten(), portfolio_allocation.flatten()))
                )
                actor_loss = -q_value_for_actor

                # Regularization
                if self.l1_lambda > 0:
                    l1_reg = sum(p.abs().sum() for p in self.actor.parameters())
                    actor_loss = actor_loss + self.l1_lambda * l1_reg
                if self.l2_lambda > 0:
                    l2_reg = sum(p.pow(2).sum() for p in self.actor.parameters())
                    actor_loss = actor_loss + self.l2_lambda * l2_reg

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                
                # [NEW] Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                
                actor_grad_norm = self._compute_gradient_norm(self.actor)
                self.actor_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                
                # === [CHANGE 4] Log diagnostics ===
                if self.track_diagnostics:
                    self.diagnostics.log(
                        actor_loss=actor_loss.item(),
                        critic_loss=critic_loss.item(),
                        reward=float(reward),
                        q_value=q_value.item(),
                        weights=portfolio_allocation.detach().numpy(),
                        actor_grad_norm=actor_grad_norm,
                        critic_grad_norm=critic_grad_norm,
                    )

            # === END OF EPOCH ===
            avg_actor_loss = total_actor_loss / max(step_count, 1)
            avg_critic_loss = total_critic_loss / max(step_count, 1)

            # === VALIDATION ===
            if self.early_stopper:
                with torch.no_grad():
                    val_critic_loss = 0
                    val_steps = 0
                    
                    for state, next_state in val_loader:
                        val_steps += 1
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
                            correction=0,
                        ).detach().cpu()
                        reward = avg_profit + self.risk_preference * volatility

                        target_q_value = reward + self.gamma * next_q_value
                        val_critic_loss += (target_q_value - q_value).pow(2).item()

                    avg_val_critic_loss = val_critic_loss / max(val_steps, 1)

                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                          f'Actor Loss: {avg_actor_loss:.6f}, '
                          f'Critic Loss: {avg_critic_loss:.6f}, '
                          f'Val Loss: {avg_val_critic_loss:.6f}')

                # === [CHANGE 5] NS episode end ===
                if self.use_ns:
                    blended = self.ns.on_episode_end(episode_task_return)
                    if verbose > 0:
                        print(f"  NS blended: {blended:.6f} | task_return: {episode_task_return:.6f}")

                # Early stopping check
                if self.early_stopper.early_stop(avg_val_critic_loss, verbose=verbose):
                    print(f"[EARLY STOP] at epoch {epoch+1}")
                    break
                    
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                          f'Actor Loss: {avg_actor_loss:.6f}, '
                          f'Critic Loss: {avg_critic_loss:.6f}')

            # === TARGET NETWORK UPDATE ===
            if self.soft_update:
                self._soft_update(self.target_actor, self.actor, self.tau)
                self._soft_update(self.target_critic, self.critic, self.tau)
            
            # === [NEW] Learning rate scheduling ===
            if self.use_lr_scheduler:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
        
        # === [CHANGE 4] Print diagnostics summary ===
        if self.track_diagnostics:
            summary = self.diagnostics.get_summary()
            print("\n[TRAINING SUMMARY]")
            for k, v in summary.items():
                if v is not None:
                    print(f"  {k}: {v}")
    
    def save_diagnostics(self, path: str = 'ddpg_diagnostics.png'):
        """Save diagnostic plots."""
        if self.track_diagnostics and self.diagnostics:
            self.diagnostics.plot(path)