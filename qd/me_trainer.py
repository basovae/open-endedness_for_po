# qd/me_trainer.py
"""
MAP-Elites Trainer for Portfolio Policy Search

This module implements a population-based quality-diversity algorithm
that maintains a grid of diverse, high-performing portfolio policies.
"""

import torch
import numpy as np
from typing import Callable, Optional


class MAPElitesTrainer:
    """
    MAP-Elites trainer for portfolio policy search.
    
    Instead of gradient descent, uses evolutionary search:
      1. Random initialization to seed archive
      2. Selection: sample elite from archive
      3. Mutation: perturb policy weights
      4. Evaluation: run episode, compute fitness + BD
      5. Update: add to archive if best in cell
    
    This approach naturally discovers diverse portfolio strategies
    across the behavior space (e.g., conservative vs aggressive,
    concentrated vs diversified).
    """
    
    def __init__(
        self,
        policy_factory: Callable,
        archive,  # MAPElitesArchive
        evaluator: Callable,
        mutation_sigma: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize MAP-Elites trainer.
        
        Args:
            policy_factory: Callable that returns a new policy network
            archive: MAPElitesArchive instance
            evaluator: Callable(policy) -> (fitness, bd) that evaluates a policy
            mutation_sigma: Standard deviation for Gaussian weight perturbation
            seed: Random seed
        """
        self.policy_factory = policy_factory
        self.archive = archive
        self.evaluator = evaluator
        self.mutation_sigma = mutation_sigma
        self.rng = np.random.default_rng(seed)
        
        # Training stats
        self.n_evaluations = 0
        self.n_additions = 0
    
    def initialize(self, n_random: int = 100):
        """
        Seed archive with random policies.
        
        Args:
            n_random: Number of random policies to evaluate
        """
        for _ in range(n_random):
            policy = self.policy_factory()
            fitness, bd = self.evaluator(policy)
            added = self.archive.add(bd, fitness, policy.state_dict())
            
            self.n_evaluations += 1
            if added:
                self.n_additions += 1
    
    def step(self) -> bool:
        """
        Perform one MAP-Elites iteration.
        
        Returns:
            True if a new elite was added to archive
        """
        # Sample parent elite
        parent_state = self.archive.sample_elite()
        if parent_state is None:
            # Archive is empty, initialize with random policy
            policy = self.policy_factory()
            fitness, bd = self.evaluator(policy)
            added = self.archive.add(bd, fitness, policy.state_dict())
            self.n_evaluations += 1
            if added:
                self.n_additions += 1
            return added
        
        # Create child policy and load parent weights
        child = self.policy_factory()
        child.load_state_dict(parent_state)
        
        # Mutate: add Gaussian noise to all parameters
        with torch.no_grad():
            for param in child.parameters():
                noise = torch.randn_like(param) * self.mutation_sigma
                param.add_(noise)
        
        # Evaluate mutated policy
        fitness, bd = self.evaluator(child)
        added = self.archive.add(bd, fitness, child.state_dict())
        
        self.n_evaluations += 1
        if added:
            self.n_additions += 1
        
        return added
    
    def train(
        self,
        n_iterations: int = 10000,
        log_interval: int = 1000,
        verbose: bool = True
    ):
        """
        Run full MAP-Elites training loop.
        
        Args:
            n_iterations: Total number of iterations
            log_interval: How often to print progress
            verbose: Whether to print progress
        """
        # Initialize archive with random policies
        if verbose:
            print("Initializing archive...")
        self.initialize(n_random=min(100, n_iterations // 10))
        
        if verbose:
            print(f"Initial coverage: {self.archive.coverage():.2%}")
            print(f"Initial QD-score: {self.archive.qd_score():.4f}")
            print("\nStarting MAP-Elites iterations...")
        
        for i in range(n_iterations):
            self.step()
            
            if verbose and (i + 1) % log_interval == 0:
                print(
                    f"Iter {i+1:6d}: "
                    f"coverage={self.archive.coverage():.2%}, "
                    f"QD-score={self.archive.qd_score():.4f}, "
                    f"max_fit={self.archive.max_fitness():.4f}, "
                    f"additions={self.n_additions}"
                )
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final coverage: {self.archive.coverage():.2%}")
            print(f"Final QD-score: {self.archive.qd_score():.4f}")
            print(f"Total evaluations: {self.n_evaluations}")
            print(f"Total additions: {self.n_additions}")
    
    def get_best_policy(self):
        """
        Get the policy with highest fitness from archive.
        
        Returns:
            Policy network with best fitness, or None if archive is empty
        """
        if len(self.archive) == 0:
            return None
        
        # Find elite with max fitness
        best_cell = max(self.archive.grid.keys(), 
                       key=lambda c: self.archive.grid[c][0])
        best_state = self.archive.grid[best_cell][2]
        
        policy = self.policy_factory()
        policy.load_state_dict(best_state)
        return policy
    
    def get_diverse_policies(self, n: int = 5):
        """
        Get n diverse policies from different regions of behavior space.
        
        Args:
            n: Number of policies to return
            
        Returns:
            List of (cell, fitness, policy) tuples
        """
        if len(self.archive) == 0:
            return []
        
        # Get all elites sorted by fitness
        elites = self.archive.get_all_elites()
        elites.sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse subset
        selected = []
        selected_cells = set()
        
        for cell, fitness, bd, state in elites:
            if len(selected) >= n:
                break
            
            # Skip if too close to already selected
            too_close = False
            for sel_cell, _, _ in selected:
                dist = abs(cell[0] - sel_cell[0]) + abs(cell[1] - sel_cell[1])
                if dist < self.archive.dims[0] // 4:  # At least 25% apart
                    too_close = True
                    break
            
            if not too_close:
                policy = self.policy_factory()
                policy.load_state_dict(state)
                selected.append((cell, fitness, policy))
                selected_cells.add(cell)
        
        return selected