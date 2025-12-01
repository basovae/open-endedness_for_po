# qd/me_trainer.py
class MAPElitesTrainer:
    """
    MAP-Elites for portfolio policy search.
    Instead of gradient descent, uses:
      1. Random initialization to seed archive
      2. Selection: sample elite from archive
      3. Mutation: perturb policy weights
      4. Evaluation: run episode, compute fitness + BD
      5. Update: add to archive if best in cell
    """
    def __init__(self, policy_factory, archive, evaluator, mutation_sigma=0.1):
        self.policy_factory = policy_factory
        self.archive = archive
        self.evaluator = evaluator
        self.mutation_sigma = mutation_sigma
    
    def initialize(self, n_random: int = 100):
        """Seed archive with random policies."""
        for _ in range(n_random):
            policy = self.policy_factory()
            fitness, bd = self.evaluator(policy)
            self.archive.add(bd, fitness, policy.state_dict())
    
    def step(self):
        """One MAP-Elites iteration."""
        # Sample parent
        parent_state = self.archive.sample_elite()
        if parent_state is None:
            return
        
        # Create mutated offspring
        child = self.policy_factory()
        child.load_state_dict(parent_state)
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * self.mutation_sigma)
        
        # Evaluate
        fitness, bd = self.evaluator(child)
        self.archive.add(bd, fitness, child.state_dict())
    
    def train(self, n_iterations: int = 10000):
        self.initialize()
        for i in range(n_iterations):
            self.step()
            if i % 1000 == 0:
                print(f"Iter {i}: coverage={self.archive.coverage():.2%}, "
                      f"QD-score={self.archive.qd_score():.4f}")