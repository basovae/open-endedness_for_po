# qd/map_elites.py
import numpy as np
from typing import Dict, Tuple, Optional
import copy

class MAPElitesArchive:
    """
    2D grid archive mapping behavior dimensions to elite solutions.
    For portfolio optimization:
      - Dim 1: Mean allocation concentration (sparse vs. diversified)
      - Dim 2: Volatility preference (conservative vs. aggressive)
    """
    def __init__(
        self,
        dims: Tuple[int, int] = (20, 20),  # grid resolution
        bd_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 0.1)),
        seed: int = 42
    ):
        self.dims = dims
        self.bd_bounds = bd_bounds
        self.rng = np.random.default_rng(seed)
        
        # Grid stores (fitness, behavior_desc, policy_state_dict)
        self.grid: Dict[Tuple[int, int], Tuple[float, np.ndarray, dict]] = {}
    
    def _to_cell(self, bd: np.ndarray) -> Tuple[int, int]:
        """Map continuous BD to discrete grid cell."""
        cell = []
        for i, (low, high) in enumerate(self.bd_bounds):
            normalized = (bd[i] - low) / (high - low + 1e-8)
            idx = int(np.clip(normalized * self.dims[i], 0, self.dims[i] - 1))
            cell.append(idx)
        return tuple(cell)
    
    def add(self, bd: np.ndarray, fitness: float, policy_state: dict) -> bool:
        """Add solution if it's the best in its cell. Returns True if added."""
        cell = self._to_cell(bd)
        if cell not in self.grid or fitness > self.grid[cell][0]:
            self.grid[cell] = (fitness, bd.copy(), copy.deepcopy(policy_state))
            return True
        return False
    
    def sample_elite(self) -> Optional[dict]:
        """Sample a random elite for mutation."""
        if not self.grid:
            return None
        cell = self.rng.choice(list(self.grid.keys()))
        return self.grid[cell][2]  # return policy state dict
    
    def coverage(self) -> float:
        """Fraction of cells filled."""
        return len(self.grid) / (self.dims[0] * self.dims[1])
    
    def qd_score(self) -> float:
        """Sum of all elite fitnesses (QD-Score metric)."""
        return sum(v[0] for v in self.grid.values())