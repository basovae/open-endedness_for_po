# qd/map_elites.py
import numpy as np
from typing import Dict, Tuple, Optional, List
import copy


class MAPElitesArchive:
    """
    2D grid archive mapping behavior dimensions to elite solutions.
    
    For portfolio optimization:
      - Dim 1: Mean allocation concentration (sparse vs. diversified)
      - Dim 2: Volatility preference (conservative vs. aggressive)
    
    The archive maintains the best-performing solution (elite) for each
    cell in the behavior space grid.
    """
    
    def __init__(
        self,
        dims: Tuple[int, int] = (20, 20),  # grid resolution
        bd_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 0.1)),
        seed: int = 42
    ):
        """
        Initialize MAP-Elites archive.
        
        Args:
            dims: Grid resolution as (n_cells_dim1, n_cells_dim2)
            bd_bounds: Bounds for each BD dimension as ((min1, max1), (min2, max2))
            seed: Random seed for reproducibility
        """
        self.dims = dims
        self.bd_bounds = bd_bounds
        self.rng = np.random.default_rng(seed)
        
        # Grid stores (fitness, behavior_desc, policy_state_dict)
        # Keys are tuples (i, j) representing cell indices
        self.grid: Dict[Tuple[int, int], Tuple[float, np.ndarray, dict]] = {}
    
    def _to_cell(self, bd: np.ndarray) -> Tuple[int, int]:
        """
        Map continuous behavior descriptor to discrete grid cell.
        
        Args:
            bd: 2D behavior descriptor array
            
        Returns:
            Tuple of cell indices (i, j)
        """
        cell = []
        for i, (low, high) in enumerate(self.bd_bounds):
            # Normalize to [0, 1]
            normalized = (bd[i] - low) / (high - low + 1e-8)
            # Map to grid index, clipping to valid range
            idx = int(np.clip(normalized * self.dims[i], 0, self.dims[i] - 1))
            cell.append(idx)
        return tuple(cell)
    
    def add(self, bd: np.ndarray, fitness: float, policy_state: dict) -> bool:
        """
        Add solution to archive if it's the best in its cell.
        
        Args:
            bd: 2D behavior descriptor
            fitness: Fitness score (higher is better)
            policy_state: Policy network state dict to store
            
        Returns:
            True if solution was added (new cell or better fitness)
        """
        cell = self._to_cell(bd)
        
        # Add if cell is empty or new solution is better
        if cell not in self.grid or fitness > self.grid[cell][0]:
            self.grid[cell] = (fitness, bd.copy(), copy.deepcopy(policy_state))
            return True
        return False
    
    def sample_elite(self) -> Optional[dict]:
        """
        Sample a random elite's policy state for mutation.
        
        Returns:
            Policy state dict of a randomly selected elite, or None if archive is empty
        """
        if not self.grid:
            return None
        
        # Get list of cell keys and sample one
        cells = list(self.grid.keys())
        idx = self.rng.integers(0, len(cells))
        cell = cells[idx]
        
        return self.grid[cell][2]  # return policy state dict
    
    def sample_elite_with_info(self) -> Optional[Tuple[Tuple[int, int], float, np.ndarray, dict]]:
        """
        Sample a random elite with full information.
        
        Returns:
            Tuple of (cell, fitness, bd, policy_state) or None if empty
        """
        if not self.grid:
            return None
        
        cells = list(self.grid.keys())
        idx = self.rng.integers(0, len(cells))
        cell = cells[idx]
        
        fitness, bd, policy_state = self.grid[cell]
        return cell, fitness, bd, policy_state
    
    def get_elite(self, cell: Tuple[int, int]) -> Optional[Tuple[float, np.ndarray, dict]]:
        """
        Get elite at specific cell.
        
        Args:
            cell: Grid cell indices (i, j)
            
        Returns:
            Tuple of (fitness, bd, policy_state) or None if cell is empty
        """
        return self.grid.get(cell, None)
    
    def coverage(self) -> float:
        """
        Compute fraction of grid cells that are filled.
        
        Returns:
            Coverage ratio in [0, 1]
        """
        total_cells = self.dims[0] * self.dims[1]
        return len(self.grid) / total_cells
    
    def qd_score(self) -> float:
        """
        Compute QD-Score: sum of all elite fitnesses.
        
        This metric captures both quality (fitness) and diversity (coverage).
        
        Returns:
            Sum of fitness values across all elites
        """
        return sum(v[0] for v in self.grid.values())
    
    def max_fitness(self) -> float:
        """
        Get the maximum fitness in the archive.
        
        Returns:
            Highest fitness value, or -inf if empty
        """
        if not self.grid:
            return float('-inf')
        return max(v[0] for v in self.grid.values())
    
    def get_all_elites(self) -> List[Tuple[Tuple[int, int], float, np.ndarray, dict]]:
        """
        Get all elites in the archive.
        
        Returns:
            List of (cell, fitness, bd, policy_state) tuples
        """
        return [(cell, f, bd, ps) for cell, (f, bd, ps) in self.grid.items()]
    
    def get_fitness_heatmap(self) -> np.ndarray:
        """
        Get fitness values as a 2D array for visualization.
        
        Returns:
            2D numpy array with fitness values (NaN for empty cells)
        """
        heatmap = np.full(self.dims, np.nan)
        for (i, j), (fitness, _, _) in self.grid.items():
            heatmap[i, j] = fitness
        return heatmap
    
    def __len__(self) -> int:
        """Number of elites in archive."""
        return len(self.grid)
    
    def __contains__(self, cell: Tuple[int, int]) -> bool:
        """Check if cell has an elite."""
        return cell in self.grid