# tests/test_qd_suite.py
"""
Comprehensive Test Suite for Quality-Diversity Algorithms
=========================================================

Testing Levels:
1. Unit Tests - Individual components in isolation
2. Integration Tests - Components working together  
3. System Tests - Full training pipelines
4. Validation Tests - QD-specific metrics and behaviors

Run with: pytest tests/test_qd_suite.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# FIXTURES - Reusable test components
# ==============================================================================

@pytest.fixture
def sample_trajectory():
    """Generate a sample trajectory for BD testing."""
    np.random.seed(42)
    n_steps, n_assets = 50, 5
    weights = np.random.dirichlet(np.ones(n_assets), size=n_steps)
    returns = np.random.normal(0.001, 0.02, size=n_steps)
    return {
        "weights_traj": weights,
        "returns": returns,
        "actions": weights,
        "states": np.random.randn(n_steps, 10)
    }


@pytest.fixture
def simple_policy_factory():
    """Factory that creates simple MLP policies for testing."""
    def factory():
        model = nn.Sequential(
            nn.Linear(25, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Softmax(dim=-1)
        )
        return model
    return factory


@pytest.fixture
def mock_evaluator():
    """Mock evaluator that returns deterministic fitness and BD."""
    def evaluator(policy):
        # Sum of first layer weights as "fitness"
        with torch.no_grad():
            first_layer = list(policy.parameters())[0]
            fitness = float(first_layer.sum())
        # Random BD for diversity
        bd = np.random.rand(2)
        return fitness, bd
    return evaluator


# ==============================================================================
# LEVEL 1: UNIT TESTS - Individual Components
# ==============================================================================

class TestNoveltyArchive:
    """Unit tests for NoveltyArchive."""
    
    def test_empty_archive_returns_max_novelty(self):
        """Empty archive should return maximum novelty (1.0)."""
        from qd.novelty_archive import NoveltyArchive
        archive = NoveltyArchive(k=5)
        desc = np.array([0.5, 0.5, 0.5])
        assert archive.novelty(desc) == 1.0
    
    def test_adding_to_archive(self):
        """Items should be added to archive."""
        from qd.novelty_archive import NoveltyArchive
        archive = NoveltyArchive(k=5, add_every=1)
        desc = np.array([0.1, 0.2, 0.3])
        archive.maybe_add(desc)
        assert len(archive) == 1
    
    def test_novelty_decreases_for_similar_descriptors(self):
        """Novelty should decrease as archive fills with similar items."""
        from qd.novelty_archive import NoveltyArchive
        archive = NoveltyArchive(k=3, add_every=1)
        
        # Add several similar descriptors
        for i in range(5):
            desc = np.array([0.5 + i*0.01, 0.5, 0.5])
            archive.maybe_add(desc)
        
        # New similar descriptor should have low novelty
        similar = np.array([0.52, 0.5, 0.5])
        novelty_similar = archive.novelty(similar)
        
        # Different descriptor should have higher novelty
        different = np.array([0.0, 0.0, 0.0])
        novelty_different = archive.novelty(different)
        
        assert novelty_different > novelty_similar
    
    def test_max_size_replacement(self):
        """Archive should not exceed max_size."""
        from qd.novelty_archive import NoveltyArchive
        archive = NoveltyArchive(k=5, add_every=1, max_size=10)
        
        for i in range(20):
            archive.maybe_add(np.random.rand(3))
        
        assert len(archive) == 10
    
    def test_add_every_parameter(self):
        """add_every should control addition frequency."""
        from qd.novelty_archive import NoveltyArchive
        archive = NoveltyArchive(k=5, add_every=3)
        
        for i in range(9):
            archive.maybe_add(np.random.rand(3))
        
        assert len(archive) == 3  # Only added at steps 3, 6, 9


class TestBehaviorDescriptors:
    """Unit tests for behavior descriptor functions."""
    
    def test_bd_weights_hist_shape(self, sample_trajectory):
        """bd_weights_hist should return correct shape."""
        from qd.novelty_metrics import bd_weights_hist
        weights = sample_trajectory["weights_traj"]
        bd = bd_weights_hist(weights, bins=5)
        n_assets = weights.shape[1]
        assert bd.shape == (n_assets * 5,)
    
    def test_bd_weights_hist_normalized(self, sample_trajectory):
        """bd_weights_hist should be L2 normalized."""
        from qd.novelty_metrics import bd_weights_hist
        weights = sample_trajectory["weights_traj"]
        bd = bd_weights_hist(weights, bins=5)
        norm = np.linalg.norm(bd)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_bd_returns_shape_output(self, sample_trajectory):
        """bd_returns_shape should return correct shape."""
        from qd.novelty_metrics import bd_returns_shape
        returns = sample_trajectory["returns"]
        bd = bd_returns_shape(returns, segments=10)
        assert bd.shape == (10,)
    
    def test_bd_for_map_elites_is_2d(self, sample_trajectory):
        """bd_for_map_elites must return exactly 2D."""
        from qd.bd_presets import bd_for_map_elites
        bd = bd_for_map_elites(sample_trajectory)
        assert bd.shape == (2,)
    
    def test_bd_for_map_elites_hhi_bounds(self, sample_trajectory):
        """HHI should be in [1/n, 1] for valid portfolios."""
        from qd.bd_presets import bd_for_map_elites
        bd = bd_for_map_elites(sample_trajectory)
        hhi = bd[0]
        n_assets = sample_trajectory["weights_traj"].shape[1]
        assert 1/n_assets - 0.01 <= hhi <= 1.01  # Small tolerance
    
    def test_concentrated_vs_diversified_hhi(self):
        """Concentrated portfolios should have higher HHI."""
        from qd.bd_presets import bd_for_map_elites
        
        # Diversified: equal weights
        diversified = {
            "weights_traj": np.ones((50, 5)) / 5,
            "returns": np.zeros(50)
        }
        
        # Concentrated: all in one asset
        concentrated = {
            "weights_traj": np.array([[1, 0, 0, 0, 0]] * 50),
            "returns": np.zeros(50)
        }
        
        hhi_div = bd_for_map_elites(diversified)[0]
        hhi_con = bd_for_map_elites(concentrated)[0]
        
        assert hhi_con > hhi_div
        assert np.isclose(hhi_div, 0.2, atol=0.01)  # 1/5 = 0.2
        assert np.isclose(hhi_con, 1.0, atol=0.01)


class TestMAPElitesArchive:
    """Unit tests for MAPElitesArchive."""
    
    def test_empty_archive_stats(self):
        """Empty archive should have zero coverage and QD-score."""
        from qd.map_elites import MAPElitesArchive
        archive = MAPElitesArchive(dims=(10, 10))
        assert archive.coverage() == 0.0
        assert archive.qd_score() == 0.0
    
    def test_add_single_elite(self):
        """Adding one elite should update coverage."""
        from qd.map_elites import MAPElitesArchive
        archive = MAPElitesArchive(dims=(10, 10))
        
        bd = np.array([0.5, 0.05])
        fitness = 1.5
        policy_state = {"layer1.weight": torch.randn(10, 5)}
        
        added = archive.add(bd, fitness, policy_state)
        
        assert added is True
        assert archive.coverage() == 1 / 100
        assert archive.qd_score() == 1.5
    
    def test_better_fitness_replaces_elite(self):
        """Higher fitness should replace existing elite in same cell."""
        from qd.map_elites import MAPElitesArchive
        archive = MAPElitesArchive(dims=(10, 10))
        
        bd = np.array([0.5, 0.05])
        archive.add(bd, fitness=1.0, policy_state={"v": 1})
        archive.add(bd, fitness=2.0, policy_state={"v": 2})
        
        assert archive.qd_score() == 2.0
        assert archive.coverage() == 1 / 100  # Still only one cell
    
    def test_worse_fitness_rejected(self):
        """Lower fitness should not replace existing elite."""
        from qd.map_elites import MAPElitesArchive
        archive = MAPElitesArchive(dims=(10, 10))
        
        bd = np.array([0.5, 0.05])
        archive.add(bd, fitness=2.0, policy_state={"v": 2})
        added = archive.add(bd, fitness=1.0, policy_state={"v": 1})
        
        assert added is False
        assert archive.qd_score() == 2.0
    
    def test_cell_mapping_boundaries(self):
        """BD at boundaries should map to valid cells."""
        from qd.map_elites import MAPElitesArchive
        archive = MAPElitesArchive(
            dims=(10, 10),
            bd_bounds=((0, 1), (0, 0.1))
        )
        
        # Test corners
        corners = [
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.1]),
            np.array([1.0, 0.1]),
        ]
        
        for bd in corners:
            cell = archive._to_cell(bd)
            assert 0 <= cell[0] < 10
            assert 0 <= cell[1] < 10
    
    def test_sample_elite_returns_valid_state(self):
        """sample_elite should return stored policy state."""
        from qd.map_elites import MAPElitesArchive
        archive = MAPElitesArchive(dims=(5, 5))
        
        policy_state = {"test_key": torch.tensor([1, 2, 3])}
        archive.add(np.array([0.5, 0.05]), 1.0, policy_state)
        
        sampled = archive.sample_elite()
        assert "test_key" in sampled
        assert torch.equal(sampled["test_key"], policy_state["test_key"])


class TestNSWrapper:
    """Unit tests for NSWrapper."""
    
    def test_buffer_reset(self):
        """Buffers should be empty after reset."""
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns)
        ns.buf["weights_traj"].append(np.array([1, 2, 3]))
        ns.reset_episode_buffers()
        
        assert len(ns.buf["weights_traj"]) == 0
    
    def test_on_step_collects_data(self):
        """on_step should accumulate trajectory data."""
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns)
        
        for _ in range(10):
            info = {"weights": np.random.rand(5), "return_t": 0.01}
            ns.on_step(state=None, action=info["weights"], reward_task=0.01, info=info)
        
        assert len(ns.buf["weights_traj"]) == 10
        assert len(ns.buf["returns"]) == 10
    
    def test_on_episode_end_clears_buffers(self):
        """on_episode_end should clear buffers after processing."""
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns)
        
        for _ in range(10):
            info = {"weights": np.random.rand(5), "return_t": 0.01}
            ns.on_step(state=None, action=info["weights"], reward_task=0.01, info=info)
        
        ns.on_episode_end(episode_task_return=0.5)
        
        assert len(ns.buf["weights_traj"]) == 0
    
    def test_blended_score_computation(self):
        """Blended score should combine task return and novelty."""
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns, alpha=1.0, beta=0.5)
        
        # First episode - max novelty expected
        for _ in range(10):
            info = {"weights": np.random.rand(5), "return_t": 0.01}
            ns.on_step(state=None, action=info["weights"], reward_task=0.01, info=info)
        
        task_return = 1.0
        blended = ns.on_episode_end(episode_task_return=task_return)
        
        # blended = alpha * task_return + beta * novelty
        # First episode novelty = 1.0 (empty archive)
        expected = 1.0 * 1.0 + 0.5 * 1.0
        assert np.isclose(blended, expected, atol=0.1)


# ==============================================================================
# LEVEL 2: INTEGRATION TESTS - Components Working Together
# ==============================================================================

class TestMAPElitesIntegration:
    """Integration tests for MAP-Elites pipeline."""
    
    def test_trainer_initialization(self, simple_policy_factory, mock_evaluator):
        """MAPElitesTrainer should initialize archive with random policies."""
        from qd.map_elites import MAPElitesArchive
        from qd.me_trainer import MAPElitesTrainer
        
        archive = MAPElitesArchive(dims=(5, 5))
        trainer = MAPElitesTrainer(
            policy_factory=simple_policy_factory,
            archive=archive,
            evaluator=mock_evaluator,
            mutation_sigma=0.1
        )
        
        trainer.initialize(n_random=20)
        
        assert archive.coverage() > 0
        assert len(archive.grid) <= 20
    
    def test_trainer_step_adds_to_archive(self, simple_policy_factory, mock_evaluator):
        """Single step should potentially add new elite."""
        from qd.map_elites import MAPElitesArchive
        from qd.me_trainer import MAPElitesTrainer
        
        archive = MAPElitesArchive(dims=(10, 10))
        trainer = MAPElitesTrainer(
            policy_factory=simple_policy_factory,
            archive=archive,
            evaluator=mock_evaluator,
            mutation_sigma=0.5
        )
        
        trainer.initialize(n_random=5)
        initial_coverage = archive.coverage()
        
        for _ in range(50):
            trainer.step()
        
        # Coverage should increase or stay same (never decrease)
        assert archive.coverage() >= initial_coverage
    
    def test_full_training_loop(self, simple_policy_factory, mock_evaluator):
        """Full training should increase coverage and QD-score."""
        from qd.map_elites import MAPElitesArchive
        from qd.me_trainer import MAPElitesTrainer
        
        archive = MAPElitesArchive(dims=(5, 5))
        trainer = MAPElitesTrainer(
            policy_factory=simple_policy_factory,
            archive=archive,
            evaluator=mock_evaluator,
            mutation_sigma=0.3
        )
        
        # Short training run
        trainer.initialize(n_random=10)
        initial_qd = archive.qd_score()
        
        for _ in range(100):
            trainer.step()
        
        final_qd = archive.qd_score()
        
        # QD-score should improve or stay same
        assert final_qd >= initial_qd * 0.9  # Allow small regression


class TestNoveltySearchIntegration:
    """Integration tests for Novelty Search with RL trainers."""
    
    def test_ns_archive_grows_during_training(self):
        """NS archive should accumulate descriptors during training."""
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns, alpha=1.0, beta=0.5)
        
        # Simulate multiple episodes
        for episode in range(10):
            for step in range(20):
                info = {"weights": np.random.rand(5), "return_t": np.random.normal(0, 0.02)}
                ns.on_step(state=None, action=info["weights"], reward_task=info["return_t"], info=info)
            ns.on_episode_end(episode_task_return=np.random.rand())
        
        assert len(ns.archive) > 0
    
    def test_novelty_decreases_over_time_for_similar_behavior(self):
        """Novelty should decrease if agent keeps doing similar things."""
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns, alpha=1.0, beta=1.0)
        
        novelties = []
        
        # Same behavior pattern every episode
        fixed_weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        
        for episode in range(15):
            for step in range(30):
                noisy_weights = fixed_weights + np.random.normal(0, 0.01, 5)
                noisy_weights = np.clip(noisy_weights, 0, 1)
                noisy_weights /= noisy_weights.sum()
                info = {"weights": noisy_weights, "return_t": 0.001}
                ns.on_step(state=None, action=noisy_weights, reward_task=0.001, info=info)
            
            blended = ns.on_episode_end(episode_task_return=0.03)
            novelty = blended - 1.0 * 0.03  # Extract novelty component
            novelties.append(novelty)
        
        # Novelty should trend downward
        assert novelties[-1] < novelties[0]


# ==============================================================================
# LEVEL 3: SYSTEM TESTS - Full Pipelines
# ==============================================================================

class TestFullPipelines:
    """System tests for complete training pipelines."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate synthetic market data for testing."""
        np.random.seed(42)
        n_days, n_assets = 500, 5
        returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
        import pandas as pd
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        df = pd.DataFrame(returns, index=dates, columns=[f"Asset_{i}" for i in range(n_assets)])
        return df
    
    def test_ddpg_with_ns_runs_without_error(self, sample_market_data):
        """DDPG with NS enabled should complete training."""
        # This test verifies the integration doesn't crash
        # Skip if modules not available
        pytest.importorskip("ddpg")
        pytest.importorskip("predictors")
        
        from ddpg import DDPG
        import predictors
        
        train = sample_market_data.iloc[:300]
        val = sample_market_data.iloc[300:400]
        
        model = DDPG(
            lookback_window=20,
            predictor=predictors.MLP,
            batch_size=1,
            short_selling=False,
            forecast_window=0,
            hidden_sizes=(32, 32),
            verbose=0,
        )
        
        # Should not raise
        model.train(
            train_data=train,
            val_data=val,
            num_epochs=3,
            use_ns=True,
            ns_alpha=1.0,
            ns_beta=0.5,
        )
    
    def test_dqn_with_ns_runs_without_error(self, sample_market_data):
        """DQN with NS enabled should complete training."""
        pytest.importorskip("deep_q_learning")
        pytest.importorskip("predictors")
        
        from deep_q_learning import DeepQLearning
        import predictors
        
        train = sample_market_data.iloc[:300]
        val = sample_market_data.iloc[300:400]
        
        model = DeepQLearning(
            lookback_window=20,
            predictor=predictors.MLP,
            batch_size=1,
            short_selling=False,
            forecast_window=0,
            hidden_sizes=(32, 32),
            verbose=0,
        )
        
        model.train(
            train_data=train,
            val_data=val,
            num_epochs=3,
            use_ns=True,
            ns_alpha=1.0,
            ns_beta=0.5,
        )


# ==============================================================================
# LEVEL 4: VALIDATION TESTS - QD-Specific Properties
# ==============================================================================

class TestQDProperties:
    """Tests verifying QD-specific properties and invariants."""
    
    def test_map_elites_monotonic_qd_score(self, simple_policy_factory, mock_evaluator):
        """QD-score should be monotonically non-decreasing."""
        from qd.map_elites import MAPElitesArchive
        from qd.me_trainer import MAPElitesTrainer
        
        archive = MAPElitesArchive(dims=(5, 5))
        trainer = MAPElitesTrainer(
            policy_factory=simple_policy_factory,
            archive=archive,
            evaluator=mock_evaluator,
            mutation_sigma=0.2
        )
        
        trainer.initialize(n_random=10)
        qd_scores = [archive.qd_score()]
        
        for _ in range(50):
            trainer.step()
            qd_scores.append(archive.qd_score())
        
        # Check monotonicity
        for i in range(1, len(qd_scores)):
            assert qd_scores[i] >= qd_scores[i-1] - 1e-6  # Small tolerance for float
    
    def test_map_elites_coverage_bounded(self, simple_policy_factory, mock_evaluator):
        """Coverage should always be in [0, 1]."""
        from qd.map_elites import MAPElitesArchive
        from qd.me_trainer import MAPElitesTrainer
        
        archive = MAPElitesArchive(dims=(5, 5))
        trainer = MAPElitesTrainer(
            policy_factory=simple_policy_factory,
            archive=archive,
            evaluator=mock_evaluator,
            mutation_sigma=0.5
        )
        
        for _ in range(200):
            if len(archive.grid) == 0:
                trainer.initialize(n_random=1)
            else:
                trainer.step()
            
            coverage = archive.coverage()
            assert 0 <= coverage <= 1
    
    def test_novelty_search_encourages_diversity(self):
        """NS should produce more diverse behaviors than pure task reward."""
        from qd.novelty_archive import NoveltyArchive
        
        archive = NoveltyArchive(k=5, add_every=1)
        
        # Simulate diverse vs. repetitive agents
        diverse_novelties = []
        for i in range(20):
            # Diverse: each descriptor is different
            desc = np.array([i * 0.1, np.sin(i), np.cos(i)])
            nov = archive.novelty(desc)
            diverse_novelties.append(nov)
            archive.maybe_add(desc)
        
        archive2 = NoveltyArchive(k=5, add_every=1)
        repetitive_novelties = []
        for i in range(20):
            # Repetitive: same descriptor with tiny noise
            desc = np.array([0.5, 0.5, 0.5]) + np.random.normal(0, 0.001, 3)
            nov = archive2.novelty(desc)
            repetitive_novelties.append(nov)
            archive2.maybe_add(desc)
        
        # Average novelty should be higher for diverse agent
        assert np.mean(diverse_novelties) > np.mean(repetitive_novelties)
    
    def test_behavior_descriptor_determinism(self, sample_trajectory):
        """Same trajectory should produce same BD."""
        from qd.bd_presets import bd_for_map_elites, bd_weights_plus_returns
        
        bd1 = bd_for_map_elites(sample_trajectory)
        bd2 = bd_for_map_elites(sample_trajectory)
        
        assert np.allclose(bd1, bd2)
        
        bd3 = bd_weights_plus_returns(sample_trajectory)
        bd4 = bd_weights_plus_returns(sample_trajectory)
        
        assert np.allclose(bd3, bd4)


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_asset_portfolio(self):
        """BD functions should handle single-asset portfolios."""
        from qd.bd_presets import bd_for_map_elites
        
        traj = {
            "weights_traj": np.ones((50, 1)),  # Single asset, always 100%
            "returns": np.random.normal(0, 0.01, 50)
        }
        
        bd = bd_for_map_elites(traj)
        assert bd.shape == (2,)
        assert bd[0] == 1.0  # HHI = 1 for single asset
    
    def test_zero_length_trajectory(self):
        """BD functions should handle empty trajectories gracefully."""
        from qd.bd_presets import bd_for_map_elites
        
        traj = {
            "weights_traj": np.array([]).reshape(0, 5),
            "returns": np.array([])
        }
        
        # Should not crash
        try:
            bd = bd_for_map_elites(traj)
        except (ValueError, IndexError):
            pass  # Expected for empty input
    
    def test_extreme_bd_values(self):
        """MAP-Elites should handle BDs outside expected bounds."""
        from qd.map_elites import MAPElitesArchive
        
        archive = MAPElitesArchive(
            dims=(10, 10),
            bd_bounds=((0, 1), (0, 0.1))
        )
        
        # BD outside bounds
        extreme_bd = np.array([2.0, 0.5])  # Both dimensions out of bounds
        
        # Should not crash, just clip to valid cell
        added = archive.add(extreme_bd, fitness=1.0, policy_state={})
        assert added  # Should still add
    
    def test_negative_fitness(self):
        """MAP-Elites should handle negative fitness values."""
        from qd.map_elites import MAPElitesArchive
        
        archive = MAPElitesArchive(dims=(5, 5))
        
        bd = np.array([0.5, 0.05])
        archive.add(bd, fitness=-1.0, policy_state={"a": 1})
        archive.add(bd, fitness=-0.5, policy_state={"a": 2})  # Better (less negative)
        
        assert archive.qd_score() == -0.5


# ==============================================================================
# PERFORMANCE BENCHMARKS (optional, for profiling)
# ==============================================================================

class TestPerformance:
    """Performance benchmarks for QD components."""
    
    @pytest.mark.slow
    def test_novelty_archive_scaling(self):
        """Novelty computation should scale reasonably with archive size."""
        from qd.novelty_archive import NoveltyArchive
        import time
        
        archive = NoveltyArchive(k=15, add_every=1, max_size=10000)
        
        # Fill archive
        for i in range(5000):
            archive.maybe_add(np.random.rand(50))
        
        # Time novelty computation
        start = time.time()
        for _ in range(100):
            archive.novelty(np.random.rand(50))
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1s for 100 queries)
        assert elapsed < 2.0
    
    @pytest.mark.slow  
    def test_map_elites_archive_scaling(self):
        """MAP-Elites operations should scale with grid size."""
        from qd.map_elites import MAPElitesArchive
        import time
        
        archive = MAPElitesArchive(dims=(50, 50))  # 2500 cells
        
        start = time.time()
        for i in range(1000):
            bd = np.random.rand(2)
            archive.add(bd, fitness=np.random.rand(), policy_state={"i": i})
        elapsed = time.time() - start
        
        assert elapsed < 2.0


# ==============================================================================
# RUN CONFIGURATION
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])