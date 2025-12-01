# tests/smoke_test_qd.py
"""
Quick Smoke Tests for QD Algorithms
====================================

Fast validation that core components work.
Run with: python tests/smoke_test_qd.py

Exit codes:
  0 = All tests passed
  1 = Some tests failed
"""

import sys
import numpy as np
import traceback

# Add project root
sys.path.insert(0, '.')

def test_result(name, passed, error=None):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if error and not passed:
        print(f"       Error: {error}")
    return passed


def run_smoke_tests():
    """Run quick smoke tests on all QD components."""
    print("\n" + "="*60)
    print("QD ALGORITHM SMOKE TESTS")
    print("="*60 + "\n")
    
    all_passed = True
    
    # -------------------------------------------------------------------------
    # Test 1: NoveltyArchive
    # -------------------------------------------------------------------------
    print("[1] NoveltyArchive")
    try:
        from qd.novelty_archive import NoveltyArchive
        archive = NoveltyArchive(k=5, add_every=1, max_size=100)
        
        # Test empty archive novelty
        nov = archive.novelty(np.array([0.5, 0.5]))
        all_passed &= test_result("Empty archive returns novelty=1.0", nov == 1.0)
        
        # Test adding
        archive.maybe_add(np.array([0.1, 0.2]))
        all_passed &= test_result("Adding to archive", len(archive) == 1)
        
        # Test novelty computation
        nov = archive.novelty(np.array([0.1, 0.2]))
        all_passed &= test_result("Novelty for identical descriptor is low", nov < 0.1)
        
    except Exception as e:
        all_passed &= test_result("NoveltyArchive import/basic ops", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 2: Behavior Descriptors
    # -------------------------------------------------------------------------
    print("\n[2] Behavior Descriptors")
    try:
        from qd.novelty_metrics import bd_weights_hist, bd_returns_shape
        from qd.bd_presets import bd_for_map_elites, bd_weights_plus_returns
        
        # Create sample trajectory
        traj = {
            "weights_traj": np.random.dirichlet(np.ones(5), size=30),
            "returns": np.random.normal(0, 0.02, 30)
        }
        
        # Test each BD function
        bd1 = bd_weights_hist(traj["weights_traj"], bins=5)
        all_passed &= test_result("bd_weights_hist shape", bd1.shape == (25,))
        
        bd2 = bd_returns_shape(traj["returns"], segments=10)
        all_passed &= test_result("bd_returns_shape shape", bd2.shape == (10,))
        
        bd3 = bd_for_map_elites(traj)
        all_passed &= test_result("bd_for_map_elites is 2D", bd3.shape == (2,))
        
        bd4 = bd_weights_plus_returns(traj)
        all_passed &= test_result("bd_weights_plus_returns runs", bd4.shape[0] > 0)
        
    except Exception as e:
        all_passed &= test_result("Behavior descriptors", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 3: MAPElitesArchive
    # -------------------------------------------------------------------------
    print("\n[3] MAPElitesArchive")
    try:
        from qd.map_elites import MAPElitesArchive
        
        archive = MAPElitesArchive(dims=(10, 10), bd_bounds=((0, 1), (0, 0.1)))
        
        # Test empty
        all_passed &= test_result("Empty coverage is 0", archive.coverage() == 0.0)
        
        # Test adding
        import torch
        policy_state = {"w": torch.randn(5)}
        added = archive.add(np.array([0.5, 0.05]), fitness=1.0, policy_state=policy_state)
        all_passed &= test_result("Adding elite", added and archive.coverage() > 0)
        
        # Test sampling
        sampled = archive.sample_elite()
        all_passed &= test_result("Sampling elite", sampled is not None and "w" in sampled)
        
        # Test replacement
        archive.add(np.array([0.5, 0.05]), fitness=2.0, policy_state={"w": torch.randn(5)})
        all_passed &= test_result("Better fitness replaces", archive.qd_score() == 2.0)
        
    except Exception as e:
        all_passed &= test_result("MAPElitesArchive", False, str(e))
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Test 4: NSWrapper
    # -------------------------------------------------------------------------
    print("\n[4] NSWrapper")
    try:
        from qd.wrappers import NSWrapper
        from qd.bd_presets import bd_weights_plus_returns
        
        ns = NSWrapper(bd_fn=bd_weights_plus_returns, alpha=1.0, beta=0.5)
        
        # Simulate episode
        for _ in range(20):
            w = np.random.dirichlet(np.ones(5))
            info = {"weights": w, "return_t": 0.01}
            ns.on_step(state=None, action=w, reward_task=0.01, info=info)
        
        all_passed &= test_result("on_step collects data", len(ns.buf["weights_traj"]) == 20)
        
        blended = ns.on_episode_end(episode_task_return=0.5)
        all_passed &= test_result("on_episode_end returns blended score", isinstance(blended, float))
        all_passed &= test_result("Buffers cleared after episode", len(ns.buf["weights_traj"]) == 0)
        
    except Exception as e:
        all_passed &= test_result("NSWrapper", False, str(e))
    
    # -------------------------------------------------------------------------
    # Test 5: MAPElitesTrainer
    # -------------------------------------------------------------------------
    print("\n[5] MAPElitesTrainer")
    try:
        from qd.map_elites import MAPElitesArchive
        from qd.me_trainer import MAPElitesTrainer
        import torch
        import torch.nn as nn
        
        # Simple policy factory
        def policy_factory():
            return nn.Sequential(
                nn.Linear(10, 16),
                nn.ReLU(),
                nn.Linear(16, 5),
                nn.Softmax(dim=-1)
            )
        
        # Simple evaluator
        def evaluator(policy):
            with torch.no_grad():
                x = torch.randn(10)
                out = policy(x)
                fitness = float(out.max())
            bd = np.random.rand(2)
            return fitness, bd
        
        archive = MAPElitesArchive(dims=(5, 5))
        trainer = MAPElitesTrainer(
            policy_factory=policy_factory,
            archive=archive,
            evaluator=evaluator,
            mutation_sigma=0.1
        )
        
        # Initialize
        trainer.initialize(n_random=10)
        all_passed &= test_result("Trainer initialization", archive.coverage() > 0)
        
        # Steps
        initial_qd = archive.qd_score()
        for _ in range(20):
            trainer.step()
        all_passed &= test_result("Trainer step runs", True)
        
    except Exception as e:
        all_passed &= test_result("MAPElitesTrainer", False, str(e))
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Test 6: Integration with existing trainers (if available)
    # -------------------------------------------------------------------------
    print("\n[6] Integration Check")
    try:
        from ddpg_trainer import DDPGTrainer
        from deep_q_learning_trainer import DeepQLearningTrainer
        
        all_passed &= test_result("DDPGTrainer has use_ns parameter", 
                                  'use_ns' in DDPGTrainer.__init__.__code__.co_varnames)
        all_passed &= test_result("DQNTrainer has use_ns parameter",
                                  'use_ns' in DeepQLearningTrainer.__init__.__code__.co_varnames)
        
    except ImportError as e:
        all_passed &= test_result("Trainer imports", False, f"Import error: {e}")
    except Exception as e:
        all_passed &= test_result("Trainer integration check", False, str(e))
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    if all_passed:
        print("ALL SMOKE TESTS PASSED ✓")
        print("="*60 + "\n")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_tests()
    sys.exit(exit_code)