# qd/wrappers.py
import numpy as np
from typing import Dict, Any, Optional, Callable
from .novelty_archive import NoveltyArchive

class NSWrapper:
    """
    Adds Novelty Search to an existing RL training loop by:
      1) collecting per-episode trajectories,
      2) computing a behavior descriptor (callable you pass),
      3) computing novelty against an archive,
      4) blending task reward with novelty: R' = alpha * R_task + beta * novelty.
    """
    def __init__(
        self,
        bd_fn: Callable[[Dict[str, Any]], np.ndarray],
        alpha: float = 1.0,        # weight for task reward
        beta: float = 1.0,         # weight for novelty
        archive: Optional[NoveltyArchive] = None,
    ):
        self.bd_fn = bd_fn
        self.alpha = alpha
        self.beta = beta
        self.archive = archive or NoveltyArchive()
        # rolling buffers for a single episode
        self.reset_episode_buffers()

    def reset_episode_buffers(self):
        self.buf = {
            "weights_traj": [],     # list of np.ndarray (n_assets,)
            "returns": [],          # list of floats (r_t)
            "actions": [],          # optional
            "states": []            # optional
        }

    def on_step(self, state, action, reward_task: float, info: Dict[str, Any]):
        # you can push more depending on your trainer
        if "weights" in info:
            self.buf["weights_traj"].append(np.asarray(info["weights"], dtype=float))
        if "return_t" in info:
            self.buf["returns"].append(float(info["return_t"]))
        self.buf["actions"].append(np.asarray(action).ravel())
        self.buf["states"].append(np.asarray(state).ravel())
        # step reward remains task reward; novelty is applied at episode end
        return reward_task

    def on_episode_end(self, episode_task_return: float) -> float:
        # build behavior descriptor
        traj = {k: (np.vstack(v) if len(v)>0 and isinstance(v[0], (list, np.ndarray)) else np.array(v))
                for k, v in self.buf.items()}
        desc = self.bd_fn(traj)
        nov = self.archive.novelty(desc)
        self.archive.maybe_add(desc)
        # blend rewards at the episode level
        blended = self.alpha * float(episode_task_return) + self.beta * float(nov)
        # clear buffers for next episode
        self.reset_episode_buffers()
        return blended
