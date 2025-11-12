# qd/novelty_archive.py
import numpy as np

class NoveltyArchive:
    def __init__(self, k: int = 10, add_every: int = 1, max_size: int = 5000, seed: int = 42):
        self.k = k
        self.add_every = add_every
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)
        self._descs = []  # list of np.ndarray
        self._t = 0

    def __len__(self):
        return len(self._descs)

    def novelty(self, desc: np.ndarray) -> float:
        if len(self._descs) == 0:
            return 1.0  # maximum novelty when archive is empty
        X = np.vstack(self._descs)
        # squared Euclidean
        d2 = np.sum((X - desc[None, :])**2, axis=1)
        k = min(self.k, len(d2))
        near = np.partition(d2, k-1)[:k]
        return float(np.mean(np.sqrt(near)))

    def maybe_add(self, desc: np.ndarray) -> None:
        self._t += 1
        if (self._t % self.add_every) != 0:
            return
        if len(self._descs) >= self.max_size:
            # random replacement to keep distribution broad
            idx = self.rng.integers(0, self.max_size)
            self._descs[idx] = desc.astype(np.float32)
        else:
            self._descs.append(desc.astype(np.float32))
