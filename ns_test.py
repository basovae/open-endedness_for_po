import numpy as np
from qd.wrappers import NSWrapper
from qd.bd_presets import bd_weights_plus_returns

ns = NSWrapper(bd_fn=bd_weights_plus_returns, alpha=1.0, beta=0.5)

# fake 50-step episode with 5 assets
for t in range(50):
    w = np.random.dirichlet(np.ones(5))
    r = np.random.normal(0, 0.01)
    ns.on_step(state=None, action=w, reward_task=r, info={"weights": w, "return_t": r})

score = ns.on_episode_end(episode_task_return=0.12)  # say +12% task return
print("Blended episode score:", score)
