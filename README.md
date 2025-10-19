Example for repo structure:

open-endedness_for_po/
├─ testing_algo.py # keep for quick smoke tests
├─ deep_q_learning.py # DQN agent
├─ ddpg.py # DDPG agent
├─ deep_q_learning_trainer.py # DQN trainer
├─ ddpg_trainer.py # DDPG trainer
├─ rl_evaluator.py # evaluation utils (extend)
├─ predictors.py # keep ML blocks here (LSTM/MLP/etc.)
├─ utility_functions.py # keep common utils here
├─ baselines/
│ ├─ classical.py # Markowitz, MinVar, Risk Parity, EW
│ └─ econometric.py # ARIMA/GARCH/Kalman (optional)
├─ qd/
│ ├─ novelty_metrics.py # behavior descriptors, k-NN distance
│ ├─ map_elites.py # optional QD container
│ └─ wrappers.py # adapters that wrap DQN/DDPG with QD/NS
├─ core/
│ ├─ interfaces.py # Defines a common class template (interface) for all methods (classical, RL, ML); BaseStrategy, FitConfig, StepOutput
│ ├─ registry.py # Keeps a dictionary of all available models so you can call them by name; METHOD_REGISTRY = {"MVP": ..., "DQN": ...}
│ ├─ data.py # Loads and splits your CSV data (train, validation, test)
│ ├─ metrics.py # Defines standard evaluation metrics (Sharpe ratio, drawdown, turnover..)
│ └─ runner.py # Runs experiments: loads data → trains model → tests model → saves results; the orchestrator (CLI callable)
├─ configs/
│ ├─ mvp.yaml
│ ├─ dqn.yaml
│ ├─ ddpg.yaml
│ ├─ dqn_qd.yaml
│ ├─ lstm.yaml
│ └─ mlp.yaml
└─ notebooks/
└─ 00_compare_methods.ipynb # your orchestrating notebook
