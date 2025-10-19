# run_dql_example.py
import pandas as pd
import torch.nn as nn
from deep_q_learning import DeepQLearning
import predictors

# 1) Load local data
df = pd.read_csv("multiasset_daily_returns.csv", index_col=0, parse_dates=True).sort_index()

# 2) Pick features & split
#features = df[["open","high","low","close","volume"]].pct_change().dropna()
#split = int(len(features)*0.8)
#train_df = features.iloc[:split]
#val_df   = features.iloc[split:-252]   # keep last ~1y for test
#test_df  = features.iloc[-252:]

split1 = int(len(df) * 0.7)
split2 = int(len(df) * 0.85)
train_df = df.iloc[:split1]
val_df   = df.iloc[split1:split2]
test_df  = df.iloc[split2:]

# 3) Configure and train
model = DeepQLearning(
    lookback_window=50,             # number of past days in each state
    predictor=predictors.MLP,       # the function approximator used for the actor/critic
    batch_size=1, # the number of training samples (or experiences) the model processes before updating its weights once
    short_selling=False,            # clamps negative weights (no short selling).
    forecast_window=0,
    reduce_negatives=True,
    verbose=1,
    hidden_sizes=(64, 64),          # forwarded to predictors.MLP via **kwargs 
                                    # architecture of the neural net inside predictors.MLP
)

model.train(
    train_data=train_df,
    val_data=val_df,
    actor_lr=1e-3,
    critic_lr=1e-3,
    num_epochs=5,                      # small smoke test
)

# 4) Evaluate
print("SPO/DPO on held-out test:")
print(model.evaluate(test_data=test_df, dpo=True))

