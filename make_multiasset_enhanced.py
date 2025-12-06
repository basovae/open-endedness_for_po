import pandas as pd
import numpy as np

def add_features(returns_df):
    """Add momentum and volatility features"""
    all_features = [returns_df]  # Start with returns
    
    # Momentum (4 per asset)
    for col in returns_df.columns:
        all_features.append(pd.DataFrame({
            f'{col}_mom5': returns_df[col].rolling(5).mean(),
            f'{col}_mom20': returns_df[col].rolling(20).mean(),
            f'{col}_mom60': returns_df[col].rolling(60).mean(),
            f'{col}_mom_acc': returns_df[col].rolling(20).mean() - returns_df[col].rolling(60).mean()
        }))
    
    # Volatility (3 per asset)
    for col in returns_df.columns:
        sq = returns_df[col] ** 2
        all_features.append(pd.DataFrame({
            f'{col}_vol20': returns_df[col].rolling(20).std(),
            f'{col}_vol60': returns_df[col].rolling(60).std(),
            f'{col}_ewma': np.sqrt(sq.ewm(alpha=0.06).mean())
        }))
    
    return pd.concat(all_features, axis=1).dropna()

# Load existing data
df = pd.read_csv("multiasset_daily_returns.csv", index_col=0, parse_dates=True)

# Add features
enhanced = add_features(df)

# Save
enhanced.to_csv("multiasset_daily_returns_enhanced.csv")

print(f"✅ Enhanced: {df.shape[1]} assets × 8 features = {enhanced.shape[1]} columns")