# experiments/diagnose_data.py
"""
Data Diagnostics for Portfolio Optimization
============================================

Checks for common data issues that might affect results.

Run: python experiments/diagnose_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("="*70)
    print("DATA DIAGNOSTICS")
    print("="*70)
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             "multiasset_daily_returns.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
    
    print(f"\n📊 BASIC INFO")
    print("-"*70)
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total days: {len(df)}")
    print(f"  Assets: {list(df.columns)}")
    
    # Check for missing values
    print(f"\n📊 MISSING VALUES")
    print("-"*70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values")
    else:
        print(f"  ⚠️ Missing values found:")
        print(missing[missing > 0])
    
    # Check for outliers
    print(f"\n📊 RETURN STATISTICS")
    print("-"*70)
    print(f"{'Asset':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Sharpe':>10}")
    print("-"*70)
    
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        sharpe = mean / std * np.sqrt(252)
        print(f"{col:<10} {mean*100:>9.4f}% {std*100:>9.4f}% {min_val*100:>9.2f}% {max_val*100:>9.2f}% {sharpe:>10.2f}")
    
    # Check for extreme values
    print(f"\n📊 OUTLIER CHECK (|return| > 10%)")
    print("-"*70)
    for col in df.columns:
        extreme = (df[col].abs() > 0.10).sum()
        if extreme > 0:
            print(f"  ⚠️ {col}: {extreme} days with |return| > 10%")
    if all((df[col].abs() > 0.10).sum() == 0 for col in df.columns):
        print("  ✓ No extreme outliers (|return| > 10%)")
    
    # Equal Weight performance by period
    print(f"\n📊 EQUAL WEIGHT PERFORMANCE BY PERIOD")
    print("-"*70)
    
    n_assets = len(df.columns)
    ew_returns = df.mean(axis=1)  # Equal weight daily returns
    
    # Split into periods
    yearly_sharpes = {}
    for year in df.index.year.unique():
        year_data = ew_returns[ew_returns.index.year == year]
        if len(year_data) > 20:
            sharpe = year_data.mean() / year_data.std() * np.sqrt(252)
            annual_ret = (1 + year_data.mean()) ** 252 - 1
            yearly_sharpes[year] = sharpe
            print(f"  {year}: Sharpe={sharpe:.2f}, Return={annual_ret*100:.1f}%")
    
    # Check correlation
    print(f"\n📊 ASSET CORRELATIONS")
    print("-"*70)
    corr = df.corr()
    print(corr.round(2))
    
    avg_corr = corr.values[np.triu_indices(n_assets, k=1)].mean()
    print(f"\n  Average pairwise correlation: {avg_corr:.3f}")
    if avg_corr < 0.3:
        print("  ✓ Low correlation - good for diversification")
    elif avg_corr < 0.6:
        print("  ⚠️ Moderate correlation")
    else:
        print("  ⚠️ High correlation - limited diversification benefit")
    
    # Why Equal Weight is winning
    print(f"\n📊 WHY EQUAL WEIGHT MIGHT BE WINNING")
    print("-"*70)
    
    # 1. Check if all assets have positive expected returns
    positive_assets = (df.mean() > 0).sum()
    print(f"  1. Assets with positive mean return: {positive_assets}/{n_assets}")
    if positive_assets == n_assets:
        print("     → All assets profitable = EW benefits from all")
    
    # 2. Check volatility differences
    vols = df.std()
    vol_ratio = vols.max() / vols.min()
    print(f"  2. Volatility ratio (max/min): {vol_ratio:.2f}")
    if vol_ratio < 2:
        print("     → Similar volatilities = no penalty for equal weighting")
    
    # 3. Check Sharpe ratios
    sharpes = df.mean() / df.std() * np.sqrt(252)
    print(f"  3. Individual Sharpe ratios: {sharpes.min():.2f} to {sharpes.max():.2f}")
    if sharpes.min() > 0.5:
        print("     → All assets have good Sharpe = EW captures all upside")
    
    # 4. Test period analysis
    print(f"\n📊 TEST PERIOD ANALYSIS (last 20%)")
    print("-"*70)
    
    test_start = int(len(df) * 0.8)
    test_data = df.iloc[test_start:]
    
    print(f"  Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print(f"  Test days: {len(test_data)}")
    
    for col in df.columns:
        test_sharpe = test_data[col].mean() / test_data[col].std() * np.sqrt(252)
        test_ret = (1 + test_data[col].mean()) ** 252 - 1
        print(f"    {col}: Sharpe={test_sharpe:.2f}, Return={test_ret*100:.1f}%")
    
    # Equal weight in test
    ew_test = test_data.mean(axis=1)
    ew_test_sharpe = ew_test.mean() / ew_test.std() * np.sqrt(252)
    ew_test_ret = (1 + ew_test.mean()) ** 252 - 1
    print(f"\n  Equal Weight: Sharpe={ew_test_sharpe:.2f}, Return={ew_test_ret*100:.1f}%")
    
    # Diagnosis
    print(f"\n📊 DIAGNOSIS")
    print("-"*70)
    
    if ew_test_sharpe > 2.0:
        print("  🔴 VERY HIGH Equal Weight Sharpe in test period!")
        print("     This is unusual and may indicate:")
        print("     - Strong bull market in test period")
        print("     - Low volatility environment")
        print("     - Data might not include challenging periods")
    
    if avg_corr < 0.4 and positive_assets == n_assets:
        print("\n  🔴 LOW CORRELATION + ALL POSITIVE RETURNS")
        print("     This is ideal for Equal Weight!")
        print("     RL would need to find an even better allocation,")
        print("     which is very hard when 1/N is already near-optimal.")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cumulative returns
    ax1 = axes[0, 0]
    cum_returns = (1 + df).cumprod()
    for col in df.columns:
        ax1.plot(cum_returns[col], label=col, linewidth=1)
    ax1.plot((1 + df.mean(axis=1)).cumprod(), 'k--', label='Equal Weight', linewidth=2)
    ax1.set_title('Cumulative Returns (All Data)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Growth of $1')
    ax1.grid(True, alpha=0.3)
    
    # Rolling correlation
    ax2 = axes[0, 1]
    rolling_corr = df[df.columns[0]].rolling(60).corr(df[df.columns[1]])
    ax2.plot(rolling_corr, alpha=0.7)
    ax2.axhline(y=avg_corr, color='red', linestyle='--', label=f'Mean: {avg_corr:.2f}')
    ax2.set_title(f'Rolling 60-Day Correlation ({df.columns[0]} vs {df.columns[1]})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Yearly Sharpe
    ax3 = axes[1, 0]
    years = list(yearly_sharpes.keys())
    sharpes_list = list(yearly_sharpes.values())
    colors = ['green' if s > 1 else 'orange' if s > 0 else 'red' for s in sharpes_list]
    ax3.bar(years, sharpes_list, color=colors, edgecolor='black')
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Equal Weight Sharpe by Year', fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio')
    
    # Test period
    ax4 = axes[1, 1]
    test_cum = (1 + test_data).cumprod()
    for col in test_data.columns:
        ax4.plot(test_cum[col], label=col, linewidth=1)
    ax4.plot((1 + test_data.mean(axis=1)).cumprod(), 'k--', label='Equal Weight', linewidth=2)
    ax4.set_title('Cumulative Returns (Test Period Only)', fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.set_ylabel('Growth of $1')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to data_diagnostics.png")
    
    plt.show()


if __name__ == "__main__":
    main()