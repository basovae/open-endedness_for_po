"""
Simple Portfolio Data Retrieval - 50+ ETFs, 20+ Years
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 50+ ETFs with 20+ year history (no survivorship bias)
ASSETS = {
    # US Equity - Sectors (11)
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLP': 'Staples',
    'XLY': 'Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Materials', 'XLC': 'Communication',
    
    # US Equity - Size/Style (9)
    'SPY': 'Large Cap', 'IWM': 'Small Cap', 'MDY': 'Mid Cap',
    'VTV': 'Value', 'VUG': 'Growth', 'MTUM': 'Momentum',
    'QUAL': 'Quality', 'SIZE': 'Size Factor', 'USMV': 'Min Vol',
    
    # International (10)
    'EFA': 'Developed ex-US', 'EEM': 'Emerging Markets',
    'VGK': 'Europe', 'EWJ': 'Japan', 'FXI': 'China',
    'EWY': 'South Korea', 'INDA': 'India', 'EWZ': 'Brazil',
    'EWG': 'Germany', 'EWU': 'UK',
    
    # Fixed Income (10)
    'AGG': 'Total Bond', 'TLT': 'Long Treasury', 'IEF': 'Intermediate Treasury',
    'SHY': 'Short Treasury', 'LQD': 'IG Corporate', 'HYG': 'High Yield',
    'MUB': 'Municipal', 'TIP': 'TIPS', 'EMB': 'EM Bonds',
    'BND': 'Total Bond Market',
    
    # Alternatives (7)
    'GLD': 'Gold', 'SLV': 'Silver', 'VNQ': 'REITs',
    'DBA': 'Agriculture', 'GSG': 'Commodities',
    'IAU': 'Gold Alt', 'GDX': 'Gold Miners',

    'VOO': 'Vanguard S&P 500',
    'QQQ': 'Nasdaq 100',
    'SCHD': 'US Dividend',
    'VXUS': 'Total International',
    'VWO': 'Emerging Markets',
}

def main():
    print("="*80)
    print("DOWNLOADING PORTFOLIO DATA")
    print("="*80)
    print(f"\nAssets: {len(ASSETS)}")
    print("Period: 20 years")
    
    # Download data
    tickers = list(ASSETS.keys())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20*365)
    
    print("\nDownloading...")
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
        group_by='ticker'  # This is the fix!
    )
    
    # Extract close prices (handle different return formats)
    prices = {}
    min_days = 20 * 252 * 0.8  # ~20 years with 80% threshold
    
    print("\n" + "="*80)
    print("PROCESSING ASSETS")
    print("="*80)
    
    for ticker in tickers:
        try:
            # Try to get Close prices
            if len(tickers) == 1:
                close_prices = data['Close']
            else:
                close_prices = data[ticker]['Close']
            
            # Check if we have enough data
            valid_data = close_prices.dropna()
            
            if len(valid_data) >= min_days:
                prices[ticker] = valid_data
                print(f"✅ {ticker}: {len(valid_data)} days ({valid_data.index[0].date()} to {valid_data.index[-1].date()})")
            else:
                print(f"⚠️  {ticker}: Only {len(valid_data)} days - SKIPPED (need {int(min_days)}+)")
        
        except Exception as e:
            print(f"❌ {ticker}: Failed - {e}")
    
    if not prices:
        print("\n❌ ERROR: No valid assets downloaded!")
        return
    
    # Create DataFrame and align dates
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.dropna()  # Keep only dates where all assets have data
    
    # Compute returns
    returns = prices_df.pct_change().dropna()
    
    # Clip extreme outliers
    returns = returns.clip(lower=-0.5, upper=0.5)
    
    # Check correlation (detect fake data)
    corr = returns.corr()
    off_diag = corr.values[~np.eye(len(corr), dtype=bool)]
    avg_corr = np.mean(np.abs(off_diag))
    
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}")
    print(f"Valid assets: {len(returns.columns)}")
    print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Trading days: {len(returns)}")
    print(f"Average |correlation|: {avg_corr:.3f}")
    
    if avg_corr > 0.85:
        print("\n⚠️  WARNING: High correlations - possible fake data!")
    else:
        print("\n✅ Correlations look realistic")
    
    # Save
    returns.to_csv("multiasset_daily_returns.csv")
    
    print(f"\n{'='*80}")
    print("✅ SAVED: multiasset_daily_returns.csv")
    print(f"{'='*80}")
    print(f"Shape: {returns.shape}")
    print(f"Assets: {list(returns.columns)}")
    print("\nReady to use with your existing code!")

if __name__ == "__main__":
    main()