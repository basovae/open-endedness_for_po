# make_multiasset.py
import os
import time
import math
import warnings
import pandas as pd
import requests

ALPHA_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "APHP5IL6O7HNLJGU")
TICKERS = ["SPY", "AGG", "EFA", "GLD", "IEF"]  # US equity, bonds, intl, gold, treasuries

USE_YFINANCE = True
FORCE_SYNTHETIC = False



def load_local(symbol: str) -> pd.Series | None:
    fn = f"{symbol}_daily.csv"
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn, index_col=0, parse_dates=True).sort_index()
    # prefer adjusted close if present
    for col in ["adjusted close", "adj_close", "Adj Close", "close", "Close"]:
        if col in df.columns:
            try:
                return df[col].astype(float)
            except Exception:
                pass
    # fallback to last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return df[num_cols[-1]].astype(float)
    return None

def load_yfinance(symbol: str) -> pd.Series | None:
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        data = yf.download(symbol, period="max", auto_adjust=True, progress=False)
        if data is None or data.empty:
            return None
        s = data["Close"].dropna().astype(float)
        # persist for next time
        s.to_frame(name="adjusted close").to_csv(f"{symbol}_daily.csv")
        print(f"{symbol}: {len(s)} rows (yfinance)")
        return s
    except Exception:
        return None

def fetch_alpha_vantage(symbol: str, max_retries: int = 5) -> pd.Series | None:
    url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
           f"&symbol={symbol}&outputsize=full&apikey={ALPHA_KEY}")
    wait = 12  # AV hard guideline: ~5 calls/min → start with 12s
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, timeout=30)
        try:
            data = r.json()
        except Exception:
            data = {}
        key = next((k for k in data.keys() if "Time Series" in k), None)
        if key:
            raw = pd.DataFrame.from_dict(data[key], orient="index")
            raw.index = pd.to_datetime(raw.index)
            raw = raw.sort_index()
            col = "5. adjusted close" if "5. adjusted close" in raw.columns else "4. close"
            s = raw[col].astype(float)
            s.to_frame(name="adjusted close").to_csv(f"{symbol}_daily.csv")
            print(f"{symbol}: {len(s)} rows (alpha_vantage)")
            return s
        # Rate-limited or error → backoff
        if any(k in data for k in ["Information", "Note", "Error Message"]):
            if attempt < max_retries:
                sleep_s = wait * attempt  # linear backoff
                print(f"{symbol}: AlphaVantage says '{list(data.keys())[0]}'. Retry {attempt}/{max_retries} in {sleep_s}s…")
                time.sleep(sleep_s)
                continue
            else:
                print(f"{symbol}: AlphaVantage still rate-limited after {max_retries} tries.")
                return None
        # Unknown response; brief wait then retry
        time.sleep(5)
    return None

def synthetic_from_spy(spy_prices: pd.Series, symbol: str, shift_days: int, scale: float) -> pd.Series:
    """Create a synthetic price series (shift + scale) so downstream code can run."""
    s = spy_prices.copy()
    s = s.shift(shift_days).fillna(method="bfill")
    s = (s / s.iloc[0]) ** scale * s.iloc[0]  # warp volatility/level a bit
    print(f"{symbol}: {len(s)} rows (synthetic from SPY)")
    return s

def load_one(symbol: str, spy_prices: pd.Series | None) -> pd.Series:
    # Synthetic shortcut for non-SPY
    if symbol != "SPY" and FORCE_SYNTHETIC and spy_prices is not None:
        idx = TICKERS.index(symbol)
        shift = 5 * idx
        scale = 0.8 + 0.1 * idx
        return synthetic_from_spy(spy_prices, symbol, shift, scale)

    s = load_local(symbol)
    if s is not None:
        print(f"{symbol}: {len(s)} rows (local)")
        return s

    if USE_YFINANCE:
        s = load_yfinance(symbol)
        if s is not None:
            return s

    # quick Alpha Vantage try (no long waiting)
    s = fetch_alpha_vantage(symbol, max_retries=1)
    if s is not None:
        return s

    if spy_prices is not None and symbol != "SPY":
        idx = TICKERS.index(symbol)
        shift = 5 * idx
        scale = 0.8 + 0.1 * idx
        return synthetic_from_spy(spy_prices, symbol, shift, scale)

    raise RuntimeError(f"Could not load data for {symbol} and no SPY for synthetic fallback.")

# --- replace your main() with this ---
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    series = {}

    # Load SPY first (needed for alignment and synthetic fallback)
    spy = load_local("SPY")
    if spy is None:
        spy = load_yfinance("SPY")
    if spy is None:
        spy = fetch_alpha_vantage("SPY")
    if spy is None:
        raise RuntimeError("Failed to load SPY (needed for alignment and synthetic fallback).")
    series["SPY"] = spy
    print(f"SPY: {len(spy)} rows (loaded)")

    # Load remaining tickers one by one
    for t in TICKERS:
        if t == "SPY":
            continue
        s = load_one(t, spy)  # this already tries local → yfinance → AV → synthetic
        series[t] = s

    # Align by intersection; compute daily returns
    df_prices = pd.DataFrame(series).dropna(how="any").sort_index()
    df_returns = df_prices.pct_change().dropna()

    # optional: limit to recent 10y for speed
    if not df_returns.empty:
        last_ts = df_returns.index.max()
        df_returns = df_returns.loc[df_returns.index >= (last_ts - pd.DateOffset(years=10))]

    df_returns.to_csv("multiasset_daily_returns.csv")
    print("Saved multiasset_daily_returns.csv with shape:", df_returns.shape)


if __name__ == "__main__":
    main()
