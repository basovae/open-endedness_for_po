import requests
import pandas as pd

API_KEY = "APHP5IL6O7HNLJGU"
symbol = "SPY"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}"
r = requests.get(url)
data = r.json()

# Check what keys exist
print(data.keys())


df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
df.index = pd.to_datetime(df.index)
df = df.astype(float)
    
    # Save to CSV
df.to_csv(f"{symbol}_daily.csv")
print(f"Saved {symbol}_daily.csv with {len(df)} rows")
print(df.head())
