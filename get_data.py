import yfinance as yf
import pandas as pd
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Download Gold futures (closest to XAUUSD)
df = yf.download("GC=F", period="60d", interval="5m")

# Reset index
df = df.reset_index()

# Rename columns to match SmartChart
df.rename(columns={
    "Datetime": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
}, inplace=True)

# Keep only needed columns
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# Save to CSV
df.to_csv("data/XAUUSD.csv", index=False)

print("✅ Data saved to data/XAUUSD.csv")
