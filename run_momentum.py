from core.d_momentum import run_d_momentum
import pandas as pd

print("Using MT5 data...")

df = pd.read_csv("data/xauusd_mt5_m1.csv")

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
df.set_index("datetime", inplace=True)

result = run_d_momentum(df)

print(result[[
    "close",
    "mom_rsi",
    "mom_bull_score",
    "mom_bear_score",
    "mom_raw_bias_state",
    "sc_mom_state",
    "sc_mom_dir"
]].tail(5))