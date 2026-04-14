import pandas as pd
from core.c_ema import run_ema_engine

df = pd.read_csv("data/xauusd_mt5_m1.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime").sort_index()

result = run_ema_engine(df)

cols = [
    "ema14", "ema20", "ema33", "ema50", "ema100", "ema200",
    "sc_ema_mtf1", "sc_ema_mtf5", "sc_ema_mtf15", "sc_ema_mtf60", "sc_ema_mtf240", "sc_ema_mtfD",
    "sc_ema_mtf_avg_dir",
    "sc_ema_local_dir",
    "sc_ema_final_dir",
    "sc_ema_rt_1420", "sc_ema_rt_3350", "sc_ema_rt_100200",
    "sc_ema_reclaim_20", "sc_ema_reclaim_3350", "sc_ema_reclaim_100200"
]

print(result[cols].tail(20).to_string())