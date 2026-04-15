import json
import time
from pathlib import Path

import pandas as pd

from core.b_trend import build_trend_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/trend_latest.json")
REFRESH_SECONDS = 10


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    return df


def run_trend_build() -> None:
    try:
        df = load_price_data()
        payload = build_trend_latest_payload(df)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("Trend cache updated")

    except Exception as e:
        print(f"Trend build error: {e}")


if __name__ == "__main__":
    while True:
        run_trend_build()
        time.sleep(REFRESH_SECONDS)