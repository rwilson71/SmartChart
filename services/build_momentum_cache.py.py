import json
import time
from pathlib import Path

import pandas as pd

from core.d_momentum import build_momentum_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/momentum_latest.json")
REFRESH_SECONDS = 10


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    return df


def run_momentum_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("Momentum build error: empty dataset")
            return

        payload = build_momentum_latest_payload(df)

        if not payload:
            print("Momentum build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("Momentum cache updated")

    except Exception as e:
        print(f"Momentum build error: {e}")


if __name__ == "__main__":
    print("Starting Momentum cache service...")

    while True:
        run_momentum_build()
        time.sleep(REFRESH_SECONDS)