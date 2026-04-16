import json
import time
from pathlib import Path

import pandas as pd

from core.n_volatility import build_volatility_latest_payload
from core.data_loader import load_price_data

CACHE_PATH = Path("data/cache/volatility_latest.json")
REFRESH_SECONDS = 10


def run_volatility_build() -> None:
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            print("Volatility cache build error: empty dataset")
            return

        payload = build_volatility_latest_payload(df)

        if not payload:
            print("Volatility cache build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        print(f"Volatility cache updated: {CACHE_PATH}")
    except Exception as e:
        print(f"Volatility cache build failed: {e}")


if __name__ == "__main__":
    print("Starting volatility cache builder...")
    while True:
        run_volatility_build()
        time.sleep(REFRESH_SECONDS)