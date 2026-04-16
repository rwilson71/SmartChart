import json
import time
from pathlib import Path

import pandas as pd

from core.data_loader import load_price_data
from core.k_orderflow import build_orderflow_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/orderflow_latest.json")
REFRESH_SECONDS = 10


def ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_orderflow_build() -> None:
    try:
        df = load_price_data()

        if df is None or df.empty:
            print("Orderflow build error: empty dataset")
            return

        payload = build_orderflow_latest_payload(df)

        if not payload:
            print("Orderflow build error: empty payload")
            return

        ensure_cache_dir()
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(
            "Orderflow cache updated:",
            payload.get("timestamp", "unknown"),
            "| dir:",
            payload.get("state", {}).get("text", "unknown"),
        )

    except Exception as e:
        print(f"Orderflow build error: {e}")


def main() -> None:
    print("Starting Orderflow cache builder...")
    ensure_cache_dir()

    while True:
        run_orderflow_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()