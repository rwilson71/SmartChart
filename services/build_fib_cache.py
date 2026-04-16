import json
import time
from pathlib import Path

import pandas as pd

from core.h_fib import build_fib_latest_payload
from core.data_loader import load_price_data

CACHE_PATH = Path("data/cache/fib_latest.json")
REFRESH_SECONDS = 10


def ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_fib_build() -> None:
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            print("Fib cache build error: empty dataset")
            return

        payload = build_fib_latest_payload(df)

        if not payload:
            print("Fib cache build error: empty payload")
            return

        ensure_cache_dir()

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Fib cache updated: {payload.get('timestamp', 'unknown')}")

    except Exception as e:
        print(f"Fib cache build error: {e}")


def main() -> None:
    print("Starting Fib cache builder...")
    ensure_cache_dir()

    while True:
        run_fib_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()