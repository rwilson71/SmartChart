from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from core.data_loader import load_price_data
from core.s_macd_reversal import build_macd_reversal_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/macd_reversal_latest.json")
REFRESH_SECONDS = 10


def ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_macd_reversal_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("MACD Reversal build error: empty dataset")
            return

        payload = build_macd_reversal_latest_payload(df)

        if not payload:
            print("MACD Reversal build error: empty payload")
            return

        ensure_cache_dir()

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(
            f"MACD Reversal cache updated successfully: "
            f"{payload.get('timestamp', 'no timestamp')}"
        )

    except Exception as e:
        print(f"MACD Reversal cache build failed: {e}")


def main() -> None:
    print("Starting MACD Reversal cache builder...")

    while True:
        run_macd_reversal_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()