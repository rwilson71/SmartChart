import json
import time
from pathlib import Path

import pandas as pd

from core.data_loader import load_price_data
from core.l_confluence_cloud import build_confluence_cloud_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/confluence_cloud_latest.json")
REFRESH_SECONDS = 10


def run_confluence_cloud_build() -> None:
    try:
        df = load_price_data()

        if df is None or df.empty:
            print("Confluence Cloud cache build error: empty dataset")
            return

        payload = build_confluence_cloud_latest_payload(df)

        if not payload:
            print("Confluence Cloud cache build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        print(f"Confluence Cloud cache updated: {CACHE_PATH}")

    except Exception as e:
        print(f"Confluence Cloud cache build failed: {e}")


def main() -> None:
    while True:
        run_confluence_cloud_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()