import json
import time
from pathlib import Path

import pandas as pd

from core.data_loader import load_price_data
from core.r_confluence import build_confluence_latest_payload

CACHE_PATH = Path("data/cache/confluence_latest.json")
REFRESH_SECONDS = 10


def run_confluence_build() -> None:
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            print("Confluence build error: empty dataset")
            return

        payload = build_confluence_latest_payload(df)

        if not payload:
            print("Confluence build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Confluence cache updated: {CACHE_PATH}")

    except Exception as e:
        print(f"Confluence build error: {e}")


if __name__ == "__main__":
    while True:
        run_confluence_build()
        time.sleep(REFRESH_SECONDS)