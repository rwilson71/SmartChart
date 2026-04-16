import json
import time
from pathlib import Path

import pandas as pd

from core.data_loader import load_price_data
from core.u_regime import build_regime_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/regime_latest.json")
REFRESH_SECONDS = 10


def run_regime_build() -> None:
    try:
        df = load_price_data().tail(2000).copy()

        if df.empty:
            print("Regime build error: empty dataset")
            return

        payload = build_regime_latest_payload(df)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Regime cache updated: {CACHE_PATH}")
    except Exception as e:
        print(f"Regime build error: {e}")


if __name__ == "__main__":
    while True:
        run_regime_build()
        time.sleep(REFRESH_SECONDS)