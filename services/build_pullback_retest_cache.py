from __future__ import annotations

import json
import time
from pathlib import Path

from core.data_loader import load_price_data
from core.t_pullback_retest import build_pullback_retest_latest_payload

CACHE_PATH = Path("data/cache/pullback_retest_latest.json")
REFRESH_SECONDS = 10


def ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_pullback_retest_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("Pullback/Retest build error: empty dataset")
            return

        payload = build_pullback_retest_latest_payload(df)

        if not payload:
            print("Pullback/Retest build error: empty payload")
            return

        ensure_cache_dir()

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(
            f"Pullback/Retest cache updated successfully: "
            f"{payload.get('timestamp', 'no timestamp')}"
        )

    except Exception as e:
        print(f"Pullback/Retest cache build failed: {e}")


def main() -> None:
    print("Starting Pullback/Retest cache builder...")

    while True:
        run_pullback_retest_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()