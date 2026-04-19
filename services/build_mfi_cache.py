from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.q_mfi import build_mfi_latest_payload


CACHE_PATH = Path("data/cache/mfi_latest.json")
REFRESH_SECONDS = 10


def run_mfi_build() -> Dict[str, Any] | None:
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            print("MFI build error: empty dataset")
            return None

        payload = build_mfi_latest_payload(df)

        if not payload:
            print("MFI build error: empty payload")
            return None

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        print(f"MFI cache updated: {CACHE_PATH}")
        return payload

    except Exception as e:
        print(f"MFI build error: {e}")
        return None


if __name__ == "__main__":
    while True:
        run_mfi_build()
        time.sleep(REFRESH_SECONDS)