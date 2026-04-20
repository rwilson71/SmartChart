from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.p_exhaustion import build_exhaustion_latest_payload


CACHE_PATH = Path("data/cache/exhaustion_latest.json")
REFRESH_SECONDS = 10


def build_exhaustion_cache() -> Dict[str, Any]:
    df = load_price_data().tail(3000).copy()

    if df.empty:
        raise ValueError("Exhaustion cache build error: empty dataset")

    payload = build_exhaustion_latest_payload(df=df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_exhaustion_build() -> None:
    try:
        payload = build_exhaustion_cache()

        print(
            f"Exhaustion cache updated: "
            f"state={payload.get('state')} "
            f"bias={payload.get('bias_label')} "
            f"time={payload.get('timestamp')}"
        )

    except Exception as e:
        print(f"Exhaustion cache build error: {e}")


if __name__ == "__main__":
    print("Starting Exhaustion cache builder...")
    while True:
        run_exhaustion_build()
        time.sleep(REFRESH_SECONDS)