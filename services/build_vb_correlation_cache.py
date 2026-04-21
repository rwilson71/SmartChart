from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.vb_correlation import build_correlation_latest_payload


CACHE_PATH = Path("data/cache/forecaster_correlation_latest.json")
REFRESH_SECONDS = 10


def build_vb_correlation_cache() -> Dict[str, Any]:
    df = load_price_data().tail(50000).copy()

    if df.empty:
        raise ValueError("Correlation cache build error: empty dataset")

    payload = build_correlation_latest_payload(df=df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CACHE_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_vb_correlation_cache_loop() -> None:
    print("Starting Correlation cache loop...")

    while True:
        try:
            build_vb_correlation_cache()
        except Exception as e:
            print(f"Correlation cache error: {e}")

        time.sleep(REFRESH_SECONDS)