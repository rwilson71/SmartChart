from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.vf_pattern import build_pattern_latest_payload


CACHE_PATH = Path("data/cache/forecaster_pattern_latest.json")
REFRESH_SECONDS = 10


def build_vf_pattern_cache() -> Dict[str, Any]:
    df = load_price_data().tail(50000).copy()

    if df.empty:
        raise ValueError("Pattern cache build error: empty dataset")

    payload = build_pattern_latest_payload(df=df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CACHE_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_vf_pattern_cache_loop() -> None:
    print("Starting Pattern cache loop...")

    while True:
        try:
            build_vf_pattern_cache()
        except Exception as e:
            print(f"Pattern cache error: {e}")

        time.sleep(REFRESH_SECONDS)