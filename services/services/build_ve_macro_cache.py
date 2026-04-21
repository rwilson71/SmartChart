from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.ve_macro import build_macro_latest_payload


CACHE_PATH = Path("data/cache/forecaster_macro_latest.json")
REFRESH_SECONDS = 10


def build_ve_macro_cache() -> Dict[str, Any]:
    df = load_price_data().tail(50000).copy()

    if df.empty:
        raise ValueError("Macro cache build error: empty dataset")

    payload = build_macro_latest_payload(df=df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CACHE_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def run_ve_macro_cache_loop() -> None:
    print("Starting Macro cache loop...")

    while True:
        try:
            build_ve_macro_cache()
        except Exception as e:
            print(f"Macro cache error: {e}")

        time.sleep(REFRESH_SECONDS)