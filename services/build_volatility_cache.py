from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.n_volatility import build_volatility_latest_payload
from core.data_loader import load_price_data


CACHE_PATH = Path("data/cache/volatility_latest.json")
REFRESH_SECONDS = 10


def build_volatility_cache() -> Dict[str, Any]:
    df = load_price_data().tail(5000).copy()

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("No valid price data returned for volatility cache build.")

    payload = build_volatility_latest_payload(df)

    if not payload:
        raise ValueError("Volatility payload build returned empty payload.")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload


def run_volatility_build() -> None:
    try:
        payload = build_volatility_cache()
        print(
            "Volatility cache updated:",
            CACHE_PATH,
            "| state=",
            payload.get("state"),
            "| bias_signal=",
            payload.get("bias_signal"),
            "| bias_label=",
            payload.get("bias_label"),
        )
    except Exception as e:
        print(f"Volatility cache build failed: {e}")


if __name__ == "__main__":
    print("Starting volatility cache builder...")
    while True:
        run_volatility_build()
        time.sleep(REFRESH_SECONDS)