from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.m_volume import build_volume_latest_payload
from services.market_data import get_ohlcv_df


CACHE_PATH = Path("data/cache/volume_latest.json")


def build_volume_cache() -> Dict[str, Any]:
    df = get_ohlcv_df()

    if df is None or len(df) == 0:
        raise ValueError("No OHLCV data returned from market data service.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Market data service did not return a DataFrame.")

    payload = build_volume_latest_payload(df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload


if __name__ == "__main__":
    result = build_volume_cache()
    print("Volume cache built successfully.")
    print(result.get("timestamp"))