from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.m_volume import build_volume_latest_payload


CACHE_PATH = Path("data/cache/volume_latest.json")


def _get_ohlcv_df() -> pd.DataFrame:
    """
    Pull the live OHLCV dataframe from the project's existing data pipeline.
    Lazy import avoids top-level circular import issues.
    """
    from main_api import get_ohlcv_df

    df = get_ohlcv_df()

    if df is None or len(df) == 0:
        raise ValueError("No OHLCV data returned.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("OHLCV loader did not return a pandas DataFrame.")

    return df


def build_volume_cache() -> Dict[str, Any]:
    df = _get_ohlcv_df()
    payload = build_volume_latest_payload(df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload


if __name__ == "__main__":
    result = build_volume_cache()
    print("Volume cache built successfully.")
    print(result.get("timestamp"))