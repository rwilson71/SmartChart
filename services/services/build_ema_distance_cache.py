from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.cc_ema_distance_calibration import (
    EmaDistanceConfig,
    build_latest_payload,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "xauusd.csv"

CACHE_TTL_SECONDS = 10.0

_CACHE: Dict[str, Dict[str, Any]] = {
    "latest": {
        "timestamp": 0.0,
        "payload": None,
    }
}


def load_price_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    path = data_path or DATA_PATH
    df = pd.read_csv(path)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    return df


def get_ema_distance_latest(
    force_refresh: bool = False,
    cfg: Optional[EmaDistanceConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceConfig()

    now = time.time()
    cached = _CACHE["latest"]

    if (
        not force_refresh
        and cached["payload"] is not None
        and (now - float(cached["timestamp"])) < CACHE_TTL_SECONDS
    ):
        return cached["payload"]

    df = load_price_data()
    payload = build_latest_payload(df, cfg)

    cached["timestamp"] = now
    cached["payload"] = payload

    return payload


def get_ema_distance_live_result(
    force_refresh: bool = False,
    cfg: Optional[EmaDistanceConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceConfig()
    latest = get_ema_distance_latest(force_refresh=force_refresh, cfg=cfg)

    return {
        "module": cfg.module_name,
        "debug_version": cfg.debug_version,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "latest": latest,
    }