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


def _fallback_payload(
    cfg: EmaDistanceConfig,
    message: str,
) -> Dict[str, Any]:
    return {
        "module": cfg.module_name,
        "debug_version": cfg.debug_version,
        "ready": False,
        "timestamp": None,
        "state": "Transition",
        "bias_signal": 0,
        "bias_label": "NEUTRAL",
        "indicator_strength": 0.0,
        "market_bias": "NEUTRAL",
        "stage": 0,
        "stage_label": "Transition",
        "distance_pct": None,
        "distance_signed_pct": None,
        "ema20": None,
        "ema200": None,
        "ema20_slope_pct": None,
        "ema200_slope_pct": None,
        "trend_side": 0,
        "price_side_vs_ema20": 0,
        "bars_in_state": 0,
        "message": message,
    }


def load_price_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    path = data_path or DATA_PATH

    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.dropna(subset=[col]).set_index(col)
            except Exception:
                pass
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

    try:
        df = load_price_data()
        payload = build_latest_payload(df, cfg)
    except Exception as exc:
        payload = _fallback_payload(
            cfg,
            f"EMA Distance cache fallback triggered: {exc}",
        )

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