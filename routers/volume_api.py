from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from core.m_volume import build_volume_latest_payload
from services.build_volume_cache import build_volume_cache
from services.market_data import get_ohlcv_df


router = APIRouter(prefix="/website/volume", tags=["website-volume"])

CACHE_PATH = Path("data/cache/volume_latest.json")


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError("Volume cache file not found.")

    with CACHE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_live_payload() -> Dict[str, Any]:
    df = get_ohlcv_df()

    if df is None or len(df) == 0:
        raise ValueError("No OHLCV data returned from market data service.")

    return build_volume_latest_payload(df)


@router.get("/latest")
def get_volume_latest() -> Dict[str, Any]:
    try:
        return _load_cache()
    except Exception:
        try:
            return _build_live_payload()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load volume payload from cache or live build: {e}",
            )


@router.get("/rebuild")
def rebuild_volume_latest() -> Dict[str, Any]:
    try:
        return build_volume_cache()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild volume cache: {e}",
        )