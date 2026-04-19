from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from core.n_volatility import build_volatility_latest_payload
from core.data_loader import load_price_data


router = APIRouter()

CACHE_PATH = Path("data/cache/volatility_latest.json")


def _load_cache() -> Dict[str, Any]:
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read volatility cache: {e}"
        )


def _build_live() -> Dict[str, Any]:
    try:
        df = load_price_data().tail(5000).copy()

        if df is None or df.empty:
            raise HTTPException(
                status_code=500,
                detail="Price data is empty"
            )

        payload = build_volatility_latest_payload(df)

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="Volatility payload build returned empty"
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build live volatility payload: {e}"
        )


@router.get("/website/volatility/latest")
def get_volatility_latest() -> Dict[str, Any]:
    """
    SmartChart Volatility Endpoint
    - Cache-first
    - Live fallback
    - Website contract ready
    """

    if CACHE_PATH.exists():
        payload = _load_cache()
    else:
        payload = _build_live()

    # Optional debug/version tag (VERY useful later)
    payload["debug_source"] = "cache" if CACHE_PATH.exists() else "live"
    payload["debug_endpoint"] = "volatility_latest_v1"

    return payload