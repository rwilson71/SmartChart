from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.o_ob_os import build_ob_os_latest_payload


router = APIRouter()

CACHE_PATH = Path("data/cache/ob_os_latest.json")


def _load_cache() -> Dict[str, Any]:
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read OB/OS cache: {e}"
        )


def _build_live() -> Dict[str, Any]:
    try:
        df = load_price_data()

        if df is None or df.empty:
            raise HTTPException(
                status_code=500,
                detail="OB/OS live build failed: empty dataset"
            )

        payload = build_ob_os_latest_payload(df)

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="OB/OS live build failed: empty payload"
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OB/OS live build failed: {e}"
        )


@router.get("/website/ob-os/latest")
def get_ob_os_latest() -> Dict[str, Any]:
    """
    SmartChart OB/OS + Divergence Endpoint
    - Cache-first
    - Live fallback
    - Website contract ready
    """

    used_cache = CACHE_PATH.exists()

    if used_cache:
        payload = _load_cache()
    else:
        payload = _build_live()

    payload["debug_source"] = "cache" if used_cache else "live"
    payload["debug_endpoint"] = "ob_os_latest_v1"

    return payload