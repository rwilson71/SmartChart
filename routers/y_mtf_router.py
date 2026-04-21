from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.y_mtf import build_mtf_latest_payload


router = APIRouter()

CACHE_PATH = Path("data/cache/y_mtf_latest.json")


@router.get("/website/y-mtf/latest")
def get_y_mtf_latest() -> Dict[str, Any]:
    # Cache-first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read Y_MTF cache: {e}"
            )

    # Live fallback
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="No price data available for Y_MTF live build"
            )

        return build_mtf_latest_payload(df=df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build Y_MTF payload live: {e}"
        )