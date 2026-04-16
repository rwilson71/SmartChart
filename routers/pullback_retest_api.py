from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.t_pullback_retest import build_pullback_retest_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/pullback_retest_latest.json")


@router.get("/website/pullback-retest/latest")
def get_pullback_retest_latest():
    # Serve cache first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read Pullback/Retest cache: {e}"
            )

    # Fallback to live build
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Price data is empty"
            )

        payload = build_pullback_retest_latest_payload(df)

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="Pullback/Retest payload build returned empty result"
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build Pullback/Retest payload: {e}"
        )