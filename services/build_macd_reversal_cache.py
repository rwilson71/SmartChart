from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.s_macd_reversal import build_macd_reversal_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/macd_reversal_latest.json")


@router.get("/website/macd-reversal/latest")
def get_macd_reversal_latest():
    # Serve cache first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read MACD Reversal cache: {e}"
            )

    # Fallback to live build
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Price data is empty"
            )

        payload = build_macd_reversal_latest_payload(df)

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="MACD Reversal payload build returned empty result"
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build MACD Reversal payload: {e}"
        )