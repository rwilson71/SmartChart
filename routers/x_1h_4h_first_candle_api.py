import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.x_1h_4h_first_candle import build_1h_4h_first_candle_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/htf_first_candle_latest.json")


@router.get("/website/htf-first-candle/latest")
def get_htf_first_candle_latest():
    # Cache first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read HTF First Candle cache: {e}"
            )

    # Live fallback
    try:
        df = load_price_data().tail(3000).copy()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="HTF First Candle live build failed: price data is empty."
            )

        payload = build_1h_4h_first_candle_payload(df)

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="HTF First Candle live build failed: empty payload."
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"HTF First Candle live build failed: {e}"
        )