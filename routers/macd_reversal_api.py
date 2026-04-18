import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.s_macd_reversal import build_macd_reversal_latest_payload
from core.data_loader import load_price_data

router = APIRouter()

CACHE_PATH = Path("data/cache/macd_reversal_latest.json")


@router.get("/website/macd-reversal/latest")
def get_macd_reversal_latest():
    # Use cache if exists
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read MACD Reversal cache: {e}"
            )

    # Fallback → live build
    try:
        df = load_price_data()
        return build_macd_reversal_latest_payload(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"MACD Reversal live build failed: {e}"
        )