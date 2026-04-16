import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.c_ema import build_ema_latest_payload
from core.data_loader import load_price_data

router = APIRouter()

CACHE_PATH = Path("data/cache/ema_latest.json")


@router.get("/website/ema/latest")
def get_ema_latest():
    # Use cache if exists
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read EMA cache: {e}"
            )

    # Fallback → live build
    try:
        df = load_price_data()
        return build_ema_latest_payload(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"EMA live build failed: {e}"
        )