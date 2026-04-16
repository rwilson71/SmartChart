import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.b_trend import build_trend_latest_payload
from core.data_loader import load_price_data

router = APIRouter()

CACHE_PATH = Path("data/cache/trend_latest.json")


@router.get("/website/trend/latest")
def get_trend_latest():
    # Use cache if exists
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read trend cache: {e}"
            )

    # Fallback to live build
    try:
        df = load_price_data()
        return build_trend_latest_payload(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trend live build failed: {e}"
        )