import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

CACHE_PATH = Path("data/cache/trend_latest.json")


@router.get("/website/trend/latest")
def get_trend_latest():
    if not CACHE_PATH.exists():
        raise HTTPException(status_code=503, detail="Trend cache not ready")

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trend cache: {e}")