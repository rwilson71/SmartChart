import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

CACHE_PATH = Path("data/cache/ema_latest.json")


@router.get("/website/ema/latest")
def get_ema_latest():
    if not CACHE_PATH.exists():
        raise HTTPException(status_code=503, detail="EMA cache not ready")

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read EMA cache: {e}")