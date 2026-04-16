import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.r_confluence import build_confluence_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/confluence_latest.json")


@router.get("/website/confluence/latest")
def get_confluence_latest():
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read confluence cache: {e}",
            )

    try:
        df = load_price_data().tail(5000).copy()
        if df.empty:
            raise HTTPException(status_code=500, detail="Price data is empty.")
        return build_confluence_latest_payload(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build live confluence payload: {e}",
        )