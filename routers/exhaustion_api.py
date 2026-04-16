import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.p_exhaustion import build_exhaustion_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/exhaustion_latest.json")


@router.get("/website/exhaustion/latest")
def get_exhaustion_latest():
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read exhaustion cache: {e}"
            )

    try:
        df = load_price_data().tail(3000).copy()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Exhaustion live build failed: empty dataset"
            )

        return build_exhaustion_latest_payload(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Exhaustion live build failed: {e}"
        )