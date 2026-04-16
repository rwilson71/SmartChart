import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.n_volatility import build_volatility_latest_payload
from core.data_loader import load_price_data

router = APIRouter()

CACHE_PATH = Path("data/cache/volatility_latest.json")


@router.get("/website/volatility/latest")
def get_volatility_latest():
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read volatility cache: {e}"
            )

    try:
        df = load_price_data().tail(5000).copy()
        if df.empty:
            raise HTTPException(status_code=500, detail="Price data is empty")

        return build_volatility_latest_payload(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build live volatility payload: {e}"
        )