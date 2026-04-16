import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.f_market_structure import build_market_structure_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/market_structure_latest.json")


@router.get("/website/market-structure/latest")
def get_market_structure_latest():
    # Cache-first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read Market Structure cache: {e}",
            )

    # Live fallback
    try:
        df = load_price_data().tail(5000).copy()
        return build_market_structure_latest_payload(
            df,
            config=None,
            mtf_frames={"tf1": "5", "tf2": "15", "tf3": "60", "tf4": "240", "tf5": "D"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build live Market Structure payload: {e}",
        )