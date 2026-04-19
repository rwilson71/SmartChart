import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.j_fvg import build_fvg_latest_payload, FvgConfig

router = APIRouter(prefix="/website/fvg", tags=["FVG"])

CACHE_PATH = Path("data/cache/fvg_latest.json")


@router.get("/latest")
def get_fvg_latest():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        df = load_price_data()
        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="FVG dataset is empty")

        return build_fvg_latest_payload(
            df,
            FvgConfig(symbol="XAUUSD", timeframe="1m"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FVG live build failed: {e}")