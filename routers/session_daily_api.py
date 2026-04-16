import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.g_session_daily import build_session_daily_latest_payload
from core.data_loader import load_price_data

router = APIRouter()

CACHE_PATH = Path("data/cache/session_daily_latest.json")


@router.get("/website/session-daily/latest")
def get_session_daily_latest():
    # Cache-first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read Session/Daily cache: {e}"
            )

    # Live fallback
    try:
        df = load_price_data().tail(5000).copy()
        if df.empty:
            raise HTTPException(status_code=404, detail="No price data available for Session/Daily live build")

        return build_session_daily_latest_payload(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build Session/Daily payload live: {e}"
        )