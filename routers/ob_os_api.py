import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.o_ob_os import build_ob_os_latest_payload


router = APIRouter()

CACHE_PATH = Path("data/cache/ob_os_latest.json")


@router.get("/website/ob-os/latest")
def get_ob_os_latest():
    """
    Cache-first endpoint for SmartChart OB/OS + Divergence Engine.
    Matches the same production pattern used by Trend / EMA:
    1. Read cache if available
    2. Fallback to live build from shared loader
    """
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read OB/OS cache: {e}"
            )

    try:
        df = load_price_data()
        if df is None or df.empty:
            raise HTTPException(
                status_code=500,
                detail="OB/OS live build failed: empty dataset"
            )

        payload = build_ob_os_latest_payload(df)

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="OB/OS live build failed: empty payload"
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OB/OS live build failed: {e}"
        )