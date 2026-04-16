import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.q_mfi import build_mfi_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/mfi_latest.json")


@router.get("/website/mfi/latest")
def get_mfi_latest():
    # Cache-first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read MFI cache: {e}"
            )

    # Live fallback
    try:
        df = load_price_data().tail(5000).copy()

        if df.empty:
            raise HTTPException(status_code=500, detail="Price data is empty")

        return build_mfi_latest_payload(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build live MFI payload: {e}"
        )