import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.e_ai_rsi import AiRsiConfig, build_ai_rsi_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/ai_rsi_latest.json")


@router.get("/website/ai-rsi/latest")
def get_ai_rsi_latest():
    # Cache first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read AI RSI cache: {e}"
            )

    # Live fallback
    try:
        df = load_price_data().tail(3000).copy()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="AI RSI live build failed: price data is empty."
            )

        payload = build_ai_rsi_latest_payload(
            df,
            config=AiRsiConfig()
        )

        if not payload:
            raise HTTPException(
                status_code=500,
                detail="AI RSI live build failed: empty payload."
            )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI RSI live build failed: {e}"
        )