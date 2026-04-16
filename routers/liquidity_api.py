import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.i_liquidity import build_liquidity_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/liquidity_latest.json")


def load_mtf_frames():
    """
    Placeholder MTF loader using the same base dataset.
    Replace with proper timeframe-resampled feeds if already available
    in your SmartChart pipeline.
    """
    df = load_price_data().copy()
    return {
        "struct": df.copy(),
        "tf1": df.copy(),
        "tf2": df.copy(),
        "tf3": df.copy(),
        "tf4": df.copy(),
    }


@router.get("/website/liquidity/latest")
def get_liquidity_latest():
    # Serve cache first
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read liquidity cache: {e}"
            )

    # Fallback to live build
    try:
        df = load_price_data()
        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Liquidity live build failed: empty dataset"
            )

        mtf_frames = load_mtf_frames()

        payload = build_liquidity_latest_payload(
            df=df,
            mtf_frames=mtf_frames,
        )
        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Liquidity live build failed: {e}"
        )