import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.k_orderflow import build_orderflow_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/orderflow_latest.json")


@router.get("/website/orderflow/latest")
def get_orderflow_latest():
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read orderflow cache: {e}",
            )

    try:
        df = load_price_data()

        if df is None or df.empty:
            raise HTTPException(
                status_code=500,
                detail="Orderflow live build failed: empty dataset",
            )

        return build_orderflow_latest_payload(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Orderflow live build failed: {e}",
        )