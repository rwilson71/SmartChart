import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.l_confluence_cloud import build_confluence_cloud_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/confluence_cloud_latest.json")


@router.get("/website/confluence-cloud/latest")
def get_confluence_cloud_latest():
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read Confluence Cloud cache: {e}",
            )

    try:
        df = load_price_data()

        if df is None or df.empty:
            raise HTTPException(
                status_code=500,
                detail="Price data is empty for Confluence Cloud live build.",
            )

        return build_confluence_cloud_latest_payload(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build Confluence Cloud payload: {e}",
        )