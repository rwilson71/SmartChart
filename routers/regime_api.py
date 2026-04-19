from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from core.data_loader import load_price_data
from core.u_regime import build_regime_latest_payload

router = APIRouter()

CACHE_PATH = Path("data/cache/regime_latest.json")


def _read_cached_payload() -> Dict[str, Any]:
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/website/regime/latest")
def get_regime_latest() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return _read_cached_payload()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read regime cache: {e}",
            )

    try:
        df = load_price_data().tail(2000).copy()

        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Regime live build failed: empty dataset",
            )

        return build_regime_latest_payload(df)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build regime payload: {e}",
        )