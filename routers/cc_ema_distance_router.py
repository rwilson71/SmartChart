from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.services.build_ema_distance_cache import (
    get_ema_distance_latest,
    get_ema_distance_live_result,
)

router = APIRouter(prefix="/website/ema-distance", tags=["EMA Distance"])


@router.get("/latest")
def ema_distance_latest(
    refresh: bool = Query(False, description="Force cache refresh"),
):
    try:
        return get_ema_distance_latest(force_refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EMA Distance latest failed: {e}")


@router.get("/live")
def ema_distance_live(
    refresh: bool = Query(False, description="Force cache refresh"),
):
    try:
        return get_ema_distance_live_result(force_refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EMA Distance live failed: {e}")
