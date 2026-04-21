from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from services.build_vb_correlation_cache import build_vb_correlation_cache


router = APIRouter()

CACHE_PATH = Path("data/cache/forecaster_correlation_latest.json")


@router.get("/website/correlation/latest")
def get_correlation_latest() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # fallback to live build
    return build_vb_correlation_cache()