from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from services.services.build_ve_macro_cache import build_ve_macro_cache
router = APIRouter()

CACHE_PATH = Path("data/cache/forecaster_macro_latest.json")


@router.get("/website/macro/latest")
def get_macro_latest() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return build_ve_macro_cache()