from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from services.build_vf_pattern_cache import build_vf_pattern_cache


router = APIRouter()

CACHE_PATH = Path("data/cache/forecaster_pattern_latest.json")


@router.get("/website/pattern/latest")
def get_pattern_latest() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return build_vf_pattern_cache()