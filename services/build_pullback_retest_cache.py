from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.t_pullback_retest import build_pullback_retest_latest_payload


CACHE_PATH = Path("data/cache/pullback_retest_latest.json")
REFRESH_SECONDS = 10


def build_pullback_retest_cache() -> Dict[str, Any]:
    df = load_price_data().tail(2000).copy()

    if df.empty:
        raise ValueError("Pullback/Retest build error: empty dataset")

    payload = build_pullback_retest_latest_payload(df=df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def run_pullback_retest_build() -> None:
    try:
        payload = build_pullback_retest_cache()
        print(
            f"Pullback/Retest cache updated: {CACHE_PATH} | "
            f"state={payload.get('state')} | "
            f"bias={payload.get('bias_label')} | "
            f"market_bias={payload.get('market_bias')} | "
            f"trigger_score={payload.get('trigger_final_score')}"
        )
    except Exception as e:
        print(f"Pullback/Retest build error: {e}")


if __name__ == "__main__":
    while True:
        run_pullback_retest_build()
        time.sleep(REFRESH_SECONDS)