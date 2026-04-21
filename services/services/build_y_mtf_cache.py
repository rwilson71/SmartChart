from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from core.data_loader import load_price_data
from core.y_mtf import build_mtf_latest_payload


CACHE_PATH = Path("data/cache/y_mtf_latest.json")
REFRESH_SECONDS = 10


def build_y_mtf_cache() -> Dict[str, Any]:
    df = load_price_data().tail(5000).copy()

    if df.empty:
        raise ValueError("Y_MTF build error: empty dataset")

    payload = build_mtf_latest_payload(df=df)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload


def run_y_mtf_cache_loop(refresh_seconds: int = REFRESH_SECONDS) -> None:
    print(f"[Y_MTF] cache loop started | refresh={refresh_seconds}s | path={CACHE_PATH}")

    while True:
        try:
            payload = build_y_mtf_cache()
            print(
                f"[Y_MTF] cache updated | "
                f"timestamp={payload.get('timestamp', 'n/a')} | "
                f"bias={payload.get('bias_label', 'n/a')} | "
                f"strength={payload.get('indicator_strength', 'n/a')} | "
                f"confidence={payload.get('mtf_confidence_label', 'n/a')} | "
                f"agreement={payload.get('mtf_agreement_label', 'n/a')}"
            )
        except KeyboardInterrupt:
            print("[Y_MTF] cache loop stopped by user")
            break
        except Exception as e:
            print(f"[Y_MTF] cache build error: {e}")

        time.sleep(refresh_seconds)


if __name__ == "__main__":
    run_y_mtf_cache_loop()