import json
import time
from pathlib import Path

from core.data_loader import load_price_data
from core.j_fvg import build_fvg_latest_payload, FvgConfig

CACHE_PATH = Path("data/cache/fvg_latest.json")
REFRESH_SECONDS = 10


def build_fvg_cache() -> dict:
    df = load_price_data()

    if df is None or df.empty:
        raise ValueError("FVG build error: empty dataset")

    payload = build_fvg_latest_payload(
        df,
        FvgConfig(symbol="XAUUSD", timeframe="1m"),
    )

    if not payload:
        raise ValueError("FVG build error: empty payload")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def run_fvg_build() -> None:
    try:
        build_fvg_cache()
        print("FVG cache updated")
    except Exception as e:
        print(f"FVG build error: {e}")


if __name__ == "__main__":
    while True:
        run_fvg_build()
        time.sleep(REFRESH_SECONDS)