import json
import time
from pathlib import Path

from core.data_loader import load_price_data
from core.p_exhaustion import build_exhaustion_latest_payload

CACHE_PATH = Path("data/cache/exhaustion_latest.json")
REFRESH_SECONDS = 10


def run_exhausion_build() -> None:
    try:
        df = load_price_data().tail(3000).copy()

        if df.empty:
            print("Exhausion cache build error: empty dataset")
            return

        payload = build_exhaustion_latest_payload(df)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"Exhausion cache updated: "
            f"state={payload.get('state')} "
            f"text={payload.get('state_text')} "
            f"time={payload.get('timestamp')}"
        )

    except Exception as e:
        print(f"Exhausion cache build error: {e}")


if __name__ == "__main__":
    print("Starting Exhausion cache builder...")
    while True:
        run_exhausion_build()
        time.sleep(REFRESH_SECONDS)