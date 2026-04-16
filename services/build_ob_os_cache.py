import json
import time
from pathlib import Path

import pandas as pd

from core.data_loader import load_price_data
from core.o_ob_os import build_ob_os_latest_payload


DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/ob_os_latest.json")
REFRESH_SECONDS = 10


def ensure_cache_dir() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_ob_os_data() -> pd.DataFrame:
    """
    Standard loader path for cache service.
    Uses shared loader first to stay aligned with Trend / EMA pipeline.
    """
    try:
        df = load_price_data()
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Fallback local CSV loader
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    return df


def run_ob_os_build() -> None:
    try:
        df = load_ob_os_data()

        if df is None or df.empty:
            print("OB/OS cache build error: empty dataset")
            return

        payload = build_ob_os_latest_payload(df)

        if not payload:
            print("OB/OS cache build error: empty payload")
            return

        ensure_cache_dir()
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"OB/OS cache updated: "
            f"{payload.get('timestamp', 'unknown')} | "
            f"state={payload.get('state_text', 'NA')} | "
            f"grade={payload.get('grade_text', 'NA')}"
        )

    except Exception as e:
        print(f"OB/OS cache build failed: {e}")


def main() -> None:
    ensure_cache_dir()
    print("Starting OB/OS cache builder...")

    while True:
        run_ob_os_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()