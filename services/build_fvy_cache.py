import json
import time
from pathlib import Path

import pandas as pd

from core.j_fvg import build_fvg_latest_payload, FvgConfig

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/fvg_latest.json")
REFRESH_SECONDS = 10


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    return df


def run_fvg_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("FVG build error: empty dataset")
            return

        payload = build_fvg_latest_payload(
            df,
            FvgConfig(symbol="XAUUSD", timeframe="1m"),
        )

        if not payload:
            print("FVG build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("FVG cache updated")

    except Exception as e:
        print(f"FVG build error: {e}")


if __name__ == "__main__":
    while True:
        run_fvg_build()
        time.sleep(REFRESH_SECONDS)