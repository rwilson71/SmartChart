import json
import time
from pathlib import Path

import pandas as pd

from core.f_market_structure import build_market_structure_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/market_structure_latest.json")
REFRESH_SECONDS = 10


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Price data must contain a datetime-like column for Market Structure MTF alignment.")

    return df


def run_market_structure_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("Market Structure build error: empty dataset")
            return

        payload = build_market_structure_latest_payload(
            df,
            config=None,
            mtf_frames={"tf1": "5", "tf2": "15", "tf3": "60", "tf4": "240", "tf5": "D"},
        )

        if not payload:
            print("Market Structure build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Market Structure cache updated: {CACHE_PATH}")

    except Exception as e:
        print(f"Market Structure build error: {e}")


if __name__ == "__main__":
    while True:
        run_market_structure_build()
        time.sleep(REFRESH_SECONDS)