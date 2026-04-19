import json
import time
from pathlib import Path

import pandas as pd

from core.w_5m_15m_first_candle import build_5m_15m_first_candle_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/ltf_first_candle_latest.json")
REFRESH_SECONDS = 10


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"LTF First Candle missing columns: {missing}")

    return df


def run_ltf_first_candle_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("LTF First Candle build error: empty dataset")
            return

        payload = build_5m_15m_first_candle_payload(df.tail(3000).copy())

        if not payload:
            print("LTF First Candle build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"LTF First Candle updated: {payload.get('timestamp', 'unknown')}")

    except Exception as e:
        print(f"LTF First Candle build error: {e}")


if __name__ == "__main__":
    print("Starting LTF First Candle cache builder...")
    while True:
        run_ltf_first_candle_build()
        time.sleep(REFRESH_SECONDS)