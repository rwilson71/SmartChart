import json
import time
from pathlib import Path

import pandas as pd

from core.e_ai_rsi import AiRsiConfig, build_ai_rsi_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/ai_rsi_latest.json")
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
        raise ValueError(f"AI RSI cache build missing required columns: {missing}")

    return df


def run_ai_rsi_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("AI RSI build error: empty dataset")
            return

        payload = build_ai_rsi_latest_payload(
            df.tail(3000).copy(),
            config=AiRsiConfig()
        )

        if not payload:
            print("AI RSI build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"AI RSI cache updated: {payload.get('timestamp', 'unknown timestamp')}"
        )

    except Exception as e:
        print(f"AI RSI build error: {e}")


if __name__ == "__main__":
    print("Starting AI RSI cache builder...")
    while True:
        run_ai_rsi_build()
        time.sleep(REFRESH_SECONDS)