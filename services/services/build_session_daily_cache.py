import json
import time
from pathlib import Path

import pandas as pd

from core.g_session_daily import build_session_daily_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/session_daily_latest.json")
REFRESH_SECONDS = 10


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Session/Daily cache build error: no datetime column could be used as index")

    return df


def run_session_daily_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("Session/Daily build error: empty dataset")
            return

        payload = build_session_daily_latest_payload(df)

        if not payload:
            print("Session/Daily build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(
            f"Session/Daily cache updated: {payload.get('timestamp', 'unknown')} "
            f"| session={payload.get('state', {}).get('active_session', 'NONE')}"
        )

    except Exception as e:
        print(f"Session/Daily build error: {e}")


def main() -> None:
    print("Starting Session/Daily cache builder...")
    while True:
        run_session_daily_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()