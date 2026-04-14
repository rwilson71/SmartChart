from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://api.twelvedata.com"
SYMBOL = "XAU/USD"
INTERVAL = "1min"
OUTPUT_CSV = Path("data/xauusd_m1_full.csv")
PAGE_SIZE = 5000
MAX_RETRIES = 5
SLEEP_BETWEEN_PAGES = 8


def get_earliest_timestamp(api_key: str, symbol: str = SYMBOL, interval: str = INTERVAL) -> str:
    url = f"{BASE_URL}/earliest_timestamp"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message', data)}")

    dt = data.get("datetime")
    if not dt:
        raise RuntimeError(f"Could not find earliest timestamp in response: {data}")

    return dt


def main() -> None:
    api_key = os.getenv("TWELVE_DATA_API_KEY", "b88c6414fc6e49039bfe45c1afa0575c")

    if not api_key:
        raise RuntimeError(
            "Missing API key. Set TWELVE_DATA_API_KEY in your environment "
            "or paste your key into the api_key fallback string."
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching earliest available timestamp for {SYMBOL} {INTERVAL}...")
    earliest = get_earliest_timestamp(api_key=api_key, symbol=SYMBOL, interval=INTERVAL)
    print(f"Earliest available: {earliest}")

    all_frames: list[pd.DataFrame] = []
    end_date: Optional[str] = None
    page_num = 0
    earliest_ts = pd.Timestamp(earliest, tz="UTC")

    while True:
        page_num += 1

        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "apikey": api_key,
            "outputsize": PAGE_SIZE,
            "order": "desc",
            "timezone": "UTC",
            "format": "JSON",
            "start_date": earliest,
        }
        if end_date:
            params["end_date"] = end_date

        print(f"Requesting page {page_num}..." + (f" end_date={end_date}" if end_date else ""))

        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            response = requests.get(f"{BASE_URL}/time_series", params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "error":
                break

            message = str(data.get("message", "Unknown API error"))
            if "run out of api credits" in message.lower():
                wait_time = 10 + (attempt - 1) * 5
                print(f"Rate limit hit on attempt {attempt}/{MAX_RETRIES}. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            raise RuntimeError(f"Twelve Data error: {message}")
        else:
            raise RuntimeError("Max retries exceeded due to rate limits.")

        values = data.get("values", [])
        if not values:
            print("No more data returned. Stopping.")
            break

        df = pd.DataFrame(values)
        if df.empty:
            print("Empty page received. Stopping.")
            break

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

        if df.empty:
            print("Page had no valid datetimes after parsing. Stopping.")
            break

        all_frames.append(df)

        oldest_dt = df["datetime"].min()
        newest_dt = df["datetime"].max()

        print(f"Page {page_num}: {len(df):,} rows | {oldest_dt} -> {newest_dt}")

        if oldest_dt <= earliest_ts:
            print("Reached earliest available timestamp. Stopping.")
            break

        next_end = oldest_dt - pd.Timedelta(seconds=1)
        end_date = next_end.strftime("%Y-%m-%d %H:%M:%S")

        time.sleep(SLEEP_BETWEEN_PAGES)

    if not all_frames:
        raise RuntimeError("No data was fetched.")

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = (
        full_df.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    preferred_cols = ["datetime", "open", "high", "low", "close", "volume"]
    existing_cols = [c for c in preferred_cols if c in full_df.columns]
    full_df = full_df[existing_cols]

    full_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(full_df):,} rows to: {OUTPUT_CSV}")
    print(f"Range: {full_df['datetime'].min()} -> {full_df['datetime'].max()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)