import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from core.data_loader import load_price_data
from core.i_liquidity import build_liquidity_latest_payload

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/liquidity_latest.json")
REFRESH_SECONDS = 10


def load_mtf_frames() -> Dict[str, pd.DataFrame]:
    """
    Placeholder MTF loader using the same base dataset.
    Replace with proper timeframe-resampled feeds if already available
    in your SmartChart pipeline.
    """
    df = load_price_data().copy()

    mtf_frames: Dict[str, pd.DataFrame] = {
        "struct": df.copy(),
        "tf1": df.copy(),
        "tf2": df.copy(),
        "tf3": df.copy(),
        "tf4": df.copy(),
    }
    return mtf_frames


def run_liquidity_build() -> None:
    try:
        df = load_price_data()

        if df.empty:
            print("Liquidity build error: empty dataset")
            return

        mtf_frames = load_mtf_frames()

        payload = build_liquidity_latest_payload(
            df=df,
            mtf_frames=mtf_frames,
        )

        if not payload:
            print("Liquidity build error: empty payload")
            return

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"Liquidity cache updated: "
            f"{payload.get('timestamp', 'unknown')} | "
            f"state={payload.get('state', 'na')} | "
            f"score={payload.get('final_score', 'na')}"
        )

    except Exception as e:
        print(f"Liquidity build error: {e}")


def main() -> None:
    print("Starting Liquidity cache builder...")
    while True:
        run_liquidity_build()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()