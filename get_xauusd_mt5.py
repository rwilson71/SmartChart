from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

OUT_PATH = Path("data/xauusd_mt5_m1.csv")
SYMBOL = "XAUUSD"
BARS = 5000


def main():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            raise RuntimeError(f"Symbol not found: {SYMBOL}")

        if not info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                raise RuntimeError(f"Could not select symbol: {SYMBOL}")

        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, BARS)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "tick_volume"})

        keep = ["datetime", "open", "high", "low", "close", "tick_volume"]
        df = df[keep].copy()

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)

        print(f"Saved {len(df)} rows to {OUT_PATH}")
        print("Columns:", df.columns.tolist())
        print(df.tail().to_string(index=False))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()