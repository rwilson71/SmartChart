from __future__ import annotations

import pandas as pd

from core.o_ob_os import ObOsConfig, ObOsEngine


def load_xauusd_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Adjust this section if your CSV column names differ
    colmap = {c.lower(): c for c in df.columns}

    time_col = None
    for candidate in ["time", "timestamp", "datetime", "date"]:
        if candidate in colmap:
            time_col = colmap[candidate]
            break

    if time_col is None:
        raise ValueError("Could not find a time column in CSV.")

    rename_map = {}
    for src in ["open", "high", "low", "close", "volume"]:
        if src in colmap:
            rename_map[colmap[src]] = src

    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    return df[["open", "high", "low", "close", "volume"]].copy()


if __name__ == "__main__":
    df = load_xauusd_csv("data/xauusd.csv")

    config = ObOsConfig(
        rsi_len=14,
        stoch_len=14,
        stoch_smooth=3,
        mfi_len=14,
        cci_len=20,
        sig_smooth_len=5,
        ob_level=0.65,
        os_level=-0.65,
        extreme_ob_level=0.82,
        extreme_os_level=-0.82,
        div_left=3,
        div_right=3,
        max_pivot_gap=40,
        min_pivot_gap=2,
        use_hidden_div=True,
        require_ob_os_for_div=True,
        mtf_on=True,
        mtf_weights={"tf1": 1.0, "tf2": 1.0, "tf3": 1.0, "tf4": 1.0},
        mtf_rules={"tf1": "15min", "tf2": "1h", "tf3": "4h", "tf4": "D"},
        trigger_cross_lookback=2,
        reversal_gate_level=0.60,
        trend_gate_level=0.25,
        min_reversal_strength=0.20,
    )

    engine = ObOsEngine(config=config)
    full = engine.calculate(df)
    latest = engine.latest(df)
print("\n=== OB/OS PARITY RUN (REAL CSV) ===")

print(
    full[
        [
            "composite",
            "signal",
            "obos_state",
            "bullish_div",
            "bearish_div",
            "hidden_bullish_div",
            "hidden_bearish_div",
            "mtf_avg",
            "bull_reversal_score",
            "bear_reversal_score",
            "bull_continuation_score",
            "bear_continuation_score",
            "exhaustion_score",
            "sc_state",
            "sc_grade",
        ]
    ].tail(15)
)

print("\n=== LATEST OUTPUT ===")
print(latest)