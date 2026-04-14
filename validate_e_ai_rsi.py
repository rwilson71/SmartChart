# =============================================================================
# AI RSI PARITY VALIDATION — CLEAN RUN VERSION
# =============================================================================

from __future__ import annotations

import pandas as pd

from core.e_ai_rsi import run_ai_rsi_engine


# =============================================================================
# CONFIG
# =============================================================================

TV_FILE = "data/FX_XAUUSD_1 (6).csv"

COMPARE_COLUMNS = [
    ("AI RSI Raw CSV", "sc_ai_rsi_raw"),
    ("AI RSI Signal CSV", "sc_ai_rsi_signal"),
    ("AI RSI Dir CSV", "sc_ai_rsi_dir"),
    ("AI RSI Strength CSV", "sc_ai_rsi_strength"),
    ("AI RSI Neutral CSV", "sc_ai_rsi_is_neutral"),

    ("DBG RSI Val CSV", "rsiVal"),
    ("DBG Y RSI CSV", "y_rsi"),
    ("DBG X RSI CSV", "x_rsi"),
    ("DBG Pred Z CSV", "pred_rsi_z"),
    ("DBG RSI Mean CSV", "rsi_mean"),
    ("DBG RSI Std CSV", "rsi_std"),
    ("DBG Pred RSI CSV", "pred_rsi"),
    ("DBG Raw CSV", "rsiWeight"),
    ("DBG X RSI RAW CSV", "rsiVal_shift1"),
]


# =============================================================================
# LOAD + NORMALIZE
# =============================================================================

def _normalize_tv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize TradingView CSV columns while preserving original exported parity/debug columns.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map: dict[str, str] = {}

    for c in df.columns:
        cl = c.lower().strip()

        if cl in {"time", "date", "datetime", "timestamp"}:
            rename_map[c] = "datetime"
        elif cl == "open":
            rename_map[c] = "open"
        elif cl == "high":
            rename_map[c] = "high"
        elif cl == "low":
            rename_map[c] = "low"
        elif cl == "close":
            rename_map[c] = "close"
        elif cl == "volume":
            rename_map[c] = "volume"

    df = df.rename(columns=rename_map)

    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required TradingView OHLCV columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    return df


def load_tv() -> pd.DataFrame:
    df = pd.read_csv(TV_FILE)
    df = _normalize_tv_columns(df)

    print("\n=== NORMALIZED TRADINGVIEW COLUMNS ===")
    for c in df.columns:
        print(c)

    return df


# =============================================================================
# VALIDATION
# =============================================================================

def _compare_series(tv: pd.Series, py: pd.Series) -> pd.DataFrame:
    merged = pd.concat([tv, py], axis=1)
    merged.columns = ["tv", "py"]
    merged = merged.dropna().copy()
    merged["abs_diff"] = (merged["tv"] - merged["py"]).abs()
    return merged


def run_validation() -> None:
    tv = load_tv()

    # Keep only OHLCV for engine input, but preserve full TV frame for comparisons
    py_input = tv[["open", "high", "low", "close", "volume"]].copy()

    # Run Python engine
    py = run_ai_rsi_engine(py_input)

    # Extra parity helper field for the final debug comparison
    py["rsiVal_shift1"] = py["rsiVal"].shift(1)

    print("\n=== VALIDATION START ===\n")

    summary_rows: list[dict[str, object]] = []

    for tv_col, py_col in COMPARE_COLUMNS:
        if tv_col not in tv.columns:
            print(f"⚠️ Missing TV column: {tv_col}")
            summary_rows.append({
                "tv_column": tv_col,
                "py_column": py_col,
                "status": "missing_tv",
                "rows": 0,
                "max_diff": None,
                "mean_diff": None,
            })
            continue

        if py_col not in py.columns:
            print(f"⚠️ Missing PY column: {py_col}")
            summary_rows.append({
                "tv_column": tv_col,
                "py_column": py_col,
                "status": "missing_py",
                "rows": 0,
                "max_diff": None,
                "mean_diff": None,
            })
            continue

        merged = _compare_series(tv[tv_col], py[py_col])

        if len(merged) == 0:
            print(f"⚠️ No overlap: {tv_col}")
            summary_rows.append({
                "tv_column": tv_col,
                "py_column": py_col,
                "status": "no_overlap",
                "rows": 0,
                "max_diff": None,
                "mean_diff": None,
            })
            continue

        max_diff = float(merged["abs_diff"].max())
        mean_diff = float(merged["abs_diff"].mean())

        print(tv_col)
        print(f"   rows      : {len(merged)}")
        print(f"   max diff  : {max_diff:.10f}")
        print(f"   mean diff : {mean_diff:.10f}")
        print("-" * 50)

        summary_rows.append({
            "tv_column": tv_col,
            "py_column": py_col,
            "status": "ok",
            "rows": len(merged),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        })

    summary = pd.DataFrame(summary_rows)

    print("\n=== VALIDATION SUMMARY ===")
    print(summary.to_string(index=False))


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_validation()
    