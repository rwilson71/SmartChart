from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MFIConfig:
    len_slow: int = 21
    len_fast: int = 5

    mid_level: float = 50.0
    high_level: float = 80.0
    low_level: float = 20.0

    mtf_on: bool = True
    mtf_weights: tuple[float, float, float, float, float, float] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    )


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first" if "open" in df.columns else "last",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(rule).agg(agg).dropna()


def _align_to_base(series: pd.Series, base_index: pd.Index) -> pd.Series:
    return series.reindex(base_index, method="ffill")


def _mfi(hlc3: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    money_flow = hlc3 * volume
    hlc3_delta = hlc3.diff()

    pos_flow = pd.Series(
        np.where(hlc3_delta > 0, money_flow, 0.0),
        index=hlc3.index,
        dtype=float,
    )
    neg_flow = pd.Series(
        np.where(hlc3_delta < 0, money_flow, 0.0),
        index=hlc3.index,
        dtype=float,
    )

    pos_sum = pos_flow.rolling(length, min_periods=1).sum()
    neg_sum = neg_flow.rolling(length, min_periods=1).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    out = out.fillna(50.0).clip(0.0, 100.0)
    return out


def _mfi_score_from_slow(df: pd.DataFrame, length: int, mid_level: float) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    mfi_slow = _mfi(hlc3, df["volume"], length)
    score = pd.Series(
        np.where(mfi_slow > mid_level, 1.0, np.where(mfi_slow < mid_level, -1.0, 0.0)),
        index=df.index,
        dtype=float,
    )
    return score


# =============================================================================
# CORE LOGIC
# =============================================================================

def _compute_base_mfi(df: pd.DataFrame, cfg: MFIConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0

    mfi21 = _mfi(hlc3, df["volume"], cfg.len_slow)
    mfi5 = _mfi(hlc3, df["volume"], cfg.len_fast)

    mfi_trend_bull = mfi21 > cfg.mid_level
    mfi_trend_bear = mfi21 < cfg.mid_level

    mfi_signal_bull = mfi5 > cfg.mid_level
    mfi_signal_bear = mfi5 < cfg.mid_level

    mfi_extreme_high = (mfi5 > cfg.high_level) | (mfi21 > cfg.high_level)
    mfi_extreme_low = (mfi5 < cfg.low_level) | (mfi21 < cfg.low_level)

    mfi_cross_up_50 = (mfi5 > cfg.mid_level) & (mfi5.shift(1) <= cfg.mid_level)
    mfi_cross_down_50 = (mfi5 < cfg.mid_level) & (mfi5.shift(1) >= cfg.mid_level)

    # Combined directional state
    mfi_state = np.select(
        [
            mfi_extreme_high,
            mfi_extreme_low,
            mfi_trend_bull & mfi_signal_bull,
            mfi_trend_bear & mfi_signal_bear,
            mfi_trend_bull,
            mfi_trend_bear,
        ],
        [3, -3, 2, -2, 1, -1],
        default=0,
    )

    out["mfi_slow"] = mfi21
    out["mfi_fast"] = mfi5

    out["mfi_trend_bull"] = mfi_trend_bull.astype(int)
    out["mfi_trend_bear"] = mfi_trend_bear.astype(int)

    out["mfi_signal_bull"] = mfi_signal_bull.astype(int)
    out["mfi_signal_bear"] = mfi_signal_bear.astype(int)

    out["mfi_extreme_high"] = mfi_extreme_high.astype(int)
    out["mfi_extreme_low"] = mfi_extreme_low.astype(int)

    out["mfi_cross_up_50"] = mfi_cross_up_50.fillna(False).astype(int)
    out["mfi_cross_down_50"] = mfi_cross_down_50.fillna(False).astype(int)

    out["mfi_state"] = pd.Series(mfi_state, index=df.index, dtype=int)
    out["mfi_bias"] = pd.Series(
        np.where(out["mfi_state"] > 0, 1, np.where(out["mfi_state"] < 0, -1, 0)),
        index=df.index,
        dtype=int,
    )

    return out


# =============================================================================
# MTF AVERAGE
# =============================================================================

def _compute_mtf_average(
    df: pd.DataFrame,
    cfg: MFIConfig,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if not cfg.mtf_on:
        out["mfi_mtf_1"] = 0.0
        out["mfi_mtf_2"] = 0.0
        out["mfi_mtf_3"] = 0.0
        out["mfi_mtf_4"] = 0.0
        out["mfi_mtf_5"] = 0.0
        out["mfi_mtf_6"] = 0.0
        out["mfi_mtf_avg"] = 0.0
        return out

    tf_map = timeframe_map or {
        "tf1": "5min",
        "tf2": "15min",
        "tf3": "30min",
        "tf4": "1h",
        "tf5": "4h",
        "tf6": "1D",
    }

    scores = []
    for key in ["tf1", "tf2", "tf3", "tf4", "tf5", "tf6"]:
        tf_rule = tf_map[key]
        tf_df = _resample_ohlcv(df, tf_rule)
        tf_score = _mfi_score_from_slow(tf_df, cfg.len_slow, cfg.mid_level)
        tf_score = _align_to_base(tf_score, df.index).fillna(0.0)
        scores.append(tf_score)

    out["mfi_mtf_1"] = scores[0]
    out["mfi_mtf_2"] = scores[1]
    out["mfi_mtf_3"] = scores[2]
    out["mfi_mtf_4"] = scores[3]
    out["mfi_mtf_5"] = scores[4]
    out["mfi_mtf_6"] = scores[5]

    weights = np.array(cfg.mtf_weights, dtype=float)
    w_sum = float(weights.sum())
    w_safe = 1.0 if w_sum <= 0 else w_sum

    out["mfi_mtf_avg"] = (
        out["mfi_mtf_1"] * weights[0]
        + out["mfi_mtf_2"] * weights[1]
        + out["mfi_mtf_3"] * weights[2]
        + out["mfi_mtf_4"] * weights[3]
        + out["mfi_mtf_5"] * weights[4]
        + out["mfi_mtf_6"] * weights[5]
    ) / w_safe

    return out


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_mfi(
    df: pd.DataFrame,
    config: Optional[MFIConfig] = None,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    SmartChart MFI Engine v1 backend conversion.

    Expected input:
        DataFrame indexed by datetime with:
        high, low, close, volume
        open optional for resampling quality

    Returns:
        DataFrame with base MFI logic, state outputs, and MTF average.
    """
    cfg = config or MFIConfig()
    _validate_ohlcv(df)

    base = df.copy().sort_index()

    core = _compute_base_mfi(base, cfg)
    mtf = _compute_mtf_average(base, cfg, timeframe_map=timeframe_map)

    out = pd.concat([core, mtf], axis=1)

    # Export-ready fields for confluence / pullback / truth engine
    out["mfi_state_export"] = out["mfi_state"].astype(int)
    out["mfi_bias_export"] = out["mfi_bias"].astype(int)

    out["mfi_trend_bull_export"] = out["mfi_trend_bull"].astype(int)
    out["mfi_trend_bear_export"] = out["mfi_trend_bear"].astype(int)

    out["mfi_signal_bull_export"] = out["mfi_signal_bull"].astype(int)
    out["mfi_signal_bear_export"] = out["mfi_signal_bear"].astype(int)

    out["mfi_extreme_high_export"] = out["mfi_extreme_high"].astype(int)
    out["mfi_extreme_low_export"] = out["mfi_extreme_low"].astype(int)

    out["mfi_cross_up_50_export"] = out["mfi_cross_up_50"].astype(int)
    out["mfi_cross_down_50_export"] = out["mfi_cross_down_50"].astype(int)

    out["mfi_mtf_avg_export"] = out["mfi_mtf_avg"].astype(float)

    # SmartChart contract helpers
    out["mfi_direction"] = np.where(out["mfi_state"] > 0, 1, np.where(out["mfi_state"] < 0, -1, 0)).astype(int)
    out["mfi_strength"] = np.select(
        [
            out["mfi_state"].abs() == 3,
            out["mfi_state"].abs() == 2,
            out["mfi_state"].abs() == 1,
        ],
        [1.0, 0.66, 0.33],
        default=0.0,
    )
    out["mfi_signal"] = ((out["mfi_signal_bull"] == 1) | (out["mfi_signal_bear"] == 1)).astype(int)

    return out


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=500, freq="5min")
    np.random.seed(42)

    base_price = 100 + np.cumsum(np.random.normal(0, 0.30, len(rng)))
    close = pd.Series(base_price, index=rng)
    open_ = close.shift(1).fillna(close.iloc[0])

    high = pd.concat([open_, close], axis=1).max(axis=1) + np.random.uniform(0.02, 0.25, len(rng))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.random.uniform(0.02, 0.25, len(rng))
    volume = pd.Series(np.random.randint(100, 5000, len(rng)), index=rng)

    test_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=rng,
    )

    result = calculate_mfi(test_df)

    cols = [
        "mfi_slow",
        "mfi_fast",
        "mfi_state_export",
        "mfi_bias_export",
        "mfi_trend_bull_export",
        "mfi_trend_bear_export",
        "mfi_signal_bull_export",
        "mfi_signal_bear_export",
        "mfi_extreme_high_export",
        "mfi_extreme_low_export",
        "mfi_cross_up_50_export",
        "mfi_cross_down_50_export",
        "mfi_mtf_avg_export",
        "mfi_direction",
        "mfi_strength",
        "mfi_signal",
    ]

    print("SmartChart MFI Engine v1 — direct test")
    print(result[cols].tail(20))