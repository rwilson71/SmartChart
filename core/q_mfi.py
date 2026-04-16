from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MFIConfig:
    # Core
    len_slow: int = 21
    len_fast: int = 5

    # Levels
    mid_level: float = 50.0
    high_level: float = 80.0
    low_level: float = 20.0

    # MTF Average
    mtf_on: bool = True
    tf1: str = "5min"
    tf2: str = "15min"
    tf3: str = "30min"
    tf4: str = "1h"
    tf5: str = "4h"
    tf6: str = "1D"

    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0
    w4: float = 1.0
    w5: float = 1.0
    w6: float = 1.0


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")


def _safe_series(series: pd.Series, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    return pd.Series(series, index=index).replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg: Dict[str, str] = {
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    if "open" in df.columns:
        agg["open"] = "first"

    out = df.resample(rule).agg(agg).dropna(subset=["high", "low", "close"])
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = out["volume"].fillna(0.0)
    return out


def _align_to_base(series: pd.Series, base_index: pd.Index) -> pd.Series:
    return series.reindex(base_index, method="ffill").fillna(0.0)


def _mfi(hlc3: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """
    Pine-aligned MFI approximation using:
      ta.mfi(hlc3, length)

    Formula:
      raw_money_flow = hlc3 * volume
      positive flow when hlc3 > hlc3[1]
      negative flow when hlc3 < hlc3[1]
      mfi = 100 - 100 / (1 + pos_sum / neg_sum)
    """
    hlc3 = hlc3.astype(float)
    volume = volume.astype(float).fillna(0.0)

    raw_money_flow = hlc3 * volume
    delta = hlc3.diff()

    pos_flow = pd.Series(
        np.where(delta > 0, raw_money_flow, 0.0),
        index=hlc3.index,
        dtype=float,
    )
    neg_flow = pd.Series(
        np.where(delta < 0, raw_money_flow, 0.0),
        index=hlc3.index,
        dtype=float,
    )

    pos_sum = pos_flow.rolling(length, min_periods=length).sum()
    neg_sum = neg_flow.rolling(length, min_periods=length).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))

    # Pine-friendly behavior for early / undefined areas
    mfi = mfi.replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0.0, 100.0)
    return mfi


def _mfi_score_from_slow(df: pd.DataFrame, cfg: MFIConfig) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    slow = _mfi(hlc3, df["volume"], cfg.len_slow)

    score = pd.Series(
        np.where(slow > cfg.mid_level, 1.0, np.where(slow < cfg.mid_level, -1.0, 0.0)),
        index=df.index,
        dtype=float,
    )
    return score


def _sign_label(v: float) -> str:
    if v > 0:
        return "bullish"
    if v < 0:
        return "bearish"
    return "neutral"


# =============================================================================
# CORE ENGINE
# =============================================================================

def calculate_mfi(
    df: pd.DataFrame,
    config: Optional[MFIConfig] = None,
) -> pd.DataFrame:
    cfg = config or MFIConfig()
    _validate_ohlcv(df)

    base = df.copy().sort_index()

    out = pd.DataFrame(index=base.index)

    hlc3 = (base["high"] + base["low"] + base["close"]) / 3.0

    # Pine:
    # mfi21 = ta.mfi(hlc3, lenSlow)
    # mfi5  = ta.mfi(hlc3, lenFast)
    out["mfi_slow"] = _mfi(hlc3, base["volume"], cfg.len_slow)
    out["mfi_fast"] = _mfi(hlc3, base["volume"], cfg.len_fast)

    # Pine core logic
    out["mfi_trend_bull"] = (out["mfi_slow"] > cfg.mid_level).astype(int)
    out["mfi_trend_bear"] = (out["mfi_slow"] < cfg.mid_level).astype(int)

    out["mfi_signal_bull"] = (out["mfi_fast"] > cfg.mid_level).astype(int)
    out["mfi_signal_bear"] = (out["mfi_fast"] < cfg.mid_level).astype(int)

    out["mfi_extreme_high"] = (
        (out["mfi_fast"] > cfg.high_level) | (out["mfi_slow"] > cfg.high_level)
    ).astype(int)

    out["mfi_extreme_low"] = (
        (out["mfi_fast"] < cfg.low_level) | (out["mfi_slow"] < cfg.low_level)
    ).astype(int)

    return out


# =============================================================================
# MTF ENGINE
# =============================================================================

def _compute_mtf_average(df: pd.DataFrame, cfg: MFIConfig) -> pd.DataFrame:
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

    tf_rules = [cfg.tf1, cfg.tf2, cfg.tf3, cfg.tf4, cfg.tf5, cfg.tf6]
    weights = np.array([cfg.w1, cfg.w2, cfg.w3, cfg.w4, cfg.w5, cfg.w6], dtype=float)

    scores = []
    for rule in tf_rules:
        tf_df = _resample_ohlcv(df, rule)
        tf_score = _mfi_score_from_slow(tf_df, cfg)
        tf_score = _align_to_base(tf_score, df.index)
        scores.append(tf_score)

    out["mfi_mtf_1"] = scores[0]
    out["mfi_mtf_2"] = scores[1]
    out["mfi_mtf_3"] = scores[2]
    out["mfi_mtf_4"] = scores[3]
    out["mfi_mtf_5"] = scores[4]
    out["mfi_mtf_6"] = scores[5]

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
# STATE ENGINE
# =============================================================================

def build_mfi_dataframe(
    df: pd.DataFrame,
    config: Optional[MFIConfig] = None,
) -> pd.DataFrame:
    cfg = config or MFIConfig()

    core = calculate_mfi(df, cfg)
    mtf = _compute_mtf_average(df.copy().sort_index(), cfg)

    out = pd.concat([core, mtf], axis=1)

    # Direction from slow MFI only, matching Pine intent
    out["mfi_direction"] = np.where(
        out["mfi_slow"] > cfg.mid_level,
        1,
        np.where(out["mfi_slow"] < cfg.mid_level, -1, 0),
    ).astype(int)

    # Signal direction from fast MFI only
    out["mfi_signal_direction"] = np.where(
        out["mfi_fast"] > cfg.mid_level,
        1,
        np.where(out["mfi_fast"] < cfg.mid_level, -1, 0),
    ).astype(int)

    # MTF state buckets taken directly from Pine visual thresholds
    out["mfi_mtf_state"] = np.select(
        [
            out["mfi_mtf_avg"] > 0.6,
            out["mfi_mtf_avg"] > 0.2,
            out["mfi_mtf_avg"] < -0.6,
            out["mfi_mtf_avg"] < -0.2,
        ],
        [2, 1, -2, -1],
        default=0,
    ).astype(int)

    return out


# =============================================================================
# TEXT MAPPING
# =============================================================================

def _build_mfi_text_fields(row: pd.Series) -> Dict[str, Any]:
    slow = float(row.get("mfi_slow", 50.0))
    fast = float(row.get("mfi_fast", 50.0))
    mtf_avg = float(row.get("mfi_mtf_avg", 0.0))

    trend_label = _sign_label(float(row.get("mfi_direction", 0.0)))
    signal_label = _sign_label(float(row.get("mfi_signal_direction", 0.0)))

    if row.get("mfi_extreme_high", 0) == 1:
        extreme_label = "extreme_high"
    elif row.get("mfi_extreme_low", 0) == 1:
        extreme_label = "extreme_low"
    else:
        extreme_label = "normal"

    if mtf_avg > 0.6:
        mtf_label = "strong_bullish"
    elif mtf_avg > 0.2:
        mtf_label = "bullish"
    elif mtf_avg < -0.6:
        mtf_label = "strong_bearish"
    elif mtf_avg < -0.2:
        mtf_label = "bearish"
    else:
        mtf_label = "neutral"

    return {
        "trend_label": trend_label,
        "signal_label": signal_label,
        "extreme_label": extreme_label,
        "mtf_label": mtf_label,
        "summary": f"{trend_label} trend / {signal_label} signal / {extreme_label}",
        "slow_value_text": f"{slow:.2f}",
        "fast_value_text": f"{fast:.2f}",
        "mtf_avg_text": f"{mtf_avg:.2f}",
    }


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_mfi_latest_payload(
    df: pd.DataFrame,
    config: Optional[MFIConfig] = None,
) -> Dict[str, Any]:
    cfg = config or MFIConfig()
    out = build_mfi_dataframe(df, cfg)

    if out.empty:
        return {}

    last = out.iloc[-1]
    ts = out.index[-1]

    text = _build_mfi_text_fields(last)

    payload: Dict[str, Any] = {
        "indicator": "mfi",
        "name": "SmartChart MFI Engine v1",
        "timestamp": str(ts),
        "debug_version": "mfi_payload_v1",
        "config": asdict(cfg),

        # Core values
        "mfi_slow": round(float(last.get("mfi_slow", 50.0)), 4),
        "mfi_fast": round(float(last.get("mfi_fast", 50.0)), 4),

        # Core states
        "mfi_trend_bull": int(last.get("mfi_trend_bull", 0)),
        "mfi_trend_bear": int(last.get("mfi_trend_bear", 0)),
        "mfi_signal_bull": int(last.get("mfi_signal_bull", 0)),
        "mfi_signal_bear": int(last.get("mfi_signal_bear", 0)),
        "mfi_extreme_high": int(last.get("mfi_extreme_high", 0)),
        "mfi_extreme_low": int(last.get("mfi_extreme_low", 0)),

        # Direction / state
        "mfi_direction": int(last.get("mfi_direction", 0)),
        "mfi_signal_direction": int(last.get("mfi_signal_direction", 0)),
        "mfi_mtf_state": int(last.get("mfi_mtf_state", 0)),

        # MTF components
        "mfi_mtf_1": round(float(last.get("mfi_mtf_1", 0.0)), 4),
        "mfi_mtf_2": round(float(last.get("mfi_mtf_2", 0.0)), 4),
        "mfi_mtf_3": round(float(last.get("mfi_mtf_3", 0.0)), 4),
        "mfi_mtf_4": round(float(last.get("mfi_mtf_4", 0.0)), 4),
        "mfi_mtf_5": round(float(last.get("mfi_mtf_5", 0.0)), 4),
        "mfi_mtf_6": round(float(last.get("mfi_mtf_6", 0.0)), 4),
        "mfi_mtf_avg": round(float(last.get("mfi_mtf_avg", 0.0)), 4),

        # Text mapping
        "trend_label": text["trend_label"],
        "signal_label": text["signal_label"],
        "extreme_label": text["extreme_label"],
        "mtf_label": text["mtf_label"],
        "summary": text["summary"],
        "slow_value_text": text["slow_value_text"],
        "fast_value_text": text["fast_value_text"],
        "mtf_avg_text": text["mtf_avg_text"],
    }

    return payload


# =============================================================================
# DIRECT TEST
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=1000, freq="1min")
    np.random.seed(42)

    close = pd.Series(100 + np.cumsum(np.random.normal(0, 0.15, len(rng))), index=rng)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.random.uniform(0.01, 0.12, len(rng))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.random.uniform(0.01, 0.12, len(rng))
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

    payload = build_mfi_latest_payload(test_df)
    print(payload)