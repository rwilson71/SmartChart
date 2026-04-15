from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd


# =============================================================================
# SMARTCHART • EMA DISTANCE CORE
# File: core/cc_ema_distance_calibration.py
# =============================================================================
# Purpose
# - Locked model 3 live module for EMA distance
# - Preserve Pine-authority parity for:
#   1. EMA Core
#   2. Distance Engine
#   3. Signal Filter Engine
#   4. Bucket Engine
#   5. Stage Engine
# - Provide lightweight latest payload for website/API use
# - Keep lower-pane / oscillator-style structure preserved in outputs
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class EmaDistanceConfig:
    # Core
    ema_fast_len: int = 20
    ema_slow_len: int = 200
    slope_len: int = 5

    # Signal filters
    require_price_on_trend_side: bool = True
    require_fast_slope_confirm: bool = True
    require_slow_slope_confirm: bool = False
    min_abs_distance_pct: float = 0.0

    # Pine bucket boundaries
    bucket_edges_pct: Tuple[float, ...] = (
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
        2.50,
        3.00,
        math.inf,
    )

    # Pine stage thresholds
    zone1_max_pct: float = 0.75
    zone2_max_pct: float = 1.50
    zone3_min_pct: float = 1.50

    # Column names
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    time_col: Optional[str] = None

    # Payload metadata
    debug_version: str = "ema_distance_payload_v1"
    module_name: str = "ema_distance"


# =============================================================================
# HELPERS
# =============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    """TradingView-style EMA warmup alignment."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def pct_change_from_past(series: pd.Series, lookback: int) -> pd.Series:
    prev = series.shift(lookback)
    out = np.where(prev != 0.0, ((series - prev) / prev) * 100.0, np.nan)
    return pd.Series(out, index=series.index, dtype=float)


def validate_ohlc(df: pd.DataFrame, cfg: EmaDistanceConfig) -> None:
    required = [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    min_rows = max(cfg.ema_slow_len, cfg.slope_len) + 5
    if len(df) < min_rows:
        raise ValueError(
            f"Not enough rows for EMA warmup and slope calculation. "
            f"Need at least {min_rows}, got {len(df)}."
        )


def build_bucket_labels(edges: Sequence[float]) -> list[str]:
    labels: list[str] = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if math.isinf(right):
            labels.append(f"{left:.2f}%+")
        else:
            labels.append(f"{left:.2f}-{right:.2f}%")
    return labels


def _bucket_index_from_abs_dist(abs_dist: pd.Series, edges: Sequence[float]) -> pd.Series:
    values = abs_dist.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=int)

    finite_edges = list(edges[:-1])
    if len(finite_edges) < 10:
        raise ValueError("bucket_edges_pct must include Pine-style edges plus infinity.")

    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = 0
        elif v < finite_edges[1]:
            out[i] = 1
        elif v < finite_edges[2]:
            out[i] = 2
        elif v < finite_edges[3]:
            out[i] = 3
        elif v < finite_edges[4]:
            out[i] = 4
        elif v < finite_edges[5]:
            out[i] = 5
        elif v < finite_edges[6]:
            out[i] = 6
        elif v < finite_edges[7]:
            out[i] = 7
        elif v < finite_edges[8]:
            out[i] = 8
        elif v < finite_edges[9]:
            out[i] = 9
        else:
            out[i] = 10

    return pd.Series(out, index=abs_dist.index, dtype=int)


def _bucket_text_from_index(bucket_index: pd.Series, edges: Sequence[float]) -> pd.Series:
    labels = build_bucket_labels(edges)
    mapping: Dict[int, str] = {
        0: "NA",
        1: labels[0],
        2: labels[1],
        3: labels[2],
        4: labels[3],
        5: labels[4],
        6: labels[5],
        7: labels[6],
        8: labels[7],
        9: labels[8],
        10: labels[9],
    }
    return bucket_index.map(mapping).fillna("NA")


def _stage_from_abs_dist(abs_dist: pd.Series, cfg: EmaDistanceConfig) -> pd.Series:
    values = abs_dist.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=int)

    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = 0
        elif v < cfg.zone1_max_pct:
            out[i] = 1
        elif v < cfg.zone2_max_pct:
            out[i] = 2
        elif v >= cfg.zone3_min_pct:
            out[i] = 3
        else:
            out[i] = 0

    return pd.Series(out, index=abs_dist.index, dtype=int)


def _stage_text(stage: pd.Series) -> pd.Series:
    mapping = {
        0: "NONE",
        1: "ZONE 1",
        2: "ZONE 2",
        3: "ZONE 3",
    }
    return stage.map(mapping).fillna("NONE")


def _direction_text(v: int) -> str:
    if v == 1:
        return "BULL"
    if v == -1:
        return "BEAR"
    return "NEUTRAL"


def _safe_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def _safe_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


def _safe_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _extract_last_timestamp(
    features: pd.DataFrame,
    cfg: EmaDistanceConfig,
) -> Optional[str]:
    if cfg.time_col and cfg.time_col in features.columns:
        value = features.iloc[-1][cfg.time_col]
        if pd.notna(value):
            ts = pd.to_datetime(value, errors="coerce")
            if pd.notna(ts):
                return ts.isoformat()
            return str(value)

    if isinstance(features.index, pd.DatetimeIndex) and len(features.index) > 0:
        ts = features.index[-1]
        if pd.notna(ts):
            return ts.isoformat()

    return None


# =============================================================================
# EMA CORE (PINE PARITY)
# =============================================================================

def build_ema_core(df: pd.DataFrame, cfg: EmaDistanceConfig) -> pd.DataFrame:
    validate_ohlc(df, cfg)

    out = df.copy()
    close = out[cfg.close_col].astype(float)

    out["ema20"] = ema(close, cfg.ema_fast_len)
    out["ema200"] = ema(close, cfg.ema_slow_len)

    out["ema20_slope_pct"] = pct_change_from_past(out["ema20"], cfg.slope_len)
    out["ema200_slope_pct"] = pct_change_from_past(out["ema200"], cfg.slope_len)

    out["trend_side"] = np.where(
        out["ema20"] > out["ema200"],
        1,
        np.where(out["ema20"] < out["ema200"], -1, 0),
    )

    out["price_side_vs_ema20"] = np.where(
        out[cfg.close_col] > out["ema20"],
        1,
        np.where(out[cfg.close_col] < out["ema20"], -1, 0),
    )

    return out


# =============================================================================
# DISTANCE ENGINE (PINE PARITY)
# =============================================================================

def build_distance_engine(features: pd.DataFrame, cfg: EmaDistanceConfig) -> pd.DataFrame:
    out = features.copy()

    out["e20_to_e200_signed"] = out["ema20"] - out["ema200"]
    out["e20_to_e200_pct"] = np.where(
        out["ema200"] != 0.0,
        ((out["ema20"] - out["ema200"]) / out["ema200"]) * 100.0,
        np.nan,
    )
    out["abs_e20_to_e200_pct"] = out["e20_to_e200_pct"].abs()

    return out


# =============================================================================
# SIGNAL FILTER ENGINE (PINE PARITY)
# =============================================================================

def build_signal_engine(features: pd.DataFrame, cfg: EmaDistanceConfig) -> pd.DataFrame:
    out = features.copy()

    base_signal = (
        (out["trend_side"] != 0)
        & out["abs_e20_to_e200_pct"].notna()
        & (out["abs_e20_to_e200_pct"] >= cfg.min_abs_distance_pct)
    )

    if cfg.require_price_on_trend_side:
        price_filter = out["price_side_vs_ema20"] == out["trend_side"]
    else:
        price_filter = pd.Series(True, index=out.index)

    if cfg.require_fast_slope_confirm:
        fast_slope_filter = np.where(
            out["trend_side"] == 1,
            out["ema20_slope_pct"] > 0,
            out["ema20_slope_pct"] < 0,
        )
        fast_slope_filter = pd.Series(fast_slope_filter, index=out.index)
    else:
        fast_slope_filter = pd.Series(True, index=out.index)

    if cfg.require_slow_slope_confirm:
        slow_slope_filter = np.where(
            out["trend_side"] == 1,
            out["ema200_slope_pct"] >= 0,
            out["ema200_slope_pct"] <= 0,
        )
        slow_slope_filter = pd.Series(slow_slope_filter, index=out.index)
    else:
        slow_slope_filter = pd.Series(True, index=out.index)

    out["base_signal"] = base_signal.fillna(False)
    out["price_filter"] = price_filter.fillna(False)
    out["fast_slope_filter"] = fast_slope_filter.fillna(False)
    out["slow_slope_filter"] = slow_slope_filter.fillna(False)

    out["research_signal"] = (
        out["base_signal"]
        & out["price_filter"]
        & out["fast_slope_filter"]
        & out["slow_slope_filter"]
        & out["ema20"].notna()
        & out["ema200"].notna()
    ).fillna(False)

    return out


# =============================================================================
# BUCKET ENGINE (PINE PARITY)
# =============================================================================

def build_bucket_engine(features: pd.DataFrame, cfg: EmaDistanceConfig) -> pd.DataFrame:
    out = features.copy()

    out["bucket_index"] = _bucket_index_from_abs_dist(
        out["abs_e20_to_e200_pct"],
        cfg.bucket_edges_pct,
    )
    out["bucket_text"] = _bucket_text_from_index(
        out["bucket_index"],
        cfg.bucket_edges_pct,
    )

    return out


# =============================================================================
# STAGE ENGINE (PINE PARITY)
# =============================================================================

def build_stage_engine(features: pd.DataFrame, cfg: EmaDistanceConfig) -> pd.DataFrame:
    out = features.copy()

    out["stage"] = _stage_from_abs_dist(out["abs_e20_to_e200_pct"], cfg)
    out["stage_text"] = _stage_text(out["stage"])
    out["trend_text"] = out["trend_side"].map(_direction_text)

    return out


# =============================================================================
# FEATURE FRAME
# =============================================================================

def build_feature_frame(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EmaDistanceConfig()

    out = build_ema_core(df, cfg)
    out = build_distance_engine(out, cfg)
    out = build_signal_engine(out, cfg)
    out = build_bucket_engine(out, cfg)
    out = build_stage_engine(out, cfg)

    return out


# =============================================================================
# WEBSITE PAYLOAD
# =============================================================================

def build_latest_payload(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceConfig()
    features = build_feature_frame(df, cfg)

    if features.empty:
        return {
            "module": cfg.module_name,
            "debug_version": cfg.debug_version,
            "ready": False,
            "message": "No data available.",
        }

    last = features.iloc[-1]
    timestamp = _extract_last_timestamp(features, cfg)

    payload: Dict[str, Any] = {
        "module": cfg.module_name,
        "debug_version": cfg.debug_version,
        "ready": True,
        "timestamp": timestamp,
        "trend_side": _safe_int(last["trend_side"]),
        "trend_text": str(last["trend_text"]),
        "price_side_vs_ema20": _safe_int(last["price_side_vs_ema20"]),
        "ema20": _safe_float(last["ema20"]),
        "ema200": _safe_float(last["ema200"]),
        "ema20_slope_pct": _safe_float(last["ema20_slope_pct"]),
        "ema200_slope_pct": _safe_float(last["ema200_slope_pct"]),
        "e20_to_e200_signed": _safe_float(last["e20_to_e200_signed"]),
        "e20_to_e200_pct": _safe_float(last["e20_to_e200_pct"]),
        "abs_e20_to_e200_pct": _safe_float(last["abs_e20_to_e200_pct"]),
        "bucket_index": _safe_int(last["bucket_index"]),
        "bucket_text": str(last["bucket_text"]),
        "stage": _safe_int(last["stage"]),
        "stage_text": str(last["stage_text"]),
        "research_signal": _safe_bool(last["research_signal"]),
    }

    return payload


def build_live_result(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceConfig()
    payload = build_latest_payload(df, cfg)
    return {
        "config": asdict(cfg),
        "latest": payload,
    }