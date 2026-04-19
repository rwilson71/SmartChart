from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# SMARTCHART • EMA DISTANCE CORE
# File: core/cc_ema_distance_calibration.py
# =============================================================================
# Production rebuild
# - Calibration dependency removed
# - Safe fallbacks for missing columns / empty dataframe
# - Website contract aligned
# - Stable latest payload for API / cache / website use
# - Build order:
#   helpers -> engine -> state -> memory -> export -> payload
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class EmaDistanceConfig:
    ema_fast_len: int = 20
    ema_slow_len: int = 200
    slope_len: int = 5

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    time_col: Optional[str] = None

    stage0_max_pct: float = 0.20
    stage1_max_pct: float = 0.50
    stage2_max_pct: float = 0.75

    module_name: str = "ema_distance"
    debug_version: str = "ema_distance_payload_v2_production"


# =============================================================================
# HELPERS
# =============================================================================

def _empty_series(index: pd.Index, dtype: str = "float64") -> pd.Series:
    return pd.Series(index=index, dtype=dtype)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def _safe_timestamp_from_df(df: pd.DataFrame, cfg: EmaDistanceConfig) -> Optional[str]:
    if df is None or len(df) == 0:
        return None

    try:
        if cfg.time_col and cfg.time_col in df.columns:
            ts = pd.to_datetime(df.iloc[-1][cfg.time_col], errors="coerce")
            if pd.notna(ts):
                return ts.isoformat()

        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            ts = df.index[-1]
            if pd.notna(ts):
                return pd.Timestamp(ts).isoformat()
    except Exception:
        return None

    return None


def _normalize_ohlc(
    df: Optional[pd.DataFrame],
    cfg: EmaDistanceConfig,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=[cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col])

    out = df.copy()

    for col in [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if cfg.time_col and cfg.time_col in out.columns:
        try:
            out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")
        except Exception:
            pass

    return out


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def _pct_change_from_past(series: pd.Series, lookback: int) -> pd.Series:
    prev = series.shift(lookback)
    result = np.where(
        prev.notna() & (prev != 0.0),
        ((series - prev) / prev) * 100.0,
        np.nan,
    )
    return pd.Series(result, index=series.index, dtype=float)


def _default_payload(
    cfg: EmaDistanceConfig,
    timestamp: Optional[str] = None,
    message: str = "EMA Distance fallback payload.",
) -> Dict[str, Any]:
    return {
        "module": cfg.module_name,
        "debug_version": cfg.debug_version,
        "ready": False,
        "timestamp": timestamp,
        "state": "Transition",
        "bias_signal": 0,
        "bias_label": "NEUTRAL",
        "indicator_strength": 0.0,
        "market_bias": "NEUTRAL",
        "stage": 0,
        "stage_label": "Transition",
        "distance_pct": None,
        "distance_signed_pct": None,
        "ema20": None,
        "ema200": None,
        "ema20_slope_pct": None,
        "ema200_slope_pct": None,
        "trend_side": 0,
        "price_side_vs_ema20": 0,
        "bars_in_state": 0,
        "message": message,
    }


# =============================================================================
# ENGINE
# =============================================================================

def build_ema_distance_engine(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EmaDistanceConfig()
    out = _normalize_ohlc(df, cfg)

    if out.empty:
        out["ema20"] = _empty_series(out.index)
        out["ema200"] = _empty_series(out.index)
        out["ema20_slope_pct"] = _empty_series(out.index)
        out["ema200_slope_pct"] = _empty_series(out.index)
        out["trend_side"] = _empty_series(out.index, dtype="int64")
        out["price_side_vs_ema20"] = _empty_series(out.index, dtype="int64")
        out["distance_signed"] = _empty_series(out.index)
        out["distance_signed_pct"] = _empty_series(out.index)
        out["distance_pct"] = _empty_series(out.index)
        return out

    close = out[cfg.close_col]

    out["ema20"] = _ema(close, cfg.ema_fast_len)
    out["ema200"] = _ema(close, cfg.ema_slow_len)
    out["ema20_slope_pct"] = _pct_change_from_past(out["ema20"], cfg.slope_len)
    out["ema200_slope_pct"] = _pct_change_from_past(out["ema200"], cfg.slope_len)

    out["trend_side"] = np.where(
        out["ema20"] > out["ema200"],
        1,
        np.where(out["ema20"] < out["ema200"], -1, 0),
    ).astype(int)

    out["price_side_vs_ema20"] = np.where(
        close > out["ema20"],
        1,
        np.where(close < out["ema20"], -1, 0),
    ).astype(int)

    out["distance_signed"] = out["ema20"] - out["ema200"]
    out["distance_signed_pct"] = np.where(
        out["ema200"].notna() & (out["ema200"] != 0.0),
        ((out["ema20"] - out["ema200"]) / out["ema200"]) * 100.0,
        np.nan,
    )
    out["distance_pct"] = out["distance_signed_pct"].abs()

    return out


# =============================================================================
# STATE
# =============================================================================

def build_ema_distance_state(
    features: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EmaDistanceConfig()
    out = features.copy()

    if out.empty:
        out["stage"] = _empty_series(out.index, dtype="int64")
        out["stage_label"] = _empty_series(out.index, dtype="object")
        out["state"] = _empty_series(out.index, dtype="object")
        out["bias_signal"] = _empty_series(out.index, dtype="int64")
        out["bias_label"] = _empty_series(out.index, dtype="object")
        out["indicator_strength"] = _empty_series(out.index)
        out["market_bias"] = _empty_series(out.index, dtype="object")
        return out

    dist = pd.to_numeric(out["distance_pct"], errors="coerce").fillna(0.0)

    stage = np.select(
        [
            dist < cfg.stage0_max_pct,
            (dist >= cfg.stage0_max_pct) & (dist < cfg.stage1_max_pct),
            (dist >= cfg.stage1_max_pct) & (dist < cfg.stage2_max_pct),
            dist >= cfg.stage2_max_pct,
        ],
        [0, 1, 2, 3],
        default=0,
    )

    out["stage"] = pd.Series(stage, index=out.index, dtype=int)

    stage_map = {
        0: "Transition",
        1: "Optimal Trend",
        2: "Extended Trend",
        3: "Exhaustion",
    }
    out["stage_label"] = out["stage"].map(stage_map).fillna("Transition")
    out["state"] = out["stage_label"]

    bias_ok = (
        out["trend_side"].notna()
        & out["price_side_vs_ema20"].notna()
        & (
            (out["trend_side"] == out["price_side_vs_ema20"])
            | (out["price_side_vs_ema20"] == 0)
        )
    )

    out["bias_signal"] = np.where(
        bias_ok,
        out["trend_side"].fillna(0).astype(int),
        0,
    ).astype(int)

    bias_map = {
        1: "BULLISH",
        0: "NEUTRAL",
        -1: "BEARISH",
    }
    out["bias_label"] = out["bias_signal"].map(bias_map).fillna("NEUTRAL")
    out["market_bias"] = out["bias_label"]

    strength = np.select(
        [
            out["stage"] == 0,
            out["stage"] == 1,
            out["stage"] == 2,
            out["stage"] == 3,
        ],
        [25.0, 70.0, 85.0, 60.0],
        default=0.0,
    )

    out["indicator_strength"] = np.where(
        out["bias_signal"] == 0,
        np.minimum(strength, 35.0),
        strength,
    ).astype(float)

    return out


# =============================================================================
# MEMORY
# =============================================================================

def build_ema_distance_memory(
    features: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> pd.DataFrame:
    _ = cfg or EmaDistanceConfig()
    out = features.copy()

    if out.empty or "stage" not in out.columns:
        out["bars_in_state"] = _empty_series(out.index, dtype="int64")
        return out

    groups = (out["stage"] != out["stage"].shift(1)).cumsum()
    out["bars_in_state"] = out.groupby(groups).cumcount() + 1
    out["bars_in_state"] = out["bars_in_state"].fillna(0).astype(int)

    return out


# =============================================================================
# EXPORT
# =============================================================================

def build_feature_frame(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or EmaDistanceConfig()

    out = build_ema_distance_engine(df, cfg)
    out = build_ema_distance_state(out, cfg)
    out = build_ema_distance_memory(out, cfg)

    return out


# =============================================================================
# PAYLOAD
# =============================================================================

def build_latest_payload(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceConfig()
    timestamp = _safe_timestamp_from_df(df if isinstance(df, pd.DataFrame) else None, cfg)

    try:
        features = build_feature_frame(df, cfg)

        if features.empty:
            return _default_payload(
                cfg=cfg,
                timestamp=timestamp,
                message="EMA Distance received empty dataframe.",
            )

        last = features.iloc[-1]

        payload: Dict[str, Any] = {
            "module": cfg.module_name,
            "debug_version": cfg.debug_version,
            "ready": True,
            "timestamp": timestamp,
            "state": _safe_str(last.get("state"), "Transition"),
            "bias_signal": _safe_int(last.get("bias_signal"), 0),
            "bias_label": _safe_str(last.get("bias_label"), "NEUTRAL"),
            "indicator_strength": float(
                max(0.0, min(100.0, _safe_float(last.get("indicator_strength"), 0.0) or 0.0))
            ),
            "market_bias": _safe_str(last.get("market_bias"), "NEUTRAL"),
            "stage": _safe_int(last.get("stage"), 0),
            "stage_label": _safe_str(last.get("stage_label"), "Transition"),
            "distance_pct": _safe_float(last.get("distance_pct")),
            "distance_signed_pct": _safe_float(last.get("distance_signed_pct")),
            "ema20": _safe_float(last.get("ema20")),
            "ema200": _safe_float(last.get("ema200")),
            "ema20_slope_pct": _safe_float(last.get("ema20_slope_pct")),
            "ema200_slope_pct": _safe_float(last.get("ema200_slope_pct")),
            "trend_side": _safe_int(last.get("trend_side"), 0),
            "price_side_vs_ema20": _safe_int(last.get("price_side_vs_ema20"), 0),
            "bars_in_state": _safe_int(last.get("bars_in_state"), 0),
        }

        return payload

    except Exception as exc:
        return _default_payload(
            cfg=cfg,
            timestamp=timestamp,
            message=f"EMA Distance fallback triggered: {exc}",
        )


def build_ema_distance_latest_payload(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
) -> Dict[str, Any]:
    return build_latest_payload(df, cfg)


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