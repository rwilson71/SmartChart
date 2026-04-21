from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class PatternConfig:
    lookback_bars: int = 50000
    min_samples: int = 120

    swing_window: int = 5
    trend_window: int = 20
    volatility_window: int = 20
    pattern_lookback: int = 12

    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45

    strong_pattern_threshold: float = 0.70
    moderate_pattern_threshold: float = 0.45


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pattern input missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Pattern input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Pattern input dataframe must use a DatetimeIndex")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(_clamp((value - low) / (high - low), 0.0, 1.0))


def _bias_from_score(score: float, bull_th: float, bear_th: float) -> Tuple[int, str]:
    if score >= bull_th:
        return 1, "BULLISH"
    if score <= bear_th:
        return -1, "BEARISH"
    return 0, "NEUTRAL"


def _prepare_df(df: pd.DataFrame, cfg: PatternConfig) -> pd.DataFrame:
    out = df.copy().sort_index()

    if cfg.lookback_bars > 0:
        out = out.tail(cfg.lookback_bars).copy()

    out["ret_pct"] = ((out["close"] / out["open"]) - 1.0) * 100.0
    out["hl_range"] = out["high"] - out["low"]
    out["body"] = out["close"] - out["open"]
    out["body_abs"] = out["body"].abs()

    out["range_mean"] = out["hl_range"].rolling(
        cfg.volatility_window,
        min_periods=max(5, cfg.volatility_window // 2),
    ).mean()

    out["ret_mean"] = out["ret_pct"].rolling(
        cfg.trend_window,
        min_periods=max(5, cfg.trend_window // 2),
    ).mean()

    out["close_ma"] = out["close"].rolling(
        cfg.trend_window,
        min_periods=max(5, cfg.trend_window // 2),
    ).mean()

    out["dist_ma_pct"] = ((out["close"] / out["close_ma"]) - 1.0) * 100.0

    out["body_frac"] = (
        out["body_abs"] / out["hl_range"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["upper_wick"] = out["high"] - np.maximum(out["open"], out["close"])
    out["lower_wick"] = np.minimum(out["open"], out["close"]) - out["low"]

    out["upper_wick_frac"] = (
        out["upper_wick"] / out["hl_range"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["lower_wick_frac"] = (
        out["lower_wick"] / out["hl_range"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def _detect_single_bar_pattern(row: pd.Series) -> Dict[str, Any]:
    body_frac = _safe_float(row.get("body_frac"))
    upper = _safe_float(row.get("upper_wick_frac"))
    lower = _safe_float(row.get("lower_wick_frac"))
    body = _safe_float(row.get("body"))
    ret_pct = _safe_float(row.get("ret_pct"))

    bullish_pin = lower >= 0.45 and body_frac <= 0.35 and body >= 0
    bearish_pin = upper >= 0.45 and body_frac <= 0.35 and body <= 0

    bullish_impulse = body_frac >= 0.60 and ret_pct > 0
    bearish_impulse = body_frac >= 0.60 and ret_pct < 0

    indecision = body_frac <= 0.20 and upper >= 0.25 and lower >= 0.25

    if bullish_pin:
        return {"pattern_name": "BULLISH_PINBAR", "pattern_score": 0.72, "direction": 1}
    if bearish_pin:
        return {"pattern_name": "BEARISH_PINBAR", "pattern_score": 0.72, "direction": -1}
    if bullish_impulse:
        return {"pattern_name": "BULLISH_IMPULSE", "pattern_score": 0.78, "direction": 1}
    if bearish_impulse:
        return {"pattern_name": "BEARISH_IMPULSE", "pattern_score": 0.78, "direction": -1}
    if indecision:
        return {"pattern_name": "INDECISION", "pattern_score": 0.40, "direction": 0}

    return {"pattern_name": "NONE", "pattern_score": 0.20, "direction": 0}


def _detect_structure_pattern(work: pd.DataFrame, cfg: PatternConfig) -> Dict[str, Any]:
    tail = work.tail(cfg.pattern_lookback).copy()
    if len(tail) < max(5, cfg.swing_window):
        return {
            "structure_name": "INSUFFICIENT_DATA",
            "structure_score": 0.0,
            "direction": 0,
        }

    highs = tail["high"]
    lows = tail["low"]
    closes = tail["close"]
    ret_mean = _safe_float(tail["ret_mean"].iloc[-1])
    dist_ma_pct = _safe_float(tail["dist_ma_pct"].iloc[-1])

    hh = _safe_float(highs.iloc[-1]) > _safe_float(highs.iloc[:-1].max())
    ll = _safe_float(lows.iloc[-1]) < _safe_float(lows.iloc[:-1].min())

    higher_lows = lows.diff().dropna().tail(4).gt(0).sum() >= 3
    lower_highs = highs.diff().dropna().tail(4).lt(0).sum() >= 3

    compression = (highs.max() - lows.min()) / max(_safe_float(closes.mean(), 1.0), 1e-9) < 0.0035

    if hh and higher_lows and ret_mean > 0 and dist_ma_pct > 0:
        return {"structure_name": "BULLISH_BREAKOUT_BUILD", "structure_score": 0.82, "direction": 1}

    if ll and lower_highs and ret_mean < 0 and dist_ma_pct < 0:
        return {"structure_name": "BEARISH_BREAKDOWN_BUILD", "structure_score": 0.82, "direction": -1}

    if compression and ret_mean > 0:
        return {"structure_name": "BULLISH_COMPRESSION", "structure_score": 0.62, "direction": 1}

    if compression and ret_mean < 0:
        return {"structure_name": "BEARISH_COMPRESSION", "structure_score": 0.62, "direction": -1}

    if abs(ret_mean) < 0.01:
        return {"structure_name": "RANGE_ROTATION", "structure_score": 0.45, "direction": 0}

    return {"structure_name": "TRANSITION", "structure_score": 0.35, "direction": 0}


def _combine_pattern_scores(
    single_bar: Dict[str, Any],
    structure: Dict[str, Any],
) -> Dict[str, Any]:
    sb_score = _safe_float(single_bar.get("pattern_score"))
    st_score = _safe_float(structure.get("structure_score"))

    sb_dir = int(single_bar.get("direction", 0))
    st_dir = int(structure.get("direction", 0))

    agreement = 1.0 if sb_dir == st_dir and sb_dir != 0 else 0.5 if (sb_dir == 0 or st_dir == 0) else 0.0

    combined = _clamp((0.40 * sb_score) + (0.60 * st_score), 0.0, 1.0)
    combined = _clamp((combined * 0.80) + (agreement * 0.20), 0.0, 1.0)

    if st_dir != 0:
        final_dir = st_dir
    else:
        final_dir = sb_dir

    directional_score = 0.5
    if final_dir > 0:
        directional_score = 0.5 + (combined * 0.5)
    elif final_dir < 0:
        directional_score = 0.5 - (combined * 0.5)

    directional_score = _clamp(directional_score, 0.0, 1.0)

    return {
        "combined_score": round(combined, 4),
        "agreement_score": round(agreement, 4),
        "direction": final_dir,
        "directional_score": round(directional_score, 4),
    }


def _pattern_state(score: float, bias_label: str, cfg: PatternConfig) -> str:
    if score >= cfg.strong_pattern_threshold:
        return f"{bias_label}_STRONG_PATTERN"
    if score >= cfg.moderate_pattern_threshold:
        return f"{bias_label}_DEVELOPING_PATTERN"
    return "NEUTRAL_PATTERN"


# =============================================================================
# ENGINE
# =============================================================================

def build_pattern_payload(
    df: pd.DataFrame,
    config: Optional[PatternConfig] = None,
) -> Dict[str, Any]:
    cfg = config or PatternConfig()

    _validate_ohlc(df)
    work = _prepare_df(df, cfg)

    if len(work) < cfg.min_samples:
        raise ValueError(f"Pattern requires at least {cfg.min_samples} rows, got {len(work)}")

    latest_row = work.iloc[-1]

    single_bar = _detect_single_bar_pattern(latest_row)
    structure = _detect_structure_pattern(work, cfg)
    combo = _combine_pattern_scores(single_bar, structure)

    bias_signal, bias_label = _bias_from_score(
        combo["directional_score"],
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    pattern_state = _pattern_state(combo["combined_score"], bias_label, cfg)

    latest_ts = work.index[-1]

    payload: Dict[str, Any] = {
        "indicator": "FORECASTER_PATTERN",
        "timestamp": latest_ts.isoformat(),
        "state": pattern_state,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": combo["combined_score"],

        "pattern_state": pattern_state,
        "primary_pattern": single_bar["pattern_name"],
        "structure_pattern": structure["structure_name"],

        "pattern_score": round(_safe_float(single_bar["pattern_score"]), 4),
        "structure_score": round(_safe_float(structure["structure_score"]), 4),
        "combined_score": combo["combined_score"],
        "agreement_score": combo["agreement_score"],
        "directional_score": combo["directional_score"],

        "ret_pct": round(_safe_float(latest_row.get("ret_pct")), 4),
        "body_frac": round(_safe_float(latest_row.get("body_frac")), 4),
        "upper_wick_frac": round(_safe_float(latest_row.get("upper_wick_frac")), 4),
        "lower_wick_frac": round(_safe_float(latest_row.get("lower_wick_frac")), 4),
        "dist_ma_pct": round(_safe_float(latest_row.get("dist_ma_pct")), 4),
        "ret_mean": round(_safe_float(latest_row.get("ret_mean")), 4),

        "data_mode": "PRICE_PATTERN_PROXY_V1",
        "future_mode": "HISTORICAL_PATTERN_MATCHING_V2",

        "config": asdict(cfg),
    }

    return payload


def build_pattern_latest_payload(
    df: pd.DataFrame,
    config: Optional[PatternConfig] = None,
) -> Dict[str, Any]:
    """
    Alias kept to match SmartChart payload-builder naming.
    """
    return build_pattern_payload(df=df, config=config)