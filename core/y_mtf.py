from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MTFConfig:
    # Base / validation
    base_tf: str = "1min"
    min_rows: int = 300

    # EMA structure
    ema_fast: int = 20
    ema_slow: int = 50
    ema_signal: int = 200

    # Slope / momentum
    slope_lookback: int = 5
    strength_clip: float = 100.0

    # ATR / structure
    atr_len: int = 14
    trend_flat_atr_pct: float = 0.12

    # Calendar behavior
    calendar_lookback_weeks: int = 12

    # Macro anchors (rolling lookback windows, not true macro bars)
    macro_anchor_years: Tuple[int, ...] = (15, 10, 7, 5, 3)

    # Structural ladder
    structural_frames: Tuple[str, ...] = ("M", "W", "D", "4H", "1H")

    # Execution ladder
    execution_frames: Tuple[str, ...] = ("30M", "15M", "5M", "2M", "1M")

    # Weights: macro
    weight_15y: float = 1.30
    weight_10y: float = 1.20
    weight_7y: float = 1.10
    weight_5y: float = 1.00
    weight_3y: float = 0.90

    # Weights: structural
    weight_M: float = 1.25
    weight_W: float = 1.20
    weight_D: float = 1.10
    weight_4H: float = 1.00
    weight_1H: float = 0.95

    # Weights: execution
    weight_30M: float = 1.00
    weight_15M: float = 0.95
    weight_5M: float = 0.90
    weight_2M: float = 0.85
    weight_1M: float = 0.80

    # Group weights
    group_weight_macro: float = 1.30
    group_weight_structural: float = 1.10
    group_weight_execution: float = 0.90

    # Agreement / conflict thresholds
    strong_alignment_threshold: float = 0.60
    weak_alignment_threshold: float = 0.20
    strong_conflict_threshold: float = 0.55

    # Confidence thresholds
    high_confidence_threshold: float = 75.0
    medium_confidence_threshold: float = 45.0

    # Transition thresholds
    direction_trend_delta_threshold: float = 8.0
    conflict_trend_delta_threshold: float = 0.10
    alignment_trend_delta_threshold: float = 0.10

    # Website contract fields
    debug_version: str = "y_mtf_v2"


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MTF build error: missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("MTF build error: input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("MTF build error: dataframe index must be a DatetimeIndex")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sign_label(
    v: float,
    pos_label: str = "BULLISH",
    neg_label: str = "BEARISH",
    neutral_label: str = "NEUTRAL",
) -> str:
    if v > 0:
        return pos_label
    if v < 0:
        return neg_label
    return neutral_label


def _sign_int(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _state_from_score(v: float) -> str:
    av = abs(v)
    if av >= 75:
        return "strong"
    if av >= 45:
        return "moderate"
    if av >= 20:
        return "weak"
    return "neutral"


def _agreement_label(score: float, strong_thr: float, weak_thr: float) -> str:
    a = abs(score)
    if a >= strong_thr:
        return "STRONG"
    if a >= weak_thr:
        return "MIXED"
    return "NEUTRAL"


def _conflict_label(score: float, strong_thr: float) -> str:
    if score >= strong_thr:
        return "HIGH_CONFLICT"
    if score >= 0.25:
        return "MODERATE_CONFLICT"
    return "LOW_CONFLICT"


def _confidence_label(score: float, cfg: MTFConfig) -> str:
    if score >= cfg.high_confidence_threshold:
        return "HIGH"
    if score >= cfg.medium_confidence_threshold:
        return "MEDIUM"
    return "LOW"


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return _true_range(df).rolling(length, min_periods=1).mean()


def _resample_rule(tf: str) -> str:
    mapping = {
        "1M": "1min",
        "2M": "2min",
        "5M": "5min",
        "15M": "15min",
        "30M": "30min",
        "1H": "1h",
        "4H": "4h",
        "D": "1D",
        "W": "1W",
        "M": "1ME",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _resample_rule(tf)

    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg_map["volume"] = "sum"

    out = df.resample(rule).agg(agg_map).dropna(subset=["open", "high", "low", "close"])
    return out


def _bars_per_day(base_df: pd.DataFrame) -> float:
    if len(base_df) < 10:
        return 1440.0

    deltas = base_df.index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return 1440.0

    median_sec = max(deltas.median(), 1.0)
    return 86400.0 / median_sec


def _approx_window_bars_for_years(base_df: pd.DataFrame, years: int) -> int:
    bpd = _bars_per_day(base_df)
    return max(int(round(years * 365.25 * bpd)), 50)


def _last_n_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        return df.copy()
    return df.tail(n).copy()


def _pct_distance(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return ((a - b) / abs(b)) * 100.0


def _normalize_signed(value: float, scale: float, clip_val: float = 100.0) -> float:
    if abs(scale) < 1e-12:
        return 0.0
    return _clip((value / scale) * 100.0, -clip_val, clip_val)


def _weighted_average(values: List[float], weights: List[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0

    weight_sum = sum(abs(w) for w in weights)
    if weight_sum <= 0:
        return 0.0

    return float(sum(v * w for v, w in zip(values, weights)) / weight_sum)


def _dominant_regime(frame_features: List[Dict[str, Any]]) -> str:
    valid = [f for f in frame_features if f.get("valid", False)]
    if not valid:
        return "unknown"

    counts: Dict[str, int] = {}
    for f in valid:
        key = str(f.get("regime_state", "unknown"))
        counts[key] = counts.get(key, 0) + 1

    return max(counts.items(), key=lambda x: x[1])[0]


def _frame_weight(frame_name: str, cfg: MTFConfig) -> float:
    mapping = {
        "15Y": cfg.weight_15y,
        "10Y": cfg.weight_10y,
        "7Y": cfg.weight_7y,
        "5Y": cfg.weight_5y,
        "3Y": cfg.weight_3y,
        "M": cfg.weight_M,
        "W": cfg.weight_W,
        "D": cfg.weight_D,
        "4H": cfg.weight_4H,
        "1H": cfg.weight_1H,
        "30M": cfg.weight_30M,
        "15M": cfg.weight_15M,
        "5M": cfg.weight_5M,
        "2M": cfg.weight_2M,
        "1M": cfg.weight_1M,
    }
    return mapping.get(frame_name, 1.0)


def _group_weight(group_name: str, cfg: MTFConfig) -> float:
    mapping = {
        "macro": cfg.group_weight_macro,
        "structural": cfg.group_weight_structural,
        "execution": cfg.group_weight_execution,
    }
    return mapping.get(group_name, 1.0)


def _safe_prev(series: pd.Series, steps_back: int, default: float = 0.0) -> float:
    if len(series) <= steps_back:
        return default
    return _safe_float(series.iloc[-steps_back - 1], default)


# =============================================================================
# PER-FRAME FEATURE EXTRACTION
# =============================================================================

def _compute_frame_features(
    df: pd.DataFrame,
    cfg: MTFConfig,
    frame_name: str,
    frame_group: str,
) -> Dict[str, Any]:
    min_needed = max(cfg.ema_signal + 5, cfg.atr_len + 5, cfg.slope_lookback + 5)

    if len(df) < min_needed:
        close_now = _safe_float(df["close"].iloc[-1]) if len(df) else 0.0
        return {
            "frame": frame_name,
            "group": frame_group,
            "rows": int(len(df)),
            "valid": False,
            "direction": 0.0,
            "direction_label": "NEUTRAL",
            "direction_sign": 0,
            "strength": 0.0,
            "strength_label": "neutral",
            "confidence_score": 0.0,
            "confidence_label": "LOW",
            "trend_state": "insufficient_data",
            "regime_state": "unknown",
            "ema_fast": np.nan,
            "ema_slow": np.nan,
            "ema_signal": np.nan,
            "close": close_now,
            "atr": 0.0,
            "fast_vs_slow_pct": 0.0,
            "close_vs_signal_pct": 0.0,
            "fast_slope": 0.0,
            "slow_slope": 0.0,
            "atr_pct": 0.0,
            "window_quality": 0.0,
            "structure_quality": 0.0,
            "slope_quality": 0.0,
        }

    x = df.copy()
    x["ema_fast"] = _ema(x["close"], cfg.ema_fast)
    x["ema_slow"] = _ema(x["close"], cfg.ema_slow)
    x["ema_signal"] = _ema(x["close"], cfg.ema_signal)
    x["atr"] = _atr(x, cfg.atr_len)

    last = x.iloc[-1]
    lb = max(cfg.slope_lookback, 1)

    ema_fast_now = _safe_float(last["ema_fast"])
    ema_slow_now = _safe_float(last["ema_slow"])
    ema_signal_now = _safe_float(last["ema_signal"])
    close_now = _safe_float(last["close"])
    atr_now = max(_safe_float(last["atr"]), 1e-9)

    ema_fast_prev = _safe_float(x["ema_fast"].iloc[-lb - 1]) if len(x) > lb else ema_fast_now
    ema_slow_prev = _safe_float(x["ema_slow"].iloc[-lb - 1]) if len(x) > lb else ema_slow_now

    fast_slope = ema_fast_now - ema_fast_prev
    slow_slope = ema_slow_now - ema_slow_prev

    fast_vs_slow = _pct_distance(ema_fast_now, ema_slow_now)
    close_vs_signal = _pct_distance(close_now, ema_signal_now)

    trend_dir_raw = 0.0
    if ema_fast_now > ema_slow_now and close_now >= ema_signal_now:
        trend_dir_raw = 1.0
    elif ema_fast_now < ema_slow_now and close_now <= ema_signal_now:
        trend_dir_raw = -1.0
    else:
        trend_dir_raw = 0.0

    slope_score = _normalize_signed(fast_slope + (0.5 * slow_slope), atr_now, cfg.strength_clip)
    structure_score = _clip((fast_vs_slow * 1.5) + (close_vs_signal * 0.8), -100.0, 100.0)

    raw_direction = _clip(
        (trend_dir_raw * 45.0) + (0.35 * slope_score) + (0.20 * structure_score),
        -100.0,
        100.0,
    )
    strength = abs(raw_direction)

    atr_pct = abs(_pct_distance(atr_now, close_now))
    flat_threshold = cfg.trend_flat_atr_pct

    if abs(raw_direction) < 12:
        trend_state = "neutral"
    elif raw_direction > 0:
        trend_state = "bullish"
    else:
        trend_state = "bearish"

    if atr_pct < flat_threshold and abs(fast_vs_slow) < 0.10:
        regime_state = "compression"
    elif abs(raw_direction) >= 55 and atr_pct >= flat_threshold:
        regime_state = "expansion"
    elif abs(raw_direction) >= 25:
        regime_state = "trend"
    else:
        regime_state = "transition"

    # Confidence subcomponents
    window_quality = _clip((len(x) / float(min_needed)) * 100.0, 0.0, 100.0)
    structure_quality = _clip((abs(fast_vs_slow) * 25.0) + (abs(close_vs_signal) * 10.0), 0.0, 100.0)
    slope_quality = _clip(abs(slope_score), 0.0, 100.0)
    regime_bonus = 15.0 if regime_state in {"trend", "expansion"} else 0.0
    neutral_penalty = 20.0 if trend_state == "neutral" else 0.0

    confidence_score = _clip(
        (0.25 * window_quality)
        + (0.30 * structure_quality)
        + (0.30 * slope_quality)
        + (0.25 * strength)
        + regime_bonus
        - neutral_penalty,
        0.0,
        100.0,
    )

    return {
        "frame": frame_name,
        "group": frame_group,
        "rows": int(len(x)),
        "valid": True,
        "direction": _safe_float(raw_direction),
        "direction_label": _sign_label(raw_direction),
        "direction_sign": _sign_int(raw_direction),
        "strength": _safe_float(strength),
        "strength_label": _state_from_score(strength),
        "confidence_score": _safe_float(confidence_score),
        "confidence_label": _confidence_label(confidence_score, cfg),
        "trend_state": trend_state,
        "regime_state": regime_state,
        "ema_fast": ema_fast_now,
        "ema_slow": ema_slow_now,
        "ema_signal": ema_signal_now,
        "close": close_now,
        "atr": atr_now,
        "fast_vs_slow_pct": _safe_float(fast_vs_slow),
        "close_vs_signal_pct": _safe_float(close_vs_signal),
        "fast_slope": _safe_float(fast_slope),
        "slow_slope": _safe_float(slow_slope),
        "atr_pct": _safe_float(atr_pct),
        "window_quality": _safe_float(window_quality),
        "structure_quality": _safe_float(structure_quality),
        "slope_quality": _safe_float(slope_quality),
    }


# =============================================================================
# AGREEMENT / CONFLICT / CONFIDENCE
# =============================================================================

def _agreement_score(frame_features: List[Dict[str, Any]]) -> float:
    valid = [f for f in frame_features if f.get("valid", False)]
    if not valid:
        return 0.0

    dirs = [np.sign(_safe_float(f["direction"])) for f in valid]
    if not dirs:
        return 0.0

    return float(np.mean(dirs))


def _conflict_score(frame_features: List[Dict[str, Any]]) -> float:
    valid = [f for f in frame_features if f.get("valid", False)]
    if len(valid) < 2:
        return 0.0

    dirs = np.array([np.sign(_safe_float(f["direction"])) for f in valid], dtype=float)
    pos = np.mean(dirs > 0)
    neg = np.mean(dirs < 0)

    return float(min(pos, neg) * 2.0)


def _build_group_summary(
    group_name: str,
    frame_features: List[Dict[str, Any]],
    cfg: MTFConfig,
) -> Dict[str, Any]:
    valid = [f for f in frame_features if f.get("valid", False)]

    if not valid:
        return {
            "group": group_name,
            "valid_count": 0,
            "weighted_direction": 0.0,
            "weighted_strength": 0.0,
            "group_confidence_score": 0.0,
            "group_confidence_label": "LOW",
            "bias_label": "NEUTRAL",
            "agreement": 0.0,
            "agreement_label": "NEUTRAL",
            "conflict_score": 0.0,
            "conflict_label": "LOW_CONFLICT",
            "dominant_regime": "unknown",
            "frames": frame_features,
        }

    directions = [_safe_float(f["direction"]) for f in valid]
    strengths = [_safe_float(f["strength"]) for f in valid]
    confidences = [_safe_float(f["confidence_score"]) for f in valid]
    weights = [_frame_weight(str(f["frame"]), cfg) for f in valid]

    weighted_direction = _weighted_average(directions, weights)
    weighted_strength = _weighted_average(strengths, weights)
    weighted_confidence = _weighted_average(confidences, weights)
    agreement = _agreement_score(valid)
    conflict = _conflict_score(valid)

    group_confidence = _clip(
        (0.45 * weighted_confidence)
        + (0.25 * abs(weighted_strength))
        + (0.20 * abs(agreement) * 100.0)
        - (0.25 * conflict * 100.0),
        0.0,
        100.0,
    )

    return {
        "group": group_name,
        "valid_count": len(valid),
        "weighted_direction": _safe_float(weighted_direction),
        "weighted_strength": _safe_float(weighted_strength),
        "group_confidence_score": _safe_float(group_confidence),
        "group_confidence_label": _confidence_label(group_confidence, cfg),
        "bias_label": _sign_label(weighted_direction),
        "agreement": _safe_float(agreement),
        "agreement_label": _agreement_label(
            agreement,
            strong_thr=cfg.strong_alignment_threshold,
            weak_thr=cfg.weak_alignment_threshold,
        ),
        "conflict_score": _safe_float(conflict),
        "conflict_label": _conflict_label(conflict, cfg.strong_conflict_threshold),
        "dominant_regime": _dominant_regime(valid),
        "frames": frame_features,
    }


# =============================================================================
# CALENDAR BEHAVIOR
# =============================================================================

def _build_calendar_context(df: pd.DataFrame, cfg: MTFConfig) -> Dict[str, Any]:
    if len(df) < 20:
        return {
            "day_of_week": "unknown",
            "calendar_bias": "NEUTRAL",
            "calendar_strength": 0.0,
            "calendar_context": "insufficient_data",
            "day_behavior_score": 0.0,
        }

    x = df.copy()
    x["ret_1"] = x["close"].pct_change()
    x["weekday"] = x.index.day_name()

    current_day = x.index[-1].day_name()

    lookback_days = int(cfg.calendar_lookback_weeks * 5)
    x = x.tail(max(lookback_days * 1440, 2000)) if len(x) > 2000 else x

    day_rows = x[x["weekday"] == current_day].copy()
    if day_rows.empty:
        return {
            "day_of_week": current_day,
            "calendar_bias": "NEUTRAL",
            "calendar_strength": 0.0,
            "calendar_context": "no_history",
            "day_behavior_score": 0.0,
        }

    mean_ret = _safe_float(day_rows["ret_1"].mean(), 0.0)
    std_ret = _safe_float(day_rows["ret_1"].std(), 0.0)

    if std_ret <= 1e-12:
        score = 0.0
    else:
        score = _clip((mean_ret / std_ret) * 25.0, -100.0, 100.0)

    if score >= 15:
        context = "continuation_support"
    elif score <= -15:
        context = "reversal_or_exhaustion_risk"
    else:
        context = "neutral_behavior"

    return {
        "day_of_week": current_day,
        "calendar_bias": _sign_label(score),
        "calendar_strength": abs(_safe_float(score)),
        "calendar_context": context,
        "day_behavior_score": _safe_float(score),
    }


# =============================================================================
# FRAME BUILDERS
# =============================================================================

def _build_macro_frames(df: pd.DataFrame, cfg: MTFConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for years in cfg.macro_anchor_years:
        bars = _approx_window_bars_for_years(df, years)
        sub = _last_n_rows(df, bars)

        frame_name = f"{years}Y"
        feat = _compute_frame_features(
            df=sub,
            cfg=cfg,
            frame_name=frame_name,
            frame_group="macro",
        )
        feat["anchor_window_years"] = years
        feat["anchor_window_bars"] = int(bars)
        feat["anchor_model"] = "rolling_lookback_window"
        out.append(feat)

    return out


def _build_structural_frames(df: pd.DataFrame, cfg: MTFConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for tf in cfg.structural_frames:
        rs = _resample_ohlcv(df, tf)
        feat = _compute_frame_features(
            df=rs,
            cfg=cfg,
            frame_name=tf,
            frame_group="structural",
        )
        out.append(feat)

    return out


def _build_execution_frames(df: pd.DataFrame, cfg: MTFConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for tf in cfg.execution_frames:
        if tf == "1M":
            rs = df.copy()
        else:
            rs = _resample_ohlcv(df, tf)

        feat = _compute_frame_features(
            df=rs,
            cfg=cfg,
            frame_name=tf,
            frame_group="execution",
        )
        out.append(feat)

    return out


# =============================================================================
# CONTRADICTION / TRANSITION ENGINE
# =============================================================================

def _build_contradiction_map(
    macro: Dict[str, Any],
    structural: Dict[str, Any],
    execution: Dict[str, Any],
    cfg: MTFConfig,
) -> Dict[str, Any]:
    groups = {
        "macro": _safe_float(macro.get("weighted_direction", 0.0)),
        "structural": _safe_float(structural.get("weighted_direction", 0.0)),
        "execution": _safe_float(execution.get("weighted_direction", 0.0)),
    }

    signs = {k: _sign_int(v) for k, v in groups.items()}
    abs_vals = {k: abs(v) for k, v in groups.items()}

    pos_groups = [k for k, s in signs.items() if s > 0]
    neg_groups = [k for k, s in signs.items() if s < 0]
    neutral_groups = [k for k, s in signs.items() if s == 0]

    contradiction_type = "none"
    contradiction_source = "none"
    dominant_opposing_group = "none"

    if len(pos_groups) > 0 and len(neg_groups) > 0:
        contradiction_type = "directional_opposition"

        if signs["macro"] != 0 and signs["execution"] != 0 and signs["macro"] != signs["execution"]:
            contradiction_source = "macro_vs_execution"
        elif signs["macro"] != 0 and signs["structural"] != 0 and signs["macro"] != signs["structural"]:
            contradiction_source = "macro_vs_structural"
        elif signs["structural"] != 0 and signs["execution"] != 0 and signs["structural"] != signs["execution"]:
            contradiction_source = "structural_vs_execution"
        else:
            contradiction_source = "mixed_group_opposition"

        opposing_candidates = neg_groups if len(pos_groups) > len(neg_groups) else pos_groups
        if opposing_candidates:
            dominant_opposing_group = max(opposing_candidates, key=lambda x: abs_vals.get(x, 0.0))

    elif len(neutral_groups) >= 1 and (len(pos_groups) >= 1 or len(neg_groups) >= 1):
        contradiction_type = "partial_alignment_with_neutral_drag"
        contradiction_source = ",".join(neutral_groups)

    severity_raw = _clip(
        (
            0.55 * _safe_float(min(
                macro.get("conflict_score", 0.0),
                1.0
            ) * 100.0)
            + 0.20 * abs(_safe_float(macro.get("weighted_direction", 0.0)) - _safe_float(structural.get("weighted_direction", 0.0))) / 2.0
            + 0.25 * abs(_safe_float(structural.get("weighted_direction", 0.0)) - _safe_float(execution.get("weighted_direction", 0.0))) / 2.0
        ),
        0.0,
        100.0,
    )

    # Better aggregate severity from all conflicts
    avg_conflict = np.mean([
        _safe_float(macro.get("conflict_score", 0.0)),
        _safe_float(structural.get("conflict_score", 0.0)),
        _safe_float(execution.get("conflict_score", 0.0)),
    ])

    group_dispersion = np.std([
        _safe_float(macro.get("weighted_direction", 0.0)),
        _safe_float(structural.get("weighted_direction", 0.0)),
        _safe_float(execution.get("weighted_direction", 0.0)),
    ])

    severity = _clip(
        (0.55 * avg_conflict * 100.0) + (0.45 * group_dispersion),
        0.0,
        100.0,
    )

    if contradiction_type == "none":
        severity = min(severity, 20.0)

    if severity >= 70:
        severity_label = "HIGH"
    elif severity >= 40:
        severity_label = "MODERATE"
    else:
        severity_label = "LOW"

    return {
        "contradiction_type": contradiction_type,
        "contradiction_source": contradiction_source,
        "contradiction_severity": _safe_float(severity),
        "contradiction_severity_label": severity_label,
        "dominant_opposing_group": dominant_opposing_group,
    }


def _compare_trend(current: float, previous: float, threshold: float, up_label: str, down_label: str, flat_label: str) -> str:
    delta = current - previous
    if delta >= threshold:
        return up_label
    if delta <= -threshold:
        return down_label
    return flat_label


def _build_transition_summary(
    df: pd.DataFrame,
    cfg: MTFConfig,
    current_summary: Dict[str, Any],
) -> Dict[str, Any]:
    x = df.sort_index().copy()
    if len(x) < max(cfg.min_rows + 5, 50):
        return {
            "direction_trend": "stable",
            "alignment_trend": "stable",
            "conflict_trend": "stable",
            "state_transition": "insufficient_history",
            "previous_mtf_bias_label": "UNKNOWN",
            "previous_mtf_agreement_label": "UNKNOWN",
        }

    prev_df = x.iloc[:-1].copy()
    if len(prev_df) < cfg.min_rows:
        return {
            "direction_trend": "stable",
            "alignment_trend": "stable",
            "conflict_trend": "stable",
            "state_transition": "insufficient_history",
            "previous_mtf_bias_label": "UNKNOWN",
            "previous_mtf_agreement_label": "UNKNOWN",
        }

    prev_intel = build_mtf_intelligence(prev_df, cfg=cfg, include_transition=False)
    prev_summary = prev_intel["summary"]

    curr_dir = _safe_float(current_summary.get("mtf_weighted_direction", 0.0))
    prev_dir = _safe_float(prev_summary.get("mtf_weighted_direction", 0.0))

    curr_alignment = abs(_safe_float(current_summary.get("mtf_agreement", 0.0)))
    prev_alignment = abs(_safe_float(prev_summary.get("mtf_agreement", 0.0)))

    curr_conflict = _safe_float(current_summary.get("mtf_conflict_score", 0.0))
    prev_conflict = _safe_float(prev_summary.get("mtf_conflict_score", 0.0))

    direction_trend = _compare_trend(
        current=abs(curr_dir),
        previous=abs(prev_dir),
        threshold=cfg.direction_trend_delta_threshold,
        up_label="strengthening",
        down_label="weakening",
        flat_label="stable",
    )

    alignment_trend = _compare_trend(
        current=curr_alignment,
        previous=prev_alignment,
        threshold=cfg.alignment_trend_delta_threshold,
        up_label="improving",
        down_label="deteriorating",
        flat_label="stable",
    )

    conflict_trend = _compare_trend(
        current=curr_conflict,
        previous=prev_conflict,
        threshold=cfg.conflict_trend_delta_threshold,
        up_label="rising",
        down_label="falling",
        flat_label="stable",
    )

    curr_bias = str(current_summary.get("mtf_bias_label", "NEUTRAL"))
    prev_bias = str(prev_summary.get("mtf_bias_label", "NEUTRAL"))

    curr_agreement_label = str(current_summary.get("mtf_agreement_label", "NEUTRAL"))
    prev_agreement_label = str(prev_summary.get("mtf_agreement_label", "NEUTRAL"))

    if prev_bias != curr_bias and curr_bias != "NEUTRAL":
        state_transition = f"bias_shift_to_{curr_bias.lower()}"
    elif prev_agreement_label != curr_agreement_label:
        state_transition = f"agreement_shift_to_{curr_agreement_label.lower()}"
    elif direction_trend == "strengthening" and alignment_trend == "improving":
        state_transition = "trend_alignment_improving"
    elif direction_trend == "weakening" and conflict_trend == "rising":
        state_transition = "trend_alignment_deteriorating"
    elif curr_bias == "NEUTRAL" and curr_conflict >= cfg.strong_conflict_threshold:
        state_transition = "high_conflict_neutralization"
    else:
        state_transition = "stable_state"

    return {
        "direction_trend": direction_trend,
        "alignment_trend": alignment_trend,
        "conflict_trend": conflict_trend,
        "state_transition": state_transition,
        "previous_mtf_bias_label": prev_bias,
        "previous_mtf_agreement_label": prev_agreement_label,
    }


# =============================================================================
# CROSS-GROUP INTELLIGENCE ENGINE
# =============================================================================

def _build_cross_group_summary(
    macro: Dict[str, Any],
    structural: Dict[str, Any],
    execution: Dict[str, Any],
    calendar_ctx: Dict[str, Any],
    cfg: MTFConfig,
) -> Dict[str, Any]:
    group_dirs = [
        _safe_float(macro["weighted_direction"]),
        _safe_float(structural["weighted_direction"]),
        _safe_float(execution["weighted_direction"]),
    ]
    group_strengths = [
        _safe_float(macro["weighted_strength"]),
        _safe_float(structural["weighted_strength"]),
        _safe_float(execution["weighted_strength"]),
    ]
    group_confidences = [
        _safe_float(macro["group_confidence_score"]),
        _safe_float(structural["group_confidence_score"]),
        _safe_float(execution["group_confidence_score"]),
    ]
    group_weights = [
        cfg.group_weight_macro,
        cfg.group_weight_structural,
        cfg.group_weight_execution,
    ]

    weighted_direction = _weighted_average(group_dirs, group_weights)
    weighted_strength = _weighted_average(group_strengths, group_weights)
    weighted_confidence = _weighted_average(group_confidences, group_weights)

    signs = np.array([np.sign(v) for v in group_dirs], dtype=float)
    agreement = float(np.mean(signs)) if len(signs) else 0.0

    pos = np.mean(signs > 0) if len(signs) else 0.0
    neg = np.mean(signs < 0) if len(signs) else 0.0
    conflict = float(min(pos, neg) * 2.0)

    avg_strength_norm = _clip(weighted_strength / 100.0, 0.0, 1.0)
    agreement_norm = abs(agreement)
    confluence_score = _clip(
        (0.55 * agreement_norm) + (0.45 * avg_strength_norm) - (0.50 * conflict),
        0.0,
        1.0,
    )

    confidence_score = _clip(
        (0.50 * weighted_confidence)
        + (0.20 * abs(agreement) * 100.0)
        + (0.20 * weighted_strength)
        - (0.25 * conflict * 100.0),
        0.0,
        100.0,
    )

    macro_abs = abs(_safe_float(macro["weighted_direction"]))
    structural_abs = abs(_safe_float(structural["weighted_direction"]))
    execution_abs = abs(_safe_float(execution["weighted_direction"]))

    dominant_group = "macro"
    dominant_val = macro_abs
    if structural_abs > dominant_val:
        dominant_group = "structural"
        dominant_val = structural_abs
    if execution_abs > dominant_val:
        dominant_group = "execution"
        dominant_val = execution_abs

    cal_score = _safe_float(calendar_ctx.get("day_behavior_score", 0.0))
    regime_pressure = _clip((0.80 * weighted_direction) + (0.20 * cal_score), -100.0, 100.0)

    return {
        "mtf_weighted_direction": _safe_float(weighted_direction),
        "mtf_weighted_strength": _safe_float(weighted_strength),
        "mtf_confidence_score": _safe_float(confidence_score),
        "mtf_confidence_label": _confidence_label(confidence_score, cfg),
        "mtf_bias_label": _sign_label(weighted_direction),
        "mtf_agreement": _safe_float(agreement),
        "mtf_agreement_label": _agreement_label(
            agreement,
            strong_thr=cfg.strong_alignment_threshold,
            weak_thr=cfg.weak_alignment_threshold,
        ),
        "mtf_conflict_score": _safe_float(conflict),
        "mtf_conflict_label": _conflict_label(conflict, cfg.strong_conflict_threshold),
        "mtf_confluence_score": _safe_float(confluence_score * 100.0),
        "mtf_confluence_label": _state_from_score(confluence_score * 100.0),
        "dominant_timeframe_group": dominant_group,
        "macro_bias": str(macro["bias_label"]),
        "macro_strength": _safe_float(macro["weighted_strength"]),
        "macro_confidence": _safe_float(macro["group_confidence_score"]),
        "structural_alignment": str(structural["bias_label"]),
        "structural_strength": _safe_float(structural["weighted_strength"]),
        "structural_confidence": _safe_float(structural["group_confidence_score"]),
        "execution_alignment": str(execution["bias_label"]),
        "execution_strength": _safe_float(execution["weighted_strength"]),
        "execution_confidence": _safe_float(execution["group_confidence_score"]),
        "mtf_regime_pressure": _safe_float(regime_pressure),
        "mtf_regime_pressure_label": _sign_label(regime_pressure),
    }


# =============================================================================
# PUBLIC ENGINE
# =============================================================================

def build_mtf_intelligence(
    df: pd.DataFrame,
    cfg: Optional[MTFConfig] = None,
    include_transition: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or MTFConfig()

    _validate_ohlcv(df)

    x = df.sort_index().copy()
    if len(x) < cfg.min_rows:
        raise ValueError(f"MTF build error: need at least {cfg.min_rows} rows, got {len(x)}")

    macro_frames = _build_macro_frames(x, cfg)
    structural_frames = _build_structural_frames(x, cfg)
    execution_frames = _build_execution_frames(x, cfg)
    calendar_ctx = _build_calendar_context(x, cfg)

    macro_summary = _build_group_summary("macro", macro_frames, cfg)
    structural_summary = _build_group_summary("structural", structural_frames, cfg)
    execution_summary = _build_group_summary("execution", execution_frames, cfg)

    cross_group = _build_cross_group_summary(
        macro=macro_summary,
        structural=structural_summary,
        execution=execution_summary,
        calendar_ctx=calendar_ctx,
        cfg=cfg,
    )

    contradiction = _build_contradiction_map(
        macro=macro_summary,
        structural=structural_summary,
        execution=execution_summary,
        cfg=cfg,
    )

    if include_transition:
        transition = _build_transition_summary(
            df=x,
            cfg=cfg,
            current_summary=cross_group,
        )
    else:
        transition = {
            "direction_trend": "stable",
            "alignment_trend": "stable",
            "conflict_trend": "stable",
            "state_transition": "skipped",
            "previous_mtf_bias_label": "UNKNOWN",
            "previous_mtf_agreement_label": "UNKNOWN",
        }

    latest_ts = x.index[-1]

    return {
        "timestamp": latest_ts.isoformat(),
        "config": asdict(cfg),
        "macro": macro_summary,
        "structural": structural_summary,
        "execution": execution_summary,
        "calendar": calendar_ctx,
        "summary": cross_group,
        "contradiction": contradiction,
        "transition": transition,
    }


# =============================================================================
# FEATURE EXPORT FOR FORECASTER / HISTORY DATASET
# =============================================================================

def build_mtf_feature_row(
    df: pd.DataFrame,
    cfg: Optional[MTFConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MTFConfig()
    intel = build_mtf_intelligence(df=df, cfg=cfg, include_transition=True)

    summary = intel["summary"]
    macro = intel["macro"]
    structural = intel["structural"]
    execution = intel["execution"]
    calendar_ctx = intel["calendar"]
    contradiction = intel["contradiction"]
    transition = intel["transition"]

    direction = _safe_float(summary["mtf_weighted_direction"])
    bias_signal = _sign_int(direction)

    row: Dict[str, Any] = {
        "timestamp": intel["timestamp"],

        # Core classes
        "mtf_bias_signal": bias_signal,
        "mtf_bias_label": str(summary["mtf_bias_label"]),
        "mtf_direction_value": direction,
        "mtf_strength_value": _safe_float(summary["mtf_weighted_strength"]),
        "mtf_confidence_score": _safe_float(summary["mtf_confidence_score"]),
        "mtf_confidence_label": str(summary["mtf_confidence_label"]),
        "mtf_agreement_value": _safe_float(summary["mtf_agreement"]),
        "mtf_agreement_label": str(summary["mtf_agreement_label"]),
        "mtf_conflict_score": _safe_float(summary["mtf_conflict_score"]),
        "mtf_conflict_label": str(summary["mtf_conflict_label"]),
        "mtf_confluence_score": _safe_float(summary["mtf_confluence_score"]),
        "mtf_confluence_label": str(summary["mtf_confluence_label"]),
        "mtf_regime_pressure": _safe_float(summary["mtf_regime_pressure"]),
        "mtf_regime_pressure_label": str(summary["mtf_regime_pressure_label"]),
        "dominant_timeframe_group": str(summary["dominant_timeframe_group"]),

        # Group direction
        "macro_bias": str(summary["macro_bias"]),
        "macro_strength": _safe_float(summary["macro_strength"]),
        "macro_confidence": _safe_float(summary["macro_confidence"]),
        "structural_alignment": str(summary["structural_alignment"]),
        "structural_strength": _safe_float(summary["structural_strength"]),
        "structural_confidence": _safe_float(summary["structural_confidence"]),
        "execution_alignment": str(summary["execution_alignment"]),
        "execution_strength": _safe_float(summary["execution_strength"]),
        "execution_confidence": _safe_float(summary["execution_confidence"]),

        # Calendar
        "day_of_week": str(calendar_ctx["day_of_week"]),
        "calendar_bias": str(calendar_ctx["calendar_bias"]),
        "calendar_strength": _safe_float(calendar_ctx["calendar_strength"]),
        "calendar_context": str(calendar_ctx["calendar_context"]),
        "day_behavior_score": _safe_float(calendar_ctx["day_behavior_score"]),

        # Contradiction
        "contradiction_type": str(contradiction["contradiction_type"]),
        "contradiction_source": str(contradiction["contradiction_source"]),
        "contradiction_severity": _safe_float(contradiction["contradiction_severity"]),
        "contradiction_severity_label": str(contradiction["contradiction_severity_label"]),
        "dominant_opposing_group": str(contradiction["dominant_opposing_group"]),

        # Transition
        "direction_trend": str(transition["direction_trend"]),
        "alignment_trend": str(transition["alignment_trend"]),
        "conflict_trend": str(transition["conflict_trend"]),
        "state_transition": str(transition["state_transition"]),
        "previous_mtf_bias_label": str(transition["previous_mtf_bias_label"]),
        "previous_mtf_agreement_label": str(transition["previous_mtf_agreement_label"]),
    }

    # Add compact per-frame feature columns for research
    for group_key in ["macro", "structural", "execution"]:
        group_block = intel[group_key]
        frames = group_block.get("frames", []) or []

        for frame in frames:
            frame_name = str(frame.get("frame", "UNK")).replace(" ", "_")
            prefix = f"{group_key}_{frame_name}"

            row[f"{prefix}_valid"] = 1 if bool(frame.get("valid", False)) else 0
            row[f"{prefix}_direction"] = _safe_float(frame.get("direction", 0.0))
            row[f"{prefix}_strength"] = _safe_float(frame.get("strength", 0.0))
            row[f"{prefix}_confidence"] = _safe_float(frame.get("confidence_score", 0.0))
            row[f"{prefix}_trend_state"] = str(frame.get("trend_state", "unknown"))
            row[f"{prefix}_regime_state"] = str(frame.get("regime_state", "unknown"))
            row[f"{prefix}_direction_sign"] = _safe_int(frame.get("direction_sign", 0))

    return row


# =============================================================================
# WEBSITE / FORECASTER / DASHBOARD CONTRACT
# =============================================================================

def build_mtf_latest_payload(
    df: pd.DataFrame,
    cfg: Optional[MTFConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MTFConfig()
    intel = build_mtf_intelligence(df=df, cfg=cfg, include_transition=True)

    summary = intel["summary"]
    macro = intel["macro"]
    structural = intel["structural"]
    execution = intel["execution"]
    calendar_ctx = intel["calendar"]
    contradiction = intel["contradiction"]
    transition = intel["transition"]

    bias_signal = _sign_int(summary["mtf_weighted_direction"])

    payload: Dict[str, Any] = {
        # Shared website contract
        "debug_version": cfg.debug_version,
        "timestamp": intel["timestamp"],
        "state": str(summary["mtf_agreement_label"]),
        "bias_signal": bias_signal,
        "bias_label": str(summary["mtf_bias_label"]),
        "indicator_strength": _safe_float(summary["mtf_weighted_strength"]),
        "market_bias": str(summary["mtf_bias_label"]),

        # Core MTF summary
        "mtf_weighted_direction": _safe_float(summary["mtf_weighted_direction"]),
        "mtf_weighted_strength": _safe_float(summary["mtf_weighted_strength"]),
        "mtf_confidence_score": _safe_float(summary["mtf_confidence_score"]),
        "mtf_confidence_label": str(summary["mtf_confidence_label"]),
        "mtf_agreement": _safe_float(summary["mtf_agreement"]),
        "mtf_agreement_label": str(summary["mtf_agreement_label"]),
        "mtf_conflict_score": _safe_float(summary["mtf_conflict_score"]),
        "mtf_conflict_label": str(summary["mtf_conflict_label"]),
        "mtf_confluence_score": _safe_float(summary["mtf_confluence_score"]),
        "mtf_confluence_label": str(summary["mtf_confluence_label"]),
        "dominant_timeframe_group": str(summary["dominant_timeframe_group"]),
        "mtf_regime_pressure": _safe_float(summary["mtf_regime_pressure"]),
        "mtf_regime_pressure_label": str(summary["mtf_regime_pressure_label"]),

        # Group outputs for Forecaster + Ultimate Truth Dashboard
        "macro_bias": str(summary["macro_bias"]),
        "macro_strength": _safe_float(summary["macro_strength"]),
        "macro_confidence": _safe_float(summary["macro_confidence"]),
        "structural_alignment": str(summary["structural_alignment"]),
        "structural_strength": _safe_float(summary["structural_strength"]),
        "structural_confidence": _safe_float(summary["structural_confidence"]),
        "execution_alignment": str(summary["execution_alignment"]),
        "execution_strength": _safe_float(summary["execution_strength"]),
        "execution_confidence": _safe_float(summary["execution_confidence"]),

        # Calendar overlay
        "day_of_week": str(calendar_ctx["day_of_week"]),
        "calendar_bias": str(calendar_ctx["calendar_bias"]),
        "calendar_strength": _safe_float(calendar_ctx["calendar_strength"]),
        "calendar_context": str(calendar_ctx["calendar_context"]),
        "day_behavior_score": _safe_float(calendar_ctx["day_behavior_score"]),

        # Contradiction
        "contradiction_type": str(contradiction["contradiction_type"]),
        "contradiction_source": str(contradiction["contradiction_source"]),
        "contradiction_severity": _safe_float(contradiction["contradiction_severity"]),
        "contradiction_severity_label": str(contradiction["contradiction_severity_label"]),
        "dominant_opposing_group": str(contradiction["dominant_opposing_group"]),

        # Transition
        "direction_trend": str(transition["direction_trend"]),
        "alignment_trend": str(transition["alignment_trend"]),
        "conflict_trend": str(transition["conflict_trend"]),
        "state_transition": str(transition["state_transition"]),
        "previous_mtf_bias_label": str(transition["previous_mtf_bias_label"]),
        "previous_mtf_agreement_label": str(transition["previous_mtf_agreement_label"]),

        # Compact dashboard labels
        "macro_agreement_label": str(macro["agreement_label"]),
        "structural_agreement_label": str(structural["agreement_label"]),
        "execution_agreement_label": str(execution["agreement_label"]),
        "macro_conflict_label": str(macro["conflict_label"]),
        "structural_conflict_label": str(structural["conflict_label"]),
        "execution_conflict_label": str(execution["conflict_label"]),
        "macro_confidence_label": str(macro["group_confidence_label"]),
        "structural_confidence_label": str(structural["group_confidence_label"]),
        "execution_confidence_label": str(execution["group_confidence_label"]),

        # Deep detail
        "macro": macro,
        "structural": structural,
        "execution": execution,
        "calendar": calendar_ctx,
        "summary": summary,
        "contradiction": contradiction,
        "transition": transition,
        "feature_row": build_mtf_feature_row(df=df, cfg=cfg),
    }

    return payload