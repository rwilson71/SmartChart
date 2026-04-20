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

    # Macro anchors
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

    # Website contract fields
    debug_version: str = "y_mtf_v1"


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


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sign_label(v: float, pos_label: str = "BULLISH", neg_label: str = "BEARISH", neutral_label: str = "NEUTRAL") -> str:
    if v > 0:
        return pos_label
    if v < 0:
        return neg_label
    return neutral_label


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
        "M": "1ME",   # month-end
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


# =============================================================================
# PER-FRAME FEATURE EXTRACTION
# =============================================================================

def _compute_frame_features(
    df: pd.DataFrame,
    cfg: MTFConfig,
    frame_name: str,
    frame_group: str,
) -> Dict[str, Any]:
    if len(df) < max(cfg.ema_signal + 5, cfg.atr_len + 5, cfg.slope_lookback + 5):
        return {
            "frame": frame_name,
            "group": frame_group,
            "rows": int(len(df)),
            "valid": False,
            "direction": 0.0,
            "direction_label": "NEUTRAL",
            "strength": 0.0,
            "strength_label": "neutral",
            "trend_state": "insufficient_data",
            "regime_state": "unknown",
            "ema_fast": np.nan,
            "ema_slow": np.nan,
            "ema_signal": np.nan,
            "close": _safe_float(df["close"].iloc[-1]) if len(df) else 0.0,
            "atr": 0.0,
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

    return {
        "frame": frame_name,
        "group": frame_group,
        "rows": int(len(x)),
        "valid": True,
        "direction": _safe_float(raw_direction),
        "direction_label": _sign_label(raw_direction),
        "strength": _safe_float(strength),
        "strength_label": _state_from_score(strength),
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
    }


# =============================================================================
# WEIGHTING / AGREEMENT / CONFLICT
# =============================================================================

def _weighted_average(values: List[float], weights: List[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0

    weight_sum = sum(abs(w) for w in weights)
    if weight_sum <= 0:
        return 0.0

    return float(sum(v * w for v, w in zip(values, weights)) / weight_sum)


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
    weights = [_frame_weight(str(f["frame"]), cfg) for f in valid]

    weighted_direction = _weighted_average(directions, weights)
    weighted_strength = _weighted_average(strengths, weights)
    agreement = _agreement_score(valid)
    conflict = _conflict_score(valid)

    return {
        "group": group_name,
        "valid_count": len(valid),
        "weighted_direction": _safe_float(weighted_direction),
        "weighted_strength": _safe_float(weighted_strength),
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
# MACRO / STRUCTURAL / EXECUTION BUILDERS
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
    group_weights = [
        cfg.group_weight_macro,
        cfg.group_weight_structural,
        cfg.group_weight_execution,
    ]

    weighted_direction = _weighted_average(group_dirs, group_weights)
    weighted_strength = _weighted_average(group_strengths, group_weights)

    # Agreement across group signs
    signs = np.array([np.sign(v) for v in group_dirs], dtype=float)
    agreement = float(np.mean(signs)) if len(signs) else 0.0

    pos = np.mean(signs > 0) if len(signs) else 0.0
    neg = np.mean(signs < 0) if len(signs) else 0.0
    conflict = float(min(pos, neg) * 2.0)

    # Confluence prefers agreement + strength, penalizes conflict
    avg_strength_norm = _clip(weighted_strength / 100.0, 0.0, 1.0)
    agreement_norm = abs(agreement)
    confluence_score = _clip((0.55 * agreement_norm) + (0.45 * avg_strength_norm) - (0.50 * conflict), 0.0, 1.0)

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

    # Calendar nudge as contextual overlay only, not core direction override
    cal_score = _safe_float(calendar_ctx.get("day_behavior_score", 0.0))
    regime_pressure = _clip((0.80 * weighted_direction) + (0.20 * cal_score), -100.0, 100.0)

    return {
        "mtf_weighted_direction": _safe_float(weighted_direction),
        "mtf_weighted_strength": _safe_float(weighted_strength),
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
        "structural_alignment": str(structural["bias_label"]),
        "structural_strength": _safe_float(structural["weighted_strength"]),
        "execution_alignment": str(execution["bias_label"]),
        "execution_strength": _safe_float(execution["weighted_strength"]),
        "mtf_regime_pressure": _safe_float(regime_pressure),
        "mtf_regime_pressure_label": _sign_label(regime_pressure),
    }


# =============================================================================
# PUBLIC ENGINE
# =============================================================================

def build_mtf_intelligence(
    df: pd.DataFrame,
    cfg: Optional[MTFConfig] = None,
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

    latest_ts = x.index[-1]

    return {
        "timestamp": latest_ts.isoformat(),
        "config": asdict(cfg),
        "macro": macro_summary,
        "structural": structural_summary,
        "execution": execution_summary,
        "calendar": calendar_ctx,
        "summary": cross_group,
    }


# =============================================================================
# WEBSITE / FORECASTER / DASHBOARD CONTRACT
# =============================================================================

def build_mtf_latest_payload(
    df: pd.DataFrame,
    cfg: Optional[MTFConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MTFConfig()
    intel = build_mtf_intelligence(df=df, cfg=cfg)

    summary = intel["summary"]
    macro = intel["macro"]
    structural = intel["structural"]
    execution = intel["execution"]
    calendar_ctx = intel["calendar"]

    bias_signal = 0
    if summary["mtf_weighted_direction"] > 0:
        bias_signal = 1
    elif summary["mtf_weighted_direction"] < 0:
        bias_signal = -1

    payload: Dict[str, Any] = {
        # Shared website contract
        "debug_version": cfg.debug_version,
        "timestamp": intel["timestamp"],
        "state": summary["mtf_agreement_label"],
        "bias_signal": bias_signal,
        "bias_label": summary["mtf_bias_label"],
        "indicator_strength": _safe_float(summary["mtf_weighted_strength"]),
        "market_bias": summary["mtf_bias_label"],

        # Core MTF summary
        "mtf_weighted_direction": _safe_float(summary["mtf_weighted_direction"]),
        "mtf_weighted_strength": _safe_float(summary["mtf_weighted_strength"]),
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
        "structural_alignment": str(summary["structural_alignment"]),
        "structural_strength": _safe_float(summary["structural_strength"]),
        "execution_alignment": str(summary["execution_alignment"]),
        "execution_strength": _safe_float(summary["execution_strength"]),

        # Calendar overlay
        "day_of_week": str(calendar_ctx["day_of_week"]),
        "calendar_bias": str(calendar_ctx["calendar_bias"]),
        "calendar_strength": _safe_float(calendar_ctx["calendar_strength"]),
        "calendar_context": str(calendar_ctx["calendar_context"]),
        "day_behavior_score": _safe_float(calendar_ctx["day_behavior_score"]),

        # Compact dashboard labels
        "macro_agreement_label": str(macro["agreement_label"]),
        "structural_agreement_label": str(structural["agreement_label"]),
        "execution_agreement_label": str(execution["agreement_label"]),
        "macro_conflict_label": str(macro["conflict_label"]),
        "structural_conflict_label": str(structural["conflict_label"]),
        "execution_conflict_label": str(execution["conflict_label"]),

        # Deep detail for consumers/debug
        "macro": macro,
        "structural": structural,
        "execution": execution,
        "calendar": calendar_ctx,
        "summary": summary,
    }

    return payload