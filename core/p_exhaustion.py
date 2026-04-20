from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ExhaustionConfig:
    # Core exhaustion engine
    ema_len: int = 20
    stretch_len: int = 20
    push_len: int = 5
    small_body_thresh: float = 0.35
    strong_body_thresh: float = 0.60
    stretch_mult: float = 1.50
    extreme_stretch_mult: float = 2.20
    decay_thresh: float = 0.55

    # Memory
    confirm_bars: int = 2
    hold_bars: int = 3
    bias_refresh_bars: int = 1

    # Delta / footprint
    delta_body_weight: float = 0.70
    delta_close_location_weight: float = 0.30
    delta_range_norm_floor: float = 1e-9

    # Range-delta / top-bottom context
    top_bottom_lookback: int = 12
    range_delta_len: int = 8
    absorption_body_frac_max: float = 0.45
    opposing_control_min: float = 20.0
    weakening_participation_threshold: float = 0.85

    # Signal shaping
    dashboard_strength_extreme: float = 1.00
    dashboard_strength_normal: float = 0.65
    dashboard_strength_neutral: float = 0.00


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()


def _clamp(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


def _safe_float(value: Any, default: float = 0.0) -> float:
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
        if value is None:
            return default
        if pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def _bars_since_change(series: pd.Series) -> pd.Series:
    """
    Stable count tracker:
        0 on the changed bar
        +1 each unchanged bar after that
    """
    out = np.zeros(len(series), dtype=int)
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index)

    vals = series.to_numpy()
    out[0] = 0

    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            out[i] = 0
        else:
            out[i] = out[i - 1] + 1

    return pd.Series(out, index=series.index, dtype=int)


# =============================================================================
# DELTA / FOOTPRINT HELPERS
# =============================================================================

def _compute_single_candle_delta(
    base: pd.DataFrame,
    body_frac: pd.Series,
    cfg: ExhaustionConfig,
) -> pd.DataFrame:
    """
    Proxy delta / footprint model using candle structure only.

    Positive values:
        buyer control
    Negative values:
        seller control
    Magnitude:
        dominance / strength
    """
    close_pos = pd.Series(
        np.where(
            (base["high"] - base["low"]) != 0,
            (base["close"] - base["low"]) / (base["high"] - base["low"]),
            0.5,
        ),
        index=base.index,
        dtype=float,
    )
    close_pos = _clamp(close_pos, 0.0, 1.0)

    close_location_signed = (close_pos - 0.5) * 2.0
    candle_dir = np.sign(base["close"] - base["open"]).astype(float)

    body_component = candle_dir * body_frac * 100.0
    close_component = close_location_signed * 100.0

    single_candle_delta = (
        body_component * cfg.delta_body_weight
        + close_component * cfg.delta_close_location_weight
    )
    single_candle_delta = _clamp(single_candle_delta, -100.0, 100.0)

    bull_control_score = _clamp(single_candle_delta, 0.0, 100.0)
    bear_control_score = _clamp(-single_candle_delta, 0.0, 100.0)
    control_strength = single_candle_delta.abs()

    control_winner = np.where(
        single_candle_delta > 0,
        "BULL",
        np.where(single_candle_delta < 0, "BEAR", "NEUTRAL"),
    )

    out = pd.DataFrame(index=base.index)
    out["close_pos"] = close_pos
    out["close_location_signed"] = close_location_signed
    out["single_candle_delta"] = single_candle_delta.astype(float)
    out["bull_control_score"] = bull_control_score.astype(float)
    out["bear_control_score"] = bear_control_score.astype(float)
    out["control_strength"] = control_strength.astype(float)
    out["control_winner_text"] = control_winner

    return out


def _compute_top_bottom_context(
    base: pd.DataFrame,
    cfg: ExhaustionConfig,
) -> pd.DataFrame:
    """
    Simple local-extreme context:
    - TOP when current high is at/near rolling high
    - BOTTOM when current low is at/near rolling low
    """
    roll_high = base["high"].rolling(cfg.top_bottom_lookback, min_periods=1).max()
    roll_low = base["low"].rolling(cfg.top_bottom_lookback, min_periods=1).min()

    is_top_context = base["high"] >= roll_high
    is_bottom_context = base["low"] <= roll_low

    top_bottom_context = np.where(
        is_top_context & ~is_bottom_context,
        "TOP",
        np.where(
            is_bottom_context & ~is_top_context,
            "BOTTOM",
            np.where(is_top_context & is_bottom_context, "RANGE_EDGE", "NONE"),
        ),
    )

    out = pd.DataFrame(index=base.index)
    out["rolling_high"] = roll_high
    out["rolling_low"] = roll_low
    out["is_top_context"] = is_top_context.astype(int)
    out["is_bottom_context"] = is_bottom_context.astype(int)
    out["top_bottom_context"] = top_bottom_context

    return out


def _compute_range_delta_and_absorption(
    base: pd.DataFrame,
    body_frac: pd.Series,
    single_candle_delta: pd.Series,
    top_bottom_context: pd.Series,
    cfg: ExhaustionConfig,
) -> pd.DataFrame:
    """
    Range delta = rolling sum / mean of single candle delta.
    Used with top / bottom context to detect:
    - absorption
    - aggressive continuation
    - weakening participation
    - opposing control in the range
    """
    range_delta_sum = single_candle_delta.rolling(cfg.range_delta_len, min_periods=1).sum()
    range_delta_avg = single_candle_delta.rolling(cfg.range_delta_len, min_periods=1).mean()
    range_delta_abs_avg = single_candle_delta.abs().rolling(cfg.range_delta_len, min_periods=1).mean()

    prior_delta_abs_avg = range_delta_abs_avg.shift(1).fillna(range_delta_abs_avg)
    weakening_participation = range_delta_abs_avg <= (
        prior_delta_abs_avg * cfg.weakening_participation_threshold
    )

    # Opposing control at a top:
    # price is at top context, but single candle delta or range delta is bearish enough
    bear_absorption_signal = (
        (top_bottom_context == "TOP")
        & (body_frac <= cfg.absorption_body_frac_max)
        & (
            (single_candle_delta <= -cfg.opposing_control_min)
            | (range_delta_avg <= -cfg.opposing_control_min)
        )
    )

    # Opposing control at a bottom:
    # price is at bottom context, but single candle delta or range delta is bullish enough
    bull_absorption_signal = (
        (top_bottom_context == "BOTTOM")
        & (body_frac <= cfg.absorption_body_frac_max)
        & (
            (single_candle_delta >= cfg.opposing_control_min)
            | (range_delta_avg >= cfg.opposing_control_min)
        )
    )

    aggressive_bull_continuation = (
        (top_bottom_context == "TOP")
        & (single_candle_delta > 0)
        & (range_delta_avg > 0)
        & (~bear_absorption_signal)
    )

    aggressive_bear_continuation = (
        (top_bottom_context == "BOTTOM")
        & (single_candle_delta < 0)
        & (range_delta_avg < 0)
        & (~bull_absorption_signal)
    )

    reversal_pressure = np.where(
        bear_absorption_signal,
        -1,
        np.where(bull_absorption_signal, 1, 0),
    )

    range_delta_bias = np.where(
        range_delta_avg > 0,
        1,
        np.where(range_delta_avg < 0, -1, 0),
    )

    range_delta_bias_text = np.where(
        range_delta_bias == 1,
        "BULL",
        np.where(range_delta_bias == -1, "BEAR", "NEUTRAL"),
    )

    out = pd.DataFrame(index=base.index)
    out["range_delta_sum"] = range_delta_sum.astype(float)
    out["range_delta_avg"] = range_delta_avg.astype(float)
    out["range_delta_abs_avg"] = range_delta_abs_avg.astype(float)
    out["weakening_participation"] = weakening_participation.astype(int)
    out["bull_absorption_signal"] = bull_absorption_signal.astype(int)
    out["bear_absorption_signal"] = bear_absorption_signal.astype(int)
    out["aggressive_bull_continuation"] = aggressive_bull_continuation.astype(int)
    out["aggressive_bear_continuation"] = aggressive_bear_continuation.astype(int)
    out["reversal_pressure"] = reversal_pressure.astype(int)
    out["range_delta_bias"] = pd.Series(range_delta_bias, index=base.index, dtype=int)
    out["range_delta_bias_text"] = pd.Series(range_delta_bias_text, index=base.index, dtype=object)

    return out


def _build_exhaustion_quality(
    exh_state: pd.Series,
    stretch_ratio: pd.Series,
    control_strength: pd.Series,
    weakening_participation: pd.Series,
    bull_absorption_signal: pd.Series,
    bear_absorption_signal: pd.Series,
) -> pd.Series:
    """
    Trigger quality score for dashboard / API use.
    """
    state_weight = exh_state.abs().astype(float) / 2.0
    stretch_weight = _clamp(stretch_ratio / 3.0, 0.0, 1.0)
    control_weight = _clamp(control_strength / 100.0, 0.0, 1.0)
    weakening_weight = weakening_participation.astype(float) * 0.20
    absorption_weight = (
        bull_absorption_signal.astype(float) + bear_absorption_signal.astype(float)
    ) * 0.20

    quality = (
        state_weight * 0.35
        + stretch_weight * 0.25
        + control_weight * 0.20
        + weakening_weight
        + absorption_weight
    ) * 100.0

    return _clamp(quality, 0.0, 100.0)


# =============================================================================
# CORE ENGINE
# =============================================================================

def calculate_exhaustion(
    df: pd.DataFrame,
    config: Optional[ExhaustionConfig] = None,
) -> pd.DataFrame:
    """
    SmartChart Exhaustion Module v2
    Production rebuild with:
    - original exhaustion engine
    - single-candle delta footprint
    - range delta
    - absorption logic
    - top/bottom context
    - dashboard / website export fields
    """
    cfg = config or ExhaustionConfig()
    _validate_ohlc(df)

    base = df.copy().sort_index()
    out = pd.DataFrame(index=base.index)

    # -------------------------------------------------------------------------
    # Core series
    # -------------------------------------------------------------------------
    ema_base = _ema(base["close"], cfg.ema_len)

    bar_range = base["high"] - base["low"]
    body_size = (base["close"] - base["open"]).abs()
    body_frac0 = pd.Series(
        np.where(bar_range != 0, body_size / bar_range, 0.0),
        index=base.index,
        dtype=float,
    )
    body_frac = _clamp(body_frac0, 0.0, 1.0)

    is_bull_bar = base["close"] > base["open"]
    is_bear_bar = base["close"] < base["open"]
    is_small_body = body_frac <= cfg.small_body_thresh
    is_strong_body = body_frac >= cfg.strong_body_thresh

    dist_from_ema = base["close"] - ema_base
    abs_dist_from_ema = dist_from_ema.abs()

    stretch_base = _sma(abs_dist_from_ema, cfg.stretch_len)
    stretch_ratio0 = pd.Series(
        np.where(stretch_base != 0, abs_dist_from_ema / stretch_base, 1.0),
        index=base.index,
        dtype=float,
    )
    stretch_ratio = _clamp(stretch_ratio0, 0.0, 10.0)

    bull_push_raw = pd.Series(
        np.where(is_bull_bar, body_frac, 0.0),
        index=base.index,
        dtype=float,
    )
    bear_push_raw = pd.Series(
        np.where(is_bear_bar, body_frac, 0.0),
        index=base.index,
        dtype=float,
    )

    bull_push = _sma(bull_push_raw, cfg.push_len) * cfg.push_len
    bear_push = _sma(bear_push_raw, cfg.push_len) * cfg.push_len

    bull_bars = _sma(is_bull_bar.astype(float), cfg.push_len) * cfg.push_len
    bear_bars = _sma(is_bear_bar.astype(float), cfg.push_len) * cfg.push_len

    bull_dominant = (bull_bars > bear_bars) & (bull_push > bear_push)
    bear_dominant = (bear_bars > bull_bars) & (bear_push > bull_push)

    recent_strong_bull = _sma((is_bull_bar & is_strong_body).astype(float), cfg.push_len) * cfg.push_len
    recent_strong_bear = _sma((is_bear_bar & is_strong_body).astype(float), cfg.push_len) * cfg.push_len
    recent_small_bull = _sma((is_bull_bar & is_small_body).astype(float), cfg.push_len) * cfg.push_len
    recent_small_bear = _sma((is_bear_bar & is_small_body).astype(float), cfg.push_len) * cfg.push_len

    # -------------------------------------------------------------------------
    # Decay / fatigue logic
    # -------------------------------------------------------------------------
    bull_decay = (
        bull_dominant
        & (dist_from_ema > 0)
        & (stretch_ratio >= cfg.stretch_mult)
        & (recent_small_bull >= recent_strong_bull)
        & (body_frac <= cfg.decay_thresh)
    )

    bear_decay = (
        bear_dominant
        & (dist_from_ema < 0)
        & (stretch_ratio >= cfg.stretch_mult)
        & (recent_small_bear >= recent_strong_bear)
        & (body_frac <= cfg.decay_thresh)
    )

    bull_extreme = bull_decay & (stretch_ratio >= cfg.extreme_stretch_mult)
    bear_extreme = bear_decay & (stretch_ratio >= cfg.extreme_stretch_mult)

    raw_exh_state = pd.Series(
        np.select(
            [
                bull_extreme,
                bull_decay,
                bear_extreme,
                bear_decay,
            ],
            [
                2,
                1,
                -2,
                -1,
            ],
            default=0,
        ),
        index=base.index,
        dtype=int,
    )

    raw_exh_bias = pd.Series(
        np.where(raw_exh_state > 0, 1, np.where(raw_exh_state < 0, -1, 0)),
        index=base.index,
        dtype=int,
    )

    # -------------------------------------------------------------------------
    # Memory
    # -------------------------------------------------------------------------
    raw_stable_count = _bars_since_change(raw_exh_state)
    raw_stable_bars = raw_stable_count + 1
    state_ready = raw_stable_bars >= cfg.confirm_bars

    bias_stable_count = _bars_since_change(raw_exh_bias)
    bias_stable_bars = bias_stable_count + 1
    bias_ready = bias_stable_bars >= cfg.bias_refresh_bars

    exh_state = np.zeros(len(base), dtype=int)
    exh_bias = np.zeros(len(base), dtype=int)
    exh_age = np.zeros(len(base), dtype=int)

    if len(base) > 0:
        exh_state[0] = int(raw_exh_state.iloc[0])
        exh_bias[0] = int(raw_exh_bias.iloc[0])
        exh_age[0] = 0

    for i in range(1, len(base)):
        can_flip = exh_age[i - 1] >= cfg.hold_bars
        allow_state_flip = (
            bool(state_ready.iloc[i])
            and int(raw_exh_state.iloc[i]) != int(exh_state[i - 1])
            and can_flip
        )
        allow_bias_refresh = (
            bool(bias_ready.iloc[i])
            and int(raw_exh_bias.iloc[i]) != 0
            and int(raw_exh_bias.iloc[i]) != int(exh_bias[i - 1])
        )

        if allow_state_flip:
            exh_state[i] = int(raw_exh_state.iloc[i])
            exh_bias[i] = int(raw_exh_bias.iloc[i])
            exh_age[i] = 0
        else:
            exh_state[i] = exh_state[i - 1]
            exh_bias[i] = exh_bias[i - 1]
            exh_age[i] = exh_age[i - 1] + 1
            if allow_bias_refresh:
                exh_bias[i] = int(raw_exh_bias.iloc[i])

    # -------------------------------------------------------------------------
    # Delta / footprint
    # -------------------------------------------------------------------------
    delta_block = _compute_single_candle_delta(base=base, body_frac=body_frac, cfg=cfg)

    # -------------------------------------------------------------------------
    # Top / bottom context
    # -------------------------------------------------------------------------
    context_block = _compute_top_bottom_context(base=base, cfg=cfg)

    # -------------------------------------------------------------------------
    # Range delta / absorption
    # -------------------------------------------------------------------------
    range_delta_block = _compute_range_delta_and_absorption(
        base=base,
        body_frac=body_frac,
        single_candle_delta=delta_block["single_candle_delta"],
        top_bottom_context=context_block["top_bottom_context"],
        cfg=cfg,
    )

    # -------------------------------------------------------------------------
    # Exhaustion quality / trigger quality
    # -------------------------------------------------------------------------
    trigger_quality = _build_exhaustion_quality(
        exh_state=pd.Series(exh_state, index=base.index, dtype=int),
        stretch_ratio=stretch_ratio,
        control_strength=delta_block["control_strength"],
        weakening_participation=range_delta_block["weakening_participation"],
        bull_absorption_signal=range_delta_block["bull_absorption_signal"],
        bear_absorption_signal=range_delta_block["bear_absorption_signal"],
    )

    # -------------------------------------------------------------------------
    # Base outputs
    # -------------------------------------------------------------------------
    out["ema_base"] = ema_base
    out["bar_range"] = bar_range
    out["body_size"] = body_size
    out["body_frac"] = body_frac
    out["is_bull_bar"] = is_bull_bar.astype(int)
    out["is_bear_bar"] = is_bear_bar.astype(int)
    out["is_small_body"] = is_small_body.astype(int)
    out["is_strong_body"] = is_strong_body.astype(int)

    out["dist_from_ema"] = dist_from_ema
    out["abs_dist_from_ema"] = abs_dist_from_ema
    out["stretch_base"] = stretch_base
    out["stretch_ratio"] = stretch_ratio

    out["bull_push"] = bull_push
    out["bear_push"] = bear_push
    out["bull_bars"] = bull_bars
    out["bear_bars"] = bear_bars
    out["bull_dominant"] = bull_dominant.astype(int)
    out["bear_dominant"] = bear_dominant.astype(int)

    out["recent_strong_bull"] = recent_strong_bull
    out["recent_strong_bear"] = recent_strong_bear
    out["recent_small_bull"] = recent_small_bull
    out["recent_small_bear"] = recent_small_bear

    out["bull_decay"] = bull_decay.astype(int)
    out["bear_decay"] = bear_decay.astype(int)
    out["bull_extreme"] = bull_extreme.astype(int)
    out["bear_extreme"] = bear_extreme.astype(int)

    out["raw_exh_state"] = raw_exh_state
    out["raw_exh_bias"] = raw_exh_bias
    out["raw_stable_bars"] = raw_stable_bars.astype(int)
    out["bias_stable_bars"] = bias_stable_bars.astype(int)

    out["exh_state"] = pd.Series(exh_state, index=base.index, dtype=int)
    out["exh_bias"] = pd.Series(exh_bias, index=base.index, dtype=int)
    out["exh_age"] = pd.Series(exh_age, index=base.index, dtype=int)

    # Delta / context / range-delta blocks
    for col in delta_block.columns:
        out[col] = delta_block[col]

    for col in context_block.columns:
        out[col] = context_block[col]

    for col in range_delta_block.columns:
        out[col] = range_delta_block[col]

    out["trigger_quality"] = trigger_quality.astype(float)

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------
    out["exh_changed"] = out["exh_state"].ne(out["exh_state"].shift(1)).fillna(False).astype(int)
    out["bull_exh_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 1)).astype(int)
    out["bull_extreme_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 2)).astype(int)
    out["bear_exh_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -1)).astype(int)
    out["bear_extreme_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -2)).astype(int)

    # -------------------------------------------------------------------------
    # Text mapping
    # -------------------------------------------------------------------------
    out["exh_text"] = np.select(
        [
            out["exh_state"] == 2,
            out["exh_state"] == 1,
            out["exh_state"] == -1,
            out["exh_state"] == -2,
        ],
        [
            "BULL EXTREME",
            "BULL EXHAUSTION",
            "BEAR EXHAUSTION",
            "BEAR EXTREME",
        ],
        default="NEUTRAL",
    )

    out["bias_text"] = np.select(
        [
            out["exh_bias"] == 1,
            out["exh_bias"] == -1,
        ],
        [
            "BULL",
            "BEAR",
        ],
        default="NEUTRAL",
    )

    out["top_bottom_text"] = out["top_bottom_context"].astype(str)

    # -------------------------------------------------------------------------
    # Website / dashboard-facing normalized fields
    # -------------------------------------------------------------------------
    out["direction"] = np.where(
        out["exh_state"] > 0,
        1,
        np.where(out["exh_state"] < 0, -1, 0),
    ).astype(int)

    out["signal_on"] = (out["exh_state"] != 0).astype(int)

    out["strength"] = np.where(
        out["exh_state"].abs() == 2,
        cfg.dashboard_strength_extreme,
        np.where(
            out["exh_state"].abs() == 1,
            cfg.dashboard_strength_normal,
            cfg.dashboard_strength_neutral,
        ),
    ).astype(float)

    out["indicator_strength"] = (
        out["strength"] * 0.60
        + (_clamp(out["trigger_quality"], 0.0, 100.0) / 100.0) * 0.40
    ).astype(float)

    out["bias_signal"] = np.where(
        out["exh_bias"] > 0,
        1,
        np.where(out["exh_bias"] < 0, -1, 0),
    ).astype(int)

    out["bias_label"] = np.where(
        out["bias_signal"] == 1,
        "BULLISH",
        np.where(out["bias_signal"] == -1, "BEARISH", "NEUTRAL"),
    )

    out["market_bias"] = np.where(
        (out["bias_signal"] == 1) & (out["bull_absorption_signal"] == 1),
        "BULLISH REVERSAL PRESSURE",
        np.where(
            (out["bias_signal"] == -1) & (out["bear_absorption_signal"] == 1),
            "BEARISH REVERSAL PRESSURE",
            np.where(
                out["bias_signal"] == 1,
                "BULLISH EXHAUSTION",
                np.where(
                    out["bias_signal"] == -1,
                    "BEARISH EXHAUSTION",
                    "NEUTRAL",
                ),
            ),
        ),
    )

    out["exhaustion_direction"] = out["direction"].astype(int)
    out["exhaustion_state_export"] = out["exh_state"].astype(int)
    out["exhaustion_bias_export"] = out["bias_signal"].astype(int)
    out["signal_active"] = out["signal_on"].astype(int)
    out["reversal_risk"] = out["reversal_pressure"].astype(int)

    out["control_winner"] = out["control_winner_text"].astype(str)
    out["control_winner_signal"] = np.where(
        out["control_winner"] == "BULL",
        1,
        np.where(out["control_winner"] == "BEAR", -1, 0),
    ).astype(int)

    out["weak_follow_through"] = out["weakening_participation"].astype(int)

    return out


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_exhaustion_latest_payload(
    df: pd.DataFrame,
    config: Optional[ExhaustionConfig] = None,
) -> Dict[str, Any]:
    """
    Single source of truth payload builder for website/API/dashboard use.
    """
    cfg = config or ExhaustionConfig()
    result = calculate_exhaustion(df=df, config=cfg)

    if result.empty:
        raise ValueError("Exhaustion payload build failed: empty result")

    last = result.iloc[-1]

    payload: Dict[str, Any] = {
        # Core identity
        "module": "exhaustion",
        "version": "exhaustion_payload_v2",
        "timestamp": str(result.index[-1]),

        # Shared website contract
        "state": _safe_str(last.get("exh_text"), "NEUTRAL"),
        "bias_signal": _safe_int(last.get("bias_signal"), 0),
        "bias_label": _safe_str(last.get("bias_label"), "NEUTRAL"),
        "indicator_strength": round(_safe_float(last.get("indicator_strength"), 0.0), 4),
        "market_bias": _safe_str(last.get("market_bias"), "NEUTRAL"),

        # Dashboard-facing summary
        "signal_active": _safe_int(last.get("signal_active"), 0),
        "exhaustion_state": _safe_int(last.get("exhaustion_state_export"), 0),
        "exhaustion_direction": _safe_int(last.get("exhaustion_direction"), 0),
        "reversal_risk": _safe_int(last.get("reversal_risk"), 0),
        "trigger_quality": round(_safe_float(last.get("trigger_quality"), 0.0), 4),

        # Specialist exhaustion fields
        "state_text": _safe_str(last.get("exh_text"), "NEUTRAL"),
        "bias_text": _safe_str(last.get("bias_text"), "NEUTRAL"),
        "top_bottom_context": _safe_str(last.get("top_bottom_text"), "NONE"),
        "state_age": _safe_int(last.get("exh_age"), 0),
        "raw_state": _safe_int(last.get("raw_exh_state"), 0),
        "raw_bias": _safe_int(last.get("raw_exh_bias"), 0),
        "raw_stable_bars": _safe_int(last.get("raw_stable_bars"), 0),
        "bias_stable_bars": _safe_int(last.get("bias_stable_bars"), 0),

        # Exhaustion event outputs
        "bull_exhaustion": _safe_int(last.get("bull_exh_pulse"), 0),
        "bull_extreme": _safe_int(last.get("bull_extreme_pulse"), 0),
        "bear_exhaustion": _safe_int(last.get("bear_exh_pulse"), 0),
        "bear_extreme": _safe_int(last.get("bear_extreme_pulse"), 0),

        # Delta / footprint fields
        "single_candle_delta": round(_safe_float(last.get("single_candle_delta"), 0.0), 4),
        "bull_control_score": round(_safe_float(last.get("bull_control_score"), 0.0), 4),
        "bear_control_score": round(_safe_float(last.get("bear_control_score"), 0.0), 4),
        "control_winner": _safe_str(last.get("control_winner"), "NEUTRAL"),
        "control_winner_signal": _safe_int(last.get("control_winner_signal"), 0),
        "control_strength": round(_safe_float(last.get("control_strength"), 0.0), 4),

        # Range-delta / absorption fields
        "range_delta": round(_safe_float(last.get("range_delta_avg"), 0.0), 4),
        "range_delta_sum": round(_safe_float(last.get("range_delta_sum"), 0.0), 4),
        "range_delta_bias": _safe_int(last.get("range_delta_bias"), 0),
        "range_delta_bias_text": _safe_str(last.get("range_delta_bias_text"), "NEUTRAL"),
        "bull_absorption_signal": _safe_int(last.get("bull_absorption_signal"), 0),
        "bear_absorption_signal": _safe_int(last.get("bear_absorption_signal"), 0),
        "weak_follow_through": _safe_int(last.get("weak_follow_through"), 0),
        "aggressive_bull_continuation": _safe_int(last.get("aggressive_bull_continuation"), 0),
        "aggressive_bear_continuation": _safe_int(last.get("aggressive_bear_continuation"), 0),

        # Diagnostic / support fields
        "ema_base": round(_safe_float(last.get("ema_base"), 0.0), 6),
        "body_fraction": round(_safe_float(last.get("body_frac"), 0.0), 6),
        "stretch_ratio": round(_safe_float(last.get("stretch_ratio"), 0.0), 6),
        "distance_from_ema": round(_safe_float(last.get("dist_from_ema"), 0.0), 6),
        "bull_dominant": _safe_int(last.get("bull_dominant"), 0),
        "bear_dominant": _safe_int(last.get("bear_dominant"), 0),
        "bull_decay": _safe_int(last.get("bull_decay"), 0),
        "bear_decay": _safe_int(last.get("bear_decay"), 0),
        "bull_extreme_flag": _safe_int(last.get("bull_extreme"), 0),
        "bear_extreme_flag": _safe_int(last.get("bear_extreme"), 0),

        # Ultimate Truth dashboard compatibility
        "direction": _safe_int(last.get("direction"), 0),
        "signal_on": _safe_int(last.get("signal_on"), 0),
        "strength": round(_safe_float(last.get("strength"), 0.0), 4),

        # Config
        "config": asdict(cfg),
    }

    return payload


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=500, freq="5min")
    np.random.seed(42)

    close = pd.Series(100 + np.cumsum(np.random.normal(0, 0.25, len(rng))), index=rng)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.random.uniform(0.02, 0.20, len(rng))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.random.uniform(0.02, 0.20, len(rng))

    test_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=rng,
    )

    result = calculate_exhaustion(test_df)
    payload = build_exhaustion_latest_payload(test_df)

    cols = [
        "exh_state",
        "exh_bias",
        "bias_signal",
        "bias_label",
        "indicator_strength",
        "market_bias",
        "single_candle_delta",
        "bull_control_score",
        "bear_control_score",
        "control_winner",
        "range_delta_avg",
        "bull_absorption_signal",
        "bear_absorption_signal",
        "weak_follow_through",
        "top_bottom_text",
        "trigger_quality",
    ]

    print("SmartChart Exhaustion Module v2 — Production backend")
    print(result[cols].tail(20))
    print("\nLatest payload:")
    print(payload)