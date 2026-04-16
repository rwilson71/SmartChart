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
    # Core
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


def _bars_since_change(series: pd.Series) -> pd.Series:
    """
    Pine parity helper for:
        ta.barssince(series != series[1])
    but expressed as a stable-count tracker.

    Result:
        0 on a changed bar
        +1 each bar that remains unchanged after that
    """
    out = np.zeros(len(series), dtype=int)
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index)

    out[0] = 0
    vals = series.to_numpy()

    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            out[i] = 0
        else:
            out[i] = out[i - 1] + 1

    return pd.Series(out, index=series.index, dtype=int)


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


# =============================================================================
# CORE ENGINE
# =============================================================================

def calculate_exhaustion(
    df: pd.DataFrame,
    config: Optional[ExhaustionConfig] = None,
) -> pd.DataFrame:
    """
    SmartChart Exhaustion Module v1
    Clean backend parity rebuild from Pine authority.

    Pine authority states:
        2  = Bull Extreme Exhaustion
        1  = Bull Exhaustion
        0  = Neutral
        -1 = Bear Exhaustion
        -2 = Bear Extreme Exhaustion
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

    # -------------------------------------------------------------------------
    # Raw state
    # Pine order:
    # bullExtreme ? 2 : bullDecay ? 1 : bearExtreme ? -2 : bearDecay ? -1 : 0
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------
    out["exh_changed"] = out["exh_state"].ne(out["exh_state"].shift(1)).fillna(False).astype(int)

    out["bull_exh_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 1)).astype(int)
    out["bull_extreme_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 2)).astype(int)
    out["bear_exh_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -1)).astype(int)
    out["bear_extreme_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -2)).astype(int)

        # -------------------------------------------------------------------------
    # Export-ready fields
    # -------------------------------------------------------------------------
    out["exh_state_export"] = out["exh_state"].astype(int)
    out["exh_bias_export"] = out["exh_bias"].astype(int)
    out["exh_flip_export"] = out["exh_changed"].astype(int)
    out["exh_bull_export"] = out["bull_exh_pulse"].astype(int)
    out["exh_bull_extreme_export"] = out["bull_extreme_pulse"].astype(int)
    out["exh_bear_export"] = out["bear_exh_pulse"].astype(int)
    out["exh_bear_extreme_export"] = out["bear_extreme_pulse"].astype(int)

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

    # -------------------------------------------------------------------------
    # Website-facing normalized fields
    # -------------------------------------------------------------------------
    out["direction"] = np.where(out["exh_state"] > 0, 1, np.where(out["exh_state"] < 0, -1, 0)).astype(int)
    out["signal_on"] = (out["exh_state"] != 0).astype(int)
    out["strength"] = np.where(
        out["exh_state"].abs() == 2, 1.0,
        np.where(out["exh_state"].abs() == 1, 0.65, 0.0)
    ).astype(float)

    return out


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_exhaustion_latest_payload(
    df: pd.DataFrame,
    config: Optional[ExhaustionConfig] = None,
) -> Dict[str, Any]:
    """
    Single source of truth payload builder for website/API use.
    """
    result = calculate_exhaustion(df=df, config=config)

    if result.empty:
        raise ValueError("Exhaustion payload build failed: empty result")

    last = result.iloc[-1]

    payload: Dict[str, Any] = {
        "module": "exhaustion",
        "version": "exhaustion_payload_v1",
        "timestamp": str(result.index[-1]),

        "state": _safe_int(last.get("exh_state_export"), 0),
        "bias": _safe_int(last.get("exh_bias_export"), 0),
        "flip": _safe_int(last.get("exh_flip_export"), 0),

        "bull_exhaustion": _safe_int(last.get("exh_bull_export"), 0),
        "bull_extreme": _safe_int(last.get("exh_bull_extreme_export"), 0),
        "bear_exhaustion": _safe_int(last.get("exh_bear_export"), 0),
        "bear_extreme": _safe_int(last.get("exh_bear_extreme_export"), 0),

        "state_text": str(last.get("exh_text", "NEUTRAL")),
        "bias_text": str(last.get("bias_text", "NEUTRAL")),

        "direction": _safe_int(last.get("direction"), 0),
        "signal_on": _safe_int(last.get("signal_on"), 0),
        "strength": round(_safe_float(last.get("strength"), 0.0), 4),

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

        "raw_state": _safe_int(last.get("raw_exh_state"), 0),
        "raw_bias": _safe_int(last.get("raw_exh_bias"), 0),
        "state_age": _safe_int(last.get("exh_age"), 0),
        "raw_stable_bars": _safe_int(last.get("raw_stable_bars"), 0),
        "bias_stable_bars": _safe_int(last.get("bias_stable_bars"), 0),

        "config": asdict(config or ExhaustionConfig()),
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
        "exh_state_export",
        "exh_bias_export",
        "exh_flip_export",
        "exh_bull_export",
        "exh_bull_extreme_export",
        "exh_bear_export",
        "exh_bear_extreme_export",
        "direction",
        "signal_on",
        "strength",
        "exh_text",
        "bias_text",
    ]

    print("SmartChart Exhaustion Module v1 — Pine parity backend")
    print(result[cols].tail(20))
    print("\nLatest payload:")
    print(payload)