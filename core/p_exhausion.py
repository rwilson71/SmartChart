from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ExhaustionConfig:
    ema_len: int = 20
    stretch_len: int = 20
    push_len: int = 5
    small_body_thresh: float = 0.35
    strong_body_thresh: float = 0.60
    stretch_mult: float = 1.50
    extreme_stretch_mult: float = 2.20
    decay_thresh: float = 0.55

    use_htf: bool = True
    htf_ema_len: int = 20
    htf_stretch_len: int = 20
    htf_stretch_weight: float = 0.50
    require_htf_align: bool = False

    mtf_on: bool = True
    mtf_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

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
    out = np.zeros(len(series), dtype=int)
    for i in range(1, len(series)):
        out[i] = 0 if series.iloc[i] != series.iloc[i - 1] else out[i - 1] + 1
    return pd.Series(out, index=series.index)


def _state_to_score(state: int) -> float:
    if state == 3:
        return 1.00
    if state == 2:
        return 0.75
    if state == 1:
        return 0.40
    if state == -1:
        return -0.40
    if state == -2:
        return -0.75
    if state == -3:
        return -1.00
    return 0.0


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out


def _align_to_base(higher_tf_series: pd.Series, base_index: pd.Index) -> pd.Series:
    aligned = higher_tf_series.reindex(base_index, method="ffill")
    return aligned


def _compute_raw_exhaustion(
    df: pd.DataFrame,
    cfg: ExhaustionConfig,
    htf_context: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    ema_base = _ema(df["close"], cfg.ema_len)

    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    body_size = (df["close"] - df["open"]).abs()
    body_frac = (body_size / bar_range).fillna(0.0)
    body_frac = _clamp(body_frac, 0.0, 1.0)

    is_bull_bar = df["close"] > df["open"]
    is_bear_bar = df["close"] < df["open"]
    is_small_body = body_frac <= cfg.small_body_thresh
    is_strong_body = body_frac >= cfg.strong_body_thresh

    dist_from_ema = df["close"] - ema_base
    abs_dist_from_ema = dist_from_ema.abs()

    stretch_base = _sma(abs_dist_from_ema, cfg.stretch_len).replace(0, np.nan)
    stretch_ratio = (abs_dist_from_ema / stretch_base).fillna(1.0)
    stretch_ratio = _clamp(stretch_ratio, 0.0, 10.0)

    bull_push_raw = np.where(is_bull_bar, body_frac, 0.0)
    bear_push_raw = np.where(is_bear_bar, body_frac, 0.0)

    bull_push = _sma(pd.Series(bull_push_raw, index=df.index), cfg.push_len) * cfg.push_len
    bear_push = _sma(pd.Series(bear_push_raw, index=df.index), cfg.push_len) * cfg.push_len

    bull_bars = _sma(pd.Series(is_bull_bar.astype(float), index=df.index), cfg.push_len) * cfg.push_len
    bear_bars = _sma(pd.Series(is_bear_bar.astype(float), index=df.index), cfg.push_len) * cfg.push_len

    recent_strong_bull = _sma(pd.Series((is_bull_bar & is_strong_body).astype(float), index=df.index), cfg.push_len) * cfg.push_len
    recent_strong_bear = _sma(pd.Series((is_bear_bar & is_strong_body).astype(float), index=df.index), cfg.push_len) * cfg.push_len

    recent_small_bull = _sma(pd.Series((is_bull_bar & is_small_body).astype(float), index=df.index), cfg.push_len) * cfg.push_len
    recent_small_bear = _sma(pd.Series((is_bear_bar & is_small_body).astype(float), index=df.index), cfg.push_len) * cfg.push_len

    bull_dominant = (bull_bars > bear_bars) & (bull_push > bear_push)
    bear_dominant = (bear_bars > bull_bars) & (bear_push > bull_push)

    if htf_context is not None and cfg.use_htf:
        htf_dist_from_ema = htf_context["htf_close"] - htf_context["htf_ema"]
        htf_abs_dist_from_ema = htf_dist_from_ema.abs()
        htf_stretch_base = htf_context["htf_stretch_base"].replace(0, np.nan)
        htf_stretch_ratio = (htf_abs_dist_from_ema / htf_stretch_base).fillna(1.0)
        htf_stretch_ratio = _clamp(htf_stretch_ratio, 0.0, 10.0)

        htf_bull_side = htf_dist_from_ema > 0
        htf_bear_side = htf_dist_from_ema < 0

        bull_htf_aligned = (~pd.Series(cfg.require_htf_align, index=df.index)) | htf_bull_side
        bear_htf_aligned = (~pd.Series(cfg.require_htf_align, index=df.index)) | htf_bear_side

        combined_stretch = stretch_ratio + (htf_stretch_ratio * cfg.htf_stretch_weight)
    else:
        htf_stretch_ratio = pd.Series(0.0, index=df.index)
        htf_bull_side = pd.Series(False, index=df.index)
        htf_bear_side = pd.Series(False, index=df.index)
        bull_htf_aligned = pd.Series(True, index=df.index)
        bear_htf_aligned = pd.Series(True, index=df.index)
        combined_stretch = stretch_ratio

    bull_decay = (
        bull_dominant
        & (dist_from_ema > 0)
        & bull_htf_aligned
        & (combined_stretch >= cfg.stretch_mult)
        & (recent_small_bull >= recent_strong_bull)
        & (body_frac <= cfg.decay_thresh)
    )

    bear_decay = (
        bear_dominant
        & (dist_from_ema < 0)
        & bear_htf_aligned
        & (combined_stretch >= cfg.stretch_mult)
        & (recent_small_bear >= recent_strong_bear)
        & (body_frac <= cfg.decay_thresh)
    )

    bull_extreme = bull_decay & (combined_stretch >= cfg.extreme_stretch_mult)
    bear_extreme = bear_decay & (combined_stretch >= cfg.extreme_stretch_mult)

    bull_htf_exhaust = bull_decay & cfg.use_htf & htf_bull_side & (htf_stretch_ratio >= cfg.stretch_mult)
    bear_htf_exhaust = bear_decay & cfg.use_htf & htf_bear_side & (htf_stretch_ratio >= cfg.stretch_mult)

    raw_exh_state = np.select(
        [
            bull_htf_exhaust,
            bull_extreme,
            bull_decay,
            bear_htf_exhaust,
            bear_extreme,
            bear_decay,
        ],
        [3, 2, 1, -3, -2, -1],
        default=0,
    )
    raw_exh_state = pd.Series(raw_exh_state, index=df.index, dtype=int)
    raw_exh_bias = pd.Series(np.where(raw_exh_state > 0, 1, np.where(raw_exh_state < 0, -1, 0)), index=df.index, dtype=int)

    out["ema_base"] = ema_base
    out["body_frac"] = body_frac
    out["dist_from_ema"] = dist_from_ema
    out["stretch_ratio"] = stretch_ratio
    out["combined_stretch"] = combined_stretch
    out["bull_decay"] = bull_decay.astype(int)
    out["bear_decay"] = bear_decay.astype(int)
    out["bull_extreme"] = bull_extreme.astype(int)
    out["bear_extreme"] = bear_extreme.astype(int)
    out["bull_htf_exhaust"] = pd.Series(bull_htf_exhaust, index=df.index).astype(int)
    out["bear_htf_exhaust"] = pd.Series(bear_htf_exhaust, index=df.index).astype(int)
    out["raw_exh_state"] = raw_exh_state
    out["raw_exh_bias"] = raw_exh_bias
    out["htf_stretch_ratio"] = htf_stretch_ratio

    return out


def _apply_memory(raw_state: pd.Series, raw_bias: pd.Series, cfg: ExhaustionConfig) -> pd.DataFrame:
    raw_stable_count = _bars_since_change(raw_state)
    raw_stable_bars = raw_stable_count + 1
    state_ready = raw_stable_bars >= cfg.confirm_bars

    bias_stable_count = _bars_since_change(raw_bias)
    bias_stable_bars = bias_stable_count + 1
    bias_ready = bias_stable_bars >= cfg.bias_refresh_bars

    exh_state = np.zeros(len(raw_state), dtype=int)
    exh_bias = np.zeros(len(raw_state), dtype=int)
    exh_age = np.zeros(len(raw_state), dtype=int)

    if len(raw_state) == 0:
        return pd.DataFrame(index=raw_state.index)

    exh_state[0] = int(raw_state.iloc[0])
    exh_bias[0] = int(raw_bias.iloc[0])
    exh_age[0] = 0

    for i in range(1, len(raw_state)):
        can_flip = exh_age[i - 1] >= cfg.hold_bars
        allow_state_flip = bool(state_ready.iloc[i]) and int(raw_state.iloc[i]) != int(exh_state[i - 1]) and can_flip
        allow_bias_refresh = bool(bias_ready.iloc[i]) and int(raw_bias.iloc[i]) != 0 and int(raw_bias.iloc[i]) != int(exh_bias[i - 1])

        if allow_state_flip:
            exh_state[i] = int(raw_state.iloc[i])
            exh_bias[i] = int(raw_bias.iloc[i])
            exh_age[i] = 0
        else:
            exh_state[i] = exh_state[i - 1]
            exh_bias[i] = exh_bias[i - 1]
            exh_age[i] = exh_age[i - 1] + 1
            if allow_bias_refresh:
                exh_bias[i] = int(raw_bias.iloc[i])

    out = pd.DataFrame(index=raw_state.index)
    out["raw_stable_bars"] = raw_stable_bars.astype(int)
    out["bias_stable_bars"] = bias_stable_bars.astype(int)
    out["exh_state"] = pd.Series(exh_state, index=raw_state.index, dtype=int)
    out["exh_bias"] = pd.Series(exh_bias, index=raw_state.index, dtype=int)
    out["exh_age"] = pd.Series(exh_age, index=raw_state.index, dtype=int)
    out["exh_changed"] = out["exh_state"].ne(out["exh_state"].shift(1)).fillna(False).astype(int)

    return out


def _compute_mtf_average(
    df: pd.DataFrame,
    cfg: ExhaustionConfig,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if not cfg.mtf_on:
        out["mtf_exh_avg"] = 0.0
        out["mtf_exh_1"] = 0.0
        out["mtf_exh_2"] = 0.0
        out["mtf_exh_3"] = 0.0
        out["mtf_exh_4"] = 0.0
        return out

    tf_map = timeframe_map or {
        "mtf_1": "15min",
        "mtf_2": "1h",
        "mtf_3": "4h",
        "mtf_4": "1D",
    }

    mtf_scores: list[pd.Series] = []

    for key in ["mtf_1", "mtf_2", "mtf_3", "mtf_4"]:
        rule = tf_map[key]
        tf_df = _resample_ohlc(df, rule)
        tf_raw = _compute_raw_exhaustion(tf_df, cfg)
        tf_score = tf_raw["raw_exh_state"].map(_state_to_score).astype(float)
        tf_score = _align_to_base(tf_score, df.index).fillna(0.0)
        mtf_scores.append(tf_score)

    out["mtf_exh_1"] = mtf_scores[0]
    out["mtf_exh_2"] = mtf_scores[1]
    out["mtf_exh_3"] = mtf_scores[2]
    out["mtf_exh_4"] = mtf_scores[3]

    weights = np.array(cfg.mtf_weights, dtype=float)
    weight_sum = float(weights.sum())

    if weight_sum > 0:
        out["mtf_exh_avg"] = (
            out["mtf_exh_1"] * weights[0]
            + out["mtf_exh_2"] * weights[1]
            + out["mtf_exh_3"] * weights[2]
            + out["mtf_exh_4"] * weights[3]
        ) / weight_sum
    else:
        out["mtf_exh_avg"] = 0.0

    return out


def _build_htf_context(
    df: pd.DataFrame,
    cfg: ExhaustionConfig,
    htf_rule: str = "1h",
) -> pd.DataFrame:
    htf_df = _resample_ohlc(df, htf_rule)
    htf_ema = _ema(htf_df["close"], cfg.htf_ema_len)
    htf_stretch_base = _sma((htf_df["close"] - htf_ema).abs(), cfg.htf_stretch_len)

    ctx = pd.DataFrame(index=htf_df.index)
    ctx["htf_close"] = htf_df["close"]
    ctx["htf_ema"] = htf_ema
    ctx["htf_stretch_base"] = htf_stretch_base

    aligned = ctx.reindex(df.index, method="ffill")
    return aligned


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_exhaustion(
    df: pd.DataFrame,
    config: Optional[ExhaustionConfig] = None,
    htf_rule: str = "1h",
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    SmartChart Exhaustion Module v2 backend conversion.

    Expected input:
        DataFrame indexed by datetime with columns:
        open, high, low, close

    Returns:
        DataFrame with SmartChart exhaustion outputs and diagnostics.
    """
    cfg = config or ExhaustionConfig()
    _validate_ohlc(df)

    base = df.copy().sort_index()

    htf_context = _build_htf_context(base, cfg, htf_rule=htf_rule) if cfg.use_htf else None
    raw = _compute_raw_exhaustion(base, cfg, htf_context=htf_context)
    mem = _apply_memory(raw["raw_exh_state"], raw["raw_exh_bias"], cfg)
    mtf = _compute_mtf_average(base, cfg, timeframe_map=timeframe_map)

    out = pd.concat([raw, mem, mtf], axis=1)

    out["bull_exh_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 1)).astype(int)
    out["bull_extreme_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 2)).astype(int)
    out["bull_htf_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == 3)).astype(int)

    out["bear_exh_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -1)).astype(int)
    out["bear_extreme_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -2)).astype(int)
    out["bear_htf_pulse"] = ((out["exh_changed"] == 1) & (out["exh_state"] == -3)).astype(int)

    # Export-ready fields
    out["exh_state_export"] = out["exh_state"].astype(int)
    out["exh_bias_export"] = out["exh_bias"].astype(int)
    out["exh_flip_export"] = out["exh_changed"].astype(int)
    out["exh_bull_export"] = out["bull_exh_pulse"].astype(int)
    out["exh_bull_extreme_export"] = out["bull_extreme_pulse"].astype(int)
    out["exh_bull_htf_export"] = out["bull_htf_pulse"].astype(int)
    out["exh_bear_export"] = out["bear_exh_pulse"].astype(int)
    out["exh_bear_extreme_export"] = out["bear_extreme_pulse"].astype(int)
    out["exh_bear_htf_export"] = out["bear_htf_pulse"].astype(int)

    # Normalized outputs for SmartChart contract
    out["exh_direction"] = np.where(out["exh_state"] > 0, 1, np.where(out["exh_state"] < 0, -1, 0)).astype(int)
    out["exh_strength"] = (out["exh_state"].abs() / 3.0).astype(float)
    out["exh_signal"] = (out["exh_state"] != 0).astype(int)

    return out


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=500, freq="5min")
    np.random.seed(42)

    base = 100 + np.cumsum(np.random.normal(0, 0.25, len(rng)))
    close = pd.Series(base, index=rng)
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

    cols = [
        "exh_state_export",
        "exh_bias_export",
        "exh_flip_export",
        "exh_bull_export",
        "exh_bull_extreme_export",
        "exh_bull_htf_export",
        "exh_bear_export",
        "exh_bear_extreme_export",
        "exh_bear_htf_export",
        "exh_direction",
        "exh_strength",
        "exh_signal",
        "mtf_exh_avg",
    ]

    print("SmartChart Exhaustion Module v2 — direct test")
    print(result[cols].tail(20))