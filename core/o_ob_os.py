from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ObOsConfig:
    # Core
    rsi_len: int = 14
    stoch_len: int = 14
    stoch_smooth: int = 3
    mfi_len: int = 14
    cci_len: int = 20
    sig_smooth_len: int = 5

    # Stretch
    ob_level: float = 0.65
    os_level: float = -0.65
    extreme_ob_level: float = 0.82
    extreme_os_level: float = -0.82

    # Divergence
    div_left: int = 3
    div_right: int = 3
    use_hidden_div: bool = True
    require_ob_os_for_div: bool = True

    # MTF
    mtf_on: bool = True
    tf1: str = "15min"
    tf2: str = "1h"
    tf3: str = "4h"
    tf4: str = "1D"
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0
    w4: float = 1.0

    # Advanced
    trigger_cross_lookback: int = 2
    reversal_gate_level: float = 0.60
    trend_gate_level: float = 0.25
    min_reversal_strength: float = 0.20


# =============================================================================
# HELPERS
# =============================================================================

def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.rolling(length, min_periods=1).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    ll = low.rolling(length, min_periods=1).min()
    hh = high.rolling(length, min_periods=1).max()
    denom = (hh - ll).replace(0.0, np.nan)
    stoch = 100.0 * (close - ll) / denom
    return stoch.fillna(50.0)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    tp = (high + low + close) / 3.0
    rmf = tp * volume.fillna(0.0)

    tp_diff = tp.diff()
    pos_flow = pd.Series(np.where(tp_diff > 0, rmf, 0.0), index=tp.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, rmf, 0.0), index=tp.index)

    pos_sum = pos_flow.rolling(length, min_periods=1).sum()
    neg_sum = neg_flow.rolling(length, min_periods=1).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    return mfi.fillna(50.0)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(length, min_periods=1).mean()

    mad = tp.rolling(length, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    denom = (0.015 * mad).replace(0.0, np.nan)
    cci = (tp - sma_tp) / denom
    return cci.fillna(0.0)


def _clip_norm(series: pd.Series, denom: float) -> pd.Series:
    out = series / denom
    return out.clip(-1.0, 1.0)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _bars_since(condition: pd.Series) -> pd.Series:
    out = np.full(len(condition), np.nan)
    last_idx = -1
    vals = condition.fillna(False).to_numpy()

    for i, v in enumerate(vals):
        if v:
            last_idx = i
            out[i] = 0.0
        elif last_idx >= 0:
            out[i] = float(i - last_idx)

    return pd.Series(out, index=condition.index)


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    left = max(1, int(left))
    right = max(1, int(right))
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)

    for i in range(left, len(vals) - right):
        center = vals[i]
        if np.isnan(center):
            continue
        left_slice = vals[i - left:i]
        right_slice = vals[i + 1:i + 1 + right]
        if np.all(center < left_slice) and np.all(center <= right_slice):
            out[i + right] = center  # Pine confirms pivot right bars later

    return pd.Series(out, index=series.index)


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    left = max(1, int(left))
    right = max(1, int(right))
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)

    for i in range(left, len(vals) - right):
        center = vals[i]
        if np.isnan(center):
            continue
        left_slice = vals[i - left:i]
        right_slice = vals[i + 1:i + 1 + right]
        if np.all(center > left_slice) and np.all(center >= right_slice):
            out[i + right] = center  # Pine confirms pivot right bars later

    return pd.Series(out, index=series.index)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        ohlc["volume"] = "sum"

    out = df.resample(rule).agg(ohlc).dropna(subset=["open", "high", "low", "close"])
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


def _compute_composite_only(df: pd.DataFrame, cfg: ObOsConfig) -> pd.Series:
    rsi_raw = _rsi(df["close"], cfg.rsi_len)
    stoch_raw = _sma(_stoch_k(df["close"], df["high"], df["low"], cfg.stoch_len), cfg.stoch_smooth)
    mfi_raw = _mfi(df["high"], df["low"], df["close"], df["volume"], cfg.mfi_len)
    cci_raw = _cci(df["high"], df["low"], df["close"], cfg.cci_len)

    rsi_norm = ((rsi_raw - 50.0) / 50.0).clip(-1.0, 1.0)
    stoch_norm = ((stoch_raw - 50.0) / 50.0).clip(-1.0, 1.0)
    mfi_norm = ((mfi_raw - 50.0) / 50.0).clip(-1.0, 1.0)
    cci_norm = _clip_norm(cci_raw, 200.0)

    composite_raw = (
        rsi_norm * 0.35
        + stoch_norm * 0.20
        + mfi_norm * 0.25
        + cci_norm * 0.20
    )

    return _ema(composite_raw, cfg.sig_smooth_len)


def _mtf_to_base(df: pd.DataFrame, rule: str, cfg: ObOsConfig) -> pd.Series:
    tf_df = _resample_ohlcv(df, rule)
    tf_comp = _compute_composite_only(tf_df, cfg)
    return tf_comp.reindex(df.index, method="ffill").fillna(0.0)


# =============================================================================
# ENGINE
# =============================================================================

def run_ob_os_engine(
    df: pd.DataFrame,
    config: Optional[ObOsConfig] = None,
) -> pd.DataFrame:
    """
    SmartChart Backend — o_ob_os.py

    OB/OS + Divergence Engine v2
    Python parity rebuild from TradingView reference.

    Expected columns:
        open, high, low, close
    Optional:
        volume

    Returns:
        original dataframe + OB/OS engine outputs
    """
    cfg = config or ObOsConfig()
    out = df.copy()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"run_ob_os_engine: missing required columns: {missing}")

    if "volume" not in out.columns:
        out["volume"] = 0.0

    # -------------------------------------------------------------------------
    # CORE ENGINE
    # -------------------------------------------------------------------------
    out["sc_obos_rsi_raw"] = _rsi(out["close"], cfg.rsi_len)
    out["sc_obos_stoch_raw"] = _sma(
        _stoch_k(out["close"], out["high"], out["low"], cfg.stoch_len),
        cfg.stoch_smooth,
    )
    out["sc_obos_mfi_raw"] = _mfi(out["high"], out["low"], out["close"], out["volume"], cfg.mfi_len)
    out["sc_obos_cci_raw"] = _cci(out["high"], out["low"], out["close"], cfg.cci_len)

    out["sc_obos_rsi_norm"] = ((out["sc_obos_rsi_raw"] - 50.0) / 50.0).clip(-1.0, 1.0)
    out["sc_obos_stoch_norm"] = ((out["sc_obos_stoch_raw"] - 50.0) / 50.0).clip(-1.0, 1.0)
    out["sc_obos_mfi_norm"] = ((out["sc_obos_mfi_raw"] - 50.0) / 50.0).clip(-1.0, 1.0)
    out["sc_obos_cci_norm"] = _clip_norm(out["sc_obos_cci_raw"], 200.0)

    out["sc_obos_composite_raw"] = (
        out["sc_obos_rsi_norm"] * 0.35
        + out["sc_obos_stoch_norm"] * 0.20
        + out["sc_obos_mfi_norm"] * 0.25
        + out["sc_obos_cci_norm"] * 0.20
    )

    out["sc_obos_composite"] = _ema(out["sc_obos_composite_raw"], cfg.sig_smooth_len)
    out["sc_obos_signal"] = _ema(out["sc_obos_composite"], max(2, cfg.sig_smooth_len))

    out["sc_is_ob"] = out["sc_obos_composite"] >= cfg.ob_level
    out["sc_is_os"] = out["sc_obos_composite"] <= cfg.os_level
    out["sc_is_extreme_ob"] = out["sc_obos_composite"] >= cfg.extreme_ob_level
    out["sc_is_extreme_os"] = out["sc_obos_composite"] <= cfg.extreme_os_level

    out["sc_obos_state"] = np.select(
        [
            out["sc_is_extreme_ob"],
            out["sc_is_ob"],
            out["sc_is_extreme_os"],
            out["sc_is_os"],
        ],
        [2, 1, -2, -1],
        default=0,
    ).astype(int)

    # -------------------------------------------------------------------------
    # DIVERGENCE
    # -------------------------------------------------------------------------
    piv_low_price = _pivot_low(out["low"], cfg.div_left, cfg.div_right)
    piv_high_price = _pivot_high(out["high"], cfg.div_left, cfg.div_right)
    piv_low_osc = _pivot_low(out["sc_obos_composite"], cfg.div_left, cfg.div_right)
    piv_high_osc = _pivot_high(out["sc_obos_composite"], cfg.div_left, cfg.div_right)

    bullish_div = np.zeros(len(out), dtype=bool)
    bearish_div = np.zeros(len(out), dtype=bool)
    hidden_bullish_div = np.zeros(len(out), dtype=bool)
    hidden_bearish_div = np.zeros(len(out), dtype=bool)
    bull_div_strength = np.zeros(len(out), dtype=float)
    bear_div_strength = np.zeros(len(out), dtype=float)

    prev_bull_price = np.nan
    prev_bull_osc = np.nan
    prev_bear_price = np.nan
    prev_bear_osc = np.nan

    comp_vals = out["sc_obos_composite"].to_numpy()
    low_p_vals = piv_low_price.to_numpy()
    low_o_vals = piv_low_osc.to_numpy()
    high_p_vals = piv_high_price.to_numpy()
    high_o_vals = piv_high_osc.to_numpy()

    for i in range(len(out)):
        # Bull side
        if not np.isnan(low_p_vals[i]) and not np.isnan(low_o_vals[i]):
            ctx_bull = comp_vals[i - cfg.div_right] if i - cfg.div_right >= 0 else np.nan
            ctx_bull_ok = (not cfg.require_ob_os_for_div) or (ctx_bull <= cfg.os_level)

            reg_bull = (
                not np.isnan(prev_bull_price)
                and not np.isnan(prev_bull_osc)
                and low_p_vals[i] < prev_bull_price
                and low_o_vals[i] > prev_bull_osc
                and ctx_bull_ok
            )
            hid_bull = (
                cfg.use_hidden_div
                and not np.isnan(prev_bull_price)
                and not np.isnan(prev_bull_osc)
                and low_p_vals[i] > prev_bull_price
                and low_o_vals[i] < prev_bull_osc
                and ctx_bull_ok
            )

            bullish_div[i] = reg_bull
            hidden_bullish_div[i] = hid_bull

            if reg_bull:
                bull_div_strength[i] = float(np.clip(abs(ctx_bull) + 0.25, 0.0, 1.0))
            elif hid_bull:
                bull_div_strength[i] = 0.65

            prev_bull_price = low_p_vals[i]
            prev_bull_osc = low_o_vals[i]

        # Bear side
        if not np.isnan(high_p_vals[i]) and not np.isnan(high_o_vals[i]):
            ctx_bear = comp_vals[i - cfg.div_right] if i - cfg.div_right >= 0 else np.nan
            ctx_bear_ok = (not cfg.require_ob_os_for_div) or (ctx_bear >= cfg.ob_level)

            reg_bear = (
                not np.isnan(prev_bear_price)
                and not np.isnan(prev_bear_osc)
                and high_p_vals[i] > prev_bear_price
                and high_o_vals[i] < prev_bear_osc
                and ctx_bear_ok
            )
            hid_bear = (
                cfg.use_hidden_div
                and not np.isnan(prev_bear_price)
                and not np.isnan(prev_bear_osc)
                and high_p_vals[i] < prev_bear_price
                and high_o_vals[i] > prev_bear_osc
                and ctx_bear_ok
            )

            bearish_div[i] = reg_bear
            hidden_bearish_div[i] = hid_bear

            if reg_bear:
                bear_div_strength[i] = float(np.clip(abs(ctx_bear) + 0.25, 0.0, 1.0))
            elif hid_bear:
                bear_div_strength[i] = 0.65

            prev_bear_price = high_p_vals[i]
            prev_bear_osc = high_o_vals[i]

    out["sc_bullish_div"] = bullish_div
    out["sc_bearish_div"] = bearish_div
    out["sc_hidden_bullish_div"] = hidden_bullish_div
    out["sc_hidden_bearish_div"] = hidden_bearish_div
    out["sc_bull_div_strength"] = bull_div_strength
    out["sc_bear_div_strength"] = bear_div_strength

    out["sc_div_bull_any"] = out["sc_bullish_div"] | out["sc_hidden_bullish_div"]
    out["sc_div_bear_any"] = out["sc_bearish_div"] | out["sc_hidden_bearish_div"]

    # -------------------------------------------------------------------------
    # MTF AVG
    # -------------------------------------------------------------------------
    total_weight = max(cfg.w1 + cfg.w2 + cfg.w3 + cfg.w4, 0.0001)

    if cfg.mtf_on:
        out["sc_tf1_comp"] = _mtf_to_base(out, cfg.tf1, cfg) if cfg.w1 > 0 else 0.0
        out["sc_tf2_comp"] = _mtf_to_base(out, cfg.tf2, cfg) if cfg.w2 > 0 else 0.0
        out["sc_tf3_comp"] = _mtf_to_base(out, cfg.tf3, cfg) if cfg.w3 > 0 else 0.0
        out["sc_tf4_comp"] = _mtf_to_base(out, cfg.tf4, cfg) if cfg.w4 > 0 else 0.0
    else:
        out["sc_tf1_comp"] = 0.0
        out["sc_tf2_comp"] = 0.0
        out["sc_tf3_comp"] = 0.0
        out["sc_tf4_comp"] = 0.0

    out["sc_mtf_avg"] = (
        out["sc_tf1_comp"] * cfg.w1
        + out["sc_tf2_comp"] * cfg.w2
        + out["sc_tf3_comp"] * cfg.w3
        + out["sc_tf4_comp"] * cfg.w4
    ) / total_weight

    out["sc_mtf_aligned_bull"] = cfg.mtf_on & (out["sc_mtf_avg"] > 0.15)
    out["sc_mtf_aligned_bear"] = cfg.mtf_on & (out["sc_mtf_avg"] < -0.15)
    out["sc_mtf_extreme_bull"] = cfg.mtf_on & (out["sc_mtf_avg"] >= cfg.extreme_ob_level)
    out["sc_mtf_extreme_bear"] = cfg.mtf_on & (out["sc_mtf_avg"] <= cfg.extreme_os_level)

    # -------------------------------------------------------------------------
    # DECISION ENGINE
    # -------------------------------------------------------------------------
    out["sc_bull_cross_up"] = _cross_up(out["sc_obos_composite"], out["sc_obos_signal"])
    out["sc_bear_cross_down"] = _cross_down(out["sc_obos_composite"], out["sc_obos_signal"])

    out["sc_bars_since_bull_cross"] = _bars_since(out["sc_bull_cross_up"])
    out["sc_bars_since_bear_cross"] = _bars_since(out["sc_bear_cross_down"])

    out["sc_bull_recent_cross"] = (
        out["sc_bars_since_bull_cross"].notna()
        & (out["sc_bars_since_bull_cross"] >= 0)
        & (out["sc_bars_since_bull_cross"] <= cfg.trigger_cross_lookback)
    )
    out["sc_bear_recent_cross"] = (
        out["sc_bars_since_bear_cross"].notna()
        & (out["sc_bars_since_bear_cross"] >= 0)
        & (out["sc_bars_since_bear_cross"] <= cfg.trigger_cross_lookback)
    )

    out["sc_bull_zero_reclaim"] = (
        (out["sc_obos_composite"] > 0) & (out["sc_obos_composite"].shift(1) <= 0)
    )
    out["sc_bear_zero_reject"] = (
        (out["sc_obos_composite"] < 0) & (out["sc_obos_composite"].shift(1) >= 0)
    )

    out["sc_bull_recovery_structure"] = (
        (out["sc_obos_composite"] < 0)
        & (out["sc_obos_composite"] > out["sc_obos_composite"].shift(1))
        & (out["sc_obos_signal"] > out["sc_obos_signal"].shift(1))
    )
    out["sc_bear_rollover_structure"] = (
        (out["sc_obos_composite"] > 0)
        & (out["sc_obos_composite"] < out["sc_obos_composite"].shift(1))
        & (out["sc_obos_signal"] < out["sc_obos_signal"].shift(1))
    )

    out["sc_bull_turn_confirmed"] = out["sc_bull_recent_cross"] & out["sc_bull_recovery_structure"]
    out["sc_bear_turn_confirmed"] = out["sc_bear_recent_cross"] & out["sc_bear_rollover_structure"]

    out["sc_bull_strength_ok"] = out["sc_obos_composite"].abs() > cfg.min_reversal_strength
    out["sc_bear_strength_ok"] = out["sc_obos_composite"].abs() > cfg.min_reversal_strength

    out["sc_bull_reversal_score"] = np.minimum(
        1.0,
        np.where(out["sc_is_os"] | out["sc_is_extreme_os"], 0.25, 0.0)
        + out["sc_bull_div_strength"] * 0.25
        + np.where(out["sc_bull_turn_confirmed"], 0.20, 0.0)
        + np.where(out["sc_bull_zero_reclaim"] | (out["sc_obos_composite"] > out["sc_obos_signal"]), 0.12, 0.0)
        + np.where(out["sc_bull_strength_ok"], 0.10, 0.0)
        + np.where(~out["sc_mtf_aligned_bear"], 0.08, 0.0),
    )

    out["sc_bear_reversal_score"] = np.minimum(
        1.0,
        np.where(out["sc_is_ob"] | out["sc_is_extreme_ob"], 0.25, 0.0)
        + out["sc_bear_div_strength"] * 0.25
        + np.where(out["sc_bear_turn_confirmed"], 0.20, 0.0)
        + np.where(out["sc_bear_zero_reject"] | (out["sc_obos_composite"] < out["sc_obos_signal"]), 0.12, 0.0)
        + np.where(out["sc_bear_strength_ok"], 0.10, 0.0)
        + np.where(~out["sc_mtf_aligned_bull"], 0.08, 0.0),
    )

    out["sc_bull_continuation_score"] = np.minimum(
        1.0,
        np.where(out["sc_obos_composite"] > cfg.trend_gate_level, 0.30, 0.0)
        + np.where(out["sc_obos_composite"] > out["sc_obos_signal"], 0.20, 0.0)
        + np.where(out["sc_mtf_aligned_bull"], 0.20, 0.0)
        + np.where(~out["sc_is_ob"] & ~out["sc_is_extreme_ob"], 0.15, 0.0)
        + np.where(~out["sc_div_bear_any"], 0.15, 0.0),
    )

    out["sc_bear_continuation_score"] = np.minimum(
        1.0,
        np.where(out["sc_obos_composite"] < -cfg.trend_gate_level, 0.30, 0.0)
        + np.where(out["sc_obos_composite"] < out["sc_obos_signal"], 0.20, 0.0)
        + np.where(out["sc_mtf_aligned_bear"], 0.20, 0.0)
        + np.where(~out["sc_is_os"] & ~out["sc_is_extreme_os"], 0.15, 0.0)
        + np.where(~out["sc_div_bull_any"], 0.15, 0.0),
    )

    out["sc_exhaustion_score"] = np.minimum(
        1.0,
        np.where(out["sc_is_extreme_ob"] | out["sc_is_extreme_os"], 0.30, 0.0)
        + np.maximum(out["sc_bull_div_strength"], out["sc_bear_div_strength"]) * 0.25
        + np.where(out["sc_obos_composite"].abs() > 0.70, 0.15, 0.0)
        + np.where((out["sc_obos_composite"] - out["sc_obos_signal"]).abs() < 0.08, 0.10, 0.0)
        + np.where(
            (out["sc_mtf_aligned_bull"] & out["sc_is_ob"])
            | (out["sc_mtf_aligned_bear"] & out["sc_is_os"]),
            0.20,
            0.0,
        ),
    )

    # -------------------------------------------------------------------------
    # FINAL STATE
    # -------------------------------------------------------------------------
    sc_state = np.zeros(len(out), dtype=int)
    active_score = np.zeros(len(out), dtype=float)

    bull_rev = out["sc_bull_reversal_score"].to_numpy()
    bear_rev = out["sc_bear_reversal_score"].to_numpy()
    bull_cont = out["sc_bull_continuation_score"].to_numpy()
    bear_cont = out["sc_bear_continuation_score"].to_numpy()

    for i in range(len(out)):
        if bull_rev[i] >= cfg.reversal_gate_level and bull_rev[i] > bear_rev[i]:
            sc_state[i] = 2
            active_score[i] = bull_rev[i]
        elif bear_rev[i] >= cfg.reversal_gate_level and bear_rev[i] > bull_rev[i]:
            sc_state[i] = -2
            active_score[i] = bear_rev[i]
        elif bull_cont[i] >= cfg.reversal_gate_level and bull_cont[i] > bear_cont[i]:
            sc_state[i] = 1
            active_score[i] = bull_cont[i]
        elif bear_cont[i] >= cfg.reversal_gate_level and bear_cont[i] > bull_cont[i]:
            sc_state[i] = -1
            active_score[i] = bear_cont[i]

    out["sc_obos_state_final"] = sc_state
    out["sc_obos_active_score"] = active_score

    out["sc_obos_grade"] = np.select(
        [
            out["sc_obos_active_score"] >= 0.85,
            out["sc_obos_active_score"] >= 0.72,
            out["sc_obos_active_score"] >= 0.60,
        ],
        [3, 2, 1],
        default=0,
    ).astype(int)

    out["sc_bull_pressure"] = out["sc_bull_continuation_score"]
    out["sc_bear_pressure"] = out["sc_bear_continuation_score"]
    out["sc_stretch_score"] = np.minimum(1.0, out["sc_obos_composite"].abs())
    out["sc_exhaustion_risk"] = out["sc_exhaustion_score"]
    out["sc_continuation_strength"] = np.maximum(
        out["sc_bull_continuation_score"],
        out["sc_bear_continuation_score"],
    )
    out["sc_reversal_strength"] = np.maximum(
        out["sc_bull_reversal_score"],
        out["sc_bear_reversal_score"],
    )

    out["sc_bull_reversal_ready"] = out["sc_obos_state_final"] == 2
    out["sc_bear_reversal_ready"] = out["sc_obos_state_final"] == -2

    out["sc_bull_trend_exhaustion_risk"] = (
        out["sc_mtf_aligned_bull"]
        & (out["sc_is_ob"] | out["sc_is_extreme_ob"])
        & out["sc_div_bear_any"]
    )
    out["sc_bear_trend_exhaustion_risk"] = (
        out["sc_mtf_aligned_bear"]
        & (out["sc_is_os"] | out["sc_is_extreme_os"])
        & out["sc_div_bull_any"]
    )

    out["sc_obos_dir"] = np.select(
        [
            out["sc_obos_composite"] > 0.10,
            out["sc_obos_composite"] < -0.10,
        ],
        [1, -1],
        default=0,
    ).astype(int)

    out["sc_stretch_state"] = np.select(
        [
            out["sc_is_extreme_ob"],
            out["sc_is_ob"],
            out["sc_is_extreme_os"],
            out["sc_is_os"],
        ],
        [2, 1, -2, -1],
        default=0,
    ).astype(int)

    out["sc_div_state"] = np.select(
        [
            out["sc_bearish_div"],
            out["sc_hidden_bearish_div"],
            out["sc_bullish_div"],
            out["sc_hidden_bullish_div"],
        ],
        [-2, -1, 2, 1],
        default=0,
    ).astype(int)

    out["sc_mtf_dir"] = np.select(
        [
            cfg.mtf_on and True,
            cfg.mtf_on and True,
        ],
        [
            np.where(out["sc_mtf_avg"] > 0.10, 1, 0),
            np.where(out["sc_mtf_avg"] < -0.10, -1, 0),
        ],
        default=0,
    )

    # Clean export aliases
    out["obos_composite"] = out["sc_obos_composite"]
    out["obos_signal"] = out["sc_obos_signal"]
    out["obos_state"] = out["sc_obos_state_final"]
    out["obos_grade"] = out["sc_obos_grade"]
    out["obos_active_score"] = out["sc_obos_active_score"]
    out["obos_div_state"] = out["sc_div_state"]
    out["obos_mtf_avg"] = out["sc_mtf_avg"]
    out["obos_mtf_dir"] = out["sc_mtf_dir"]
    out["obos_reversal_strength"] = out["sc_reversal_strength"]
    out["obos_continuation_strength"] = out["sc_continuation_strength"]
    out["obos_exhaustion_score"] = out["sc_exhaustion_score"]

    return out


# =============================================================================
# DEFAULT EXPORT CONTRACT
# =============================================================================

DEFAULT_OB_OS_CONFIG: Dict[str, Any] = asdict(ObOsConfig())


# =============================================================================
# DIRECT TEST
# =============================================================================

if __name__ == "__main__":
    idx = pd.date_range("2026-01-01", periods=500, freq="5min")
    rng = np.random.default_rng(42)

    base = 2650 + np.cumsum(rng.normal(0, 1.8, len(idx)))
    high = base + rng.uniform(0.2, 1.8, len(idx))
    low = base - rng.uniform(0.2, 1.8, len(idx))
    open_ = base + rng.normal(0, 0.5, len(idx))
    close = base + rng.normal(0, 0.5, len(idx))
    volume = rng.integers(100, 1000, len(idx))

    test_df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume,
        },
        index=idx,
    )

    result = run_ob_os_engine(test_df)

    cols = [
        "sc_obos_composite",
        "sc_obos_signal",
        "sc_obos_state",
        "sc_div_state",
        "sc_mtf_avg",
        "sc_obos_state_final",
        "sc_obos_grade",
        "sc_obos_active_score",
        "sc_bull_reversal_score",
        "sc_bear_reversal_score",
        "sc_bull_continuation_score",
        "sc_bear_continuation_score",
        "sc_exhaustion_score",
        "sc_mtf_dir",
    ]
    print(result[cols].tail(10))