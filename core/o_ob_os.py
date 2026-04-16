from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

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


DEFAULT_OB_OS_CONFIG: Dict[str, Any] = asdict(ObOsConfig())


# =============================================================================
# HELPERS
# =============================================================================

def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


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

    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    loss_zero = (avg_loss == 0.0) & (avg_gain > 0.0)
    gain_zero = (avg_gain == 0.0) & (avg_loss > 0.0)

    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask(loss_zero, 100.0)
    rsi = rsi.mask(gain_zero, 0.0)

    return rsi.fillna(50.0)


def _stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    ll = low.rolling(length, min_periods=1).min()
    hh = high.rolling(length, min_periods=1).max()
    denom = (hh - ll).replace(0.0, np.nan)
    stoch = 100.0 * (close - ll) / denom
    return stoch.fillna(50.0)


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int,
) -> pd.Series:
    length = max(1, int(length))
    tp = (high + low + close) / 3.0
    rmf = tp * volume.fillna(0.0)

    tp_diff = tp.diff()
    pos_flow = pd.Series(np.where(tp_diff > 0, rmf, 0.0), index=tp.index, dtype=float)
    neg_flow = pd.Series(np.where(tp_diff < 0, rmf, 0.0), index=tp.index, dtype=float)

    pos_sum = pos_flow.rolling(length, min_periods=1).sum()
    neg_sum = neg_flow.rolling(length, min_periods=1).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))

    both_zero = (pos_sum == 0.0) & (neg_sum == 0.0)
    neg_zero = (neg_sum == 0.0) & (pos_sum > 0.0)
    pos_zero = (pos_sum == 0.0) & (neg_sum > 0.0)

    mfi = mfi.mask(both_zero, 50.0)
    mfi = mfi.mask(neg_zero, 100.0)
    mfi = mfi.mask(pos_zero, 0.0)

    return mfi.fillna(50.0)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(length, min_periods=1).mean()

    mad = tp.rolling(length, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    denom = (0.015 * mad).replace(0.0, np.nan)
    cci = (tp - sma_tp) / denom
    return cci.fillna(0.0)


def _clip_norm(series: pd.Series, denom: float) -> pd.Series:
    return (series / denom).clip(-1.0, 1.0)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _bars_since(condition: pd.Series) -> pd.Series:
    out = np.full(len(condition), np.nan, dtype=float)
    last_idx = -1
    vals = condition.fillna(False).to_numpy(dtype=bool)

    for i, v in enumerate(vals):
        if v:
            last_idx = i
            out[i] = 0.0
        elif last_idx >= 0:
            out[i] = float(i - last_idx)

    return pd.Series(out, index=condition.index, dtype=float)


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    left = max(1, int(left))
    right = max(1, int(right))
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(left, len(vals) - right):
        center = vals[i]
        if np.isnan(center):
            continue
        left_slice = vals[i - left:i]
        right_slice = vals[i + 1:i + 1 + right]
        if np.all(center < left_slice) and np.all(center <= right_slice):
            out[i + right] = center  # matches Pine pivot confirmation timing

    return pd.Series(out, index=series.index, dtype=float)


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    left = max(1, int(left))
    right = max(1, int(right))
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(left, len(vals) - right):
        center = vals[i]
        if np.isnan(center):
            continue
        left_slice = vals[i - left:i]
        right_slice = vals[i + 1:i + 1 + right]
        if np.all(center > left_slice) and np.all(center >= right_slice):
            out[i + right] = center  # matches Pine pivot confirmation timing

    return pd.Series(out, index=series.index, dtype=float)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
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
    if tf_df.empty:
        return pd.Series(0.0, index=df.index, dtype=float)
    tf_comp = _compute_composite_only(tf_df, cfg)
    return tf_comp.reindex(df.index, method="ffill").fillna(0.0)


def _state_text(v: int) -> str:
    if v == 2:
        return "BULL REV"
    if v == 1:
        return "BULL CONT"
    if v == -1:
        return "BEAR CONT"
    if v == -2:
        return "BEAR REV"
    return "NEUTRAL"


def _div_text(v: int) -> str:
    if v == 2:
        return "BULL DIV"
    if v == 1:
        return "H BULL"
    if v == -1:
        return "H BEAR"
    if v == -2:
        return "BEAR DIV"
    return "NONE"


def _grade_text(v: int) -> str:
    if v == 3:
        return "A"
    if v == 2:
        return "B"
    if v == 1:
        return "C"
    return "N"


def _stretch_text(v: int) -> str:
    if v == 2:
        return "EXTREME OB"
    if v == 1:
        return "OB"
    if v == -1:
        return "OS"
    if v == -2:
        return "EXTREME OS"
    return "NEUTRAL"


def _dir_text(v: int) -> str:
    if v > 0:
        return "BULL"
    if v < 0:
        return "BEAR"
    return "NEUTRAL"


def _bool(v: Any) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return bool(v)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return default


def _int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return default


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
    Production parity rebuild from TradingView authority.

    Required columns:
        open, high, low, close
    Optional:
        volume
    """
    cfg = config or ObOsConfig()
    out = df.copy()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"run_ob_os_engine: missing required columns: {missing}")

    out["open"] = _safe_series(out, "open")
    out["high"] = _safe_series(out, "high")
    out["low"] = _safe_series(out, "low")
    out["close"] = _safe_series(out, "close")
    out["volume"] = _safe_series(out, "volume", default=0.0)

    # -------------------------------------------------------------------------
    # CORE ENGINE
    # -------------------------------------------------------------------------
    out["sc_obos_rsi_raw"] = _rsi(out["close"], cfg.rsi_len)
    out["sc_obos_stoch_raw"] = _sma(
        _stoch_k(out["close"], out["high"], out["low"], cfg.stoch_len),
        cfg.stoch_smooth,
    )
    out["sc_obos_mfi_raw"] = _mfi(
        out["high"],
        out["low"],
        out["close"],
        out["volume"],
        cfg.mfi_len,
    )
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

    comp_vals = out["sc_obos_composite"].to_numpy(dtype=float)
    low_p_vals = piv_low_price.to_numpy(dtype=float)
    low_o_vals = piv_low_osc.to_numpy(dtype=float)
    high_p_vals = piv_high_price.to_numpy(dtype=float)
    high_o_vals = piv_high_osc.to_numpy(dtype=float)

    for i in range(len(out)):
        if not np.isnan(low_p_vals[i]) and not np.isnan(low_o_vals[i]):
            ctx_idx = i - cfg.div_right
            ctx_bull = comp_vals[ctx_idx] if ctx_idx >= 0 else np.nan
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

        if not np.isnan(high_p_vals[i]) and not np.isnan(high_o_vals[i]):
            ctx_idx = i - cfg.div_right
            ctx_bear = comp_vals[ctx_idx] if ctx_idx >= 0 else np.nan
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

    out["sc_mtf_aligned_bull"] = bool(cfg.mtf_on) & (out["sc_mtf_avg"] > 0.15)
    out["sc_mtf_aligned_bear"] = bool(cfg.mtf_on) & (out["sc_mtf_avg"] < -0.15)
    out["sc_mtf_extreme_bull"] = bool(cfg.mtf_on) & (out["sc_mtf_avg"] >= cfg.extreme_ob_level)
    out["sc_mtf_extreme_bear"] = bool(cfg.mtf_on) & (out["sc_mtf_avg"] <= cfg.extreme_os_level)

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
        (out["sc_obos_composite"] > 0.0)
        & (out["sc_obos_composite"].shift(1) <= 0.0)
    )
    out["sc_bear_zero_reject"] = (
        (out["sc_obos_composite"] < 0.0)
        & (out["sc_obos_composite"].shift(1) >= 0.0)
    )

    out["sc_bull_recovery_structure"] = (
        (out["sc_obos_composite"] < 0.0)
        & (out["sc_obos_composite"] > out["sc_obos_composite"].shift(1))
        & (out["sc_obos_signal"] > out["sc_obos_signal"].shift(1))
    )
    out["sc_bear_rollover_structure"] = (
        (out["sc_obos_composite"] > 0.0)
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
        + np.where(
            out["sc_bull_zero_reclaim"] | (out["sc_obos_composite"] > out["sc_obos_signal"]),
            0.12,
            0.0,
        )
        + np.where(out["sc_bull_strength_ok"], 0.10, 0.0)
        + np.where(~out["sc_mtf_aligned_bear"], 0.08, 0.0),
    )

    out["sc_bear_reversal_score"] = np.minimum(
        1.0,
        np.where(out["sc_is_ob"] | out["sc_is_extreme_ob"], 0.25, 0.0)
        + out["sc_bear_div_strength"] * 0.25
        + np.where(out["sc_bear_turn_confirmed"], 0.20, 0.0)
        + np.where(
            out["sc_bear_zero_reject"] | (out["sc_obos_composite"] < out["sc_obos_signal"]),
            0.12,
            0.0,
        )
        + np.where(out["sc_bear_strength_ok"], 0.10, 0.0)
        + np.where(~out["sc_mtf_aligned_bull"], 0.08, 0.0),
    )

    out["sc_bull_continuation_score"] = np.minimum(
        1.0,
        np.where(out["sc_obos_composite"] > cfg.trend_gate_level, 0.30, 0.0)
        + np.where(out["sc_obos_composite"] > out["sc_obos_signal"], 0.20, 0.0)
        + np.where(out["sc_mtf_aligned_bull"], 0.20, 0.0)
        + np.where((~out["sc_is_ob"]) & (~out["sc_is_extreme_ob"]), 0.15, 0.0)
        + np.where(~out["sc_div_bear_any"], 0.15, 0.0),
    )

    out["sc_bear_continuation_score"] = np.minimum(
        1.0,
        np.where(out["sc_obos_composite"] < -cfg.trend_gate_level, 0.30, 0.0)
        + np.where(out["sc_obos_composite"] < out["sc_obos_signal"], 0.20, 0.0)
        + np.where(out["sc_mtf_aligned_bear"], 0.20, 0.0)
        + np.where((~out["sc_is_os"]) & (~out["sc_is_extreme_os"]), 0.15, 0.0)
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

    bull_rev = out["sc_bull_reversal_score"].to_numpy(dtype=float)
    bear_rev = out["sc_bear_reversal_score"].to_numpy(dtype=float)
    bull_cont = out["sc_bull_continuation_score"].to_numpy(dtype=float)
    bear_cont = out["sc_bear_continuation_score"].to_numpy(dtype=float)

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
            bool(cfg.mtf_on) & (out["sc_mtf_avg"] > 0.10),
            bool(cfg.mtf_on) & (out["sc_mtf_avg"] < -0.10),
        ],
        [1, -1],
        default=0,
    ).astype(int)

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
# PAYLOAD BUILDER
# =============================================================================

def build_ob_os_latest_payload(
    df: pd.DataFrame,
    config: Optional[ObOsConfig] = None,
) -> Dict[str, Any]:
    """
    Build latest OB/OS payload for website/API use.
    Cache service and router should both use this same payload builder.
    """
    cfg = config or ObOsConfig()
    result = run_ob_os_engine(df, cfg)

    if result.empty:
        return {}

    row = result.iloc[-1]

    timestamp = result.index[-1]
    if isinstance(timestamp, pd.Timestamp):
        timestamp_iso = timestamp.isoformat()
    else:
        timestamp_iso = str(timestamp)

    state = _int(row.get("sc_obos_state_final", 0))
    grade = _int(row.get("sc_obos_grade", 0))
    div_state = _int(row.get("sc_div_state", 0))
    stretch_state = _int(row.get("sc_stretch_state", 0))
    obos_dir = _int(row.get("sc_obos_dir", 0))
    mtf_dir = _int(row.get("sc_mtf_dir", 0))

    payload: Dict[str, Any] = {
        "debug_version": "ob_os_payload_v1",
        "indicator": "o_ob_os",
        "title": "OB/OS + Divergence Engine",
        "timestamp": timestamp_iso,

        "composite": round(_num(row.get("sc_obos_composite")), 6),
        "signal": round(_num(row.get("sc_obos_signal")), 6),
        "mtf_avg": round(_num(row.get("sc_mtf_avg")), 6),

        "state": state,
        "state_text": _state_text(state),
        "grade": grade,
        "grade_text": _grade_text(grade),

        "obos_state": _int(row.get("sc_obos_state", 0)),
        "stretch_state": stretch_state,
        "stretch_text": _stretch_text(stretch_state),

        "div_state": div_state,
        "div_text": _div_text(div_state),

        "obos_dir": obos_dir,
        "obos_dir_text": _dir_text(obos_dir),
        "mtf_dir": mtf_dir,
        "mtf_dir_text": _dir_text(mtf_dir),

        "active_score": round(_num(row.get("sc_obos_active_score")), 6),
        "bull_reversal_score": round(_num(row.get("sc_bull_reversal_score")), 6),
        "bear_reversal_score": round(_num(row.get("sc_bear_reversal_score")), 6),
        "bull_continuation_score": round(_num(row.get("sc_bull_continuation_score")), 6),
        "bear_continuation_score": round(_num(row.get("sc_bear_continuation_score")), 6),
        "exhaustion_score": round(_num(row.get("sc_exhaustion_score")), 6),

        "reversal_strength": round(_num(row.get("sc_reversal_strength")), 6),
        "continuation_strength": round(_num(row.get("sc_continuation_strength")), 6),
        "stretch_score": round(_num(row.get("sc_stretch_score")), 6),
        "bull_pressure": round(_num(row.get("sc_bull_pressure")), 6),
        "bear_pressure": round(_num(row.get("sc_bear_pressure")), 6),

        "bullish_div": _bool(row.get("sc_bullish_div")),
        "bearish_div": _bool(row.get("sc_bearish_div")),
        "hidden_bullish_div": _bool(row.get("sc_hidden_bullish_div")),
        "hidden_bearish_div": _bool(row.get("sc_hidden_bearish_div")),
        "bull_div_strength": round(_num(row.get("sc_bull_div_strength")), 6),
        "bear_div_strength": round(_num(row.get("sc_bear_div_strength")), 6),

        "is_ob": _bool(row.get("sc_is_ob")),
        "is_os": _bool(row.get("sc_is_os")),
        "is_extreme_ob": _bool(row.get("sc_is_extreme_ob")),
        "is_extreme_os": _bool(row.get("sc_is_extreme_os")),

        "bull_reversal_ready": _bool(row.get("sc_bull_reversal_ready")),
        "bear_reversal_ready": _bool(row.get("sc_bear_reversal_ready")),
        "bull_trend_exhaustion_risk": _bool(row.get("sc_bull_trend_exhaustion_risk")),
        "bear_trend_exhaustion_risk": _bool(row.get("sc_bear_trend_exhaustion_risk")),

        "mtf_aligned_bull": _bool(row.get("sc_mtf_aligned_bull")),
        "mtf_aligned_bear": _bool(row.get("sc_mtf_aligned_bear")),
        "mtf_extreme_bull": _bool(row.get("sc_mtf_extreme_bull")),
        "mtf_extreme_bear": _bool(row.get("sc_mtf_extreme_bear")),

        "config": {
            "rsi_len": cfg.rsi_len,
            "stoch_len": cfg.stoch_len,
            "stoch_smooth": cfg.stoch_smooth,
            "mfi_len": cfg.mfi_len,
            "cci_len": cfg.cci_len,
            "sig_smooth_len": cfg.sig_smooth_len,
            "ob_level": cfg.ob_level,
            "os_level": cfg.os_level,
            "extreme_ob_level": cfg.extreme_ob_level,
            "extreme_os_level": cfg.extreme_os_level,
            "div_left": cfg.div_left,
            "div_right": cfg.div_right,
            "use_hidden_div": cfg.use_hidden_div,
            "require_ob_os_for_div": cfg.require_ob_os_for_div,
            "mtf_on": cfg.mtf_on,
            "tf1": cfg.tf1,
            "tf2": cfg.tf2,
            "tf3": cfg.tf3,
            "tf4": cfg.tf4,
            "w1": cfg.w1,
            "w2": cfg.w2,
            "w3": cfg.w3,
            "w4": cfg.w4,
            "trigger_cross_lookback": cfg.trigger_cross_lookback,
            "reversal_gate_level": cfg.reversal_gate_level,
            "trend_gate_level": cfg.trend_gate_level,
            "min_reversal_strength": cfg.min_reversal_strength,
        },
    }

    return payload