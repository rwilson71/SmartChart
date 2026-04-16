"""
SmartChart Backend — i_liquidity.py

Master Liquidity Engine
Clean Python parity rebuild from merged Pine authority.

Merged parity source:
- Liquidity Engine v1
- Liquidity Engine v2
- Master merged logic

Core features:
- Pivot-based liquidity levels
- ATR-based liquidity zones
- Sweep / reclaim logic
- Sweep quality scoring
- EMA-distance filter
- RSI OB/OS + divergence layer
- Displacement engine
- Premium / Discount / Equilibrium state
- Structural MTF trend alignment
- Weighted MTF liquidity-state agreement
- Final liquidity state engine
- SmartChart-ready output contract

Production rules:
- Pine authority first
- No TradingView visuals
- No test block inside core
- Payload builder in this file is the single source of truth
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_LIQUIDITY_CONFIG: Dict[str, Any] = {
    # Core
    "pivot_left": 3,
    "pivot_right": 3,
    "atr_len": 14,
    "zone_atr_mult": 0.20,
    "reclaim_bars": 2,
    "sweep_mem_bars": 12,
    "reclaim_mem_bars": 8,
    "cluster_window": 20,

    # Zones
    "pd_lookback": 96,
    "eq_buffer_atr_mult": 0.05,

    # Quality / OBOS / Divergence
    "liq_vol_len": 20,
    "liq_vol_mult": 1.2,
    "rsi_len": 14,
    "ob_level": 70,
    "os_level": 30,
    "div_pivot_left": 3,
    "div_pivot_right": 3,
    "liq_use_ema_dist": True,
    "liq_ema_len": 20,
    "liq_min_dist_pct": 0.10,

    # Displacement
    "disp_atr_len": 14,
    "disp_range_mult": 1.20,
    "disp_body_frac_min": 0.60,
    "vol_confirm_mult": 1.10,

    # Structural trend
    "struct_fast_len": 20,
    "struct_slow_len": 50,

    # Weighted liquidity MTF
    "liq_mtf_bull_thr": 0.20,
    "liq_mtf_bear_thr": -0.20,
    "mtf_w1": 1.0,
    "mtf_w2": 1.0,
    "mtf_w3": 1.0,
    "mtf_w4": 1.0,
}


# =============================================================================
# HELPERS
# =============================================================================

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    avg_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_up / avg_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(left, len(vals) - right):
        window = vals[i - left:i + right + 1]
        center = vals[i]
        if np.isfinite(center) and center == np.nanmax(window):
            if np.sum(np.isclose(window, center, equal_nan=False)) == 1:
                out[i + right] = center

    return pd.Series(out, index=series.index)


def _pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(left, len(vals) - right):
        window = vals[i - left:i + right + 1]
        center = vals[i]
        if np.isfinite(center) and center == np.nanmin(window):
            if np.sum(np.isclose(window, center, equal_nan=False)) == 1:
                out[i + right] = center

    return pd.Series(out, index=series.index)


def _barssince(condition: pd.Series) -> pd.Series:
    out = np.full(len(condition), np.nan, dtype=float)
    last_true = None
    vals = condition.fillna(False).to_numpy(dtype=bool)

    for i, v in enumerate(vals):
        if v:
            last_true = i
            out[i] = 0.0
        elif last_true is not None:
            out[i] = float(i - last_true)

    return pd.Series(out, index=condition.index)


def _clamp(series: pd.Series | float, lo: float, hi: float) -> pd.Series | float:
    return np.minimum(np.maximum(series, lo), hi)


def _safe_div(a: pd.Series | float, b: pd.Series | float, fill: float = 0.0):
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a_s = a if isinstance(a, pd.Series) else pd.Series(a, index=b.index)  # type: ignore[arg-type]
        b_s = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)  # type: ignore[arg-type]
        out = a_s / b_s.replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan).fillna(fill)

    if b == 0:
        return fill
    return a / b


def _rolling_prev_day_high_low(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Approximation of Pine request.security(..., "D", high[1]/low[1]) using
    the dataframe's DatetimeIndex. Falls back gracefully if index is not datetime.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        prev_high = df["high"].rolling(1440, min_periods=1).max().shift(1)
        prev_low = df["low"].rolling(1440, min_periods=1).min().shift(1)
        return prev_high, prev_low

    day_key = df.index.normalize()
    day_high = df.groupby(day_key)["high"].transform("max")
    day_low = df.groupby(day_key)["low"].transform("min")

    daily = pd.DataFrame({"day_high": day_high, "day_low": day_low}, index=df.index)
    daily_unique = daily.groupby(day_key).first()
    prev_daily = daily_unique.shift(1)

    prev_high = day_key.map(prev_daily["day_high"])
    prev_low = day_key.map(prev_daily["day_low"])

    return (
        pd.Series(prev_high, index=df.index, dtype=float),
        pd.Series(prev_low, index=df.index, dtype=float),
    )


def _state_text_from_dir_and_scores(
    liq_dir: pd.Series,
    bull_reversal: pd.Series,
    bear_reversal: pd.Series,
    bull_breakout: pd.Series,
    bear_breakout: pd.Series,
    bull_sweep_active: pd.Series,
    bear_sweep_active: pd.Series,
) -> pd.Series:
    state = pd.Series("neutral", index=liq_dir.index, dtype=object)
    state = state.mask(bull_reversal, "bull_reversal")
    state = state.mask(bear_reversal, "bear_reversal")
    state = state.mask(bull_breakout, "bull_breakout")
    state = state.mask(bear_breakout, "bear_breakout")
    state = state.mask(bull_sweep_active, "bull_sweep_active")
    state = state.mask(bear_sweep_active, "bear_sweep_active")
    return state


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def _dir_text(v: int) -> str:
    return "Bull" if v > 0 else "Bear" if v < 0 else "Neutral"


def _pd_text(pd_state: int) -> str:
    return "Discount" if pd_state > 0 else "Premium" if pd_state < 0 else "Equilibrium"


def _obos_div_text(state: int) -> str:
    if state == 2:
        return "Bull Peak"
    if state == -2:
        return "Bear Peak"
    if state == 1:
        return "Bull OBOS"
    if state == -1:
        return "Bear OBOS"
    return "Neutral"


def _agreement_text(v: float, bull_thr: float, bear_thr: float) -> str:
    if v > bull_thr:
        return "Bull"
    if v < bear_thr:
        return "Bear"
    return "Neutral"


# =============================================================================
# V1-STYLE LOCAL LIQUIDITY STATE FOR MTF CONFIRMATION
# =============================================================================

def _compute_simple_liq_state(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Lightweight parity helper for weighted MTF liquidity agreement.
    Mirrors the v1-style helper used inside request.security in Pine.
    """
    out = pd.DataFrame(index=df.index)

    pv_lo = _pivot_low(df["low"], cfg["pivot_left"], cfg["pivot_right"])
    pv_hi = _pivot_high(df["high"], cfg["pivot_left"], cfg["pivot_right"])

    last_lo = pv_lo.ffill()
    last_hi = pv_hi.ffill()

    liq_ema = _ema(df["close"], cfg["liq_ema_len"])
    bull_ema_dist_pct = _safe_div((liq_ema - df["low"]) * 100.0, liq_ema, fill=0.0)
    bear_ema_dist_pct = _safe_div((df["high"] - liq_ema) * 100.0, liq_ema, fill=0.0)

    bull_ema_ok = bull_ema_dist_pct >= cfg["liq_min_dist_pct"]
    bear_ema_ok = bear_ema_dist_pct >= cfg["liq_min_dist_pct"]

    vol_avg = _sma(df["volume"], cfg["liq_vol_len"])
    high_vol = (vol_avg > 0) & (df["volume"] > vol_avg * cfg["liq_vol_mult"])

    swept_below = (last_lo.notna()) & (df["low"] < last_lo)
    swept_above = (last_hi.notna()) & (df["high"] > last_hi)

    reclaimed_lo = (last_lo.notna()) & (df["close"] > last_lo)
    reclaimed_hi = (last_hi.notna()) & (df["close"] < last_hi)

    bull_lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    bear_upper_wick = df["high"] - np.maximum(df["open"], df["close"])

    bull_wick_reject = (bull_lower_wick > 0) & (
        bull_lower_wick >= (df["close"] - df["open"]).abs() * 0.5
    )
    bear_wick_reject = (bear_upper_wick > 0) & (
        bear_upper_wick >= (df["close"] - df["open"]).abs() * 0.5
    )

    hl2 = (df["high"] + df["low"]) / 2.0
    bull_close_intent = (df["close"] > df["open"]) | (df["close"] > hl2)
    bear_close_intent = (df["close"] < df["open"]) | (df["close"] < hl2)

    rsi_value = _rsi(df["close"], cfg["rsi_len"])
    is_oversold = rsi_value <= cfg["os_level"]
    is_overbought = rsi_value >= cfg["ob_level"]

    px_lo = _pivot_low(df["low"], cfg["div_pivot_left"], cfg["div_pivot_right"])
    px_hi = _pivot_high(df["high"], cfg["div_pivot_left"], cfg["div_pivot_right"])
    rsi_lo = _pivot_low(rsi_value, cfg["div_pivot_left"], cfg["div_pivot_right"])
    rsi_hi = _pivot_high(rsi_value, cfg["div_pivot_left"], cfg["div_pivot_right"])

    prev_px_lo = px_lo.ffill().shift(1)
    prev_rsi_lo = rsi_lo.ffill().shift(1)
    prev_px_hi = px_hi.ffill().shift(1)
    prev_rsi_hi = rsi_hi.ffill().shift(1)

    bull_div = (
        px_lo.notna()
        & rsi_lo.notna()
        & prev_px_lo.notna()
        & prev_rsi_lo.notna()
        & (px_lo < prev_px_lo)
        & (rsi_lo > prev_rsi_lo)
    )
    bear_div = (
        px_hi.notna()
        & rsi_hi.notna()
        & prev_px_hi.notna()
        & prev_rsi_hi.notna()
        & (px_hi > prev_px_hi)
        & (rsi_hi < prev_rsi_hi)
    )

    raw_bull = swept_below & reclaimed_lo & ((not cfg["liq_use_ema_dist"]) | bull_ema_ok)
    raw_bear = swept_above & reclaimed_hi & ((not cfg["liq_use_ema_dist"]) | bear_ema_ok)

    bull_qual = (
        raw_bull.astype(int)
        + high_vol.astype(int)
        + (bull_wick_reject | bull_close_intent).astype(int)
        + (raw_bull & is_oversold).astype(int)
        + (raw_bull & is_oversold & bull_div).astype(int)
    )

    bear_qual = (
        raw_bear.astype(int)
        + high_vol.astype(int)
        + (bear_wick_reject | bear_close_intent).astype(int)
        + (raw_bear & is_overbought).astype(int)
        + (raw_bear & is_overbought & bear_div).astype(int)
    )

    bull_sweep = raw_bull & (bull_qual >= 2) & ~(raw_bull & (bull_qual >= 2)).shift(1).fillna(False)
    bear_sweep = raw_bear & (bear_qual >= 2) & ~(raw_bear & (bear_qual >= 2)).shift(1).fillna(False)

    bull_recent = _barssince(bull_sweep) <= cfg["sweep_mem_bars"]
    bear_recent = _barssince(bear_sweep) <= cfg["sweep_mem_bars"]

    bull_age = _barssince(bull_sweep).fillna(100000.0)
    bear_age = _barssince(bear_sweep).fillna(100000.0)

    liq_state = pd.Series(0, index=df.index, dtype=int)
    liq_state = liq_state.mask(bull_recent & ~bear_recent, 1)
    liq_state = liq_state.mask(bear_recent & ~bull_recent, -1)
    liq_state = liq_state.mask(bull_recent & bear_recent & (bull_age < bear_age), 1)
    liq_state = liq_state.mask(bull_recent & bear_recent & (bear_age < bull_age), -1)

    liq_sweep_state = pd.Series(0, index=df.index, dtype=int)
    liq_sweep_state = liq_sweep_state.mask(bull_sweep, 1)
    liq_sweep_state = liq_sweep_state.mask(bear_sweep, -1)

    last_bull_level = last_lo.where(bull_sweep).ffill()
    last_bear_level = last_hi.where(bear_sweep).ffill()

    level = pd.Series(np.nan, index=df.index, dtype=float)
    level = level.mask(liq_state == 1, last_bull_level)
    level = level.mask(liq_state == -1, last_bear_level)

    qual = pd.Series(0, index=df.index, dtype=int)
    qual = qual.mask(liq_sweep_state == 1, bull_qual)
    qual = qual.mask(liq_sweep_state == -1, bear_qual)
    qual = qual.mask(
        (liq_sweep_state == 0) & (liq_state == 1),
        bull_qual.where(bull_sweep).ffill().fillna(0).astype(int),
    )
    qual = qual.mask(
        (liq_sweep_state == 0) & (liq_state == -1),
        bear_qual.where(bear_sweep).ffill().fillna(0).astype(int),
    )

    out["liq_state"] = liq_state
    out["liq_sweep_state"] = liq_sweep_state
    out["liq_strength"] = qual
    out["liq_level"] = level
    return out


# =============================================================================
# MAIN ENGINE
# =============================================================================

def compute_liquidity(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    mtf_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Compute SmartChart Master Liquidity Engine parity on a dataframe.

    Required columns:
        open, high, low, close, volume

    Optional mtf_frames keys:
        "struct" : dataframe for structural MTF trend
        "tf1", "tf2", "tf3", "tf4" : dataframes for weighted liquidity MTF

    Returns:
        DataFrame with full intermediate and final output fields.
    """
    cfg = {**DEFAULT_LIQUIDITY_CONFIG, **(config or {})}

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    # -------------------------------------------------------------------------
    # Base calcs
    # -------------------------------------------------------------------------
    out["atr_val"] = _atr(out, cfg["atr_len"])
    out["atr_disp"] = _atr(out, cfg["disp_atr_len"])

    out["ema20"] = _ema(out["close"], 20)
    out["ema50"] = _ema(out["close"], 50)
    out["ema200"] = _ema(out["close"], 200)
    out["liq_ema"] = _ema(out["close"], cfg["liq_ema_len"])

    out["vol_ma"] = _sma(out["volume"], cfg["liq_vol_len"])
    out["vol_ratio"] = _safe_div(out["volume"], out["vol_ma"], fill=1.0)
    out["high_vol"] = (out["vol_ma"] > 0) & (out["volume"] > out["vol_ma"] * cfg["liq_vol_mult"])

    out["rng"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_frac"] = _safe_div(out["body"], out["rng"], fill=0.0)
    out["close_pos"] = _safe_div(out["close"] - out["low"], out["rng"], fill=0.5)
    out["wick_top_frac"] = _safe_div(
        out["high"] - np.maximum(out["open"], out["close"]),
        out["rng"],
        fill=0.0,
    )
    out["wick_bot_frac"] = _safe_div(
        np.minimum(out["open"], out["close"]) - out["low"],
        out["rng"],
        fill=0.0,
    )

    out["trend_up"] = (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"])
    out["trend_dn"] = (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"])
    out["trend_state"] = np.select(
        [out["trend_up"], out["trend_dn"]],
        [1, -1],
        default=0,
    )

    # -------------------------------------------------------------------------
    # Pivots / liquidity levels
    # -------------------------------------------------------------------------
    out["ph"] = _pivot_high(out["high"], cfg["pivot_left"], cfg["pivot_right"])
    out["pl"] = _pivot_low(out["low"], cfg["pivot_left"], cfg["pivot_right"])

    out["last_ph"] = out["ph"].ffill()
    out["last_pl"] = out["pl"].ffill()

    out["zone_tol"] = out["atr_val"] * cfg["zone_atr_mult"]

    out["liq_high_top"] = out["last_ph"] + out["zone_tol"]
    out["liq_high_bot"] = out["last_ph"] - out["zone_tol"]
    out["liq_low_top"] = out["last_pl"] + out["zone_tol"]
    out["liq_low_bot"] = out["last_pl"] - out["zone_tol"]

    # -------------------------------------------------------------------------
    # Cluster strength
    # -------------------------------------------------------------------------
    ph_cluster_hit = (
        out["ph"].notna()
        & out["last_ph"].notna()
        & ((out["ph"] - out["last_ph"]).abs() <= out["zone_tol"])
    )
    pl_cluster_hit = (
        out["pl"].notna()
        & out["last_pl"].notna()
        & ((out["pl"] - out["last_pl"]).abs() <= out["zone_tol"])
    )

    out["ph_cluster_strength"] = (
        ph_cluster_hit.astype(float).rolling(cfg["cluster_window"], min_periods=1).sum()
    )
    out["pl_cluster_strength"] = (
        pl_cluster_hit.astype(float).rolling(cfg["cluster_window"], min_periods=1).sum()
    )

    # -------------------------------------------------------------------------
    # Sweep / reclaim structure
    # -------------------------------------------------------------------------
    out["sweep_high"] = (
        out["last_ph"].notna()
        & (out["high"] > out["liq_high_top"])
        & (out["close"] < out["last_ph"])
        & (out["wick_top_frac"] >= 0.20)
        & (out["body_frac"] >= 0.15)
    )

    out["sweep_low"] = (
        out["last_pl"].notna()
        & (out["low"] < out["liq_low_bot"])
        & (out["close"] > out["last_pl"])
        & (out["wick_bot_frac"] >= 0.20)
        & (out["body_frac"] >= 0.15)
    )

    out["bars_since_sweep_high"] = _barssince(out["sweep_high"])
    out["bars_since_sweep_low"] = _barssince(out["sweep_low"])

    out["recent_sweep_high"] = out["bars_since_sweep_high"] <= cfg["reclaim_bars"]
    out["recent_sweep_low"] = out["bars_since_sweep_low"] <= cfg["reclaim_bars"]

    out["reclaim_high_now"] = (
        out["last_ph"].notna()
        & (out["close"] < out["last_ph"])
        & (out["close_pos"] <= 0.40)
    )
    out["reclaim_low_now"] = (
        out["last_pl"].notna()
        & (out["close"] > out["last_pl"])
        & (out["close_pos"] >= 0.60)
    )

    out["reclaim_high"] = out["recent_sweep_high"] & out["reclaim_high_now"]
    out["reclaim_low"] = out["recent_sweep_low"] & out["reclaim_low_now"]

    out["sweep_high_mem"] = out["bars_since_sweep_high"] <= cfg["sweep_mem_bars"]
    out["sweep_low_mem"] = out["bars_since_sweep_low"] <= cfg["sweep_mem_bars"]

    out["bars_since_reclaim_high"] = _barssince(out["reclaim_high"])
    out["bars_since_reclaim_low"] = _barssince(out["reclaim_low"])

    out["reclaim_high_mem"] = out["bars_since_reclaim_high"] <= cfg["reclaim_mem_bars"]
    out["reclaim_low_mem"] = out["bars_since_reclaim_low"] <= cfg["reclaim_mem_bars"]

    # -------------------------------------------------------------------------
    # V1-style sweep quality / EMA dist / OBOS / divergence
    # -------------------------------------------------------------------------
    out["bull_ema_dist_pct"] = _safe_div(
        (out["liq_ema"] - out["low"]) * 100.0,
        out["liq_ema"],
        fill=0.0,
    )
    out["bear_ema_dist_pct"] = _safe_div(
        (out["high"] - out["liq_ema"]) * 100.0,
        out["liq_ema"],
        fill=0.0,
    )

    out["bull_ema_dist_ok"] = out["bull_ema_dist_pct"] >= cfg["liq_min_dist_pct"]
    out["bear_ema_dist_ok"] = out["bear_ema_dist_pct"] >= cfg["liq_min_dist_pct"]

    bull_lower_wick = np.minimum(out["open"], out["close"]) - out["low"]
    bear_upper_wick = out["high"] - np.maximum(out["open"], out["close"])
    hl2 = (out["high"] + out["low"]) / 2.0

    out["bull_wick_reject"] = (bull_lower_wick > 0) & (bull_lower_wick >= out["body"] * 0.5)
    out["bear_wick_reject"] = (bear_upper_wick > 0) & (bear_upper_wick >= out["body"] * 0.5)

    out["bull_close_intent"] = (out["close"] > out["open"]) | (out["close"] > hl2)
    out["bear_close_intent"] = (out["close"] < out["open"]) | (out["close"] < hl2)

    out["rsi_value"] = _rsi(out["close"], cfg["rsi_len"])
    out["is_oversold"] = out["rsi_value"] <= cfg["os_level"]
    out["is_overbought"] = out["rsi_value"] >= cfg["ob_level"]

    out["px_lo"] = _pivot_low(out["low"], cfg["div_pivot_left"], cfg["div_pivot_right"])
    out["px_hi"] = _pivot_high(out["high"], cfg["div_pivot_left"], cfg["div_pivot_right"])
    out["rsi_lo"] = _pivot_low(out["rsi_value"], cfg["div_pivot_left"], cfg["div_pivot_right"])
    out["rsi_hi"] = _pivot_high(out["rsi_value"], cfg["div_pivot_left"], cfg["div_pivot_right"])

    prev_px_lo = out["px_lo"].ffill().shift(1)
    prev_rsi_lo = out["rsi_lo"].ffill().shift(1)
    prev_px_hi = out["px_hi"].ffill().shift(1)
    prev_rsi_hi = out["rsi_hi"].ffill().shift(1)

    out["bullish_div"] = (
        out["px_lo"].notna()
        & out["rsi_lo"].notna()
        & prev_px_lo.notna()
        & prev_rsi_lo.notna()
        & (out["px_lo"] < prev_px_lo)
        & (out["rsi_lo"] > prev_rsi_lo)
    )

    out["bearish_div"] = (
        out["px_hi"].notna()
        & out["rsi_hi"].notna()
        & prev_px_hi.notna()
        & prev_rsi_hi.notna()
        & (out["px_hi"] > prev_px_hi)
        & (out["rsi_hi"] < prev_rsi_hi)
    )

    ema_gate_bull = (
        out["bull_ema_dist_ok"]
        if cfg["liq_use_ema_dist"]
        else pd.Series(True, index=out.index)
    )
    ema_gate_bear = (
        out["bear_ema_dist_ok"]
        if cfg["liq_use_ema_dist"]
        else pd.Series(True, index=out.index)
    )

    out["raw_bull_sweep_qual"] = out["sweep_low"] & out["reclaim_low_now"] & ema_gate_bull
    out["raw_bear_sweep_qual"] = out["sweep_high"] & out["reclaim_high_now"] & ema_gate_bear

    out["bull_sweep_obos"] = out["raw_bull_sweep_qual"] & out["is_oversold"]
    out["bear_sweep_obos"] = out["raw_bear_sweep_qual"] & out["is_overbought"]

    out["bull_sweep_peak"] = (
        out["raw_bull_sweep_qual"] & out["is_oversold"] & out["bullish_div"]
    )
    out["bear_sweep_peak"] = (
        out["raw_bear_sweep_qual"] & out["is_overbought"] & out["bearish_div"]
    )

    out["bull_qual"] = (
        out["raw_bull_sweep_qual"].astype(int)
        + out["high_vol"].astype(int)
        + (out["bull_wick_reject"] | out["bull_close_intent"]).astype(int)
        + out["bull_sweep_obos"].astype(int)
        + out["bull_sweep_peak"].astype(int)
    )

    out["bear_qual"] = (
        out["raw_bear_sweep_qual"].astype(int)
        + out["high_vol"].astype(int)
        + (out["bear_wick_reject"] | out["bear_close_intent"]).astype(int)
        + out["bear_sweep_obos"].astype(int)
        + out["bear_sweep_peak"].astype(int)
    )

    out["bull_quality_ok"] = out["raw_bull_sweep_qual"] & (out["bull_qual"] >= 2)
    out["bear_quality_ok"] = out["raw_bear_sweep_qual"] & (out["bear_qual"] >= 2)

    out["quality_score_bull"] = _clamp((out["bull_qual"] / 5.0) * 100.0, 0.0, 100.0)
    out["quality_score_bear"] = _clamp((out["bear_qual"] / 5.0) * 100.0, 0.0, 100.0)

    out["obos_div_state"] = np.select(
        [
            out["bull_sweep_obos"] & out["bullish_div"],
            out["bear_sweep_obos"] & out["bearish_div"],
            out["bull_sweep_obos"],
            out["bear_sweep_obos"],
        ],
        [2, -2, 1, -1],
        default=0,
    )

    # -------------------------------------------------------------------------
    # Event memory for quality sweeps
    # -------------------------------------------------------------------------
    out["bull_sweep_event"] = out["bull_quality_ok"] & ~out["bull_quality_ok"].shift(1).fillna(False)
    out["bear_sweep_event"] = out["bear_quality_ok"] & ~out["bear_quality_ok"].shift(1).fillna(False)

    out["last_bull_sweep_level"] = out["last_pl"].where(out["bull_sweep_event"]).ffill()
    out["last_bear_sweep_level"] = out["last_ph"].where(out["bear_sweep_event"]).ffill()

    bull_recent = _barssince(out["bull_sweep_event"]) <= cfg["sweep_mem_bars"]
    bear_recent = _barssince(out["bear_sweep_event"]) <= cfg["sweep_mem_bars"]
    bull_age = _barssince(out["bull_sweep_event"]).fillna(100000.0)
    bear_age = _barssince(out["bear_sweep_event"]).fillna(100000.0)

    out["bull_recent"] = bull_recent
    out["bear_recent"] = bear_recent

    out["liq_bias_mem"] = np.select(
        [
            bull_recent & ~bear_recent,
            bear_recent & ~bull_recent,
            bull_recent & bear_recent & (bull_age < bear_age),
            bull_recent & bear_recent & (bear_age < bull_age),
        ],
        [1, -1, 1, -1],
        default=0,
    )

    out["bull_qual_mem"] = out["bull_qual"].where(out["bull_sweep_event"]).ffill().fillna(0)
    out["bear_qual_mem"] = out["bear_qual"].where(out["bear_sweep_event"]).ffill().fillna(0)

    out["liq_bias_qual"] = np.select(
        [out["liq_bias_mem"] == 1, out["liq_bias_mem"] == -1],
        [out["bull_qual_mem"], out["bear_qual_mem"]],
        default=0.0,
    )

    out["liq_bias_level"] = np.select(
        [out["liq_bias_mem"] == 1, out["liq_bias_mem"] == -1],
        [out["last_bull_sweep_level"], out["last_bear_sweep_level"]],
        default=np.nan,
    )

    # -------------------------------------------------------------------------
    # Displacement
    # -------------------------------------------------------------------------
    out["disp_up_raw"] = (
        (out["close"] > out["open"])
        & (out["rng"] > out["atr_disp"] * cfg["disp_range_mult"])
        & (out["body_frac"] >= cfg["disp_body_frac_min"])
        & (out["close"] > out["ema20"])
    )

    out["disp_dn_raw"] = (
        (out["close"] < out["open"])
        & (out["rng"] > out["atr_disp"] * cfg["disp_range_mult"])
        & (out["body_frac"] >= cfg["disp_body_frac_min"])
        & (out["close"] < out["ema20"])
    )

    out["vol_confirm"] = out["vol_ratio"] >= cfg["vol_confirm_mult"]
    out["displacement_up"] = out["disp_up_raw"] & out["vol_confirm"]
    out["displacement_dn"] = out["disp_dn_raw"] & out["vol_confirm"]

    out["disp_strength"] = (
        50.0 * _clamp(
            _safe_div(out["rng"], out["atr_disp"] * cfg["disp_range_mult"], fill=0.0),
            0.0,
            1.0,
        )
        + 25.0 * _clamp(
            _safe_div(out["body_frac"], cfg["disp_body_frac_min"], fill=0.0),
            0.0,
            1.0,
        )
        + 25.0 * _clamp(
            _safe_div(out["vol_ratio"], cfg["vol_confirm_mult"], fill=0.0),
            0.0,
            1.0,
        )
    )
    out["disp_strength_clamped"] = np.minimum(100.0, out["disp_strength"])
    out["disp_dir"] = np.select(
        [out["displacement_up"], out["displacement_dn"]],
        [1, -1],
        default=0,
    )

    # -------------------------------------------------------------------------
    # Premium / Discount / Equilibrium
    # -------------------------------------------------------------------------
    prev_day_high, prev_day_low = _rolling_prev_day_high_low(out)
    out["prev_day_high"] = prev_day_high
    out["prev_day_low"] = prev_day_low

    out["fb_high"] = out["high"].shift(1).rolling(cfg["pd_lookback"], min_periods=1).max()
    out["fb_low"] = out["low"].shift(1).rolling(cfg["pd_lookback"], min_periods=1).min()

    out["range_high"] = out["prev_day_high"].fillna(out["fb_high"])
    out["range_low"] = out["prev_day_low"].fillna(out["fb_low"])

    out["equilibrium"] = (out["range_high"] + out["range_low"]) / 2.0
    out["eq_buffer"] = out["atr_val"] * cfg["eq_buffer_atr_mult"]

    out["in_premium"] = out["close"] > out["equilibrium"] + out["eq_buffer"]
    out["in_discount"] = out["close"] < out["equilibrium"] - out["eq_buffer"]
    out["in_equilibrium"] = ~out["in_premium"] & ~out["in_discount"]

    out["pd_state"] = np.select(
        [out["in_discount"], out["in_premium"]],
        [1, -1],
        default=0,
    )

    # -------------------------------------------------------------------------
    # Structural MTF alignment
    # -------------------------------------------------------------------------
    struct_df = None if not mtf_frames else mtf_frames.get("struct")
    if struct_df is not None:
        struct_fast = _ema(struct_df["close"], cfg["struct_fast_len"])
        struct_slow = _ema(struct_df["close"], cfg["struct_slow_len"])
        struct_state = pd.Series(
            np.select(
                [struct_fast > struct_slow, struct_fast < struct_slow],
                [1, -1],
                default=0,
            ),
            index=struct_df.index,
        )
        out["mtf_trend_state"] = struct_state.reindex(out.index, method="ffill").fillna(0).astype(int)
    else:
        out["mtf_trend_state"] = 0

    out["mtf_align_long"] = out["mtf_trend_state"] == 1
    out["mtf_align_short"] = out["mtf_trend_state"] == -1

    # -------------------------------------------------------------------------
    # Weighted liquidity MTF agreement
    # -------------------------------------------------------------------------
    mtf_keys = ["tf1", "tf2", "tf3", "tf4"]
    mtf_weights = [cfg["mtf_w1"], cfg["mtf_w2"], cfg["mtf_w3"], cfg["mtf_w4"]]

    liq_mtf_series: list[pd.Series] = []
    used_weights: list[float] = []

    if mtf_frames:
        for key, weight in zip(mtf_keys, mtf_weights):
            mtf_df = mtf_frames.get(key)
            if mtf_df is None or weight <= 0:
                continue

            liq_state_df = _compute_simple_liq_state(mtf_df, cfg)
            aligned = liq_state_df["liq_state"].reindex(out.index, method="ffill").fillna(0.0)
            liq_mtf_series.append(aligned)
            used_weights.append(weight)

    if liq_mtf_series and sum(used_weights) > 0:
        weighted_sum = sum(s * w for s, w in zip(liq_mtf_series, used_weights))
        weight_sum = float(sum(used_weights))
        out["liq_mtf_avg_dir"] = weighted_sum / max(weight_sum, 1e-6)
    else:
        out["liq_mtf_avg_dir"] = 0.0

    out["liq_mtf_agreement_score"] = out["liq_mtf_avg_dir"].abs()
    out["liq_mtf_agreement_dir"] = np.select(
        [
            out["liq_mtf_avg_dir"] > cfg["liq_mtf_bull_thr"],
            out["liq_mtf_avg_dir"] < cfg["liq_mtf_bear_thr"],
        ],
        [1, -1],
        default=0,
    )

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------
    out["zone_long_score"] = np.select(
        [out["in_discount"], out["in_equilibrium"]],
        [100.0, 45.0],
        default=10.0,
    )
    out["zone_short_score"] = np.select(
        [out["in_premium"], out["in_equilibrium"]],
        [100.0, 45.0],
        default=10.0,
    )

    out["sweep_long_score"] = np.where(
        out["sweep_low_mem"],
        np.where(out["bull_quality_ok"], 100.0, 50.0),
        0.0,
    )
    out["sweep_short_score"] = np.where(
        out["sweep_high_mem"],
        np.where(out["bear_quality_ok"], 100.0, 50.0),
        0.0,
    )

    out["reclaim_long_score"] = np.where(out["reclaim_low_mem"], 100.0, 0.0)
    out["reclaim_short_score"] = np.where(out["reclaim_high_mem"], 100.0, 0.0)

    out["disp_long_score"] = np.where(
        out["displacement_up"],
        out["disp_strength_clamped"],
        out["disp_strength_clamped"] * 0.35,
    )
    out["disp_short_score"] = np.where(
        out["displacement_dn"],
        out["disp_strength_clamped"],
        out["disp_strength_clamped"] * 0.35,
    )

    out["mtf_struct_long_score"] = np.where(out["mtf_align_long"], 100.0, 0.0)
    out["mtf_struct_short_score"] = np.where(out["mtf_align_short"], 100.0, 0.0)

    out["liq_mtf_long_score"] = np.where(
        out["liq_mtf_agreement_dir"] == 1,
        out["liq_mtf_agreement_score"] * 100.0,
        0.0,
    )
    out["liq_mtf_short_score"] = np.where(
        out["liq_mtf_agreement_dir"] == -1,
        out["liq_mtf_agreement_score"] * 100.0,
        0.0,
    )

    out["quality_long_score"] = out["quality_score_bull"]
    out["quality_short_score"] = out["quality_score_bear"]

    out["obos_div_long_score"] = np.select(
        [out["bull_sweep_peak"], out["bull_sweep_obos"], out["bullish_div"]],
        [100.0, 70.0, 50.0],
        default=0.0,
    )
    out["obos_div_short_score"] = np.select(
        [out["bear_sweep_peak"], out["bear_sweep_obos"], out["bearish_div"]],
        [100.0, 70.0, 50.0],
        default=0.0,
    )

    w_sweep = 18.0
    w_reclaim = 18.0
    w_disp = 16.0
    w_zone = 12.0
    w_struct_mtf = 10.0
    w_liq_mtf = 10.0
    w_quality = 10.0
    w_obos_div = 6.0

    out["final_long_score"] = (
        w_sweep * (out["sweep_long_score"] / 100.0)
        + w_reclaim * (out["reclaim_long_score"] / 100.0)
        + w_disp * (out["disp_long_score"] / 100.0)
        + w_zone * (out["zone_long_score"] / 100.0)
        + w_struct_mtf * (out["mtf_struct_long_score"] / 100.0)
        + w_liq_mtf * (out["liq_mtf_long_score"] / 100.0)
        + w_quality * (out["quality_long_score"] / 100.0)
        + w_obos_div * (out["obos_div_long_score"] / 100.0)
    )

    out["final_short_score"] = (
        w_sweep * (out["sweep_short_score"] / 100.0)
        + w_reclaim * (out["reclaim_short_score"] / 100.0)
        + w_disp * (out["disp_short_score"] / 100.0)
        + w_zone * (out["zone_short_score"] / 100.0)
        + w_struct_mtf * (out["mtf_struct_short_score"] / 100.0)
        + w_liq_mtf * (out["liq_mtf_short_score"] / 100.0)
        + w_quality * (out["quality_short_score"] / 100.0)
        + w_obos_div * (out["obos_div_short_score"] / 100.0)
    )

    out["liq_long_ready"] = (
        out["sweep_low_mem"]
        & out["reclaim_low_mem"]
        & (out["zone_long_score"] >= 45.0)
        & (out["bull_qual"] >= 2)
        & (out["displacement_up"] | (out["disp_long_score"] >= 55.0))
    )

    out["liq_short_ready"] = (
        out["sweep_high_mem"]
        & out["reclaim_high_mem"]
        & (out["zone_short_score"] >= 45.0)
        & (out["bear_qual"] >= 2)
        & (out["displacement_dn"] | (out["disp_short_score"] >= 55.0))
    )

    out["liq_breakout_up"] = (
        out["last_ph"].notna()
        & (out["high"] > out["liq_high_top"])
        & (out["close"] > out["last_ph"])
        & ~out["reclaim_high_mem"]
        & out["displacement_up"]
    )

    out["liq_breakout_dn"] = (
        out["last_pl"].notna()
        & (out["low"] < out["liq_low_bot"])
        & (out["close"] < out["last_pl"])
        & ~out["reclaim_low_mem"]
        & out["displacement_dn"]
    )

    # -------------------------------------------------------------------------
    # Final state engine
    # -------------------------------------------------------------------------
    bull_reversal = out["liq_long_ready"] & (out["final_long_score"] >= 60.0)
    bear_reversal = out["liq_short_ready"] & (out["final_short_score"] >= 60.0)
    bull_breakout = out["liq_breakout_up"] & (out["final_long_score"] >= out["final_short_score"])
    bear_breakout = out["liq_breakout_dn"] & (out["final_short_score"] > out["final_long_score"])
    bull_sweep_active = (
        out["bull_recent"] & (out["final_long_score"] >= 45.0) & (out["liq_bias_mem"] == 1)
    )
    bear_sweep_active = (
        out["bear_recent"] & (out["final_short_score"] >= 45.0) & (out["liq_bias_mem"] == -1)
    )

    out["liq_dir"] = np.select(
        [
            bull_reversal,
            bear_reversal,
            bull_breakout,
            bear_breakout,
            bull_sweep_active,
            bear_sweep_active,
        ],
        [1, -1, 1, -1, 1, -1],
        default=0,
    )

    out["liq_state"] = _state_text_from_dir_and_scores(
        out["liq_dir"],
        bull_reversal,
        bear_reversal,
        bull_breakout,
        bear_breakout,
        bull_sweep_active,
        bear_sweep_active,
    )

    out["liq_final_score"] = np.where(
        out["liq_dir"] > 0,
        out["final_long_score"],
        np.where(
            out["liq_dir"] < 0,
            out["final_short_score"],
            np.maximum(out["final_long_score"], out["final_short_score"]),
        ),
    )

    out["active_level_price"] = np.select(
        [
            out["liq_dir"] > 0,
            out["liq_dir"] < 0,
            out["final_short_score"] > out["final_long_score"],
        ],
        [
            out["last_pl"],
            out["last_ph"],
            out["last_ph"],
        ],
        default=out["last_pl"],
    )

    out["active_level_type"] = np.select(
        [
            out["liq_dir"] > 0,
            out["liq_dir"] < 0,
            out["final_short_score"] > out["final_long_score"],
        ],
        [
            "Sellside",
            "Buyside",
            "Buyside",
        ],
        default="Sellside",
    )

    out["active_cluster_strength"] = np.select(
        [
            out["liq_dir"] > 0,
            out["liq_dir"] < 0,
            out["final_short_score"] > out["final_long_score"],
        ],
        [
            out["pl_cluster_strength"],
            out["ph_cluster_strength"],
            out["ph_cluster_strength"],
        ],
        default=out["pl_cluster_strength"],
    )

    out["liq_quality_score"] = np.where(
        out["liq_dir"] > 0,
        out["quality_score_bull"],
        np.where(
            out["liq_dir"] < 0,
            out["quality_score_bear"],
            np.maximum(out["quality_score_bull"], out["quality_score_bear"]),
        ),
    )

    # -------------------------------------------------------------------------
    # SmartChart-ready exports
    # -------------------------------------------------------------------------
    out["sc_liq_dir"] = out["liq_dir"]
    out["sc_liq_state_text"] = out["liq_state"]
    out["sc_liq_final_score"] = out["liq_final_score"]
    out["sc_liq_quality_score"] = out["liq_quality_score"]
    out["sc_liq_sweep_high_mem"] = out["sweep_high_mem"].astype(float)
    out["sc_liq_sweep_low_mem"] = out["sweep_low_mem"].astype(float)
    out["sc_liq_reclaim_high_mem"] = out["reclaim_high_mem"].astype(float)
    out["sc_liq_reclaim_low_mem"] = out["reclaim_low_mem"].astype(float)
    out["sc_liq_disp_dir"] = out["disp_dir"]
    out["sc_liq_disp_score"] = out["disp_strength_clamped"]
    out["sc_liq_pd_state"] = out["pd_state"]
    out["sc_liq_cluster_strength"] = out["active_cluster_strength"]
    out["sc_liq_active_level"] = out["active_level_price"]
    out["sc_liq_mtf_trend_dir"] = out["mtf_trend_state"]
    out["sc_liq_mtf_avg_dir"] = out["liq_mtf_avg_dir"]
    out["sc_liq_obos_div_state"] = out["obos_div_state"]
    out["sc_liq_bull_qual"] = out["bull_qual"]
    out["sc_liq_bear_qual"] = out["bear_qual"]
    out["sc_liq_memory_bias"] = out["liq_bias_mem"]

    return out


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_liquidity_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    mtf_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """
    Build latest SmartChart Liquidity payload for website/API use.

    This is the single source of truth for cache + router output.
    """
    result = compute_liquidity(df=df, config=config, mtf_frames=mtf_frames)

    if result.empty:
        raise ValueError("Liquidity build error: empty result dataframe")

    last = result.iloc[-1]
    cfg = {**DEFAULT_LIQUIDITY_CONFIG, **(config or {})}

    liq_dir = _safe_int(last.get("sc_liq_dir"), 0)
    pd_state = _safe_int(last.get("sc_liq_pd_state"), 0)
    obos_div_state = _safe_int(last.get("sc_liq_obos_div_state"), 0)
    mtf_avg_dir = _safe_float(last.get("sc_liq_mtf_avg_dir"), 0.0)

    payload: Dict[str, Any] = {
        "indicator": "liquidity",
        "name": "SmartChart Liquidity Engine",
        "debug_version": "liquidity_payload_v1",
        "timestamp": str(result.index[-1]),

        # headline state
        "dir": liq_dir,
        "dir_text": _dir_text(liq_dir),
        "state": _safe_str(last.get("sc_liq_state_text"), "neutral"),
        "final_score": round(_safe_float(last.get("sc_liq_final_score"), 0.0), 2),
        "quality_score": round(_safe_float(last.get("sc_liq_quality_score"), 0.0), 2),

        # liquidity memory / reclaim / sweep
        "sweep_high_mem": _safe_float(last.get("sc_liq_sweep_high_mem"), 0.0),
        "sweep_low_mem": _safe_float(last.get("sc_liq_sweep_low_mem"), 0.0),
        "reclaim_high_mem": _safe_float(last.get("sc_liq_reclaim_high_mem"), 0.0),
        "reclaim_low_mem": _safe_float(last.get("sc_liq_reclaim_low_mem"), 0.0),
        "memory_bias": _safe_int(last.get("sc_liq_memory_bias"), 0),
        "memory_bias_text": _dir_text(_safe_int(last.get("sc_liq_memory_bias"), 0)),

        # displacement
        "disp_dir": _safe_int(last.get("sc_liq_disp_dir"), 0),
        "disp_dir_text": _dir_text(_safe_int(last.get("sc_liq_disp_dir"), 0)),
        "disp_score": round(_safe_float(last.get("sc_liq_disp_score"), 0.0), 2),

        # PD / EQ
        "pd_state": pd_state,
        "pd_state_text": _pd_text(pd_state),

        # active liquidity structure
        "cluster_strength": round(_safe_float(last.get("sc_liq_cluster_strength"), 0.0), 2),
        "active_level": round(_safe_float(last.get("sc_liq_active_level"), np.nan), 4),
        "active_level_type": _safe_str(last.get("active_level_type"), ""),

        # MTF
        "mtf_trend_dir": _safe_int(last.get("sc_liq_mtf_trend_dir"), 0),
        "mtf_trend_text": _dir_text(_safe_int(last.get("sc_liq_mtf_trend_dir"), 0)),
        "mtf_avg_dir": round(mtf_avg_dir, 4),
        "mtf_agreement_text": _agreement_text(
            mtf_avg_dir,
            cfg["liq_mtf_bull_thr"],
            cfg["liq_mtf_bear_thr"],
        ),

        # OBOS + divergence
        "obos_div_state": obos_div_state,
        "obos_div_text": _obos_div_text(obos_div_state),

        # quality internals
        "bull_qual": _safe_int(last.get("sc_liq_bull_qual"), 0),
        "bear_qual": _safe_int(last.get("sc_liq_bear_qual"), 0),

        # useful supporting fields for Elementor / debugging
        "final_long_score": round(_safe_float(last.get("final_long_score"), 0.0), 2),
        "final_short_score": round(_safe_float(last.get("final_short_score"), 0.0), 2),
        "zone_long_score": round(_safe_float(last.get("zone_long_score"), 0.0), 2),
        "zone_short_score": round(_safe_float(last.get("zone_short_score"), 0.0), 2),
        "liq_breakout_up": bool(last.get("liq_breakout_up", False)),
        "liq_breakout_dn": bool(last.get("liq_breakout_dn", False)),
        "liq_long_ready": bool(last.get("liq_long_ready", False)),
        "liq_short_ready": bool(last.get("liq_short_ready", False)),
    }

    return payload