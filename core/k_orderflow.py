"""
==============================================================================
SMARTCHART BACKEND — ORDERFLOW ENGINE
File: k_orderflow.py
==============================================================================

🔒 LOCKED STATE — ORDERFLOW PARITY VERSION (APRIL 2026)

This module is the **official backend parity implementation** of the
SmartChart Order Flow Engine.

It has been aligned with:
- TradingView Pine Orderflow Engine (source of truth)
- Pine Parity Visual + Table Validator (chart validation layer)

------------------------------------------------------------------------------
CORE PURPOSE
------------------------------------------------------------------------------
This module provides backend-safe, non-visual Orderflow logic for:

- SmartChart main.py integration
- Truth table generation
- Scanner / multi-instrument processing
- Website / API data output
- Strategy decision layer (read-only)

------------------------------------------------------------------------------
DO NOT MODIFY WITHOUT PARITY VALIDATION
------------------------------------------------------------------------------
Any changes MUST be validated against:

1. Pine Orderflow Parity Visual Script
2. TradingView chart behavior
3. Expected outputs:
   - Direction
   - Strength
   - Quality
   - Zone activation states

------------------------------------------------------------------------------
PRIMARY OUTPUTS (TRUTH FIELDS)
------------------------------------------------------------------------------
These fields are considered **LOCKED OUTPUT CONTRACT**:

sc_orderflow_dir
sc_orderflow_text
sc_orderflow_strength
sc_orderflow_quality

sc_of_bull_score
sc_of_bear_score
sc_of_mtf_avg_dir

sc_of_density_bull_active
sc_of_density_bear_active

sc_of_imbalance_bull_active
sc_of_imbalance_bear_active

sc_of_zone_bull_active
sc_of_zone_bear_active

sc_of_accept_bull
sc_of_accept_bear

------------------------------------------------------------------------------
ARCHITECTURE ROLE
------------------------------------------------------------------------------
- This module is part of the SmartChart "Structure Layer"
- It feeds into the SmartChart "Truth Engine"
- It MUST NOT contain:
    - UI logic
    - Plotting
    - Trading execution logic

------------------------------------------------------------------------------
RELATIONSHIP TO OTHER MODULES
------------------------------------------------------------------------------
Depends on:
- a_indicators (EMA, ATR, etc.)

Feeds into:
- main.py
- truth_engine.py
- scanner_engine.py
- playbook_engine.py

------------------------------------------------------------------------------
IMPORTANT NOTES
------------------------------------------------------------------------------
- Python MTF is a proxy (not exact TradingView request.security)
- Zone lifecycle is simplified but state-aligned
- Visual parity is validated in Pine, not here

------------------------------------------------------------------------------
STATUS
------------------------------------------------------------------------------
✔ Backend parity achieved
✔ Visual parity validated (Pine)
✔ Ready for integration into main.py

------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# FALLBACK INDICATORS
# =============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.rolling(length, min_periods=1).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    length = max(1, int(length))
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    length = max(1, int(length))

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_rma = pd.Series(tr, index=close.index).ewm(alpha=1.0 / length, adjust=False).mean()
    plus_rma = pd.Series(plus_dm, index=close.index).ewm(alpha=1.0 / length, adjust=False).mean()
    minus_rma = pd.Series(minus_dm, index=close.index).ewm(alpha=1.0 / length, adjust=False).mean()

    plus_di = np.where(tr_rma != 0, 100.0 * plus_rma / tr_rma, 0.0)
    minus_di = np.where(tr_rma != 0, 100.0 * minus_rma / tr_rma, 0.0)

    denom = plus_di + minus_di
    dx = np.where(denom != 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    return pd.Series(dx, index=close.index).ewm(alpha=1.0 / length, adjust=False).mean()


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class OrderflowConfig:
    # Core
    atr_len: int = 14
    tf_minutes: float = 5.0

    # Opening range
    or_mode: str = "Auto"              # Auto | Manual
    or_auto_k: float = 3.0
    or_manual_minutes: int = 30

    # Trend
    trend_mode: str = "EMA+ADX"        # Off | EMA | EMA+ADX
    ema_fast_len: int = 34
    ema_slow_len: int = 89
    adx_len: int = 14
    adx_thr: float = 18.0

    # Delta
    delta_mode: str = "RangeWeighted"  # RangeWeighted | CloseVsPrev | CloseVsOpen
    delta_smooth_len: int = 5
    delta_norm_len: int = 50
    delta_impulse_mult: float = 1.35

    # Volume
    vol_len: int = 20
    rv_impulse_mult: float = 1.5
    vol_expansion_mult: float = 1.2

    # Density
    den_on: bool = True
    den_metric_mode: str = "AbsDelta"  # AbsDelta | Volume | Hybrid
    den_q_len: int = 200
    den_q_pct: float = 92.0
    den_depth_atr: float = 0.45
    den_extend_bars: int = 150
    den_merge_atr: float = 0.30
    den_break_vol_mult: float = 1.2

    # Imbalance
    imb_on: bool = True
    imb_atr_len: int = 28
    imb_atr_mult: float = 1.5
    imb_body_pct: float = 70.0

    # Structure / pivot zones
    zone_on: bool = True
    zone_depth_atr: float = 0.60
    pivot_a_left: int = 10
    pivot_a_right: int = 10
    pivot_b_left: int = 5
    pivot_b_right: int = 5

    # Compact profile
    profile_on: bool = True
    profile_lookback: int = 150
    profile_bins: int = 24
    acceptance_pct: float = 70.0

    # MTF proxy
    mtf_on: bool = True
    mtf_weight_1: float = 1.0
    mtf_weight_2: float = 1.0
    mtf_weight_3: float = 1.0
    mtf_weight_4: float = 1.0
    mtf_weight_5: float = 1.0
    mtf_span_1: int = 3
    mtf_span_2: int = 6
    mtf_span_3: int = 12
    mtf_span_4: int = 48
    mtf_span_5: int = 288

    # Session reset
    session_col: Optional[str] = None  # optional session/day key if already present


# =============================================================================
# HELPERS
# =============================================================================

def _clamp(v: pd.Series | float, lo: float, hi: float):
    if isinstance(v, pd.Series):
        return v.clip(lower=lo, upper=hi)
    return max(lo, min(hi, v))


def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series).astype(int)


def _bool_score(series: pd.Series | np.ndarray | bool) -> pd.Series:
    if isinstance(series, bool):
        return pd.Series([1.0 if series else 0.0])
    return pd.Series(np.where(series, 1.0, 0.0), index=series.index if isinstance(series, pd.Series) else None)


def _norm_signed(v: pd.Series, base: pd.Series) -> pd.Series:
    denom = base.where(base > 0, np.nan)
    out = (v / denom).clip(-1.0, 1.0)
    return out.fillna(0.0)


def _rolling_percentile(series: pd.Series, window: int, pct: float) -> pd.Series:
    q = pct / 100.0
    window = max(5, int(window))
    return series.rolling(window, min_periods=max(5, min(window, 20))).quantile(q)


def _body_pct(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    candle_range = (high - low).abs().replace(0.0, np.nan)
    body = (close - open_).abs()
    return (body / candle_range * 100.0).fillna(0.0)


def _pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    win = left + right + 1
    roll_max = high.rolling(win, center=True, min_periods=win).max()
    is_pivot = high.eq(roll_max)
    return high.where(is_pivot).shift(right)


def _pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    win = left + right + 1
    roll_min = low.rolling(win, center=True, min_periods=win).min()
    is_pivot = low.eq(roll_min)
    return low.where(is_pivot).shift(right)


def _resample_mtf_state(close: pd.Series, cfg: OrderflowConfig, span: int) -> pd.Series:
    span = max(1, int(span))
    c = close.iloc[::span].copy()
    ef = ema(c, cfg.ema_fast_len)
    es = ema(c, cfg.ema_slow_len)
    state = pd.Series(np.where((c > ef) & (ef > es), 1.0, np.where((c < ef) & (ef < es), -1.0, 0.0)), index=c.index)
    state = state.reindex(close.index).ffill().fillna(0.0)
    return state


# =============================================================================
# ENGINE
# =============================================================================

def run_orderflow_engine(df: pd.DataFrame, config: Optional[OrderflowConfig] = None) -> pd.DataFrame:
    cfg = config or OrderflowConfig()
    out = df.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Orderflow engine missing required columns: {sorted(missing)}")

    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    # -------------------------------------------------------------------------
    # Core
    # -------------------------------------------------------------------------
    out["sc_of_atr"] = atr(high, low, close, cfg.atr_len)
    hlc3 = (high + low + close) / 3.0

    if cfg.session_col and cfg.session_col in out.columns:
        new_sess = out[cfg.session_col].ne(out[cfg.session_col].shift(1)).fillna(True)
    elif isinstance(out.index, pd.DatetimeIndex):
        new_sess = pd.Series(out.index.normalize(), index=out.index).ne(
            pd.Series(out.index.normalize(), index=out.index).shift(1)
        ).fillna(True)
    else:
        bars_per_day = max(1, int(round(1440 / max(cfg.tf_minutes, 1.0))))
        new_sess = pd.Series(False, index=out.index)
        new_sess.iloc[::bars_per_day] = True
        new_sess.iloc[0] = True

    cum_pv = np.zeros(len(out), dtype=float)
    cum_v = np.zeros(len(out), dtype=float)
    pv_running = 0.0
    v_running = 0.0
    for i in range(len(out)):
        if bool(new_sess.iloc[i]):
            pv_running = 0.0
            v_running = 0.0
        pv_running += float(hlc3.iloc[i] * volume.iloc[i])
        v_running += float(volume.iloc[i])
        cum_pv[i] = pv_running
        cum_v[i] = v_running
    out["sc_of_vwap"] = np.where(cum_v > 0, cum_pv / cum_v, np.nan)

    # -------------------------------------------------------------------------
    # Opening range
    # -------------------------------------------------------------------------
    or_len_min = cfg.or_auto_k * cfg.tf_minutes if cfg.or_mode == "Auto" else float(cfg.or_manual_minutes)
    or_bars = max(1, int(np.ceil(or_len_min / max(cfg.tf_minutes, 1.0))))

    or_high = np.full(len(out), np.nan)
    or_low = np.full(len(out), np.nan)
    count = 0
    cur_hi = np.nan
    cur_lo = np.nan

    for i in range(len(out)):
        if bool(new_sess.iloc[i]):
            count = 0
            cur_hi = np.nan
            cur_lo = np.nan

        if count < or_bars:
            cur_hi = high.iloc[i] if np.isnan(cur_hi) else max(cur_hi, high.iloc[i])
            cur_lo = low.iloc[i] if np.isnan(cur_lo) else min(cur_lo, low.iloc[i])
            count += 1

        or_high[i] = cur_hi
        or_low[i] = cur_lo

    out["sc_of_or_high"] = or_high
    out["sc_of_or_low"] = or_low
    out["sc_of_above_or"] = (close > out["sc_of_or_high"]).astype(int)
    out["sc_of_below_or"] = (close < out["sc_of_or_low"]).astype(int)

    # -------------------------------------------------------------------------
    # Trend
    # -------------------------------------------------------------------------
    out["sc_of_trend_ema_fast"] = ema(close, cfg.ema_fast_len)
    out["sc_of_trend_ema_slow"] = ema(close, cfg.ema_slow_len)
    out["sc_of_adx"] = adx(high, low, close, cfg.adx_len)

    trend_up = out["sc_of_trend_ema_fast"] > out["sc_of_trend_ema_slow"]
    trend_dn = out["sc_of_trend_ema_fast"] < out["sc_of_trend_ema_slow"]

    if cfg.trend_mode == "Off":
        trend_long_ok = pd.Series(True, index=out.index)
        trend_short_ok = pd.Series(True, index=out.index)
    elif cfg.trend_mode == "EMA":
        trend_long_ok = trend_up
        trend_short_ok = trend_dn
    else:
        trend_long_ok = trend_up & (out["sc_of_adx"] >= cfg.adx_thr)
        trend_short_ok = trend_dn & (out["sc_of_adx"] >= cfg.adx_thr)

    out["sc_of_trend_long_ok"] = trend_long_ok.astype(int)
    out["sc_of_trend_short_ok"] = trend_short_ok.astype(int)
    out["sc_of_trend_dir"] = np.where(trend_long_ok, 1, np.where(trend_short_ok, -1, 0))

    # -------------------------------------------------------------------------
    # Delta
    # -------------------------------------------------------------------------
    rng = (high - low).abs().replace(0.0, np.nan)
    delta_rw = (volume * (close - open_) / rng).fillna(0.0)
    delta_cp = np.where(close >= close.shift(1), volume, -volume)
    delta_co = np.where(close >= open_, volume, -volume)

    if cfg.delta_mode == "RangeWeighted":
        delta_raw = pd.Series(delta_rw, index=out.index)
    elif cfg.delta_mode == "CloseVsOpen":
        delta_raw = pd.Series(delta_co, index=out.index)
    else:
        delta_raw = pd.Series(delta_cp, index=out.index)

    out["sc_of_delta_raw"] = delta_raw
    out["sc_of_delta_sm"] = ema(delta_raw, cfg.delta_smooth_len)
    out["sc_of_delta_abs"] = out["sc_of_delta_sm"].abs()
    out["sc_of_delta_base"] = sma(out["sc_of_delta_abs"], cfg.delta_norm_len).clip(lower=1.0)
    out["sc_of_delta_norm"] = _norm_signed(out["sc_of_delta_sm"], out["sc_of_delta_base"])
    out["sc_of_delta_dir"] = _sign(out["sc_of_delta_sm"])

    out["sc_of_bull_delta_impulse"] = (
        (out["sc_of_delta_sm"] > 0)
        & (out["sc_of_delta_abs"] > (out["sc_of_delta_base"] * cfg.delta_impulse_mult).clip(lower=1.0))
    ).astype(int)

    out["sc_of_bear_delta_impulse"] = (
        (out["sc_of_delta_sm"] < 0)
        & (out["sc_of_delta_abs"] > (out["sc_of_delta_base"] * cfg.delta_impulse_mult).clip(lower=1.0))
    ).astype(int)

    # -------------------------------------------------------------------------
    # Participation / volume
    # -------------------------------------------------------------------------
    out["sc_of_vol_ma"] = sma(volume, cfg.vol_len)
    out["sc_of_rel_vol"] = np.where(out["sc_of_vol_ma"] > 0, volume / out["sc_of_vol_ma"], 1.0)
    out["sc_of_vol_expansion"] = (out["sc_of_rel_vol"] >= cfg.vol_expansion_mult).astype(int)
    out["sc_of_vol_impulse"] = (out["sc_of_rel_vol"] >= cfg.rv_impulse_mult).astype(int)

    out["sc_of_bull_participation"] = ((out["sc_of_delta_sm"] > 0) & (out["sc_of_vol_impulse"] == 1)).astype(int)
    out["sc_of_bear_participation"] = ((out["sc_of_delta_sm"] < 0) & (out["sc_of_vol_impulse"] == 1)).astype(int)

    out["sc_of_participation_score"] = (
        ((_clamp(out["sc_of_rel_vol"] / max(cfg.rv_impulse_mult, 1.0), 0.0, 2.0)) * 0.6)
        + (out["sc_of_delta_norm"].abs() * 0.4)
    ).clip(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Density
    # -------------------------------------------------------------------------
    if cfg.den_metric_mode == "AbsDelta":
        den_metric = out["sc_of_delta_sm"].abs()
    elif cfg.den_metric_mode == "Volume":
        den_metric = volume
    else:
        den_metric = out["sc_of_delta_sm"].abs() * 0.6 + volume * 0.4

    out["sc_of_den_metric"] = den_metric
    out["sc_of_den_thr"] = _rolling_percentile(den_metric, cfg.den_q_len, cfg.den_q_pct)

    density_hit = cfg.den_on & (out["sc_of_den_metric"] >= out["sc_of_den_thr"])
    density_mid = close
    density_top = density_mid + out["sc_of_atr"] * cfg.den_depth_atr
    density_bot = density_mid - out["sc_of_atr"] * cfg.den_depth_atr

    out["sc_of_density_event"] = density_hit.astype(int)
    out["sc_of_density_mid"] = np.where(density_hit, density_mid, np.nan)
    out["sc_of_density_top"] = np.where(density_hit, density_top, np.nan)
    out["sc_of_density_bot"] = np.where(density_hit, density_bot, np.nan)

    bull_density_zone = (
        density_hit.shift(1).fillna(False)
        & (out["sc_of_delta_sm"].shift(1).fillna(0.0) >= 0)
        & (high >= density_bot.shift(1))
        & (low <= density_top.shift(1))
    )
    bear_density_zone = (
        density_hit.shift(1).fillna(False)
        & (out["sc_of_delta_sm"].shift(1).fillna(0.0) < 0)
        & (high >= density_bot.shift(1))
        & (low <= density_top.shift(1))
    )

    out["sc_of_density_bull_active"] = bull_density_zone.astype(int)
    out["sc_of_density_bear_active"] = bear_density_zone.astype(int)

    # -------------------------------------------------------------------------
    # Imbalance
    # -------------------------------------------------------------------------
    out["sc_of_imb_atr"] = atr(high, low, close, cfg.imb_atr_len)
    body_pct = _body_pct(open_, high, low, close)

    is_up_1 = open_.shift(1) <= close.shift(1)
    price_diff = (high.shift(1) - low.shift(1)).abs()
    big_body = body_pct.shift(1) >= cfg.imb_body_pct

    gap_closed = np.where(
        is_up_1.fillna(False),
        high.shift(2) >= low,
        low.shift(2) <= high,
    )

    imbalance_zone = (
        cfg.imb_on
        & (price_diff > (out["sc_of_imb_atr"] * cfg.imb_atr_mult))
        & big_body.fillna(False)
        & (~pd.Series(gap_closed, index=out.index).fillna(False))
    )

    out["sc_of_imbalance_event"] = imbalance_zone.astype(int)
    out["sc_of_imbalance_dir"] = np.where(imbalance_zone & is_up_1.fillna(False), 1, np.where(imbalance_zone, -1, 0))

    imb_top = pd.Series(
        np.where(
            is_up_1.fillna(False),
            low,
            low.shift(2),
        ),
        index=out.index,
    )
    imb_bot = pd.Series(
        np.where(
            is_up_1.fillna(False),
            high.shift(2),
            high,
        ),
        index=out.index,
    )

    out["sc_of_imbalance_top"] = np.where(imbalance_zone, imb_top, np.nan)
    out["sc_of_imbalance_bot"] = np.where(imbalance_zone, imb_bot, np.nan)

    out["sc_of_imbalance_bull_active"] = (
        imbalance_zone.shift(1).fillna(False)
        & (out["sc_of_imbalance_dir"].shift(1) == 1)
        & (high >= out["sc_of_imbalance_bot"].shift(1))
        & (low <= out["sc_of_imbalance_top"].shift(1))
    ).astype(int)

    out["sc_of_imbalance_bear_active"] = (
        imbalance_zone.shift(1).fillna(False)
        & (out["sc_of_imbalance_dir"].shift(1) == -1)
        & (high >= out["sc_of_imbalance_bot"].shift(1))
        & (low <= out["sc_of_imbalance_top"].shift(1))
    ).astype(int)

    # -------------------------------------------------------------------------
    # Structure zones
    # -------------------------------------------------------------------------
    ph_a = _pivot_high(high, cfg.pivot_a_left, cfg.pivot_a_right)
    pl_a = _pivot_low(low, cfg.pivot_a_left, cfg.pivot_a_right)
    ph_b = _pivot_high(high, cfg.pivot_b_left, cfg.pivot_b_right)
    pl_b = _pivot_low(low, cfg.pivot_b_left, cfg.pivot_b_right)

    out["sc_of_pivot_high_a"] = ph_a
    out["sc_of_pivot_low_a"] = pl_a
    out["sc_of_pivot_high_b"] = ph_b
    out["sc_of_pivot_low_b"] = pl_b

    sup_a_top = ph_a.ffill()
    sup_a_bot = sup_a_top - out["sc_of_atr"] * cfg.zone_depth_atr
    dem_a_bot = pl_a.ffill()
    dem_a_top = dem_a_bot + out["sc_of_atr"] * cfg.zone_depth_atr

    sup_b_top = ph_b.ffill()
    sup_b_bot = sup_b_top - out["sc_of_atr"] * cfg.zone_depth_atr
    dem_b_bot = pl_b.ffill()
    dem_b_top = dem_b_bot + out["sc_of_atr"] * cfg.zone_depth_atr

    out["sc_of_zone_a_sup_top"] = sup_a_top
    out["sc_of_zone_a_sup_bot"] = sup_a_bot
    out["sc_of_zone_a_dem_top"] = dem_a_top
    out["sc_of_zone_a_dem_bot"] = dem_a_bot
    out["sc_of_zone_b_sup_top"] = sup_b_top
    out["sc_of_zone_b_sup_bot"] = sup_b_bot
    out["sc_of_zone_b_dem_top"] = dem_b_top
    out["sc_of_zone_b_dem_bot"] = dem_b_bot

    zone_bull_active = (
        ((high >= dem_a_bot) & (low <= dem_a_top))
        | ((high >= dem_b_bot) & (low <= dem_b_top))
    ).fillna(False)

    zone_bear_active = (
        ((high >= sup_a_bot) & (low <= sup_a_top))
        | ((high >= sup_b_bot) & (low <= sup_b_top))
    ).fillna(False)

    out["sc_of_zone_bull_active"] = (cfg.zone_on & zone_bull_active).astype(int)
    out["sc_of_zone_bear_active"] = (cfg.zone_on & zone_bear_active).astype(int)

    # -------------------------------------------------------------------------
    # Compact profile
    # -------------------------------------------------------------------------
    poc_price = np.full(len(out), np.nan)
    acc_top = np.full(len(out), np.nan)
    acc_bot = np.full(len(out), np.nan)
    poc_bull_share = np.full(len(out), np.nan)
    poc_bear_share = np.full(len(out), np.nan)

    if cfg.profile_on:
        bins = max(10, int(cfg.profile_bins))
        lb = max(25, int(cfg.profile_lookback))

        for i in range(len(out)):
            if i < lb - 1:
                continue

            h_slice = high.iloc[i - lb + 1 : i + 1]
            l_slice = low.iloc[i - lb + 1 : i + 1]
            c_slice = close.iloc[i - lb + 1 : i + 1]
            o_slice = open_.iloc[i - lb + 1 : i + 1]
            v_slice = volume.iloc[i - lb + 1 : i + 1]

            range_hi = float(h_slice.max())
            range_lo = float(l_slice.min())
            step = (range_hi - range_lo) / bins

            if step <= 0:
                continue

            vol_arr = np.zeros(bins)
            bull_arr = np.zeros(bins)
            bear_arr = np.zeros(bins)

            for px, vv, oo, cc in zip(c_slice.values, v_slice.values, o_slice.values, c_slice.values):
                idx = int(np.floor((px - range_lo) / step))
                idx = max(0, min(bins - 1, idx))
                vol_arr[idx] += vv
                if cc >= oo:
                    bull_arr[idx] += vv
                else:
                    bear_arr[idx] += vv

            poc_idx = int(np.argmax(vol_arr))
            poc_vol = vol_arr[poc_idx]
            poc_bot_ = range_lo + step * poc_idx
            poc_top_ = poc_bot_ + step

            poc_price[i] = (poc_top_ + poc_bot_) * 0.5

            poc_bull = bull_arr[poc_idx]
            poc_bear = bear_arr[poc_idx]
            poc_sum = poc_bull + poc_bear

            poc_bull_share[i] = (poc_bull / poc_sum) if poc_sum > 0 else 0.5
            poc_bear_share[i] = (poc_bear / poc_sum) if poc_sum > 0 else 0.5

            thr_vol = poc_vol * (cfg.acceptance_pct / 100.0)
            low_idx = poc_idx
            hi_idx = poc_idx

            k = poc_idx - 1
            while k >= 0:
                if vol_arr[k] >= thr_vol:
                    low_idx = k
                    k -= 1
                else:
                    break

            k = poc_idx + 1
            while k < bins:
                if vol_arr[k] >= thr_vol:
                    hi_idx = k
                    k += 1
                else:
                    break

            acc_bot[i] = range_lo + step * low_idx
            acc_top[i] = range_lo + step * (hi_idx + 1)

    out["sc_of_poc_price"] = poc_price
    out["sc_of_acc_top"] = acc_top
    out["sc_of_acc_bot"] = acc_bot
    out["sc_of_poc_bull_share"] = poc_bull_share
    out["sc_of_poc_bear_share"] = poc_bear_share

    in_acceptance = (
        out["sc_of_acc_top"].notna()
        & out["sc_of_acc_bot"].notna()
        & (high >= out["sc_of_acc_bot"])
        & (low <= out["sc_of_acc_top"])
    )

    out["sc_of_in_acceptance"] = in_acceptance.astype(int)
    out["sc_of_accept_bull"] = (in_acceptance & (out["sc_of_poc_bull_share"] > 0.55)).astype(int)
    out["sc_of_accept_bear"] = (in_acceptance & (out["sc_of_poc_bear_share"] > 0.55)).astype(int)

    # -------------------------------------------------------------------------
    # MTF average
    # -------------------------------------------------------------------------
    if cfg.mtf_on:
        mtf1 = _resample_mtf_state(close, cfg, cfg.mtf_span_1)
        mtf2 = _resample_mtf_state(close, cfg, cfg.mtf_span_2)
        mtf3 = _resample_mtf_state(close, cfg, cfg.mtf_span_3)
        mtf4 = _resample_mtf_state(close, cfg, cfg.mtf_span_4)
        mtf5 = _resample_mtf_state(close, cfg, cfg.mtf_span_5)

        wsum = max(cfg.mtf_weight_1 + cfg.mtf_weight_2 + cfg.mtf_weight_3 + cfg.mtf_weight_4 + cfg.mtf_weight_5, 1e-9)
        mtf_raw = (
            mtf1 * cfg.mtf_weight_1
            + mtf2 * cfg.mtf_weight_2
            + mtf3 * cfg.mtf_weight_3
            + mtf4 * cfg.mtf_weight_4
            + mtf5 * cfg.mtf_weight_5
        )
        mtf_avg_dir = mtf_raw / wsum
    else:
        mtf1 = mtf2 = mtf3 = mtf4 = mtf5 = pd.Series(0.0, index=out.index)
        mtf_avg_dir = pd.Series(0.0, index=out.index)

    out["sc_of_mtf_s1"] = mtf1
    out["sc_of_mtf_s2"] = mtf2
    out["sc_of_mtf_s3"] = mtf3
    out["sc_of_mtf_s4"] = mtf4
    out["sc_of_mtf_s5"] = mtf5
    out["sc_of_mtf_avg_dir"] = mtf_avg_dir
    out["sc_of_mtf_support_bull"] = (mtf_avg_dir > 0.20).astype(int)
    out["sc_of_mtf_support_bear"] = (mtf_avg_dir < -0.20).astype(int)
    out["sc_of_mtf_agreement_score"] = mtf_avg_dir.abs()

    # -------------------------------------------------------------------------
    # Final state
    # -------------------------------------------------------------------------
    bull_score_raw = (
        ((close > out["sc_of_vwap"]).astype(float) * 0.12)
        + ((close > out["sc_of_or_high"]).astype(float) * 0.08)
        + ((out["sc_of_delta_sm"] > 0).astype(float) * 0.15)
        + (out["sc_of_bull_delta_impulse"].astype(float) * 0.12)
        + (out["sc_of_bull_participation"].astype(float) * 0.10)
        + (out["sc_of_density_bull_active"].astype(float) * 0.08)
        + (out["sc_of_imbalance_bull_active"].astype(float) * 0.10)
        + (out["sc_of_zone_bull_active"].astype(float) * 0.07)
        + (out["sc_of_accept_bull"].astype(float) * 0.08)
        + (out["sc_of_trend_long_ok"].astype(float) * 0.05)
        + (out["sc_of_mtf_support_bull"].astype(float) * 0.05)
    )

    bear_score_raw = (
        ((close < out["sc_of_vwap"]).astype(float) * 0.12)
        + ((close < out["sc_of_or_low"]).astype(float) * 0.08)
        + ((out["sc_of_delta_sm"] < 0).astype(float) * 0.15)
        + (out["sc_of_bear_delta_impulse"].astype(float) * 0.12)
        + (out["sc_of_bear_participation"].astype(float) * 0.10)
        + (out["sc_of_density_bear_active"].astype(float) * 0.08)
        + (out["sc_of_imbalance_bear_active"].astype(float) * 0.10)
        + (out["sc_of_zone_bear_active"].astype(float) * 0.07)
        + (out["sc_of_accept_bear"].astype(float) * 0.08)
        + (out["sc_of_trend_short_ok"].astype(float) * 0.05)
        + (out["sc_of_mtf_support_bear"].astype(float) * 0.05)
    )

    out["sc_of_bull_score"] = bull_score_raw.clip(0.0, 1.0)
    out["sc_of_bear_score"] = bear_score_raw.clip(0.0, 1.0)

    out["sc_orderflow_dir"] = np.where(
        (out["sc_of_bull_score"] > out["sc_of_bear_score"]) & (out["sc_of_bull_score"] >= 0.35),
        1,
        np.where(
            (out["sc_of_bear_score"] > out["sc_of_bull_score"]) & (out["sc_of_bear_score"] >= 0.35),
            -1,
            0,
        ),
    )

    max_side_score = np.maximum(out["sc_of_bull_score"], out["sc_of_bear_score"])
    out["sc_orderflow_strength"] = (
        max_side_score * 0.60
        + out["sc_of_participation_score"] * 0.20
        + out["sc_of_mtf_agreement_score"] * 0.20
    ).clip(0.0, 1.0)

    quality_trend_ok = (
        ((out["sc_orderflow_dir"] == 1) & (out["sc_of_trend_long_ok"] == 1))
        | ((out["sc_orderflow_dir"] == -1) & (out["sc_of_trend_short_ok"] == 1))
    )
    quality_mtf_ok = (
        ((out["sc_orderflow_dir"] == 1) & (out["sc_of_mtf_support_bull"] == 1))
        | ((out["sc_orderflow_dir"] == -1) & (out["sc_of_mtf_support_bear"] == 1))
    )
    quality_flow_ok = (
        ((out["sc_orderflow_dir"] == 1) & (
            (out["sc_of_imbalance_bull_active"] == 1)
            | (out["sc_of_density_bull_active"] == 1)
            | (out["sc_of_zone_bull_active"] == 1)
        ))
        | ((out["sc_orderflow_dir"] == -1) & (
            (out["sc_of_imbalance_bear_active"] == 1)
            | (out["sc_of_density_bear_active"] == 1)
            | (out["sc_of_zone_bear_active"] == 1)
        ))
    )
    quality_delta_ok = out["sc_of_delta_norm"].abs() > 0.35

    out["sc_orderflow_quality"] = (
        (in_acceptance.astype(float) * 0.15)
        + (out["sc_of_vol_expansion"].astype(float) * 0.15)
        + (quality_delta_ok.astype(float) * 0.15)
        + (quality_trend_ok.astype(float) * 0.15)
        + (quality_mtf_ok.astype(float) * 0.15)
        + (quality_flow_ok.astype(float) * 0.25)
    ).clip(0.0, 1.0)

    # parity-style aliases
    out["ofDir"] = out["sc_orderflow_dir"]
    out["ofStrengthScore"] = out["sc_orderflow_strength"]
    out["ofQualityScore"] = out["sc_orderflow_quality"]
    out["bullScore"] = out["sc_of_bull_score"]
    out["bearScore"] = out["sc_of_bear_score"]
    out["mtfAvgDir"] = out["sc_of_mtf_avg_dir"]

    out["ofImpulseBull"] = ((out["sc_of_bull_delta_impulse"] == 1) | (out["sc_of_bull_participation"] == 1)).astype(int)
    out["ofImpulseBear"] = ((out["sc_of_bear_delta_impulse"] == 1) | (out["sc_of_bear_participation"] == 1)).astype(int)
    out["ofAcceptBull"] = out["sc_of_accept_bull"]
    out["ofAcceptBear"] = out["sc_of_accept_bear"]
    out["ofImbalanceBull"] = out["sc_of_imbalance_bull_active"]
    out["ofImbalanceBear"] = out["sc_of_imbalance_bear_active"]
    out["ofDensityBull"] = out["sc_of_density_bull_active"]
    out["ofDensityBear"] = out["sc_of_density_bear_active"]

    out["sc_orderflow_text"] = np.where(
        out["sc_orderflow_dir"] == 1,
        "BULLISH",
        np.where(out["sc_orderflow_dir"] == -1, "BEARISH", "NEUTRAL"),
    )

    return out


# =============================================================================
# MAIN.PY HELPER
# =============================================================================

def apply_k_orderflow(df: pd.DataFrame, config: Optional[OrderflowConfig] = None) -> pd.DataFrame:
    return run_orderflow_engine(df, config=config)


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

def _make_test_data(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=rows, freq="5min")

    base = 2650 + np.cumsum(rng.normal(0, 1.8, rows))
    open_ = pd.Series(base + rng.normal(0, 0.8, rows), index=idx)
    close = pd.Series(base + rng.normal(0, 0.8, rows), index=idx)
    high = pd.Series(np.maximum(open_, close) + np.abs(rng.normal(1.2, 0.5, rows)), index=idx)
    low = pd.Series(np.minimum(open_, close) - np.abs(rng.normal(1.2, 0.5, rows)), index=idx)
    volume = pd.Series(rng.integers(100, 2500, rows), index=idx)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    df = _make_test_data()
    result = run_orderflow_engine(df)

    cols = [
        "close",
        "sc_orderflow_text",
        "sc_orderflow_dir",
        "sc_orderflow_strength",
        "sc_orderflow_quality",
        "sc_of_bull_score",
        "sc_of_bear_score",
        "sc_of_mtf_avg_dir",
        "sc_of_density_bull_active",
        "sc_of_density_bear_active",
        "sc_of_imbalance_bull_active",
        "sc_of_imbalance_bear_active",
        "sc_of_zone_bull_active",
        "sc_of_zone_bear_active",
        "sc_of_accept_bull",
        "sc_of_accept_bear",
    ]
    print("\nCONFIG:")
    print(asdict(OrderflowConfig()))
    print("\nLAST 20 ROWS:")
    print(result[cols].tail(20).round(4))