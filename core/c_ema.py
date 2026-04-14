from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from core.a_indicators import ema, atr


DEFAULT_EMA_CONFIG: Dict[str, Any] = {
    # Logic
    "slope_len": 5,
    "spread_lookback": 10,
    "exp_1420_min_pct": 0.02,
    "exp_3350_min_pct": 0.05,
    "exp_grow_pct": 0.01,
    "flat_band_pct": 0.02,
    "strict_trend_mode": False,

    # MTF
    "use_mtf_confirm": True,
    "mtf_weight_local": 0.60,
    "mtf_weight_avg": 0.40,
    "mtf_bull_thresh": 0.20,
    "mtf_bear_thresh": -0.20,

    # Distance
    "dist_atr_len": 14,
    "stretch20_atr": 1.00,
    "stretch200_atr": 2.00,

    # Retest
    "rt_on": True,
    "rt_lookback_bars": 12,
    "rt_pad_atr_mult": 0.15,
    "rt_require_close": True,

    # Reclaim
    "reclaim_on": True,
    "reclaim_lookback": 3,

    # Slow structure constants
    "e20200_lookback": 10,
    "e20200_near_pct": 0.15,
    "e20200_wide_pct": 0.60,
    "e20200_grow_pct": 0.05,
    "e20200_flat_pct": 0.02,
    "e100e200_near_pct": 0.10,
    "e100e200_wide_pct": 0.35,
}


# =============================================================================
# HELPERS
# =============================================================================

def _to_float_series(series: pd.Series, index: pd.Index) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=index, dtype=float)
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


def safe_div(a, b, default=np.nan):
    if np.isscalar(b):
        return a / b if b != 0 else default

    if isinstance(a, pd.Series):
        a_series = pd.to_numeric(a, errors="coerce")
    else:
        a_series = pd.Series(a, index=b.index if isinstance(b, pd.Series) else None, dtype=float)

    if isinstance(b, pd.Series):
        b_series = pd.to_numeric(b, errors="coerce")
    else:
        b_series = pd.Series(b, index=a_series.index, dtype=float)

    out = pd.Series(default, index=a_series.index, dtype=float)
    mask = b_series.notna() & (b_series != 0)
    out.loc[mask] = a_series.loc[mask] / b_series.loc[mask]
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_prior_high(high: pd.Series, lookback: int) -> pd.Series:
    return high.shift(1).rolling(max(1, int(lookback)), min_periods=1).max()


def rolling_prior_low(low: pd.Series, lookback: int) -> pd.Series:
    return low.shift(1).rolling(max(1, int(lookback)), min_periods=1).min()


def sign_dir(series: pd.Series, index: pd.Index) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(s > 0, 1, np.where(s < 0, -1, 0)), index=index, dtype=int)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        .dropna()
    )
    return out


def _mtf_state_from_ohlc(df_tf: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(df_tf["close"], errors="coerce")

    e14 = pd.to_numeric(ema(c, 14), errors="coerce")
    e20 = pd.to_numeric(ema(c, 20), errors="coerce")
    e33 = pd.to_numeric(ema(c, 33), errors="coerce")
    e50 = pd.to_numeric(ema(c, 50), errors="coerce")
    e100 = pd.to_numeric(ema(c, 100), errors="coerce")
    e200 = pd.to_numeric(ema(c, 200), errors="coerce")

    bull_fast = (e14 > e20) & (e20 > e33) & (e33 > e50)
    bear_fast = (e14 < e20) & (e20 < e33) & (e33 < e50)
    bull_slow = (e50 > e100) & (e100 > e200)
    bear_slow = (e50 < e100) & (e100 < e200)

    band_bull = e33 > e50
    band_bear = e33 < e50

    state = pd.Series(0.0, index=df_tf.index, dtype=float)
    state.loc[bull_fast & bull_slow] = 1.0
    state.loc[bear_fast & bear_slow] = -1.0
    state.loc[(state == 0.0) & bull_fast & (e100 >= e200)] = 1.0
    state.loc[(state == 0.0) & bear_fast & (e100 <= e200)] = -1.0
    state.loc[(state == 0.0) & band_bull] = 0.5
    state.loc[(state == 0.0) & band_bear] = -0.5

    return state


def build_mtf_resample_states(base_df: pd.DataFrame) -> Dict[str, pd.Series]:
    if not isinstance(base_df.index, pd.DatetimeIndex):
        raise ValueError("MTF resample engine requires a DatetimeIndex.")

    tf_map = {
        "mtf1": "1min",
        "mtf5": "5min",
        "mtf15": "15min",
        "mtf60": "60min",
        "mtf240": "240min",
        "mtfD": "1D",
    }

    out: Dict[str, pd.Series] = {}

    for key, rule in tf_map.items():
        tf_ohlc = _resample_ohlc(base_df, rule)
        tf_state = _mtf_state_from_ohlc(tf_ohlc)
        mapped = tf_state.reindex(base_df.index, method="ffill")
        out[key] = mapped.astype(float).fillna(0.0)

    return out


def recent_event_memory(
    bull_raw: pd.Series,
    bear_raw: pd.Series,
    direction: pd.Series,
    lookback_bars: int,
    index: pd.Index,
) -> pd.Series:
    bull_tbar = pd.Series(np.nan, index=index, dtype=float)
    bear_tbar = pd.Series(np.nan, index=index, dtype=float)

    n = len(index)
    for i in range(n):
        if i > 0:
            bull_tbar.iloc[i] = bull_tbar.iloc[i - 1]
            bear_tbar.iloc[i] = bear_tbar.iloc[i - 1]

        if bool(bull_raw.iloc[i]):
            bull_tbar.iloc[i] = i
        if bool(bear_raw.iloc[i]):
            bear_tbar.iloc[i] = i

    bar_idx = pd.Series(np.arange(n), index=index, dtype=float)

    out = pd.Series(False, index=index, dtype=bool)
    bull_mask = direction == 1
    bear_mask = direction == -1

    out.loc[bull_mask] = bull_tbar.loc[bull_mask].notna() & (
        (bar_idx.loc[bull_mask] - bull_tbar.loc[bull_mask]) <= lookback_bars
    )
    out.loc[bear_mask] = bear_tbar.loc[bear_mask].notna() & (
        (bar_idx.loc[bear_mask] - bear_tbar.loc[bear_mask]) <= lookback_bars
    )
    return out


# =============================================================================
# MAIN ENGINE
# =============================================================================

def compute_ema_engine(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_EMA_CONFIG, **(config or {})}

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    idx = out.index

    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("c_ema.py now requires a DatetimeIndex for full MTF parity.")

    open_ = _to_float_series(out["open"], idx)
    high = _to_float_series(out["high"], idx)
    low = _to_float_series(out["low"], idx)
    close = _to_float_series(out["close"], idx)

    calc_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )

    # =========================================================================
    # EMA CORE
    # =========================================================================

    ema14 = pd.to_numeric(ema(close, 14), errors="coerce")
    ema20 = pd.to_numeric(ema(close, 20), errors="coerce")
    ema33 = pd.to_numeric(ema(close, 33), errors="coerce")
    ema50 = pd.to_numeric(ema(close, 50), errors="coerce")
    ema100 = pd.to_numeric(ema(close, 100), errors="coerce")
    ema200 = pd.to_numeric(ema(close, 200), errors="coerce")

    band_lo = pd.concat([ema33, ema50], axis=1).min(axis=1)
    band_hi = pd.concat([ema33, ema50], axis=1).max(axis=1)

    # =========================================================================
    # DISTANCE ENGINE
    # =========================================================================

    atr_dist = pd.to_numeric(
        atr(high, low, close, int(cfg["dist_atr_len"])),
        errors="coerce",
    )

    p_to_e20_pts = (close - ema20).abs()
    p_to_e20_pct = safe_div((close - ema20) * 100.0, ema20)
    p_to_e20_signed = close - ema20

    p_to_e200_pts = (close - ema200).abs()
    p_to_e200_pct = safe_div((close - ema200) * 100.0, ema200)
    p_to_e200_signed = close - ema200

    e20_to_e200_pts = (ema20 - ema200).abs()
    e20_to_e200_pct = safe_div((ema20 - ema200) * 100.0, ema200)
    e20_to_e200_signed = ema20 - ema200

    is_stretched_from20 = (
        atr_dist.notna()
        & (atr_dist > 0)
        & (safe_div((close - ema20).abs(), atr_dist, default=0.0) >= float(cfg["stretch20_atr"]))
    )

    is_stretched_from200 = (
        atr_dist.notna()
        & (atr_dist > 0)
        & (safe_div((close - ema200).abs(), atr_dist, default=0.0) >= float(cfg["stretch200_atr"]))
    )

    dist20_dir = pd.Series(np.where(close > ema20, 1, np.where(close < ema20, -1, 0)), index=idx, dtype=int)
    dist200_dir = pd.Series(np.where(close > ema200, 1, np.where(close < ema200, -1, 0)), index=idx, dtype=int)

    # =========================================================================
    # SLOPES
    # =========================================================================

    slope_len = max(1, int(cfg["slope_len"]))
    flat_band_pct = float(cfg["flat_band_pct"])

    ema14_prev = ema14.shift(slope_len)
    ema20_prev = ema20.shift(slope_len)
    ema33_prev = ema33.shift(slope_len)
    ema50_prev = ema50.shift(slope_len)
    ema100_prev = ema100.shift(slope_len)
    ema200_prev = ema200.shift(slope_len)

    ema14_slope_pct = safe_div((ema14 - ema14_prev) * 100.0, ema14_prev)
    ema20_slope_pct = safe_div((ema20 - ema20_prev) * 100.0, ema20_prev)
    ema33_slope_pct = safe_div((ema33 - ema33_prev) * 100.0, ema33_prev)
    ema50_slope_pct = safe_div((ema50 - ema50_prev) * 100.0, ema50_prev)
    ema100_slope_pct = safe_div((ema100 - ema100_prev) * 100.0, ema100_prev)
    ema200_slope_pct = safe_div((ema200 - ema200_prev) * 100.0, ema200_prev)

    ema14_slope_dir = sign_dir(ema14_slope_pct, idx)
    ema20_slope_dir = sign_dir(ema20_slope_pct, idx)
    ema33_slope_dir = sign_dir(ema33_slope_pct, idx)
    ema50_slope_dir = sign_dir(ema50_slope_pct, idx)
    ema100_slope_dir = sign_dir(ema100_slope_pct, idx)
    ema200_slope_dir = sign_dir(ema200_slope_pct, idx)

    ema_slope_fast = (ema20 - ema20.shift(1)).fillna(0.0)
    ema_slope_band = (ema50 - ema50.shift(1)).fillna(0.0)
    ema_slope_slow = (ema200 - ema200.shift(1)).fillna(0.0)

    ema_slope_fast_state = sign_dir(ema_slope_fast, idx)
    ema_slope_band_state = sign_dir(ema_slope_band, idx)
    ema_slope_slow_state = sign_dir(ema_slope_slow, idx)

    # =========================================================================
    # SLOW STRUCTURE ENGINE
    # =========================================================================

    e20e200_now = e20_to_e200_pct
    e20e200_prev = e20_to_e200_pct.shift(int(cfg["e20200_lookback"]))
    e20e200_grow = e20e200_now - e20e200_prev

    e20e200_has = e20e200_now.notna()
    e20200_near = e20e200_has & (e20e200_now.abs() <= float(cfg["e20200_near_pct"]))
    e20200_wide = e20e200_has & (e20e200_now.abs() >= float(cfg["e20200_wide_pct"]))
    e20200_expand = e20200_wide & e20e200_grow.notna() & (e20e200_grow.abs() >= float(cfg["e20200_grow_pct"]))

    e20200_flat = (
        ema20_slope_pct.notna()
        & ema200_slope_pct.notna()
        & (ema20_slope_pct.abs() <= float(cfg["e20200_flat_pct"]))
        & (ema200_slope_pct.abs() <= float(cfg["e20200_flat_pct"]))
    )

    e20200_state = pd.Series(
        np.where(
            e20200_flat,
            4,
            np.where(e20200_expand, 3, np.where(e20200_near, 1, np.where(e20200_wide, 2, 0))),
        ),
        index=idx,
        dtype=int,
    )

    e100e200_pct = safe_div((ema100 - ema200) * 100.0, ema200)

    e100e200_near = e100e200_pct.notna() & (e100e200_pct.abs() <= float(cfg["e100e200_near_pct"]))
    e100e200_wide = e100e200_pct.notna() & (e100e200_pct.abs() >= float(cfg["e100e200_wide_pct"]))

    slow_anchor_bias = pd.Series(
        np.where(
            (ema100 > ema200) & (ema100_slope_dir >= 0) & (ema200_slope_dir >= 0),
            1,
            np.where(
                (ema100 < ema200) & (ema100_slope_dir <= 0) & (ema200_slope_dir <= 0),
                -1,
                0,
            ),
        ),
        index=idx,
        dtype=int,
    )

    slow_anchor_state = pd.Series(
        np.where(e100e200_near, 1, np.where(e100e200_wide, 2, 0)),
        index=idx,
        dtype=int,
    )

    # =========================================================================
    # FAST BAND / STACK ENGINE
    # =========================================================================

    spread_lookback = max(1, int(cfg["spread_lookback"]))

    bull1420_raw = ema14 > ema20
    bear1420_raw = ema14 < ema20

    bull1420 = bull1420_raw & (ema14_slope_dir >= 0) & (ema20_slope_dir >= 0)
    bear1420 = bear1420_raw & (ema14_slope_dir <= 0) & (ema20_slope_dir <= 0)

    spread1420_now = (ema14 - ema20).abs()
    spread1420_pct = safe_div(spread1420_now * 100.0, ema20)
    spread1420_prev = spread1420_pct.shift(spread_lookback)

    has1420_spread = spread1420_pct.notna() & spread1420_prev.notna()
    wide1420 = has1420_spread & (spread1420_pct >= float(cfg["exp_1420_min_pct"]))
    growing1420 = has1420_spread & ((spread1420_pct - spread1420_prev) >= float(cfg["exp_grow_pct"]))
    shrinking1420 = has1420_spread & ((spread1420_pct - spread1420_prev) < 0)

    ema1420_state = pd.Series(
        np.where(bull1420, 1, np.where(bear1420, -1, 0)),
        index=idx,
        dtype=int,
    )
    ema1420_exp = pd.Series(
        np.where(wide1420 & growing1420, 2, np.where(wide1420, 1, 0)),
        index=idx,
        dtype=int,
    )

    band1420_flat = (
        ema14_slope_pct.notna()
        & ema20_slope_pct.notna()
        & (ema14_slope_pct.abs() <= flat_band_pct)
        & (ema20_slope_pct.abs() <= flat_band_pct)
    )

    bull3350_raw = ema33 > ema50
    bear3350_raw = ema33 < ema50

    bull3350_strict = bull3350_raw & (ema33_slope_dir >= 0) & (ema50_slope_dir >= 0)
    bear3350_strict = bear3350_raw & (ema33_slope_dir <= 0) & (ema50_slope_dir <= 0)

    spread3350_now = (ema33 - ema50).abs()
    spread3350_pct = safe_div(spread3350_now * 100.0, ema50)
    spread3350_prev = spread3350_pct.shift(spread_lookback)

    has3350_spread = spread3350_pct.notna() & spread3350_prev.notna()
    wide3350 = has3350_spread & (spread3350_pct >= float(cfg["exp_3350_min_pct"]))
    growing3350 = has3350_spread & ((spread3350_pct - spread3350_prev) >= float(cfg["exp_grow_pct"]))
    shrinking3350 = has3350_spread & ((spread3350_pct - spread3350_prev) < 0)

    band3350_flat = (
        ema33_slope_pct.notna()
        & ema50_slope_pct.notna()
        & (ema33_slope_pct.abs() <= flat_band_pct)
        & (ema50_slope_pct.abs() <= flat_band_pct)
    )

    bull3350_soft = bull3350_raw & (~band3350_flat) & (close >= band_lo)
    bear3350_soft = bear3350_raw & (~band3350_flat) & (close <= band_hi)

    if bool(cfg["strict_trend_mode"]):
        ema_band_state = pd.Series(
            np.where(bull3350_strict, 1, np.where(bear3350_strict, -1, 0)),
            index=idx,
            dtype=int,
        )
    else:
        ema_band_state = pd.Series(
            np.where(bull3350_soft, 1, np.where(bear3350_soft, -1, 0)),
            index=idx,
            dtype=int,
        )

    bull_stack_fast = (ema14 > ema20) & (ema20 > ema33) & (ema33 > ema50)
    bear_stack_fast = (ema14 < ema20) & (ema20 < ema33) & (ema33 < ema50)

    bull_stack_slow = (ema50 > ema100) & (ema100 > ema200)
    bear_stack_slow = (ema50 < ema100) & (ema100 < ema200)

    bull_stack_all = bull_stack_fast & bull_stack_slow
    bear_stack_all = bear_stack_fast & bear_stack_slow

    # =========================================================================
    # EMA BEHAVIOR ENGINE
    # =========================================================================

    ema_behavior_dir = pd.Series(
        np.where(
            bull_stack_all,
            1,
            np.where(
                bear_stack_all,
                -1,
                np.where(
                    bull_stack_fast & (slow_anchor_bias >= 0) & (ema_band_state == 1),
                    1,
                    np.where(
                        bear_stack_fast & (slow_anchor_bias <= 0) & (ema_band_state == -1),
                        -1,
                        ema_band_state,
                    ),
                ),
            ),
        ),
        index=idx,
        dtype=int,
    )

    ema_bull_expansion = (
        (ema_behavior_dir == 1)
        & (ema1420_state == 1)
        & (ema_band_state == 1)
        & (ema1420_exp >= 1)
        & wide3350
        & (ema20_slope_dir >= 0)
        & (ema100_slope_dir >= 0)
        & (slow_anchor_bias >= 0)
        & (~e20200_near)
    )

    ema_bear_expansion = (
        (ema_behavior_dir == -1)
        & (ema1420_state == -1)
        & (ema_band_state == -1)
        & (ema1420_exp >= 1)
        & wide3350
        & (ema20_slope_dir <= 0)
        & (ema100_slope_dir <= 0)
        & (slow_anchor_bias <= 0)
        & (~e20200_near)
    )

    ema_compression = (
        (band3350_flat | (~wide1420) | e20200_near | e20200_flat)
        & ((~wide3350) | band3350_flat)
    )

    bull_decay = (
        (ema_behavior_dir == 1)
        & (
            shrinking1420
            | shrinking3350
            | (ema14 <= ema20)
            | (close < ema20)
            | (ema20_slope_dir < 0)
            | ((ema100_slope_dir <= 0) & (slow_anchor_bias <= 0))
        )
    )

    bear_decay = (
        (ema_behavior_dir == -1)
        & (
            shrinking1420
            | shrinking3350
            | (ema14 >= ema20)
            | (close > ema20)
            | (ema20_slope_dir > 0)
            | ((ema100_slope_dir >= 0) & (slow_anchor_bias >= 0))
        )
    )

    ema_behavior_type = pd.Series(
        np.where(
            ema_compression,
            1,
            np.where(ema_bull_expansion | ema_bear_expansion, 2, np.where(bull_decay | bear_decay, 3, 0)),
        ),
        index=idx,
        dtype=int,
    )

    ema_fast_compression = pd.Series(np.where(band1420_flat | (~wide1420), 1, 0), index=idx, dtype=int)

    ema_fast_decay = pd.Series(
        np.where(bull_decay, 1, np.where(bear_decay, -1, 0)),
        index=idx,
        dtype=int,
    )

    ema_slow_compression = pd.Series(np.where(e20200_near | e20200_flat, 1, 0), index=idx, dtype=int)

    ema_slow_expansion = pd.Series(
        np.where(
            (ema_behavior_dir == 1) & e20200_expand & (ema20_slope_dir >= 0) & (ema200_slope_dir >= 0),
            1,
            np.where(
                (ema_behavior_dir == -1) & e20200_expand & (ema20_slope_dir <= 0) & (ema200_slope_dir <= 0),
                -1,
                0,
            ),
        ),
        index=idx,
        dtype=int,
    )

    ema_trend_quality = pd.Series(
        np.where(
            ema_behavior_dir == 0,
            0,
            np.where(
                (ema_behavior_type == 2) & (e20_to_e200_pct.abs() >= float(cfg["e20200_wide_pct"])),
                2,
                np.where((ema_behavior_type == 2) | (ema_band_state != 0), 1, 0),
            ),
        ),
        index=idx,
        dtype=int,
    )

    local_ema_dir = ema_behavior_dir.copy()

    # =========================================================================
    # ACCEPTANCE / MATURITY ENGINE
    # =========================================================================

    bull_accept20 = (close > ema20) & (ema20 >= ema20.shift(1))
    bear_accept20 = (close < ema20) & (ema20 <= ema20.shift(1))

    bull_accept_band = (close >= band_lo) & (ema20 >= ema33)
    bear_accept_band = (close <= band_hi) & (ema20 <= ema33)

    bull_struct = (ema20 > ema33) & (ema33 >= ema50)
    bear_struct = (ema20 < ema33) & (ema33 <= ema50)

    mature_bull = (close > ema200) & (ema20 > ema50) & (e20_to_e200_pct > 0) & (slow_anchor_bias >= 0)
    mature_bear = (close < ema200) & (ema20 < ema50) & (e20_to_e200_pct < 0) & (slow_anchor_bias <= 0)

    early_bull = bull_accept20 & bull_accept_band
    early_bear = bear_accept20 & bear_accept_band

    # =========================================================================
    # EMA RETEST ENGINE
    # =========================================================================

    rt_pad = atr_dist * float(cfg["rt_pad_atr_mult"])

    zone1420_lo = pd.concat([ema14, ema20], axis=1).min(axis=1)
    zone1420_hi = pd.concat([ema14, ema20], axis=1).max(axis=1)

    zone3350_lo = pd.concat([ema33, ema50], axis=1).min(axis=1)
    zone3350_hi = pd.concat([ema33, ema50], axis=1).max(axis=1)

    zone100200_lo = pd.concat([ema100, ema200], axis=1).min(axis=1)
    zone100200_hi = pd.concat([ema100, ema200], axis=1).max(axis=1)

    rt_on = bool(cfg["rt_on"])
    rt_lookback_bars = max(1, int(cfg["rt_lookback_bars"]))
    rt_require_close = bool(cfg["rt_require_close"])

    touch1420 = rt_on & (low <= (zone1420_hi + rt_pad)) & (high >= (zone1420_lo - rt_pad))
    touch3350 = rt_on & (low <= (zone3350_hi + rt_pad)) & (high >= (zone3350_lo - rt_pad))
    touch100200 = rt_on & (low <= (zone100200_hi + rt_pad)) & (high >= (zone100200_lo - rt_pad))

    bull_retest1420_raw = touch1420 & (local_ema_dir == 1) & ((~rt_require_close) | (close >= zone1420_lo))
    bear_retest1420_raw = touch1420 & (local_ema_dir == -1) & ((~rt_require_close) | (close <= zone1420_hi))

    bull_retest3350_raw = touch3350 & (local_ema_dir == 1) & ((~rt_require_close) | (close >= zone3350_lo))
    bear_retest3350_raw = touch3350 & (local_ema_dir == -1) & ((~rt_require_close) | (close <= zone3350_hi))

    bull_retest100200_raw = touch100200 & (local_ema_dir == 1) & ((~rt_require_close) | (close >= zone100200_lo))
    bear_retest100200_raw = touch100200 & (local_ema_dir == -1) & ((~rt_require_close) | (close <= zone100200_hi))

    rt_ema1420_ok = recent_event_memory(
        bull_raw=bull_retest1420_raw,
        bear_raw=bear_retest1420_raw,
        direction=local_ema_dir,
        lookback_bars=rt_lookback_bars,
        index=idx,
    )

    rt_ema3350_ok = recent_event_memory(
        bull_raw=bull_retest3350_raw,
        bear_raw=bear_retest3350_raw,
        direction=local_ema_dir,
        lookback_bars=rt_lookback_bars,
        index=idx,
    )

    rt_ema100200_ok = recent_event_memory(
        bull_raw=bull_retest100200_raw,
        bear_raw=bear_retest100200_raw,
        direction=local_ema_dir,
        lookback_bars=rt_lookback_bars,
        index=idx,
    )

    rt_ema1420_dir = pd.Series(np.where(rt_ema1420_ok, local_ema_dir, 0), index=idx, dtype=int)
    rt_ema3350_dir = pd.Series(np.where(rt_ema3350_ok, local_ema_dir, 0), index=idx, dtype=int)
    rt_ema100200_dir = pd.Series(np.where(rt_ema100200_ok, local_ema_dir, 0), index=idx, dtype=int)

    active_rt_family = pd.Series(
        np.where(rt_ema1420_ok, 1, np.where(rt_ema3350_ok, 2, np.where(rt_ema100200_ok, 3, 0))),
        index=idx,
        dtype=int,
    )

    any_rt_ema = (rt_ema1420_ok | rt_ema3350_ok | rt_ema100200_ok).astype(int)

    # =========================================================================
    # RECLAIM ENGINE
    # =========================================================================

    prior_high = rolling_prior_high(high, int(cfg["reclaim_lookback"]))
    prior_low = rolling_prior_low(low, int(cfg["reclaim_lookback"]))

    reclaim_on = bool(cfg["reclaim_on"])

    bull_reclaim20 = reclaim_on & rt_ema1420_ok & (close > ema20) & (close > prior_high)
    bear_reclaim20 = reclaim_on & rt_ema1420_ok & (close < ema20) & (close < prior_low)

    bull_reclaim_band = reclaim_on & rt_ema3350_ok & (close > band_hi) & (close > prior_high)
    bear_reclaim_band = reclaim_on & rt_ema3350_ok & (close < band_lo) & (close < prior_low)

    bull_reclaim100200 = reclaim_on & rt_ema100200_ok & (close > zone100200_hi) & (close > prior_high)
    bear_reclaim100200 = reclaim_on & rt_ema100200_ok & (close < zone100200_lo) & (close < prior_low)

    reclaim20_dir = pd.Series(np.where(bull_reclaim20, 1, np.where(bear_reclaim20, -1, 0)), index=idx, dtype=int)
    reclaim_band_dir = pd.Series(np.where(bull_reclaim_band, 1, np.where(bear_reclaim_band, -1, 0)), index=idx, dtype=int)
    reclaim100200_dir = pd.Series(np.where(bull_reclaim100200, 1, np.where(bear_reclaim100200, -1, 0)), index=idx, dtype=int)

    # =========================================================================
    # MTF EMA AVERAGE ENGINE
    # =========================================================================

    mtf_source_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=idx,
    )

    mtf = build_mtf_resample_states(mtf_source_df)

    mtf1 = mtf["mtf1"]
    mtf5 = mtf["mtf5"]
    mtf15 = mtf["mtf15"]
    mtf60 = mtf["mtf60"]
    mtf240 = mtf["mtf240"]
    mtfD = mtf["mtfD"]

    mtf_avg_dir = (mtf1 + mtf5 + mtf15 + mtf60 + mtf240 + mtfD) / 6.0

    # =========================================================================
    # FINAL ZONE ENGINE
    # =========================================================================

    local_dir_score = pd.Series(
        np.where(local_ema_dir == 1, 1.0, np.where(local_ema_dir == -1, -1.0, 0.0)),
        index=idx,
        dtype=float,
    )

    if bool(cfg["use_mtf_confirm"]):
        final_dir_score = (
            local_dir_score * float(cfg["mtf_weight_local"])
            + mtf_avg_dir * float(cfg["mtf_weight_avg"])
        )
    else:
        final_dir_score = local_dir_score.copy()

    final_ema_zone_dir = pd.Series(
        np.where(
            final_dir_score >= float(cfg["mtf_bull_thresh"]),
            1,
            np.where(final_dir_score <= float(cfg["mtf_bear_thresh"]), -1, 0),
        ),
        index=idx,
        dtype=int,
    )

    final_zone_dir = pd.Series(
        np.where(ema_compression & (final_dir_score.abs() < 0.50), 0, final_ema_zone_dir),
        index=idx,
        dtype=int,
    )

    # =========================================================================
    # OUTPUT CONTRACT
    # =========================================================================

    contract = pd.DataFrame(index=idx)

    contract["sc_ema_final_dir"] = final_zone_dir.astype(int)
    contract["sc_ema_zone_dir"] = final_zone_dir.astype(int)
    contract["sc_ema_local_dir"] = local_ema_dir.astype(int)
    contract["sc_ema_mtf1"] = mtf1.astype(float)
    contract["sc_ema_mtf5"] = mtf5.astype(float)
    contract["sc_ema_mtf15"] = mtf15.astype(float)
    contract["sc_ema_mtf60"] = mtf60.astype(float)
    contract["sc_ema_mtf240"] = mtf240.astype(float)
    contract["sc_ema_mtfD"] = mtfD.astype(float)
    contract["sc_ema_mtf_avg_dir"] = mtf_avg_dir.astype(float)
    contract["sc_ema_final_dir_score"] = final_dir_score.astype(float)

    contract["sc_ema_behavior_dir"] = ema_behavior_dir.astype(int)
    contract["sc_ema_behavior_type"] = ema_behavior_type.astype(int)
    contract["sc_ema_fast_compression"] = ema_fast_compression.astype(int)
    contract["sc_ema_fast_decay"] = ema_fast_decay.astype(int)
    contract["sc_ema_slow_compression"] = ema_slow_compression.astype(int)
    contract["sc_ema_slow_expansion"] = ema_slow_expansion.astype(int)
    contract["sc_ema_trend_quality"] = ema_trend_quality.astype(int)
    contract["sc_ema_compression"] = ema_compression.astype(int)

    contract["sc_ema_band_state"] = ema_band_state.astype(int)
    contract["sc_ema_1420_state"] = ema1420_state.astype(int)
    contract["sc_ema_1420_exp"] = ema1420_exp.astype(int)

    contract["sc_bull_stack_fast"] = bull_stack_fast.astype(int)
    contract["sc_bear_stack_fast"] = bear_stack_fast.astype(int)
    contract["sc_bull_stack_slow"] = bull_stack_slow.astype(int)
    contract["sc_bear_stack_slow"] = bear_stack_slow.astype(int)
    contract["sc_bull_stack_all"] = bull_stack_all.astype(int)
    contract["sc_bear_stack_all"] = bear_stack_all.astype(int)

    contract["sc_bull_accept20"] = bull_accept20.astype(int)
    contract["sc_bear_accept20"] = bear_accept20.astype(int)
    contract["sc_bull_accept_band"] = bull_accept_band.astype(int)
    contract["sc_bear_accept_band"] = bear_accept_band.astype(int)
    contract["sc_bull_struct"] = bull_struct.astype(int)
    contract["sc_bear_struct"] = bear_struct.astype(int)
    contract["sc_mature_bull"] = mature_bull.astype(int)
    contract["sc_mature_bear"] = mature_bear.astype(int)
    contract["sc_early_bull"] = early_bull.astype(int)
    contract["sc_early_bear"] = early_bear.astype(int)

    contract["sc_ema_rt_1420"] = rt_ema1420_ok.astype(int)
    contract["sc_ema_rt_3350"] = rt_ema3350_ok.astype(int)
    contract["sc_ema_rt_100200"] = rt_ema100200_ok.astype(int)

    contract["sc_ema_rt_1420_dir"] = rt_ema1420_dir.astype(int)
    contract["sc_ema_rt_3350_dir"] = rt_ema3350_dir.astype(int)
    contract["sc_ema_rt_100200_dir"] = rt_ema100200_dir.astype(int)

    contract["sc_ema_rt_any"] = any_rt_ema.astype(int)
    contract["sc_ema_rt_family"] = active_rt_family.astype(int)

    contract["sc_ema_reclaim_20"] = reclaim20_dir.astype(int)
    contract["sc_ema_reclaim_3350"] = reclaim_band_dir.astype(int)
    contract["sc_ema_reclaim_100200"] = reclaim100200_dir.astype(int)

    contract["sc_price_to_ema20_pts"] = p_to_e20_pts.astype(float)
    contract["sc_price_to_ema20_pct"] = p_to_e20_pct.astype(float)
    contract["sc_price_to_ema20_signed"] = p_to_e20_signed.astype(float)

    contract["sc_price_to_ema200_pts"] = p_to_e200_pts.astype(float)
    contract["sc_price_to_ema200_pct"] = p_to_e200_pct.astype(float)
    contract["sc_price_to_ema200_signed"] = p_to_e200_signed.astype(float)

    contract["sc_ema20_to_ema200_pts"] = e20_to_e200_pts.astype(float)
    contract["sc_ema20_to_ema200_pct"] = e20_to_e200_pct.astype(float)
    contract["sc_ema20_to_ema200_signed"] = e20_to_e200_signed.astype(float)

    contract["sc_is_stretched_from20"] = is_stretched_from20.astype(int)
    contract["sc_is_stretched_from200"] = is_stretched_from200.astype(int)
    contract["sc_dist20_dir"] = dist20_dir.astype(int)
    contract["sc_dist200_dir"] = dist200_dir.astype(int)

    contract["sc_e20200_state"] = e20200_state.astype(int)
    contract["sc_slow_anchor_bias"] = slow_anchor_bias.astype(int)
    contract["sc_slow_anchor_state"] = slow_anchor_state.astype(int)

    contract["sc_ema14_slope_pct"] = ema14_slope_pct.astype(float)
    contract["sc_ema20_slope_pct"] = ema20_slope_pct.astype(float)
    contract["sc_ema33_slope_pct"] = ema33_slope_pct.astype(float)
    contract["sc_ema50_slope_pct"] = ema50_slope_pct.astype(float)
    contract["sc_ema100_slope_pct"] = ema100_slope_pct.astype(float)
    contract["sc_ema200_slope_pct"] = ema200_slope_pct.astype(float)

    contract["sc_ema14_slope_dir"] = ema14_slope_dir.astype(int)
    contract["sc_ema20_slope_dir"] = ema20_slope_dir.astype(int)
    contract["sc_ema33_slope_dir"] = ema33_slope_dir.astype(int)
    contract["sc_ema50_slope_dir"] = ema50_slope_dir.astype(int)
    contract["sc_ema100_slope_dir"] = ema100_slope_dir.astype(int)
    contract["sc_ema200_slope_dir"] = ema200_slope_dir.astype(int)

    contract["sc_ema_fast_slope_state"] = ema_slope_fast_state.astype(int)
    contract["sc_ema_band_slope_state"] = ema_slope_band_state.astype(int)
    contract["sc_ema_slow_slope_state"] = ema_slope_slow_state.astype(int)

    contract["ema14"] = ema14.astype(float)
    contract["ema20"] = ema20.astype(float)
    contract["ema33"] = ema33.astype(float)
    contract["ema50"] = ema50.astype(float)
    contract["ema100"] = ema100.astype(float)
    contract["ema200"] = ema200.astype(float)
    contract["band_lo"] = band_lo.astype(float)
    contract["band_hi"] = band_hi.astype(float)
    contract["atr_dist"] = atr_dist.astype(float)

    contract["ema_dir"] = contract["sc_ema_final_dir"].fillna(0).astype(int)
    contract["ema_quality"] = pd.Series(
        np.where(
            contract["sc_ema_trend_quality"] >= 2,
            1.0,
            np.where(contract["sc_ema_trend_quality"] == 1, 0.65, 0.0),
        ),
        index=idx,
        dtype=float,
    )

    contract["ema_stack_bull"] = contract["sc_bull_stack_all"].fillna(0).astype(int)
    contract["ema_stack_bear"] = contract["sc_bear_stack_all"].fillna(0).astype(int)

    contract["rt_ema_1420"] = contract["sc_ema_rt_1420"].fillna(0).astype(int)
    contract["rt_ema_3350"] = contract["sc_ema_rt_3350"].fillna(0).astype(int)
    contract["rt_ema_100200"] = contract["sc_ema_rt_100200"].fillna(0).astype(int)

    contract["ema_rt_any"] = contract["sc_ema_rt_any"].fillna(0).astype(int)
    contract["ema_rt_family"] = contract["sc_ema_rt_family"].fillna(0).astype(int)
    contract["ema_rt_dir"] = pd.Series(
        np.where(
            contract["sc_ema_rt_1420"] == 1,
            contract["sc_ema_rt_1420_dir"],
            np.where(
                contract["sc_ema_rt_3350"] == 1,
                contract["sc_ema_rt_3350_dir"],
                np.where(contract["sc_ema_rt_100200"] == 1, contract["sc_ema_rt_100200_dir"], 0),
            ),
        ),
        index=idx,
        dtype=int,
    )

    contract["ema_slope_fast"] = contract["sc_ema_fast_slope_state"].fillna(0).astype(int)
    contract["ema_slope_band"] = contract["sc_ema_band_slope_state"].fillna(0).astype(int)
    contract["ema_slope_slow"] = contract["sc_ema_slow_slope_state"].fillna(0).astype(int)

    contract["ema14_slope"] = contract["sc_ema14_slope_dir"].fillna(0).astype(int)
    contract["ema20_slope"] = contract["sc_ema20_slope_dir"].fillna(0).astype(int)
    contract["ema33_slope"] = contract["sc_ema33_slope_dir"].fillna(0).astype(int)
    contract["ema50_slope"] = contract["sc_ema50_slope_dir"].fillna(0).astype(int)
    contract["ema100_slope"] = contract["sc_ema100_slope_dir"].fillna(0).astype(int)
    contract["ema200_slope"] = contract["sc_ema200_slope_dir"].fillna(0).astype(int)

    contract["price_to_ema20_pct"] = contract["sc_price_to_ema20_pct"].astype(float)
    contract["price_to_ema20_pts"] = contract["sc_price_to_ema20_pts"].astype(float)
    contract["ema20_to_ema200_pct"] = contract["sc_ema20_to_ema200_pct"].astype(float)
    contract["ema20_to_ema200_pts"] = contract["sc_ema20_to_ema200_pts"].astype(float)

    out = pd.concat([out, contract], axis=1)
    return out


def build_ema(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return compute_ema_engine(df, config=config)


def run_ema_engine(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return compute_ema_engine(df, config=config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EMA engine on CSV for parity/debug.")
    parser.add_argument("csv_path", nargs="?", default="data/xauusd_mt5_m1.csv")
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

    result = run_ema_engine(df)

    cols = [
        "sc_ema_final_dir",
        "sc_ema_local_dir",
        "sc_ema_mtf1",
        "sc_ema_mtf5",
        "sc_ema_mtf15",
        "sc_ema_mtf60",
        "sc_ema_mtf240",
        "sc_ema_mtfD",
        "sc_ema_mtf_avg_dir",
        "sc_ema_behavior_dir",
        "sc_ema_behavior_type",
        "sc_ema_rt_1420",
        "sc_ema_rt_3350",
        "sc_ema_rt_100200",
        "sc_ema_reclaim_20",
        "sc_ema_reclaim_3350",
        "sc_ema_reclaim_100200",
        "sc_price_to_ema20_pct",
        "sc_ema20_to_ema200_pct",
    ]
    print(result[cols].tail(args.tail).to_string())