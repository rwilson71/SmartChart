from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_EMA_CONFIG: Dict[str, Any] = {
    "slope_len": 5,
    "spread_lookback": 10,
    "exp_1420_min_pct": 0.02,
    "exp_3350_min_pct": 0.05,
    "exp_grow_pct": 0.01,
    "flat_band_pct": 0.02,
    "strict_trend_mode": False,
    "use_mtf_confirm": True,
    "mtf_weight_local": 0.60,
    "mtf_weight_avg": 0.40,
    "mtf_bull_thresh": 0.20,
    "mtf_bear_thresh": -0.20,
    "dist_atr_len": 14,
    "stretch20_atr": 1.00,
    "stretch200_atr": 2.00,
    "rt_on": True,
    "rt_lookback_bars": 12,
    "rt_pad_atr_mult": 0.15,
    "rt_require_close": True,
    "reclaim_on": True,
    "reclaim_lookback": 3,
    "e20200_lookback": 10,
    "e20200_near_pct": 0.15,
    "e20200_wide_pct": 0.60,
    "e20200_grow_pct": 0.05,
    "e20200_flat_pct": 0.02,
    "e100e200_near_pct": 0.10,
    "e100e200_wide_pct": 0.35,
}


def _require_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EMA engine requires columns: {sorted(required)}. Missing: {sorted(missing)}")


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length, min_periods=length).mean()


def _safe_pct_diff(now: pd.Series, base: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=now.index, dtype=float)
    mask = base != 0.0
    out.loc[mask] = ((now.loc[mask] - base.loc[mask]) / base.loc[mask]) * 100.0
    return out


def _dir_from_value(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(series > 0, 1, np.where(series < 0, -1, 0)),
        index=series.index,
        dtype=float,
    )


def _highest_prev(series: pd.Series, lookback: int) -> pd.Series:
    return series.shift(1).rolling(lookback, min_periods=lookback).max()


def _lowest_prev(series: pd.Series, lookback: int) -> pd.Series:
    return series.shift(1).rolling(lookback, min_periods=lookback).min()


def _rolling_state_memory(raw_signal: pd.Series, lookback: int) -> pd.Series:
    last_true_idx = None
    out = []

    vals = raw_signal.fillna(False).astype(bool).tolist()
    for i, is_true in enumerate(vals):
        if is_true:
            last_true_idx = i
        out.append(last_true_idx is not None and (i - last_true_idx <= lookback))

    return pd.Series(out, index=raw_signal.index, dtype=bool)


def _mtf_dir_from_resample(df: pd.DataFrame, rule: str) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(0.0, index=df.index)

    agg = (
        df[["open", "high", "low", "close"]]
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    e14 = _ema(agg["close"], 14)
    e20 = _ema(agg["close"], 20)
    e33 = _ema(agg["close"], 33)
    e50 = _ema(agg["close"], 50)
    e100 = _ema(agg["close"], 100)
    e200 = _ema(agg["close"], 200)

    bull_fast = (e14 > e20) & (e20 > e33) & (e33 > e50)
    bear_fast = (e14 < e20) & (e20 < e33) & (e33 < e50)
    bull_slow = (e50 > e100) & (e100 > e200)
    bear_slow = (e50 < e100) & (e100 < e200)

    band_bull = e33 > e50
    band_bear = e33 < e50

    direction = pd.Series(0.0, index=agg.index, dtype=float)
    direction.loc[bull_fast & bull_slow] = 1.0
    direction.loc[bear_fast & bear_slow] = -1.0
    direction.loc[(direction == 0.0) & bull_fast & (e100 >= e200)] = 1.0
    direction.loc[(direction == 0.0) & bear_fast & (e100 <= e200)] = -1.0
    direction.loc[(direction == 0.0) & band_bull] = 0.5
    direction.loc[(direction == 0.0) & band_bear] = -0.5

    return direction.reindex(df.index, method="ffill").fillna(0.0)


def build_ema(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    _require_ohlc(df)

    cfg = {**DEFAULT_EMA_CONFIG, **(config or {})}
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]

    ema14 = _ema(close, 14)
    ema20 = _ema(close, 20)
    ema33 = _ema(close, 33)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)
    ema200 = _ema(close, 200)

    out["ema14"] = ema14
    out["ema20"] = ema20
    out["ema33"] = ema33
    out["ema50"] = ema50
    out["ema100"] = ema100
    out["ema200"] = ema200

    band_lo = np.minimum(ema33, ema50)
    band_hi = np.maximum(ema33, ema50)

    atr_dist = _atr(out, cfg["dist_atr_len"])
    out["atr_dist"] = atr_dist

    p_to_e20_pct = _safe_pct_diff(close, ema20)
    p_to_e200_pct = _safe_pct_diff(close, ema200)
    e20_to_e200_pct = _safe_pct_diff(ema20, ema200)

    out["sc_price_to_ema20_pct"] = p_to_e20_pct
    out["sc_price_to_ema200_pct"] = p_to_e200_pct
    out["sc_ema20_to_ema200_pct"] = e20_to_e200_pct

    out["sc_is_stretched_from20"] = ((close - ema20).abs() / atr_dist) >= cfg["stretch20_atr"]
    out["sc_is_stretched_from200"] = ((close - ema200).abs() / atr_dist) >= cfg["stretch200_atr"]

    slope_len = cfg["slope_len"]
    ema14_slope_pct = _safe_pct_diff(ema14, ema14.shift(slope_len))
    ema20_slope_pct = _safe_pct_diff(ema20, ema20.shift(slope_len))
    ema33_slope_pct = _safe_pct_diff(ema33, ema33.shift(slope_len))
    ema50_slope_pct = _safe_pct_diff(ema50, ema50.shift(slope_len))
    ema100_slope_pct = _safe_pct_diff(ema100, ema100.shift(slope_len))
    ema200_slope_pct = _safe_pct_diff(ema200, ema200.shift(slope_len))

    ema14_slope_dir = _dir_from_value(ema14_slope_pct)
    ema20_slope_dir = _dir_from_value(ema20_slope_pct)
    ema33_slope_dir = _dir_from_value(ema33_slope_pct)
    ema50_slope_dir = _dir_from_value(ema50_slope_pct)
    ema100_slope_dir = _dir_from_value(ema100_slope_pct)
    ema200_slope_dir = _dir_from_value(ema200_slope_pct)

    e20e200_prev = e20_to_e200_pct.shift(cfg["e20200_lookback"])
    e20e200_grow = e20_to_e200_pct - e20e200_prev

    e20200_near = e20_to_e200_pct.abs() <= cfg["e20200_near_pct"]
    e20200_wide = e20_to_e200_pct.abs() >= cfg["e20200_wide_pct"]
    e20200_expand = e20200_wide & (e20e200_grow.abs() >= cfg["e20200_grow_pct"])
    e20200_flat = (ema20_slope_pct.abs() <= cfg["e20200_flat_pct"]) & (ema200_slope_pct.abs() <= cfg["e20200_flat_pct"])

    out["e20200_state"] = np.select(
        [e20200_flat, e20200_expand, e20200_near, e20200_wide],
        [4, 3, 1, 2],
        default=0,
    )

    e100e200_pct = _safe_pct_diff(ema100, ema200)
    e100e200_near = e100e200_pct.abs() <= cfg["e100e200_near_pct"]
    e100e200_wide = e100e200_pct.abs() >= cfg["e100e200_wide_pct"]

    slow_anchor_bias = np.select(
        [
            (ema100 > ema200) & (ema100_slope_dir >= 0) & (ema200_slope_dir >= 0),
            (ema100 < ema200) & (ema100_slope_dir <= 0) & (ema200_slope_dir <= 0),
        ],
        [1, -1],
        default=0,
    )
    out["slow_anchor_bias"] = slow_anchor_bias

    out["slow_anchor_state"] = np.select(
        [e100e200_near, e100e200_wide],
        [1, 2],
        default=0,
    )

    bull1420_raw = ema14 > ema20
    bear1420_raw = ema14 < ema20

    bull1420 = bull1420_raw & (ema14_slope_dir >= 0) & (ema20_slope_dir >= 0)
    bear1420 = bear1420_raw & (ema14_slope_dir <= 0) & (ema20_slope_dir <= 0)

    spread1420_pct = ((ema14 - ema20).abs() / ema20) * 100.0
    spread1420_prev = spread1420_pct.shift(cfg["spread_lookback"])
    has1420_spread = spread1420_pct.notna() & spread1420_prev.notna()
    wide1420 = has1420_spread & (spread1420_pct >= cfg["exp_1420_min_pct"])
    growing1420 = has1420_spread & ((spread1420_pct - spread1420_prev) >= cfg["exp_grow_pct"])
    shrinking1420 = has1420_spread & ((spread1420_pct - spread1420_prev) < 0)

    ema1420_state = np.select([bull1420, bear1420], [1, -1], default=0)
    ema1420_exp = np.select([wide1420 & growing1420, wide1420], [2, 1], default=0)

    band1420_flat = (ema14_slope_pct.abs() <= cfg["flat_band_pct"]) & (ema20_slope_pct.abs() <= cfg["flat_band_pct"])

    bull3350_raw = ema33 > ema50
    bear3350_raw = ema33 < ema50

    bull3350_strict = bull3350_raw & (ema33_slope_dir >= 0) & (ema50_slope_dir >= 0)
    bear3350_strict = bear3350_raw & (ema33_slope_dir <= 0) & (ema50_slope_dir <= 0)

    spread3350_pct = ((ema33 - ema50).abs() / ema50) * 100.0
    spread3350_prev = spread3350_pct.shift(cfg["spread_lookback"])
    has3350_spread = spread3350_pct.notna() & spread3350_prev.notna()
    wide3350 = has3350_spread & (spread3350_pct >= cfg["exp_3350_min_pct"])
    shrinking3350 = has3350_spread & ((spread3350_pct - spread3350_prev) < 0)

    band3350_flat = (ema33_slope_pct.abs() <= cfg["flat_band_pct"]) & (ema50_slope_pct.abs() <= cfg["flat_band_pct"])

    bull3350_soft = bull3350_raw & (~band3350_flat) & (close >= band_lo)
    bear3350_soft = bear3350_raw & (~band3350_flat) & (close <= band_hi)

    ema_band_state = np.select(
        [bull3350_strict, bear3350_strict],
        [1, -1],
        default=0,
    ) if cfg["strict_trend_mode"] else np.select(
        [bull3350_soft, bear3350_soft],
        [1, -1],
        default=0,
    )

    bull_stack_fast = (ema14 > ema20) & (ema20 > ema33) & (ema33 > ema50)
    bear_stack_fast = (ema14 < ema20) & (ema20 < ema33) & (ema33 < ema50)
    bull_stack_slow = (ema50 > ema100) & (ema100 > ema200)
    bear_stack_slow = (ema50 < ema100) & (ema100 < ema200)

    bull_stack_all = bull_stack_fast & bull_stack_slow
    bear_stack_all = bear_stack_fast & bear_stack_slow

    ema_behavior_dir = np.select(
        [
            bull_stack_all,
            bear_stack_all,
            bull_stack_fast & (slow_anchor_bias >= 0) & (ema_band_state == 1),
            bear_stack_fast & (slow_anchor_bias <= 0) & (ema_band_state == -1),
        ],
        [1, -1, 1, -1],
        default=ema_band_state,
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

    ema_behavior_type = np.select(
        [ema_compression, (ema_bull_expansion | ema_bear_expansion), (bull_decay | bear_decay)],
        [1, 2, 3],
        default=0,
    )

    ema_fast_compression = np.where((band1420_flat | (~wide1420)), 1, 0)
    ema_fast_decay = np.select([bull_decay, bear_decay], [1, -1], default=0)
    ema_slow_compression = np.where((e20200_near | e20200_flat), 1, 0)
    ema_slow_expansion = np.select(
        [
            (ema_behavior_dir == 1) & e20200_expand & (ema20_slope_dir >= 0) & (ema200_slope_dir >= 0),
            (ema_behavior_dir == -1) & e20200_expand & (ema20_slope_dir <= 0) & (ema200_slope_dir <= 0),
        ],
        [1, -1],
        default=0,
    )

    ema_trend_quality = np.select(
        [
            ema_behavior_dir == 0,
            (ema_behavior_type == 2) & (e20_to_e200_pct.abs() >= cfg["e20200_wide_pct"]),
            (ema_behavior_type == 2) | (ema_band_state != 0),
        ],
        [0, 2, 1],
        default=0,
    )

    local_ema_dir = ema_behavior_dir

    rt_pad = atr_dist * cfg["rt_pad_atr_mult"]

    zone1420_lo = np.minimum(ema14, ema20)
    zone1420_hi = np.maximum(ema14, ema20)
    zone3350_lo = np.minimum(ema33, ema50)
    zone3350_hi = np.maximum(ema33, ema50)
    zone100200_lo = np.minimum(ema100, ema200)
    zone100200_hi = np.maximum(ema100, ema200)

    touch1420 = (
        cfg["rt_on"]
        & (low <= (zone1420_hi + rt_pad))
        & (high >= (zone1420_lo - rt_pad))
    )
    touch3350 = (
        cfg["rt_on"]
        & (low <= (zone3350_hi + rt_pad))
        & (high >= (zone3350_lo - rt_pad))
    )
    touch100200 = (
        cfg["rt_on"]
        & (low <= (zone100200_hi + rt_pad))
        & (high >= (zone100200_lo - rt_pad))
    )

    if cfg["rt_require_close"]:
        bull_retest1420_raw = touch1420 & (local_ema_dir == 1) & (close >= zone1420_lo)
        bear_retest1420_raw = touch1420 & (local_ema_dir == -1) & (close <= zone1420_hi)
        bull_retest3350_raw = touch3350 & (local_ema_dir == 1) & (close >= zone3350_lo)
        bear_retest3350_raw = touch3350 & (local_ema_dir == -1) & (close <= zone3350_hi)
        bull_retest100200_raw = touch100200 & (local_ema_dir == 1) & (close >= zone100200_lo)
        bear_retest100200_raw = touch100200 & (local_ema_dir == -1) & (close <= zone100200_hi)
    else:
        bull_retest1420_raw = touch1420 & (local_ema_dir == 1)
        bear_retest1420_raw = touch1420 & (local_ema_dir == -1)
        bull_retest3350_raw = touch3350 & (local_ema_dir == 1)
        bear_retest3350_raw = touch3350 & (local_ema_dir == -1)
        bull_retest100200_raw = touch100200 & (local_ema_dir == 1)
        bear_retest100200_raw = touch100200 & (local_ema_dir == -1)

    rt_ema1420_ok = np.select(
        [local_ema_dir == 1, local_ema_dir == -1],
        [
            _rolling_state_memory(bull_retest1420_raw, cfg["rt_lookback_bars"]),
            _rolling_state_memory(bear_retest1420_raw, cfg["rt_lookback_bars"]),
        ],
        default=False,
    )

    rt_ema3350_ok = np.select(
        [local_ema_dir == 1, local_ema_dir == -1],
        [
            _rolling_state_memory(bull_retest3350_raw, cfg["rt_lookback_bars"]),
            _rolling_state_memory(bear_retest3350_raw, cfg["rt_lookback_bars"]),
        ],
        default=False,
    )

    rt_ema100200_ok = np.select(
        [local_ema_dir == 1, local_ema_dir == -1],
        [
            _rolling_state_memory(bull_retest100200_raw, cfg["rt_lookback_bars"]),
            _rolling_state_memory(bear_retest100200_raw, cfg["rt_lookback_bars"]),
        ],
        default=False,
    )

    prior_high = _highest_prev(high, cfg["reclaim_lookback"])
    prior_low = _lowest_prev(low, cfg["reclaim_lookback"])

    bull_reclaim20 = (
        cfg["reclaim_on"]
        & pd.Series(rt_ema1420_ok, index=out.index).astype(bool)
        & (close > ema20)
        & (close > prior_high)
    )
    bear_reclaim20 = (
        cfg["reclaim_on"]
        & pd.Series(rt_ema1420_ok, index=out.index).astype(bool)
        & (close < ema20)
        & (close < prior_low)
    )

    bull_reclaim3350 = (
        cfg["reclaim_on"]
        & pd.Series(rt_ema3350_ok, index=out.index).astype(bool)
        & (close > band_hi)
        & (close > prior_high)
    )
    bear_reclaim3350 = (
        cfg["reclaim_on"]
        & pd.Series(rt_ema3350_ok, index=out.index).astype(bool)
        & (close < band_lo)
        & (close < prior_low)
    )

    bull_reclaim100200 = (
        cfg["reclaim_on"]
        & pd.Series(rt_ema100200_ok, index=out.index).astype(bool)
        & (close > zone100200_hi)
        & (close > prior_high)
    )
    bear_reclaim100200 = (
        cfg["reclaim_on"]
        & pd.Series(rt_ema100200_ok, index=out.index).astype(bool)
        & (close < zone100200_lo)
        & (close < prior_low)
    )

    mtf1 = _mtf_dir_from_resample(out, "1min")
    mtf5 = _mtf_dir_from_resample(out, "5min")
    mtf15 = _mtf_dir_from_resample(out, "15min")
    mtf60 = _mtf_dir_from_resample(out, "60min")
    mtf240 = _mtf_dir_from_resample(out, "240min")
    mtfD = _mtf_dir_from_resample(out, "1D")

    mtf_avg_dir = (mtf1 + mtf5 + mtf15 + mtf60 + mtf240 + mtfD) / 6.0

    local_dir_score = np.select(
        [local_ema_dir == 1, local_ema_dir == -1],
        [1.0, -1.0],
        default=0.0,
    )

    final_dir_score = (
        local_dir_score * cfg["mtf_weight_local"] + mtf_avg_dir * cfg["mtf_weight_avg"]
        if cfg["use_mtf_confirm"]
        else pd.Series(local_dir_score, index=out.index, dtype=float)
    )

    final_ema_zone_dir = np.select(
        [
            final_dir_score >= cfg["mtf_bull_thresh"],
            final_dir_score <= cfg["mtf_bear_thresh"],
        ],
        [1, -1],
        default=0,
    )

    final_zone_dir = np.where(
        ema_compression & (np.abs(final_dir_score) < 0.50),
        0,
        final_ema_zone_dir,
    )

    sc_ema_rt_any = (
        pd.Series(rt_ema1420_ok, index=out.index).astype(bool)
        | pd.Series(rt_ema3350_ok, index=out.index).astype(bool)
        | pd.Series(rt_ema100200_ok, index=out.index).astype(bool)
    )

    sc_ema_rt_family = np.select(
        [
            pd.Series(rt_ema1420_ok, index=out.index).astype(bool),
            pd.Series(rt_ema3350_ok, index=out.index).astype(bool),
            pd.Series(rt_ema100200_ok, index=out.index).astype(bool),
        ],
        [1, 2, 3],
        default=0,
    )

    ema_rt_dir = np.select(
        [local_ema_dir > 0, local_ema_dir < 0],
        [1, -1],
        default=0,
    )

    sc_ema_reclaim_20 = np.select([bull_reclaim20, bear_reclaim20], [1, -1], default=0)
    sc_ema_reclaim_3350 = np.select([bull_reclaim3350, bear_reclaim3350], [1, -1], default=0)
    sc_ema_reclaim_100200 = np.select([bull_reclaim100200, bear_reclaim100200], [1, -1], default=0)

    out["sc_ema_local_dir"] = local_ema_dir
    out["sc_ema_final_dir"] = final_zone_dir
    out["sc_ema_final_dir_score"] = final_dir_score
    out["sc_ema_behavior_type"] = ema_behavior_type
    out["sc_ema_trend_quality"] = ema_trend_quality
    out["sc_ema_compression"] = np.where(ema_compression, 1, 0)
    out["sc_ema_fast_compression"] = ema_fast_compression
    out["sc_ema_slow_compression"] = ema_slow_compression
    out["sc_ema_slow_expansion"] = ema_slow_expansion
    out["sc_ema_rt_any"] = sc_ema_rt_any
    out["sc_ema_rt_family"] = sc_ema_rt_family
    out["ema_rt_dir"] = ema_rt_dir
    out["sc_ema_reclaim_20"] = sc_ema_reclaim_20
    out["sc_ema_reclaim_3350"] = sc_ema_reclaim_3350
    out["sc_ema_reclaim_100200"] = sc_ema_reclaim_100200
    out["sc_ema_mtf_avg_dir"] = mtf_avg_dir

    return out