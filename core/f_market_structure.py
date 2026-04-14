from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MarketStructureConfig:
    # Core
    atr_len: int = 14
    base_len: int = 20
    range_len: int = 24
    compress_len: int = 16
    impulse_len: int = 8
    vol_len: int = 16

    # Displacement
    disp_body_mult: float = 0.75
    disp_range_mult: float = 0.85
    disp_close_pos_bull: float = 0.58
    disp_close_pos_bear: float = 0.42
    disp_vol_mult: float = 1.00
    disp_follow_bars: int = 2

    # Contraction / Expansion
    contr_atr_frac: float = 0.98
    expand_atr_frac: float = 1.02
    contr_range_frac: float = 0.98
    expand_range_frac: float = 1.02

    # Accumulation / Distribution
    ad_lookback: int = 20
    ad_band_atr: float = 3.2
    ad_vol_bias_mult: float = 1.00
    ad_delta_bias_mult: float = 1.00  # kept for Pine parity

    # Trend support
    trend_on: bool = True
    ema_fast_len: int = 20
    ema_slow_len: int = 50
    ema_slope_len: int = 5

    # MTF
    mtf_on: bool = True
    mtf_weight_1: float = 1.0
    mtf_weight_2: float = 1.0
    mtf_weight_3: float = 1.0
    mtf_weight_4: float = 1.0
    mtf_weight_5: float = 1.0
    mtf_bull_thresh: float = 0.20
    mtf_bear_thresh: float = -0.20


DEFAULT_MARKET_STRUCTURE_CONFIG: Dict[str, Any] = asdict(MarketStructureConfig())


# =============================================================================
# HELPERS
# =============================================================================


def _series_float(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype(float)


def clamp(series_or_value: Any, lo: float, hi: float):
    if isinstance(series_or_value, pd.Series):
        return series_or_value.clip(lower=lo, upper=hi)
    return max(lo, min(hi, float(series_or_value)))


def bool_score(cond: pd.Series) -> pd.Series:
    return cond.fillna(False).astype(float)


def sign(series: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=series.index)
    out = out.mask(series > 0, 1.0)
    out = out.mask(series < 0, -1.0)
    return out


def sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.rolling(length, min_periods=1).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / max(1, int(length)), adjust=False, min_periods=1).mean()


def close_pos(high: pd.Series, low: pd.Series, close: pd.Series, mintick: float = 1e-9) -> pd.Series:
    rng = (high - low).clip(lower=mintick)
    return (close - low) / rng


def highest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(max(1, int(length)), min_periods=1).max()


def lowest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(max(1, int(length)), min_periods=1).min()


def state_text(ms_state: pd.Series, contraction: pd.Series, expansion: pd.Series) -> pd.Series:
    out = pd.Series("TRANSITION", index=ms_state.index, dtype=object)
    out = out.mask(expansion.fillna(False), "EXPANSION")
    out = out.mask(contraction.fillna(False), "CONTRACTION")
    out = out.mask(ms_state == -1, "DISTRIBUTION")
    out = out.mask(ms_state == -2, "BEAR DISPLACEMENT")
    out = out.mask(ms_state == 1, "ACCUMULATION")
    out = out.mask(ms_state == 2, "BULL DISPLACEMENT")
    return out


def dir_text(ms_dir: pd.Series) -> pd.Series:
    out = pd.Series("NEUTRAL", index=ms_dir.index, dtype=object)
    out = out.mask(ms_dir > 0, "BULL")
    out = out.mask(ms_dir < 0, "BEAR")
    return out


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf_map = {
        "1": "1min",
        "3": "3min",
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "45": "45min",
        "60": "60min",
        "120": "120min",
        "180": "180min",
        "240": "240min",
        "D": "1D",
        "W": "1W",
    }
    rule = tf_map.get(str(timeframe), str(timeframe))

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("MTF mode requires a DatetimeIndex on the input DataFrame.")

    ohlcv = pd.DataFrame(index=df.index)
    src = df[["open", "high", "low", "close"]].copy()
    if "volume" in df.columns:
        src["volume"] = _series_float(df["volume"]).fillna(0.0)
    else:
        src["volume"] = 1.0

    agg = src.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    return agg


def _mtf_state_from_ohlc(
    ohlc: pd.DataFrame,
    fast_len: int,
    slow_len: int,
    slope_len: int,
) -> pd.Series:
    c = _series_float(ohlc["close"])
    ef = ema(c, fast_len)
    es = ema(c, slow_len)
    ef_slope = ef - ef.shift(max(1, int(slope_len))).fillna(ef)
    es_slope = es - es.shift(max(1, int(slope_len))).fillna(es)

    bull = (ef > es) & (ef_slope > 0) & (es_slope >= 0)
    bear = (ef < es) & (ef_slope < 0) & (es_slope <= 0)

    out = pd.Series(0.0, index=ohlc.index)
    out = out.mask(bull, 1.0)
    out = out.mask(bear, -1.0)
    return out


def _align_htf_state_to_ltf(htf_state: pd.Series, ltf_index: pd.Index) -> pd.Series:
    aligned = htf_state.reindex(ltf_index, method="ffill")
    return aligned.fillna(0.0)


# =============================================================================
# ENGINE
# =============================================================================


def run_market_structure_engine(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    mtf_frames: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    SmartChart Backend — f_market_structure.py

    Clean rebuild from Pine authority.
    Build order:
    inputs → helpers → core structure logic → displacement → compression/expansion
    → regime classification → state engine → MTF layer → export fields → test block.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: open, high, low, close.
        Volume is optional; if missing, it is replaced with 1.0 for Pine-style fallback.
    config : Optional[Dict[str, Any]]
        Partial config override.
    mtf_frames : Optional[Dict[str, str]]
        Real MTF timeframes. Expected keys: tf1, tf2, tf3, tf4, tf5.
        Example for 1m base chart: {"tf1": "5", "tf2": "15", "tf3": "60", "tf4": "240", "tf5": "D"}
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cfg = MarketStructureConfig(**{**DEFAULT_MARKET_STRUCTURE_CONFIG, **(config or {})})
    out = df.copy()

    # -------------------------------------------------------------------------
    # INPUT SERIES
    # -------------------------------------------------------------------------
    o = _series_float(out["open"])
    h = _series_float(out["high"])
    l = _series_float(out["low"])
    c = _series_float(out["close"])
    v = _series_float(out["volume"]) if "volume" in out.columns else pd.Series(1.0, index=out.index)
    v = v.fillna(1.0)

    mintick = 1e-9

    # -------------------------------------------------------------------------
    # CORE CALCULATIONS
    # -------------------------------------------------------------------------
    atr_val = atr(h, l, c, cfg.atr_len)
    atr_ma = sma(atr_val, cfg.range_len)

    bar_range = (h - l).clip(lower=0.0)
    body_size = (c - o).abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    close_pos_val = close_pos(h, l, c, mintick=mintick)

    range_ma = sma(bar_range, cfg.range_len)
    vol_src = v.copy()
    vol_ma = sma(vol_src, cfg.vol_len)

    rel_atr = np.where(atr_ma > 0, atr_val / atr_ma, 1.0)
    rel_range = np.where(range_ma > 0, bar_range / range_ma, 1.0)
    rel_vol = np.where(vol_ma > 0, vol_src / vol_ma, 1.0)

    rel_atr = pd.Series(rel_atr, index=out.index, dtype=float)
    rel_range = pd.Series(rel_range, index=out.index, dtype=float)
    rel_vol = pd.Series(rel_vol, index=out.index, dtype=float)

    rng_safe = (h - l).clip(lower=mintick)
    delta_raw = vol_src * (c - o) / rng_safe
    delta_abs = delta_raw.abs()
    delta_base = sma(delta_abs, cfg.vol_len).clip(lower=1.0)
    delta_norm = clamp(delta_raw / delta_base, -1.0, 1.0)

    price_change = c - c.shift(cfg.impulse_len).fillna(c)
    impulse_atr = pd.Series(np.where(atr_val > 0, price_change / atr_val.clip(lower=mintick), 0.0), index=out.index)
    impulse_dir = sign(impulse_atr)

    # -------------------------------------------------------------------------
    # TREND SUPPORT
    # -------------------------------------------------------------------------
    ema_fast = ema(c, cfg.ema_fast_len)
    ema_slow = ema(c, cfg.ema_slow_len)

    ema_fast_slope = ema_fast - ema_fast.shift(cfg.ema_slope_len).fillna(ema_fast)
    ema_slow_slope = ema_slow - ema_slow.shift(cfg.ema_slope_len).fillna(ema_slow)

    trend_bull = (ema_fast > ema_slow) & (ema_fast_slope > 0) & (ema_slow_slope >= 0) if cfg.trend_on else pd.Series(False, index=out.index)
    trend_bear = (ema_fast < ema_slow) & (ema_fast_slope < 0) & (ema_slow_slope <= 0) if cfg.trend_on else pd.Series(False, index=out.index)
    trend_neutral = ~trend_bull & ~trend_bear

    trend_support_bull = trend_bull if cfg.trend_on else pd.Series(True, index=out.index)
    trend_support_bear = trend_bear if cfg.trend_on else pd.Series(True, index=out.index)

    # -------------------------------------------------------------------------
    # DISPLACEMENT ENGINE
    # -------------------------------------------------------------------------
    bull_disp_raw = (
        (c > o)
        & (body_size > atr_val * cfg.disp_body_mult)
        & (bar_range > atr_val * cfg.disp_range_mult)
        & (close_pos_val >= cfg.disp_close_pos_bull)
        & (rel_vol >= cfg.disp_vol_mult)
    )
    bear_disp_raw = (
        (c < o)
        & (body_size > atr_val * cfg.disp_body_mult)
        & (bar_range > atr_val * cfg.disp_range_mult)
        & (close_pos_val <= cfg.disp_close_pos_bear)
        & (rel_vol >= cfg.disp_vol_mult)
    )

    bull_follow = highest(h, cfg.disp_follow_bars) > h.shift(1).fillna(h)
    bear_follow = lowest(l, cfg.disp_follow_bars) < l.shift(1).fillna(l)

    bull_displacement = bull_disp_raw & bull_follow
    bear_displacement = bear_disp_raw & bear_follow

    disp_strength_raw = pd.Series(np.where(atr_val > 0, body_size / atr_val.clip(lower=mintick), 0.0), index=out.index)

    bull_disp_strength = pd.Series(0.0, index=out.index)
    bull_disp_strength = bull_disp_strength.mask(
        bull_displacement,
        clamp(
            (disp_strength_raw / max(cfg.disp_body_mult, 0.1)) * 0.6
            + clamp(rel_vol / max(cfg.disp_vol_mult, 1.0), 0.0, 2.0) * 0.4,
            0.0,
            1.0,
        ),
    )

    bear_disp_strength = pd.Series(0.0, index=out.index)
    bear_disp_strength = bear_disp_strength.mask(
        bear_displacement,
        clamp(
            (disp_strength_raw / max(cfg.disp_body_mult, 0.1)) * 0.6
            + clamp(rel_vol / max(cfg.disp_vol_mult, 1.0), 0.0, 2.0) * 0.4,
            0.0,
            1.0,
        ),
    )

    # -------------------------------------------------------------------------
    # COMPRESSION / EXPANSION
    # -------------------------------------------------------------------------
    range_compression = highest(h, cfg.compress_len) - lowest(l, cfg.compress_len)

    contraction = (
        (rel_atr < cfg.contr_atr_frac)
        & (rel_range < cfg.contr_range_frac)
        & (range_compression < atr_val * cfg.ad_band_atr * 1.2)
    )
    expansion = (
        (rel_atr > cfg.expand_atr_frac)
        & (rel_range > cfg.expand_range_frac)
        & (rel_vol > 1.0)
    )

    # -------------------------------------------------------------------------
    # ACCUMULATION / DISTRIBUTION
    # -------------------------------------------------------------------------
    ad_high = highest(h, cfg.ad_lookback)
    ad_low = lowest(l, cfg.ad_lookback)
    ad_range = ad_high - ad_low
    ad_mid = (ad_high + ad_low) * 0.5

    tight_range = ad_range <= (atr_val * cfg.ad_band_atr)
    base_forming = tight_range & contraction

    ad_len = max(3, int(np.floor(cfg.ad_lookback / 3.0)))
    delta_avg = sma(delta_raw, ad_len)
    vol_avg = sma(vol_src, ad_len)

    bull_bias_inside_base = (
        (c >= ad_mid)
        & (delta_avg > 0)
        & (vol_avg >= (vol_ma / max(cfg.ad_vol_bias_mult, 1e-7)))
    )
    bear_bias_inside_base = (
        (c <= ad_mid)
        & (delta_avg < 0)
        & (vol_avg >= (vol_ma / max(cfg.ad_vol_bias_mult, 1e-7)))
    )

    accumulation = base_forming & bull_bias_inside_base & ~bear_bias_inside_base
    distribution = base_forming & bear_bias_inside_base & ~bull_bias_inside_base

    # -------------------------------------------------------------------------
    # TRANSITION
    # -------------------------------------------------------------------------
    transition = (
        ~accumulation
        & ~distribution
        & ~bull_displacement
        & ~bear_displacement
        & ~expansion
        & ~contraction
    )

    # -------------------------------------------------------------------------
    # MTF LAYER
    # -------------------------------------------------------------------------
    default_frames = {"tf1": "5", "tf2": "15", "tf3": "60", "tf4": "240", "tf5": "D"}
    frames = {**default_frames, **(mtf_frames or {})}

    if cfg.mtf_on:
        s1 = _align_htf_state_to_ltf(
            _mtf_state_from_ohlc(_resample_ohlcv(out, frames["tf1"]), cfg.ema_fast_len, cfg.ema_slow_len, cfg.ema_slope_len),
            out.index,
        )
        s2 = _align_htf_state_to_ltf(
            _mtf_state_from_ohlc(_resample_ohlcv(out, frames["tf2"]), cfg.ema_fast_len, cfg.ema_slow_len, cfg.ema_slope_len),
            out.index,
        )
        s3 = _align_htf_state_to_ltf(
            _mtf_state_from_ohlc(_resample_ohlcv(out, frames["tf3"]), cfg.ema_fast_len, cfg.ema_slow_len, cfg.ema_slope_len),
            out.index,
        )
        s4 = _align_htf_state_to_ltf(
            _mtf_state_from_ohlc(_resample_ohlcv(out, frames["tf4"]), cfg.ema_fast_len, cfg.ema_slow_len, cfg.ema_slope_len),
            out.index,
        )
        s5 = _align_htf_state_to_ltf(
            _mtf_state_from_ohlc(_resample_ohlcv(out, frames["tf5"]), cfg.ema_fast_len, cfg.ema_slow_len, cfg.ema_slope_len),
            out.index,
        )
    else:
        s1 = pd.Series(0.0, index=out.index)
        s2 = pd.Series(0.0, index=out.index)
        s3 = pd.Series(0.0, index=out.index)
        s4 = pd.Series(0.0, index=out.index)
        s5 = pd.Series(0.0, index=out.index)

    mtf_wsum = max(
        cfg.mtf_weight_1 + cfg.mtf_weight_2 + cfg.mtf_weight_3 + cfg.mtf_weight_4 + cfg.mtf_weight_5,
        1e-4,
    )
    mtf_raw = (
        s1 * cfg.mtf_weight_1
        + s2 * cfg.mtf_weight_2
        + s3 * cfg.mtf_weight_3
        + s4 * cfg.mtf_weight_4
        + s5 * cfg.mtf_weight_5
    )
    mtf_avg_dir = (mtf_raw / mtf_wsum) if cfg.mtf_on else pd.Series(0.0, index=out.index)

    mtf_support_bull = mtf_avg_dir > cfg.mtf_bull_thresh
    mtf_support_bear = mtf_avg_dir < cfg.mtf_bear_thresh
    mtf_agreement_score = mtf_avg_dir.abs()

    # -------------------------------------------------------------------------
    # FINAL STATE LOGIC / REGIME CLASSIFICATION
    # -------------------------------------------------------------------------
    bull_part_1 = bool_score(accumulation) * 0.18
    bull_part_2 = bool_score(bull_displacement) * 0.24
    bull_part_3 = bool_score(expansion & (c > ad_mid)) * 0.12
    bull_part_4 = bool_score(trend_support_bull) * 0.12
    bull_part_5 = bool_score(mtf_support_bull) * 0.12
    bull_part_6 = bool_score(delta_norm > 0) * 0.10
    bull_part_7 = bool_score(impulse_dir > 0) * 0.12

    bear_part_1 = bool_score(distribution) * 0.18
    bear_part_2 = bool_score(bear_displacement) * 0.24
    bear_part_3 = bool_score(expansion & (c < ad_mid)) * 0.12
    bear_part_4 = bool_score(trend_support_bear) * 0.12
    bear_part_5 = bool_score(mtf_support_bear) * 0.12
    bear_part_6 = bool_score(delta_norm < 0) * 0.10
    bear_part_7 = bool_score(impulse_dir < 0) * 0.12

    bull_context_score = bull_part_1 + bull_part_2 + bull_part_3 + bull_part_4 + bull_part_5 + bull_part_6 + bull_part_7
    bear_context_score = bear_part_1 + bear_part_2 + bear_part_3 + bear_part_4 + bear_part_5 + bear_part_6 + bear_part_7

    bull_score = clamp(bull_context_score, 0.0, 1.0)
    bear_score = clamp(bear_context_score, 0.0, 1.0)

    ms_dir = pd.Series(0.0, index=out.index)
    ms_dir = ms_dir.mask((bull_score > bear_score) & (bull_score >= 0.35), 1.0)
    ms_dir = ms_dir.mask((bear_score > bull_score) & (bear_score >= 0.35), -1.0)

    ms_state = pd.Series(0.0, index=out.index)
    ms_state = ms_state.mask(distribution, -1.0)
    ms_state = ms_state.mask(bear_displacement, -2.0)
    ms_state = ms_state.mask(accumulation, 1.0)
    ms_state = ms_state.mask(bull_displacement, 2.0)

    ms_bull = ms_dir == 1.0
    ms_bear = ms_dir == -1.0

    max_side_score = pd.concat([bull_score, bear_score], axis=1).max(axis=1)
    disp_strength = pd.concat([bull_disp_strength, bear_disp_strength], axis=1).max(axis=1)
    phase_strength = bool_score(accumulation | distribution) * 0.25
    volatility_strength = bool_score(expansion) * 0.20 + bool_score(contraction) * 0.10

    ms_strength_score = clamp(
        max_side_score * 0.45
        + disp_strength * 0.25
        + mtf_agreement_score * 0.15
        + volatility_strength * 0.10
        + phase_strength * 0.05,
        0.0,
        1.0,
    )

    quality_trend = ((ms_bull & trend_support_bull) | (ms_bear & trend_support_bear) | (ms_state == 0.0))
    quality_mtf = ((ms_bull & mtf_support_bull) | (ms_bear & mtf_support_bear) | (ms_state == 0.0))
    quality_delta = ((ms_bull & (delta_norm > 0)) | (ms_bear & (delta_norm < 0)) | (ms_state == 0.0))
    quality_expansion = pd.Series(True, index=out.index)
    quality_expansion = quality_expansion.mask(
        bull_displacement | bear_displacement,
        expansion | (rel_vol > 1.0),
    )

    ms_quality_score = clamp(
        bool_score(quality_trend) * 0.25
        + bool_score(quality_mtf) * 0.20
        + bool_score(quality_delta) * 0.20
        + bool_score(quality_expansion) * 0.20
        + bool_score(~transition) * 0.15,
        0.0,
        1.0,
    )

    sc_state_text = state_text(ms_state, contraction, expansion)
    sc_dir_text = dir_text(ms_dir)

    # -------------------------------------------------------------------------
    # EXPORT FIELDS
    # -------------------------------------------------------------------------
    out["atr_val"] = atr_val
    out["atr_ma"] = atr_ma
    out["bar_range"] = bar_range
    out["body_size"] = body_size
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick
    out["close_pos"] = close_pos_val
    out["range_ma"] = range_ma
    out["vol_ma"] = vol_ma
    out["rel_atr"] = rel_atr
    out["rel_range"] = rel_range
    out["rel_vol"] = rel_vol
    out["delta_raw"] = delta_raw
    out["delta_norm"] = delta_norm
    out["price_change"] = price_change
    out["impulse_atr"] = impulse_atr
    out["impulse_dir"] = impulse_dir

    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["ema_fast_slope"] = ema_fast_slope
    out["ema_slow_slope"] = ema_slow_slope
    out["trend_bull"] = trend_bull.astype(int)
    out["trend_bear"] = trend_bear.astype(int)
    out["trend_neutral"] = trend_neutral.astype(int)
    out["trend_support_bull"] = trend_support_bull.astype(int)
    out["trend_support_bear"] = trend_support_bear.astype(int)

    out["bull_disp_raw"] = bull_disp_raw.astype(int)
    out["bear_disp_raw"] = bear_disp_raw.astype(int)
    out["bull_follow"] = bull_follow.astype(int)
    out["bear_follow"] = bear_follow.astype(int)
    out["bull_displacement"] = bull_displacement.astype(int)
    out["bear_displacement"] = bear_displacement.astype(int)
    out["bull_disp_strength"] = bull_disp_strength
    out["bear_disp_strength"] = bear_disp_strength

    out["range_compression"] = range_compression
    out["contraction"] = contraction.astype(int)
    out["expansion"] = expansion.astype(int)

    out["ad_high"] = ad_high
    out["ad_low"] = ad_low
    out["ad_range"] = ad_range
    out["ad_mid"] = ad_mid
    out["tight_range"] = tight_range.astype(int)
    out["base_forming"] = base_forming.astype(int)
    out["delta_avg"] = delta_avg
    out["vol_avg"] = vol_avg
    out["bull_bias_inside_base"] = bull_bias_inside_base.astype(int)
    out["bear_bias_inside_base"] = bear_bias_inside_base.astype(int)
    out["accumulation"] = accumulation.astype(int)
    out["distribution"] = distribution.astype(int)
    out["transition"] = transition.astype(int)

    out["mtf_s1"] = s1
    out["mtf_s2"] = s2
    out["mtf_s3"] = s3
    out["mtf_s4"] = s4
    out["mtf_s5"] = s5
    out["mtf_avg_dir"] = mtf_avg_dir
    out["mtf_support_bull"] = mtf_support_bull.astype(int)
    out["mtf_support_bear"] = mtf_support_bear.astype(int)
    out["mtf_agreement_score"] = mtf_agreement_score

    out["bull_score"] = bull_score
    out["bear_score"] = bear_score
    out["ms_dir"] = ms_dir
    out["ms_state"] = ms_state
    out["ms_bull"] = ms_bull.astype(int)
    out["ms_bear"] = ms_bear.astype(int)
    out["ms_strength_score"] = ms_strength_score
    out["ms_quality_score"] = ms_quality_score
    out["state_text"] = sc_state_text
    out["dir_text"] = sc_dir_text

    # SmartChart export contract
    out["sc_ms_state"] = ms_state
    out["sc_ms_state_text"] = sc_state_text
    out["sc_ms_dir"] = ms_dir
    out["sc_ms_dir_text"] = sc_dir_text
    out["sc_ms_bull_score"] = bull_score
    out["sc_ms_bear_score"] = bear_score
    out["sc_ms_strength"] = ms_strength_score
    out["sc_ms_quality"] = ms_quality_score

    out["sc_ms_accumulation"] = accumulation.astype(float)
    out["sc_ms_distribution"] = distribution.astype(float)
    out["sc_ms_bull_displacement"] = bull_displacement.astype(float)
    out["sc_ms_bear_displacement"] = bear_displacement.astype(float)
    out["sc_ms_expansion"] = expansion.astype(float)
    out["sc_ms_contraction"] = contraction.astype(float)
    out["sc_ms_transition"] = transition.astype(float)

    out["sc_ms_delta"] = delta_norm
    out["sc_ms_impulse_dir"] = impulse_dir

    out["sc_ms_mtf_avg"] = mtf_avg_dir
    out["sc_ms_mtf_support_bull"] = mtf_support_bull.astype(float)
    out["sc_ms_mtf_support_bear"] = mtf_support_bear.astype(float)
    out["sc_ms_mtf_agreement"] = mtf_agreement_score

    return out


# =============================================================================
# TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    idx = pd.date_range("2026-01-01 08:00:00", periods=600, freq="1min")
    rng = np.random.default_rng(7)

    base = 4750 + np.cumsum(rng.normal(0, 0.9, len(idx)))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0] - 0.3)
    spread = np.abs(rng.normal(0.8, 0.3, len(idx))) + 0.05
    high = pd.concat([open_, close], axis=1).max(axis=1) + spread
    low = pd.concat([open_, close], axis=1).min(axis=1) - spread
    volume = pd.Series(rng.integers(100, 1000, len(idx)), index=idx, dtype=float)

    test_df = pd.DataFrame(
        {
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume.values,
        },
        index=idx,
    )

    result = run_market_structure_engine(
        test_df,
        config=None,
        mtf_frames={"tf1": "5", "tf2": "15", "tf3": "60", "tf4": "240", "tf5": "D"},
    )

    cols = [
        "sc_ms_state",
        "sc_ms_state_text",
        "sc_ms_dir",
        "sc_ms_dir_text",
        "sc_ms_bull_score",
        "sc_ms_bear_score",
        "sc_ms_strength",
        "sc_ms_quality",
        "sc_ms_accumulation",
        "sc_ms_distribution",
        "sc_ms_bull_displacement",
        "sc_ms_bear_displacement",
        "sc_ms_expansion",
        "sc_ms_contraction",
        "sc_ms_transition",
        "sc_ms_delta",
        "sc_ms_impulse_dir",
        "sc_ms_mtf_avg",
    ]
    print(result[cols].tail(10).to_string())
