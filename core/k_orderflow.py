from __future__ import annotations

from dataclasses import dataclass
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
    or_mode: str = "Auto"  # Auto | Manual
    or_auto_k: float = 3.0
    or_manual_minutes: int = 30

    # Trend
    trend_mode: str = "EMA+ADX"  # Off | EMA | EMA+ADX
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
    imb_box_count: int = 10
    imb_shrink_on_touch: bool = True

    # Structure / pivot zones
    zone_on: bool = True
    zone_depth_atr: float = 0.60
    zone_extend_bars: int = 120
    pivot_a_left: int = 10
    pivot_a_right: int = 10
    pivot_b_left: int = 5
    pivot_b_right: int = 5
    only_trend_zones: bool = False

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
    session_col: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

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


def _clamp(v: pd.Series | float, lo: float, hi: float):
    if isinstance(v, pd.Series):
        return v.clip(lower=lo, upper=hi)
    return max(lo, min(hi, v))


def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series).astype(int)


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
    state = pd.Series(
        np.where((c > ef) & (ef > es), 1.0, np.where((c < ef) & (ef < es), -1.0, 0.0)),
        index=c.index,
    )
    state = state.reindex(close.index).ffill().fillna(0.0)
    return state


def _build_session_reset_series(index: pd.Index, tf_minutes: float, session_col_series: Optional[pd.Series] = None) -> pd.Series:
    if session_col_series is not None:
        return session_col_series.ne(session_col_series.shift(1)).fillna(True)

    if isinstance(index, pd.DatetimeIndex):
        normalized = pd.Series(index.normalize(), index=index)
        return normalized.ne(normalized.shift(1)).fillna(True)

    bars_per_day = max(1, int(round(1440 / max(tf_minutes, 1.0))))
    new_sess = pd.Series(False, index=index)
    new_sess.iloc[::bars_per_day] = True
    new_sess.iloc[0] = True
    return new_sess


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

    session_series = out[cfg.session_col] if cfg.session_col and cfg.session_col in out.columns else None
    new_sess = _build_session_reset_series(out.index, cfg.tf_minutes, session_series)

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
    or_count = np.zeros(len(out), dtype=int)

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
        or_count[i] = count

    out["sc_of_or_high"] = or_high
    out["sc_of_or_low"] = or_low
    out["sc_of_or_count"] = or_count
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
        (_clamp(out["sc_of_rel_vol"] / max(cfg.rv_impulse_mult, 1.0), 0.0, 2.0) * 0.6)
        + (out["sc_of_delta_norm"].abs() * 0.4)
    ).clip(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Density
    # Pine uses zone memory arrays.
    # Backend keeps a simplified but state-aligned rolling persistence layer.
    # -------------------------------------------------------------------------
    if cfg.den_metric_mode == "AbsDelta":
        den_metric = out["sc_of_delta_sm"].abs()
    elif cfg.den_metric_mode == "Volume":
        den_metric = volume
    else:
        den_metric = out["sc_of_delta_sm"].abs() * 0.6 + volume * 0.4

    out["sc_of_den_metric"] = den_metric
    out["sc_of_den_thr"] = _rolling_percentile(den_metric, cfg.den_q_len, cfg.den_q_pct)

    density_hit = bool(cfg.den_on) & (out["sc_of_den_metric"] >= out["sc_of_den_thr"])
    out["sc_of_density_event"] = density_hit.astype(int)

    den_top_arr = np.full(len(out), np.nan)
    den_bot_arr = np.full(len(out), np.nan)
    den_mid_arr = np.full(len(out), np.nan)
    den_dir_arr = np.zeros(len(out), dtype=int)
    den_bull_active_arr = np.zeros(len(out), dtype=int)
    den_bear_active_arr = np.zeros(len(out), dtype=int)

    density_zones: list[dict[str, Any]] = []

    for i in range(len(out)):
        current_atr = _safe_float(out["sc_of_atr"].iloc[i], 0.0)
        current_close = _safe_float(close.iloc[i], 0.0)
        current_high = _safe_float(high.iloc[i], 0.0)
        current_low = _safe_float(low.iloc[i], 0.0)
        current_vol = _safe_float(volume.iloc[i], 0.0)
        current_vol_ma = _safe_float(out["sc_of_vol_ma"].iloc[i], 0.0)
        current_delta_sm = _safe_float(out["sc_of_delta_sm"].iloc[i], 0.0)

        if bool(density_hit.iloc[i]):
            den_mid_price = current_close
            merged = False
            merge_dist = current_atr * cfg.den_merge_atr
            for z in density_zones:
                if abs(den_mid_price - z["mid"]) <= merge_dist:
                    z["end_bar"] = i + cfg.den_extend_bars
                    merged = True
                    break

            if not merged:
                half = current_atr * cfg.den_depth_atr
                density_zones.append(
                    {
                        "top": den_mid_price + half,
                        "bot": den_mid_price - half,
                        "mid": den_mid_price,
                        "end_bar": i + cfg.den_extend_bars,
                        "dir": 1 if current_delta_sm >= 0 else -1,
                    }
                )
                den_top_arr[i] = den_mid_price + half
                den_bot_arr[i] = den_mid_price - half
                den_mid_arr[i] = den_mid_price
                den_dir_arr[i] = 1 if current_delta_sm >= 0 else -1

        keep_zones: list[dict[str, Any]] = []
        bull_active = False
        bear_active = False

        for z in density_zones:
            break_up = current_close > z["top"] and current_vol > current_vol_ma * cfg.den_break_vol_mult
            break_down = current_close < z["bot"] and current_vol > current_vol_ma * cfg.den_break_vol_mult
            eaten = (z["dir"] == 1 and break_down) or (z["dir"] == -1 and break_up)
            expired = i >= z["end_bar"]

            in_density = current_high >= z["bot"] and current_low <= z["top"]
            if in_density:
                if z["dir"] == 1:
                    bull_active = True
                elif z["dir"] == -1:
                    bear_active = True

            if not eaten and not expired:
                keep_zones.append(z)

        density_zones = keep_zones
        den_bull_active_arr[i] = 1 if bull_active else 0
        den_bear_active_arr[i] = 1 if bear_active else 0

    out["sc_of_density_mid"] = den_mid_arr
    out["sc_of_density_top"] = den_top_arr
    out["sc_of_density_bot"] = den_bot_arr
    out["sc_of_density_dir"] = den_dir_arr
    out["sc_of_density_bull_active"] = den_bull_active_arr
    out["sc_of_density_bear_active"] = den_bear_active_arr

    # -------------------------------------------------------------------------
    # Imbalance
    # Pine uses memory arrays + shrink/mitigation.
    # Backend reproduces that state behavior directly.
    # -------------------------------------------------------------------------
    out["sc_of_imb_atr"] = atr(high, low, close, cfg.imb_atr_len)
    body_pct = _body_pct(open_, high, low, close)

    out["sc_of_imbalance_event"] = 0
    out["sc_of_imbalance_dir"] = 0
    out["sc_of_imbalance_top"] = np.nan
    out["sc_of_imbalance_bot"] = np.nan
    out["sc_of_imbalance_bull_active"] = 0
    out["sc_of_imbalance_bear_active"] = 0

    imbalance_boxes: list[dict[str, Any]] = []

    for i in range(len(out)):
        if cfg.imb_on and i >= 2:
            price_diff = abs(_safe_float(high.iloc[i - 1]) - _safe_float(low.iloc[i - 1]))
            big_body = _safe_float(body_pct.iloc[i - 1]) >= cfg.imb_body_pct
            up_candle = _safe_float(open_.iloc[i - 1]) <= _safe_float(close.iloc[i - 1])

            gap_closed = (
                _safe_float(high.iloc[i - 2]) >= _safe_float(low.iloc[i])
                if up_candle
                else _safe_float(low.iloc[i - 2]) <= _safe_float(high.iloc[i])
            )

            is_zone = (
                price_diff > _safe_float(out["sc_of_imb_atr"].iloc[i]) * cfg.imb_atr_mult
                and big_body
                and not gap_closed
            )

            if is_zone:
                top_ = _safe_float(low.iloc[i]) if up_candle else _safe_float(low.iloc[i - 2])
                bot_ = _safe_float(high.iloc[i - 2]) if up_candle else _safe_float(high.iloc[i])

                zone = {
                    "top": max(top_, bot_),
                    "bot": min(top_, bot_),
                    "dir": 1 if up_candle else -1,
                }
                imbalance_boxes.append(zone)
                if len(imbalance_boxes) > max(1, cfg.imb_box_count):
                    imbalance_boxes.pop(0)

                out.iloc[i, out.columns.get_loc("sc_of_imbalance_event")] = 1
                out.iloc[i, out.columns.get_loc("sc_of_imbalance_dir")] = zone["dir"]
                out.iloc[i, out.columns.get_loc("sc_of_imbalance_top")] = zone["top"]
                out.iloc[i, out.columns.get_loc("sc_of_imbalance_bot")] = zone["bot"]

        bull_active = False
        bear_active = False
        keep_boxes: list[dict[str, Any]] = []

        current_open = _safe_float(open_.iloc[i])
        current_high = _safe_float(high.iloc[i])
        current_low = _safe_float(low.iloc[i])

        for z in imbalance_boxes:
            top_ = float(z["top"])
            bot_ = float(z["bot"])

            if cfg.imb_shrink_on_touch:
                if current_open >= top_ and current_low < top_ and current_low > bot_:
                    top_ = current_low
                if current_open <= bot_ and current_high > bot_ and current_high < top_:
                    bot_ = current_high

            fully_mitigated = (
                (current_open <= bot_ and current_high > bot_ and current_high >= top_)
                or (current_open >= top_ and current_low < top_ and current_low <= bot_)
            )

            in_imb = current_high >= bot_ and current_low <= top_
            if in_imb:
                if z["dir"] == 1:
                    bull_active = True
                elif z["dir"] == -1:
                    bear_active = True

            if not fully_mitigated:
                z["top"] = top_
                z["bot"] = bot_
                keep_boxes.append(z)

        imbalance_boxes = keep_boxes
        out.iloc[i, out.columns.get_loc("sc_of_imbalance_bull_active")] = 1 if bull_active else 0
        out.iloc[i, out.columns.get_loc("sc_of_imbalance_bear_active")] = 1 if bear_active else 0

    # -------------------------------------------------------------------------
    # Structure zones
    # Pine uses born/expire and trend-aware zone state.
    # Backend reproduces the active-state logic closely.
    # -------------------------------------------------------------------------
    ph_a = _pivot_high(high, cfg.pivot_a_left, cfg.pivot_a_right)
    pl_a = _pivot_low(low, cfg.pivot_a_left, cfg.pivot_a_right)
    ph_b = _pivot_high(high, cfg.pivot_b_left, cfg.pivot_b_right)
    pl_b = _pivot_low(low, cfg.pivot_b_left, cfg.pivot_b_right)

    out["sc_of_pivot_high_a"] = ph_a
    out["sc_of_pivot_low_a"] = pl_a
    out["sc_of_pivot_high_b"] = ph_b
    out["sc_of_pivot_low_b"] = pl_b

    zone_cols = [
        "sc_of_zone_a_sup_top",
        "sc_of_zone_a_sup_bot",
        "sc_of_zone_a_dem_top",
        "sc_of_zone_a_dem_bot",
        "sc_of_zone_b_sup_top",
        "sc_of_zone_b_sup_bot",
        "sc_of_zone_b_dem_top",
        "sc_of_zone_b_dem_bot",
        "sc_of_zone_a_sup_active",
        "sc_of_zone_a_dem_active",
        "sc_of_zone_b_sup_active",
        "sc_of_zone_b_dem_active",
    ]
    for col in zone_cols:
        out[col] = np.nan if "top" in col or "bot" in col else 0

    pair_a = {
        "sup": {"top": np.nan, "bot": np.nan, "born": None, "expire": None, "in_trend": False},
        "dem": {"top": np.nan, "bot": np.nan, "born": None, "expire": None, "in_trend": False},
    }
    pair_b = {
        "sup": {"top": np.nan, "bot": np.nan, "born": None, "expire": None, "in_trend": False},
        "dem": {"top": np.nan, "bot": np.nan, "born": None, "expire": None, "in_trend": False},
    }

    zone_bull_active_arr = np.zeros(len(out), dtype=int)
    zone_bear_active_arr = np.zeros(len(out), dtype=int)

    for i in range(len(out)):
        current_atr = _safe_float(out["sc_of_atr"].iloc[i], 0.0)

        if cfg.zone_on and pd.notna(ph_a.iloc[i]):
            ph = _safe_float(ph_a.iloc[i])
            pair_a["sup"] = {
                "top": ph,
                "bot": ph - cfg.zone_depth_atr * current_atr,
                "born": i - cfg.pivot_a_right,
                "expire": i + cfg.zone_extend_bars,
                "in_trend": bool(out["sc_of_trend_short_ok"].iloc[i] == 1),
            }

        if cfg.zone_on and pd.notna(pl_a.iloc[i]):
            pl = _safe_float(pl_a.iloc[i])
            pair_a["dem"] = {
                "top": pl + cfg.zone_depth_atr * current_atr,
                "bot": pl,
                "born": i - cfg.pivot_a_right,
                "expire": i + cfg.zone_extend_bars,
                "in_trend": bool(out["sc_of_trend_long_ok"].iloc[i] == 1),
            }

        if cfg.zone_on and pd.notna(ph_b.iloc[i]):
            ph = _safe_float(ph_b.iloc[i])
            pair_b["sup"] = {
                "top": ph,
                "bot": ph - cfg.zone_depth_atr * current_atr,
                "born": i - cfg.pivot_b_right,
                "expire": i + cfg.zone_extend_bars,
                "in_trend": bool(out["sc_of_trend_short_ok"].iloc[i] == 1),
            }

        if cfg.zone_on and pd.notna(pl_b.iloc[i]):
            pl = _safe_float(pl_b.iloc[i])
            pair_b["dem"] = {
                "top": pl + cfg.zone_depth_atr * current_atr,
                "bot": pl,
                "born": i - cfg.pivot_b_right,
                "expire": i + cfg.zone_extend_bars,
                "in_trend": bool(out["sc_of_trend_long_ok"].iloc[i] == 1),
            }

        def _zone_active(z: dict[str, Any]) -> bool:
            expire = z["expire"]
            if pd.isna(z["top"]) or pd.isna(z["bot"]) or expire is None:
                return False
            if i > expire:
                return False
            if cfg.only_trend_zones and not z["in_trend"]:
                return False
            return True

        a_sup_active = _zone_active(pair_a["sup"])
        a_dem_active = _zone_active(pair_a["dem"])
        b_sup_active = _zone_active(pair_b["sup"])
        b_dem_active = _zone_active(pair_b["dem"])

        out.iloc[i, out.columns.get_loc("sc_of_zone_a_sup_top")] = pair_a["sup"]["top"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_a_sup_bot")] = pair_a["sup"]["bot"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_a_dem_top")] = pair_a["dem"]["top"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_a_dem_bot")] = pair_a["dem"]["bot"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_b_sup_top")] = pair_b["sup"]["top"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_b_sup_bot")] = pair_b["sup"]["bot"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_b_dem_top")] = pair_b["dem"]["top"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_b_dem_bot")] = pair_b["dem"]["bot"]
        out.iloc[i, out.columns.get_loc("sc_of_zone_a_sup_active")] = 1 if a_sup_active else 0
        out.iloc[i, out.columns.get_loc("sc_of_zone_a_dem_active")] = 1 if a_dem_active else 0
        out.iloc[i, out.columns.get_loc("sc_of_zone_b_sup_active")] = 1 if b_sup_active else 0
        out.iloc[i, out.columns.get_loc("sc_of_zone_b_dem_active")] = 1 if b_dem_active else 0

        current_high = _safe_float(high.iloc[i])
        current_low = _safe_float(low.iloc[i])

        zone_bull_active = (
            (a_dem_active and current_high >= pair_a["dem"]["bot"] and current_low <= pair_a["dem"]["top"])
            or (b_dem_active and current_high >= pair_b["dem"]["bot"] and current_low <= pair_b["dem"]["top"])
        )
        zone_bear_active = (
            (a_sup_active and current_high >= pair_a["sup"]["bot"] and current_low <= pair_a["sup"]["top"])
            or (b_sup_active and current_high >= pair_b["sup"]["bot"] and current_low <= pair_b["sup"]["top"])
        )

        zone_bull_active_arr[i] = 1 if zone_bull_active else 0
        zone_bear_active_arr[i] = 1 if zone_bear_active else 0

    out["sc_of_zone_bull_active"] = (cfg.zone_on and True) * zone_bull_active_arr
    out["sc_of_zone_bear_active"] = (cfg.zone_on and True) * zone_bear_active_arr

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

        wsum = max(
            cfg.mtf_weight_1 + cfg.mtf_weight_2 + cfg.mtf_weight_3 + cfg.mtf_weight_4 + cfg.mtf_weight_5,
            1e-9,
        )
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
# PAYLOAD BUILDER
# =============================================================================

def build_orderflow_latest_payload(
    df: pd.DataFrame,
    config: Optional[OrderflowConfig] = None,
) -> Dict[str, Any]:
    result = run_orderflow_engine(df, config=config)

    if result.empty:
        raise ValueError("Orderflow payload build failed: empty result dataframe")

    last = result.iloc[-1]
    ts = result.index[-1]

    zone_summary = "bull_active" if _safe_int(last.get("sc_of_zone_bull_active")) == 1 else (
        "bear_active" if _safe_int(last.get("sc_of_zone_bear_active")) == 1 else "neutral"
    )
    density_summary = "bull_active" if _safe_int(last.get("sc_of_density_bull_active")) == 1 else (
        "bear_active" if _safe_int(last.get("sc_of_density_bear_active")) == 1 else "neutral"
    )
    imbalance_summary = "bull_active" if _safe_int(last.get("sc_of_imbalance_bull_active")) == 1 else (
        "bear_active" if _safe_int(last.get("sc_of_imbalance_bear_active")) == 1 else "neutral"
    )
    acceptance_summary = "bull_accept" if _safe_int(last.get("sc_of_accept_bull")) == 1 else (
        "bear_accept" if _safe_int(last.get("sc_of_accept_bear")) == 1 else "neutral"
    )

    payload: Dict[str, Any] = {
        "indicator": "orderflow",
        "debug_version": "orderflow_payload_v1",
        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),

        "state": {
            "dir": _safe_int(last.get("sc_orderflow_dir")),
            "text": str(last.get("sc_orderflow_text", "NEUTRAL")),
            "strength": round(_safe_float(last.get("sc_orderflow_strength")), 6),
            "quality": round(_safe_float(last.get("sc_orderflow_quality")), 6),
        },

        "scores": {
            "bull_score": round(_safe_float(last.get("sc_of_bull_score")), 6),
            "bear_score": round(_safe_float(last.get("sc_of_bear_score")), 6),
            "mtf_avg_dir": round(_safe_float(last.get("sc_of_mtf_avg_dir")), 6),
            "participation_score": round(_safe_float(last.get("sc_of_participation_score")), 6),
            "mtf_agreement_score": round(_safe_float(last.get("sc_of_mtf_agreement_score")), 6),
            "delta_norm": round(_safe_float(last.get("sc_of_delta_norm")), 6),
        },

        "flow": {
            "bull_delta_impulse": bool(_safe_int(last.get("sc_of_bull_delta_impulse")) == 1),
            "bear_delta_impulse": bool(_safe_int(last.get("sc_of_bear_delta_impulse")) == 1),
            "bull_participation": bool(_safe_int(last.get("sc_of_bull_participation")) == 1),
            "bear_participation": bool(_safe_int(last.get("sc_of_bear_participation")) == 1),
            "vol_expansion": bool(_safe_int(last.get("sc_of_vol_expansion")) == 1),
            "trend_long_ok": bool(_safe_int(last.get("sc_of_trend_long_ok")) == 1),
            "trend_short_ok": bool(_safe_int(last.get("sc_of_trend_short_ok")) == 1),
            "trend_dir": _safe_int(last.get("sc_of_trend_dir")),
        },

        "density": {
            "bull_active": bool(_safe_int(last.get("sc_of_density_bull_active")) == 1),
            "bear_active": bool(_safe_int(last.get("sc_of_density_bear_active")) == 1),
            "event": bool(_safe_int(last.get("sc_of_density_event")) == 1),
            "summary": density_summary,
            "top": _safe_float(last.get("sc_of_density_top"), np.nan),
            "bot": _safe_float(last.get("sc_of_density_bot"), np.nan),
            "mid": _safe_float(last.get("sc_of_density_mid"), np.nan),
        },

        "imbalance": {
            "bull_active": bool(_safe_int(last.get("sc_of_imbalance_bull_active")) == 1),
            "bear_active": bool(_safe_int(last.get("sc_of_imbalance_bear_active")) == 1),
            "event": bool(_safe_int(last.get("sc_of_imbalance_event")) == 1),
            "summary": imbalance_summary,
            "top": _safe_float(last.get("sc_of_imbalance_top"), np.nan),
            "bot": _safe_float(last.get("sc_of_imbalance_bot"), np.nan),
        },

        "zones": {
            "bull_active": bool(_safe_int(last.get("sc_of_zone_bull_active")) == 1),
            "bear_active": bool(_safe_int(last.get("sc_of_zone_bear_active")) == 1),
            "summary": zone_summary,

            "a_supply": {
                "top": _safe_float(last.get("sc_of_zone_a_sup_top"), np.nan),
                "bot": _safe_float(last.get("sc_of_zone_a_sup_bot"), np.nan),
                "active": bool(_safe_int(last.get("sc_of_zone_a_sup_active")) == 1),
            },
            "a_demand": {
                "top": _safe_float(last.get("sc_of_zone_a_dem_top"), np.nan),
                "bot": _safe_float(last.get("sc_of_zone_a_dem_bot"), np.nan),
                "active": bool(_safe_int(last.get("sc_of_zone_a_dem_active")) == 1),
            },
            "b_supply": {
                "top": _safe_float(last.get("sc_of_zone_b_sup_top"), np.nan),
                "bot": _safe_float(last.get("sc_of_zone_b_sup_bot"), np.nan),
                "active": bool(_safe_int(last.get("sc_of_zone_b_sup_active")) == 1),
            },
            "b_demand": {
                "top": _safe_float(last.get("sc_of_zone_b_dem_top"), np.nan),
                "bot": _safe_float(last.get("sc_of_zone_b_dem_bot"), np.nan),
                "active": bool(_safe_int(last.get("sc_of_zone_b_dem_active")) == 1),
            },
        },

        "profile": {
            "in_acceptance": bool(_safe_int(last.get("sc_of_in_acceptance")) == 1),
            "accept_bull": bool(_safe_int(last.get("sc_of_accept_bull")) == 1),
            "accept_bear": bool(_safe_int(last.get("sc_of_accept_bear")) == 1),
            "summary": acceptance_summary,
            "poc_price": _safe_float(last.get("sc_of_poc_price"), np.nan),
            "acc_top": _safe_float(last.get("sc_of_acc_top"), np.nan),
            "acc_bot": _safe_float(last.get("sc_of_acc_bot"), np.nan),
            "poc_bull_share": round(_safe_float(last.get("sc_of_poc_bull_share")), 6),
            "poc_bear_share": round(_safe_float(last.get("sc_of_poc_bear_share")), 6),
        },

        "levels": {
            "vwap": _safe_float(last.get("sc_of_vwap"), np.nan),
            "or_high": _safe_float(last.get("sc_of_or_high"), np.nan),
            "or_low": _safe_float(last.get("sc_of_or_low"), np.nan),
        },

        "mtf": {
            "s1": round(_safe_float(last.get("sc_of_mtf_s1")), 6),
            "s2": round(_safe_float(last.get("sc_of_mtf_s2")), 6),
            "s3": round(_safe_float(last.get("sc_of_mtf_s3")), 6),
            "s4": round(_safe_float(last.get("sc_of_mtf_s4")), 6),
            "s5": round(_safe_float(last.get("sc_of_mtf_s5")), 6),
            "avg_dir": round(_safe_float(last.get("sc_of_mtf_avg_dir")), 6),
            "support_bull": bool(_safe_int(last.get("sc_of_mtf_support_bull")) == 1),
            "support_bear": bool(_safe_int(last.get("sc_of_mtf_support_bear")) == 1),
        },

        "raw_contract": {
            "sc_orderflow_dir": _safe_int(last.get("sc_orderflow_dir")),
            "sc_orderflow_text": str(last.get("sc_orderflow_text", "NEUTRAL")),
            "sc_orderflow_strength": round(_safe_float(last.get("sc_orderflow_strength")), 6),
            "sc_orderflow_quality": round(_safe_float(last.get("sc_orderflow_quality")), 6),

            "sc_of_bull_score": round(_safe_float(last.get("sc_of_bull_score")), 6),
            "sc_of_bear_score": round(_safe_float(last.get("sc_of_bear_score")), 6),
            "sc_of_mtf_avg_dir": round(_safe_float(last.get("sc_of_mtf_avg_dir")), 6),

            "sc_of_density_bull_active": _safe_int(last.get("sc_of_density_bull_active")),
            "sc_of_density_bear_active": _safe_int(last.get("sc_of_density_bear_active")),

            "sc_of_imbalance_bull_active": _safe_int(last.get("sc_of_imbalance_bull_active")),
            "sc_of_imbalance_bear_active": _safe_int(last.get("sc_of_imbalance_bear_active")),

            "sc_of_zone_bull_active": _safe_int(last.get("sc_of_zone_bull_active")),
            "sc_of_zone_bear_active": _safe_int(last.get("sc_of_zone_bear_active")),

            "sc_of_accept_bull": _safe_int(last.get("sc_of_accept_bull")),
            "sc_of_accept_bear": _safe_int(last.get("sc_of_accept_bear")),
        },
    }

    return payload