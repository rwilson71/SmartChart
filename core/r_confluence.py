from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.a_indicators import ema, atr


DEFAULT_R_CONFLUENCE_CONFIG: Dict[str, Any] = {
    # Main
    "engine_on": True,

    # MTF Trend
    "ema_fast_len": 20,
    "ema_slow_len": 200,
    "mtf_weights": [1.0, 1.0, 1.0, 1.0],
    "mtf_bias_bull_min": 0.25,
    "mtf_bias_bear_max": -0.25,

    # Macro Location
    "fib_pivot_len": 5,
    "macro_fib_mode": "33-50",  # 25-33 | 33-50 | 50-61.5 | 66-78
    "macro_pad_pct": 0.20,
    "macro_min_abs": 20.0,
    "macro_atr_mult": 1.2,

    # Activity Layers
    "atr_len": 14,
    "vol_len": 20,
    "delta_smooth_len": 5,
    "delta_norm_len": 50,
    "den_q_len": 200,
    "den_q_pct": 92.0,
    "den_depth_atr": 0.45,
    "imb_atr_len": 28,
    "imb_atr_mult": 1.5,
    "imb_body_pct": 70.0,
    "liq_pivot_len": 5,
    "liq_sweep_ttl": 12,
    "core_pad_pct": 0.08,
    "core_min_abs": 6.0,
    "core_atr_mult": 0.50,

    # Volume Profile
    "vp_lookback": 120,
    "vp_bins": 24,
    "vp_va_percent": 68,
    "vp_volume_type": "Both",  # Both | Bullish | Bearish
    "vp_bias_tol_pct": 0.05,

    # Trigger / State
    "mom_mode": "MACD",  # MACD | MFI
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_sig_len": 9,
    "mfi_fast_len": 5,
    "mfi_slow_len": 21,
    "body_min_pct": 0.25,
    "trigger_ttl": 8,
    "hold_bars": 20,
}


# =============================================================================
# HELPERS
# =============================================================================

def _clamp(v: pd.Series | float, lo: float, hi: float):
    if isinstance(v, pd.Series):
        return v.clip(lower=lo, upper=hi)
    return max(lo, min(hi, float(v)))


def _boolf(v: pd.Series | bool) -> pd.Series | float:
    if isinstance(v, pd.Series):
        return v.astype(float)
    return 1.0 if bool(v) else 0.0


def _safe_div(a: pd.Series, b: pd.Series | float, fill: float = 0.0) -> pd.Series:
    if isinstance(b, (int, float)):
        b = pd.Series(float(b), index=a.index)
    out = a / b.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _rolling_percentile(series: pd.Series, window: int, q: float) -> pd.Series:
    qf = float(q) / 100.0
    return series.rolling(window, min_periods=window).quantile(qf)


def _bars_since(cond: pd.Series) -> pd.Series:
    cond = cond.fillna(False).astype(bool)
    out = np.full(len(cond), np.nan, dtype=float)
    last_true = -1
    for i, v in enumerate(cond.to_numpy()):
        if v:
            last_true = i
            out[i] = 0.0
        elif last_true >= 0:
            out[i] = float(i - last_true)
    return pd.Series(out, index=cond.index)


def _pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    vals = high.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    for i in range(left, len(vals) - right):
        c = vals[i]
        if np.isnan(c):
            continue
        left_slice = vals[i - left:i]
        right_slice = vals[i + 1:i + right + 1]
        if np.all(c > left_slice) and np.all(c >= right_slice):
            out[i] = c
    return pd.Series(out, index=high.index)


def _pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    vals = low.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    for i in range(left, len(vals) - right):
        c = vals[i]
        if np.isnan(c):
            continue
        left_slice = vals[i - left:i]
        right_slice = vals[i + 1:i + right + 1]
        if np.all(c < left_slice) and np.all(c <= right_slice):
            out[i] = c
    return pd.Series(out, index=low.index)


def _valuewhen_last(source: pd.Series, value: pd.Series) -> pd.Series:
    out = np.full(len(source), np.nan, dtype=float)
    last = np.nan
    s = source.to_numpy(dtype=float)
    v = value.to_numpy(dtype=float)
    for i in range(len(source)):
        if not np.isnan(s[i]):
            last = v[i]
        out[i] = last
    return pd.Series(out, index=source.index)


def _fib_level(direction: pd.Series, anchor: pd.Series, rng: pd.Series, pct: float) -> pd.Series:
    out = pd.Series(np.nan, index=anchor.index, dtype=float)
    bull = direction == 1
    bear = direction == -1
    out.loc[bull] = anchor.loc[bull] + rng.loc[bull] * pct
    out.loc[bear] = anchor.loc[bear] - rng.loc[bear] * pct
    return out


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df[["open", "high", "low", "close", "volume"]].resample(rule).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _map_htf_state(df: pd.DataFrame, rule: str, ema_fast_len: int, ema_slow_len: int) -> pd.Series:
    htf = _resample_ohlcv(df, rule)
    htf_fast = ema(htf["close"], ema_fast_len)
    htf_slow = ema(htf["close"], ema_slow_len)

    state = pd.Series(0.0, index=htf.index)
    bull = (htf["close"] > htf_fast) & (htf_fast > htf_slow)
    bear = (htf["close"] < htf_fast) & (htf_fast < htf_slow)
    state.loc[bull] = 1.0
    state.loc[bear] = -1.0

    return state.reindex(df.index, method="ffill").fillna(0.0)


def _infer_freq_rule(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "1min"

    diffs = pd.Series(index[1:] - index[:-1]).dropna()
    if diffs.empty:
        return "1min"

    med = diffs.median()
    mins = max(int(round(med.total_seconds() / 60.0)), 1)
    return f"{mins}min"


def _rule_from_multiplier(base_rule: str, mult: int) -> str:
    try:
        base = int(base_rule.replace("min", ""))
    except Exception:
        base = 1
    return f"{base * mult}min"


def _compute_volume_profile_window(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    vol_arr: np.ndarray,
    volume_type: str,
    bins: int,
    va_percent: int,
) -> Tuple[float, float, float, float, float, float, float, float, float, int, int, int, float]:
    highest = float(np.nanmax(high_arr))
    lowest = float(np.nanmin(low_arr))
    step = (highest - lowest) / max(bins - 1, 1)

    if not np.isfinite(step) or step <= 0:
        return (np.nan, np.nan, np.nan, 0.5, 0.5, highest, lowest, np.nan, 0.0, 0, 0, 0, 0.0)

    arr = np.zeros(bins, dtype=float)
    bull_arr = np.zeros(bins, dtype=float)
    bear_arr = np.zeros(bins, dtype=float)

    for i in range(len(high_arr)):
        is_bull = close_arr[i] >= open_arr[i]
        include = (
            True if volume_type == "Both"
            else is_bull if volume_type == "Bullish"
            else (not is_bull)
        )
        if not include:
            continue

        for j in range(bins):
            lvl = lowest + step * j
            if lvl >= low_arr[i] and lvl < high_arr[i]:
                arr[j] += vol_arr[i]
                if is_bull:
                    bull_arr[j] += vol_arr[i]
                else:
                    bear_arr[j] += vol_arr[i]

    total_vol = float(arr.sum())
    if total_vol <= 0:
        return (np.nan, np.nan, np.nan, 0.5, 0.5, highest, lowest, step, 0.0, 0, 0, 0, 0.0)

    poc_vol = float(arr.max())
    poc_idx = int(arr.argmax())
    poc = lowest + step * poc_idx

    poc_bull = bull_arr[poc_idx]
    poc_bear = bear_arr[poc_idx]
    poc_sum = poc_bull + poc_bear
    bull_share = float(poc_bull / poc_sum) if poc_sum > 0 else 0.5
    bear_share = float(poc_bear / poc_sum) if poc_sum > 0 else 0.5

    va_target = total_vol * va_percent / 100.0
    va_sum = poc_vol
    va_dn = poc_idx
    va_up = poc_idx

    while va_sum < va_target:
        v_up = arr[va_up + 1] if va_up < bins - 1 else 0.0
        v_dn = arr[va_dn - 1] if va_dn > 0 else 0.0
        if v_up == 0 and v_dn == 0:
            break
        if v_up >= v_dn:
            va_up += 1
            va_sum += v_up
        else:
            va_dn -= 1
            va_sum += v_dn

    vah = lowest + step * va_up
    val = lowest + step * va_dn

    return (
        poc,
        vah,
        val,
        bull_share,
        bear_share,
        highest,
        lowest,
        step,
        poc_vol,
        poc_idx,
        va_dn,
        va_up,
        total_vol,
    )


def _compute_volume_profile_series(
    df: pd.DataFrame,
    lookback: int,
    bins: int,
    va_percent: int,
    volume_type: str,
) -> pd.DataFrame:
    n = len(df)
    cols = {
        "vp_poc": np.full(n, np.nan),
        "vp_vah": np.full(n, np.nan),
        "vp_val": np.full(n, np.nan),
        "vp_bull_share": np.full(n, np.nan),
        "vp_bear_share": np.full(n, np.nan),
        "vp_highest": np.full(n, np.nan),
        "vp_lowest": np.full(n, np.nan),
        "vp_step": np.full(n, np.nan),
        "vp_max_vol": np.full(n, np.nan),
        "vp_poc_idx": np.full(n, np.nan),
        "vp_va_dn": np.full(n, np.nan),
        "vp_va_up": np.full(n, np.nan),
        "vp_total_vol": np.full(n, np.nan),
    }

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    keys = list(cols.keys())

    for i in range(lookback - 1, n):
        s = slice(i - lookback + 1, i + 1)
        vals = _compute_volume_profile_window(
            h[s], l[s], o[s], c[s], v[s], volume_type, bins, va_percent
        )
        for k, key in enumerate(keys):
            cols[key][i] = vals[k]

    return pd.DataFrame(cols, index=df.index)


def _dir_label(v: float | int | None) -> str:
    if v == 1:
        return "bull"
    if v == -1:
        return "bear"
    return "neutral"


def _strength_label(v: float | int | None) -> str:
    if v == 3:
        return "A+"
    if v == 2:
        return "B"
    if v == 1:
        return "C"
    return "N"


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        return None


def _safe_bool(v: Any) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return bool(v)


# =============================================================================
# MAIN ENGINE
# =============================================================================

def r_confluence(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_R_CONFLUENCE_CONFIG, **(config or {})}
    out = df.copy()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"r_confluence requires columns: {required}. Missing: {missing}")
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("r_confluence requires a DatetimeIndex for MTF and previous-day logic.")

    # -------------------------------------------------------------------------
    # Base series
    # -------------------------------------------------------------------------
    out["atr_now"] = atr(out["high"], out["low"], out["close"], int(cfg["atr_len"]))
    out["vol_ma"] = out["volume"].rolling(int(cfg["vol_len"]), min_periods=1).mean()
    out["rng"] = (out["high"] - out["low"]).clip(lower=1e-9)

    # -------------------------------------------------------------------------
    # 1) MTF trend agreement
    # -------------------------------------------------------------------------
    base_rule = _infer_freq_rule(out.index)
    rules = [
        _rule_from_multiplier(base_rule, 15),
        _rule_from_multiplier(base_rule, 60),
        _rule_from_multiplier(base_rule, 240),
        "1D",
    ]

    out["mtf_s1"] = _map_htf_state(out, rules[0], int(cfg["ema_fast_len"]), int(cfg["ema_slow_len"]))
    out["mtf_s2"] = _map_htf_state(out, rules[1], int(cfg["ema_fast_len"]), int(cfg["ema_slow_len"]))
    out["mtf_s3"] = _map_htf_state(out, rules[2], int(cfg["ema_fast_len"]), int(cfg["ema_slow_len"]))
    out["mtf_s4"] = _map_htf_state(out, rules[3], int(cfg["ema_fast_len"]), int(cfg["ema_slow_len"]))

    w = cfg["mtf_weights"]
    wsum = max(float(sum(w)), 1e-4)
    out["mtf_raw"] = (
        out["mtf_s1"] * w[0]
        + out["mtf_s2"] * w[1]
        + out["mtf_s3"] * w[2]
        + out["mtf_s4"] * w[3]
    )
    out["mtf_avg"] = out["mtf_raw"] / wsum

    out["mtf_dir"] = 0
    out.loc[out["mtf_avg"] >= float(cfg["mtf_bias_bull_min"]), "mtf_dir"] = 1
    out.loc[out["mtf_avg"] <= float(cfg["mtf_bias_bear_max"]), "mtf_dir"] = -1

    # -------------------------------------------------------------------------
    # 2) Macro location
    # -------------------------------------------------------------------------
    daily = _resample_ohlcv(out, "1D")
    out["prev_day_high"] = daily["high"].shift(1).reindex(out.index, method="ffill")
    out["prev_day_low"] = daily["low"].shift(1).reindex(out.index, method="ffill")

    piv = int(cfg["fib_pivot_len"])
    out["ph"] = _pivot_high(out["high"], piv, piv)
    out["pl"] = _pivot_low(out["low"], piv, piv)
    out["last_pivot_high"] = out["ph"].ffill()
    out["last_pivot_low"] = out["pl"].ffill()

    out["bull_fib_ready"] = (
        out["prev_day_low"].notna()
        & out["last_pivot_high"].notna()
        & (out["last_pivot_high"] > out["prev_day_low"])
    )
    out["bear_fib_ready"] = (
        out["prev_day_high"].notna()
        & out["last_pivot_low"].notna()
        & (out["last_pivot_low"] < out["prev_day_high"])
    )

    out["macro_dir"] = 0
    out.loc[(out["mtf_dir"] == 1) & out["bull_fib_ready"], "macro_dir"] = 1
    out.loc[(out["mtf_dir"] == -1) & out["bear_fib_ready"], "macro_dir"] = -1

    out["fib_a"] = np.where(
        out["macro_dir"] == 1,
        out["prev_day_low"],
        np.where(out["macro_dir"] == -1, out["prev_day_high"], np.nan),
    )
    out["fib_b"] = np.where(
        out["macro_dir"] == 1,
        out["last_pivot_high"],
        np.where(out["macro_dir"] == -1, out["last_pivot_low"], np.nan),
    )
    out["fib_range"] = np.where(out["macro_dir"] != 0, np.abs(out["fib_b"] - out["fib_a"]), np.nan)

    mode = str(cfg["macro_fib_mode"])
    if mode == "25-33":
        out["macro_l1"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.25)
        out["macro_l2"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.33)
    elif mode == "33-50":
        out["macro_l1"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.33)
        out["macro_l2"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.50)
    elif mode == "50-61.5":
        out["macro_l1"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.50)
        out["macro_l2"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.615)
    else:
        out["macro_l1"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.66)
        out["macro_l2"] = _fib_level(out["macro_dir"], out["fib_a"], out["fib_range"], 0.78)

    out["raw_macro_lo"] = np.where(
        (out["macro_dir"] != 0) & out["macro_l1"].notna() & out["macro_l2"].notna(),
        np.minimum(out["macro_l1"], out["macro_l2"]),
        np.nan,
    )
    out["raw_macro_hi"] = np.where(
        (out["macro_dir"] != 0) & out["macro_l1"].notna() & out["macro_l2"].notna(),
        np.maximum(out["macro_l1"], out["macro_l2"]),
        np.nan,
    )

    out["macro_pad"] = (out["raw_macro_hi"] - out["raw_macro_lo"]) * float(cfg["macro_pad_pct"])
    out["base_macro_lo"] = out["raw_macro_lo"] - out["macro_pad"]
    out["base_macro_hi"] = out["raw_macro_hi"] + out["macro_pad"]
    out["base_macro_size"] = (out["base_macro_hi"] - out["base_macro_lo"]).abs()
    out["macro_floor"] = out["atr_now"] * float(cfg["macro_atr_mult"])
    out["final_macro_size"] = np.maximum(
        out["base_macro_size"],
        np.maximum(out["macro_floor"], float(cfg["macro_min_abs"])),
    )
    out["macro_extra_half"] = np.maximum(
        0.0,
        (out["final_macro_size"] - out["base_macro_size"]) * 0.5,
    )
    out["macro_zone_lo"] = out["base_macro_lo"] - out["macro_extra_half"]
    out["macro_zone_hi"] = out["base_macro_hi"] + out["macro_extra_half"]
    out["macro_zone_ready"] = (
        (out["macro_dir"] != 0)
        & out["macro_zone_lo"].notna()
        & out["macro_zone_hi"].notna()
    )
    out["in_macro_zone"] = (
        out["macro_zone_ready"]
        & (out["close"] >= out["macro_zone_lo"])
        & (out["close"] <= out["macro_zone_hi"])
    )

    # -------------------------------------------------------------------------
    # 3) Smart-money activity layers
    # -------------------------------------------------------------------------
    out["delta_raw"] = out["volume"] * (out["close"] - out["open"]) / out["rng"]
    out["delta_sm"] = ema(out["delta_raw"], int(cfg["delta_smooth_len"]))
    out["delta_abs"] = out["delta_sm"].abs()
    out["delta_base"] = out["delta_abs"].rolling(int(cfg["delta_norm_len"]), min_periods=1).mean()
    out["delta_norm"] = _clamp(_safe_div(out["delta_sm"], out["delta_base"]), -1.0, 1.0)

    out["den_metric"] = out["delta_sm"].abs()
    out["den_thr"] = _rolling_percentile(out["den_metric"], int(cfg["den_q_len"]), float(cfg["den_q_pct"]))
    out["density_event"] = out["den_metric"] >= out["den_thr"]
    out["density_top"] = out["close"] + out["atr_now"] * float(cfg["den_depth_atr"])
    out["density_bot"] = out["close"] - out["atr_now"] * float(cfg["den_depth_atr"])
    out["density_bull"] = out["density_event"] & (out["delta_sm"] >= 0)
    out["density_bear"] = out["density_event"] & (out["delta_sm"] < 0)

    out["density_dir"] = 0
    out.loc[(out["macro_dir"] == 1) & out["density_bull"], "density_dir"] = 1
    out.loc[(out["macro_dir"] == -1) & out["density_bear"], "density_dir"] = -1

    out["imb_atr"] = atr(out["high"], out["low"], out["close"], int(cfg["imb_atr_len"]))
    out["body_pct"] = ((out["close"] - out["open"]).abs() / out["rng"]) * 100.0
    out["is_up_1"] = out["open"].shift(1) <= out["close"].shift(1)
    out["price_diff"] = (out["high"].shift(1) - out["low"].shift(1)).abs()
    out["big_body"] = out["body_pct"].shift(1) >= float(cfg["imb_body_pct"])

    out["gap_closed"] = np.where(
        out["is_up_1"],
        out["high"].shift(2) >= out["low"],
        out["low"].shift(2) <= out["high"],
    )

    out["imbalance_event"] = (
        (out["price_diff"] > out["imb_atr"] * float(cfg["imb_atr_mult"]))
        & out["big_body"]
        & (~pd.Series(out["gap_closed"], index=out.index).fillna(False))
    )
    out["imb_dir"] = np.where(out["imbalance_event"], np.where(out["is_up_1"], 1, -1), 0)
    out["imb_top"] = np.where(out["is_up_1"], out["low"], out["low"].shift(2))
    out["imb_bot"] = np.where(out["is_up_1"], out["high"].shift(2), out["high"])

    out["imb_bull_active"] = (
        out["imbalance_event"].shift(1).fillna(False)
        & (pd.Series(out["imb_dir"], index=out.index).shift(1) == 1)
        & (out["high"] >= pd.Series(out["imb_bot"], index=out.index).shift(1))
        & (out["low"] <= pd.Series(out["imb_top"], index=out.index).shift(1))
    )
    out["imb_bear_active"] = (
        out["imbalance_event"].shift(1).fillna(False)
        & (pd.Series(out["imb_dir"], index=out.index).shift(1) == -1)
        & (out["high"] >= pd.Series(out["imb_bot"], index=out.index).shift(1))
        & (out["low"] <= pd.Series(out["imb_top"], index=out.index).shift(1))
    )

    out["imb_dir_active"] = 0
    out.loc[(out["macro_dir"] == 1) & out["imb_bull_active"], "imb_dir_active"] = 1
    out.loc[(out["macro_dir"] == -1) & out["imb_bear_active"], "imb_dir_active"] = -1

    vp = _compute_volume_profile_series(
        out,
        int(cfg["vp_lookback"]),
        int(cfg["vp_bins"]),
        int(cfg["vp_va_percent"]),
        str(cfg["vp_volume_type"]),
    )
    out = pd.concat([out, vp], axis=1)

    out["vp_ready"] = out["vp_poc"].notna() & out["vp_vah"].notna() & out["vp_val"].notna()
    out["vp_inside_va"] = out["vp_ready"] & (out["close"] >= out["vp_val"]) & (out["close"] <= out["vp_vah"])
    out["vp_outside_va"] = out["vp_ready"] & (~out["vp_inside_va"])
    out["vp_dist_to_poc_pct"] = np.where(
        out["vp_ready"] & (out["vp_poc"] != 0),
        (out["close"] - out["vp_poc"]) / out["vp_poc"] * 100.0,
        np.nan,
    )
    out["vp_neutral_band"] = out["vp_ready"] & (out["vp_dist_to_poc_pct"].abs() <= float(cfg["vp_bias_tol_pct"]))

    out["vp_bias"] = 0
    out.loc[out["vp_ready"] & (~out["vp_neutral_band"]) & (out["close"] > out["vp_poc"]), "vp_bias"] = 1
    out.loc[out["vp_ready"] & (~out["vp_neutral_band"]) & (out["close"] < out["vp_poc"]), "vp_bias"] = -1

    out["vp_dir"] = 0
    bull_vp = (
        (out["macro_dir"] == 1)
        & (out["vp_bias"] >= 0)
        & (out["vp_inside_va"] | (out["close"] >= out["vp_poc"]))
    )
    bear_vp = (
        (out["macro_dir"] == -1)
        & (out["vp_bias"] <= 0)
        & (out["vp_inside_va"] | (out["close"] <= out["vp_poc"]))
    )
    out.loc[bull_vp, "vp_dir"] = 1
    out.loc[bear_vp, "vp_dir"] = -1

    lp = int(cfg["liq_pivot_len"])
    out["pivot_high"] = _pivot_high(out["high"], lp, lp)
    out["pivot_low"] = _pivot_low(out["low"], lp, lp)
    out["liq_high"] = _valuewhen_last(out["pivot_high"], out["pivot_high"])
    out["liq_low"] = _valuewhen_last(out["pivot_low"], out["pivot_low"])

    out["bull_sweep_now"] = (
        (out["macro_dir"] == 1)
        & out["liq_low"].notna()
        & (out["low"] < out["liq_low"])
        & (out["close"] > out["liq_low"])
    )
    out["bear_sweep_now"] = (
        (out["macro_dir"] == -1)
        & out["liq_high"].notna()
        & (out["high"] > out["liq_high"])
        & (out["close"] < out["liq_high"])
    )
    out["bull_sweep_bars"] = _bars_since(out["bull_sweep_now"])
    out["bear_sweep_bars"] = _bars_since(out["bear_sweep_now"])
    out["liq_bull_active"] = (
        out["bull_sweep_bars"].notna()
        & (out["bull_sweep_bars"] >= 0)
        & (out["bull_sweep_bars"] <= int(cfg["liq_sweep_ttl"]))
    )
    out["liq_bear_active"] = (
        out["bear_sweep_bars"].notna()
        & (out["bear_sweep_bars"] >= 0)
        & (out["bear_sweep_bars"] <= int(cfg["liq_sweep_ttl"]))
    )

    out["liq_dir"] = 0
    out.loc[(out["macro_dir"] == 1) & out["liq_bull_active"], "liq_dir"] = 1
    out.loc[(out["macro_dir"] == -1) & out["liq_bear_active"], "liq_dir"] = -1

    # -------------------------------------------------------------------------
    # 4) Active confluence core
    # -------------------------------------------------------------------------
    out["act_count_bull"] = (
        (out["density_dir"] == 1).astype(int)
        + (out["imb_dir_active"] == 1).astype(int)
        + (out["vp_dir"] == 1).astype(int)
        + (out["liq_dir"] == 1).astype(int)
    )
    out["act_count_bear"] = (
        (out["density_dir"] == -1).astype(int)
        + (out["imb_dir_active"] == -1).astype(int)
        + (out["vp_dir"] == -1).astype(int)
        + (out["liq_dir"] == -1).astype(int)
    )

    out["activity_dir"] = 0
    out.loc[(out["macro_dir"] == 1) & (out["act_count_bull"] >= 2), "activity_dir"] = 1
    out.loc[(out["macro_dir"] == -1) & (out["act_count_bear"] >= 2), "activity_dir"] = -1

    core_raw_lo = np.full(len(out), np.nan)
    core_raw_hi = np.full(len(out), np.nan)

    imb_bot_shift = out["imb_bot"].shift(1)
    imb_top_shift = out["imb_top"].shift(1)

    for i in range(len(out)):
        ad = int(out["activity_dir"].iat[i])
        if ad == 0:
            continue

        tmp_lo = out["macro_zone_lo"].iat[i]
        tmp_hi = out["macro_zone_hi"].iat[i]
        if np.isnan(tmp_lo) or np.isnan(tmp_hi):
            continue

        if ad == 1:
            if out["density_dir"].iat[i] == 1:
                tmp_lo = max(tmp_lo, out["density_bot"].iat[i])
                tmp_hi = min(tmp_hi, out["density_top"].iat[i])

            if out["imb_dir_active"].iat[i] == 1:
                imb_bot_prev = imb_bot_shift.iat[i]
                imb_top_prev = imb_top_shift.iat[i]
                if np.isfinite(imb_bot_prev) and np.isfinite(imb_top_prev):
                    tmp_lo = max(tmp_lo, imb_bot_prev)
                    tmp_hi = min(tmp_hi, imb_top_prev)

            if out["vp_dir"].iat[i] == 1 and bool(out["vp_ready"].iat[i]):
                tmp_lo = max(tmp_lo, out["vp_val"].iat[i])
                tmp_hi = min(tmp_hi, out["vp_vah"].iat[i])

        elif ad == -1:
            if out["density_dir"].iat[i] == -1:
                tmp_lo = max(tmp_lo, out["density_bot"].iat[i])
                tmp_hi = min(tmp_hi, out["density_top"].iat[i])

            if out["imb_dir_active"].iat[i] == -1:
                imb_bot_prev = imb_bot_shift.iat[i]
                imb_top_prev = imb_top_shift.iat[i]
                if np.isfinite(imb_bot_prev) and np.isfinite(imb_top_prev):
                    tmp_lo = max(tmp_lo, imb_bot_prev)
                    tmp_hi = min(tmp_hi, imb_top_prev)

            if out["vp_dir"].iat[i] == -1 and bool(out["vp_ready"].iat[i]):
                tmp_lo = max(tmp_lo, out["vp_val"].iat[i])
                tmp_hi = min(tmp_hi, out["vp_vah"].iat[i])

        core_raw_lo[i] = tmp_lo
        core_raw_hi[i] = tmp_hi

    out["core_raw_lo"] = core_raw_lo
    out["core_raw_hi"] = core_raw_hi
    out["core_overlap_valid"] = (
        (out["activity_dir"] != 0)
        & out["core_raw_lo"].notna()
        & out["core_raw_hi"].notna()
        & (out["core_raw_hi"] > out["core_raw_lo"])
    )
    out["core_raw_size"] = np.where(
        out["core_overlap_valid"],
        (out["core_raw_hi"] - out["core_raw_lo"]).abs(),
        np.nan,
    )
    out["core_pad"] = np.where(
        out["core_overlap_valid"],
        out["core_raw_size"] * float(cfg["core_pad_pct"]),
        np.nan,
    )
    out["core_base_lo"] = np.where(out["core_overlap_valid"], out["core_raw_lo"] - out["core_pad"], np.nan)
    out["core_base_hi"] = np.where(out["core_overlap_valid"], out["core_raw_hi"] + out["core_pad"], np.nan)
    out["core_base_size"] = np.where(
        out["core_overlap_valid"],
        (out["core_base_hi"] - out["core_base_lo"]).abs(),
        np.nan,
    )
    out["core_floor"] = out["atr_now"] * float(cfg["core_atr_mult"])
    out["core_final_size"] = np.where(
        out["core_overlap_valid"],
        np.maximum(out["core_base_size"], np.maximum(out["core_floor"], float(cfg["core_min_abs"]))),
        np.nan,
    )
    out["core_extra_half"] = np.where(
        out["core_overlap_valid"],
        np.maximum(0.0, (out["core_final_size"] - out["core_base_size"]) * 0.5),
        np.nan,
    )
    out["core_zone_lo"] = np.where(out["core_overlap_valid"], out["core_base_lo"] - out["core_extra_half"], np.nan)
    out["core_zone_hi"] = np.where(out["core_overlap_valid"], out["core_base_hi"] + out["core_extra_half"], np.nan)
    out["core_zone_ready"] = (
        (out["activity_dir"] != 0)
        & out["core_zone_lo"].notna()
        & out["core_zone_hi"].notna()
    )
    out["in_core_zone"] = (
        out["core_zone_ready"]
        & (out["close"] >= out["core_zone_lo"])
        & (out["close"] <= out["core_zone_hi"])
    )
    out["vp_agrees_with_confluence"] = (out["vp_dir"] != 0) & (out["vp_dir"] == out["activity_dir"])

    # -------------------------------------------------------------------------
    # 5) Momentum + trigger + state
    # -------------------------------------------------------------------------
    out["macd_line"] = ema(out["close"], int(cfg["macd_fast"])) - ema(out["close"], int(cfg["macd_slow"]))
    out["macd_sig"] = ema(out["macd_line"], int(cfg["macd_sig_len"]))

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    rmf = tp * out["volume"]
    tp_diff = tp.diff()

    pos_flow = rmf.where(tp_diff > 0, 0.0)
    neg_flow = rmf.where(tp_diff < 0, 0.0).abs()

    mfi_fast_pos = pos_flow.rolling(int(cfg["mfi_fast_len"]), min_periods=int(cfg["mfi_fast_len"])).sum()
    mfi_fast_neg = neg_flow.rolling(int(cfg["mfi_fast_len"]), min_periods=int(cfg["mfi_fast_len"])).sum()
    mfi_slow_pos = pos_flow.rolling(int(cfg["mfi_slow_len"]), min_periods=int(cfg["mfi_slow_len"])).sum()
    mfi_slow_neg = neg_flow.rolling(int(cfg["mfi_slow_len"]), min_periods=int(cfg["mfi_slow_len"])).sum()

    out["mfi_fast"] = 100.0 - (
        100.0 / (1.0 + _safe_div(mfi_fast_pos, mfi_fast_neg.replace(0.0, np.nan), fill=np.nan))
    )
    out["mfi_slow"] = 100.0 - (
        100.0 / (1.0 + _safe_div(mfi_slow_pos, mfi_slow_neg.replace(0.0, np.nan), fill=np.nan))
    )

    mom_mode = str(cfg["mom_mode"]).upper()
    out["mom_bull_ok"] = np.where(
        mom_mode == "MACD",
        (out["macd_line"] > out["macd_sig"]) & (out["macd_line"] >= out["macd_line"].shift(1)),
        (out["mfi_slow"] > 50.0) & (out["mfi_fast"] > out["mfi_fast"].shift(1)),
    )
    out["mom_bear_ok"] = np.where(
        mom_mode == "MACD",
        (out["macd_line"] < out["macd_sig"]) & (out["macd_line"] <= out["macd_line"].shift(1)),
        (out["mfi_slow"] < 50.0) & (out["mfi_fast"] < out["mfi_fast"].shift(1)),
    )

    out["momentum_ok"] = False
    out.loc[out["activity_dir"] == 1, "momentum_ok"] = pd.Series(out["mom_bull_ok"], index=out.index)
    out.loc[out["activity_dir"] == -1, "momentum_ok"] = pd.Series(out["mom_bear_ok"], index=out.index)

    out["body"] = (out["close"] - out["open"]).abs()
    out["body_pct_now"] = out["body"] / out["rng"]
    out["bull_body"] = (out["close"] > out["open"]) & (out["body_pct_now"] >= float(cfg["body_min_pct"]))
    out["bear_body"] = (out["close"] < out["open"]) & (out["body_pct_now"] >= float(cfg["body_min_pct"]))

    out["ema_fast"] = ema(out["close"], int(cfg["ema_fast_len"]))

    out["core_retest_bull"] = (
        out["core_zone_ready"]
        & (out["low"] <= out["core_zone_hi"])
        & (out["close"] >= out["core_zone_lo"])
    )
    out["core_retest_bear"] = (
        out["core_zone_ready"]
        & (out["high"] >= out["core_zone_lo"])
        & (out["close"] <= out["core_zone_hi"])
    )
    out["ema_retest_bull"] = (out["low"] <= out["ema_fast"]) & (out["close"] >= out["ema_fast"])
    out["ema_retest_bear"] = (out["high"] >= out["ema_fast"]) & (out["close"] <= out["ema_fast"])

    out["retest_ok"] = False
    out.loc[out["activity_dir"] == 1, "retest_ok"] = out["core_retest_bull"] | out["ema_retest_bull"]
    out.loc[out["activity_dir"] == -1, "retest_ok"] = out["core_retest_bear"] | out["ema_retest_bear"]

    out["break_ok"] = False
    out.loc[out["activity_dir"] == 1, "break_ok"] = out["close"] > out["high"].shift(1)
    out.loc[out["activity_dir"] == -1, "break_ok"] = out["close"] < out["low"].shift(1)

    out["trigger_now"] = False
    out.loc[out["activity_dir"] == 1, "trigger_now"] = out["bull_body"] & out["retest_ok"] & out["break_ok"]
    out.loc[out["activity_dir"] == -1, "trigger_now"] = out["bear_body"] & out["retest_ok"] & out["break_ok"]

    out["trigger_bars"] = _bars_since(out["trigger_now"])
    out["trigger_ok"] = (
        out["trigger_bars"].notna()
        & (out["trigger_bars"] >= 0)
        & (out["trigger_bars"] <= int(cfg["trigger_ttl"]))
    )

    out["ic_ready"] = (
        bool(cfg["engine_on"])
        & out["macro_zone_ready"]
        & out["core_zone_ready"]
        & (out["activity_dir"] != 0)
        & out["momentum_ok"]
    )
    out["ic_valid"] = out["ic_ready"] & out["trigger_ok"]
    out["ic_in_macro"] = out["ic_ready"] & out["in_macro_zone"]
    out["ic_in_core"] = out["ic_ready"] & out["in_core_zone"]

    out["ic_ttl_raw"] = _bars_since(out["ic_in_core"])
    out["ic_active"] = (
        out["ic_valid"]
        & out["ic_ttl_raw"].notna()
        & (out["ic_ttl_raw"] >= 0)
        & (out["ic_ttl_raw"] <= int(cfg["hold_bars"]))
    )
    out["ic_ttl"] = np.where(
        out["ic_active"],
        np.maximum(0, int(cfg["hold_bars"]) - out["ic_ttl_raw"]),
        0,
    )

    out["ic_score"] = 0
    out["ic_score"] += (out["mtf_dir"] != 0).astype(int)
    out["ic_score"] += out["macro_zone_ready"].astype(int)
    out["ic_score"] += out["core_zone_ready"].astype(int)
    out["ic_score"] += out["momentum_ok"].astype(int)
    out["ic_score"] += out["trigger_ok"].astype(int)

    out["ic_strength"] = 0
    out.loc[out["ic_score"] == 5, "ic_strength"] = 3
    out.loc[(out["ic_score"] >= 4) & (out["ic_score"] < 5), "ic_strength"] = 2
    out.loc[(out["ic_score"] >= 3) & (out["ic_score"] < 4), "ic_strength"] = 1

    out["ic_quality"] = _clamp(
        _boolf(out["in_macro_zone"]) * 0.15
        + _boolf(out["in_core_zone"]) * 0.20
        + _boolf(out["momentum_ok"]) * 0.15
        + _boolf(out["trigger_ok"]) * 0.20
        + _boolf((out["act_count_bull"] >= 2) | (out["act_count_bear"] >= 2)) * 0.15
        + _boolf(out["ic_active"]) * 0.15,
        0.0,
        1.0,
    )

    # -------------------------------------------------------------------------
    # Export aliases for parity / website
    # -------------------------------------------------------------------------
    out["sc_confl_dir"] = out["activity_dir"]
    out["sc_confl_mtf_avg"] = out["mtf_avg"]
    out["sc_confl_mtf_dir"] = out["mtf_dir"]
    out["sc_confl_macro_dir"] = out["macro_dir"]
    out["sc_confl_macro_ready"] = out["macro_zone_ready"].astype(float)
    out["sc_confl_in_macro"] = out["in_macro_zone"].astype(float)
    out["sc_confl_core_ready"] = out["core_zone_ready"].astype(float)
    out["sc_confl_in_core"] = out["in_core_zone"].astype(float)
    out["sc_confl_density_dir"] = out["density_dir"]
    out["sc_confl_density_event"] = out["density_event"].astype(float)
    out["sc_confl_delta_norm"] = out["delta_norm"]
    out["sc_confl_imb_dir"] = out["imb_dir_active"]
    out["sc_confl_imb_live"] = (out["imb_bull_active"] | out["imb_bear_active"]).astype(float)
    out["sc_confl_vp_bias"] = out["vp_bias"]
    out["sc_confl_vp_dir"] = out["vp_dir"]
    out["sc_confl_vp_inside_va"] = out["vp_inside_va"].astype(float)
    out["sc_confl_vp_dist_to_poc_pct"] = out["vp_dist_to_poc_pct"]
    out["sc_confl_vp_agree"] = out["vp_agrees_with_confluence"].astype(float)
    out["sc_confl_liq_dir"] = out["liq_dir"]
    out["sc_confl_momentum_ok"] = out["momentum_ok"].astype(float)
    out["sc_confl_trigger_ok"] = out["trigger_ok"].astype(float)
    out["sc_confl_ready"] = out["ic_ready"].astype(float)
    out["sc_confl_valid"] = out["ic_valid"].astype(float)
    out["sc_confl_active"] = out["ic_active"].astype(float)
    out["sc_confl_ttl"] = out["ic_ttl"]
    out["sc_confl_score"] = out["ic_score"]
    out["sc_confl_strength"] = out["ic_strength"]
    out["sc_confl_quality"] = out["ic_quality"]
    out["sc_confl_macro_lo"] = out["macro_zone_lo"]
    out["sc_confl_macro_hi"] = out["macro_zone_hi"]
    out["sc_confl_core_lo"] = out["core_zone_lo"]
    out["sc_confl_core_hi"] = out["core_zone_hi"]
    out["sc_confl_vp_val"] = out["vp_val"]
    out["sc_confl_vp_poc"] = out["vp_poc"]
    out["sc_confl_vp_vah"] = out["vp_vah"]

    return out

def build_confluence_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = r_confluence(df, config=config)
    if result.empty:
        raise ValueError("Confluence payload build failed: empty dataframe result.")

    row = result.iloc[-1]

    direction = _safe_int(row.get("sc_confl_dir")) or 0
    mtf_dir = _safe_int(row.get("sc_confl_mtf_dir")) or 0
    macro_dir = _safe_int(row.get("sc_confl_macro_dir")) or 0
    vp_bias = _safe_int(row.get("sc_confl_vp_bias")) or 0
    vp_dir = _safe_int(row.get("sc_confl_vp_dir")) or 0
    liq_dir = _safe_int(row.get("sc_confl_liq_dir")) or 0
    density_dir = _safe_int(row.get("sc_confl_density_dir")) or 0
    imb_dir = _safe_int(row.get("sc_confl_imb_dir")) or 0
    ic_strength = _safe_int(row.get("sc_confl_strength")) or 0
    ic_score = _safe_int(row.get("sc_confl_score")) or 0
    ic_ttl = _safe_int(row.get("sc_confl_ttl")) or 0

    macro_ready = _safe_bool(row.get("sc_confl_macro_ready"))
    in_macro = _safe_bool(row.get("sc_confl_in_macro"))
    core_ready = _safe_bool(row.get("sc_confl_core_ready"))
    in_core = _safe_bool(row.get("sc_confl_in_core"))
    density_event = _safe_bool(row.get("sc_confl_density_event"))
    imbalance_live = _safe_bool(row.get("sc_confl_imb_live"))
    vp_inside_va = _safe_bool(row.get("sc_confl_vp_inside_va"))
    vp_agrees = _safe_bool(row.get("sc_confl_vp_agree"))
    momentum_ok = _safe_bool(row.get("sc_confl_momentum_ok"))
    trigger_ok = _safe_bool(row.get("sc_confl_trigger_ok"))
    ready = _safe_bool(row.get("sc_confl_ready"))
    valid = _safe_bool(row.get("sc_confl_valid"))
    active = _safe_bool(row.get("sc_confl_active"))

    mtf_avg = _safe_float(row.get("sc_confl_mtf_avg"))
    delta_norm = _safe_float(row.get("sc_confl_delta_norm"))
    quality = _safe_float(row.get("sc_confl_quality"))

    # -------------------------------------------------------------------------
    # ULTIMATE TRUTH DASHBOARD CONTRACT
    # -------------------------------------------------------------------------
    bias_signal = direction

    if bias_signal > 0:
        bias_label = "BULLISH"
    elif bias_signal < 0:
        bias_label = "BEARISH"
    else:
        bias_label = "NEUTRAL"

    # Market bias can stay aligned to confluence direction.
    # Fallback to MTF direction if confluence direction is neutral.
    if direction > 0:
        market_bias = "BULLISH"
    elif direction < 0:
        market_bias = "BEARISH"
    elif mtf_dir > 0:
        market_bias = "BULLISH"
    elif mtf_dir < 0:
        market_bias = "BEARISH"
    else:
        market_bias = "NEUTRAL"

    # Website / truth-table state hierarchy
    if direction == 1 and active:
        state = "BULLISH_ACTIVE"
    elif direction == -1 and active:
        state = "BEARISH_ACTIVE"
    elif direction == 1 and valid:
        state = "BULLISH_VALID"
    elif direction == -1 and valid:
        state = "BEARISH_VALID"
    elif direction == 1 and ready:
        state = "BULLISH_READY"
    elif direction == -1 and ready:
        state = "BEARISH_READY"
    elif direction == 1 and macro_ready and core_ready:
        state = "BULLISH_BUILDING"
    elif direction == -1 and macro_ready and core_ready:
        state = "BEARISH_BUILDING"
    elif direction == 1:
        state = "BULLISH"
    elif direction == -1:
        state = "BEARISH"
    else:
        state = "NEUTRAL"

    # Normalize strength to shared dashboard scale
    # Primary mapping from ic_strength, with a small uplift if score/quality are strong
    base_strength_map = {
        0: 0.0,
        1: 40.0,
        2: 70.0,
        3: 100.0,
    }
    indicator_strength = base_strength_map.get(ic_strength, 0.0)

    if quality is not None:
        indicator_strength = max(indicator_strength, round(float(quality) * 100.0, 2))

    if ic_score >= 5:
        indicator_strength = max(indicator_strength, 100.0)
    elif ic_score == 4:
        indicator_strength = max(indicator_strength, 75.0)
    elif ic_score == 3:
        indicator_strength = max(indicator_strength, 55.0)

    indicator_strength = round(min(100.0, max(0.0, indicator_strength)), 2)

    # Useful top-level text helpers for website cards
    if active:
        status_label = "ACTIVE"
    elif valid:
        status_label = "VALID"
    elif ready:
        status_label = "READY"
    else:
        status_label = "IDLE"

    summary = (
        f"{bias_label} confluence / "
        f"{status_label} / "
        f"strength {_strength_label(ic_strength)} / "
        f"score {ic_score}"
    )

    payload: Dict[str, Any] = {
        "indicator": "confluence",
        "name": "SmartChart Confluence Engine",
        "debug_version": "confluence_payload_v2",
        "timestamp": str(result.index[-1]),

        # ---------------------------------------------------------------------
        # SHARED ULTIMATE TRUTH DASHBOARD CONTRACT
        # ---------------------------------------------------------------------
        "state": state,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": indicator_strength,
        "market_bias": market_bias,

        # Helpful shared extras
        "status_label": status_label,
        "summary": summary,

        # Flat website-facing fields
        "direction": direction,
        "direction_label": _dir_label(direction),
        "mtf_dir": mtf_dir,
        "mtf_dir_label": _dir_label(mtf_dir),
        "macro_dir": macro_dir,
        "macro_dir_label": _dir_label(macro_dir),
        "vp_bias": vp_bias,
        "vp_bias_label": _dir_label(vp_bias),
        "vp_dir": vp_dir,
        "vp_dir_label": _dir_label(vp_dir),
        "liq_dir": liq_dir,
        "liq_dir_label": _dir_label(liq_dir),
        "density_dir": density_dir,
        "density_dir_label": _dir_label(density_dir),
        "imbalance_dir": imb_dir,
        "imbalance_dir_label": _dir_label(imb_dir),

        "mtf_avg": mtf_avg,
        "delta_norm": delta_norm,
        "score": ic_score,
        "strength": ic_strength,
        "strength_label": _strength_label(ic_strength),
        "quality": quality,

        "macro_ready": macro_ready,
        "in_macro": in_macro,
        "core_ready": core_ready,
        "in_core": in_core,
        "density_event": density_event,
        "imbalance_live": imbalance_live,
        "vp_inside_va": vp_inside_va,
        "vp_agrees_with_confluence": vp_agrees,
        "momentum_ok": momentum_ok,
        "trigger_ok": trigger_ok,
        "ready": ready,
        "valid": valid,
        "active": active,
        "ttl": ic_ttl,

        "macro_zone_lo": _safe_float(row.get("sc_confl_macro_lo")),
        "macro_zone_hi": _safe_float(row.get("sc_confl_macro_hi")),
        "core_zone_lo": _safe_float(row.get("sc_confl_core_lo")),
        "core_zone_hi": _safe_float(row.get("sc_confl_core_hi")),

        "vp_val": _safe_float(row.get("sc_confl_vp_val")),
        "vp_poc": _safe_float(row.get("sc_confl_vp_poc")),
        "vp_vah": _safe_float(row.get("sc_confl_vp_vah")),
        "vp_dist_to_poc_pct": _safe_float(row.get("sc_confl_vp_dist_to_poc_pct")),

        "close": _safe_float(row.get("close")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),

        # ---------------------------------------------------------------------
        # KEEP NESTED BACKEND STRUCTURE TOO
        # ---------------------------------------------------------------------
        "state_detail": {
            "direction": direction,
            "direction_label": _dir_label(direction),
            "mtf_dir": mtf_dir,
            "mtf_dir_label": _dir_label(mtf_dir),
            "macro_dir": macro_dir,
            "macro_dir_label": _dir_label(macro_dir),
            "vp_bias": vp_bias,
            "vp_bias_label": _dir_label(vp_bias),
            "vp_dir": vp_dir,
            "vp_dir_label": _dir_label(vp_dir),
            "liq_dir": liq_dir,
            "liq_dir_label": _dir_label(liq_dir),
            "density_dir": density_dir,
            "density_dir_label": _dir_label(density_dir),
            "imbalance_dir": imb_dir,
            "imbalance_dir_label": _dir_label(imb_dir),
        },

        "scores": {
            "mtf_avg": mtf_avg,
            "delta_norm": delta_norm,
            "score": ic_score,
            "strength": ic_strength,
            "strength_label": _strength_label(ic_strength),
            "quality": quality,
        },

        "status": {
            "macro_ready": macro_ready,
            "in_macro": in_macro,
            "core_ready": core_ready,
            "in_core": in_core,
            "density_event": density_event,
            "imbalance_live": imbalance_live,
            "vp_inside_va": vp_inside_va,
            "vp_agrees_with_confluence": vp_agrees,
            "momentum_ok": momentum_ok,
            "trigger_ok": trigger_ok,
            "ready": ready,
            "valid": valid,
            "active": active,
            "ttl": ic_ttl,
        },

        "zones": {
            "macro_zone_lo": _safe_float(row.get("sc_confl_macro_lo")),
            "macro_zone_hi": _safe_float(row.get("sc_confl_macro_hi")),
            "core_zone_lo": _safe_float(row.get("sc_confl_core_lo")),
            "core_zone_hi": _safe_float(row.get("sc_confl_core_hi")),
        },

        "volume_profile": {
            "val": _safe_float(row.get("sc_confl_vp_val")),
            "poc": _safe_float(row.get("sc_confl_vp_poc")),
            "vah": _safe_float(row.get("sc_confl_vp_vah")),
            "dist_to_poc_pct": _safe_float(row.get("sc_confl_vp_dist_to_poc_pct")),
        },

        "price": {
            "close": _safe_float(row.get("close")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
        },
    }

    return payload