from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_FIB_CONFIG: Dict[str, Any] = {
    # Main
    "fib_on": True,
    "lb": 3,
    "rb": 3,

    # Previous-day institutional anchors
    "anchor_tz": "Europe/London",

    # MTF layer
    "fib_mtf_on": True,
    "tf1": "5",
    "tf2": "15",
    "tf3": "30",
    "tf4": "60",
    "tf5": "240",
    "tf6": "D",

    "fib_w1": 1.0,
    "fib_w2": 1.0,
    "fib_w3": 1.0,
    "fib_w4": 1.0,
    "fib_w5": 1.0,
    "fib_w6": 1.0,
}


# =============================================================================
# HELPERS
# =============================================================================

def _to_float_series(series: pd.Series, index: pd.Index) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=index)
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


def _require_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex")


def _ensure_tz_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC")
    return index


def _tf_to_pandas_freq(tf: str) -> str:
    tf = str(tf).strip().upper()
    if tf == "D":
        return "1D"
    if tf == "W":
        return "1W"
    if tf == "M":
        return "1ME"
    if tf.isdigit():
        return f"{int(tf)}min"
    raise ValueError(f"Unsupported timeframe: {tf}")


def _build_day_key(index: pd.DatetimeIndex, tz_name: str) -> pd.Series:
    idx = _ensure_tz_index(index).tz_convert(tz_name)
    return pd.Series(idx.strftime("%Y-%m-%d"), index=index, dtype=object)


def _prev_day_anchor_series(
    df: pd.DataFrame,
    tz_name: str,
) -> tuple[pd.Series, pd.Series]:
    day_key = _build_day_key(df.index, tz_name)

    daily = (
        pd.DataFrame(
            {
                "day_key": day_key.values,
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
            },
            index=df.index,
        )
        .groupby("day_key", sort=True)
        .agg({"high": "max", "low": "min"})
    )

    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)

    prev_day_high = pd.to_numeric(day_key.map(daily["prev_high"]), errors="coerce").astype(float)
    prev_day_low = pd.to_numeric(day_key.map(daily["prev_low"]), errors="coerce").astype(float)

    return prev_day_high, prev_day_low


def _resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    freq = _tf_to_pandas_freq(tf)
    src = df.copy()
    src.index = _ensure_tz_index(src.index)

    ohlc = (
        src[["open", "high", "low", "close"]]
        .resample(freq, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    return ohlc


def _pivot_high_confirmed(series: pd.Series, left: int, right: int) -> pd.Series:
    """
    Pine mirror for ta.pivothigh(high, left, right):
    return pivot price on the confirmation bar, not the pivot center bar.
    """
    left = max(1, int(left))
    right = max(1, int(right))

    n = len(series)
    out = pd.Series(np.nan, index=series.index, dtype=float)

    values = pd.to_numeric(series, errors="coerce").astype(float).values

    for center in range(left, n - right):
        window = values[center - left : center + right + 1]
        center_val = values[center]
        if np.isnan(center_val):
            continue
        if center_val == np.nanmax(window) and np.sum(window == center_val) == 1:
            confirm_bar = center + right
            if confirm_bar < n:
                out.iloc[confirm_bar] = float(center_val)

    return out


def _pivot_low_confirmed(series: pd.Series, left: int, right: int) -> pd.Series:
    """
    Pine mirror for ta.pivotlow(low, left, right):
    return pivot price on the confirmation bar, not the pivot center bar.
    """
    left = max(1, int(left))
    right = max(1, int(right))

    n = len(series)
    out = pd.Series(np.nan, index=series.index, dtype=float)

    values = pd.to_numeric(series, errors="coerce").astype(float).values

    for center in range(left, n - right):
        window = values[center - left : center + right + 1]
        center_val = values[center]
        if np.isnan(center_val):
            continue
        if center_val == np.nanmin(window) and np.sum(window == center_val) == 1:
            confirm_bar = center + right
            if confirm_bar < n:
                out.iloc[confirm_bar] = float(center_val)

    return out


def _last_confirmed_pivot_value_and_bar(
    pivot_confirmed: pd.Series,
    right: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Mirrors:
      ta.valuewhen(not na(ta.pivothigh(...)), ta.pivothigh(...), 0)
      ta.valuewhen(not na(ta.pivothigh(...)), bar_index - right, 0)
    """
    idx = pivot_confirmed.index
    n = len(idx)

    pivot_value = pivot_confirmed.ffill().astype(float)

    pivot_bar_source = pd.Series(
        np.where(pivot_confirmed.notna(), np.arange(n) - int(right), np.nan),
        index=idx,
        dtype=float,
    )
    pivot_bar = pivot_bar_source.ffill().astype(float)

    return pivot_value, pivot_bar


def _align_htf_series_to_base(
    htf_series: pd.Series,
    base_index: pd.DatetimeIndex,
) -> pd.Series:
    aligned = htf_series.reindex(base_index, method="ffill")
    return pd.to_numeric(aligned, errors="coerce").astype(float)


def _score_tf(
    df: pd.DataFrame,
    tf: str,
    prev_day_low_base: pd.Series,
    prev_day_high_base: pd.Series,
    lb: int,
    rb: int,
) -> pd.Series:
    """
    Pine mirror of the MTF scoring function.
    """
    htf = _resample_ohlc(df, tf)

    htf_close = pd.to_numeric(htf["close"], errors="coerce").astype(float)

    htf_ph_confirmed = _pivot_high_confirmed(htf["high"], lb, rb)
    htf_pl_confirmed = _pivot_low_confirmed(htf["low"], lb, rb)

    htf_last_ph, _ = _last_confirmed_pivot_value_and_bar(htf_ph_confirmed, rb)
    htf_last_pl, _ = _last_confirmed_pivot_value_and_bar(htf_pl_confirmed, rb)

    c = _align_htf_series_to_base(htf_close, df.index)
    ph = _align_htf_series_to_base(htf_last_ph, df.index)
    pl = _align_htf_series_to_base(htf_last_pl, df.index)

    bull_ok = ph.notna() & prev_day_low_base.notna() & (ph > prev_day_low_base)
    bear_ok = pl.notna() & prev_day_high_base.notna() & (pl < prev_day_high_base)

    bull_mid = pd.Series(
        np.where(bull_ok, prev_day_low_base + ((ph - prev_day_low_base) * 0.50), np.nan),
        index=df.index,
        dtype=float,
    )
    bear_mid = pd.Series(
        np.where(bear_ok, prev_day_high_base - ((prev_day_high_base - pl) * 0.50), np.nan),
        index=df.index,
        dtype=float,
    )

    bull_score = pd.Series(
        np.where(bull_ok & (c > bull_mid), 1.0, 0.0),
        index=df.index,
        dtype=float,
    )
    bear_score = pd.Series(
        np.where(bear_ok & (c < bear_mid), -1.0, 0.0),
        index=df.index,
        dtype=float,
    )

    return (bull_score + bear_score).astype(float)


def _zone_text(zone: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            zone == 1,
            "25–33",
            np.where(zone == 2, "50–61.5", np.where(zone == 3, "66–78", "NONE")),
        ),
        index=zone.index,
        dtype=object,
    )


def _dir_text(direction: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(direction > 0, "BULL", np.where(direction < 0, "BEAR", "NEUTRAL")),
        index=direction.index,
        dtype=object,
    )


def _mtf_state_text(avg: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            avg > 0.60,
            "STRONG BULL",
            np.where(
                avg > 0.20,
                "BULL",
                np.where(avg < -0.60, "STRONG BEAR", np.where(avg < -0.20, "BEAR", "MIXED")),
            ),
        ),
        index=avg.index,
        dtype=object,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_str(value: Any, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


# =============================================================================
# MAIN ENGINE
# =============================================================================

def compute_fib_engine(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_FIB_CONFIG, **(config or {})}

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    _require_datetime_index(df)

    out = df.copy()
    out.index = _ensure_tz_index(out.index)

    idx = out.index

    high = _to_float_series(out["high"], idx)
    low = _to_float_series(out["low"], idx)
    close = _to_float_series(out["close"], idx)

    lb = max(1, int(cfg["lb"]))
    rb = max(1, int(cfg["rb"]))
    fib_on = bool(cfg["fib_on"])

    # =========================================================================
    # PREVIOUS-DAY INSTITUTIONAL ANCHORS
    # =========================================================================

    prev_day_high, prev_day_low = _prev_day_anchor_series(out, str(cfg["anchor_tz"]))

    # =========================================================================
    # CONFIRMED PIVOTS
    # =========================================================================

    bull_ph_confirmed = _pivot_high_confirmed(high, lb, rb)
    bear_pl_confirmed = _pivot_low_confirmed(low, lb, rb)

    last_bull_pivot_high, last_bull_pivot_high_bar = _last_confirmed_pivot_value_and_bar(
        bull_ph_confirmed, rb
    )
    last_bear_pivot_low, last_bear_pivot_low_bar = _last_confirmed_pivot_value_and_bar(
        bear_pl_confirmed, rb
    )

    # =========================================================================
    # ANCHOR VALIDATION
    # =========================================================================

    bull_ready = fib_on & prev_day_low.notna() & last_bull_pivot_high.notna() & (last_bull_pivot_high > prev_day_low)
    bear_ready = fib_on & prev_day_high.notna() & last_bear_pivot_low.notna() & (last_bear_pivot_low < prev_day_high)

    bull_range = pd.Series(
        np.where(bull_ready, last_bull_pivot_high - prev_day_low, np.nan),
        index=idx,
        dtype=float,
    )
    bear_range = pd.Series(
        np.where(bear_ready, prev_day_high - last_bear_pivot_low, np.nan),
        index=idx,
        dtype=float,
    )

    bull_is_latest = bull_ready & ((~bear_ready) | (last_bull_pivot_high_bar >= last_bear_pivot_low_bar))
    bear_is_latest = bear_ready & ((~bull_ready) | (last_bear_pivot_low_bar > last_bull_pivot_high_bar))

    fib_dir = pd.Series(
        np.where(bull_is_latest, 1, np.where(bear_is_latest, -1, 0)),
        index=idx,
        dtype=int,
    )

    # =========================================================================
    # ACTIVE FIB LEVELS
    # =========================================================================

    fib0 = pd.Series(np.nan, index=idx, dtype=float)
    fib25 = pd.Series(np.nan, index=idx, dtype=float)
    fib33 = pd.Series(np.nan, index=idx, dtype=float)
    fib50 = pd.Series(np.nan, index=idx, dtype=float)
    fib615 = pd.Series(np.nan, index=idx, dtype=float)
    fib66 = pd.Series(np.nan, index=idx, dtype=float)
    fib78 = pd.Series(np.nan, index=idx, dtype=float)
    fib1 = pd.Series(np.nan, index=idx, dtype=float)

    bull_mask = (fib_dir == 1) & bull_ready
    bear_mask = (fib_dir == -1) & bear_ready

    fib0.loc[bull_mask] = prev_day_low.loc[bull_mask]
    fib25.loc[bull_mask] = prev_day_low.loc[bull_mask] + (bull_range.loc[bull_mask] * 0.25)
    fib33.loc[bull_mask] = prev_day_low.loc[bull_mask] + (bull_range.loc[bull_mask] * 0.33)
    fib50.loc[bull_mask] = prev_day_low.loc[bull_mask] + (bull_range.loc[bull_mask] * 0.50)
    fib615.loc[bull_mask] = prev_day_low.loc[bull_mask] + (bull_range.loc[bull_mask] * 0.615)
    fib66.loc[bull_mask] = prev_day_low.loc[bull_mask] + (bull_range.loc[bull_mask] * 0.66)
    fib78.loc[bull_mask] = prev_day_low.loc[bull_mask] + (bull_range.loc[bull_mask] * 0.78)
    fib1.loc[bull_mask] = last_bull_pivot_high.loc[bull_mask]

    fib0.loc[bear_mask] = prev_day_high.loc[bear_mask]
    fib25.loc[bear_mask] = prev_day_high.loc[bear_mask] - (bear_range.loc[bear_mask] * 0.25)
    fib33.loc[bear_mask] = prev_day_high.loc[bear_mask] - (bear_range.loc[bear_mask] * 0.33)
    fib50.loc[bear_mask] = prev_day_high.loc[bear_mask] - (bear_range.loc[bear_mask] * 0.50)
    fib615.loc[bear_mask] = prev_day_high.loc[bear_mask] - (bear_range.loc[bear_mask] * 0.615)
    fib66.loc[bear_mask] = prev_day_high.loc[bear_mask] - (bear_range.loc[bear_mask] * 0.66)
    fib78.loc[bear_mask] = prev_day_high.loc[bear_mask] - (bear_range.loc[bear_mask] * 0.78)
    fib1.loc[bear_mask] = last_bear_pivot_low.loc[bear_mask]

    # =========================================================================
    # ACTIVE ZONE DETECTION
    # =========================================================================

    active_zone = pd.Series(0, index=idx, dtype=int)

    has_levels = (
        (fib_dir != 0)
        & fib25.notna()
        & fib33.notna()
        & fib50.notna()
        & fib615.notna()
        & fib66.notna()
        & fib78.notna()
    )

    in_z1 = has_levels & (close >= np.minimum(fib25, fib33)) & (close <= np.maximum(fib25, fib33))
    in_z2 = has_levels & (close >= np.minimum(fib50, fib615)) & (close <= np.maximum(fib50, fib615))
    in_z3 = has_levels & (close >= np.minimum(fib66, fib78)) & (close <= np.maximum(fib66, fib78))

    active_zone.loc[in_z1] = 1
    active_zone.loc[~in_z1 & in_z2] = 2
    active_zone.loc[~in_z1 & ~in_z2 & in_z3] = 3

    # =========================================================================
    # MTF FIB AVERAGE + AGREEMENT / CONFLICT / CONTRACT
    # =========================================================================

    fib_mtf_on = bool(cfg["fib_mtf_on"])

    if fib_mtf_on:
        fib_score_1 = _score_tf(out, str(cfg["tf1"]), prev_day_low, prev_day_high, lb, rb)
        fib_score_2 = _score_tf(out, str(cfg["tf2"]), prev_day_low, prev_day_high, lb, rb)
        fib_score_3 = _score_tf(out, str(cfg["tf3"]), prev_day_low, prev_day_high, lb, rb)
        fib_score_4 = _score_tf(out, str(cfg["tf4"]), prev_day_low, prev_day_high, lb, rb)
        fib_score_5 = _score_tf(out, str(cfg["tf5"]), prev_day_low, prev_day_high, lb, rb)
        fib_score_6 = _score_tf(out, str(cfg["tf6"]), prev_day_low, prev_day_high, lb, rb)
    else:
        fib_score_1 = pd.Series(0.0, index=idx, dtype=float)
        fib_score_2 = pd.Series(0.0, index=idx, dtype=float)
        fib_score_3 = pd.Series(0.0, index=idx, dtype=float)
        fib_score_4 = pd.Series(0.0, index=idx, dtype=float)
        fib_score_5 = pd.Series(0.0, index=idx, dtype=float)
        fib_score_6 = pd.Series(0.0, index=idx, dtype=float)

    fib_w1 = float(cfg["fib_w1"])
    fib_w2 = float(cfg["fib_w2"])
    fib_w3 = float(cfg["fib_w3"])
    fib_w4 = float(cfg["fib_w4"])
    fib_w5 = float(cfg["fib_w5"])
    fib_w6 = float(cfg["fib_w6"])

    fib_wsum_raw = fib_w1 + fib_w2 + fib_w3 + fib_w4 + fib_w5 + fib_w6
    fib_wsum = 1.0 if fib_wsum_raw <= 0 else fib_wsum_raw

    fib_mtf_avg = (
        (
            (fib_score_1 * fib_w1)
            + (fib_score_2 * fib_w2)
            + (fib_score_3 * fib_w3)
            + (fib_score_4 * fib_w4)
            + (fib_score_5 * fib_w5)
            + (fib_score_6 * fib_w6)
        ) / fib_wsum
        if fib_mtf_on
        else pd.Series(0.0, index=idx, dtype=float)
    ).astype(float)

    bull_count = (
        (fib_score_1 > 0).astype(int)
        + (fib_score_2 > 0).astype(int)
        + (fib_score_3 > 0).astype(int)
        + (fib_score_4 > 0).astype(int)
        + (fib_score_5 > 0).astype(int)
        + (fib_score_6 > 0).astype(int)
    )

    bear_count = (
        (fib_score_1 < 0).astype(int)
        + (fib_score_2 < 0).astype(int)
        + (fib_score_3 < 0).astype(int)
        + (fib_score_4 < 0).astype(int)
        + (fib_score_5 < 0).astype(int)
        + (fib_score_6 < 0).astype(int)
    )

    active_tf_count = (
        (fib_score_1 != 0).astype(int)
        + (fib_score_2 != 0).astype(int)
        + (fib_score_3 != 0).astype(int)
        + (fib_score_4 != 0).astype(int)
        + (fib_score_5 != 0).astype(int)
        + (fib_score_6 != 0).astype(int)
    )

    mtf_bull_agreement = pd.Series(
        np.where(active_tf_count > 0, bull_count / active_tf_count, 0.0),
        index=idx,
        dtype=float,
    )

    mtf_bear_agreement = pd.Series(
        np.where(active_tf_count > 0, bear_count / active_tf_count, 0.0),
        index=idx,
        dtype=float,
    )

    mtf_agreement = pd.Series(
        np.where(fib_mtf_avg >= 0, mtf_bull_agreement, mtf_bear_agreement),
        index=idx,
        dtype=float,
    )

    mtf_conflict = pd.Series(
        np.where(fib_mtf_avg >= 0, mtf_bear_agreement, mtf_bull_agreement),
        index=idx,
        dtype=float,
    )

    mtf_contract_state = pd.Series(
        np.where(
            (fib_mtf_avg > 0.20) & (mtf_agreement >= 0.50),
            1,
            np.where((fib_mtf_avg < -0.20) & (mtf_agreement >= 0.50), -1, 0),
        ),
        index=idx,
        dtype=int,
    )

    # =========================================================================
    # TEXT FIELDS
    # =========================================================================

    fib_dir_text = _dir_text(fib_dir)
    active_zone_text = _zone_text(active_zone)
    fib_mtf_state_text = _mtf_state_text(fib_mtf_avg)

    # =========================================================================
    # EXPORT FIELDS — PINE PARITY
    # =========================================================================

    out["sc_fib_dir"] = fib_dir.astype(int)
    out["sc_fib_dir_text"] = fib_dir_text

    out["sc_fib_active_zone"] = active_zone.astype(int)
    out["sc_fib_active_zone_text"] = active_zone_text

    out["sc_fib_prev_day_high"] = prev_day_high.astype(float)
    out["sc_fib_prev_day_low"] = prev_day_low.astype(float)

    out["sc_fib_anchor_bull_high"] = last_bull_pivot_high.astype(float)
    out["sc_fib_anchor_bear_low"] = last_bear_pivot_low.astype(float)
    out["sc_fib_anchor_bull_bar"] = last_bull_pivot_high_bar.astype(float)
    out["sc_fib_anchor_bear_bar"] = last_bear_pivot_low_bar.astype(float)

    out["sc_fib_bull_ready"] = bull_ready.astype(float)
    out["sc_fib_bear_ready"] = bear_ready.astype(float)
    out["sc_fib_bull_latest"] = bull_is_latest.astype(float)
    out["sc_fib_bear_latest"] = bear_is_latest.astype(float)

    out["sc_fib_0"] = fib0.astype(float)
    out["sc_fib_25"] = fib25.astype(float)
    out["sc_fib_33"] = fib33.astype(float)
    out["sc_fib_50"] = fib50.astype(float)
    out["sc_fib_615"] = fib615.astype(float)
    out["sc_fib_66"] = fib66.astype(float)
    out["sc_fib_78"] = fib78.astype(float)
    out["sc_fib_1"] = fib1.astype(float)

    out["sc_fib_mtf_avg"] = fib_mtf_avg.astype(float)
    out["sc_fib_mtf_agreement"] = mtf_agreement.astype(float)
    out["sc_fib_mtf_conflict"] = mtf_conflict.astype(float)
    out["sc_fib_contract_state"] = mtf_contract_state.astype(int)
    out["sc_fib_mtf_state_text"] = fib_mtf_state_text

    out["sc_fib_score_1"] = fib_score_1.astype(float)
    out["sc_fib_score_2"] = fib_score_2.astype(float)
    out["sc_fib_score_3"] = fib_score_3.astype(float)
    out["sc_fib_score_4"] = fib_score_4.astype(float)
    out["sc_fib_score_5"] = fib_score_5.astype(float)
    out["sc_fib_score_6"] = fib_score_6.astype(float)

    # =========================================================================
    # TRUTH / COMPATIBILITY FIELDS
    # =========================================================================

    out["fib_dir"] = out["sc_fib_dir"].fillna(0).astype(int)
    out["fib_zone"] = out["sc_fib_active_zone"].fillna(0).astype(int)
    out["fib_zone_label"] = out["sc_fib_active_zone_text"].fillna("NONE")

    return out


def build_fib(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return compute_fib_engine(df, config=config)


def run_fib_engine(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return compute_fib_engine(df, config=config)


# =============================================================================
# PAYLOAD BUILDER — WEBSITE SINGLE SOURCE OF TRUTH
# =============================================================================

def build_fib_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = compute_fib_engine(df, config=config)

    if result.empty:
        raise ValueError("Fib payload build failed: computed dataframe is empty")

    last = result.iloc[-1]

    timestamp = result.index[-1]
    if isinstance(timestamp, pd.Timestamp):
        timestamp_str = timestamp.isoformat()
    else:
        timestamp_str = str(timestamp)

    payload: Dict[str, Any] = {
        "debug_version": "fib_payload_v1",
        "module": "fib",
        "name": "SmartChart Fib Engine",
        "timestamp": timestamp_str,

        "state": {
            "direction": _safe_int(last.get("sc_fib_dir"), 0),
            "direction_text": _safe_str(last.get("sc_fib_dir_text"), "NEUTRAL"),
            "active_zone": _safe_int(last.get("sc_fib_active_zone"), 0),
            "active_zone_text": _safe_str(last.get("sc_fib_active_zone_text"), "NONE"),
            "contract_state": _safe_int(last.get("sc_fib_contract_state"), 0),
            "contract_text": (
                "BULL"
                if _safe_int(last.get("sc_fib_contract_state"), 0) == 1
                else "BEAR"
                if _safe_int(last.get("sc_fib_contract_state"), 0) == -1
                else "MIXED"
            ),
            "mtf_state_text": _safe_str(last.get("sc_fib_mtf_state_text"), "MIXED"),
        },

        "anchors": {
            "prev_day_high": _safe_float(last.get("sc_fib_prev_day_high")),
            "prev_day_low": _safe_float(last.get("sc_fib_prev_day_low")),
            "bull_anchor_high": _safe_float(last.get("sc_fib_anchor_bull_high")),
            "bear_anchor_low": _safe_float(last.get("sc_fib_anchor_bear_low")),
            "bull_anchor_bar": _safe_float(last.get("sc_fib_anchor_bull_bar")),
            "bear_anchor_bar": _safe_float(last.get("sc_fib_anchor_bear_bar")),
            "bull_ready": _safe_float(last.get("sc_fib_bull_ready")),
            "bear_ready": _safe_float(last.get("sc_fib_bear_ready")),
            "bull_latest": _safe_float(last.get("sc_fib_bull_latest")),
            "bear_latest": _safe_float(last.get("sc_fib_bear_latest")),
        },

        "levels": {
            "fib_0": _safe_float(last.get("sc_fib_0")),
            "fib_25": _safe_float(last.get("sc_fib_25")),
            "fib_33": _safe_float(last.get("sc_fib_33")),
            "fib_50": _safe_float(last.get("sc_fib_50")),
            "fib_615": _safe_float(last.get("sc_fib_615")),
            "fib_66": _safe_float(last.get("sc_fib_66")),
            "fib_78": _safe_float(last.get("sc_fib_78")),
            "fib_1": _safe_float(last.get("sc_fib_1")),
        },

        "mtf": {
            "avg": _safe_float(last.get("sc_fib_mtf_avg")),
            "agreement": _safe_float(last.get("sc_fib_mtf_agreement")),
            "conflict": _safe_float(last.get("sc_fib_mtf_conflict")),
            "score_1": _safe_float(last.get("sc_fib_score_1")),
            "score_2": _safe_float(last.get("sc_fib_score_2")),
            "score_3": _safe_float(last.get("sc_fib_score_3")),
            "score_4": _safe_float(last.get("sc_fib_score_4")),
            "score_5": _safe_float(last.get("sc_fib_score_5")),
            "score_6": _safe_float(last.get("sc_fib_score_6")),
        },

        "truth": {
            "fib_dir": _safe_int(last.get("fib_dir"), 0),
            "fib_zone": _safe_int(last.get("fib_zone"), 0),
            "fib_zone_label": _safe_str(last.get("fib_zone_label"), "NONE"),
        },
    }

    return payload