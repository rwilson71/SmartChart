from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class PullbackRetestConfig:
    # EMA groups
    ema_14_len: int = 14
    ema_20_len: int = 20
    ema_33_len: int = 33
    ema_50_len: int = 50
    ema_100_len: int = 100
    ema_200_len: int = 200

    # Retest tolerance
    price_tolerance_pct: float = 0.0015   # 0.15%
    use_atr_tolerance: bool = True
    atr_len: int = 14
    atr_mult: float = 0.20

    # Session settings
    session_start_hour: int = 0
    session_start_minute: int = 0

    # Trigger memory
    ttl_bars: int = 8

    # Fib settings
    fib_pivot_len: int = 5

    # Trigger candle scoring
    volume_ma_len: int = 20
    displacement_atr_mult: float = 1.20
    absorption_wick_ratio_min: float = 0.35
    strong_close_pos_bull: float = 0.65
    strong_close_pos_bear: float = 0.35


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return pd.to_numeric(series, errors="coerce").ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return pd.to_numeric(series, errors="coerce").rolling(length, min_periods=1).mean()


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _sma(tr, length)


def _bars_since(condition: pd.Series) -> pd.Series:
    out = np.full(len(condition), np.nan, dtype=float)
    last_true = -1
    vals = condition.fillna(False).astype(bool).to_numpy()

    for i, flag in enumerate(vals):
        if flag:
            last_true = i
            out[i] = 0.0
        elif last_true >= 0:
            out[i] = float(i - last_true)

    return pd.Series(out, index=condition.index)


def _within_tolerance(
    price_low: pd.Series,
    price_high: pd.Series,
    level: pd.Series,
    tol: pd.Series,
) -> pd.Series:
    return level.notna() & (price_low <= (level + tol)) & (price_high >= (level - tol))


def _zone_touch(
    price_low: pd.Series,
    price_high: pd.Series,
    zone_lo: pd.Series,
    zone_hi: pd.Series,
    tol: pd.Series,
) -> pd.Series:
    return (
        zone_lo.notna()
        & zone_hi.notna()
        & (price_low <= (zone_hi + tol))
        & (price_high >= (zone_lo - tol))
    )


def _rolling_pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    vals = high.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)

    for i in range(left, n - right):
        center = vals[i]
        if np.isnan(center):
            continue
        window = vals[i - left : i + right + 1]
        if np.nanargmax(window) == left and np.nanmax(window) == center:
            out[i] = center

    return pd.Series(out, index=high.index)


def _rolling_pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    vals = low.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)

    for i in range(left, n - right):
        center = vals[i]
        if np.isnan(center):
            continue
        window = vals[i - left : i + right + 1]
        if np.nanargmin(window) == left and np.nanmin(window) == center:
            out[i] = center

    return pd.Series(out, index=low.index)


def _value_when_not_na(series: pd.Series) -> pd.Series:
    return series.ffill()


def _prev_daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    prev = daily.shift(1).rename(columns={"high": "prev_day_high", "low": "prev_day_low"})
    return prev.reindex(df.index, method="ffill")


def _first_bar_levels(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    tf = df.resample(rule).agg(agg).dropna()
    session_date = tf.index.normalize()

    first_mask = ~session_date.duplicated()
    first_bars = tf.loc[first_mask, ["high", "low"]].copy()
    first_bars.columns = [f"first_{rule}_high", f"first_{rule}_low"]

    first_bars.index = first_bars.index.normalize()
    return first_bars.reindex(df.index.normalize(), method="ffill").set_index(df.index)


def _session_high_low(df: pd.DataFrame) -> pd.DataFrame:
    day_key = df.index.normalize()
    session_high = df.groupby(day_key)["high"].cummax()
    session_low = df.groupby(day_key)["low"].cummin()
    return pd.DataFrame(
        {
            "session_high": session_high,
            "session_low": session_low,
        },
        index=df.index,
    )


def _fib_level(direction: pd.Series, anchor_a: pd.Series, fib_range: pd.Series, pct: float) -> pd.Series:
    bull = anchor_a + fib_range * pct
    bear = anchor_a - fib_range * pct
    return pd.Series(
        np.where(direction == 1, bull, np.where(direction == -1, bear, np.nan)),
        index=direction.index,
        dtype=float,
    )


def _ttl_flag(raw_touch: pd.Series, ttl_bars: int) -> pd.DataFrame:
    bars = _bars_since(raw_touch)
    active = bars.notna() & (bars >= 0) & (bars <= ttl_bars)
    ttl = pd.Series(
        np.where(active, np.maximum(0, ttl_bars - bars), 0),
        index=raw_touch.index,
        dtype=int,
    )
    return pd.DataFrame(
        {
            "touch_now": raw_touch.astype(int),
            "touch_active": active.astype(int),
            "touch_ttl": ttl,
        },
        index=raw_touch.index,
    )


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        numerator / denominator.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _body_fraction(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).abs()
    body = (df["close"] - df["open"]).abs()
    return _safe_div(body, rng)


def _close_position(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).abs()
    return _safe_div(df["close"] - df["low"], rng).clip(0.0, 1.0)


def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)).clip(lower=0.0)


def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return (pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]).clip(lower=0.0)


def _to_native(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _get_reindexed_series(
    source_df: Optional[pd.DataFrame],
    index: pd.Index,
    candidates: list[str],
) -> pd.Series:
    if source_df is None:
        return pd.Series(np.nan, index=index, dtype=float)

    for col in candidates:
        if col in source_df.columns:
            return pd.to_numeric(source_df[col].reindex(index), errors="coerce")
    return pd.Series(np.nan, index=index, dtype=float)


# =============================================================================
# CORE
# =============================================================================

def calculate_pullback_retest(
    df: pd.DataFrame,
    config: Optional[PullbackRetestConfig] = None,
    confluence_df: Optional[pd.DataFrame] = None,
    trend_df: Optional[pd.DataFrame] = None,
    fvg_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    SmartChart Pullback / Retest Engine

    Optional inputs:
    - confluence_df: may contain cloud zone and/or confluence zone fields
    - trend_df: may contain trend direction export if available
    - fvg_df: may contain active FVG zone fields

    If trend_df is not provided, direction is inferred from EMA stack.
    """
    cfg = config or PullbackRetestConfig()
    _validate_ohlcv(df)

    base = df.copy().sort_index()
    out = pd.DataFrame(index=base.index)

    # -------------------------------------------------------------------------
    # CORE LEVELS
    # -------------------------------------------------------------------------
    ema14 = _ema(base["close"], cfg.ema_14_len)
    ema20 = _ema(base["close"], cfg.ema_20_len)
    ema33 = _ema(base["close"], cfg.ema_33_len)
    ema50 = _ema(base["close"], cfg.ema_50_len)
    ema100 = _ema(base["close"], cfg.ema_100_len)
    ema200 = _ema(base["close"], cfg.ema_200_len)
    atr_now = _atr(base, cfg.atr_len)

    tol_pct = base["close"].abs() * cfg.price_tolerance_pct
    tol_atr = atr_now * cfg.atr_mult
    tol = np.maximum(tol_pct, tol_atr) if cfg.use_atr_tolerance else tol_pct
    tol = pd.Series(tol, index=base.index, dtype=float)

    out["ema14"] = ema14
    out["ema20"] = ema20
    out["ema33"] = ema33
    out["ema50"] = ema50
    out["ema100"] = ema100
    out["ema200"] = ema200
    out["atr_now"] = atr_now
    out["retest_tolerance"] = tol

    # -------------------------------------------------------------------------
    # DIRECTION
    # -------------------------------------------------------------------------
    if trend_df is not None and "trend_dir" in trend_df.columns:
        trend_dir = trend_df["trend_dir"].reindex(base.index).fillna(0).astype(int)
    elif trend_df is not None and "trend_dir_export" in trend_df.columns:
        trend_dir = trend_df["trend_dir_export"].reindex(base.index).fillna(0).astype(int)
    else:
        trend_dir = pd.Series(
            np.where(
                (ema14 > ema20) & (ema20 > ema33) & (ema33 > ema50),
                1,
                np.where(
                    (ema14 < ema20) & (ema20 < ema33) & (ema33 < ema50),
                    -1,
                    0,
                ),
            ),
            index=base.index,
            dtype=int,
        )

    out["pb_trend_dir"] = trend_dir

    # -------------------------------------------------------------------------
    # EMA RETESTS
    # -------------------------------------------------------------------------
    ema1420_mid = (ema14 + ema20) / 2.0
    ema3350_mid = (ema33 + ema50) / 2.0
    ema100200_mid = (ema100 + ema200) / 2.0

    rt_ema1420_now = _within_tolerance(base["low"], base["high"], ema1420_mid, tol)
    rt_ema3350_now = _within_tolerance(base["low"], base["high"], ema3350_mid, tol)
    rt_ema100200_now = _within_tolerance(base["low"], base["high"], ema100200_mid, tol)

    ema1420_ttl = _ttl_flag(rt_ema1420_now, cfg.ttl_bars)
    ema3350_ttl = _ttl_flag(rt_ema3350_now, cfg.ttl_bars)
    ema100200_ttl = _ttl_flag(rt_ema100200_now, cfg.ttl_bars)

    out["rt_ema1420_now"] = ema1420_ttl["touch_now"]
    out["rt_ema1420_active"] = ema1420_ttl["touch_active"]
    out["rt_ema1420_ttl"] = ema1420_ttl["touch_ttl"]

    out["rt_ema3350_now"] = ema3350_ttl["touch_now"]
    out["rt_ema3350_active"] = ema3350_ttl["touch_active"]
    out["rt_ema3350_ttl"] = ema3350_ttl["touch_ttl"]

    out["rt_ema100200_now"] = ema100200_ttl["touch_now"]
    out["rt_ema100200_active"] = ema100200_ttl["touch_active"]
    out["rt_ema100200_ttl"] = ema100200_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # SESSION H/L RETESTS
    # -------------------------------------------------------------------------
    sess = _session_high_low(base)
    session_high = sess["session_high"]
    session_low = sess["session_low"]

    rt_session_high_now = _within_tolerance(base["low"], base["high"], session_high, tol)
    rt_session_low_now = _within_tolerance(base["low"], base["high"], session_low, tol)

    session_high_ttl = _ttl_flag(rt_session_high_now, cfg.ttl_bars)
    session_low_ttl = _ttl_flag(rt_session_low_now, cfg.ttl_bars)

    out["session_high"] = session_high
    out["session_low"] = session_low

    out["rt_session_high_now"] = session_high_ttl["touch_now"]
    out["rt_session_high_active"] = session_high_ttl["touch_active"]
    out["rt_session_high_ttl"] = session_high_ttl["touch_ttl"]

    out["rt_session_low_now"] = session_low_ttl["touch_now"]
    out["rt_session_low_active"] = session_low_ttl["touch_active"]
    out["rt_session_low_ttl"] = session_low_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # PREVIOUS DAY H/L RETESTS
    # -------------------------------------------------------------------------
    prev_day = _prev_daily_levels(base)
    prev_day_high = prev_day["prev_day_high"]
    prev_day_low = prev_day["prev_day_low"]

    rt_prev_day_high_now = _within_tolerance(base["low"], base["high"], prev_day_high, tol)
    rt_prev_day_low_now = _within_tolerance(base["low"], base["high"], prev_day_low, tol)

    prev_day_high_ttl = _ttl_flag(rt_prev_day_high_now, cfg.ttl_bars)
    prev_day_low_ttl = _ttl_flag(rt_prev_day_low_now, cfg.ttl_bars)

    out["prev_day_high"] = prev_day_high
    out["prev_day_low"] = prev_day_low

    out["rt_prev_day_high_now"] = prev_day_high_ttl["touch_now"]
    out["rt_prev_day_high_active"] = prev_day_high_ttl["touch_active"]
    out["rt_prev_day_high_ttl"] = prev_day_high_ttl["touch_ttl"]

    out["rt_prev_day_low_now"] = prev_day_low_ttl["touch_now"]
    out["rt_prev_day_low_active"] = prev_day_low_ttl["touch_active"]
    out["rt_prev_day_low_ttl"] = prev_day_low_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # FIRST 5M / 15M / 1H / 4H SESSION CANDLE H/L
    # -------------------------------------------------------------------------
    first_5m = _first_bar_levels(base, "5min")
    first_15m = _first_bar_levels(base, "15min")
    first_1h = _first_bar_levels(base, "1h")
    first_4h = _first_bar_levels(base, "4h")

    first_5m_high = first_5m["first_5min_high"]
    first_5m_low = first_5m["first_5min_low"]
    first_15m_high = first_15m["first_15min_high"]
    first_15m_low = first_15m["first_15min_low"]
    first_1h_high = first_1h["first_1h_high"]
    first_1h_low = first_1h["first_1h_low"]
    first_4h_high = first_4h["first_4h_high"]
    first_4h_low = first_4h["first_4h_low"]

    rt_first_5m_high_now = _within_tolerance(base["low"], base["high"], first_5m_high, tol)
    rt_first_5m_low_now = _within_tolerance(base["low"], base["high"], first_5m_low, tol)
    rt_first_15m_high_now = _within_tolerance(base["low"], base["high"], first_15m_high, tol)
    rt_first_15m_low_now = _within_tolerance(base["low"], base["high"], first_15m_low, tol)
    rt_first_1h_high_now = _within_tolerance(base["low"], base["high"], first_1h_high, tol)
    rt_first_1h_low_now = _within_tolerance(base["low"], base["high"], first_1h_low, tol)
    rt_first_4h_high_now = _within_tolerance(base["low"], base["high"], first_4h_high, tol)
    rt_first_4h_low_now = _within_tolerance(base["low"], base["high"], first_4h_low, tol)

    first_5m_high_ttl = _ttl_flag(rt_first_5m_high_now, cfg.ttl_bars)
    first_5m_low_ttl = _ttl_flag(rt_first_5m_low_now, cfg.ttl_bars)
    first_15m_high_ttl = _ttl_flag(rt_first_15m_high_now, cfg.ttl_bars)
    first_15m_low_ttl = _ttl_flag(rt_first_15m_low_now, cfg.ttl_bars)
    first_1h_high_ttl = _ttl_flag(rt_first_1h_high_now, cfg.ttl_bars)
    first_1h_low_ttl = _ttl_flag(rt_first_1h_low_now, cfg.ttl_bars)
    first_4h_high_ttl = _ttl_flag(rt_first_4h_high_now, cfg.ttl_bars)
    first_4h_low_ttl = _ttl_flag(rt_first_4h_low_now, cfg.ttl_bars)

    out["first_5m_high"] = first_5m_high
    out["first_5m_low"] = first_5m_low
    out["first_15m_high"] = first_15m_high
    out["first_15m_low"] = first_15m_low
    out["first_1h_high"] = first_1h_high
    out["first_1h_low"] = first_1h_low
    out["first_4h_high"] = first_4h_high
    out["first_4h_low"] = first_4h_low

    out["rt_first_5m_high_now"] = first_5m_high_ttl["touch_now"]
    out["rt_first_5m_high_active"] = first_5m_high_ttl["touch_active"]
    out["rt_first_5m_high_ttl"] = first_5m_high_ttl["touch_ttl"]

    out["rt_first_5m_low_now"] = first_5m_low_ttl["touch_now"]
    out["rt_first_5m_low_active"] = first_5m_low_ttl["touch_active"]
    out["rt_first_5m_low_ttl"] = first_5m_low_ttl["touch_ttl"]

    out["rt_first_15m_high_now"] = first_15m_high_ttl["touch_now"]
    out["rt_first_15m_high_active"] = first_15m_high_ttl["touch_active"]
    out["rt_first_15m_high_ttl"] = first_15m_high_ttl["touch_ttl"]

    out["rt_first_15m_low_now"] = first_15m_low_ttl["touch_now"]
    out["rt_first_15m_low_active"] = first_15m_low_ttl["touch_active"]
    out["rt_first_15m_low_ttl"] = first_15m_low_ttl["touch_ttl"]

    out["rt_first_1h_high_now"] = first_1h_high_ttl["touch_now"]
    out["rt_first_1h_high_active"] = first_1h_high_ttl["touch_active"]
    out["rt_first_1h_high_ttl"] = first_1h_high_ttl["touch_ttl"]

    out["rt_first_1h_low_now"] = first_1h_low_ttl["touch_now"]
    out["rt_first_1h_low_active"] = first_1h_low_ttl["touch_active"]
    out["rt_first_1h_low_ttl"] = first_1h_low_ttl["touch_ttl"]

    out["rt_first_4h_high_now"] = first_4h_high_ttl["touch_now"]
    out["rt_first_4h_high_active"] = first_4h_high_ttl["touch_active"]
    out["rt_first_4h_high_ttl"] = first_4h_high_ttl["touch_ttl"]

    out["rt_first_4h_low_now"] = first_4h_low_ttl["touch_now"]
    out["rt_first_4h_low_active"] = first_4h_low_ttl["touch_active"]
    out["rt_first_4h_low_ttl"] = first_4h_low_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # CONFLUENCE CLOUD ZONE RETEST
    # -------------------------------------------------------------------------
    cloud_lo = _get_reindexed_series(
        confluence_df,
        base.index,
        ["conf_zone_lo_export", "cloud_zone_lo", "cloud_lo", "loc_lo"],
    )
    cloud_hi = _get_reindexed_series(
        confluence_df,
        base.index,
        ["conf_zone_hi_export", "cloud_zone_hi", "cloud_hi", "loc_hi"],
    )

    cloud_mid = (cloud_lo + cloud_hi) / 2.0
    rt_cloud_now = _zone_touch(base["low"], base["high"], cloud_lo, cloud_hi, tol)
    cloud_ttl = _ttl_flag(rt_cloud_now, cfg.ttl_bars)

    out["conf_cloud_lo"] = cloud_lo
    out["conf_cloud_hi"] = cloud_hi
    out["conf_cloud_mid"] = cloud_mid
    out["rt_conf_cloud_now"] = cloud_ttl["touch_now"]
    out["rt_conf_cloud_active"] = cloud_ttl["touch_active"]
    out["rt_conf_cloud_ttl"] = cloud_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # CONFLUENCE ZONE RETEST
    # -------------------------------------------------------------------------
    conf_zone_lo = _get_reindexed_series(
        confluence_df,
        base.index,
        ["confluence_zone_lo_export", "confluence_zone_lo", "zone_lo", "conf_lo"],
    )
    conf_zone_hi = _get_reindexed_series(
        confluence_df,
        base.index,
        ["confluence_zone_hi_export", "confluence_zone_hi", "zone_hi", "conf_hi"],
    )

    conf_zone_mid = (conf_zone_lo + conf_zone_hi) / 2.0
    rt_conf_zone_now = _zone_touch(base["low"], base["high"], conf_zone_lo, conf_zone_hi, tol)
    conf_zone_ttl = _ttl_flag(rt_conf_zone_now, cfg.ttl_bars)

    out["confluence_zone_lo"] = conf_zone_lo
    out["confluence_zone_hi"] = conf_zone_hi
    out["confluence_zone_mid"] = conf_zone_mid
    out["rt_confluence_zone_now"] = conf_zone_ttl["touch_now"]
    out["rt_confluence_zone_active"] = conf_zone_ttl["touch_active"]
    out["rt_confluence_zone_ttl"] = conf_zone_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # FVG RETEST
    # -------------------------------------------------------------------------
    fvg_lo = _get_reindexed_series(
        fvg_df,
        base.index,
        ["fvg_lo_export", "fvg_zone_lo", "active_fvg_lo", "zone_lo", "fvg_lo"],
    )
    fvg_hi = _get_reindexed_series(
        fvg_df,
        base.index,
        ["fvg_hi_export", "fvg_zone_hi", "active_fvg_hi", "zone_hi", "fvg_hi"],
    )

    fvg_mid = (fvg_lo + fvg_hi) / 2.0
    rt_fvg_now = _zone_touch(base["low"], base["high"], fvg_lo, fvg_hi, tol)
    fvg_ttl = _ttl_flag(rt_fvg_now, cfg.ttl_bars)

    out["fvg_lo"] = fvg_lo
    out["fvg_hi"] = fvg_hi
    out["fvg_mid"] = fvg_mid
    out["rt_fvg_now"] = fvg_ttl["touch_now"]
    out["rt_fvg_active"] = fvg_ttl["touch_active"]
    out["rt_fvg_ttl"] = fvg_ttl["touch_ttl"]

    # -------------------------------------------------------------------------
    # FIB RETESTS
    # Institutional anchor model:
    # Bull: previous day low -> last pivot high
    # Bear: previous day high -> last pivot low
    # -------------------------------------------------------------------------
    prev_day_high_s = prev_day_high
    prev_day_low_s = prev_day_low

    ph = _rolling_pivot_high(base["high"], cfg.fib_pivot_len, cfg.fib_pivot_len)
    pl = _rolling_pivot_low(base["low"], cfg.fib_pivot_len, cfg.fib_pivot_len)

    last_pivot_high = _value_when_not_na(ph)
    last_pivot_low = _value_when_not_na(pl)

    bull_fib_ready = prev_day_low_s.notna() & last_pivot_high.notna() & (last_pivot_high > prev_day_low_s)
    bear_fib_ready = prev_day_high_s.notna() & last_pivot_low.notna() & (last_pivot_low < prev_day_high_s)

    fib_a = pd.Series(
        np.where(
            trend_dir == 1,
            prev_day_low_s,
            np.where(trend_dir == -1, prev_day_high_s, np.nan),
        ),
        index=base.index,
        dtype=float,
    )
    fib_b = pd.Series(
        np.where(
            trend_dir == 1,
            last_pivot_high,
            np.where(trend_dir == -1, last_pivot_low, np.nan),
        ),
        index=base.index,
        dtype=float,
    )

    fib_range = (fib_b - fib_a).abs()
    fib_ready = pd.Series(
        np.where(
            trend_dir == 1,
            bull_fib_ready,
            np.where(trend_dir == -1, bear_fib_ready, False),
        ),
        index=base.index,
        dtype=bool,
    )

    fib25 = _fib_level(trend_dir, fib_a, fib_range, 0.25)
    fib33 = _fib_level(trend_dir, fib_a, fib_range, 0.33)
    fib50 = _fib_level(trend_dir, fib_a, fib_range, 0.50)
    fib615 = _fib_level(trend_dir, fib_a, fib_range, 0.615)
    fib66 = _fib_level(trend_dir, fib_a, fib_range, 0.66)
    fib78 = _fib_level(trend_dir, fib_a, fib_range, 0.78)

    out["fib25"] = fib25
    out["fib33"] = fib33
    out["fib50"] = fib50
    out["fib615"] = fib615
    out["fib66"] = fib66
    out["fib78"] = fib78

    for name, level in {
        "fib25": fib25,
        "fib33": fib33,
        "fib50": fib50,
        "fib615": fib615,
        "fib66": fib66,
        "fib78": fib78,
    }.items():
        raw_now = fib_ready & _within_tolerance(base["low"], base["high"], level, tol)
        ttl_df = _ttl_flag(raw_now, cfg.ttl_bars)
        out[f"rt_{name}_now"] = ttl_df["touch_now"]
        out[f"rt_{name}_active"] = ttl_df["touch_active"]
        out[f"rt_{name}_ttl"] = ttl_df["touch_ttl"]

    # -------------------------------------------------------------------------
    # FAMILY FLAGS
    # -------------------------------------------------------------------------
    out["rt_any_ema"] = (
        (out["rt_ema1420_active"] == 1)
        | (out["rt_ema3350_active"] == 1)
        | (out["rt_ema100200_active"] == 1)
    ).astype(int)

    out["rt_any_structure"] = (
        (out["rt_session_high_active"] == 1)
        | (out["rt_session_low_active"] == 1)
        | (out["rt_prev_day_high_active"] == 1)
        | (out["rt_prev_day_low_active"] == 1)
        | (out["rt_first_5m_high_active"] == 1)
        | (out["rt_first_5m_low_active"] == 1)
        | (out["rt_first_15m_high_active"] == 1)
        | (out["rt_first_15m_low_active"] == 1)
        | (out["rt_first_1h_high_active"] == 1)
        | (out["rt_first_1h_low_active"] == 1)
        | (out["rt_first_4h_high_active"] == 1)
        | (out["rt_first_4h_low_active"] == 1)
    ).astype(int)

    out["rt_any_fib"] = (
        (out["rt_fib25_active"] == 1)
        | (out["rt_fib33_active"] == 1)
        | (out["rt_fib50_active"] == 1)
        | (out["rt_fib615_active"] == 1)
        | (out["rt_fib66_active"] == 1)
        | (out["rt_fib78_active"] == 1)
    ).astype(int)

    out["rt_any_cloud"] = (out["rt_conf_cloud_active"] == 1).astype(int)
    out["rt_any_confluence"] = (out["rt_confluence_zone_active"] == 1).astype(int)
    out["rt_any_fvg"] = (out["rt_fvg_active"] == 1).astype(int)

    out["rt_any"] = (
        (out["rt_any_ema"] == 1)
        | (out["rt_any_structure"] == 1)
        | (out["rt_any_fib"] == 1)
        | (out["rt_any_cloud"] == 1)
        | (out["rt_any_confluence"] == 1)
        | (out["rt_any_fvg"] == 1)
    ).astype(int)

    # -------------------------------------------------------------------------
    # TRIGGER CANDLE SCORING (OHLCV / FOOTPRINT PROXY)
    # -------------------------------------------------------------------------
    candle_range = (base["high"] - base["low"]).abs()
    body_frac = _body_fraction(base)
    close_pos = _close_position(base)

    upper_wick = _upper_wick(base)
    lower_wick = _lower_wick(base)

    vol_ma = _sma(base["volume"], cfg.volume_ma_len)
    rel_volume = _safe_div(base["volume"], vol_ma).clip(0.0, 3.0)

    signed_pressure = np.sign(base["close"] - base["open"])
    delta_proxy_raw = signed_pressure * body_frac * rel_volume
    delta_strength = pd.Series(delta_proxy_raw, index=base.index, dtype=float).clip(-3.0, 3.0)

    displacement_flag = (
        (candle_range >= (out["atr_now"] * cfg.displacement_atr_mult))
        & (body_frac >= 0.60)
        & (rel_volume >= 1.10)
    ).astype(int)

    displacement_score = (
        (_safe_div(candle_range, out["atr_now"]).clip(0.0, 3.0) / 3.0) * 0.50
        + body_frac.clip(0.0, 1.0) * 0.30
        + (rel_volume.clip(0.0, 3.0) / 3.0) * 0.20
    ).clip(0.0, 1.0)

    wick_ratio_lower = _safe_div(lower_wick, candle_range).clip(0.0, 1.0)
    wick_ratio_upper = _safe_div(upper_wick, candle_range).clip(0.0, 1.0)

    bull_absorption_flag = (
        (wick_ratio_lower >= cfg.absorption_wick_ratio_min)
        & (close_pos >= 0.50)
        & (rel_volume >= 1.0)
    ).astype(int)

    bear_absorption_flag = (
        (wick_ratio_upper >= cfg.absorption_wick_ratio_min)
        & (close_pos <= 0.50)
        & (rel_volume >= 1.0)
    ).astype(int)

    bull_absorption_score = (
        wick_ratio_lower * 0.45
        + close_pos.clip(0.0, 1.0) * 0.30
        + (rel_volume.clip(0.0, 3.0) / 3.0) * 0.25
    ).clip(0.0, 1.0)

    bear_absorption_score = (
        wick_ratio_upper * 0.45
        + (1.0 - close_pos.clip(0.0, 1.0)) * 0.30
        + (rel_volume.clip(0.0, 3.0) / 3.0) * 0.25
    ).clip(0.0, 1.0)

    bull_score = (
        (delta_strength.clip(lower=0.0, upper=3.0) / 3.0) * 0.35
        + close_pos.clip(0.0, 1.0) * 0.20
        + body_frac.clip(0.0, 1.0) * 0.15
        + bull_absorption_score * 0.15
        + displacement_score * 0.15
    ).clip(0.0, 1.0)

    bear_score = (
        ((-delta_strength).clip(lower=0.0, upper=3.0) / 3.0) * 0.35
        + (1.0 - close_pos.clip(0.0, 1.0)) * 0.20
        + body_frac.clip(0.0, 1.0) * 0.15
        + bear_absorption_score * 0.15
        + displacement_score * 0.15
    ).clip(0.0, 1.0)

    delta_bull_score = (bull_score * 100.0).clip(0.0, 100.0)
    delta_bear_score = -(bear_score * 100.0).clip(0.0, 100.0)

    delta_winner_score = pd.Series(
        np.where(
            bull_score > bear_score,
            delta_bull_score,
            np.where(bear_score > bull_score, delta_bear_score, 0.0),
        ),
        index=base.index,
        dtype=float,
    )

    delta_winner_label = pd.Series(
        np.where(
            bull_score > bear_score,
            "bull",
            np.where(bear_score > bull_score, "bear", "neutral"),
        ),
        index=base.index,
        dtype="object",
    )

    final_trigger_score = pd.Series(
        np.where(
            trend_dir > 0,
            bull_score,
            np.where(
                trend_dir < 0,
                bear_score,
                np.maximum(bull_score, bear_score),
            ),
        ),
        index=base.index,
        dtype=float,
    ).clip(0.0, 1.0)

    trigger_bias = pd.Series(
        np.where(
            bull_score > bear_score,
            1,
            np.where(bear_score > bull_score, -1, 0),
        ),
        index=base.index,
        dtype=int,
    )

    trigger_bias_label = pd.Series(
        np.where(
            trigger_bias > 0,
            "bullish",
            np.where(trigger_bias < 0, "bearish", "neutral"),
        ),
        index=base.index,
        dtype="object",
    )

    out["trigger_body_frac"] = body_frac.astype(float)
    out["trigger_close_pos"] = close_pos.astype(float)
    out["trigger_rel_volume"] = rel_volume.astype(float)
    out["trigger_delta_strength"] = delta_strength.astype(float)

    out["trigger_displacement_flag"] = displacement_flag.astype(int)
    out["trigger_displacement_score"] = displacement_score.astype(float)

    out["trigger_bull_absorption_flag"] = bull_absorption_flag.astype(int)
    out["trigger_bear_absorption_flag"] = bear_absorption_flag.astype(int)
    out["trigger_bull_absorption_score"] = bull_absorption_score.astype(float)
    out["trigger_bear_absorption_score"] = bear_absorption_score.astype(float)

    out["trigger_bull_score"] = bull_score.astype(float)
    out["trigger_bear_score"] = bear_score.astype(float)
    out["trigger_final_score"] = final_trigger_score.astype(float)
    out["trigger_bias"] = trigger_bias.astype(int)
    out["trigger_bias_label"] = trigger_bias_label

    out["delta_bull_score"] = delta_bull_score.astype(float)
    out["delta_bear_score"] = delta_bear_score.astype(float)
    out["delta_winner_score"] = delta_winner_score.astype(float)
    out["delta_winner_label"] = delta_winner_label

    # -------------------------------------------------------------------------
    # EXPORT CONTRACT
    # -------------------------------------------------------------------------
    export_cols = [
        "rt_ema1420_active",
        "rt_ema3350_active",
        "rt_ema100200_active",
        "rt_conf_cloud_active",
        "rt_confluence_zone_active",
        "rt_fvg_active",
        "rt_session_high_active",
        "rt_session_low_active",
        "rt_prev_day_high_active",
        "rt_prev_day_low_active",
        "rt_first_5m_high_active",
        "rt_first_5m_low_active",
        "rt_first_15m_high_active",
        "rt_first_15m_low_active",
        "rt_first_1h_high_active",
        "rt_first_1h_low_active",
        "rt_first_4h_high_active",
        "rt_first_4h_low_active",
        "rt_fib25_active",
        "rt_fib33_active",
        "rt_fib50_active",
        "rt_fib615_active",
        "rt_fib66_active",
        "rt_fib78_active",
        "rt_any_ema",
        "rt_any_structure",
        "rt_any_fib",
        "rt_any_cloud",
        "rt_any_confluence",
        "rt_any_fvg",
        "rt_any",
    ]

    for col in export_cols:
        out[f"{col}_export"] = out[col].astype(int)

    out["pb_direction"] = trend_dir.astype(int)

    retest_family_strength = (
        out[
            [
                "rt_any_ema",
                "rt_any_structure",
                "rt_any_fib",
                "rt_any_cloud",
                "rt_any_confluence",
                "rt_any_fvg",
            ]
        ].sum(axis=1) / 6.0
    ).astype(float)

    out["pb_signal"] = (
        (out["rt_any"] == 1) & (out["trigger_final_score"] >= 0.45)
    ).astype(int)

    out["pb_strength"] = (
        retest_family_strength * 0.55
        + out["trigger_final_score"].clip(0.0, 1.0) * 0.45
    ).clip(0.0, 1.0).astype(float)

    return out


def build_pullback_retest(
    df: pd.DataFrame,
    config: Optional[PullbackRetestConfig] = None,
    confluence_df: Optional[pd.DataFrame] = None,
    trend_df: Optional[pd.DataFrame] = None,
    fvg_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    return calculate_pullback_retest(
        df=df,
        config=config,
        confluence_df=confluence_df,
        trend_df=trend_df,
        fvg_df=fvg_df,
    )


def run_pullback_retest(
    df: pd.DataFrame,
    config: Optional[PullbackRetestConfig] = None,
    confluence_df: Optional[pd.DataFrame] = None,
    trend_df: Optional[pd.DataFrame] = None,
    fvg_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    return calculate_pullback_retest(
        df=df,
        config=config,
        confluence_df=confluence_df,
        trend_df=trend_df,
        fvg_df=fvg_df,
    )


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

DEFAULT_PULLBACK_RETEST_CONFIG: Dict[str, Any] = asdict(PullbackRetestConfig())


def build_pullback_retest_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    confluence_df: Optional[pd.DataFrame] = None,
    trend_df: Optional[pd.DataFrame] = None,
    fvg_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    cfg_dict = DEFAULT_PULLBACK_RETEST_CONFIG.copy()
    if config:
        cfg_dict.update(config)

    cfg = PullbackRetestConfig(**cfg_dict)

    result = calculate_pullback_retest(
        df=df,
        config=cfg,
        confluence_df=confluence_df,
        trend_df=trend_df,
        fvg_df=fvg_df,
    )

    if result.empty:
        raise ValueError("Pullback/Retest payload build failed: empty dataframe result")

    last_idx = result.index[-1]
    last = result.iloc[-1]

    direction = int(_to_native(last.get("pb_direction", 0)) or 0)
    signal = int(_to_native(last.get("pb_signal", 0)) or 0)
    strength = float(_to_native(last.get("pb_strength", 0.0)) or 0.0)

    trigger_bias = int(_to_native(last.get("trigger_bias", 0)) or 0)
    trigger_bias_label = str(_to_native(last.get("trigger_bias_label")) or "neutral")

    if direction > 0:
        direction_label = "bullish"
    elif direction < 0:
        direction_label = "bearish"
    else:
        direction_label = "neutral"

    active_groups = []
    if int(_to_native(last.get("rt_any_ema_export", 0)) or 0) == 1:
        active_groups.append("ema")
    if int(_to_native(last.get("rt_any_structure_export", 0)) or 0) == 1:
        active_groups.append("structure")
    if int(_to_native(last.get("rt_any_fib_export", 0)) or 0) == 1:
        active_groups.append("fib")
    if int(_to_native(last.get("rt_any_cloud_export", 0)) or 0) == 1:
        active_groups.append("cloud")
    if int(_to_native(last.get("rt_any_confluence_export", 0)) or 0) == 1:
        active_groups.append("confluence")
    if int(_to_native(last.get("rt_any_fvg_export", 0)) or 0) == 1:
        active_groups.append("fvg")

    if signal == 1 and strength >= 0.75:
        state_label = "strong_retest"
    elif signal == 1 and strength >= 0.50:
        state_label = "active_retest"
    elif signal == 1:
        state_label = "weak_retest"
    else:
        state_label = "no_retest"

    bias_signal = direction if signal == 1 else 0
    if bias_signal > 0:
        bias_label = "BULLISH"
    elif bias_signal < 0:
        bias_label = "BEARISH"
    else:
        bias_label = "NEUTRAL"

    market_bias = bias_label if signal == 1 else "NEUTRAL"

    payload = {
        "indicator": "pullback_retest",
        "debug_version": "pullback_retest_payload_v3",
        "timestamp": last_idx.isoformat(),

        # Shared website / Ultimate Truth contract
        "state": state_label,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(strength, 4),
        "market_bias": market_bias,

        # Specialist state
        "direction": direction,
        "direction_label": direction_label,
        "signal": signal,
        "strength": round(strength, 4),
        "state_label": state_label,
        "active_groups": active_groups,

        # Trigger candle / delta specialist fields
        "trigger_delta_strength": round(float(_to_native(last.get("trigger_delta_strength", 0.0)) or 0.0), 4),
        "trigger_bull_score": round(float(_to_native(last.get("trigger_bull_score", 0.0)) or 0.0), 4),
        "trigger_bear_score": round(float(_to_native(last.get("trigger_bear_score", 0.0)) or 0.0), 4),
        "trigger_final_score": round(float(_to_native(last.get("trigger_final_score", 0.0)) or 0.0), 4),
        "trigger_bias": trigger_bias,
        "trigger_bias_label": trigger_bias_label,
        "trigger_displacement_flag": int(_to_native(last.get("trigger_displacement_flag", 0)) or 0),
        "trigger_displacement_score": round(float(_to_native(last.get("trigger_displacement_score", 0.0)) or 0.0), 4),
        "trigger_bull_absorption_flag": int(_to_native(last.get("trigger_bull_absorption_flag", 0)) or 0),
        "trigger_bear_absorption_flag": int(_to_native(last.get("trigger_bear_absorption_flag", 0)) or 0),
        "trigger_bull_absorption_score": round(float(_to_native(last.get("trigger_bull_absorption_score", 0.0)) or 0.0), 4),
        "trigger_bear_absorption_score": round(float(_to_native(last.get("trigger_bear_absorption_score", 0.0)) or 0.0), 4),
        "trigger_body_frac": round(float(_to_native(last.get("trigger_body_frac", 0.0)) or 0.0), 4),
        "trigger_close_pos": round(float(_to_native(last.get("trigger_close_pos", 0.0)) or 0.0), 4),
        "trigger_rel_volume": round(float(_to_native(last.get("trigger_rel_volume", 0.0)) or 0.0), 4),

        # Delta section
        "delta_bull_score": round(float(_to_native(last.get("delta_bull_score", 0.0)) or 0.0), 2),
        "delta_bear_score": round(float(_to_native(last.get("delta_bear_score", 0.0)) or 0.0), 2),
        "delta_winner_score": round(float(_to_native(last.get("delta_winner_score", 0.0)) or 0.0), 2),
        "delta_winner_label": str(_to_native(last.get("delta_winner_label")) or "neutral"),

        "levels": {
            "ema14": _to_native(last.get("ema14")),
            "ema20": _to_native(last.get("ema20")),
            "ema33": _to_native(last.get("ema33")),
            "ema50": _to_native(last.get("ema50")),
            "ema100": _to_native(last.get("ema100")),
            "ema200": _to_native(last.get("ema200")),
            "session_high": _to_native(last.get("session_high")),
            "session_low": _to_native(last.get("session_low")),
            "prev_day_high": _to_native(last.get("prev_day_high")),
            "prev_day_low": _to_native(last.get("prev_day_low")),
            "first_5m_high": _to_native(last.get("first_5m_high")),
            "first_5m_low": _to_native(last.get("first_5m_low")),
            "first_15m_high": _to_native(last.get("first_15m_high")),
            "first_15m_low": _to_native(last.get("first_15m_low")),
            "first_1h_high": _to_native(last.get("first_1h_high")),
            "first_1h_low": _to_native(last.get("first_1h_low")),
            "first_4h_high": _to_native(last.get("first_4h_high")),
            "first_4h_low": _to_native(last.get("first_4h_low")),
            "conf_cloud_lo": _to_native(last.get("conf_cloud_lo")),
            "conf_cloud_hi": _to_native(last.get("conf_cloud_hi")),
            "conf_cloud_mid": _to_native(last.get("conf_cloud_mid")),
            "confluence_zone_lo": _to_native(last.get("confluence_zone_lo")),
            "confluence_zone_hi": _to_native(last.get("confluence_zone_hi")),
            "confluence_zone_mid": _to_native(last.get("confluence_zone_mid")),
            "fvg_lo": _to_native(last.get("fvg_lo")),
            "fvg_hi": _to_native(last.get("fvg_hi")),
            "fvg_mid": _to_native(last.get("fvg_mid")),
            "fib25": _to_native(last.get("fib25")),
            "fib33": _to_native(last.get("fib33")),
            "fib50": _to_native(last.get("fib50")),
            "fib615": _to_native(last.get("fib615")),
            "fib66": _to_native(last.get("fib66")),
            "fib78": _to_native(last.get("fib78")),
            "atr_now": _to_native(last.get("atr_now")),
            "retest_tolerance": _to_native(last.get("retest_tolerance")),
        },

        "retests": {
            "ema": {
                "ema1420": {
                    "active": int(_to_native(last.get("rt_ema1420_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_ema1420_ttl", 0)) or 0),
                },
                "ema3350": {
                    "active": int(_to_native(last.get("rt_ema3350_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_ema3350_ttl", 0)) or 0),
                },
                "ema100200": {
                    "active": int(_to_native(last.get("rt_ema100200_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_ema100200_ttl", 0)) or 0),
                },
                "any": int(_to_native(last.get("rt_any_ema_export", 0)) or 0),
            },
            "structure": {
                "session_high": {
                    "active": int(_to_native(last.get("rt_session_high_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_session_high_ttl", 0)) or 0),
                },
                "session_low": {
                    "active": int(_to_native(last.get("rt_session_low_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_session_low_ttl", 0)) or 0),
                },
                "prev_day_high": {
                    "active": int(_to_native(last.get("rt_prev_day_high_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_prev_day_high_ttl", 0)) or 0),
                },
                "prev_day_low": {
                    "active": int(_to_native(last.get("rt_prev_day_low_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_prev_day_low_ttl", 0)) or 0),
                },
                "first_5m_high": {
                    "active": int(_to_native(last.get("rt_first_5m_high_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_5m_high_ttl", 0)) or 0),
                },
                "first_5m_low": {
                    "active": int(_to_native(last.get("rt_first_5m_low_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_5m_low_ttl", 0)) or 0),
                },
                "first_15m_high": {
                    "active": int(_to_native(last.get("rt_first_15m_high_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_15m_high_ttl", 0)) or 0),
                },
                "first_15m_low": {
                    "active": int(_to_native(last.get("rt_first_15m_low_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_15m_low_ttl", 0)) or 0),
                },
                "first_1h_high": {
                    "active": int(_to_native(last.get("rt_first_1h_high_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_1h_high_ttl", 0)) or 0),
                },
                "first_1h_low": {
                    "active": int(_to_native(last.get("rt_first_1h_low_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_1h_low_ttl", 0)) or 0),
                },
                "first_4h_high": {
                    "active": int(_to_native(last.get("rt_first_4h_high_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_4h_high_ttl", 0)) or 0),
                },
                "first_4h_low": {
                    "active": int(_to_native(last.get("rt_first_4h_low_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_first_4h_low_ttl", 0)) or 0),
                },
                "any": int(_to_native(last.get("rt_any_structure_export", 0)) or 0),
            },
            "cloud": {
                "active": int(_to_native(last.get("rt_conf_cloud_active_export", 0)) or 0),
                "ttl": int(_to_native(last.get("rt_conf_cloud_ttl", 0)) or 0),
                "any": int(_to_native(last.get("rt_any_cloud_export", 0)) or 0),
            },
            "confluence": {
                "active": int(_to_native(last.get("rt_confluence_zone_active_export", 0)) or 0),
                "ttl": int(_to_native(last.get("rt_confluence_zone_ttl", 0)) or 0),
                "any": int(_to_native(last.get("rt_any_confluence_export", 0)) or 0),
            },
            "fvg": {
                "active": int(_to_native(last.get("rt_fvg_active_export", 0)) or 0),
                "ttl": int(_to_native(last.get("rt_fvg_ttl", 0)) or 0),
                "any": int(_to_native(last.get("rt_any_fvg_export", 0)) or 0),
            },
            "fib": {
                "fib25": {
                    "active": int(_to_native(last.get("rt_fib25_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_fib25_ttl", 0)) or 0),
                },
                "fib33": {
                    "active": int(_to_native(last.get("rt_fib33_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_fib33_ttl", 0)) or 0),
                },
                "fib50": {
                    "active": int(_to_native(last.get("rt_fib50_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_fib50_ttl", 0)) or 0),
                },
                "fib615": {
                    "active": int(_to_native(last.get("rt_fib615_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_fib615_ttl", 0)) or 0),
                },
                "fib66": {
                    "active": int(_to_native(last.get("rt_fib66_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_fib66_ttl", 0)) or 0),
                },
                "fib78": {
                    "active": int(_to_native(last.get("rt_fib78_active_export", 0)) or 0),
                    "ttl": int(_to_native(last.get("rt_fib78_ttl", 0)) or 0),
                },
                "any": int(_to_native(last.get("rt_any_fib_export", 0)) or 0),
            },
            "any": int(_to_native(last.get("rt_any_export", 0)) or 0),
        },

        "price": {
            "close": _to_native(df["close"].iloc[-1]),
        },

        "config": cfg_dict,
    }

    return payload