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
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()


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


def _within_tolerance(price_low: pd.Series, price_high: pd.Series, level: pd.Series, tol: pd.Series) -> pd.Series:
    return (price_low <= (level + tol)) & (price_high >= (level - tol))


def _rolling_pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    vals = high.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    for i in range(left, n - right):
        center = vals[i]
        if np.isnan(center):
            continue
        window = vals[i - left:i + right + 1]
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
        window = vals[i - left:i + right + 1]
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
    """
    For each day, extract the first bar high/low after resampling to rule.
    """
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
    ttl = pd.Series(np.where(active, np.maximum(0, ttl_bars - bars), 0), index=raw_touch.index, dtype=int)
    return pd.DataFrame(
        {
            "touch_now": raw_touch.astype(int),
            "touch_active": active.astype(int),
            "touch_ttl": ttl,
        },
        index=raw_touch.index,
    )


# =============================================================================
# CORE
# =============================================================================

def calculate_pullback_retest(
    df: pd.DataFrame,
    config: Optional[PullbackRetestConfig] = None,
    confluence_df: Optional[pd.DataFrame] = None,
    trend_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    SmartChart Pullback / Retest Engine

    Optional inputs:
    - confluence_df: may contain conf_zone_lo_export / conf_zone_hi_export
    - trend_df: may contain trend direction export if available

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
    # FIRST 5M / 15M SESSION CANDLE H/L
    # -------------------------------------------------------------------------
    first_5m = _first_bar_levels(base, "5min")
    first_15m = _first_bar_levels(base, "15min")

    first_5m_high = first_5m["first_5min_high"]
    first_5m_low = first_5m["first_5min_low"]

    first_15m_high = first_15m["first_15min_high"]
    first_15m_low = first_15m["first_15min_low"]

    rt_first_5m_high_now = _within_tolerance(base["low"], base["high"], first_5m_high, tol)
    rt_first_5m_low_now = _within_tolerance(base["low"], base["high"], first_5m_low, tol)
    rt_first_15m_high_now = _within_tolerance(base["low"], base["high"], first_15m_high, tol)
    rt_first_15m_low_now = _within_tolerance(base["low"], base["high"], first_15m_low, tol)

    first_5m_high_ttl = _ttl_flag(rt_first_5m_high_now, cfg.ttl_bars)
    first_5m_low_ttl = _ttl_flag(rt_first_5m_low_now, cfg.ttl_bars)
    first_15m_high_ttl = _ttl_flag(rt_first_15m_high_now, cfg.ttl_bars)
    first_15m_low_ttl = _ttl_flag(rt_first_15m_low_now, cfg.ttl_bars)

    out["first_5m_high"] = first_5m_high
    out["first_5m_low"] = first_5m_low
    out["first_15m_high"] = first_15m_high
    out["first_15m_low"] = first_15m_low

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

    # -------------------------------------------------------------------------
    # CONFLUENCE CLOUD ZONE RETEST
    # -------------------------------------------------------------------------
    if confluence_df is not None:
        if "conf_zone_lo_export" in confluence_df.columns:
            cloud_lo = confluence_df["conf_zone_lo_export"].reindex(base.index)
        elif "loc_lo" in confluence_df.columns:
            cloud_lo = confluence_df["loc_lo"].reindex(base.index)
        else:
            cloud_lo = pd.Series(np.nan, index=base.index)

        if "conf_zone_hi_export" in confluence_df.columns:
            cloud_hi = confluence_df["conf_zone_hi_export"].reindex(base.index)
        elif "loc_hi" in confluence_df.columns:
            cloud_hi = confluence_df["loc_hi"].reindex(base.index)
        else:
            cloud_hi = pd.Series(np.nan, index=base.index)
    else:
        cloud_lo = pd.Series(np.nan, index=base.index)
        cloud_hi = pd.Series(np.nan, index=base.index)

    cloud_mid = (cloud_lo + cloud_hi) / 2.0
    rt_cloud_now = cloud_lo.notna() & cloud_hi.notna() & (base["low"] <= cloud_hi) & (base["high"] >= cloud_lo)
    cloud_ttl = _ttl_flag(rt_cloud_now, cfg.ttl_bars)

    out["conf_cloud_lo"] = cloud_lo
    out["conf_cloud_hi"] = cloud_hi
    out["conf_cloud_mid"] = cloud_mid
    out["rt_conf_cloud_now"] = cloud_ttl["touch_now"]
    out["rt_conf_cloud_active"] = cloud_ttl["touch_active"]
    out["rt_conf_cloud_ttl"] = cloud_ttl["touch_ttl"]

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
        np.where(trend_dir == 1, prev_day_low_s, np.where(trend_dir == -1, prev_day_high_s, np.nan)),
        index=base.index,
        dtype=float,
    )
    fib_b = pd.Series(
        np.where(trend_dir == 1, last_pivot_high, np.where(trend_dir == -1, last_pivot_low, np.nan)),
        index=base.index,
        dtype=float,
    )

    fib_range = (fib_b - fib_a).abs()
    fib_ready = pd.Series(
        np.where(trend_dir == 1, bull_fib_ready, np.where(trend_dir == -1, bear_fib_ready, False)),
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

    out["rt_any"] = (
        (out["rt_any_ema"] == 1)
        | (out["rt_any_structure"] == 1)
        | (out["rt_any_fib"] == 1)
        | (out["rt_any_cloud"] == 1)
    ).astype(int)

    # -------------------------------------------------------------------------
    # EXPORT CONTRACT
    # -------------------------------------------------------------------------
    export_cols = [
        "rt_ema1420_active",
        "rt_ema3350_active",
        "rt_ema100200_active",
        "rt_conf_cloud_active",
        "rt_session_high_active",
        "rt_session_low_active",
        "rt_prev_day_high_active",
        "rt_prev_day_low_active",
        "rt_first_5m_high_active",
        "rt_first_5m_low_active",
        "rt_first_15m_high_active",
        "rt_first_15m_low_active",
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
        "rt_any",
    ]

    for col in export_cols:
        out[f"{col}_export"] = out[col].astype(int)

    out["pb_direction"] = trend_dir.astype(int)
    out["pb_signal"] = out["rt_any"].astype(int)
    out["pb_strength"] = (
        out[
            [
                "rt_any_ema",
                "rt_any_structure",
                "rt_any_fib",
                "rt_any_cloud",
            ]
        ].sum(axis=1) / 4.0
    ).astype(float)

    return out


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=1500, freq="5min")
    np.random.seed(42)

    drift = np.linspace(0, 20, len(rng))
    noise = np.random.normal(0, 1.0, len(rng)).cumsum()
    close = pd.Series(3300 + drift + noise, index=rng)
    open_ = close.shift(1).fillna(close.iloc[0])

    high = pd.concat([open_, close], axis=1).max(axis=1) + np.random.uniform(0.2, 2.0, len(rng))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.random.uniform(0.2, 2.0, len(rng))
    volume = pd.Series(np.random.randint(100, 5000, len(rng)), index=rng)

    test_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=rng,
    )

    # Mock confluence zone for test
    mock_conf = pd.DataFrame(index=test_df.index)
    mock_conf["conf_zone_lo_export"] = _ema(test_df["close"], 20) - 5.0
    mock_conf["conf_zone_hi_export"] = _ema(test_df["close"], 20) + 5.0

    result = calculate_pullback_retest(test_df, confluence_df=mock_conf)

    cols = [
        "pb_trend_dir",
        "rt_ema1420_active_export",
        "rt_ema3350_active_export",
        "rt_ema100200_active_export",
        "rt_conf_cloud_active_export",
        "rt_session_high_active_export",
        "rt_session_low_active_export",
        "rt_prev_day_high_active_export",
        "rt_prev_day_low_active_export",
        "rt_first_5m_high_active_export",
        "rt_first_5m_low_active_export",
        "rt_first_15m_high_active_export",
        "rt_first_15m_low_active_export",
        "rt_fib25_active_export",
        "rt_fib33_active_export",
        "rt_fib50_active_export",
        "rt_fib615_active_export",
        "rt_fib66_active_export",
        "rt_fib78_active_export",
        "rt_any_ema_export",
        "rt_any_structure_export",
        "rt_any_fib_export",
        "rt_any_cloud_export",
        "rt_any_export",
        "pb_signal",
        "pb_strength",
    ]

    # =============================================================================
# PAYLOAD BUILDER
# =============================================================================

DEFAULT_PULLBACK_RETEST_CONFIG: Dict[str, Any] = asdict(PullbackRetestConfig())


def _to_native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_pullback_retest_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    confluence_df: Optional[pd.DataFrame] = None,
    trend_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build website/API payload for the latest Pullback / Retest state.
    """
    cfg_dict = DEFAULT_PULLBACK_RETEST_CONFIG.copy()
    if config:
        cfg_dict.update(config)

    cfg = PullbackRetestConfig(**cfg_dict)

    result = calculate_pullback_retest(
        df=df,
        config=cfg,
        confluence_df=confluence_df,
        trend_df=trend_df,
    )

    if result.empty:
        return {}

    last_idx = result.index[-1]
    last = result.iloc[-1]

    direction = int(_to_native(last.get("pb_direction", 0)) or 0)
    signal = int(_to_native(last.get("pb_signal", 0)) or 0)
    strength = float(_to_native(last.get("pb_strength", 0.0)) or 0.0)

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

    if signal == 1 and strength >= 0.75:
        state_label = "strong_retest"
    elif signal == 1 and strength >= 0.50:
        state_label = "active_retest"
    elif signal == 1:
        state_label = "weak_retest"
    else:
        state_label = "no_retest"

    payload = {
        "indicator": "pullback_retest",
        "timestamp": last_idx.isoformat(),
        "state": {
            "direction": direction,
            "direction_label": direction_label,
            "signal": signal,
            "strength": round(strength, 4),
            "state_label": state_label,
            "active_groups": active_groups,
        },
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
            "conf_cloud_lo": _to_native(last.get("conf_cloud_lo")),
            "conf_cloud_hi": _to_native(last.get("conf_cloud_hi")),
            "conf_cloud_mid": _to_native(last.get("conf_cloud_mid")),
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
                "any": int(_to_native(last.get("rt_any_structure_export", 0)) or 0),
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
            "cloud": {
                "active": int(_to_native(last.get("rt_conf_cloud_active_export", 0)) or 0),
                "ttl": int(_to_native(last.get("rt_conf_cloud_ttl", 0)) or 0),
                "any": int(_to_native(last.get("rt_any_cloud_export", 0)) or 0),
            },
            "any": int(_to_native(last.get("rt_any_export", 0)) or 0),
        },
        "exports": {
            "pb_direction": direction,
            "pb_signal": signal,
            "pb_strength": round(strength, 4),
            "rt_ema1420_active_export": int(_to_native(last.get("rt_ema1420_active_export", 0)) or 0),
            "rt_ema3350_active_export": int(_to_native(last.get("rt_ema3350_active_export", 0)) or 0),
            "rt_ema100200_active_export": int(_to_native(last.get("rt_ema100200_active_export", 0)) or 0),
            "rt_conf_cloud_active_export": int(_to_native(last.get("rt_conf_cloud_active_export", 0)) or 0),
            "rt_session_high_active_export": int(_to_native(last.get("rt_session_high_active_export", 0)) or 0),
            "rt_session_low_active_export": int(_to_native(last.get("rt_session_low_active_export", 0)) or 0),
            "rt_prev_day_high_active_export": int(_to_native(last.get("rt_prev_day_high_active_export", 0)) or 0),
            "rt_prev_day_low_active_export": int(_to_native(last.get("rt_prev_day_low_active_export", 0)) or 0),
            "rt_first_5m_high_active_export": int(_to_native(last.get("rt_first_5m_high_active_export", 0)) or 0),
            "rt_first_5m_low_active_export": int(_to_native(last.get("rt_first_5m_low_active_export", 0)) or 0),
            "rt_first_15m_high_active_export": int(_to_native(last.get("rt_first_15m_high_active_export", 0)) or 0),
            "rt_first_15m_low_active_export": int(_to_native(last.get("rt_first_15m_low_active_export", 0)) or 0),
            "rt_fib25_active_export": int(_to_native(last.get("rt_fib25_active_export", 0)) or 0),
            "rt_fib33_active_export": int(_to_native(last.get("rt_fib33_active_export", 0)) or 0),
            "rt_fib50_active_export": int(_to_native(last.get("rt_fib50_active_export", 0)) or 0),
            "rt_fib615_active_export": int(_to_native(last.get("rt_fib615_active_export", 0)) or 0),
            "rt_fib66_active_export": int(_to_native(last.get("rt_fib66_active_export", 0)) or 0),
            "rt_fib78_active_export": int(_to_native(last.get("rt_fib78_active_export", 0)) or 0),
            "rt_any_ema_export": int(_to_native(last.get("rt_any_ema_export", 0)) or 0),
            "rt_any_structure_export": int(_to_native(last.get("rt_any_structure_export", 0)) or 0),
            "rt_any_fib_export": int(_to_native(last.get("rt_any_fib_export", 0)) or 0),
            "rt_any_cloud_export": int(_to_native(last.get("rt_any_cloud_export", 0)) or 0),
            "rt_any_export": int(_to_native(last.get("rt_any_export", 0)) or 0),
        },
        "config": cfg_dict,
    }

    return payload