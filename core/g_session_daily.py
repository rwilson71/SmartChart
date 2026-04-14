from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# MODULE G — SESSION & DAILY ENGINE
# Pine authority:
# SmartChart • Session & Daily Validation v2
# =============================================================================

DEFAULT_SESSION_DAILY_CONFIG: Dict[str, Any] = {
    # Sessions
    "session_timezone": "Europe/London",

    # Session windows
    "asia_start": "00:00",
    "asia_end": "08:00",
    "london_start": "08:00",
    "london_end": "13:00",
    "newyork_start": "13:30",
    "newyork_end": "21:00",
}


# =============================================================================
# HELPERS
# =============================================================================

def _to_local_index(index: pd.DatetimeIndex, tz_name: str) -> pd.DatetimeIndex:
    """
    Convert incoming index to the target timezone.
    If index is naive, assume UTC first.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be a pandas DatetimeIndex")

    if index.tz is None:
        return index.tz_localize("UTC").tz_convert(tz_name)

    return index.tz_convert(tz_name)


def _parse_hhmm(value: str) -> Tuple[int, int]:
    hh, mm = value.split(":")
    return int(hh), int(mm)


def _in_session(
    local_index: pd.DatetimeIndex,
    start_str: str,
    end_str: str,
) -> pd.Series:
    """
    Match Pine session membership using local wall-clock time.
    Start inclusive, end exclusive.
    Supports overnight sessions.
    """
    sh, sm = _parse_hhmm(start_str)
    eh, em = _parse_hhmm(end_str)

    mins = local_index.hour * 60 + local_index.minute
    start_m = sh * 60 + sm
    end_m = eh * 60 + em

    if end_m >= start_m:
        mask = (mins >= start_m) & (mins < end_m)
    else:
        mask = (mins >= start_m) | (mins < end_m)

    return pd.Series(mask, index=local_index, dtype=bool)


def _touch_level(
    high: pd.Series,
    low: pd.Series,
    level: pd.Series,
) -> pd.Series:
    """
    Pine parity:
    not na(level) and high >= level and low <= level
    """
    level = pd.to_numeric(level, errors="coerce")
    return ((level.notna()) & (high >= level) & (low <= level)).astype(int)


def _session_tracker(
    high: pd.Series,
    low: pd.Series,
    in_session: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Pine parity for current and previous session high/low.

    Pine logic:
    - reset current hi/lo on session start to current bar high/low
    - update current hi/lo while inside session
    - on session end, lock prev hi/lo = completed session hi/lo
    - current hi/lo become na outside session
    """
    cur_hi = pd.Series(np.nan, index=high.index, dtype=float)
    cur_lo = pd.Series(np.nan, index=low.index, dtype=float)
    prev_hi = pd.Series(np.nan, index=high.index, dtype=float)
    prev_lo = pd.Series(np.nan, index=low.index, dtype=float)

    active_hi = np.nan
    active_lo = np.nan
    locked_hi = np.nan
    locked_lo = np.nan

    prev_in = False

    for i in range(len(high)):
        now_in = bool(in_session.iloc[i])
        start = now_in and not prev_in
        end = (not now_in) and prev_in

        if start:
            active_hi = float(high.iloc[i])
            active_lo = float(low.iloc[i])
        elif now_in:
            active_hi = max(active_hi, float(high.iloc[i])) if pd.notna(active_hi) else float(high.iloc[i])
            active_lo = min(active_lo, float(low.iloc[i])) if pd.notna(active_lo) else float(low.iloc[i])

        if end:
            locked_hi = active_hi
            locked_lo = active_lo
            active_hi = np.nan
            active_lo = np.nan

        cur_hi.iloc[i] = active_hi if now_in else np.nan
        cur_lo.iloc[i] = active_lo if now_in else np.nan
        prev_hi.iloc[i] = locked_hi
        prev_lo.iloc[i] = locked_lo

        prev_in = now_in

    return {
        "cur_hi": cur_hi,
        "cur_lo": cur_lo,
        "prev_hi": prev_hi,
        "prev_lo": prev_lo,
    }


def _opening_range_tracker(
    high: pd.Series,
    low: pd.Series,
    in_session: pd.Series,
    bars_required: int,
) -> Dict[str, pd.Series]:
    """
    Pine parity opening range tracker.

    Pine logic:
    - on session start:
        count = 1
        OR hi/lo = current bar high/low
    - while in session:
        count += 1
        if count <= bars_required:
            update OR hi/lo
    - outside session:
        values persist; Pine vars are not reset outside session

    Important:
    This is chart-bar based parity.
    On 1m:
      OR5  = first 1 bar
      OR15 = first 3 bars
    """
    or_hi = pd.Series(np.nan, index=high.index, dtype=float)
    or_lo = pd.Series(np.nan, index=low.index, dtype=float)
    count_series = pd.Series(0, index=high.index, dtype=int)

    count = 0
    current_hi = np.nan
    current_lo = np.nan
    prev_in = False

    for i in range(len(high)):
        now_in = bool(in_session.iloc[i])
        start = now_in and not prev_in

        if start:
            count = 1
            current_hi = float(high.iloc[i])
            current_lo = float(low.iloc[i])

        elif now_in:
            count += 1
            if count <= bars_required:
                current_hi = max(current_hi, float(high.iloc[i])) if pd.notna(current_hi) else float(high.iloc[i])
                current_lo = min(current_lo, float(low.iloc[i])) if pd.notna(current_lo) else float(low.iloc[i])

        or_hi.iloc[i] = current_hi
        or_lo.iloc[i] = current_lo
        count_series.iloc[i] = count if now_in else 0

        prev_in = now_in

    return {
        "or_hi": or_hi,
        "or_lo": or_lo,
        "count": count_series,
    }


def _daily_levels_from_intraday(
    df: pd.DataFrame,
    tz_name: str,
) -> Dict[str, pd.Series]:
    """
    Pine parity intent for:
      request.security(..., "D", high[1]/low[1])

    We build local trading-day buckets in the configured session timezone,
    aggregate daily high/low, then shift by 1 day and map back to intraday bars.
    """
    local_idx = _to_local_index(df.index, tz_name)
    day_key = pd.Series(local_idx.strftime("%Y-%m-%d"), index=df.index, dtype=object)

    daily = (
        pd.DataFrame(
            {
                "day_key": day_key.values,
                "high": pd.to_numeric(df["high"], errors="coerce").astype(float),
                "low": pd.to_numeric(df["low"], errors="coerce").astype(float),
            },
            index=df.index,
        )
        .groupby("day_key", sort=True)
        .agg({"high": "max", "low": "min"})
    )

    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily["prev_mid"] = (daily["prev_high"] + daily["prev_low"]) * 0.5

    prev_day_high = pd.to_numeric(day_key.map(daily["prev_high"]), errors="coerce")
    prev_day_low = pd.to_numeric(day_key.map(daily["prev_low"]), errors="coerce")
    prev_day_mid = pd.to_numeric(day_key.map(daily["prev_mid"]), errors="coerce")

    return {
        "day_key": day_key,
        "prev_day_high": prev_day_high,
        "prev_day_low": prev_day_low,
        "prev_day_mid": prev_day_mid,
    }


def _session_text(asia_in: pd.Series, london_in: pd.Series, ny_in: pd.Series) -> pd.Series:
    """
    Pine parity for table session text:
      ASIA / LONDON / NEW YORK / NONE
    """
    out = pd.Series("NONE", index=asia_in.index, dtype=object)
    out.loc[asia_in.astype(bool)] = "ASIA"
    out.loc[london_in.astype(bool)] = "LONDON"
    out.loc[ny_in.astype(bool)] = "NEW YORK"
    return out


# =============================================================================
# CORE ENGINE
# =============================================================================

def compute_session_daily_engine(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    cfg = {**DEFAULT_SESSION_DAILY_CONFIG, **(config or {})}

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex")

    out = df.copy()

    high = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    if high.isna().any() or low.isna().any() or close.isna().any():
        raise ValueError("Input OHLC columns contain non-numeric or missing values after coercion")

    local_idx = _to_local_index(out.index, cfg["session_timezone"])

    # =========================================================================
    # SESSION FLAGS
    # =========================================================================

    asia_in = _in_session(local_idx, cfg["asia_start"], cfg["asia_end"]).reindex(out.index, fill_value=False)
    london_in = _in_session(local_idx, cfg["london_start"], cfg["london_end"]).reindex(out.index, fill_value=False)
    ny_in = _in_session(local_idx, cfg["newyork_start"], cfg["newyork_end"]).reindex(out.index, fill_value=False)

    asia_start = asia_in & ~asia_in.shift(1, fill_value=False)
    london_start = london_in & ~london_in.shift(1, fill_value=False)
    ny_start = ny_in & ~ny_in.shift(1, fill_value=False)

    asia_end = ~asia_in & asia_in.shift(1, fill_value=False)
    london_end = ~london_in & london_in.shift(1, fill_value=False)
    ny_end = ~ny_in & ny_in.shift(1, fill_value=False)

    # =========================================================================
    # SESSION TRACKERS
    # =========================================================================

    asia_track = _session_tracker(high, low, asia_in)
    london_track = _session_tracker(high, low, london_in)
    ny_track = _session_tracker(high, low, ny_in)

    asia_cur_hi = asia_track["cur_hi"]
    asia_cur_lo = asia_track["cur_lo"]
    asia_prev_hi = asia_track["prev_hi"]
    asia_prev_lo = asia_track["prev_lo"]

    london_cur_hi = london_track["cur_hi"]
    london_cur_lo = london_track["cur_lo"]
    london_prev_hi = london_track["prev_hi"]
    london_prev_lo = london_track["prev_lo"]

    ny_cur_hi = ny_track["cur_hi"]
    ny_cur_lo = ny_track["cur_lo"]
    ny_prev_hi = ny_track["prev_hi"]
    ny_prev_lo = ny_track["prev_lo"]

    # =========================================================================
    # OPENING RANGE TRACKERS
    # Pine parity:
    # OR5  = first 1 bar
    # OR15 = first 3 bars
    # =========================================================================

    asia_or5 = _opening_range_tracker(high, low, asia_in, bars_required=1)
    asia_or15 = _opening_range_tracker(high, low, asia_in, bars_required=3)

    london_or5 = _opening_range_tracker(high, low, london_in, bars_required=1)
    london_or15 = _opening_range_tracker(high, low, london_in, bars_required=3)

    ny_or5 = _opening_range_tracker(high, low, ny_in, bars_required=1)
    ny_or15 = _opening_range_tracker(high, low, ny_in, bars_required=3)

    asia_or5_hi = asia_or5["or_hi"]
    asia_or5_lo = asia_or5["or_lo"]
    asia_or15_hi = asia_or15["or_hi"]
    asia_or15_lo = asia_or15["or_lo"]

    london_or5_hi = london_or5["or_hi"]
    london_or5_lo = london_or5["or_lo"]
    london_or15_hi = london_or15["or_hi"]
    london_or15_lo = london_or15["or_lo"]

    ny_or5_hi = ny_or5["or_hi"]
    ny_or5_lo = ny_or5["or_lo"]
    ny_or15_hi = ny_or15["or_hi"]
    ny_or15_lo = ny_or15["or_lo"]

    # =========================================================================
    # PREVIOUS DAY LEVELS
    # =========================================================================

    daily = _daily_levels_from_intraday(out, cfg["session_timezone"])
    day_key = daily["day_key"]
    prev_day_high = daily["prev_day_high"]
    prev_day_low = daily["prev_day_low"]
    prev_day_mid = daily["prev_day_mid"]

    # =========================================================================
    # TOUCH FLAGS
    # =========================================================================

    touch_prev_day_high = _touch_level(high, low, prev_day_high)
    touch_prev_day_low = _touch_level(high, low, prev_day_low)
    touch_prev_day_mid = _touch_level(high, low, prev_day_mid)

    touch_prev_asia_high = _touch_level(high, low, asia_prev_hi)
    touch_prev_asia_low = _touch_level(high, low, asia_prev_lo)
    touch_prev_london_high = _touch_level(high, low, london_prev_hi)
    touch_prev_london_low = _touch_level(high, low, london_prev_lo)
    touch_prev_ny_high = _touch_level(high, low, ny_prev_hi)
    touch_prev_ny_low = _touch_level(high, low, ny_prev_lo)

    touch_asia_or5_hi = _touch_level(high, low, asia_or5_hi)
    touch_asia_or5_lo = _touch_level(high, low, asia_or5_lo)
    touch_london_or5_hi = _touch_level(high, low, london_or5_hi)
    touch_london_or5_lo = _touch_level(high, low, london_or5_lo)
    touch_ny_or5_hi = _touch_level(high, low, ny_or5_hi)
    touch_ny_or5_lo = _touch_level(high, low, ny_or5_lo)

    touch_asia_or15_hi = _touch_level(high, low, asia_or15_hi)
    touch_asia_or15_lo = _touch_level(high, low, asia_or15_lo)
    touch_london_or15_hi = _touch_level(high, low, london_or15_hi)
    touch_london_or15_lo = _touch_level(high, low, london_or15_lo)
    touch_ny_or15_hi = _touch_level(high, low, ny_or15_hi)
    touch_ny_or15_lo = _touch_level(high, low, ny_or15_lo)

    # =========================================================================
    # EXPORT FIELDS
    # =========================================================================

    out["sc_asia_in"] = asia_in.astype(int)
    out["sc_london_in"] = london_in.astype(int)
    out["sc_ny_in"] = ny_in.astype(int)

    out["sc_asia_start"] = asia_start.astype(int)
    out["sc_london_start"] = london_start.astype(int)
    out["sc_ny_start"] = ny_start.astype(int)

    out["sc_asia_end"] = asia_end.astype(int)
    out["sc_london_end"] = london_end.astype(int)
    out["sc_ny_end"] = ny_end.astype(int)

    out["sc_asia_cur_hi"] = asia_cur_hi
    out["sc_asia_cur_lo"] = asia_cur_lo
    out["sc_prev_asia_hi"] = asia_prev_hi
    out["sc_prev_asia_lo"] = asia_prev_lo

    out["sc_london_cur_hi"] = london_cur_hi
    out["sc_london_cur_lo"] = london_cur_lo
    out["sc_prev_london_hi"] = london_prev_hi
    out["sc_prev_london_lo"] = london_prev_lo

    out["sc_ny_cur_hi"] = ny_cur_hi
    out["sc_ny_cur_lo"] = ny_cur_lo
    out["sc_prev_ny_hi"] = ny_prev_hi
    out["sc_prev_ny_lo"] = ny_prev_lo

    out["sc_prev_day_high"] = prev_day_high
    out["sc_prev_day_low"] = prev_day_low
    out["sc_prev_day_mid"] = prev_day_mid

    out["sc_asia_or5_hi"] = asia_or5_hi
    out["sc_asia_or5_lo"] = asia_or5_lo
    out["sc_asia_or15_hi"] = asia_or15_hi
    out["sc_asia_or15_lo"] = asia_or15_lo

    out["sc_london_or5_hi"] = london_or5_hi
    out["sc_london_or5_lo"] = london_or5_lo
    out["sc_london_or15_hi"] = london_or15_hi
    out["sc_london_or15_lo"] = london_or15_lo

    out["sc_ny_or5_hi"] = ny_or5_hi
    out["sc_ny_or5_lo"] = ny_or5_lo
    out["sc_ny_or15_hi"] = ny_or15_hi
    out["sc_ny_or15_lo"] = ny_or15_lo

    out["sc_touch_prev_day_high"] = touch_prev_day_high
    out["sc_touch_prev_day_low"] = touch_prev_day_low
    out["sc_touch_prev_day_mid"] = touch_prev_day_mid

    out["sc_touch_prev_asia_hi"] = touch_prev_asia_high
    out["sc_touch_prev_asia_lo"] = touch_prev_asia_low
    out["sc_touch_prev_london_hi"] = touch_prev_london_high
    out["sc_touch_prev_london_lo"] = touch_prev_london_low
    out["sc_touch_prev_ny_hi"] = touch_prev_ny_high
    out["sc_touch_prev_ny_lo"] = touch_prev_ny_low

    out["sc_touch_asia_or5_hi"] = touch_asia_or5_hi
    out["sc_touch_asia_or5_lo"] = touch_asia_or5_lo
    out["sc_touch_asia_or15_hi"] = touch_asia_or15_hi
    out["sc_touch_asia_or15_lo"] = touch_asia_or15_lo

    out["sc_touch_london_or5_hi"] = touch_london_or5_hi
    out["sc_touch_london_or5_lo"] = touch_london_or5_lo
    out["sc_touch_london_or15_hi"] = touch_london_or15_hi
    out["sc_touch_london_or15_lo"] = touch_london_or15_lo

    out["sc_touch_ny_or5_hi"] = touch_ny_or5_hi
    out["sc_touch_ny_or5_lo"] = touch_ny_or5_lo
    out["sc_touch_ny_or15_hi"] = touch_ny_or15_hi
    out["sc_touch_ny_or15_lo"] = touch_ny_or15_lo

    out["sc_session_text"] = _session_text(asia_in, london_in, ny_in)
    out["sc_day_key"] = day_key.values

    return out


def run_session_daily_engine(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    return compute_session_daily_engine(df, config=config)


# =============================================================================
# TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    n = 3 * 24 * 60  # 3 days of 1m bars
    idx = pd.date_range("2026-04-08 00:00:00", periods=n, freq="1min", tz="UTC")

    x = np.arange(n)
    base = 4800 + np.sin(x / 30.0) * 8 + np.sin(x / 200.0) * 20

    df = pd.DataFrame(
        {
            "open": base + np.sin(x / 13.0) * 0.8,
            "high": base + 1.2 + np.sin(x / 11.0) * 0.9,
            "low": base - 1.2 + np.sin(x / 17.0) * 0.9,
            "close": base + np.sin(x / 15.0) * 0.7,
            "volume": 1000 + (np.sin(x / 9.0) * 120),
        },
        index=idx,
    )

    result = run_session_daily_engine(df)

    cols = [
        "sc_asia_in",
        "sc_london_in",
        "sc_ny_in",
        "sc_asia_start",
        "sc_london_start",
        "sc_ny_start",
        "sc_prev_day_high",
        "sc_prev_day_low",
        "sc_prev_day_mid",
        "sc_prev_asia_hi",
        "sc_prev_asia_lo",
        "sc_prev_london_hi",
        "sc_prev_london_lo",
        "sc_prev_ny_hi",
        "sc_prev_ny_lo",
        "sc_asia_or5_hi",
        "sc_asia_or5_lo",
        "sc_asia_or15_hi",
        "sc_asia_or15_lo",
        "sc_london_or5_hi",
        "sc_london_or5_lo",
        "sc_london_or15_hi",
        "sc_london_or15_lo",
        "sc_ny_or5_hi",
        "sc_ny_or5_lo",
        "sc_ny_or15_hi",
        "sc_ny_or15_lo",
        "sc_touch_prev_day_high",
        "sc_touch_prev_day_low",
        "sc_touch_prev_day_mid",
        "sc_session_text",
    ]

    print("SmartChart Session & Daily Engine — Pine parity rebuild")
    print(result[cols].tail(40))