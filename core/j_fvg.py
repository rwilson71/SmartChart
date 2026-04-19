from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class FvgConfig:
    # Core
    fvg_on: bool = True
    fvg_min_gap_pct: float = 0.05
    fvg_hold_bars: int = 60
    fvg_max_zones: int = 12
    fvg_use_wicks: bool = False
    fvg_touch_mode: str = "Any Touch"  # Any Touch | Midpoint | Full Fill

    # Quality filters
    use_impulse_filter: bool = True
    use_body_filter: bool = True
    use_trend_filter: bool = True
    use_mtf_filter: bool = True

    impulse_atr_len: int = 14
    impulse_min_atr: float = 0.60
    body_min_pct: float = 0.55

    # Retest / freshness
    rt_score_on: bool = True
    fresh_bars_max: int = 25

    # MTF
    mtf_on: bool = True
    tf1: str = "15"
    tf2: str = "60"
    tf3: str = "240"
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0

    # Metadata
    symbol: str = "XAUUSD"
    timeframe: str = "1m"


DEFAULT_FVG_CONFIG: Dict[str, Any] = asdict(FvgConfig())


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FVG engine requires columns: {sorted(required)}. Missing: {sorted(missing)}")

    if len(df) < 3:
        raise ValueError("FVG engine requires at least 3 rows.")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out = out.sort_index()
        return out

    out = df.copy()
    for col in ["datetime", "timestamp", "time", "date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out = out.dropna(subset=[col]).set_index(col).sort_index()
            return out

    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _grade_txt(score: float) -> str:
    if score >= 0.85:
        return "A+"
    if score >= 0.65:
        return "B+"
    if score >= 0.45:
        return "B"
    if score >= 0.25:
        return "C"
    return "N"


def _body_pct(o: float, h: float, l: float, c: float, min_tick: float = 1e-9) -> float:
    rng = max(h - l, min_tick)
    return abs(c - o) / rng


def _gap_pct(top: float, bot: float, ref: float) -> float:
    if ref == 0:
        return 0.0
    return (abs(top - bot) / abs(ref)) * 100.0


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = _true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1.0 / max(length, 1), adjust=False).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=max(length, 1), adjust=False).mean()


def _age_score(current_bar: int, born_bar: int, fresh_bars_max: int, hold_bars: int) -> float:
    age = current_bar - born_bar
    if age <= fresh_bars_max:
        return 1.0

    denom = max(hold_bars - fresh_bars_max, 1)
    return _clamp(1.0 - ((age - fresh_bars_max) / denom), 0.0, 1.0)


def _timeframe_to_rule(tf: str) -> Optional[str]:
    mapping = {
        "1": "1min",
        "3": "3min",
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "45": "45min",
        "60": "1h",
        "120": "2h",
        "180": "3h",
        "240": "4h",
        "D": "1D",
        "W": "1W",
    }
    return mapping.get(tf)


def _resample_ohlcv(df: pd.DataFrame, tf: str) -> Optional[pd.DataFrame]:
    if not isinstance(df.index, pd.DatetimeIndex):
        return None

    rule = _timeframe_to_rule(tf)
    if not rule:
        return None

    agg: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out if not out.empty else None


def _f_mtf_dir(resampled_df: Optional[pd.DataFrame]) -> float:
    if resampled_df is None or resampled_df.empty:
        return 0.0

    e20 = _ema(resampled_df["close"], 20).iloc[-1]
    e50 = _ema(resampled_df["close"], 50).iloc[-1]
    e200 = _ema(resampled_df["close"], 200).iloc[-1]

    if pd.isna(e20) or pd.isna(e50) or pd.isna(e200):
        return 0.0
    if e20 > e50 and e50 > e200:
        return 1.0
    if e20 < e50 and e50 < e200:
        return -1.0
    return 0.0


def _touch_bull(mode: str, top: float, bot: float, mid: float, low: float, high: float) -> bool:
    if mode == "Any Touch":
        return low <= top and high >= bot
    if mode == "Midpoint":
        return low <= mid
    if mode == "Full Fill":
        return low <= bot
    return False


def _touch_bear(mode: str, top: float, bot: float, mid: float, low: float, high: float) -> bool:
    if mode == "Any Touch":
        return high >= bot and low <= top
    if mode == "Midpoint":
        return high >= mid
    if mode == "Full Fill":
        return high >= top
    return False


def _state_text(state: int) -> str:
    mapping = {
        0: "none",
        1: "single_side_active",
        2: "both_sides_active",
    }
    return mapping.get(int(state), "none")


def _dir_text(direction: int) -> str:
    mapping = {
        -1: "bearish",
        0: "neutral",
        1: "bullish",
    }
    return mapping.get(int(direction), "neutral")


def _retest_text(retest_state: int) -> str:
    mapping = {
        -1: "bear_ready",
        0: "none",
        1: "bull_ready",
        2: "both_ready",
    }
    return mapping.get(int(retest_state), "none")


# =============================================================================
# CORE ENGINE
# =============================================================================

def build_fvg_dataframe(
    df: pd.DataFrame,
    config: Optional[FvgConfig] = None,
) -> pd.DataFrame:
    cfg = config or FvgConfig()
    raw = _ensure_datetime_index(df)
    _validate_ohlcv(raw)

    out = raw.copy()

    # Core series
    out["ema20"] = _ema(out["close"], 20)
    out["ema50"] = _ema(out["close"], 50)
    out["ema200"] = _ema(out["close"], 200)
    out["atr"] = _atr(out, cfg.impulse_atr_len)

    trend_dir = np.where(
        (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"]),
        1,
        np.where(
            (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"]),
            -1,
            0,
        ),
    )
    out["trend_dir"] = trend_dir.astype(int)

    # MTF average
    mtf_avg_vals: List[float] = []
    if cfg.mtf_on and isinstance(out.index, pd.DatetimeIndex):
        tf_specs: List[Tuple[str, float]] = [
            (cfg.tf1, cfg.w1),
            (cfg.tf2, cfg.w2),
            (cfg.tf3, cfg.w3),
        ]
        weight_sum = max(cfg.w1 + cfg.w2 + cfg.w3, 0.0001)

        for ts in out.index:
            cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in out.columns else [])
            sub = out.loc[:ts, cols]
            dvals: List[Tuple[float, float]] = []
            for tf, w in tf_specs:
                rs = _resample_ohlcv(sub, tf)
                dvals.append((_f_mtf_dir(rs), w))
            mtf_avg = sum(v * w for v, w in dvals) / weight_sum
            mtf_avg_vals.append(float(mtf_avg))
    else:
        mtf_avg_vals = [0.0] * len(out)

    out["fvg_mtf_avg"] = mtf_avg_vals

    # Detection + state outputs
    numeric_cols = [
        "bull_raw", "bear_raw",
        "bull_gap_pct", "bear_gap_pct",
        "bull_fvg_found", "bear_fvg_found",
        "bull_base_strength", "bear_base_strength",
        "bull_best_score", "bear_best_score",
        "bull_active_now", "bear_active_now",
        "bull_retest_ready", "bear_retest_ready",
        "fvg_dir", "fvg_strength_score", "fvg_state", "fvg_retest_state",
        "active_bull_zone_count", "active_bear_zone_count",
        "top_active_bull_top", "top_active_bull_bot", "top_active_bull_mid",
        "top_active_bear_top", "top_active_bear_bot", "top_active_bear_mid",
        "latest_bull_zone_strength", "latest_bear_zone_strength",
    ]

    text_cols = [
        "fvg_state_text", "fvg_dir_text", "fvg_retest_text", "fvg_grade_text",
    ]

    for c in numeric_cols:
        out[c] = np.nan

    for c in text_cols:
        out[c] = pd.Series(index=out.index, dtype="object")

    bull_zones: List[Dict[str, Any]] = []
    bear_zones: List[Dict[str, Any]] = []

    for i in range(len(out)):
        row = out.iloc[i]

        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        atr_val = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
        tdir = int(row["trend_dir"])
        mtf_avg = float(row["fvg_mtf_avg"])

        # 3-candle FVG detection
        bull_raw = False
        bear_raw = False
        bull_gap_pct = 0.0
        bear_gap_pct = 0.0
        bull_fvg_found = False
        bear_fvg_found = False
        bull_base_strength = 0.0
        bear_base_strength = 0.0

        if cfg.fvg_on and i >= 2:
            o2 = float(out["open"].iloc[i - 2])
            c2 = float(out["close"].iloc[i - 2])

            o1 = float(out["open"].iloc[i - 1])
            h1 = float(out["high"].iloc[i - 1])
            l1 = float(out["low"].iloc[i - 1])
            c1 = float(out["close"].iloc[i - 1])

            if cfg.fvg_use_wicks:
                bull_gap_top = l
                bull_gap_bot = float(out["high"].iloc[i - 2])

                bear_gap_top = float(out["low"].iloc[i - 2])
                bear_gap_bot = h
            else:
                bull_gap_top = min(o, c)
                bull_gap_bot = max(o2, c2)

                bear_gap_top = min(o2, c2)
                bear_gap_bot = max(o, c)

            bull_raw = bull_gap_top > bull_gap_bot
            bear_raw = bear_gap_bot < bear_gap_top

            mid_body_pct = _body_pct(o1, h1, l1, c1)
            mid_impulse = abs(c1 - o1) / atr_val if atr_val > 0 else 0.0

            bull_gap_pct = _gap_pct(bull_gap_top, bull_gap_bot, c) if bull_raw else 0.0
            bear_gap_pct = _gap_pct(bear_gap_top, bear_gap_bot, c) if bear_raw else 0.0

            bull_gap_ok = bull_raw and bull_gap_pct >= cfg.fvg_min_gap_pct
            bear_gap_ok = bear_raw and bear_gap_pct >= cfg.fvg_min_gap_pct
            body_ok = (not cfg.use_body_filter) or (mid_body_pct >= cfg.body_min_pct)
            impulse_ok = (not cfg.use_impulse_filter) or (mid_impulse >= cfg.impulse_min_atr)
            trend_bull_ok = (not cfg.use_trend_filter) or (tdir == 1)
            trend_bear_ok = (not cfg.use_trend_filter) or (tdir == -1)
            mtf_bull_ok = (not cfg.use_mtf_filter) or (mtf_avg > 0.20)
            mtf_bear_ok = (not cfg.use_mtf_filter) or (mtf_avg < -0.20)

            bull_fvg_found = bull_gap_ok and body_ok and impulse_ok and trend_bull_ok and mtf_bull_ok
            bear_fvg_found = bear_gap_ok and body_ok and impulse_ok and trend_bear_ok and mtf_bear_ok

            # Detection strength score
            bull_gap_score = _clamp(
                bull_gap_pct / max(cfg.fvg_min_gap_pct * 2.5, 0.0001), 0.0, 1.0
            ) if bull_raw else 0.0
            bear_gap_score = _clamp(
                bear_gap_pct / max(cfg.fvg_min_gap_pct * 2.5, 0.0001), 0.0, 1.0
            ) if bear_raw else 0.0
            body_score = _clamp(mid_body_pct, 0.0, 1.0)
            impulse_score = _clamp(
                mid_impulse / max(cfg.impulse_min_atr * 1.5, 0.0001), 0.0, 1.0
            )

            bull_trend_score = 1.0 if tdir == 1 else 0.4 if tdir == 0 else 0.0
            bear_trend_score = 1.0 if tdir == -1 else 0.4 if tdir == 0 else 0.0
            bull_mtf_score = _clamp((mtf_avg + 1.0) / 2.0, 0.0, 1.0)
            bear_mtf_score = _clamp((1.0 - mtf_avg) / 2.0, 0.0, 1.0)

            bull_base_strength = _clamp(
                (bull_gap_score * 0.35)
                + (body_score * 0.20)
                + (impulse_score * 0.20)
                + (bull_trend_score * 0.10)
                + (bull_mtf_score * 0.15),
                0.0,
                1.0,
            )
            bear_base_strength = _clamp(
                (bear_gap_score * 0.35)
                + (body_score * 0.20)
                + (impulse_score * 0.20)
                + (bear_trend_score * 0.10)
                + (bear_mtf_score * 0.15),
                0.0,
                1.0,
            )

            # Add new bull zone
            if bull_fvg_found:
                z_top = bull_gap_top
                z_bot = bull_gap_bot
                bull_zones.insert(0, {
                    "top": float(z_top),
                    "bot": float(z_bot),
                    "mid": float((z_top + z_bot) * 0.5),
                    "born": i,
                    "touched": False,
                    "active": True,
                    "strength": float(bull_base_strength),
                })

            # Add new bear zone
            if bear_fvg_found:
                z_top = bear_gap_top
                z_bot = bear_gap_bot
                bear_zones.insert(0, {
                    "top": float(z_top),
                    "bot": float(z_bot),
                    "mid": float((z_top + z_bot) * 0.5),
                    "born": i,
                    "touched": False,
                    "active": True,
                    "strength": float(bear_base_strength),
                })

        # Limit arrays
        bull_zones = bull_zones[:cfg.fvg_max_zones]
        bear_zones = bear_zones[:cfg.fvg_max_zones]

        # Update bull zones
        for z in bull_zones:
            expired = (i - z["born"]) > cfg.fvg_hold_bars
            hit = _touch_bull(cfg.fvg_touch_mode, z["top"], z["bot"], z["mid"], l, h)

            if z["active"] and hit:
                z["touched"] = True
                z["active"] = False

            if expired:
                z["active"] = False

        # Update bear zones
        for z in bear_zones:
            expired = (i - z["born"]) > cfg.fvg_hold_bars
            hit = _touch_bear(cfg.fvg_touch_mode, z["top"], z["bot"], z["mid"], l, h)

            if z["active"] and hit:
                z["touched"] = True
                z["active"] = False

            if expired:
                z["active"] = False

        bull_active_now = False
        bear_active_now = False
        bull_best_score = 0.0
        bear_best_score = 0.0
        bull_retest_ready = False
        bear_retest_ready = False

        active_bull_zones = [z for z in bull_zones if z["active"]]
        active_bear_zones = [z for z in bear_zones if z["active"]]

        for z in bull_zones:
            a = bool(z["active"])
            t = bool(z["touched"])
            age_score = _age_score(i, int(z["born"]), cfg.fresh_bars_max, cfg.fvg_hold_bars) if cfg.rt_score_on else 1.0
            final_score = _clamp((float(z["strength"]) * 0.75) + (age_score * 0.25), 0.0, 1.0)

            if a:
                bull_active_now = True
                bull_best_score = max(bull_best_score, final_score)
            if a and (not t):
                bull_retest_ready = True

        for z in bear_zones:
            a = bool(z["active"])
            t = bool(z["touched"])
            age_score = _age_score(i, int(z["born"]), cfg.fresh_bars_max, cfg.fvg_hold_bars) if cfg.rt_score_on else 1.0
            final_score = _clamp((float(z["strength"]) * 0.75) + (age_score * 0.25), 0.0, 1.0)

            if a:
                bear_active_now = True
                bear_best_score = max(bear_best_score, final_score)
            if a and (not t):
                bear_retest_ready = True

        if bull_active_now and (not bear_active_now):
            fvg_dir = 1
        elif bear_active_now and (not bull_active_now):
            fvg_dir = -1
        elif bull_best_score > bear_best_score:
            fvg_dir = 1
        elif bear_best_score > bull_best_score:
            fvg_dir = -1
        else:
            fvg_dir = 0

        fvg_strength_score = max(bull_best_score, bear_best_score)

        if bull_active_now and bear_active_now:
            fvg_state = 2
        elif bull_active_now or bear_active_now:
            fvg_state = 1
        else:
            fvg_state = 0

        if bull_retest_ready and (not bear_retest_ready):
            fvg_retest_state = 1
        elif bear_retest_ready and (not bull_retest_ready):
            fvg_retest_state = -1
        elif bull_retest_ready and bear_retest_ready:
            fvg_retest_state = 2
        else:
            fvg_retest_state = 0

        top_active_bull = active_bull_zones[0] if active_bull_zones else None
        top_active_bear = active_bear_zones[0] if active_bear_zones else None
        latest_bull = bull_zones[0] if bull_zones else None
        latest_bear = bear_zones[0] if bear_zones else None

        out.iat[i, out.columns.get_loc("bull_raw")] = int(bull_raw)
        out.iat[i, out.columns.get_loc("bear_raw")] = int(bear_raw)
        out.iat[i, out.columns.get_loc("bull_gap_pct")] = bull_gap_pct
        out.iat[i, out.columns.get_loc("bear_gap_pct")] = bear_gap_pct
        out.iat[i, out.columns.get_loc("bull_fvg_found")] = int(bull_fvg_found)
        out.iat[i, out.columns.get_loc("bear_fvg_found")] = int(bear_fvg_found)
        out.iat[i, out.columns.get_loc("bull_base_strength")] = bull_base_strength
        out.iat[i, out.columns.get_loc("bear_base_strength")] = bear_base_strength
        out.iat[i, out.columns.get_loc("bull_best_score")] = bull_best_score
        out.iat[i, out.columns.get_loc("bear_best_score")] = bear_best_score
        out.iat[i, out.columns.get_loc("bull_active_now")] = int(bull_active_now)
        out.iat[i, out.columns.get_loc("bear_active_now")] = int(bear_active_now)
        out.iat[i, out.columns.get_loc("bull_retest_ready")] = int(bull_retest_ready)
        out.iat[i, out.columns.get_loc("bear_retest_ready")] = int(bear_retest_ready)
        out.iat[i, out.columns.get_loc("fvg_dir")] = fvg_dir
        out.iat[i, out.columns.get_loc("fvg_strength_score")] = fvg_strength_score
        out.iat[i, out.columns.get_loc("fvg_state")] = fvg_state
        out.iat[i, out.columns.get_loc("fvg_retest_state")] = fvg_retest_state
        out.iat[i, out.columns.get_loc("fvg_state_text")] = _state_text(fvg_state)
        out.iat[i, out.columns.get_loc("fvg_dir_text")] = _dir_text(fvg_dir)
        out.iat[i, out.columns.get_loc("fvg_retest_text")] = _retest_text(fvg_retest_state)
        out.iat[i, out.columns.get_loc("fvg_grade_text")] = _grade_txt(fvg_strength_score)
        out.iat[i, out.columns.get_loc("active_bull_zone_count")] = len(active_bull_zones)
        out.iat[i, out.columns.get_loc("active_bear_zone_count")] = len(active_bear_zones)

        out.iat[i, out.columns.get_loc("top_active_bull_top")] = top_active_bull["top"] if top_active_bull else np.nan
        out.iat[i, out.columns.get_loc("top_active_bull_bot")] = top_active_bull["bot"] if top_active_bull else np.nan
        out.iat[i, out.columns.get_loc("top_active_bull_mid")] = top_active_bull["mid"] if top_active_bull else np.nan

        out.iat[i, out.columns.get_loc("top_active_bear_top")] = top_active_bear["top"] if top_active_bear else np.nan
        out.iat[i, out.columns.get_loc("top_active_bear_bot")] = top_active_bear["bot"] if top_active_bear else np.nan
        out.iat[i, out.columns.get_loc("top_active_bear_mid")] = top_active_bear["mid"] if top_active_bear else np.nan

        out.iat[i, out.columns.get_loc("latest_bull_zone_strength")] = float(latest_bull["strength"]) if latest_bull else np.nan
        out.iat[i, out.columns.get_loc("latest_bear_zone_strength")] = float(latest_bear["strength"]) if latest_bear else np.nan

    return out


def _derive_market_bias(mtf_avg: float) -> str:
    if mtf_avg > 0.20:
        return "BULLISH"
    if mtf_avg < -0.20:
        return "BEARISH"
    return "NEUTRAL"


def _derive_fvg_state_label(
    state_value: int,
    direction_value: int,
    retest_state: int,
) -> str:
    if state_value == 2:
        return "DUAL ACTIVE FVG"
    if state_value == 1 and direction_value == 1:
        return "BULLISH ACTIVE FVG"
    if state_value == 1 and direction_value == -1:
        return "BEARISH ACTIVE FVG"
    if retest_state == 1:
        return "BULLISH RETEST READY"
    if retest_state == -1:
        return "BEARISH RETEST READY"
    if retest_state == 2:
        return "DUAL RETEST READY"
    return "NO ACTIVE FVG"


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_fvg_latest_payload(
    df: pd.DataFrame,
    config: Optional[FvgConfig] = None,
) -> Dict[str, Any]:
    cfg = config or FvgConfig()
    out = build_fvg_dataframe(df, cfg)
    last = out.iloc[-1]

    direction = int(last["fvg_dir"])
    direction_text = str(last["fvg_dir_text"]).upper()
    strength_score = float(last["fvg_strength_score"])
    mtf_avg = float(last["fvg_mtf_avg"])
    state_value = int(last["fvg_state"])
    retest_state = int(last["fvg_retest_state"])

    state_label = _derive_fvg_state_label(
        state_value=state_value,
        direction_value=direction,
        retest_state=retest_state,
    )

    payload: Dict[str, Any] = {
        "indicator": "fvg",
        "debug_version": "fvg_payload_v2",
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "engine_on": cfg.fvg_on,

        # Shared website contract
        "timestamp": out.index[-1].isoformat() if isinstance(out.index, pd.DatetimeIndex) else None,
        "state": state_label,
        "bias_signal": direction,
        "bias_label": direction_text,
        "indicator_strength": round(strength_score, 4),
        "market_bias": _derive_market_bias(mtf_avg),

        # Specialist FVG state
        "state_detail": {
            "direction": direction,
            "direction_text": direction_text,
            "strength_score": round(strength_score, 4),
            "strength_pct": round(strength_score * 100.0, 2),
            "grade": str(last["fvg_grade_text"]),
            "mtf_avg": round(mtf_avg, 4),
            "state_value": state_value,
            "state_text": str(last["fvg_state_text"]),
            "retest_state": retest_state,
            "retest_text": str(last["fvg_retest_text"]),
            "bull_best_score": round(float(last["bull_best_score"]), 4),
            "bear_best_score": round(float(last["bear_best_score"]), 4),
            "bull_active_now": bool(int(last["bull_active_now"])),
            "bear_active_now": bool(int(last["bear_active_now"])),
            "bull_retest_ready": bool(int(last["bull_retest_ready"])),
            "bear_retest_ready": bool(int(last["bear_retest_ready"])),
            "active_bull_zone_count": int(last["active_bull_zone_count"]),
            "active_bear_zone_count": int(last["active_bear_zone_count"]),
        },

        "zones": {
            "top_active_bull_zone": {
                "top": None if pd.isna(last["top_active_bull_top"]) else round(float(last["top_active_bull_top"]), 4),
                "bot": None if pd.isna(last["top_active_bull_bot"]) else round(float(last["top_active_bull_bot"]), 4),
                "mid": None if pd.isna(last["top_active_bull_mid"]) else round(float(last["top_active_bull_mid"]), 4),
            },
            "top_active_bear_zone": {
                "top": None if pd.isna(last["top_active_bear_top"]) else round(float(last["top_active_bear_top"]), 4),
                "bot": None if pd.isna(last["top_active_bear_bot"]) else round(float(last["top_active_bear_bot"]), 4),
                "mid": None if pd.isna(last["top_active_bear_mid"]) else round(float(last["top_active_bear_mid"]), 4),
            },
        },

        "latest_detection": {
            "bull_found": bool(int(last["bull_fvg_found"])),
            "bear_found": bool(int(last["bear_fvg_found"])),
            "bull_gap_pct": round(float(last["bull_gap_pct"]), 4) if pd.notna(last["bull_gap_pct"]) else 0.0,
            "bear_gap_pct": round(float(last["bear_gap_pct"]), 4) if pd.notna(last["bear_gap_pct"]) else 0.0,
            "bull_base_strength": round(float(last["bull_base_strength"]), 4) if pd.notna(last["bull_base_strength"]) else 0.0,
            "bear_base_strength": round(float(last["bear_base_strength"]), 4) if pd.notna(last["bear_base_strength"]) else 0.0,
        },

        "config": asdict(cfg),
    }

    return payload