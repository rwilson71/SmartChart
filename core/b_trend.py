from __future__ import annotations

from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from core.a_indicators import ema, atr, rsi


DEFAULT_TREND_CONFIG: Dict[str, Any] = {
    # Core
    "trend_len": 20,
    "atr_len": 100,
    "cluster_thr": 0.60,
    "enter_score_min": 0.68,
    "hold_score_min": 0.55,
    "exit_score_max": 0.42,
    "flip_confirm_bars": 2,
    "re_flip_lock_bars": 3,
    "neutral_edge_buf": 0.08,

    # EMA structure
    "ema_fast1_len": 14,
    "ema_fast2_len": 20,
    "ema_mid1_len": 33,
    "ema_mid2_len": 50,
    "ema_slow1_len": 100,
    "ema_slow2_len": 200,

    # Cluster supertrend
    "st_atr1": 7,
    "st_fac1": 1.5,
    "st_w1": 1.0,
    "st_atr2": 14,
    "st_fac2": 2.5,
    "st_w2": 1.2,
    "st_atr3": 21,
    "st_fac3": 3.5,
    "st_w3": 1.4,

    # Momentum
    "rsi_len": 14,
    "mom_len": 5,
    "mom_smooth": 5,
    "use_mom_check": True,

    # Regression
    "reg_len": 40,
    "reg_smooth": 5,

    # Regime
    "adx_len": 14,
    "atr_short_len": 14,
    "atr_long_len": 50,
    "trend_strength_min": 0.60,
    "trend_quality_min": 0.55,
    "range_adx_max": 18.0,
    "range_atr_ratio_max": 0.95,
    "expansion_atr_ratio_min": 1.20,
    "expansion_mom_min": 0.55,
    "exhaustion_rsi_long": 72.0,
    "exhaustion_rsi_short": 28.0,
    "exhaustion_path_min": 0.70,

    # MTF weights
    "mtf_w1": 0.5,
    "mtf_w2": 1.0,
    "mtf_w3": 1.25,
    "mtf_w4": 1.5,
    "mtf_w5": 1.75,
    "mtf_w6": 2.0,
    "mtf_w7": 2.25,

    # Real MTF generation
    "use_real_mtf": True,
    "mtf_timeframes": ["1", "5", "15", "60", "240", "D", "W"],

    # Score damping / cleanliness
    "score_path_weight": 0.70,
    "score_cluster_weight": 0.30,
    "tight_price_ema20_pct": 0.08,
    "tight_band_width_pct": 0.10,
    "tight_penalty": 0.20,
    "weak_slope_penalty": 0.12,
}


# ==============================================================================
# HELPERS
# ==============================================================================

def _to_float_series(series: pd.Series, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=index)
    return out.astype(float).replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def clamp_series(
    series: pd.Series | np.ndarray | float,
    low: float = 0.0,
    high: float = 1.0,
    index: Optional[pd.Index] = None,
) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=low, upper=high)
    )


def safe_div(a, b, default: float = 0.0):
    if np.isscalar(b):
        return a / b if b != 0 else default

    a_series = a if isinstance(a, pd.Series) else pd.Series(a)
    b_series = b if isinstance(b, pd.Series) else pd.Series(b, index=a_series.index)

    a_series = pd.to_numeric(a_series, errors="coerce")
    b_series = pd.to_numeric(b_series, errors="coerce")

    out = pd.Series(default, index=a_series.index, dtype=float)
    mask = b_series.notna() & (b_series != 0)
    out.loc[mask] = a_series.loc[mask] / b_series.loc[mask]
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(max(1, int(length)), min_periods=1).mean()


def stdev(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(max(1, int(length)), min_periods=1).std(ddof=0).fillna(0.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(0.0).astype(float)


def rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return pd.to_numeric(series, errors="coerce").fillna(0.0).ewm(alpha=1 / length, adjust=False).mean()


def linreg_series(series: pd.Series, length: int) -> pd.Series:
    length = max(2, int(length))
    x = np.arange(length, dtype=float)

    def _calc(y: np.ndarray) -> float:
        if len(y) < length:
            return np.nan
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0] * x[-1] + coeffs[1])

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)


def rolling_corr_with_index(series: pd.Series, length: int) -> pd.Series:
    length = max(2, int(length))
    idx = pd.Series(np.arange(len(series)), index=series.index, dtype=float)
    return series.rolling(length, min_periods=length).corr(idx).fillna(0.0)


def kama_like(close: pd.Series, trend_len: int) -> pd.Series:
    trend_len = max(1, int(trend_len))

    close_n = close.shift(trend_len).fillna(close)
    change = (close - close_n).abs()
    noise = sma(close.diff().abs().fillna(0.0), trend_len) * trend_len
    er = safe_div(change, noise, default=0.0)

    fast_alpha = 2.0 / 3.0
    slow_alpha = 2.0 / 31.0
    alpha = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    avt = pd.Series(index=close.index, dtype=float)
    avt.iloc[0] = float(close.iloc[0])

    for i in range(1, len(close)):
        a = float(alpha.iloc[i]) if pd.notna(alpha.iloc[i]) else 0.0
        avt.iloc[i] = a * float(close.iloc[i]) + (1.0 - a) * float(avt.iloc[i - 1])

    return avt.ffill().fillna(close)


def supertrend_line_dir(
    src: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_len: int,
    fac: float,
):
    atr_val = pd.to_numeric(
        atr(high, low, close, atr_len),
        errors="coerce",
    ).fillna(0.0)

    ub0 = src + fac * atr_val
    lb0 = src - fac * atr_val

    ub = pd.Series(index=src.index, dtype=float)
    lb = pd.Series(index=src.index, dtype=float)
    direction = pd.Series(index=src.index, dtype=float)

    for i in range(len(src)):
        if i == 0:
            ub.iloc[i] = ub0.iloc[i]
            lb.iloc[i] = lb0.iloc[i]
            direction.iloc[i] = 1.0
            continue

        prev_ub = ub.iloc[i - 1]
        prev_lb = lb.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        prev_src = src.iloc[i - 1]

        cur_ub0 = ub0.iloc[i]
        cur_lb0 = lb0.iloc[i]
        cur_src = src.iloc[i]

        ub.iloc[i] = cur_ub0 if (cur_ub0 < prev_ub or prev_src > prev_ub) else prev_ub
        lb.iloc[i] = cur_lb0 if (cur_lb0 > prev_lb or prev_src < prev_lb) else prev_lb

        if prev_dir == -1.0 and cur_src > prev_ub:
            direction.iloc[i] = 1.0
        elif prev_dir == 1.0 and cur_src < prev_lb:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = prev_dir

    line = pd.Series(np.where(direction == 1.0, lb, ub), index=src.index, dtype=float)
    return line.ffill(), direction.fillna(0.0)


def compute_adx(df: pd.DataFrame, adx_len: int):
    high = df["high"]
    low = df["low"]

    up_move = high.diff().fillna(0.0)
    down_move = (-low.diff()).fillna(0.0)

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    tr = true_range(df)
    trur = rma(tr, adx_len)

    plus_di = safe_div(100 * rma(plus_dm, adx_len), trur, default=0.0)
    minus_di = safe_div(100 * rma(minus_dm, adx_len), trur, default=0.0)

    dx = safe_div(100 * (plus_di - minus_di).abs(), plus_di + minus_di, default=0.0)
    adx_value = rma(dx, adx_len)

    return plus_di, minus_di, adx_value.fillna(0.0)


def _tv_tf_to_pandas_freq(tf: str) -> str:
    tf = str(tf).strip().upper()

    if tf == "D":
        return "1D"
    if tf == "W":
        return "1W"
    if tf == "M":
        return "1MS"

    if tf.isdigit():
        minutes = int(tf)
        return f"{minutes}min"

    raise ValueError(f"Unsupported timeframe format: {tf}")


def _infer_base_minutes(index: pd.DatetimeIndex) -> Optional[float]:
    if len(index) < 2:
        return None
    diffs = pd.Series(index[1:] - index[:-1]).dt.total_seconds() / 60.0
    diffs = diffs.replace([np.inf, -np.inf], np.nan).dropna()
    if diffs.empty:
        return None
    return float(diffs.mode().iloc[0])


def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Real MTF resampling requires a DatetimeIndex.")

    work = df.sort_index().copy()

    out = (
        work[["open", "high", "low", "close"]]
        .resample(freq, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna(subset=["open", "high", "low", "close"], how="any")
    )

    return out


def _get_mtf_states(out: pd.DataFrame, trend_state: pd.Series) -> Dict[str, pd.Series]:
    result: Dict[str, pd.Series] = {}
    for i in range(1, 8):
        col = f"mtf_trend_state_t{i}"
        if col in out.columns:
            result[f"t{i}"] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            result[f"t{i}"] = pd.to_numeric(trend_state, errors="coerce").fillna(0).astype(int)
    return result


def _compute_trend_state_core(
    close: pd.Series,
    ema20: pd.Series,
    band_lo: pd.Series,
    band_hi: pd.Series,
    cluster_bias: pd.Series,
    bull_score: pd.Series,
    bear_score: pd.Series,
    trend_path_quality: pd.Series,
    cfg: Dict[str, Any],
) -> pd.Series:
    idx = close.index
    trend_state = pd.Series(index=idx, dtype=int)

    last_flip_bar = -10000
    confirm_bars = max(1, int(cfg["flip_confirm_bars"]))
    re_flip_lock_bars = int(cfg["re_flip_lock_bars"])
    neutral_edge_buf = float(cfg["neutral_edge_buf"])
    enter_score_min = float(cfg["enter_score_min"])
    hold_score_min = float(cfg["hold_score_min"])
    exit_score_max = float(cfg["exit_score_max"])

    score_edge = bull_score - bear_score

    for i in range(len(close)):
        prev_trend = int(trend_state.iloc[i - 1]) if i > 0 and pd.notna(trend_state.iloc[i - 1]) else 0
        next_trend = prev_trend

        enter_bull_raw = bool(
            bull_score.iloc[i] >= enter_score_min
            and score_edge.iloc[i] > 0.18
            and trend_path_quality.iloc[i] >= 0.45
            and cluster_bias.iloc[i] >= 0.60
        )
        enter_bear_raw = bool(
            bear_score.iloc[i] >= enter_score_min
            and score_edge.iloc[i] < -0.18
            and trend_path_quality.iloc[i] >= 0.45
            and cluster_bias.iloc[i] <= -0.60
        )

        if confirm_bars <= 1:
            bull_confirm = enter_bull_raw
            bear_confirm = enter_bear_raw
        else:
            bull_confirm = False
            bear_confirm = False
            if i >= confirm_bars - 1:
                sl = slice(i - confirm_bars + 1, i + 1)

                bull_confirm = bool(
                    (
                        (bull_score.iloc[sl] >= enter_score_min)
                        & ((bull_score.iloc[sl] - bear_score.iloc[sl]) > 0.18)
                        & (trend_path_quality.iloc[sl] >= 0.45)
                        & (cluster_bias.iloc[sl] >= 0.60)
                    ).all()
                )
                bear_confirm = bool(
                    (
                        (bear_score.iloc[sl] >= enter_score_min)
                        & ((bull_score.iloc[sl] - bear_score.iloc[sl]) < -0.18)
                        & (trend_path_quality.iloc[sl] >= 0.45)
                        & (cluster_bias.iloc[sl] <= -0.60)
                    ).all()
                )

        enter_bull = enter_bull_raw and bull_confirm
        enter_bear = enter_bear_raw and bear_confirm

        hold_bull = bool(
            bull_score.iloc[i] >= hold_score_min
            and score_edge.iloc[i] > -neutral_edge_buf
            and cluster_bias.iloc[i] > 0.0
        )
        hold_bear = bool(
            bear_score.iloc[i] >= hold_score_min
            and score_edge.iloc[i] < neutral_edge_buf
            and cluster_bias.iloc[i] < 0.0
        )

        exit_bull = bool(
            bull_score.iloc[i] <= exit_score_max
            or cluster_bias.iloc[i] < 0.20
            or (close.iloc[i] < ema20.iloc[i] and close.iloc[i] < band_lo.iloc[i])
        )
        exit_bear = bool(
            bear_score.iloc[i] <= exit_score_max
            or cluster_bias.iloc[i] > -0.20
            or (close.iloc[i] > ema20.iloc[i] and close.iloc[i] > band_hi.iloc[i])
        )

        bars_since_flip = i - last_flip_bar
        bull_flip_allowed = bars_since_flip >= re_flip_lock_bars or prev_trend != -1
        bear_flip_allowed = bars_since_flip >= re_flip_lock_bars or prev_trend != 1

        if prev_trend == 0:
            next_trend = 1 if enter_bull else -1 if enter_bear else 0
        elif prev_trend == 1:
            next_trend = -1 if (enter_bear and bear_flip_allowed) else 1 if (hold_bull and not exit_bull) else 0
        elif prev_trend == -1:
            next_trend = 1 if (enter_bull and bull_flip_allowed) else -1 if (hold_bear and not exit_bear) else 0

        if next_trend != prev_trend:
            last_flip_bar = i

        trend_state.iloc[i] = next_trend

    return trend_state.fillna(0).astype(int)


def _build_real_mtf_states(
    base_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, pd.Series]:
    result: Dict[str, pd.Series] = {}
    base_index = base_df.index
    mtf_list: List[str] = list(cfg.get("mtf_timeframes", ["1", "5", "15", "60", "240", "D", "W"]))

    if len(mtf_list) != 7:
        raise ValueError("cfg['mtf_timeframes'] must contain exactly 7 timeframes.")

    if not isinstance(base_index, pd.DatetimeIndex):
        zero = pd.Series(0, index=base_index, dtype=int)
        for i in range(1, 8):
            result[f"t{i}"] = zero.copy()
        return result

    base_minutes = _infer_base_minutes(base_index)

    for i, tf in enumerate(mtf_list, start=1):
        key = f"t{i}"

        try:
            if tf.isdigit() and base_minutes is not None:
                tf_minutes = float(int(tf))
                if tf_minutes < base_minutes:
                    result[key] = pd.Series(0, index=base_index, dtype=int)
                    continue

            freq = _tv_tf_to_pandas_freq(tf)
            htf_df = _resample_ohlc(base_df[["open", "high", "low", "close"]], freq)

            if htf_df.empty:
                result[key] = pd.Series(0, index=base_index, dtype=int)
                continue

            child_cfg = {**cfg, "use_real_mtf": False}
            htf_out = compute_trend_engine(htf_df, config=child_cfg)
            htf_state = pd.to_numeric(htf_out["sc_trend_state"], errors="coerce").fillna(0).astype(int)

            mapped = htf_state.reindex(base_index, method="ffill").fillna(0).astype(int)
            result[key] = mapped

        except Exception:
            result[key] = pd.Series(0, index=base_index, dtype=int)

    return result


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _state_label(state: int) -> str:
    if state > 0:
        return "Bullish"
    if state < 0:
        return "Bearish"
    return "Neutral"


def _stage_label(stage: int) -> str:
    mapping = {
        3: "Weakening Bull",
        2: "Strong Bull",
        1: "Bull",
        0: "Neutral",
        -1: "Bear",
        -2: "Strong Bear",
        -3: "Weakening Bear",
    }
    return mapping.get(int(stage), "Neutral")


def _color_from_state(state: int) -> str:
    if state > 0:
        return "#22c55e"
    if state < 0:
        return "#ef4444"
    return "#9ca3af"


def build_trend_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = compute_trend_engine(df, config=config)

    if out.empty:
        return {
            "debug_version": "trend_payload_v3",
            "status": "empty",
        }

    last = out.iloc[-1]
    ts = out.index[-1]

    trend_state = _safe_int(last.get("sc_trend_state", 0))
    trend_stage = _safe_int(last.get("sc_trend_stage", 0))
    regime_text = str(last.get("sc_regime_text", "Neutral"))
    quality_text = str(last.get("trend_quality_text", "Neutral"))

    payload: Dict[str, Any] = {
        "debug_version": "trend_payload_v3",
        "status": "ok",
        "symbol": "XAUUSD",
        "timestamp": str(ts),

        "direction": _state_label(trend_state),
        "direction_value": trend_state,
        "direction_color": _color_from_state(trend_state),

        "channel_state": _state_label(trend_state),
        "channel_label": _state_label(trend_state),
        "channel_color": _color_from_state(trend_state),

        "stage": trend_stage,
        "stage_label": _stage_label(trend_stage),

        "regime": regime_text,
        "quality": quality_text,

        "trend_strength": round(_safe_float(last.get("sc_trend_strength", 0.0)), 4),
        "continuation_quality": round(_safe_float(last.get("sc_continuation_quality", 0.0)), 4),
        "structure_quality": round(_safe_float(last.get("sc_structure_quality", 0.0)), 4),
        "momentum_quality": round(_safe_float(last.get("sc_momentum_quality", 0.0)), 4),
        "path_quality": round(_safe_float(last.get("sc_path_quality", 0.0)), 4),
        "cluster_strength": round(_safe_float(last.get("sc_cluster_strength", 0.0)), 4),
        "decay": round(_safe_float(last.get("sc_trend_decay", 0.0)), 4),

        "trend_weakening": bool(_safe_int(last.get("sc_trend_stage", 0)) in (3, -3)),
        "mtf_score": round(_safe_float(last.get("sc_mtf_score", 0.0)), 4),
        "mtf_bias": _safe_int(last.get("sc_mtf_bias", 0)),
        "mtf_alignment": _safe_int(last.get("sc_mtf_alignment", 0)),

        "bull_score": round(_safe_float(last.get("bull_score", 0.0)), 4),
        "bear_score": round(_safe_float(last.get("bear_score", 0.0)), 4),
        "score_edge": round(_safe_float(last.get("score_edge", 0.0)), 4),
        "cluster_bias": round(_safe_float(last.get("cluster_bias", 0.0)), 4),

        "adx_value": round(_safe_float(last.get("adx_value", 0.0)), 4),
        "atr_ratio": round(_safe_float(last.get("atr_ratio", 0.0)), 4),
        "rsi": round(_safe_float(last.get("rsi", 0.0)), 4),
        "mom": round(_safe_float(last.get("mom", 0.0)), 4),

        "price_to_ema20_pct": round(_safe_float(last.get("price_to_ema20_pct", 0.0)), 4),
        "e20e200_pct": round(_safe_float(last.get("e20e200_pct", 0.0)), 4),
        "band_width_pct": round(_safe_float(last.get("band_width_pct", 0.0)), 4),

        "ema20": round(_safe_float(last.get("ema20", 0.0)), 4),
        "ema50": round(_safe_float(last.get("ema50", 0.0)), 4),
        "ema200": round(_safe_float(last.get("ema200", 0.0)), 4),
        "avt": round(_safe_float(last.get("avt", 0.0)), 4),

        "trend_long_signal": _safe_int(last.get("trend_long_signal", 0)),
        "trend_short_signal": _safe_int(last.get("trend_short_signal", 0)),
        "bull_weak_signal": _safe_int(last.get("bull_weak_signal", 0)),
        "bear_weak_signal": _safe_int(last.get("bear_weak_signal", 0)),

        "mtf_states": {
            "t1": _safe_int(last.get("mtf_trend_state_t1", 0)),
            "t2": _safe_int(last.get("mtf_trend_state_t2", 0)),
            "t3": _safe_int(last.get("mtf_trend_state_t3", 0)),
            "t4": _safe_int(last.get("mtf_trend_state_t4", 0)),
            "t5": _safe_int(last.get("mtf_trend_state_t5", 0)),
            "t6": _safe_int(last.get("mtf_trend_state_t6", 0)),
            "t7": _safe_int(last.get("mtf_trend_state_t7", 0)),
        },
    }

    return payload


# ==============================================================================
# MAIN ENGINE
# ==============================================================================

def compute_trend_engine(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    cfg = {**DEFAULT_TREND_CONFIG, **(config or {})}

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    idx = out.index

    open_ = _to_float_series(out["open"], idx)
    high = _to_float_series(out["high"], idx)
    low = _to_float_series(out["low"], idx)
    close = _to_float_series(out["close"], idx)

    calc_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )

    hlc3 = (high + low + close) / 3.0

    ema14 = pd.to_numeric(ema(close, cfg["ema_fast1_len"]), errors="coerce").fillna(close)
    ema20 = pd.to_numeric(ema(close, cfg["ema_fast2_len"]), errors="coerce").fillna(close)
    ema33 = pd.to_numeric(ema(close, cfg["ema_mid1_len"]), errors="coerce").fillna(close)
    ema50 = pd.to_numeric(ema(close, cfg["ema_mid2_len"]), errors="coerce").fillna(close)
    ema100 = pd.to_numeric(ema(close, cfg["ema_slow1_len"]), errors="coerce").fillna(close)
    ema200 = pd.to_numeric(ema(close, cfg["ema_slow2_len"]), errors="coerce").fillna(close)

    band_lo = pd.concat([ema33, ema50], axis=1).min(axis=1)
    band_hi = pd.concat([ema33, ema50], axis=1).max(axis=1)

    e20e200_pct = safe_div((ema20 - ema200) * 100.0, ema200, default=0.0)
    price_to_ema20_pct = safe_div((close - ema20) * 100.0, ema20, default=0.0)
    band_width_pct = safe_div((band_hi - band_lo) * 100.0, close.abs().replace(0.0, np.nan), default=0.0)

    avt = kama_like(close, cfg["trend_len"])

    st1, d1 = supertrend_line_dir(hlc3, high, low, close, cfg["st_atr1"], cfg["st_fac1"])
    st2, d2 = supertrend_line_dir(hlc3, high, low, close, cfg["st_atr2"], cfg["st_fac2"])
    st3, d3 = supertrend_line_dir(hlc3, high, low, close, cfg["st_atr3"], cfg["st_fac3"])

    st_w1 = float(cfg["st_w1"])
    st_w2 = float(cfg["st_w2"])
    st_w3 = float(cfg["st_w3"])
    w_sum = st_w1 + st_w2 + st_w3

    bull_w = pd.Series(
        np.where(d1 > 0, st_w1, 0.0) +
        np.where(d2 > 0, st_w2, 0.0) +
        np.where(d3 > 0, st_w3, 0.0),
        index=idx,
        dtype=float,
    )
    bear_w = pd.Series(
        np.where(d1 < 0, st_w1, 0.0) +
        np.where(d2 < 0, st_w2, 0.0) +
        np.where(d3 < 0, st_w3, 0.0),
        index=idx,
        dtype=float,
    )

    bull_consensus = bull_w / w_sum if w_sum != 0 else pd.Series(0.0, index=idx, dtype=float)
    bear_consensus = bear_w / w_sum if w_sum != 0 else pd.Series(0.0, index=idx, dtype=float)

    trend_stack_score = (
        (ema20 > ema33).astype(float) * 0.22 +
        (ema33 > ema50).astype(float) * 0.18 +
        (ema50 > ema100).astype(float) * 0.15 +
        (ema100 > ema200).astype(float) * 0.15 +
        (close > ema20).astype(float) * 0.12 +
        (close > ema50).astype(float) * 0.10 +
        (close > avt).astype(float) * 0.08
    )

    bear_stack_score = (
        (ema20 < ema33).astype(float) * 0.22 +
        (ema33 < ema50).astype(float) * 0.18 +
        (ema50 < ema100).astype(float) * 0.15 +
        (ema100 < ema200).astype(float) * 0.15 +
        (close < ema20).astype(float) * 0.12 +
        (close < ema50).astype(float) * 0.10 +
        (close < avt).astype(float) * 0.08
    )

    raw_cluster_bias = (
        (bull_consensus - bear_consensus) * 0.60 +
        (trend_stack_score - bear_stack_score) * 0.40
    )

    cluster_bias = clamp_series(raw_cluster_bias, low=-1.0, high=1.0, index=idx)
    cluster_strength = cluster_bias.abs()

    bull_line_num = (
        np.where(d1 > 0, st1 * st_w1, 0.0) +
        np.where(d2 > 0, st2 * st_w2, 0.0) +
        np.where(d3 > 0, st3 * st_w3, 0.0)
    )
    bear_line_num = (
        np.where(d1 < 0, st1 * st_w1, 0.0) +
        np.where(d2 < 0, st2 * st_w2, 0.0) +
        np.where(d3 < 0, st3 * st_w3, 0.0)
    )

    bull_line = pd.Series(np.where(bull_w > 0, bull_line_num / bull_w, np.nan), index=idx, dtype=float)
    bear_line = pd.Series(np.where(bear_w > 0, bear_line_num / bear_w, np.nan), index=idx, dtype=float)
    cluster_line = pd.Series(
        np.where(cluster_bias > 0, bull_line, np.where(cluster_bias < 0, bear_line, np.nan)),
        index=idx,
        dtype=float,
    )

    cluster_bull_ok = cluster_bias >= float(cfg["cluster_thr"])
    cluster_bear_ok = cluster_bias <= -float(cfg["cluster_thr"])

    rsi_val = pd.to_numeric(rsi(close, cfg["rsi_len"]), errors="coerce").fillna(50.0)
    mom_raw = (close - close.shift(int(cfg["mom_len"]))).fillna(0.0)
    mom = pd.to_numeric(ema(mom_raw, cfg["mom_smooth"]), errors="coerce").fillna(0.0)

    bull_mom_ok = (rsi_val > 50) & (mom > 0)
    bear_mom_ok = (rsi_val < 50) & (mom < 0)

    if bool(cfg["use_mom_check"]):
        mom_assist_long = bull_mom_ok
        mom_assist_short = bear_mom_ok
    else:
        mom_assist_long = pd.Series(True, index=idx)
        mom_assist_short = pd.Series(True, index=idx)

    reg_line_raw = linreg_series(close, cfg["reg_len"])
    reg_line = pd.to_numeric(ema(reg_line_raw.bfill().fillna(close), cfg["reg_smooth"]), errors="coerce").fillna(close)

    reg_slope = reg_line - reg_line.shift(1).fillna(reg_line)
    reg_corr = rolling_corr_with_index(close, cfg["reg_len"]).abs().fillna(0.0)
    regression_dir = pd.Series(np.where(reg_slope > 0, 1, np.where(reg_slope < 0, -1, 0)), index=idx, dtype=int)

    atr_trend = pd.to_numeric(
        atr(high, low, close, cfg["atr_len"]),
        errors="coerce",
    ).fillna(0.0)

    reg_residual = (close - reg_line).fillna(0.0)

    reg_smoothness = 1.0 - np.minimum(
        1.0,
        safe_div(
            stdev(reg_residual, cfg["reg_len"]),
            pd.Series(np.maximum(atr_trend, 1e-9), index=idx),
            default=0.0,
        ),
    )
    reg_smoothness = pd.Series(reg_smoothness, index=idx, dtype=float).fillna(0.0)

    reg_curvature_raw = reg_slope - reg_slope.shift(1).fillna(0.0)
    reg_curvature = pd.Series(
        np.where(reg_curvature_raw > 0, 1, np.where(reg_curvature_raw < 0, -1, 0)),
        index=idx,
        dtype=int,
    )

    trend_path_quality = clamp_series((reg_corr * 0.6) + (reg_smoothness * 0.4), index=idx)

    dist_from_avt = close - avt
    avt_disp_score = clamp_series(
        safe_div(dist_from_avt.abs(), pd.Series(np.maximum(atr_trend, 1e-9), index=idx), default=0.0),
        index=idx,
    )

    bull_struct_fast = (ema14 > ema20) & (ema20 > ema33)
    bear_struct_fast = (ema14 < ema20) & (ema20 < ema33)

    bull_struct_core = (ema20 > ema33) & (ema33 >= ema50)
    bear_struct_core = (ema20 < ema33) & (ema33 <= ema50)

    bull_struct_full = (ema20 > ema33) & (ema33 > ema50) & (ema50 > ema100) & (ema100 > ema200)
    bear_struct_full = (ema20 < ema33) & (ema33 < ema50) & (ema50 < ema100) & (ema100 < ema200)

    bull_price_accept = (close > avt) & (close > ema20) & (close >= band_lo)
    bear_price_accept = (close < avt) & (close < ema20) & (close <= band_hi)

    bull_mature = (close > ema200) & (e20e200_pct > 0)
    bear_mature = (close < ema200) & (e20e200_pct < 0)

    bull_path_agree = (regression_dir == 1) & (trend_path_quality >= 0.50)
    bear_path_agree = (regression_dir == -1) & (trend_path_quality >= 0.50)

    bull_base_score = (
        (bull_price_accept.astype(float) * 0.16) +
        (bull_struct_fast.astype(float) * 0.10) +
        (bull_struct_core.astype(float) * 0.16) +
        (bull_struct_full.astype(float) * 0.14) +
        (bull_mature.astype(float) * 0.10) +
        (cluster_bull_ok.astype(float) * 0.10) +
        (bull_path_agree.astype(float) * 0.08) +
        (mom_assist_long.astype(float) * 0.06)
    )

    bear_base_score = (
        (bear_price_accept.astype(float) * 0.16) +
        (bear_struct_fast.astype(float) * 0.10) +
        (bear_struct_core.astype(float) * 0.16) +
        (bear_struct_full.astype(float) * 0.14) +
        (bear_mature.astype(float) * 0.10) +
        (cluster_bear_ok.astype(float) * 0.10) +
        (bear_path_agree.astype(float) * 0.08) +
        (mom_assist_short.astype(float) * 0.06)
    )

    path_cluster_bull_damper = (
        trend_path_quality * float(cfg["score_path_weight"]) +
        cluster_strength * float(cfg["score_cluster_weight"])
    )
    path_cluster_bear_damper = path_cluster_bull_damper.copy()

    tight_price_penalty = (price_to_ema20_pct.abs() < float(cfg["tight_price_ema20_pct"])).astype(float) * float(cfg["tight_penalty"])
    tight_band_penalty = (band_width_pct < float(cfg["tight_band_width_pct"])).astype(float) * float(cfg["tight_penalty"])
    weak_slope_penalty = (reg_slope.abs() < atr_trend * 0.02).astype(float) * float(cfg["weak_slope_penalty"])

    total_penalty = clamp_series(tight_price_penalty + tight_band_penalty + weak_slope_penalty, low=0.0, high=0.45, index=idx)

    bull_score = clamp_series(bull_base_score * path_cluster_bull_damper * (1.0 - total_penalty), index=idx)
    bear_score = clamp_series(bear_base_score * path_cluster_bear_damper * (1.0 - total_penalty), index=idx)
    score_edge = bull_score - bear_score

    structure_quality = pd.Series(0.0, index=idx, dtype=float)
    structure_quality += ((bull_struct_core | bear_struct_core).astype(float) * 0.35)
    structure_quality += ((bull_struct_fast | bear_struct_fast).astype(float) * 0.20)
    structure_quality += ((bull_price_accept | bear_price_accept).astype(float) * 0.20)
    structure_quality += ((bull_struct_full | bear_struct_full).astype(float) * 0.25)
    structure_quality = clamp_series(structure_quality, index=idx)

    mom_distance = mom.abs()
    mom_distance_avg = pd.to_numeric(
        ema(mom_distance, max(3, int(cfg["mom_smooth"]) * 2)),
        errors="coerce",
    ).replace(0, np.nan)

    mom_power = clamp_series(safe_div(mom_distance, mom_distance_avg, default=0.0), index=idx)

    rsi_strength = pd.Series(
        np.where(
            bull_score > bear_score,
            np.maximum(0.0, np.minimum(1.0, (rsi_val - 50.0) / 20.0)),
            np.where(
                bear_score > bull_score,
                np.maximum(0.0, np.minimum(1.0, (50.0 - rsi_val) / 20.0)),
                0.0,
            ),
        ),
        index=idx,
        dtype=float,
    )

    momentum_quality = clamp_series((mom_power * 0.6) + (rsi_strength * 0.4), index=idx)

    trend_state = _compute_trend_state_core(
        close=close,
        ema20=ema20,
        band_lo=band_lo,
        band_hi=band_hi,
        cluster_bias=cluster_bias,
        bull_score=bull_score,
        bear_score=bear_score,
        trend_path_quality=trend_path_quality,
        cfg=cfg,
    )
    trend_dir = trend_state.copy()

    base_strength = (
        (cluster_strength * 0.30) +
        (avt_disp_score * 0.20) +
        (structure_quality * 0.20) +
        (momentum_quality * 0.15) +
        (trend_path_quality * 0.15)
    )
    trend_strength = clamp_series(base_strength, index=idx)

    stage_score = pd.Series(
        np.where(trend_state == 1, bull_score, np.where(trend_state == -1, bear_score, 0.0)),
        index=idx,
        dtype=float,
    )

    continuation_quality = clamp_series(
        (structure_quality * 0.35) +
        (momentum_quality * 0.25) +
        (trend_path_quality * 0.25) +
        (stage_score * 0.15),
        index=idx,
    )

    bull_weak_factors = (
        np.where((trend_state == 1) & (rsi_val < 55), 0.20, 0.0) +
        np.where((trend_state == 1) & (mom <= 0), 0.20, 0.0) +
        np.where((trend_state == 1) & (reg_curvature_raw < 0), 0.15, 0.0) +
        np.where((trend_state == 1) & (cluster_strength < 0.35), 0.15, 0.0) +
        np.where((trend_state == 1) & (close < ema20), 0.15, 0.0) +
        np.where((trend_state == 1) & (close < avt), 0.15, 0.0)
    )

    bear_weak_factors = (
        np.where((trend_state == -1) & (rsi_val > 45), 0.20, 0.0) +
        np.where((trend_state == -1) & (mom >= 0), 0.20, 0.0) +
        np.where((trend_state == -1) & (reg_curvature_raw > 0), 0.15, 0.0) +
        np.where((trend_state == -1) & (cluster_strength < 0.35), 0.15, 0.0) +
        np.where((trend_state == -1) & (close > ema20), 0.15, 0.0) +
        np.where((trend_state == -1) & (close > avt), 0.15, 0.0)
    )

    trend_decay = clamp_series(
        pd.Series(
            np.where(trend_state == 1, bull_weak_factors, np.where(trend_state == -1, bear_weak_factors, 0.0)),
            index=idx,
            dtype=float,
        ),
        index=idx,
    )

    trend_weakening = (trend_state != 0) & (trend_decay >= 0.45)

    trend_stage = pd.Series(
        np.where(
            (trend_state == 1) & trend_weakening, 3,
            np.where(
                (trend_state == 1) & (bull_score >= 0.82) & (close > ema200) & (cluster_strength >= 0.65), 2,
                np.where(
                    trend_state == 1, 1,
                    np.where(
                        (trend_state == -1) & trend_weakening, -3,
                        np.where(
                            (trend_state == -1) & (bear_score >= 0.82) & (close < ema200) & (cluster_strength >= 0.65), -2,
                            np.where(trend_state == -1, -1, 0),
                        ),
                    ),
                ),
            ),
        ),
        index=idx,
        dtype=int,
    )

    trend_quality_state = pd.Series(
        np.where(
            (trend_strength >= 0.80) & (continuation_quality >= 0.75), 3,
            np.where(
                (trend_strength >= 0.60) & (continuation_quality >= 0.55), 2,
                np.where(trend_strength >= 0.40, 1, 0),
            ),
        ),
        index=idx,
        dtype=int,
    )

    trend_quality_text = pd.Series(
        np.where(
            trend_quality_state == 3, "Strong",
            np.where(trend_quality_state == 2, "Clean", np.where(trend_quality_state == 1, "Weak", "Neutral")),
        ),
        index=idx,
    )

    trend_long_signal = ((trend_state == 1) & (trend_state.shift(1).fillna(0) <= 0)).astype(int)
    trend_short_signal = ((trend_state == -1) & (trend_state.shift(1).fillna(0) >= 0)).astype(int)
    bull_weak_signal = ((trend_stage == 3) & (trend_stage.shift(1).fillna(0) != 3)).astype(int)
    bear_weak_signal = ((trend_stage == -3) & (trend_stage.shift(1).fillna(0) != -3)).astype(int)

    if bool(cfg.get("use_real_mtf", True)):
        mtf = _build_real_mtf_states(calc_df, cfg)
    else:
        mtf = _get_mtf_states(out, trend_state)

    mtf_sum = (
        mtf["t1"] * float(cfg["mtf_w1"]) +
        mtf["t2"] * float(cfg["mtf_w2"]) +
        mtf["t3"] * float(cfg["mtf_w3"]) +
        mtf["t4"] * float(cfg["mtf_w4"]) +
        mtf["t5"] * float(cfg["mtf_w5"]) +
        mtf["t6"] * float(cfg["mtf_w6"]) +
        mtf["t7"] * float(cfg["mtf_w7"])
    )
    mtf_weight_total = (
        float(cfg["mtf_w1"]) + float(cfg["mtf_w2"]) + float(cfg["mtf_w3"]) +
        float(cfg["mtf_w4"]) + float(cfg["mtf_w5"]) + float(cfg["mtf_w6"]) + float(cfg["mtf_w7"])
    )

    mtf_score = mtf_sum / mtf_weight_total if mtf_weight_total != 0 else pd.Series(0.0, index=idx, dtype=float)
    mtf_bias = pd.Series(
        np.where(mtf_score > 0.25, 1, np.where(mtf_score < -0.25, -1, 0)),
        index=idx,
        dtype=int,
    )
    mtf_alignment = (trend_state == mtf_bias).astype(int)

    _, _, adx_value = compute_adx(calc_df, cfg["adx_len"])

    atr_short = pd.to_numeric(
        atr(high, low, close, cfg["atr_short_len"]),
        errors="coerce",
    ).fillna(0.0)

    atr_long_raw = pd.to_numeric(
        atr(high, low, close, cfg["atr_long_len"]),
        errors="coerce",
    ).fillna(0.0)
    atr_long = pd.to_numeric(ema(atr_long_raw, cfg["atr_long_len"]), errors="coerce").replace(0, np.nan)
    atr_ratio = safe_div(atr_short, atr_long, default=1.0)

    trend_impulse_mode = dist_from_avt.abs() > (0.75 * atr_trend)

    regime_trend = (
        (trend_state != 0) &
        (trend_strength >= float(cfg["trend_strength_min"])) &
        (continuation_quality >= float(cfg["trend_quality_min"])) &
        (adx_value > float(cfg["range_adx_max"])) &
        (mtf_alignment == 1) &
        (trend_decay < 0.55) &
        (cluster_strength >= 0.55)
    )

    regime_range = (
        (trend_state == 0) &
        (adx_value <= float(cfg["range_adx_max"])) &
        (atr_ratio <= float(cfg["range_atr_ratio_max"])) &
        (trend_strength < float(cfg["trend_strength_min"]))
    )

    regime_expansion = (
        (trend_state != 0) &
        (atr_ratio >= float(cfg["expansion_atr_ratio_min"])) &
        (momentum_quality >= float(cfg["expansion_mom_min"])) &
        trend_impulse_mode &
        (trend_decay < 0.45) &
        (cluster_strength >= 0.65)
    )

    regime_exhaustion_long = (
        (trend_state == 1) &
        (rsi_val >= float(cfg["exhaustion_rsi_long"])) &
        (trend_path_quality >= float(cfg["exhaustion_path_min"])) &
        (reg_curvature_raw < 0) &
        (trend_decay >= 0.45)
    )

    regime_exhaustion_short = (
        (trend_state == -1) &
        (rsi_val <= float(cfg["exhaustion_rsi_short"])) &
        (trend_path_quality >= float(cfg["exhaustion_path_min"])) &
        (reg_curvature_raw > 0) &
        (trend_decay >= 0.45)
    )

    regime_exhaustion = regime_exhaustion_long | regime_exhaustion_short

    regime_state = pd.Series(
        np.where(
            regime_exhaustion, 4,
            np.where(regime_expansion, 3, np.where(regime_trend, 1, np.where(regime_range, 2, 0))),
        ),
        index=idx,
        dtype=int,
    )

    regime_text = pd.Series(
        np.where(
            regime_state == 1, "Trend",
            np.where(regime_state == 2, "Range", np.where(regime_state == 3, "Expansion", np.where(regime_state == 4, "Exhaustion", "Neutral"))),
        ),
        index=idx,
    )

    regime_trend_bull = regime_trend & (trend_state == 1)
    regime_trend_bear = regime_trend & (trend_state == -1)
    regime_expansion_bull = regime_expansion & (trend_state == 1)
    regime_expansion_bear = regime_expansion & (trend_state == -1)
    regime_exhaust_bull = regime_exhaustion & (trend_state == 1)
    regime_exhaust_bear = regime_exhaustion & (trend_state == -1)

    out["sc_trend_state"] = trend_state.astype(int)
    out["sc_trend_dir"] = trend_dir.astype(int)
    out["sc_trend_stage"] = trend_stage.astype(int)
    out["sc_trend_strength"] = trend_strength.astype(float)
    out["sc_continuation_quality"] = continuation_quality.astype(float)

    out["sc_structure_quality"] = structure_quality.astype(float)
    out["sc_momentum_quality"] = momentum_quality.astype(float)
    out["sc_path_quality"] = trend_path_quality.astype(float)
    out["sc_cluster_strength"] = cluster_strength.astype(float)
    out["sc_trend_decay"] = trend_decay.astype(float)

    out["sc_mtf_score"] = mtf_score.astype(float)
    out["sc_mtf_bias"] = mtf_bias.astype(int)
    out["sc_mtf_alignment"] = mtf_alignment.astype(int)

    out["sc_regime_state"] = regime_state.astype(int)
    out["sc_regime_text"] = regime_text.astype(str)

    out["sc_is_trend"] = regime_trend.astype(int)
    out["sc_is_range"] = regime_range.astype(int)
    out["sc_is_expansion"] = regime_expansion.astype(int)
    out["sc_is_exhaustion"] = regime_exhaustion.astype(int)

    out["sc_trend_bull"] = regime_trend_bull.astype(int)
    out["sc_trend_bear"] = regime_trend_bear.astype(int)
    out["sc_expansion_bull"] = regime_expansion_bull.astype(int)
    out["sc_expansion_bear"] = regime_expansion_bear.astype(int)
    out["sc_exhaust_bull"] = regime_exhaust_bull.astype(int)
    out["sc_exhaust_bear"] = regime_exhaust_bear.astype(int)

    out["trend_long_signal"] = trend_long_signal.astype(int)
    out["trend_short_signal"] = trend_short_signal.astype(int)
    out["bull_weak_signal"] = bull_weak_signal.astype(int)
    out["bear_weak_signal"] = bear_weak_signal.astype(int)

    out["ema14"] = ema14.astype(float)
    out["ema20"] = ema20.astype(float)
    out["ema33"] = ema33.astype(float)
    out["ema50"] = ema50.astype(float)
    out["ema100"] = ema100.astype(float)
    out["ema200"] = ema200.astype(float)
    out["avt"] = avt.astype(float)
    out["cluster_line"] = cluster_line.astype(float)

    out["cluster_bias"] = cluster_bias.astype(float)
    out["cluster_strength"] = cluster_strength.astype(float)
    out["bull_score"] = bull_score.astype(float)
    out["bear_score"] = bear_score.astype(float)
    out["score_edge"] = score_edge.astype(float)

    out["adx_value"] = adx_value.astype(float)
    out["atr_ratio"] = atr_ratio.astype(float)
    out["rsi"] = rsi_val.astype(float)
    out["mom"] = mom.astype(float)
    out["reg_curvature"] = reg_curvature.astype(int)
    out["reg_curvature_raw"] = pd.to_numeric(reg_curvature_raw, errors="coerce").fillna(0.0).astype(float)

    out["price_to_ema20_pct"] = price_to_ema20_pct.astype(float)
    out["e20e200_pct"] = e20e200_pct.astype(float)
    out["band_width_pct"] = band_width_pct.astype(float)
    out["trend_quality_state"] = trend_quality_state.astype(int)
    out["trend_quality_text"] = trend_quality_text.astype(str)

    for i in range(1, 8):
        out[f"mtf_trend_state_t{i}"] = mtf[f"t{i}"].astype(int)

    out["trend_dir"] = out["sc_trend_dir"].fillna(0).astype(int)
    out["trend_strength"] = out["sc_trend_strength"].fillna(0.0).astype(float)
    out["regime"] = out["sc_regime_state"].fillna(0).astype(int)
    out["regime_label"] = pd.Series(
        np.where(
            out["regime"] == 1, "trend",
            np.where(
                out["regime"] == 2, "range",
                np.where(out["regime"] == 3, "expansion", np.where(out["regime"] == 4, "exhaustion", "neutral")),
            ),
        ),
        index=out.index,
    )

    return out


def build_trend(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return compute_trend_engine(df, config=config)


def run_trend_engine(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    return compute_trend_engine(df, config=config)