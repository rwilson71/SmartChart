from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import math
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MacdReversalConfig:
    # -------------------------------------------------------------------------
    # MACD Trend block (Pine authority)
    # -------------------------------------------------------------------------
    fast_length: int = 12
    slow_length: int = 26
    smooth_length: int = 9
    ma_length: int = 50

    # -------------------------------------------------------------------------
    # Reversal Probability block (Pine authority)
    # -------------------------------------------------------------------------
    osc_period: int = 20
    short_sma_len: int = 5
    long_sma_len: int = 34
    duration_std_window: int = 100

    # -------------------------------------------------------------------------
    # Optional SmartChart MTF extension
    # Not part of the original Pine signal logic; keep separate from parity
    # -------------------------------------------------------------------------
    mtf_on: bool = True
    mtf_weights: tuple[float, float, float, float, float, float] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    )


DEFAULT_MACD_REVERSAL_CONFIG = MacdReversalConfig()


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()


def _cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_under(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _cross_zero(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    return ((series > 0) & (prev <= 0)) | ((series < 0) & (prev >= 0))


def _bars_since_true(condition: pd.Series) -> pd.Series:
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


def _pine_cdf(z: float) -> float:
    """
    Pine-authority approximation from the TradingView script.
    """
    if pd.isna(z):
        return np.nan

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = -1.0 if z < 0 else 1.0
    x = abs(float(z)) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    erf_approx = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * erf_approx)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return np.nan
    return float(np.mean(values))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg: Dict[str, str] = {
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "open" in df.columns:
        agg["open"] = "first"

    out = df.resample(rule).agg(agg).dropna()
    return out.sort_index()


def _align_to_base(series: pd.Series, base_index: pd.Index) -> pd.Series:
    return series.reindex(base_index, method="ffill")


def _state_to_text(x: int) -> str:
    if x > 0:
        return "bullish"
    if x < 0:
        return "bearish"
    return "sideways"


def _clean_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        val = float(x)
    except Exception:
        return None
    if np.isnan(val) or np.isinf(val):
        return None
    return val


def _clean_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


# =============================================================================
# CORE INTERNAL CALCULATIONS
# =============================================================================

def _compute_macd_block(df: pd.DataFrame, cfg: MacdReversalConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    macd = _ema(df["close"], cfg.fast_length) - _ema(df["close"], cfg.slow_length)
    signal = _ema(macd, cfg.smooth_length)
    hist = macd - signal
    ma = _sma(df["close"], cfg.ma_length)

    bullish = df["close"] > ma
    bearish = df["close"] < ma

    bullish_dot = _cross_over(macd, signal) & bullish
    bearish_dot = _cross_under(macd, signal) & bearish
    sideways_dot = ~(bullish_dot | bearish_dot)

    market_trend_state = pd.Series(
        np.where(bullish, 1, np.where(bearish, -1, 0)),
        index=df.index,
        dtype=int,
    )

    macd_signal_state = pd.Series(
        np.where(bullish_dot, 1, np.where(bearish_dot, -1, 0)),
        index=df.index,
        dtype=int,
    )

    out["macd_line"] = macd
    out["macd_signal_line"] = signal
    out["macd_hist"] = hist
    out["trend_ma"] = ma
    out["bullish_trend"] = bullish.astype(int)
    out["bearish_trend"] = bearish.astype(int)
    out["bullish_dot"] = bullish_dot.astype(int)
    out["bearish_dot"] = bearish_dot.astype(int)
    out["sideways_dot"] = sideways_dot.astype(int)
    out["market_trend_state"] = market_trend_state
    out["macd_signal_state"] = macd_signal_state

    return out


def _compute_reversal_block(df: pd.DataFrame, cfg: MacdReversalConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    midpoint_price = (df["high"] + df["low"]) / 2.0
    short_sma = _sma(midpoint_price, cfg.short_sma_len)
    long_sma = _sma(midpoint_price, cfg.long_sma_len)
    amazing_osc = short_sma - long_sma

    ao_change = amazing_osc.diff().fillna(0.0)
    rise = _rma(ao_change.clip(lower=0.0), cfg.osc_period)
    fall = _rma((-ao_change).clip(lower=0.0), cfg.osc_period)

    custom_rsi = pd.Series(
        np.where(
            fall == 0.0,
            100.0,
            np.where(
                rise == 0.0,
                0.0,
                100.0 - (100.0 / (1.0 + rise / fall)),
            ),
        ),
        index=df.index,
        dtype=float,
    ) - 50.0

    cross_zero = _cross_zero(custom_rsi)
    cut = _bars_since_true(cross_zero)

    # -------------------------------------------------------------------------
    # Pine parity for duration memory:
    # var durations = array.new_int()
    # cut = ta.barssince(ta.cross(customRSI, 0))
    # if cut == 0 and cut != cut[1]
    #     durations.unshift(cut[1])
    # basis = durations.avg()
    # z = (cut - durations.avg()) / durations.stdev()
    # -------------------------------------------------------------------------
    durations_store: list[float] = []
    durations_avg_vals: list[float] = []
    durations_std_vals: list[float] = []
    z_vals: list[float] = []
    probability_vals: list[float] = []

    cut_vals = cut.to_numpy(dtype=float)
    cross_vals = cross_zero.fillna(False).to_numpy(dtype=bool)

    for i in range(len(df)):
        if cross_vals[i]:
            prev_cut = cut_vals[i - 1] if i > 0 else np.nan
            if not np.isnan(prev_cut):
                durations_store.insert(0, float(prev_cut))

        dur_avg = _safe_mean(durations_store)
        dur_std = _safe_std(durations_store)

        durations_avg_vals.append(dur_avg)
        durations_std_vals.append(dur_std)

        current_cut = cut_vals[i]
        if np.isnan(current_cut) or np.isnan(dur_avg) or np.isnan(dur_std) or dur_std == 0.0:
            z = np.nan
            prob = np.nan
        else:
            z = (current_cut - dur_avg) / dur_std
            prob = _pine_cdf(z)

        z_vals.append(z)
        probability_vals.append(prob)

    durations_avg = pd.Series(durations_avg_vals, index=df.index, dtype=float)
    durations_std = pd.Series(durations_std_vals, index=df.index, dtype=float)
    z = pd.Series(z_vals, index=df.index, dtype=float)
    probability = pd.Series(probability_vals, index=df.index, dtype=float)

    cut_volatility = cut.rolling(cfg.duration_std_window, min_periods=2).std() / 2.0

    high_reversal_probability = (probability > 0.84).fillna(False).astype(int)
    extreme_probability = (probability > 0.98).fillna(False).astype(int)
    low_reversal_probability = (probability < 0.14).fillna(False).astype(int)
    extreme_reversal_probability = (
        cut > (durations_avg + durations_std * 3.0)
    ).fillna(False).astype(int)
    probability_reset = (cut == 0).fillna(False).astype(int)

    reversal_bias = pd.Series(
        np.where(custom_rsi > 0, 1, np.where(custom_rsi < 0, -1, 0)),
        index=df.index,
        dtype=int,
    )

    reversal_momentum = pd.Series(
        np.where(
            ((custom_rsi > 0) & (custom_rsi > custom_rsi.shift(1)))
            | ((custom_rsi < 0) & (custom_rsi < custom_rsi.shift(1))),
            1,
            0,
        ),
        index=df.index,
        dtype=int,
    )

    out["midpoint_price"] = midpoint_price
    out["ao_short_sma"] = short_sma
    out["ao_long_sma"] = long_sma
    out["amazing_osc"] = amazing_osc
    out["custom_rsi"] = custom_rsi
    out["cut"] = cut
    out["durations_avg"] = durations_avg
    out["durations_std"] = durations_std
    out["cut_volatility"] = cut_volatility
    out["reversal_zscore"] = z
    out["reversal_probability"] = probability
    out["reversal_bias"] = reversal_bias
    out["reversal_momentum"] = reversal_momentum
    out["high_reversal_probability"] = high_reversal_probability
    out["extreme_reversal_probability"] = extreme_reversal_probability
    out["extreme_probability"] = extreme_probability
    out["low_reversal_probability"] = low_reversal_probability
    out["probability_reset"] = probability_reset

    return out


def _compute_mtf_layer(
    df: pd.DataFrame,
    cfg: MacdReversalConfig,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    if not cfg.mtf_on:
        for i in range(1, 7):
            out[f"macd_mtf_{i}"] = 0.0
            out[f"reversal_mtf_{i}"] = 0.0
        out["macd_mtf_avg"] = 0.0
        out["reversal_mtf_avg"] = 0.0
        return out

    tf_map = timeframe_map or {
        "tf1": "5min",
        "tf2": "15min",
        "tf3": "30min",
        "tf4": "1h",
        "tf5": "4h",
        "tf6": "1D",
    }

    macd_scores: list[pd.Series] = []
    reversal_scores: list[pd.Series] = []

    for key in ["tf1", "tf2", "tf3", "tf4", "tf5", "tf6"]:
        tf_df = _resample_ohlc(df, tf_map[key])

        tf_macd = _compute_macd_block(tf_df, cfg)
        tf_rev = _compute_reversal_block(tf_df, cfg)

        macd_score = _align_to_base(
            tf_macd["market_trend_state"].astype(float), df.index
        ).fillna(0.0)

        reversal_score = _align_to_base(
            tf_rev["reversal_probability"].astype(float), df.index
        ).fillna(0.0)

        macd_scores.append(macd_score)
        reversal_scores.append(reversal_score)

    for i in range(6):
        out[f"macd_mtf_{i+1}"] = macd_scores[i]
        out[f"reversal_mtf_{i+1}"] = reversal_scores[i]

    weights = np.array(cfg.mtf_weights, dtype=float)
    w_sum = float(weights.sum())
    w_safe = 1.0 if w_sum <= 0 else w_sum

    out["macd_mtf_avg"] = sum(out[f"macd_mtf_{i+1}"] * weights[i] for i in range(6)) / w_safe
    out["reversal_mtf_avg"] = sum(out[f"reversal_mtf_{i+1}"] * weights[i] for i in range(6)) / w_safe

    return out


# =============================================================================
# PUBLIC CALCULATION API
# =============================================================================

def calculate_macd_reversal(
    df: pd.DataFrame,
    config: Optional[MacdReversalConfig] = None,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    SmartChart backend parity rebuild of:
    TradingView Pine "Combined MACD & Trend Reversal"

    Input:
        DatetimeIndex
        required columns: high, low, close
        optional: open (improves MTF resampling quality)
    """
    cfg = config or MacdReversalConfig()
    _validate_ohlc(df)

    base = df.copy().sort_index()

    macd_block = _compute_macd_block(base, cfg)
    reversal_block = _compute_reversal_block(base, cfg)
    mtf_block = _compute_mtf_layer(base, cfg, timeframe_map=timeframe_map)

    out = pd.concat([macd_block, reversal_block, mtf_block], axis=1)

    # -------------------------------------------------------------------------
    # Export contract
    # -------------------------------------------------------------------------
    out["macd_trend_export"] = out["market_trend_state"].astype(int)
    out["macd_signal_export"] = out["macd_signal_state"].astype(int)
    out["macd_bullish_dot_export"] = out["bullish_dot"].astype(int)
    out["macd_bearish_dot_export"] = out["bearish_dot"].astype(int)
    out["macd_sideways_export"] = out["sideways_dot"].astype(int)

    out["reversal_bias_export"] = out["reversal_bias"].astype(int)
    out["reversal_probability_export"] = out["reversal_probability"].astype(float)
    out["reversal_high_prob_export"] = out["high_reversal_probability"].astype(int)
    out["reversal_extreme_prob_export"] = out["extreme_probability"].astype(int)
    out["reversal_low_prob_export"] = out["low_reversal_probability"].astype(int)
    out["reversal_probability_reset_export"] = out["probability_reset"].astype(int)
    out["reversal_extreme_duration_export"] = out["extreme_reversal_probability"].astype(int)

    out["macd_mtf_avg_export"] = out["macd_mtf_avg"].astype(float)
    out["reversal_mtf_avg_export"] = out["reversal_mtf_avg"].astype(float)

    # -------------------------------------------------------------------------
    # SmartChart helpers
    # -------------------------------------------------------------------------
    out["macd_direction"] = out["market_trend_state"].astype(int)
    out["macd_signal"] = (out["macd_signal_state"] != 0).astype(int)
    out["reversal_signal"] = (
        (out["high_reversal_probability"] == 1)
        | (out["extreme_probability"] == 1)
    ).astype(int)
    out["reversal_strength"] = out["reversal_probability"].astype(float)

    out["trend_state_text"] = out["market_trend_state"].apply(_state_to_text)
    out["reversal_bias_text"] = out["reversal_bias"].apply(_state_to_text)

    out["combined_state_text"] = np.where(
        (out["market_trend_state"] > 0) & (out["reversal_bias"] > 0),
        "bullish_continuation",
        np.where(
            (out["market_trend_state"] < 0) & (out["reversal_bias"] < 0),
            "bearish_continuation",
            np.where(
                out["reversal_signal"] == 1,
                "reversal_risk",
                "mixed",
            ),
        ),
    )

    return out


# =============================================================================
# PAYLOAD BUILDER
# =============================================================================

def build_macd_reversal_latest_payload(
    df: pd.DataFrame,
    config: Optional[MacdReversalConfig] = None,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    cfg = config or MacdReversalConfig()
    calc = calculate_macd_reversal(df, config=cfg, timeframe_map=timeframe_map)

    if calc.empty:
        return {
            "ok": False,
            "indicator": "macd_reversal",
            "error": "Empty calculation output",
        }

    last = calc.iloc[-1]
    last_ts = calc.index[-1]

    probability = _clean_float(last.get("reversal_probability"))
    probability_pct = None if probability is None else round(probability * 100.0, 2)

    payload: Dict[str, Any] = {
        "ok": True,
        "indicator": "macd_reversal",
        "source_authority": "TradingView Pine",
        "timestamp": str(last_ts),
        "config": asdict(cfg),

        "macd_trend": {
            "macd_line": _clean_float(last.get("macd_line")),
            "signal_line": _clean_float(last.get("macd_signal_line")),
            "histogram": _clean_float(last.get("macd_hist")),
            "moving_average": _clean_float(last.get("trend_ma")),
            "trend_state": last.get("trend_state_text"),
            "market_trend_state": _clean_int(last.get("market_trend_state")),
            "bullish_dot": _clean_int(last.get("bullish_dot")),
            "bearish_dot": _clean_int(last.get("bearish_dot")),
            "sideways_dot": _clean_int(last.get("sideways_dot")),
            "macd_signal_state": _clean_int(last.get("macd_signal_state")),
        },

        "reversal_probability": {
            "custom_rsi": _clean_float(last.get("custom_rsi")),
            "cut": _clean_float(last.get("cut")),
            "avg_duration": _clean_float(last.get("durations_avg")),
            "duration_std": _clean_float(last.get("durations_std")),
            "cut_volatility": _clean_float(last.get("cut_volatility")),
            "z_score": _clean_float(last.get("reversal_zscore")),
            "probability": probability,
            "probability_pct": probability_pct,
            "bias_state": last.get("reversal_bias_text"),
            "bias_value": _clean_int(last.get("reversal_bias")),
            "momentum": _clean_int(last.get("reversal_momentum")),
            "high_reversal": _clean_int(last.get("high_reversal_probability")),
            "extreme_reversal_duration": _clean_int(last.get("extreme_reversal_probability")),
            "extreme_probability": _clean_int(last.get("extreme_probability")),
            "low_reversal": _clean_int(last.get("low_reversal_probability")),
            "reset_state": _clean_int(last.get("probability_reset")),
        },

        "mtf": {
            "macd_mtf_avg": _clean_float(last.get("macd_mtf_avg")),
            "reversal_mtf_avg": _clean_float(last.get("reversal_mtf_avg")),
            "macd_mtf_1": _clean_float(last.get("macd_mtf_1")),
            "macd_mtf_2": _clean_float(last.get("macd_mtf_2")),
            "macd_mtf_3": _clean_float(last.get("macd_mtf_3")),
            "macd_mtf_4": _clean_float(last.get("macd_mtf_4")),
            "macd_mtf_5": _clean_float(last.get("macd_mtf_5")),
            "macd_mtf_6": _clean_float(last.get("macd_mtf_6")),
            "reversal_mtf_1": _clean_float(last.get("reversal_mtf_1")),
            "reversal_mtf_2": _clean_float(last.get("reversal_mtf_2")),
            "reversal_mtf_3": _clean_float(last.get("reversal_mtf_3")),
            "reversal_mtf_4": _clean_float(last.get("reversal_mtf_4")),
            "reversal_mtf_5": _clean_float(last.get("reversal_mtf_5")),
            "reversal_mtf_6": _clean_float(last.get("reversal_mtf_6")),
        },

        "summary": {
            "direction_bias": last.get("trend_state_text"),
            "reversal_state": last.get("reversal_bias_text"),
            "combined_state": last.get("combined_state_text"),
            "macd_signal": _clean_int(last.get("macd_signal")),
            "reversal_signal": _clean_int(last.get("reversal_signal")),
            "reversal_strength": _clean_float(last.get("reversal_strength")),
            "quality_note": (
                "Extreme reversal probability"
                if _clean_int(last.get("extreme_probability")) == 1
                else "High reversal probability"
                if _clean_int(last.get("high_reversal_probability")) == 1
                else "Normal probability state"
            ),
        },
    }

    return payload


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=1200, freq="5min")
    np.random.seed(42)

    drift = np.linspace(0, 20, len(rng))
    noise = np.random.normal(0, 0.9, len(rng)).cumsum()
    close = pd.Series(3300 + drift + noise, index=rng)

    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.random.uniform(0.2, 1.8, len(rng))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.random.uniform(0.2, 1.8, len(rng))

    test_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=rng,
    )

    result = calculate_macd_reversal(test_df)
    payload = build_macd_reversal_latest_payload(test_df)

    cols = [
        "market_trend_state",
        "macd_signal_state",
        "reversal_probability",
        "macd_mtf_avg_export",
        "reversal_mtf_avg_export",
        "macd_trend_export",
        "macd_signal_export",
        "reversal_bias_export",
        "reversal_probability_export",
        "reversal_signal",
        "reversal_strength",
        "combined_state_text",
    ]

    print("SmartChart MACD Reversal Module — direct test")
    print(result[cols].tail(10))
    print("\nLatest payload:")
    print(payload)