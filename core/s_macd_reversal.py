from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MacdReversalConfig:
    # MACD trend block
    fast_length: int = 12
    slow_length: int = 26
    smooth_length: int = 9
    ma_length: int = 50

    # Reversal probability block
    osc_period: int = 20
    short_sma_len: int = 5
    long_sma_len: int = 34

    duration_std_window: int = 100

    # MTF
    mtf_on: bool = True
    mtf_weights: tuple[float, float, float, float, float, float] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    )


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
    return ((series > 0) & (series.shift(1) <= 0)) | ((series < 0) & (series.shift(1) >= 0))


def _cdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + erf(float(x) / sqrt(2.0)))


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


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first" if "open" in df.columns else "last",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    return df.resample(rule).agg(agg).dropna()


def _align_to_base(series: pd.Series, base_index: pd.Index) -> pd.Series:
    return series.reindex(base_index, method="ffill")


# =============================================================================
# CORE INTERNAL CALCS
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
            fall == 0,
            100.0,
            np.where(rise == 0, 0.0, 100.0 - (100.0 / (1.0 + rise / fall))),
        ),
        index=df.index,
        dtype=float,
    ) - 50.0

    cross_zero = _cross_zero(custom_rsi)
    cut = _bars_since_true(cross_zero)

    segment_duration = cut.shift(1).where(cross_zero)
    durations_avg = segment_duration.expanding(min_periods=1).mean().reindex(df.index).ffill()
    durations_std = segment_duration.expanding(min_periods=2).std().reindex(df.index).ffill()

    cut_volatility = cut.rolling(cfg.duration_std_window, min_periods=2).std() / 2.0

    z = (cut - durations_avg) / durations_std.replace(0.0, np.nan)
    probability = z.fillna(0.0).apply(_cdf_scalar).clip(0.0, 1.0)

    extreme_reversal_probability = (cut > (durations_avg + durations_std * 3.0)).astype(int)
    high_reversal_probability = (probability > 0.84).astype(int)
    extreme_probability = (probability > 0.98).astype(int)
    low_reversal_probability = (probability < 0.14).astype(int)
    probability_reset = cross_zero.astype(int)

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

    macd_scores = []
    reversal_scores = []

    for key in ["tf1", "tf2", "tf3", "tf4", "tf5", "tf6"]:
        tf_df = _resample_ohlc(df, tf_map[key])

        tf_macd = _compute_macd_block(tf_df, cfg)
        tf_rev = _compute_reversal_block(tf_df, cfg)

        macd_score = _align_to_base(tf_macd["market_trend_state"].astype(float), df.index).fillna(0.0)
        reversal_score = _align_to_base(tf_rev["reversal_probability"].astype(float), df.index).fillna(0.0)

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
# PUBLIC API
# =============================================================================

def calculate_macd_reversal(
    df: pd.DataFrame,
    config: Optional[MacdReversalConfig] = None,
    timeframe_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    SmartChart backend conversion of Combined MACD & Trend Reversal,
    now with multi-timeframe averaging.

    Expected input:
        DataFrame indexed by datetime with:
        high, low, close
        open optional for resampling quality
    """
    cfg = config or MacdReversalConfig()
    _validate_ohlc(df)

    base = df.copy().sort_index()

    macd_block = _compute_macd_block(base, cfg)
    reversal_block = _compute_reversal_block(base, cfg)
    mtf_block = _compute_mtf_layer(base, cfg, timeframe_map=timeframe_map)

    out = pd.concat([macd_block, reversal_block, mtf_block], axis=1)

    # Export contract
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

    # SmartChart helpers
    out["macd_direction"] = out["market_trend_state"].astype(int)
    out["macd_signal"] = np.where(out["macd_signal_state"] != 0, 1, 0).astype(int)
    out["reversal_signal"] = ((out["high_reversal_probability"] == 1) | (out["extreme_probability"] == 1)).astype(int)
    out["reversal_strength"] = out["reversal_probability"].astype(float)

    return out


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
    ]

    print("SmartChart MACD Reversal Module MTF — direct test")
    print(result[cols].tail(30))