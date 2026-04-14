from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class RegimeConfig:
    # Core trend / structure
    ema_fast_len: int = 20
    ema_slow_len: int = 50
    adx_len: int = 14
    atr_len: int = 14
    atr_base_len: int = 50
    er_len: int = 20

    # Thresholds
    adx_trend_min: float = 23.0
    adx_range_max: float = 16.0

    atr_expand_mult: float = 1.15
    atr_contract_mult: float = 0.90

    er_trend_min: float = 0.30
    er_range_max: float = 0.18

    slope_lookback: int = 5
    compression_threshold: float = 0.0035

    # Memory
    confirm_bars: int = 2
    hold_bars: int = 2
    bias_refresh_bars: int = 1


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")


def _to_float_series(series: pd.Series, index: pd.Index, fill_value: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=index)
    return out.astype(float).replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return (
        pd.to_numeric(series, errors="coerce")
        .ffill()
        .fillna(0.0)
        .ewm(span=length, adjust=False)
        .mean()
    )


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return pd.to_numeric(series, errors="coerce").fillna(0.0).rolling(length, min_periods=1).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    alpha = 1.0 / float(length)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).ewm(alpha=alpha, adjust=False).mean()


def _bars_since_change(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index)

    out = np.zeros(len(series), dtype=int)
    values = series.to_numpy()

    for i in range(1, len(values)):
        out[i] = 0 if values[i] != values[i - 1] else out[i - 1] + 1

    return pd.Series(out, index=series.index, dtype=int)


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
    return _rma(tr, length)


def _adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

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

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = _rma(tr, length).replace(0.0, np.nan)
    plus_di = 100.0 * _rma(plus_dm, length) / atr
    minus_di = 100.0 * _rma(minus_dm, length) / atr

    dx = (
        100.0
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0.0, np.nan)
    ).fillna(0.0)

    adx = _rma(dx, length).fillna(0.0)

    return pd.DataFrame(
        {
            "plus_di": plus_di.fillna(0.0),
            "minus_di": minus_di.fillna(0.0),
            "adx": adx.astype(float),
        },
        index=df.index,
    )


def _efficiency_ratio(close: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    direction = (close - close.shift(length)).abs()
    volatility = close.diff().abs().rolling(length, min_periods=1).sum()
    er = direction / volatility.replace(0.0, np.nan)
    return er.fillna(0.0).clip(0.0, 1.0)


def _normalize_slope(series: pd.Series, lookback: int) -> pd.Series:
    lookback = max(1, int(lookback))
    raw = series - series.shift(lookback)
    denom = series.abs().replace(0.0, np.nan)
    return (raw / denom).fillna(0.0)


# =============================================================================
# CORE RAW REGIME
# =============================================================================

def _compute_raw_regime(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    ema_fast = _ema(df["close"], cfg.ema_fast_len)
    ema_slow = _ema(df["close"], cfg.ema_slow_len)

    ema_spread = (ema_fast - ema_slow).abs()
    ema_spread_pct = (ema_spread / ema_slow.replace(0.0, np.nan)).fillna(0.0)

    ema_fast_slope = _normalize_slope(ema_fast, cfg.slope_lookback)
    ema_slow_slope = _normalize_slope(ema_slow, cfg.slope_lookback)

    adx_df = _adx(df, cfg.adx_len)
    plus_di = adx_df["plus_di"]
    minus_di = adx_df["minus_di"]
    adx = adx_df["adx"]

    atr_now = _atr(df, cfg.atr_len)
    atr_base = _sma(atr_now, cfg.atr_base_len).replace(0.0, np.nan)
    atr_ratio = (atr_now / atr_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    er = _efficiency_ratio(df["close"], cfg.er_len)

    trend_dir = pd.Series(
        np.where(
            (ema_fast > ema_slow) & (plus_di >= minus_di),
            1,
            np.where((ema_fast < ema_slow) & (minus_di > plus_di), -1, 0),
        ),
        index=df.index,
        dtype=int,
    )

    compression = ema_spread_pct <= cfg.compression_threshold
    expansion = atr_ratio >= cfg.atr_expand_mult
    contraction = atr_ratio <= cfg.atr_contract_mult

    # -------------------------------------------------------------------------
    # Regime logic
    # Keep structure simple and stable:
    # - Trend: structure present + non-compressed + strength confirmation
    # - Range: compressed/neutral structure + weakness confirmation
    # - Transition: anything in-between
    # -------------------------------------------------------------------------

    trend_condition = (
    (trend_dir != 0)
    & (
        (adx >= cfg.adx_trend_min)
        | (er >= cfg.er_trend_min)
        | (atr_ratio >= cfg.atr_expand_mult)
        )
    )

    range_condition = (
        (compression | (trend_dir == 0))
        & (
            (adx <= cfg.adx_range_max)
            | (er <= cfg.er_range_max)
            | (atr_ratio <= cfg.atr_contract_mult)
        )
    )

    transition_condition = ~(trend_condition | range_condition)

    # 1 = trend, 2 = range, 3 = transition
    raw_regime_state = pd.Series(
        np.where(trend_condition, 1, np.where(range_condition, 2, 3)),
        index=df.index,
        dtype=int,
    )

    # 1 = expansion, 2 = contraction, 3 = normal
    raw_market_state = pd.Series(
        np.where(expansion, 1, np.where(contraction, 2, 3)),
        index=df.index,
        dtype=int,
    )

    raw_regime_bias = pd.Series(
        np.where(raw_regime_state == 1, trend_dir, 0),
        index=df.index,
        dtype=int,
    )

    trend_strength = (
        ((adx / max(cfg.adx_trend_min, 1.0)) * 0.40)
        + (er * 0.35)
        + ((np.minimum(atr_ratio, 2.0) / 2.0) * 0.25)
    ).clip(0.0, 1.0)

    out["ema_fast"] = ema_fast.astype(float)
    out["ema_slow"] = ema_slow.astype(float)
    out["ema_spread_pct"] = ema_spread_pct.astype(float)
    out["ema_fast_slope"] = ema_fast_slope.astype(float)
    out["ema_slow_slope"] = ema_slow_slope.astype(float)

    out["plus_di"] = plus_di.astype(float)
    out["minus_di"] = minus_di.astype(float)
    out["adx"] = adx.astype(float)

    out["atr_now"] = atr_now.astype(float)
    out["atr_base"] = atr_base.fillna(atr_now).astype(float)
    out["atr_ratio"] = atr_ratio.astype(float)
    out["efficiency_ratio"] = er.astype(float)

    out["compression"] = compression.astype(int)
    out["expansion"] = expansion.astype(int)
    out["contraction"] = contraction.astype(int)

    out["trend_dir"] = trend_dir.astype(int)
    out["trend_condition"] = trend_condition.astype(int)
    out["range_condition"] = range_condition.astype(int)
    out["transition_condition"] = transition_condition.astype(int)

    out["raw_regime_state"] = raw_regime_state.astype(int)
    out["raw_market_state"] = raw_market_state.astype(int)
    out["raw_regime_bias"] = raw_regime_bias.astype(int)
    out["raw_regime_strength"] = trend_strength.astype(float)

    return out


# =============================================================================
# MEMORY
# =============================================================================

def _apply_memory(
    raw_regime_state: pd.Series,
    raw_market_state: pd.Series,
    raw_regime_bias: pd.Series,
    cfg: RegimeConfig,
) -> pd.DataFrame:
    if len(raw_regime_state) == 0:
        return pd.DataFrame(index=raw_regime_state.index)

    regime_stable_count = _bars_since_change(raw_regime_state)
    regime_stable_bars = regime_stable_count + 1
    regime_ready = regime_stable_bars >= cfg.confirm_bars

    bias_stable_count = _bars_since_change(raw_regime_bias)
    bias_stable_bars = bias_stable_count + 1
    bias_ready = bias_stable_bars >= cfg.bias_refresh_bars

    regime_state = np.zeros(len(raw_regime_state), dtype=int)
    market_state = np.zeros(len(raw_market_state), dtype=int)
    regime_bias = np.zeros(len(raw_regime_bias), dtype=int)
    regime_age = np.zeros(len(raw_regime_state), dtype=int)

    regime_state[0] = int(raw_regime_state.iloc[0])
    market_state[0] = int(raw_market_state.iloc[0])
    regime_bias[0] = int(raw_regime_bias.iloc[0])
    regime_age[0] = 0

    for i in range(1, len(raw_regime_state)):
        prev_state = regime_state[i - 1]
        prev_market = market_state[i - 1]
        prev_bias = regime_bias[i - 1]
        prev_age = regime_age[i - 1]

        candidate_state = int(raw_regime_state.iloc[i])
        candidate_market = int(raw_market_state.iloc[i])
        candidate_bias = int(raw_regime_bias.iloc[i])

        can_flip = prev_age >= cfg.hold_bars
        allow_state_flip = bool(regime_ready.iloc[i]) and (candidate_state != prev_state) and can_flip
        allow_bias_refresh = bool(bias_ready.iloc[i]) and (candidate_bias != prev_bias)

        if allow_state_flip:
            regime_state[i] = candidate_state
            market_state[i] = candidate_market
            regime_bias[i] = candidate_bias
            regime_age[i] = 0
        else:
            regime_state[i] = prev_state
            market_state[i] = prev_market
            regime_bias[i] = prev_bias
            regime_age[i] = prev_age + 1

            if allow_bias_refresh:
                regime_bias[i] = candidate_bias

    out = pd.DataFrame(index=raw_regime_state.index)
    out["regime_state"] = pd.Series(regime_state, index=raw_regime_state.index, dtype=int)
    out["market_state"] = pd.Series(market_state, index=raw_market_state.index, dtype=int)
    out["regime_bias"] = pd.Series(regime_bias, index=raw_regime_bias.index, dtype=int)
    out["regime_age"] = pd.Series(regime_age, index=raw_regime_state.index, dtype=int)

    out["regime_changed"] = out["regime_state"].ne(out["regime_state"].shift(1)).fillna(False).astype(int)
    out["market_changed"] = out["market_state"].ne(out["market_state"].shift(1)).fillna(False).astype(int)

    return out


# =============================================================================
# TEXT MAPS
# =============================================================================

def _regime_text(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(series == 1, "trend", np.where(series == 2, "range", "transition")),
        index=series.index,
        dtype="object",
    )


def _market_text(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(series == 1, "expansion", np.where(series == 2, "contraction", "normal")),
        index=series.index,
        dtype="object",
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_regime(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """
    SmartChart Regime Engine

    Expected input:
        DataFrame indexed by datetime with:
        open, high, low, close
    """
    cfg = config or RegimeConfig()
    _validate_ohlc(df)

    base = df.copy().sort_index()
    idx = base.index

    base["open"] = _to_float_series(base["open"], idx)
    base["high"] = _to_float_series(base["high"], idx)
    base["low"] = _to_float_series(base["low"], idx)
    base["close"] = _to_float_series(base["close"], idx)

    raw = _compute_raw_regime(base, cfg)
    mem = _apply_memory(
        raw["raw_regime_state"],
        raw["raw_market_state"],
        raw["raw_regime_bias"],
        cfg,
    )

    out = pd.concat([raw, mem], axis=1)

    out["regime_text"] = _regime_text(out["regime_state"])
    out["market_text"] = _market_text(out["market_state"])

    out["is_trend"] = (out["regime_state"] == 1).astype(int)
    out["is_range"] = (out["regime_state"] == 2).astype(int)
    out["is_transition"] = (out["regime_state"] == 3).astype(int)

    out["is_expansion"] = (out["market_state"] == 1).astype(int)
    out["is_contraction"] = (out["market_state"] == 2).astype(int)
    out["is_normal_market"] = (out["market_state"] == 3).astype(int)

    # =========================================================================
    # SMARTCHART OUTPUT CONTRACT
    # =========================================================================

    out["regime_state_export"] = out["regime_state"].astype(int)
    out["regime_bias_export"] = out["regime_bias"].astype(int)
    out["regime_change_export"] = out["regime_changed"].astype(int)

    out["market_state_export"] = out["market_state"].astype(int)
    out["market_change_export"] = out["market_changed"].astype(int)

    out["trend_condition_export"] = out["trend_condition"].astype(int)
    out["range_condition_export"] = out["range_condition"].astype(int)
    out["transition_condition_export"] = out["transition_condition"].astype(int)

    out["expansion_export"] = out["expansion"].astype(int)
    out["contraction_export"] = out["contraction"].astype(int)

    out["regime_strength_export"] = out["raw_regime_strength"].astype(float)

    out["regime_direction"] = out["regime_bias"].astype(int)
    out["regime_signal"] = (out["regime_state"] == 1).astype(int)
    out["regime_strength"] = out["raw_regime_strength"].astype(float)

    # =========================================================================
    # TRUTH ENGINE CONTRACT
    # =========================================================================

    out["regime"] = out["regime_state"].fillna(0).astype(int)
    out["regime_label"] = out["regime_text"].fillna("unknown")

    out["market_condition"] = out["market_state"].fillna(0).astype(int)
    out["market_condition_label"] = out["market_text"].fillna("unknown")

    return out


def build_regime(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    return calculate_regime(df, config=config)


def run_regime(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    return calculate_regime(df, config=config)


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2026-01-01", periods=1500, freq="5min")
    np.random.seed(42)

    trend_part = np.linspace(0, 40, 500) + np.random.normal(0, 0.6, 500).cumsum()
    range_part = 20 + np.random.normal(0, 0.8, 500).cumsum() * 0.2
    exp_part = np.linspace(-10, 25, 500) + np.random.normal(0, 1.5, 500).cumsum()

    price = np.concatenate([3300 + trend_part, 3340 + range_part, 3320 + exp_part])
    close = pd.Series(price, index=rng)
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

    result = calculate_regime(test_df)

    cols = [
        "adx",
        "atr_ratio",
        "efficiency_ratio",
        "trend_dir",
        "regime_state_export",
        "regime_bias_export",
        "market_state_export",
        "trend_condition_export",
        "range_condition_export",
        "transition_condition_export",
        "expansion_export",
        "contraction_export",
        "regime_strength_export",
        "regime_text",
        "market_text",
        "regime_direction",
        "regime_signal",
        "regime_strength",
        "regime",
        "regime_label",
        "market_condition",
        "market_condition_label",
    ]

    print("SmartChart Regime Engine — direct test")
    print(result[cols].tail(30))