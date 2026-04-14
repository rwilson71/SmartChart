"""
SmartChart Backend — d_momentum.py

AI / Momentum Bias Module v2
Clean Python rebuild from Pine parity reference.

Parity blocks:
- Core inputs and helpers
- RSI / EMA momentum / signal / base EMA series
- Component state engine
- Score engine
- OB / OS stretch layer
- Raw bias classification
- Composite AI state classification
- Memory / hold / refresh logic
- Event pulse engine
- Export-ready output contract
- Clean standalone test block

Backend only:
- no TradingView visuals
- no labels / plots / drawing objects
- designed for later parity validation
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# OPTIONAL SHARED INDICATOR IMPORTS
# =============================================================================

try:
    from core.a_indicators import ema, rsi
except Exception:
    try:
        from a_indicators import ema, rsi
    except Exception:
        def ema(series: pd.Series, length: int) -> pd.Series:
            length = max(1, int(length))
            return series.ewm(span=length, adjust=False).mean()

        def rsi(series: pd.Series, length: int = 14) -> pd.Series:
            """
            Wilder-style RSI with TradingView-friendly edge-case behavior.

            Edge-case handling:
            - avg_gain == 0 and avg_loss == 0 -> 50
            - avg_loss == 0 and avg_gain > 0 -> 100
            - avg_gain == 0 and avg_loss > 0 -> 0
            """
            length = max(2, int(length))
            delta = series.diff()

            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)

            avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
            avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            out = 100.0 - (100.0 / (1.0 + rs))

            both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
            loss_zero = (avg_loss == 0.0) & (avg_gain > 0.0)
            gain_zero = (avg_gain == 0.0) & (avg_loss > 0.0)

            out = out.mask(both_zero, 50.0)
            out = out.mask(loss_zero, 100.0)
            out = out.mask(gain_zero, 0.0)

            return out.fillna(50.0)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MomentumConfig:
    # Core
    rsi_len: int = 14
    fast_len: int = 12
    slow_len: int = 26
    signal_len: int = 9
    base_ema_len: int = 20

    bull_rsi_level: float = 55.0
    bear_rsi_level: float = 45.0
    strong_score_min: float = 2.0
    extreme_score_min: float = 3.0

    # Overbought / Oversold
    ob_level: float = 70.0
    os_level: float = 30.0
    ext_ob_level: float = 80.0
    ext_os_level: float = 20.0
    use_obos_filter: bool = True

    # Memory
    confirm_bars: int = 2
    hold_bars: int = 3
    bias_refresh_bars: int = 1


DEFAULT_MOMENTUM_CONFIG: Dict[str, Any] = asdict(MomentumConfig())


# =============================================================================
# HELPERS
# =============================================================================

def _clamp(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


def _bars_since_change(series: pd.Series) -> pd.Series:
    """
    Pine parity helper for:
        ta.barssince(x != x[1])

    This returns the count of consecutive bars for which the value
    has remained unchanged.

    Pine behavior in this use:
    - change on current bar -> 0
    - unchanged next bar -> 1
    - unchanged next bar -> 2
    """
    vals = series.to_numpy()
    out = np.zeros(len(vals), dtype=float)

    if len(vals) == 0:
        return pd.Series(out, index=series.index)

    out[0] = 0.0
    for i in range(1, len(vals)):
        cur = vals[i]
        prev = vals[i - 1]

        if pd.isna(cur) or pd.isna(prev) or cur != prev:
            out[i] = 0.0
        else:
            out[i] = out[i - 1] + 1.0

    return pd.Series(out, index=series.index)


# =============================================================================
# CORE ENGINE
# =============================================================================

def run_momentum_module(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Clean SmartChart AI / Momentum Bias backend module.

    Required columns:
    - close

    Optional:
    - open, high, low, volume

    Returns:
    DataFrame with all module fields appended.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if "close" not in df.columns:
        raise ValueError("run_momentum_module() requires a DataFrame with a 'close' column.")

    cfg_raw = MomentumConfig()
    if config:
        for k, v in config.items():
            if hasattr(cfg_raw, k):
                setattr(cfg_raw, k, v)
    cfg = cfg_raw

    out = df.copy()
    close = out["close"].astype(float)

    # =========================================================================
    # CORE SERIES
    # =========================================================================

    out["mom_rsi"] = rsi(close, cfg.rsi_len)
    out["mom_ema_fast"] = ema(close, cfg.fast_len)
    out["mom_ema_slow"] = ema(close, cfg.slow_len)
    out["mom_ema_base"] = ema(close, cfg.base_ema_len)

    out["mom_line"] = out["mom_ema_fast"] - out["mom_ema_slow"]
    out["mom_signal_line"] = ema(out["mom_line"], cfg.signal_len)
    out["mom_hist_line"] = out["mom_line"] - out["mom_signal_line"]

    out["mom_line_slope"] = out["mom_line"] - out["mom_line"].shift(1)
    out["mom_signal_slope"] = out["mom_signal_line"] - out["mom_signal_line"].shift(1)

    out["mom_dist_base"] = close - out["mom_ema_base"]

    dist_base_pct0 = np.where(
        out["mom_ema_base"] != 0.0,
        (out["mom_dist_base"] / out["mom_ema_base"]) * 100.0,
        0.0,
    )
    out["mom_dist_base_pct0"] = pd.Series(dist_base_pct0, index=out.index).fillna(0.0)
    out["mom_dist_base_pct"] = _clamp(out["mom_dist_base_pct0"], -10.0, 10.0)

    # =========================================================================
    # COMPONENT STATES
    # =========================================================================

    out["mom_rsi_bull"] = (out["mom_rsi"] >= cfg.bull_rsi_level).astype(int)
    out["mom_rsi_bear"] = (out["mom_rsi"] <= cfg.bear_rsi_level).astype(int)

    out["mom_bias_bull"] = (
        (out["mom_line"] > 0.0) &
        (out["mom_hist_line"] > 0.0)
    ).astype(int)

    out["mom_bias_bear"] = (
        (out["mom_line"] < 0.0) &
        (out["mom_hist_line"] < 0.0)
    ).astype(int)

    out["mom_slope_bull"] = (
        (out["mom_line_slope"] > 0.0) &
        (out["mom_signal_slope"] >= 0.0)
    ).astype(int)

    out["mom_slope_bear"] = (
        (out["mom_line_slope"] < 0.0) &
        (out["mom_signal_slope"] <= 0.0)
    ).astype(int)

    out["mom_price_bull"] = (
        (close > out["mom_ema_base"]) &
        (out["mom_dist_base_pct"] > 0.0)
    ).astype(int)

    out["mom_price_bear"] = (
        (close < out["mom_ema_base"]) &
        (out["mom_dist_base_pct"] < 0.0)
    ).astype(int)

    # =========================================================================
    # SCORE ENGINE
    # =========================================================================

    out["mom_bull_score"] = (
        out["mom_rsi_bull"]
        + out["mom_bias_bull"]
        + out["mom_slope_bull"]
        + out["mom_price_bull"]
    ).astype(float)

    out["mom_bear_score"] = (
        out["mom_rsi_bear"]
        + out["mom_bias_bear"]
        + out["mom_slope_bear"]
        + out["mom_price_bear"]
    ).astype(float)

    out["mom_score_diff"] = out["mom_bull_score"] - out["mom_bear_score"]

    # =========================================================================
    # OB / OS LAYER
    #
    #  2  = Extreme Overbought
    #  1  = Overbought
    #  0  = Neutral
    # -1  = Oversold
    # -2  = Extreme Oversold
    # =========================================================================

    raw_stretch = np.select(
        [
            out["mom_rsi"] >= cfg.ext_ob_level,
            out["mom_rsi"] >= cfg.ob_level,
            out["mom_rsi"] <= cfg.ext_os_level,
            out["mom_rsi"] <= cfg.os_level,
        ],
        [
            2,
            1,
            -2,
            -1,
        ],
        default=0,
    )

    out["mom_raw_stretch_state"] = pd.Series(raw_stretch, index=out.index).astype(int)

    if cfg.use_obos_filter:
        out["mom_ob_state"] = out["mom_raw_stretch_state"].astype(int)
    else:
        out["mom_ob_state"] = pd.Series(0, index=out.index, dtype=int)

    out["mom_is_overbought"] = (out["mom_ob_state"] == 1).astype(int)
    out["mom_is_extreme_overbought"] = (out["mom_ob_state"] == 2).astype(int)
    out["mom_is_oversold"] = (out["mom_ob_state"] == -1).astype(int)
    out["mom_is_extreme_oversold"] = (out["mom_ob_state"] == -2).astype(int)

    # =========================================================================
    # RAW BIAS CLASSIFICATION
    #
    #  3  = Strong Bull
    #  2  = Bull
    #  1  = Weak Bull
    #  0  = Neutral
    # -1  = Weak Bear
    # -2  = Bear
    # -3  = Strong Bear
    # =========================================================================

    raw_bias_state = np.select(
        [
            (out["mom_bull_score"] >= cfg.extreme_score_min) & (out["mom_bull_score"] > out["mom_bear_score"]),
            (out["mom_bull_score"] >= cfg.strong_score_min) & (out["mom_bull_score"] > out["mom_bear_score"]),
            (out["mom_bull_score"] > out["mom_bear_score"]),
            (out["mom_bear_score"] >= cfg.extreme_score_min) & (out["mom_bear_score"] > out["mom_bull_score"]),
            (out["mom_bear_score"] >= cfg.strong_score_min) & (out["mom_bear_score"] > out["mom_bull_score"]),
            (out["mom_bear_score"] > out["mom_bull_score"]),
        ],
        [3, 2, 1, -3, -2, -1],
        default=0,
    )

    out["mom_raw_bias_state"] = pd.Series(raw_bias_state, index=out.index).astype(int)
    out["mom_raw_bias_dir"] = np.sign(out["mom_raw_bias_state"]).astype(int)

    # =========================================================================
    # COMPOSITE AI STATE
    #
    #  5  = Strong Bull + Extreme Overbought
    #  4  = Bull + Overbought
    #  3  = Strong Bull
    #  2  = Bull
    #  1  = Weak Bull
    #  0  = Neutral
    # -1  = Weak Bear
    # -2  = Bear
    # -3  = Strong Bear
    # -4  = Bear + Oversold
    # -5  = Strong Bear + Extreme Oversold
    # =========================================================================

    raw_ai_state = np.select(
        [
            (out["mom_raw_bias_state"] == 3) & (out["mom_is_extreme_overbought"] == 1),
            (out["mom_raw_bias_state"] > 0) & (out["mom_is_overbought"] == 1),
            (out["mom_raw_bias_state"] == -3) & (out["mom_is_extreme_oversold"] == 1),
            (out["mom_raw_bias_state"] < 0) & (out["mom_is_oversold"] == 1),
        ],
        [5, 4, -5, -4],
        default=out["mom_raw_bias_state"],
    )

    out["mom_raw_ai_state"] = pd.Series(raw_ai_state, index=out.index).astype(int)
    out["mom_raw_ai_dir"] = np.sign(out["mom_raw_ai_state"]).astype(int)

    # =========================================================================
    # MEMORY
    # =========================================================================

    out["mom_raw_stable_count"] = _bars_since_change(out["mom_raw_ai_state"])
    out["mom_raw_stable_bars"] = out["mom_raw_stable_count"].fillna(0.0) + 1.0
    out["mom_state_ready"] = (out["mom_raw_stable_bars"] >= cfg.confirm_bars).astype(int)

    out["mom_bias_stable_count"] = _bars_since_change(out["mom_raw_ai_dir"])
    out["mom_bias_stable_bars"] = out["mom_bias_stable_count"].fillna(0.0) + 1.0
    out["mom_bias_ready"] = (out["mom_bias_stable_bars"] >= cfg.bias_refresh_bars).astype(int)

    ai_state_vals = np.zeros(len(out), dtype=int)
    ai_dir_vals = np.zeros(len(out), dtype=int)
    ai_age_vals = np.zeros(len(out), dtype=int)

    raw_ai_state_vals = out["mom_raw_ai_state"].fillna(0).astype(int).to_numpy()
    raw_ai_dir_vals = out["mom_raw_ai_dir"].fillna(0).astype(int).to_numpy()
    state_ready_vals = out["mom_state_ready"].fillna(0).astype(int).to_numpy()
    bias_ready_vals = out["mom_bias_ready"].fillna(0).astype(int).to_numpy()

    if len(out) > 0:
        ai_state_vals[0] = raw_ai_state_vals[0]
        ai_dir_vals[0] = raw_ai_dir_vals[0]
        ai_age_vals[0] = 0

    for i in range(1, len(out)):
        prev_ai_state = ai_state_vals[i - 1]
        prev_ai_dir = ai_dir_vals[i - 1]
        prev_ai_age = ai_age_vals[i - 1]

        raw_state = raw_ai_state_vals[i]
        raw_dir = raw_ai_dir_vals[i]

        can_flip = prev_ai_age >= cfg.hold_bars
        allow_state_flip = bool(state_ready_vals[i]) and (raw_state != prev_ai_state) and can_flip
        allow_bias_refresh = bool(bias_ready_vals[i]) and (raw_dir != 0) and (raw_dir != prev_ai_dir)

        if allow_state_flip:
            ai_state_vals[i] = raw_state
            ai_dir_vals[i] = raw_dir
            ai_age_vals[i] = 0
        else:
            ai_state_vals[i] = prev_ai_state
            ai_dir_vals[i] = raw_dir if allow_bias_refresh else prev_ai_dir
            ai_age_vals[i] = prev_ai_age + 1

    out["sc_mom_ai_state"] = ai_state_vals.astype(int)
    out["sc_mom_ai_dir"] = ai_dir_vals.astype(int)
    out["sc_mom_ai_age"] = ai_age_vals.astype(int)

    # =========================================================================
    # EVENTS
    # =========================================================================

    out["sc_mom_state_changed"] = (
        out["sc_mom_ai_state"] != out["sc_mom_ai_state"].shift(1).fillna(out["sc_mom_ai_state"])
    ).astype(int)

    if len(out) > 0:
        out.iloc[0, out.columns.get_loc("sc_mom_state_changed")] = 0

    out["sc_mom_bull_weak_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == 1)
    ).astype(int)

    out["sc_mom_bull_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == 2)
    ).astype(int)

    out["sc_mom_bull_strong_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == 3)
    ).astype(int)

    out["sc_mom_bull_ob_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == 4)
    ).astype(int)

    out["sc_mom_bull_ext_ob_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == 5)
    ).astype(int)

    out["sc_mom_bear_weak_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == -1)
    ).astype(int)

    out["sc_mom_bear_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == -2)
    ).astype(int)

    out["sc_mom_bear_strong_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == -3)
    ).astype(int)

    out["sc_mom_bear_os_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == -4)
    ).astype(int)

    out["sc_mom_bear_ext_os_pulse"] = (
        (out["sc_mom_state_changed"] == 1) & (out["sc_mom_ai_state"] == -5)
    ).astype(int)

    # =========================================================================
    # EXPORT-READY FIELDS
    # =========================================================================

    out["sc_mom_state"] = out["sc_mom_ai_state"].astype(int)
    out["sc_mom_dir"] = out["sc_mom_ai_dir"].astype(int)
    out["sc_mom_flip"] = out["sc_mom_state_changed"].astype(int)

    out["sc_mom_bull_weak"] = out["sc_mom_bull_weak_pulse"].astype(int)
    out["sc_mom_bull"] = out["sc_mom_bull_pulse"].astype(int)
    out["sc_mom_bull_strong"] = out["sc_mom_bull_strong_pulse"].astype(int)
    out["sc_mom_bull_ob"] = out["sc_mom_bull_ob_pulse"].astype(int)
    out["sc_mom_bull_ext_ob"] = out["sc_mom_bull_ext_ob_pulse"].astype(int)

    out["sc_mom_bear_weak"] = out["sc_mom_bear_weak_pulse"].astype(int)
    out["sc_mom_bear"] = out["sc_mom_bear_pulse"].astype(int)
    out["sc_mom_bear_strong"] = out["sc_mom_bear_strong_pulse"].astype(int)
    out["sc_mom_bear_os"] = out["sc_mom_bear_os_pulse"].astype(int)
    out["sc_mom_bear_ext_os"] = out["sc_mom_bear_ext_os_pulse"].astype(int)

    out["sc_mom_ob_state"] = out["mom_ob_state"].astype(int)
    out["sc_mom_bull_score"] = out["mom_bull_score"].astype(float)
    out["sc_mom_bear_score"] = out["mom_bear_score"].astype(float)
    out["sc_mom_score_diff"] = out["mom_score_diff"].astype(float)

    out["sc_mom_rsi"] = out["mom_rsi"].astype(float)
    out["sc_mom_line"] = out["mom_line"].astype(float)
    out["sc_mom_signal_line"] = out["mom_signal_line"].astype(float)
    out["sc_mom_hist_line"] = out["mom_hist_line"].astype(float)
    out["sc_mom_dist_base_pct"] = out["mom_dist_base_pct"].astype(float)

    # =========================================================================
    # TEXT MAPPING
    # =========================================================================

    state_text_map = {
        5: "STRONG BULL + EXT OB",
        4: "BULL + OB",
        3: "STRONG BULL",
        2: "BULL",
        1: "WEAK BULL",
        0: "NEUTRAL",
        -1: "WEAK BEAR",
        -2: "BEAR",
        -3: "STRONG BEAR",
        -4: "BEAR + OS",
        -5: "STRONG BEAR + EXT OS",
    }

    dir_text_map = {
        1: "BULL",
        0: "NEUTRAL",
        -1: "BEAR",
    }

    obos_text_map = {
        2: "EXT OB",
        1: "OB",
        0: "NEUTRAL",
        -1: "OS",
        -2: "EXT OS",
    }

    out["sc_mom_state_text"] = out["sc_mom_state"].map(state_text_map).fillna("NEUTRAL")
    out["sc_mom_dir_text"] = out["sc_mom_dir"].map(dir_text_map).fillna("NEUTRAL")
    out["sc_mom_obos_text"] = out["sc_mom_ob_state"].map(obos_text_map).fillna("NEUTRAL")

    return out


# =============================================================================
# PUBLIC CONTRACT WRAPPER
# =============================================================================

def run_d_momentum(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    return run_momentum_module(df=df, config=config)

import pandas as pd
from core.d_momentum import run_d_momentum

# LOAD REAL DATA
df = pd.read_csv("data/xauusd_m1_full.csv")

# Ensure datetime index (IMPORTANT)
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)

# Run module
result = run_d_momentum(df)

print(result[[
    "close",
    "sc_mom_state",
    "sc_mom_dir",
    "sc_mom_ob_state",
    "sc_mom_bull_score",
    "sc_mom_bear_score",
    "sc_mom_score_diff"
]].tail(20))