from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# HELPERS
# =============================================================================

def _series_or_default(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _pick_first_existing(df: pd.DataFrame, names: list[str], default=0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _normalize_dir_series(series: pd.Series) -> pd.Series:
    s = _safe_numeric(series, 0)
    return pd.Series(
        np.where(s > 0, 1, np.where(s < 0, -1, 0)),
        index=series.index,
    ).astype(int)


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    return _safe_numeric(series, 0).fillna(0).astype(bool).astype(int)


def _label_from_dir(series: pd.Series) -> pd.Series:
    s = _normalize_dir_series(series)
    return pd.Series(
        np.where(s > 0, "bullish", np.where(s < 0, "bearish", "neutral")),
        index=series.index,
    )


def _bool_from_threshold(series: pd.Series, threshold: float) -> pd.Series:
    return (_safe_numeric(series, 0) >= threshold).astype(int)


# =============================================================================
# S1.5 STRATEGY ENGINE v1
# -----------------------------------------------------------------------------
# Locked S1.5 v1 logic:
#
# LONG
# - EMA 100-200 retest
# - MFI fast crosses above slow
# - Cross occurs above zero line
# - MACD bullish / above zeroS
# - AI bullish
#
# SHORT
# - EMA 100-200 retest
# - MFI fast crosses below slow
# - Cross occurs below zero line
# - MACD bearish / below zero
# - AI bearish
#
# Deferred for later:
# - momentum
# - volatility
# - MTF confirmation
# - forecaster
# =============================================================================

def build_strategy_s1_5(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # =========================================================================
    # BASE PASSTHROUGH
    # =========================================================================

    if "timestamp" in df.columns:
        out["timestamp"] = df["timestamp"]
    else:
        out["timestamp"] = pd.Series(df.index, index=df.index)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            out[col] = df[col]

    # =========================================================================
    # TRUTH / BIAS INPUTS
    # =========================================================================

    out["bias_dir"] = _normalize_dir_series(_pick_first_existing(df, ["bias_dir", "dir"], 0))
    out["bias_label"] = _label_from_dir(out["bias_dir"])

    out["truth_state"] = _series_or_default(df, "truth_state", "blocked")
    out["truth_strength"] = _safe_numeric(_series_or_default(df, "truth_strength", 0), 0)
    out["truth_grade"] = _series_or_default(df, "truth_grade", "C")
    out["signal_quality"] = _series_or_default(df, "signal_quality", "low")

    out["continuation_context_ok"] = _normalize_bool_series(
        _series_or_default(df, "continuation_context_ok", 0)
    )
    out["context_ok"] = _normalize_bool_series(_series_or_default(df, "context_ok", 0))

    out["trend_dir"] = _normalize_dir_series(_series_or_default(df, "trend_dir", 0))
    out["ema_dir"] = _normalize_dir_series(_series_or_default(df, "ema_dir", 0))

    # =========================================================================
    # EMA RETEST / EMA DISTANCE
    # =========================================================================

    out["rt_ema_100200"] = _normalize_bool_series(_series_or_default(df, "rt_ema_100200", 0))

    out["ema_distance_stage"] = _safe_numeric(_series_or_default(df, "ema_distance_stage", 0), 0)
    out["ema_distance_label"] = _series_or_default(df, "ema_distance_label", "transition")

    # S1.5 still wants continuation states, not exhaustion
    out["ema_distance_ok"] = out["ema_distance_stage"].isin([1, 2]).astype(int)

    # =========================================================================
    # AI
    # =========================================================================

    out["ai_dir"] = _normalize_dir_series(_series_or_default(df, "ai_dir", 0))
    out["ai_strength"] = _safe_numeric(_series_or_default(df, "ai_strength", 0), 0)

    out["ai_long_ok"] = (out["ai_dir"] == 1).astype(int)
    out["ai_short_ok"] = (out["ai_dir"] == -1).astype(int)

    # =========================================================================
    # MACD
    # -----------------------------------------------------------------------------
    # Preferred fields if available:
    # - macd_line / macd_hist / macd_signal_line
    # Fallback:
    # - macd_rev_dir / macd_reversal_dir
    # - macd_rev_signal / macd_reversal_signal
    # - macd_rev_strength / macd_reversal_strength
    # =========================================================================

    out["macd_line"] = _safe_numeric(
        _pick_first_existing(df, ["macd_line", "macd_value", "macd"], 0),
        0,
    )
    out["macd_hist"] = _safe_numeric(
        _pick_first_existing(df, ["macd_hist", "macd_histogram"], 0),
        0,
    )

    out["macd_rev_dir"] = _normalize_dir_series(
        _pick_first_existing(df, ["macd_rev_dir", "macd_reversal_dir"], 0)
    )
    out["macd_rev_signal"] = _normalize_bool_series(
        _pick_first_existing(df, ["macd_rev_signal", "macd_reversal_signal"], 0)
    )
    out["macd_rev_strength"] = _safe_numeric(
        _pick_first_existing(df, ["macd_rev_strength", "macd_reversal_strength"], 0),
        0,
    )

    # Above / below zero line
    out["macd_above_zero"] = (out["macd_line"] > 0).astype(int)
    out["macd_below_zero"] = (out["macd_line"] < 0).astype(int)

    # Fallback when raw macd line is unavailable
    out["macd_long_ok"] = np.where(
        out["macd_line"] != 0,
        out["macd_above_zero"],
        (
            (out["macd_rev_dir"] == 1) &
            (
                (out["macd_rev_signal"] == 1) |
                (out["macd_rev_strength"] > 0)
            )
        ).astype(int),
    ).astype(int)

    out["macd_short_ok"] = np.where(
        out["macd_line"] != 0,
        out["macd_below_zero"],
        (
            (out["macd_rev_dir"] == -1) &
            (
                (out["macd_rev_signal"] == 1) |
                (out["macd_rev_strength"] > 0)
            )
        ).astype(int),
    ).astype(int)

    # =========================================================================
    # MFI
    # -----------------------------------------------------------------------------
    # Preferred raw fields:
    # - mfi_fast
    # - mfi_slow
    # - mfi_cross_up
    # - mfi_cross_down
    #
    # Fallback:
    # - mfi_dir
    # - mfi_signal
    # =========================================================================

    out["mfi_dir"] = _normalize_dir_series(_series_or_default(df, "mfi_dir", 0))
    out["mfi_signal"] = _normalize_bool_series(_series_or_default(df, "mfi_signal", 0))

    out["mfi_fast"] = _safe_numeric(
        _pick_first_existing(df, ["mfi_fast", "mfi_fast_len", "mfi_fast_value"], 0),
        0,
    )
    out["mfi_slow"] = _safe_numeric(
        _pick_first_existing(df, ["mfi_slow", "mfi_slow_len", "mfi_slow_value"], 0),
        0,
    )
    out["mfi_cross_up"] = _normalize_bool_series(
        _pick_first_existing(df, ["mfi_cross_up", "mfi_fast_cross_up"], 0)
    )
    out["mfi_cross_down"] = _normalize_bool_series(
        _pick_first_existing(df, ["mfi_cross_down", "mfi_fast_cross_down"], 0)
    )

    inferred_mfi_long_cross = (
        (out["mfi_signal"] == 1) &
        (out["mfi_dir"] == 1)
    ).astype(int)

    inferred_mfi_short_cross = (
        (out["mfi_signal"] == 1) &
        (out["mfi_dir"] == -1)
    ).astype(int)

    out["mfi_long_cross"] = np.where(
        out["mfi_cross_up"] == 1,
        1,
        inferred_mfi_long_cross,
    ).astype(int)

    out["mfi_short_cross"] = np.where(
        out["mfi_cross_down"] == 1,
        1,
        inferred_mfi_short_cross,
    ).astype(int)

    out["mfi_above_zero"] = (
        (out["mfi_fast"] > 0) &
        (out["mfi_slow"] > 0)
    ).astype(int)

    out["mfi_below_zero"] = (
        (out["mfi_fast"] < 0) &
        (out["mfi_slow"] < 0)
    ).astype(int)

    # Fallback if raw MFI values are unavailable
    out["mfi_long_zero_ok"] = np.where(
        ((out["mfi_fast"] != 0) | (out["mfi_slow"] != 0)),
        out["mfi_above_zero"],
        (out["mfi_dir"] == 1).astype(int),
    ).astype(int)

    out["mfi_short_zero_ok"] = np.where(
        ((out["mfi_fast"] != 0) | (out["mfi_slow"] != 0)),
        out["mfi_below_zero"],
        (out["mfi_dir"] == -1).astype(int),
    ).astype(int)

    out["s15_long_mfi_ok"] = (
        (out["mfi_long_cross"] == 1) &
        (out["mfi_long_zero_ok"] == 1)
    ).astype(int)

    out["s15_short_mfi_ok"] = (
        (out["mfi_short_cross"] == 1) &
        (out["mfi_short_zero_ok"] == 1)
    ).astype(int)

    # =========================================================================
    # CORE S1.5 CONDITIONS
    # =========================================================================

    out["s15_long_bias_ok"] = (
        (out["bias_dir"] == 1) &
        (out["trend_dir"] == 1) &
        (out["ema_dir"] == 1)
    ).astype(int)

    out["s15_short_bias_ok"] = (
        (out["bias_dir"] == -1) &
        (out["trend_dir"] == -1) &
        (out["ema_dir"] == -1)
    ).astype(int)

    out["s15_long_setup_ok"] = (
        (out["rt_ema_100200"] == 1) &
        (out["s15_long_bias_ok"] == 1) &
        (out["s15_long_mfi_ok"] == 1) &
        (out["macd_long_ok"] == 1) &
        (out["ai_long_ok"] == 1) &
        (out["ema_distance_ok"] == 1) &
        (out["continuation_context_ok"] == 1)
    ).astype(int)

    out["s15_short_setup_ok"] = (
        (out["rt_ema_100200"] == 1) &
        (out["s15_short_bias_ok"] == 1) &
        (out["s15_short_mfi_ok"] == 1) &
        (out["macd_short_ok"] == 1) &
        (out["ai_short_ok"] == 1) &
        (out["ema_distance_ok"] == 1) &
        (out["continuation_context_ok"] == 1)
    ).astype(int)

    # =========================================================================
    # STATE / DIRECTION
    # =========================================================================

    out["s15_dir"] = pd.Series(
        np.select(
            [
                out["s15_long_setup_ok"] == 1,
                out["s15_short_setup_ok"] == 1,
            ],
            [1, -1],
            default=0,
        ),
        index=out.index,
    ).astype(int)

    out["s15_dir_label"] = _label_from_dir(out["s15_dir"])

    out["s15_candidate"] = (
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] != 0)
    ).astype(int)

    out["s15_ready"] = (out["s15_dir"] != 0).astype(int)

    out["s15_building"] = (
        (out["s15_ready"] == 0) &
        (out["s15_candidate"] == 1) &
        (out["continuation_context_ok"] == 1)
    ).astype(int)

    out["s15_state"] = pd.Series(
        np.select(
            [
                out["s15_ready"] == 1,
                out["s15_building"] == 1,
                out["s15_candidate"] == 1,
            ],
            [
                "ready",
                "building",
                "watch",
            ],
            default="blocked",
        ),
        index=out.index,
    )

    # =========================================================================
    # SCORE / GRADE
    # =========================================================================

    out["s15_score"] = (
        out["rt_ema_100200"] * 2
        + out["s15_long_bias_ok"] + out["s15_short_bias_ok"]
        + out["s15_long_mfi_ok"] + out["s15_short_mfi_ok"]
        + out["macd_long_ok"] + out["macd_short_ok"]
        + out["ai_long_ok"] + out["ai_short_ok"]
        + out["ema_distance_ok"]
        + out["continuation_context_ok"] * 2
        + _bool_from_threshold(out["truth_strength"], 5)
        + _bool_from_threshold(out["truth_strength"], 9)
    )

    out["s15_grade"] = pd.Series(
        np.select(
            [
                out["s15_score"] <= 4,
                (out["s15_score"] >= 5) & (out["s15_score"] <= 7),
                (out["s15_score"] >= 8) & (out["s15_score"] <= 10),
                out["s15_score"] >= 11,
            ],
            [
                "C",
                "B",
                "A",
                "A+",
            ],
            default="C",
        ),
        index=out.index,
    )

    # =========================================================================
    # REASON / DEBUG
    # =========================================================================

    out["s15_reason"] = pd.Series("blocked", index=out.index)

    out.loc[out["s15_candidate"] == 1, "s15_reason"] = "candidate"
    out.loc[out["s15_building"] == 1, "s15_reason"] = "waiting_for_full_alignment"
    out.loc[out["s15_ready"] == 1, "s15_reason"] = "ema_100200_rt_mfi_macd_ai_aligned"

    out.loc[
        (out["rt_ema_100200"] == 0),
        "s15_reason"
    ] = "no_ema_100200_retest"

    out.loc[
        (out["rt_ema_100200"] == 1) & (out["bias_dir"] == 0),
        "s15_reason"
    ] = "no_bias_direction"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] != 0) &
        (out["continuation_context_ok"] == 0),
        "s15_reason"
    ] = "truth_context_not_ready"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s15_long_mfi_ok"] == 0),
        "s15_reason"
    ] = "long_mfi_not_confirmed_above_zero"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s15_short_mfi_ok"] == 0),
        "s15_reason"
    ] = "short_mfi_not_confirmed_below_zero"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s15_long_mfi_ok"] == 1) &
        (out["macd_long_ok"] == 0),
        "s15_reason"
    ] = "macd_not_bullish_above_zero"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s15_short_mfi_ok"] == 1) &
        (out["macd_short_ok"] == 0),
        "s15_reason"
    ] = "macd_not_bearish_below_zero"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s15_long_mfi_ok"] == 1) &
        (out["macd_long_ok"] == 1) &
        (out["ai_long_ok"] == 0),
        "s15_reason"
    ] = "ai_not_bullish"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s15_short_mfi_ok"] == 1) &
        (out["macd_short_ok"] == 1) &
        (out["ai_short_ok"] == 0),
        "s15_reason"
    ] = "ai_not_bearish"

    out.loc[
        (out["rt_ema_100200"] == 1) &
        (out["bias_dir"] != 0) &
        (out["ema_distance_ok"] == 0),
        "s15_reason"
    ] = "ema_distance_not_continuation"

    out.loc[out["s15_ready"] == 1, "s15_reason"] = "ema_100200_rt_mfi_macd_ai_aligned"

    # =========================================================================
    # CLEAN EXPORT FIELDS
    # =========================================================================

    out["strategy_name"] = "S1.5"
    out["strategy_tier"] = "package_2"
    out["entry_type"] = "ema_100200_retest"
    out["trade_ready"] = out["s15_ready"]
    out["direction"] = out["s15_dir_label"]
    out["grade"] = out["s15_grade"]
    out["state"] = out["s15_state"]
    out["reason"] = out["s15_reason"]

    return out


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "trend_dir": [1, -1, 1, 1],
            "ema_dir": [1, -1, 1, 1],
            "bias_dir": [1, -1, 1, 0],
            "rt_ema_100200": [1, 1, 1, 0],
            "ema_distance_stage": [1, 2, 3, 1],
            "ai_dir": [1, -1, 1, 1],
            "macd_line": [0.5, -0.4, 0.2, 0.0],
            "mfi_fast": [12, -10, 8, 0],
            "mfi_slow": [5, -6, 7, 0],
            "mfi_cross_up": [1, 0, 0, 0],
            "mfi_cross_down": [0, 1, 0, 0],
            "continuation_context_ok": [1, 1, 1, 0],
            "truth_strength": [10, 9, 8, 2],
        }
    )

    s15 = build_strategy_s1_5(sample)
    print(s15.tail())