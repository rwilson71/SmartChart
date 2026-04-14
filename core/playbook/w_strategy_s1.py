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
# S1 STRATEGY ENGINE v1
# -----------------------------------------------------------------------------
# Locked S1 v1 logic:
# - EMA 14-20 retest required
# - MFI fast > slow and above zero for buy
# - MFI fast < slow and below zero for sell
# - EMA distance calibration bullish/bearish agreement required
# - AI RSI agreement required
# - MACD reversal agreement required
# - Truth context filter required
#
# Future upgrades (not included yet):
# - MTF confirmation
# - Forecaster confirmation
# =============================================================================

def build_strategy_s1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build SmartChart S1 strategy table from the truth-engine-enriched DataFrame.

    Expected input:
    - Preferably the output of truth_engine.build_truth(...)
    - Can also tolerate partially merged raw frames if key fields are present

    Output:
    - S1 strategy-specific table with direction, readiness, grade, and reasons
    """

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
    # CORE INPUTS FROM TRUTH ENGINE / MODULES
    # =========================================================================

    # Bias / truth
    out["bias_dir"] = _normalize_dir_series(_pick_first_existing(df, ["bias_dir", "dir"], 0))
    out["bias_label"] = _label_from_dir(out["bias_dir"])

    out["truth_state"] = _series_or_default(df, "truth_state", "blocked")
    out["truth_strength"] = _safe_numeric(_series_or_default(df, "truth_strength", 0), 0)
    out["truth_grade"] = _series_or_default(df, "truth_grade", "C")
    out["signal_quality"] = _series_or_default(df, "signal_quality", "low")

    out["continuation_context_ok"] = _normalize_bool_series(
        _series_or_default(df, "continuation_context_ok", 0)
    )
    out["continuation_entry_ok"] = _normalize_bool_series(
        _series_or_default(df, "continuation_entry_ok", 0)
    )
    out["trade_ready"] = _normalize_bool_series(_series_or_default(df, "trade_ready", 0))
    out["context_ok"] = _normalize_bool_series(_series_or_default(df, "context_ok", 0))

    # EMA / structure
    out["rt_ema_1420"] = _normalize_bool_series(_series_or_default(df, "rt_ema_1420", 0))
    out["ema_dir"] = _normalize_dir_series(_series_or_default(df, "ema_dir", 0))
    out["trend_dir"] = _normalize_dir_series(_series_or_default(df, "trend_dir", 0))

    out["ema_distance_stage"] = _safe_numeric(_series_or_default(df, "ema_distance_stage", 0), 0)
    out["ema_distance_label"] = _series_or_default(df, "ema_distance_label", "transition")
    out["ema_stage_optimal"] = (out["ema_distance_stage"] == 1).astype(int)
    out["ema_stage_extended"] = (out["ema_distance_stage"] == 2).astype(int)
    out["ema_stage_transition"] = (out["ema_distance_stage"] == 0).astype(int)
    out["ema_stage_exhausted"] = (out["ema_distance_stage"] == 3).astype(int)

    # AI
    out["ai_dir"] = _normalize_dir_series(_series_or_default(df, "ai_dir", 0))
    out["ai_strength"] = _safe_numeric(_series_or_default(df, "ai_strength", 0), 0)

    # MACD reversal
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

    # MFI
    out["mfi_dir"] = _normalize_dir_series(_series_or_default(df, "mfi_dir", 0))
    out["mfi_signal"] = _normalize_bool_series(_series_or_default(df, "mfi_signal", 0))

    # Support raw MFI columns if present
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

    # Exhaustion safety
    out["exhaustion_state"] = _safe_numeric(_series_or_default(df, "exhaustion_state", 0), 0)

    # =========================================================================
    # MFI CROSS LOGIC
    # -----------------------------------------------------------------------------
    # Preferred:
    # - use explicit cross columns if available
    # Fallback:
    # - infer from mfi_dir / mfi_signal
    # Zero-line condition:
    # - fast and slow above 0 for long
    # - fast and slow below 0 for short
    # =========================================================================

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

    # If raw MFI values are unavailable, allow directional MFI signal to stand in
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

    # =========================================================================
    # EMA DISTANCE AGREEMENT
    # -----------------------------------------------------------------------------
    # S1 should allow continuation states only:
    # - Stage 1 = best
    # - Stage 2 = allowed
    # - Stage 0 = no entry
    # - Stage 3 = no continuation entry
    # =========================================================================

    out["ema_distance_long_ok"] = (
        (out["bias_dir"] == 1) &
        out["ema_distance_stage"].isin([1, 2])
    ).astype(int)

    out["ema_distance_short_ok"] = (
        (out["bias_dir"] == -1) &
        out["ema_distance_stage"].isin([1, 2])
    ).astype(int)

    # =========================================================================
    # AI / MACD AGREEMENT
    # =========================================================================

    out["ai_long_ok"] = (out["ai_dir"] == 1).astype(int)
    out["ai_short_ok"] = (out["ai_dir"] == -1).astype(int)

    out["macd_long_ok"] = (
        (out["macd_rev_dir"] == 1) &
        (
            (out["macd_rev_signal"] == 1) |
            (out["macd_rev_strength"] > 0)
        )
    ).astype(int)

    out["macd_short_ok"] = (
        (out["macd_rev_dir"] == -1) &
        (
            (out["macd_rev_signal"] == 1) |
            (out["macd_rev_strength"] > 0)
        )
    ).astype(int)

    # =========================================================================
    # CORE S1 CONDITIONS
    # =========================================================================

    out["s1_long_bias_ok"] = (
        (out["bias_dir"] == 1) &
        (out["trend_dir"] == 1) &
        (out["ema_dir"] == 1)
    ).astype(int)

    out["s1_short_bias_ok"] = (
        (out["bias_dir"] == -1) &
        (out["trend_dir"] == -1) &
        (out["ema_dir"] == -1)
    ).astype(int)

    out["s1_long_mfi_ok"] = (
        (out["mfi_long_cross"] == 1) &
        (out["mfi_long_zero_ok"] == 1)
    ).astype(int)

    out["s1_short_mfi_ok"] = (
        (out["mfi_short_cross"] == 1) &
        (out["mfi_short_zero_ok"] == 1)
    ).astype(int)

    out["s1_long_setup_ok"] = (
        (out["rt_ema_1420"] == 1) &
        (out["s1_long_bias_ok"] == 1) &
        (out["s1_long_mfi_ok"] == 1) &
        (out["ema_distance_long_ok"] == 1) &
        (out["ai_long_ok"] == 1) &
        (out["macd_long_ok"] == 1) &
        (out["continuation_context_ok"] == 1) &
        (out["exhaustion_state"] < 3)
    ).astype(int)

    out["s1_short_setup_ok"] = (
        (out["rt_ema_1420"] == 1) &
        (out["s1_short_bias_ok"] == 1) &
        (out["s1_short_mfi_ok"] == 1) &
        (out["ema_distance_short_ok"] == 1) &
        (out["ai_short_ok"] == 1) &
        (out["macd_short_ok"] == 1) &
        (out["continuation_context_ok"] == 1) &
        (out["exhaustion_state"] < 3)
    ).astype(int)

    # =========================================================================
    # DIRECTION / READY STATE
    # =========================================================================

    out["s1_dir"] = pd.Series(
        np.select(
            [
                out["s1_long_setup_ok"] == 1,
                out["s1_short_setup_ok"] == 1,
            ],
            [1, -1],
            default=0,
        ),
        index=out.index,
    ).astype(int)

    out["s1_dir_label"] = _label_from_dir(out["s1_dir"])

    out["s1_candidate"] = (
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] != 0)
    ).astype(int)

    out["s1_ready"] = (out["s1_dir"] != 0).astype(int)

    out["s1_building"] = (
        (out["s1_ready"] == 0) &
        (out["s1_candidate"] == 1) &
        (out["continuation_context_ok"] == 1)
    ).astype(int)

    out["s1_state"] = pd.Series(
        np.select(
            [
                out["s1_ready"] == 1,
                out["s1_building"] == 1,
                out["s1_candidate"] == 1,
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
    # SCORING / GRADE
    # =========================================================================

    out["s1_score"] = (
        out["rt_ema_1420"] * 2
        + out["s1_long_bias_ok"] + out["s1_short_bias_ok"]
        + out["s1_long_mfi_ok"] + out["s1_short_mfi_ok"]
        + out["ema_distance_long_ok"] + out["ema_distance_short_ok"]
        + out["ai_long_ok"] + out["ai_short_ok"]
        + out["macd_long_ok"] + out["macd_short_ok"]
        + out["continuation_context_ok"] * 2
        + _bool_from_threshold(out["truth_strength"], 5)
        + _bool_from_threshold(out["truth_strength"], 9)
        - (out["exhaustion_state"] >= 3).astype(int) * 2
    )

    out["s1_grade"] = pd.Series(
        np.select(
            [
                out["s1_score"] <= 4,
                (out["s1_score"] >= 5) & (out["s1_score"] <= 7),
                (out["s1_score"] >= 8) & (out["s1_score"] <= 10),
                out["s1_score"] >= 11,
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
    # REASON / DEBUG LAYER
    # =========================================================================

    out["s1_reason"] = pd.Series("blocked", index=out.index)

    out.loc[out["s1_candidate"] == 1, "s1_reason"] = "candidate"
    out.loc[out["s1_building"] == 1, "s1_reason"] = "waiting_for_full_alignment"
    out.loc[out["s1_ready"] == 1, "s1_reason"] = "ema_1420_rt_mfi_ai_macd_distance_aligned"

    out.loc[
        (out["rt_ema_1420"] == 0),
        "s1_reason"
    ] = "no_ema_1420_retest"

    out.loc[
        (out["rt_ema_1420"] == 1) & (out["bias_dir"] == 0),
        "s1_reason"
    ] = "no_bias_direction"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] != 0) &
        (out["continuation_context_ok"] == 0),
        "s1_reason"
    ] = "truth_context_not_ready"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s1_long_mfi_ok"] == 0),
        "s1_reason"
    ] = "long_mfi_not_confirmed"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s1_short_mfi_ok"] == 0),
        "s1_reason"
    ] = "short_mfi_not_confirmed"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s1_long_mfi_ok"] == 1) &
        (out["ema_distance_long_ok"] == 0),
        "s1_reason"
    ] = "ema_distance_not_bullish"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s1_short_mfi_ok"] == 1) &
        (out["ema_distance_short_ok"] == 0),
        "s1_reason"
    ] = "ema_distance_not_bearish"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s1_long_mfi_ok"] == 1) &
        (out["ema_distance_long_ok"] == 1) &
        (out["ai_long_ok"] == 0),
        "s1_reason"
    ] = "ai_not_bullish"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s1_short_mfi_ok"] == 1) &
        (out["ema_distance_short_ok"] == 1) &
        (out["ai_short_ok"] == 0),
        "s1_reason"
    ] = "ai_not_bearish"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == 1) &
        (out["s1_long_mfi_ok"] == 1) &
        (out["ema_distance_long_ok"] == 1) &
        (out["ai_long_ok"] == 1) &
        (out["macd_long_ok"] == 0),
        "s1_reason"
    ] = "macd_not_bullish"

    out.loc[
        (out["rt_ema_1420"] == 1) &
        (out["bias_dir"] == -1) &
        (out["s1_short_mfi_ok"] == 1) &
        (out["ema_distance_short_ok"] == 1) &
        (out["ai_short_ok"] == 1) &
        (out["macd_short_ok"] == 0),
        "s1_reason"
    ] = "macd_not_bearish"

    out.loc[
        (out["exhaustion_state"] >= 3),
        "s1_reason"
    ] = "blocked_by_exhaustion"

    # Ensure ready has final priority
    out.loc[out["s1_ready"] == 1, "s1_reason"] = "ema_1420_rt_mfi_ai_macd_distance_aligned"

    # =========================================================================
    # CLEAN WEBSITE / API FIELDS
    # =========================================================================

    out["strategy_name"] = "S1"
    out["strategy_tier"] = "basic"
    out["entry_type"] = "ema_1420_retest"
    out["trade_ready"] = out["s1_ready"]
    out["direction"] = out["s1_dir_label"]
    out["grade"] = out["s1_grade"]
    out["state"] = out["s1_state"]
    out["reason"] = out["s1_reason"]

    return out


# =============================================================================
# TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "trend_dir": [1, -1, 1, 1],
            "ema_dir": [1, -1, 1, 1],
            "bias_dir": [1, -1, 1, 0],
            "rt_ema_1420": [1, 1, 1, 0],
            "ema_distance_stage": [1, 2, 3, 1],
            "ai_dir": [1, -1, 1, 1],
            "macd_rev_dir": [1, -1, 1, 1],
            "macd_rev_signal": [1, 1, 1, 0],
            "macd_rev_strength": [2, 2, 1, 0],
            "mfi_dir": [1, -1, 1, 1],
            "mfi_signal": [1, 1, 1, 0],
            "mfi_fast": [10, -12, 8, 5],
            "mfi_slow": [4, -8, 6, 3],
            "continuation_context_ok": [1, 1, 1, 0],
            "truth_strength": [10, 9, 8, 2],
            "exhaustion_state": [1, 1, 3, 0],
        }
    )

    s1 = build_strategy_s1(sample)
    print(s1.tail())