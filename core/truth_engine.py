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


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_text(series: pd.Series, default: str = "") -> pd.Series:
    return series.astype(str).fillna(default) if len(series) else pd.Series(default, index=series.index)


def _normalize_dir_series(series: pd.Series) -> pd.Series:
    s = _safe_numeric(series, 0)
    return pd.Series(
        np.where(s > 0, 1, np.where(s < 0, -1, 0)),
        index=series.index,
    ).astype(int)


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    return _safe_numeric(series, 0).fillna(0).astype(bool).astype(int)


def _same_direction(a: pd.Series, b: pd.Series) -> pd.Series:
    a1 = _normalize_dir_series(a)
    b1 = _normalize_dir_series(b)
    return ((a1 != 0) & (a1 == b1)).astype(int)


def _opposite_direction(a: pd.Series, b: pd.Series) -> pd.Series:
    a1 = _normalize_dir_series(a)
    b1 = _normalize_dir_series(b)
    return ((a1 != 0) & (b1 != 0) & (a1 == -b1)).astype(int)


def _label_from_dir(series: pd.Series) -> pd.Series:
    s = _normalize_dir_series(series)
    return pd.Series(
        np.where(s > 0, "bullish", np.where(s < 0, "bearish", "neutral")),
        index=series.index,
    )


def _clip_int(series: pd.Series, low: int, high: int) -> pd.Series:
    return _safe_numeric(series, 0).clip(low, high).astype(int)


def _bool_from_threshold(series: pd.Series, threshold: float) -> pd.Series:
    return (_safe_numeric(series, 0) >= threshold).astype(int)


def _pick_first_existing(df: pd.DataFrame, names: list[str], default=0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _sum_series(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        raise ValueError("No series provided to _sum_series")
    out = parts[0].copy()
    for p in parts[1:]:
        out = out + p
    return out


# =============================================================================
# EMA DISTANCE STAGE
# LOCKED WORKING RESEARCH STATE:
# Stage 0 = 0.00% to 0.20%
# Stage 1 = 0.20% to 0.50%
# Stage 2 = 0.50% to 0.75%
# Stage 3 = 0.75%+
# =============================================================================

def _ema_distance_stage_from_pct(pct_series: pd.Series) -> pd.Series:
    p = _safe_numeric(pct_series, 0).abs()

    return pd.Series(
        np.select(
            [
                p < 0.20,
                (p >= 0.20) & (p < 0.50),
                (p >= 0.50) & (p < 0.75),
                p >= 0.75,
            ],
            [0, 1, 2, 3],
            default=0,
        ),
        index=p.index,
    ).astype(int)


def _ema_distance_label(stage_series: pd.Series) -> pd.Series:
    s = _clip_int(stage_series, 0, 3)
    return pd.Series(
        np.select(
            [
                s == 0,
                s == 1,
                s == 2,
                s == 3,
            ],
            [
                "transition",
                "continuation_optimal",
                "continuation_extended",
                "reversal_exhausted",
            ],
            default="transition",
        ),
        index=s.index,
    )


# =============================================================================
# MAIN TRUTH ENGINE
# =============================================================================

def build_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    SmartChart Truth Engine v2

    Purpose:
    - Aggregate all module outputs into a layered institutional-grade truth table.
    - Separate bias, structure, setup, reversal pressure, context, and forecast.
    - Preserve specific retest families for future playbook / scanner execution.

    Design:
    - Safe with missing columns
    - Uses strict directional normalization
    - Continuation and reversal contexts are handled separately
    """

    out = pd.DataFrame(index=df.index)

    # =========================================================================
    # BASE / TIMESTAMP / PRICE PASSTHROUGH
    # =========================================================================

    if "timestamp" in df.columns:
        out["timestamp"] = df["timestamp"]
    else:
        out["timestamp"] = pd.Series(df.index, index=df.index)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            out[col] = df[col]

    # =========================================================================
    # CONTEXT INPUTS
    # =========================================================================

    out["session_active"] = _normalize_bool_series(_series_or_default(df, "session_active", 0))
    out["session_name"] = _series_or_default(df, "session_name", "none")
    out["day_of_week"] = _series_or_default(df, "day_of_week", "")

    out["market_phase"] = _safe_numeric(_series_or_default(df, "market_phase", 0), 0)
    out["market_phase_label"] = _series_or_default(df, "market_phase_label", "unknown")

    out["regime"] = _safe_numeric(_pick_first_existing(df, ["regime", "regime_state"], 0), 0)
    out["regime_label"] = _series_or_default(df, "regime_label", "unknown")

    out["volatility_state"] = _safe_numeric(_pick_first_existing(df, ["volatility_state", "vol_state"], 0), 0)
    out["volatility_label"] = _series_or_default(df, "volatility_label", "unknown")

    out["volume_state"] = _safe_numeric(_pick_first_existing(df, ["volume_state", "volm_state"], 0), 0)
    out["volume_label"] = _series_or_default(df, "volume_label", "unknown")

    out["market_condition"] = _safe_numeric(_series_or_default(df, "market_condition", 0), 0)
    out["market_condition_label"] = _series_or_default(df, "market_condition_label", "unknown")

    # =========================================================================
    # CORE BIAS INPUTS
    # =========================================================================

    out["trend_dir"] = _normalize_dir_series(_series_or_default(df, "trend_dir", 0))
    out["trend_strength"] = _safe_numeric(_series_or_default(df, "trend_strength", 0), 0)

    out["ema_dir"] = _normalize_dir_series(_series_or_default(df, "ema_dir", 0))
    out["ema_quality"] = _safe_numeric(_series_or_default(df, "ema_quality", 0), 0)
    out["ema_stack_bull"] = _normalize_bool_series(_series_or_default(df, "ema_stack_bull", 0))
    out["ema_stack_bear"] = _normalize_bool_series(_series_or_default(df, "ema_stack_bear", 0))

    out["ema_slope_fast"] = _normalize_dir_series(_series_or_default(df, "ema_slope_fast", 0))
    out["ema_slope_band"] = _normalize_dir_series(_series_or_default(df, "ema_slope_band", 0))
    out["ema_slope_slow"] = _normalize_dir_series(_series_or_default(df, "ema_slope_slow", 0))

    out["ema14_slope"] = _normalize_dir_series(_series_or_default(df, "ema14_slope", 0))
    out["ema20_slope"] = _normalize_dir_series(_series_or_default(df, "ema20_slope", 0))
    out["ema33_slope"] = _normalize_dir_series(_series_or_default(df, "ema33_slope", 0))
    out["ema50_slope"] = _normalize_dir_series(_series_or_default(df, "ema50_slope", 0))
    out["ema100_slope"] = _normalize_dir_series(_series_or_default(df, "ema100_slope", 0))
    out["ema200_slope"] = _normalize_dir_series(_series_or_default(df, "ema200_slope", 0))

    out["price_to_ema20_pct"] = _safe_numeric(_series_or_default(df, "price_to_ema20_pct", 0), 0)
    out["price_to_ema20_pts"] = _safe_numeric(_series_or_default(df, "price_to_ema20_pts", 0), 0)

    out["ema20_to_ema200_pct"] = _safe_numeric(
        _pick_first_existing(df, ["ema20_to_ema200_pct", "sc_ema20_to_ema200_pct", "ema_distance_pct"], 0),
        0,
    )
    out["ema20_to_ema200_pts"] = _safe_numeric(
        _pick_first_existing(df, ["ema20_to_ema200_pts", "sc_ema20_to_ema200_pts", "ema_distance_pts"], 0),
        0,
    )

    out["ema_distance_stage"] = _ema_distance_stage_from_pct(out["ema20_to_ema200_pct"])
    out["ema_distance_label"] = _ema_distance_label(out["ema_distance_stage"])

    out["momentum_dir"] = _normalize_dir_series(_series_or_default(df, "momentum_dir", 0))
    out["momentum_strength"] = _safe_numeric(_series_or_default(df, "momentum_strength", 0), 0)

    out["ai_dir"] = _normalize_dir_series(_series_or_default(df, "ai_dir", 0))
    out["ai_strength"] = _safe_numeric(_series_or_default(df, "ai_strength", 0), 0)
    out["ai_bias_score"] = _safe_numeric(_series_or_default(df, "ai_bias_score", 0), 0)

    # =========================================================================
    # RETEST INPUTS
    # =========================================================================

    out["rt_ema_1420"] = _normalize_bool_series(_series_or_default(df, "rt_ema_1420", 0))
    out["rt_ema_3350"] = _normalize_bool_series(_series_or_default(df, "rt_ema_3350", 0))
    out["rt_ema_100200"] = _normalize_bool_series(_series_or_default(df, "rt_ema_100200", 0))

    out["ema_rt_any"] = _normalize_bool_series(_series_or_default(df, "ema_rt_any", 0))
    out["ema_rt_family"] = _safe_numeric(_series_or_default(df, "ema_rt_family", 0), 0)
    out["ema_rt_dir"] = _normalize_dir_series(_series_or_default(df, "ema_rt_dir", 0))

    out["rt_fib"] = _normalize_bool_series(_series_or_default(df, "rt_fib", 0))
    out["rt_session"] = _normalize_bool_series(_series_or_default(df, "rt_session", 0))
    out["rt_orderflow"] = _normalize_bool_series(_series_or_default(df, "rt_orderflow", 0))
    out["rt_confluence_cloud"] = _normalize_bool_series(_series_or_default(df, "rt_confluence_cloud", 0))
    out["rt_confluence"] = _normalize_bool_series(
        _pick_first_existing(df, ["rt_confluence", "rt_confluence_local"], 0)
    )

    out["rt_5m_high"] = _normalize_bool_series(
        _pick_first_existing(df, ["rt_5m_high", "rt_m5_high"], 0)
    )
    out["rt_5m_low"] = _normalize_bool_series(
        _pick_first_existing(df, ["rt_5m_low", "rt_m5_low"], 0)
    )
    out["rt_15m_high"] = _normalize_bool_series(
        _pick_first_existing(df, ["rt_15m_high", "rt_m15_high"], 0)
    )
    out["rt_15m_low"] = _normalize_bool_series(
        _pick_first_existing(df, ["rt_15m_low", "rt_m15_low"], 0)
    )

    # Backward compatibility with older merged flag if present
    out["rt_m5"] = _normalize_bool_series(_series_or_default(df, "rt_m5", 0))

    # =========================================================================
    # STRUCTURE / LOCATION INPUTS
    # =========================================================================

    out["market_structure_dir"] = _normalize_dir_series(
        _pick_first_existing(df, ["market_structure_dir", "ms_dir"], 0)
    )
    out["market_structure_state"] = _safe_numeric(
        _pick_first_existing(df, ["market_structure_state", "ms_state"], 0),
        0,
    )
    out["bos_dir"] = _normalize_dir_series(_series_or_default(df, "bos_dir", 0))
    out["choch_dir"] = _normalize_dir_series(_series_or_default(df, "choch_dir", 0))

    out["fib_dir"] = _normalize_dir_series(_series_or_default(df, "fib_dir", 0))
    out["fib_zone"] = _safe_numeric(_series_or_default(df, "fib_zone", 0), 0)
    out["fib_zone_label"] = _series_or_default(df, "fib_zone_label", "none")

    out["liquidity_dir"] = _normalize_dir_series(_series_or_default(df, "liquidity_dir", 0))
    out["liquidity_position"] = _safe_numeric(_series_or_default(df, "liquidity_position", 0), 0)
    out["liquidity_state"] = _safe_numeric(_series_or_default(df, "liquidity_state", 0), 0)

    out["fvg_dir"] = _normalize_dir_series(_series_or_default(df, "fvg_dir", 0))
    out["fvg_active"] = _normalize_bool_series(_series_or_default(df, "fvg_active", 0))

    out["orderflow_dir"] = _normalize_dir_series(_series_or_default(df, "orderflow_dir", 0))
    out["orderflow_zone"] = _safe_numeric(_series_or_default(df, "orderflow_zone", 0), 0)
    out["orderflow_strength"] = _safe_numeric(_series_or_default(df, "orderflow_strength", 0), 0)

    out["session_bias"] = _normalize_dir_series(_series_or_default(df, "session_bias", 0))
    out["daily_bias"] = _normalize_dir_series(_series_or_default(df, "daily_bias", 0))

    # =========================================================================
    # CONFLUENCE INPUTS
    # =========================================================================

    out["confluence_score"] = _safe_numeric(_series_or_default(df, "confluence_score", 0), 0)
    out["confluence_dir"] = _normalize_dir_series(_series_or_default(df, "confluence_dir", 0))
    out["confluence_active"] = _normalize_bool_series(_series_or_default(df, "confluence_active", 0))

    out["cloud_dir"] = _normalize_dir_series(
        _pick_first_existing(df, ["confluence_cloud_dir", "cloud_dir"], 0)
    )
    out["cloud_active"] = _normalize_bool_series(
        _pick_first_existing(df, ["confluence_cloud_active", "cloud_active"], 0)
    )

    # =========================================================================
    # REVERSAL / PRESSURE INPUTS
    # =========================================================================

    out["macd_rev_dir"] = _normalize_dir_series(_series_or_default(df, "macd_reversal_dir", 0))
    out["macd_rev_strength"] = _safe_numeric(_series_or_default(df, "macd_reversal_strength", 0), 0)
    out["macd_rev_signal"] = _normalize_bool_series(_series_or_default(df, "macd_reversal_signal", 0))
    out["macd_rev_hist_state"] = _safe_numeric(_series_or_default(df, "macd_reversal_hist_state", 0), 0)

    out["ob_os_state"] = _safe_numeric(_series_or_default(df, "ob_os_state", 0), 0)
    out["ob_os_dir"] = _normalize_dir_series(_series_or_default(df, "ob_os_dir", 0))

    out["exhaustion_state"] = _safe_numeric(_series_or_default(df, "exhaustion_state", 0), 0)
    out["exhaustion_dir"] = _normalize_dir_series(_series_or_default(df, "exhaustion_dir", 0))
    out["exhaustion_signal"] = _normalize_bool_series(_series_or_default(df, "exhaustion_signal", 0))

    out["divergence_flag"] = _normalize_bool_series(_series_or_default(df, "divergence_flag", 0))
    out["divergence_dir"] = _normalize_dir_series(_series_or_default(df, "divergence_dir", 0))

    out["mfi_state"] = _safe_numeric(_series_or_default(df, "mfi_state", 0), 0)
    out["mfi_dir"] = _normalize_dir_series(_series_or_default(df, "mfi_dir", 0))
    out["mfi_signal"] = _normalize_bool_series(_series_or_default(df, "mfi_signal", 0))

    # =========================================================================
    # PULLBACK / RETEST INTELLIGENCE
    # =========================================================================

    out["pullback_dir"] = _normalize_dir_series(_series_or_default(df, "pullback_dir", 0))
    out["pullback_active"] = _normalize_bool_series(_series_or_default(df, "pullback_active", 0))
    out["pullback_quality"] = _safe_numeric(_series_or_default(df, "pullback_quality", 0), 0)

    # =========================================================================
    # FORECASTER INPUTS
    # =========================================================================

    out["forecast_dir"] = _normalize_dir_series(_series_or_default(df, "forecast_dir", 0))
    out["forecast_confidence"] = _safe_numeric(_series_or_default(df, "forecast_confidence", 0), 0)
    out["forecast_agreement"] = _safe_numeric(_series_or_default(df, "forecast_agreement", 0), 0)
    out["forecast_long_pct"] = _safe_numeric(_series_or_default(df, "forecast_long_pct", 0), 0)
    out["forecast_short_pct"] = _safe_numeric(_series_or_default(df, "forecast_short_pct", 0), 0)

    # =========================================================================
    # CORE BIAS LAYER
    # =========================================================================
    # Anchor = trend + EMA
    # Support = momentum + AI
    # EMA distance stage modifies continuation quality later

    out["anchor_align"] = _same_direction(out["trend_dir"], out["ema_dir"])
    out["support_momentum_align"] = _same_direction(out["momentum_dir"], out["trend_dir"])
    out["support_ai_align"] = _same_direction(out["ai_dir"], out["trend_dir"])

    out["anchor_bias_dir"] = pd.Series(
        np.where(
            (out["trend_dir"] != 0) & (out["trend_dir"] == out["ema_dir"]),
            out["trend_dir"],
            0,
        ),
        index=out.index,
    ).astype(int)

    out["support_vote_raw"] = out["momentum_dir"] + out["ai_dir"]
    out["support_bias_dir"] = _normalize_dir_series(out["support_vote_raw"])

    out["bias_vote_raw"] = (
        out["trend_dir"] * 3
        + out["ema_dir"] * 3
        + out["momentum_dir"] * 1
        + out["ai_dir"] * 1
    )
    out["bias_dir"] = _normalize_dir_series(out["bias_vote_raw"])
    out["bias_label"] = _label_from_dir(out["bias_dir"])

    out["anchor_bias_ok"] = (
        (out["anchor_bias_dir"] != 0) &
        (out["anchor_bias_dir"] == out["bias_dir"])
    ).astype(int)

    out["support_bias_ok"] = (
        (out["bias_dir"] != 0) &
        (
            _same_direction(out["momentum_dir"], out["bias_dir"])
            + _same_direction(out["ai_dir"], out["bias_dir"])
            >= 1
        )
    ).astype(int)

    out["core_alignment_score"] = (
        out["anchor_align"] * 2
        + _same_direction(out["trend_dir"], out["bias_dir"])
        + _same_direction(out["ema_dir"], out["bias_dir"])
        + _same_direction(out["momentum_dir"], out["bias_dir"])
        + _same_direction(out["ai_dir"], out["bias_dir"])
    )

    out["bias_conflict_score"] = (
        _opposite_direction(out["momentum_dir"], out["bias_dir"])
        + _opposite_direction(out["ai_dir"], out["bias_dir"])
    )

    out["core_bias_confirmed"] = (
        (out["bias_dir"] != 0) &
        (out["anchor_bias_ok"] == 1) &
        (out["core_alignment_score"] >= 4)
    ).astype(int)

    # =========================================================================
    # EMA DISTANCE QUALITY / TREND MATURITY LAYER
    # =========================================================================

    out["ema_stage_continuation_ok"] = out["ema_distance_stage"].isin([1, 2]).astype(int)
    out["ema_stage_optimal"] = (out["ema_distance_stage"] == 1).astype(int)
    out["ema_stage_extended"] = (out["ema_distance_stage"] == 2).astype(int)
    out["ema_stage_transition"] = (out["ema_distance_stage"] == 0).astype(int)
    out["ema_stage_exhausted"] = (out["ema_distance_stage"] == 3).astype(int)

    out["trend_maturity_score"] = (
        out["ema_stage_optimal"] * 3
        + out["ema_stage_extended"] * 2
        + out["ema_stage_transition"] * 0
        + out["ema_stage_exhausted"] * -2
    )

    # =========================================================================
    # STRUCTURE / LOCATION LAYER
    # =========================================================================

    out["market_structure_aligned"] = _same_direction(out["market_structure_dir"], out["bias_dir"])
    out["bos_aligned"] = _same_direction(out["bos_dir"], out["bias_dir"])
    out["choch_aligned"] = _same_direction(out["choch_dir"], out["bias_dir"])
    out["fib_aligned"] = _same_direction(out["fib_dir"], out["bias_dir"])
    out["liquidity_aligned"] = _same_direction(out["liquidity_dir"], out["bias_dir"])
    out["fvg_aligned"] = _same_direction(out["fvg_dir"], out["bias_dir"])
    out["orderflow_aligned"] = _same_direction(out["orderflow_dir"], out["bias_dir"])
    out["session_bias_aligned"] = _same_direction(out["session_bias"], out["bias_dir"])
    out["daily_bias_aligned"] = _same_direction(out["daily_bias"], out["bias_dir"])

    out["structure_score"] = (
        out["market_structure_aligned"] * 2
        + out["bos_aligned"] * 2
        + out["fib_aligned"]
        + out["liquidity_aligned"]
        + out["fvg_aligned"]
        + out["orderflow_aligned"]
        + out["session_bias_aligned"]
        + out["daily_bias_aligned"]
    )

    out["location_score"] = (
        out["fib_aligned"]
        + out["liquidity_aligned"]
        + out["orderflow_aligned"]
        + out["session_bias_aligned"]
        + out["daily_bias_aligned"]
    )

    out["structure_ok"] = (
        (out["bias_dir"] != 0) &
        (out["market_structure_aligned"] == 1) &
        (out["structure_score"] >= 4)
    ).astype(int)

    out["location_ok"] = (
        (out["bias_dir"] != 0) &
        (out["location_score"] >= 2)
    ).astype(int)

    # =========================================================================
    # RETEST / LEVEL / ENTRY LAYER
    # =========================================================================

    out["structure_retest_score"] = (
        out["rt_fib"]
        + out["rt_session"]
        + out["rt_orderflow"]
        + out["rt_confluence_cloud"]
        + out["rt_5m_high"]
        + out["rt_5m_low"]
        + out["rt_15m_high"]
        + out["rt_15m_low"]
    )

    out["entry_retest_score"] = (
        out["rt_ema_1420"]
        + out["rt_ema_3350"]
        + out["rt_ema_100200"]
        + out["rt_confluence"]
        + out["rt_m5"]
    )

    out["retest_count"] = out["structure_retest_score"] + out["entry_retest_score"]
    out["has_retest"] = (out["retest_count"] > 0).astype(int)

    out["pullback_aligned"] = _same_direction(out["pullback_dir"], out["bias_dir"])
    out["ema_retest_dir_aligned"] = (
        (out["ema_rt_dir"] == 0) |
        (out["ema_rt_dir"] == out["bias_dir"])
    ).astype(int)

    out["pullback_ok"] = (
        (out["pullback_active"] == 1) &
        (out["pullback_aligned"] == 1)
    ).astype(int)

    out["entry_setup_valid"] = (
        (out["bias_dir"] != 0) &
        (out["entry_retest_score"] >= 1) &
        (out["pullback_ok"] == 1) &
        (out["ema_retest_dir_aligned"] == 1)
    ).astype(int)

    out["structure_retest_ok"] = (out["structure_retest_score"] >= 1).astype(int)

    # =========================================================================
    # CONFLUENCE LAYER
    # =========================================================================

    out["confluence_aligned"] = _same_direction(out["confluence_dir"], out["bias_dir"])
    out["cloud_aligned"] = _same_direction(out["cloud_dir"], out["bias_dir"])

    out["cloud_structure_ok"] = (
        ((out["cloud_active"] == 1) & (out["cloud_aligned"] == 1)) |
        (out["cloud_active"] == 0)
    ).astype(int)

    out["local_confluence_ok"] = (
        ((out["confluence_active"] == 1) & (out["confluence_aligned"] == 1)) |
        (out["confluence_active"] == 0)
    ).astype(int)

    out["confluence_quality_score"] = (
        out["cloud_aligned"] * 2
        + out["confluence_aligned"]
        + _bool_from_threshold(out["confluence_score"], 1)
    )

    # =========================================================================
    # REVERSAL PRESSURE LAYER
    # =========================================================================

    out["macd_rev_active"] = (
        (out["macd_rev_signal"] == 1) &
        (out["macd_rev_strength"] > 0)
    ).astype(int)

    out["macd_rev_opposes_bias"] = _opposite_direction(out["macd_rev_dir"], out["bias_dir"])
    out["divergence_opposes_bias"] = _opposite_direction(out["divergence_dir"], out["bias_dir"])
    out["exhaustion_opposes_bias"] = _opposite_direction(out["exhaustion_dir"], out["bias_dir"])
    out["mfi_opposes_bias"] = _opposite_direction(out["mfi_dir"], out["bias_dir"])
    out["ob_os_opposes_bias"] = _opposite_direction(out["ob_os_dir"], out["bias_dir"])

    out["reversal_pressure"] = (
        (out["macd_rev_active"] * (out["macd_rev_opposes_bias"] | (out["macd_rev_dir"] == 0))).astype(int)
        + out["divergence_flag"] * ((out["divergence_opposes_bias"] == 1) | (out["divergence_dir"] == 0)).astype(int)
        + out["exhaustion_signal"] * ((out["exhaustion_opposes_bias"] == 1) | (out["exhaustion_dir"] == 0)).astype(int)
        + out["mfi_signal"] * ((out["mfi_opposes_bias"] == 1) | (out["mfi_dir"] == 0)).astype(int)
    )

    out["reversal_pressure_level"] = pd.Series(
        np.select(
            [
                out["reversal_pressure"] <= 0,
                out["reversal_pressure"] == 1,
                out["reversal_pressure"] == 2,
                out["reversal_pressure"] >= 3,
            ],
            [
                0,   # none
                1,   # mild
                2,   # moderate
                3,   # high
            ],
            default=0,
        ),
        index=out.index,
    ).astype(int)

    out["reversal_pressure_label"] = pd.Series(
        np.select(
            [
                out["reversal_pressure_level"] == 0,
                out["reversal_pressure_level"] == 1,
                out["reversal_pressure_level"] == 2,
                out["reversal_pressure_level"] == 3,
            ],
            [
                "none",
                "mild",
                "moderate",
                "high",
            ],
            default="none",
        ),
        index=out.index,
    )

    # =========================================================================
    # FORECAST LAYER
    # =========================================================================

    out["forecast_match_dir"] = (
        (out["forecast_dir"] != 0) &
        (out["forecast_dir"] == out["bias_dir"])
    ).astype(int)

    out["forecast_confident"] = _bool_from_threshold(out["forecast_confidence"], 55)
    out["forecast_permission"] = (
        (out["forecast_match_dir"] == 1) &
        (out["forecast_confident"] == 1)
    ).astype(int)

    # =========================================================================
    # CONTEXT QUALITY LAYER
    # =========================================================================

    # These thresholds are intentionally tolerant because not every module
    # may be fully standardized yet.
    out["regime_ok"] = (
        (out["regime"] != 0) |
        (out["regime_label"].str.lower().isin(["trend", "normal", "expansion"]))
    ).astype(int)

    out["volume_ok"] = (
        (out["volume_state"] != 0) |
        (out["volume_label"].str.lower().isin(["normal", "expansion", "climax"]))
    ).astype(int)

    out["volatility_ok"] = (
        (out["volatility_state"] != 0) |
        (out["volatility_label"].str.lower().isin(["normal", "expansion", "trend"]))
    ).astype(int)

    out["market_condition_ok"] = (
        (out["market_condition"] != 0) |
        (out["market_condition_label"].str.lower().isin(["normal", "expansion", "trend"]))
    ).astype(int)

    out["context_score"] = (
        out["regime_ok"]
        + out["volume_ok"]
        + out["volatility_ok"]
        + out["market_condition_ok"]
        + out["session_active"]
    )

    out["context_ok"] = (out["context_score"] >= 2).astype(int)

    # =========================================================================
    # CONTINUATION CONTEXT
    # =========================================================================

    out["continuation_context_ok"] = (
        (out["bias_dir"] != 0) &
        (out["core_bias_confirmed"] == 1) &
        (out["structure_ok"] == 1) &
        (out["location_ok"] == 1) &
        (out["ema_stage_continuation_ok"] == 1) &
        (out["cloud_structure_ok"] == 1) &
        (out["context_ok"] == 1)
    ).astype(int)

    out["continuation_entry_ok"] = (
        (out["continuation_context_ok"] == 1) &
        (out["entry_setup_valid"] == 1) &
        (out["reversal_pressure_level"] <= 1)
    ).astype(int)

    # =========================================================================
    # REVERSAL CONTEXT
    # =========================================================================

    out["reversal_context_ok"] = (
        (out["bias_dir"] != 0) &
        (
            (out["ema_stage_exhausted"] == 1) |
            (out["reversal_pressure_level"] >= 2) |
            (out["macd_rev_active"] == 1) |
            (out["divergence_flag"] == 1) |
            (out["exhaustion_signal"] == 1)
        )
    ).astype(int)

    out["reversal_entry_hint"] = (
        (out["reversal_context_ok"] == 1) &
        (
            (out["rt_confluence"] == 1) |
            (out["rt_confluence_cloud"] == 1) |
            (out["rt_fib"] == 1) |
            (out["rt_session"] == 1) |
            (out["rt_orderflow"] == 1)
        )
    ).astype(int)

    # =========================================================================
    # INVALIDATION / CONFLICT LAYER
    # =========================================================================

    out["continuation_invalidated"] = (
        (out["ema_stage_exhausted"] == 1) |
        (out["reversal_pressure_level"] >= 3)
    ).astype(int)

    out["conflict_score"] = (
        out["bias_conflict_score"]
        + _opposite_direction(out["market_structure_dir"], out["bias_dir"])
        + _opposite_direction(out["orderflow_dir"], out["bias_dir"])
        + _opposite_direction(out["confluence_dir"], out["bias_dir"])
        + _opposite_direction(out["cloud_dir"], out["bias_dir"])
    )

    # =========================================================================
    # PLAYBOOK-STYLE RETEST CLASSIFICATION
    # =========================================================================

    out["s1_retest_ok"] = out["rt_ema_1420"]
    out["s15_retest_ok"] = out["rt_ema_100200"]
    out["band_retest_ok"] = out["rt_ema_3350"]

    out["continuation_setup_type"] = pd.Series(
        np.select(
            [
                out["s1_retest_ok"] == 1,
                out["s15_retest_ok"] == 1,
                out["band_retest_ok"] == 1,
            ],
            [
                "s1",
                "s1_5",
                "band_retest",
            ],
            default="none",
        ),
        index=out.index,
    )

    # =========================================================================
    # FINAL STRENGTH MODEL
    # =========================================================================

    out["truth_strength"] = (
        out["core_alignment_score"]
        + out["structure_score"]
        + out["structure_retest_score"]
        + out["entry_retest_score"]
        + out["confluence_quality_score"]
        + out["forecast_permission"]
        + out["context_score"]
        + out["trend_maturity_score"]
        - out["conflict_score"]
        - out["reversal_pressure_level"]
    )

    out["bias_strength"] = (
        out["anchor_bias_ok"] * 3
        + out["support_bias_ok"] * 2
        + _same_direction(out["trend_dir"], out["bias_dir"])
        + _same_direction(out["ema_dir"], out["bias_dir"])
        + _same_direction(out["momentum_dir"], out["bias_dir"])
        + _same_direction(out["ai_dir"], out["bias_dir"])
    )

    out["signal_quality_score"] = (
        out["continuation_context_ok"] * 3
        + out["entry_setup_valid"] * 2
        + out["local_confluence_ok"]
        + out["forecast_permission"]
        + out["structure_retest_ok"]
        + _bool_from_threshold(out["pullback_quality"], 1)
        - out["reversal_pressure_level"]
        - out["conflict_score"]
    )

    out["signal_quality"] = pd.Series(
        np.select(
            [
                out["signal_quality_score"] <= 1,
                (out["signal_quality_score"] >= 2) & (out["signal_quality_score"] <= 4),
                (out["signal_quality_score"] >= 5) & (out["signal_quality_score"] <= 7),
                out["signal_quality_score"] >= 8,
            ],
            [
                "low",
                "moderate",
                "high",
                "elite",
            ],
            default="low",
        ),
        index=out.index,
    )

    # =========================================================================
    # FINAL STATE / EXECUTION TYPE
    # =========================================================================

    out["valid_direction"] = (out["bias_dir"] != 0).astype(int)

    out["setup_type"] = pd.Series(
        np.select(
            [
                out["continuation_entry_ok"] == 1,
                out["reversal_entry_hint"] == 1,
                out["continuation_context_ok"] == 1,
                out["reversal_context_ok"] == 1,
            ],
            [
                "continuation",
                "reversal",
                "continuation_building",
                "reversal_building",
            ],
            default="none",
        ),
        index=out.index,
    )

    out["execution_type"] = pd.Series(
        np.select(
            [
                out["continuation_entry_ok"] == 1,
                out["reversal_entry_hint"] == 1,
            ],
            [
                out["continuation_setup_type"],
                "reversal_probe",
            ],
            default="none",
        ),
        index=out.index,
    )

    out["trade_ready"] = (
        (out["continuation_entry_ok"] == 1) |
        (
            (out["reversal_entry_hint"] == 1) &
            (out["reversal_pressure_level"] >= 2)
        )
    ).astype(int)

    out["truth_state"] = pd.Series(
        np.select(
            [
                out["trade_ready"] == 1,
                out["continuation_context_ok"] == 1,
                out["reversal_context_ok"] == 1,
                out["valid_direction"] == 1,
            ],
            [
                "ready",
                "building",
                "reversal_watch",
                "watch",
            ],
            default="blocked",
        ),
        index=out.index,
    )

    out["truth_grade"] = pd.Series(
        np.select(
            [
                out["truth_strength"] <= 4,
                (out["truth_strength"] >= 5) & (out["truth_strength"] <= 8),
                (out["truth_strength"] >= 9) & (out["truth_strength"] <= 13),
                out["truth_strength"] >= 14,
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
    # WEBSITE / SMART SIGNALS FRIENDLY EXPORTS
    # =========================================================================

    out["dir"] = out["bias_dir"]
    out["dir_label"] = out["bias_label"]
    out["forecast_label"] = _label_from_dir(out["forecast_dir"])

    out["trend_label"] = _label_from_dir(out["trend_dir"])
    out["ema_label"] = _label_from_dir(out["ema_dir"])
    out["momentum_label"] = _label_from_dir(out["momentum_dir"])
    out["ai_label"] = _label_from_dir(out["ai_dir"])

    out["continuation_ready"] = out["continuation_entry_ok"]
    out["reversal_ready"] = (
        (out["reversal_entry_hint"] == 1) &
        (out["reversal_pressure_level"] >= 2)
    ).astype(int)

    out["trade_bias"] = out["bias_label"]
    out["direction_label"] = out["bias_label"]
    out["signal_strength"] = out["truth_strength"]
    out["trade_state"] = out["truth_state"]
    out["grade"] = out["truth_grade"]

    return out


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "trend_dir": [1, 1, -1, 1],
            "ema_dir": [1, 1, -1, 1],
            "momentum_dir": [1, 0, -1, 1],
            "ai_dir": [1, 1, -1, 1],
            "trend_strength": [7, 6, 8, 7],
            "ema_quality": [6, 5, 7, 8],
            "ema20_to_ema200_pct": [0.32, 0.62, 0.85, 0.18],
            "market_structure_dir": [1, 1, -1, 1],
            "bos_dir": [1, 1, -1, 0],
            "fib_dir": [1, 1, -1, 1],
            "liquidity_dir": [1, 0, -1, 1],
            "orderflow_dir": [1, 1, -1, 0],
            "session_bias": [1, 1, -1, 0],
            "daily_bias": [1, 1, -1, 1],
            "rt_ema_1420": [1, 0, 0, 1],
            "rt_ema_100200": [0, 1, 1, 0],
            "rt_fib": [1, 1, 0, 0],
            "rt_confluence_cloud": [1, 0, 1, 0],
            "rt_confluence": [1, 0, 0, 1],
            "pullback_active": [1, 1, 1, 0],
            "pullback_dir": [1, 1, -1, 0],
            "pullback_quality": [2, 1, 2, 0],
            "confluence_score": [7, 5, 8, 3],
            "confluence_dir": [1, 1, -1, 1],
            "confluence_active": [1, 1, 1, 1],
            "confluence_cloud_dir": [1, 0, -1, 1],
            "confluence_cloud_active": [1, 0, 1, 1],
            "macd_reversal_dir": [0, 1, 1, -1],
            "macd_reversal_strength": [0, 2, 3, 1],
            "macd_reversal_signal": [0, 1, 1, 1],
            "exhaustion_dir": [0, 1, 1, -1],
            "exhaustion_signal": [0, 0, 1, 1],
            "divergence_flag": [0, 1, 1, 0],
            "divergence_dir": [0, -1, 1, 0],
            "forecast_dir": [1, 1, -1, 1],
            "forecast_confidence": [72, 66, 80, 40],
            "forecast_agreement": [3, 2, 4, 1],
            "regime_label": ["trend", "trend", "trend", "normal"],
            "volume_label": ["normal", "expansion", "normal", "low"],
            "volatility_label": ["normal", "expansion", "normal", "normal"],
            "market_condition_label": ["normal", "expansion", "normal", "normal"],
            "session_active": [1, 1, 1, 0],
        }
    )

    truth = build_truth(sample)
    print(truth.tail())