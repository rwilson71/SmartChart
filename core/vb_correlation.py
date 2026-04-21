from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class CorrelationConfig:
    lookback_bars: int = 50000
    rolling_window: int = 100
    min_samples: int = 50

    # Horizon placeholders for future multi-year / multi-regime expansion
    use_3y: bool = True
    use_5y: bool = True
    use_7y: bool = True
    use_10y: bool = True
    use_15y: bool = True

    # Strength interpretation
    strong_corr_threshold: float = 0.70
    moderate_corr_threshold: float = 0.40

    # Bias interpretation from return alignment
    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Correlation input missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Correlation input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Correlation input dataframe must use a DatetimeIndex")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _prepare_df(df: pd.DataFrame, cfg: CorrelationConfig) -> pd.DataFrame:
    out = df.copy().sort_index()

    if cfg.lookback_bars > 0:
        out = out.tail(cfg.lookback_bars).copy()

    out["ret_pct"] = ((out["close"] / out["open"]) - 1.0) * 100.0
    out["ret_log"] = np.log(out["close"] / out["close"].shift(1))
    out["direction"] = np.sign(out["ret_pct"]).fillna(0)

    return out


def _normalize_corr_strength(corr: float) -> float:
    return float(_clamp(abs(corr), 0.0, 1.0))


def _corr_label(corr: float, cfg: CorrelationConfig) -> str:
    a = abs(corr)
    if a >= cfg.strong_corr_threshold:
        return "STRONG"
    if a >= cfg.moderate_corr_threshold:
        return "MODERATE"
    return "WEAK"


def _relationship_label(corr: float, cfg: CorrelationConfig) -> str:
    strength = _corr_label(corr, cfg)
    if corr > 0:
        return f"POSITIVE_{strength}"
    if corr < 0:
        return f"NEGATIVE_{strength}"
    return "NEUTRAL_WEAK"


def _bias_from_alignment(score: float, bull_th: float, bear_th: float) -> Tuple[int, str]:
    if score >= bull_th:
        return 1, "BULLISH"
    if score <= bear_th:
        return -1, "BEARISH"
    return 0, "NEUTRAL"


def _compute_anchor_series(work: pd.DataFrame) -> pd.Series:
    """
    Internal anchor series for v1.

    Since we do not yet have a second external asset series inside this engine,
    we compare short-horizon returns to slower anchor returns of the same market.
    This gives a usable internal correlation structure now, while preserving
    the architecture for true cross-asset series later.
    """
    fast = work["ret_log"].rolling(5, min_periods=3).mean()
    slow = work["ret_log"].rolling(20, min_periods=10).mean()
    anchor = slow.fillna(0.0)
    return fast.corr(anchor)  # not used directly, keeps intent obvious


def _rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    return series_a.rolling(window, min_periods=max(10, window // 2)).corr(series_b)


def _summarize_pair(
    name: str,
    series_a: pd.Series,
    series_b: pd.Series,
    cfg: CorrelationConfig,
) -> Dict[str, Any]:
    joined = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()

    if len(joined) < cfg.min_samples:
        return {
            "label": name,
            "samples": int(len(joined)),
            "correlation": 0.0,
            "rolling_correlation": 0.0,
            "relationship_state": "INSUFFICIENT_DATA",
            "alignment_rate": 0.0,
            "bias_signal": 0,
            "bias_label": "NEUTRAL",
            "strength": 0.0,
        }

    corr = _safe_float(joined["a"].corr(joined["b"]))
    rolling = _rolling_corr(joined["a"], joined["b"], cfg.rolling_window)
    rolling_last = _safe_float(rolling.iloc[-1]) if not rolling.empty else 0.0

    align = (np.sign(joined["a"]) == np.sign(joined["b"])).astype(int)
    alignment_rate = _safe_float(align.mean())

    strength = _normalize_corr_strength(rolling_last if rolling_last != 0 else corr)
    bias_signal, bias_label = _bias_from_alignment(
        alignment_rate,
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    return {
        "label": name,
        "samples": int(len(joined)),
        "correlation": round(corr, 4),
        "rolling_correlation": round(rolling_last, 4),
        "relationship_state": _relationship_label(rolling_last if rolling_last != 0 else corr, cfg),
        "alignment_rate": round(alignment_rate, 4),
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "strength": round(strength, 4),
    }


def _build_internal_horizon_map(work: pd.DataFrame, cfg: CorrelationConfig) -> Dict[str, Dict[str, Any]]:
    """
    v1 internal-horizon correlation map.

    We compare:
    - fast return structure
    - medium return structure
    - slow return structure

    Then expose them using future-ready horizon names so later we can swap in
    true multi-year or cross-asset datasets without changing the payload contract.
    """
    ret_1 = work["ret_log"]
    ret_5 = work["ret_log"].rolling(5, min_periods=3).mean()
    ret_20 = work["ret_log"].rolling(20, min_periods=10).mean()
    ret_50 = work["ret_log"].rolling(50, min_periods=20).mean()
    ret_100 = work["ret_log"].rolling(100, min_periods=30).mean()

    horizon_stats: Dict[str, Dict[str, Any]] = {}

    if cfg.use_3y:
        horizon_stats["3Y"] = _summarize_pair("3Y", ret_1, ret_20, cfg)
    if cfg.use_5y:
        horizon_stats["5Y"] = _summarize_pair("5Y", ret_5, ret_20, cfg)
    if cfg.use_7y:
        horizon_stats["7Y"] = _summarize_pair("7Y", ret_5, ret_50, cfg)
    if cfg.use_10y:
        horizon_stats["10Y"] = _summarize_pair("10Y", ret_20, ret_50, cfg)
    if cfg.use_15y:
        horizon_stats["15Y"] = _summarize_pair("15Y", ret_20, ret_100, cfg)

    return horizon_stats


def _aggregate_horizon_stats(horizon_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not horizon_stats:
        return {
            "active_horizons": [],
            "avg_correlation": 0.0,
            "avg_rolling_correlation": 0.0,
            "avg_alignment_rate": 0.0,
            "avg_strength": 0.0,
            "dominant_relationship": "NEUTRAL_WEAK",
        }

    rows = list(horizon_stats.values())

    avg_corr = float(np.mean([_safe_float(r.get("correlation")) for r in rows]))
    avg_roll = float(np.mean([_safe_float(r.get("rolling_correlation")) for r in rows]))
    avg_align = float(np.mean([_safe_float(r.get("alignment_rate")) for r in rows]))
    avg_strength = float(np.mean([_safe_float(r.get("strength")) for r in rows]))

    rel_counts: Dict[str, int] = {}
    for r in rows:
        label = str(r.get("relationship_state", "NEUTRAL_WEAK"))
        rel_counts[label] = rel_counts.get(label, 0) + 1

    dominant_relationship = max(rel_counts.items(), key=lambda kv: kv[1])[0] if rel_counts else "NEUTRAL_WEAK"

    return {
        "active_horizons": list(horizon_stats.keys()),
        "avg_correlation": round(avg_corr, 4),
        "avg_rolling_correlation": round(avg_roll, 4),
        "avg_alignment_rate": round(avg_align, 4),
        "avg_strength": round(avg_strength, 4),
        "dominant_relationship": dominant_relationship,
    }


# =============================================================================
# ENGINE
# =============================================================================

def build_correlation_payload(
    df: pd.DataFrame,
    config: Optional[CorrelationConfig] = None,
) -> Dict[str, Any]:
    cfg = config or CorrelationConfig()

    _validate_ohlc(df)
    work = _prepare_df(df, cfg)

    if len(work) < cfg.min_samples:
        raise ValueError(
            f"Correlation requires at least {cfg.min_samples} rows, got {len(work)}"
        )

    horizon_stats = _build_internal_horizon_map(work, cfg)
    summary = _aggregate_horizon_stats(horizon_stats)

    avg_alignment = _safe_float(summary.get("avg_alignment_rate", 0.0))
    avg_strength = _safe_float(summary.get("avg_strength", 0.0))
    dominant_relationship = str(summary.get("dominant_relationship", "NEUTRAL_WEAK"))

    bias_signal, bias_label = _bias_from_alignment(
        avg_alignment,
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    latest_ts = work.index[-1]

    payload: Dict[str, Any] = {
        "indicator": "FORECASTER_CORRELATION",
        "timestamp": latest_ts.isoformat(),
        "state": dominant_relationship,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(avg_strength, 4),

        "relationship_state": dominant_relationship,
        "rolling_correlation_score": round(_safe_float(summary.get("avg_rolling_correlation", 0.0)), 4),
        "alignment_score": round(avg_alignment, 4),

        "active_horizons": summary.get("active_horizons", []),
        "horizon_stats": horizon_stats,

        "data_mode": "INTERNAL_SINGLE_ASSET_V1",
        "future_mode": "MULTI_ASSET_CROSS_CORRELATION",

        "config": asdict(cfg),
    }

    return payload


def build_correlation_latest_payload(
    df: pd.DataFrame,
    config: Optional[CorrelationConfig] = None,
) -> Dict[str, Any]:
    """
    Alias kept to match the SmartChart payload-builder pattern.
    """
    return build_correlation_payload(df=df, config=config)