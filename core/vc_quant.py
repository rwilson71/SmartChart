from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class QuantConfig:
    lookback_bars: int = 50000
    min_samples: int = 50

    # Return handling
    neutral_epsilon_pct: float = 0.00001
    zscore_clip: float = 3.0

    # Bias interpretation
    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45

    # Strength weighting
    weight_win_rate: float = 0.35
    weight_expectancy: float = 0.35
    weight_payoff: float = 0.15
    weight_zscore: float = 0.15


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Quant input missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Quant input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Quant input dataframe must use a DatetimeIndex")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _prepare_df(df: pd.DataFrame, cfg: QuantConfig) -> pd.DataFrame:
    out = df.copy().sort_index()

    if cfg.lookback_bars > 0:
        out = out.tail(cfg.lookback_bars).copy()

    out["ret_pct"] = ((out["close"] / out["open"]) - 1.0) * 100.0

    eps = abs(cfg.neutral_epsilon_pct)
    out["is_bull"] = (out["ret_pct"] > eps).astype(int)
    out["is_bear"] = (out["ret_pct"] < -eps).astype(int)
    out["is_neutral"] = ((out["ret_pct"] >= -eps) & (out["ret_pct"] <= eps)).astype(int)

    return out


def _bias_from_score(score: float, bull_th: float, bear_th: float) -> Tuple[int, str]:
    if score >= bull_th:
        return 1, "BULLISH"
    if score <= bear_th:
        return -1, "BEARISH"
    return 0, "NEUTRAL"


def _probability_state(win_rate: float, expectancy: float) -> str:
    if win_rate >= 0.60 and expectancy > 0:
        return "HIGH_PROBABILITY"
    if win_rate >= 0.52 and expectancy > 0:
        return "POSITIVE_EDGE"
    if expectancy < 0 and win_rate <= 0.48:
        return "NEGATIVE_EDGE"
    return "NEUTRAL_EDGE"


def _compute_zscore(series: pd.Series, clip: float) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0

    mean = _safe_float(clean.mean())
    std = _safe_float(clean.std(ddof=0))
    last = _safe_float(clean.iloc[-1])

    if std <= 1e-12:
        return 0.0

    z = (last - mean) / std
    return float(_clamp(z, -abs(clip), abs(clip)))


def _normalize_component(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    scaled = (value - low) / (high - low)
    return float(_clamp(scaled, 0.0, 1.0))


def _build_strength(
    win_rate: float,
    expectancy_pct: float,
    payoff_ratio: float,
    zscore_return: float,
    cfg: QuantConfig,
) -> float:
    win_component = abs(win_rate - 0.5) * 2.0

    # Expectancy: map roughly -0.20%..+0.20% to 0..1 on magnitude
    expectancy_component = _normalize_component(abs(expectancy_pct), 0.0, 0.20)

    # Payoff: map 1.0..3.0 into 0..1, cap outside
    payoff_component = _normalize_component(payoff_ratio, 1.0, 3.0)

    # Z-score: map |z| 0..clip into 0..1
    z_component = _normalize_component(abs(zscore_return), 0.0, cfg.zscore_clip)

    strength = (
        cfg.weight_win_rate * win_component
        + cfg.weight_expectancy * expectancy_component
        + cfg.weight_payoff * payoff_component
        + cfg.weight_zscore * z_component
    )

    return float(_clamp(strength, 0.0, 1.0))


# =============================================================================
# ENGINE
# =============================================================================

def build_quant_payload(
    df: pd.DataFrame,
    config: Optional[QuantConfig] = None,
) -> Dict[str, Any]:
    cfg = config or QuantConfig()

    _validate_ohlc(df)
    work = _prepare_df(df, cfg)

    if len(work) < cfg.min_samples:
        raise ValueError(f"Quant requires at least {cfg.min_samples} rows, got {len(work)}")

    rets = work["ret_pct"].dropna()
    n = int(len(rets))

    bull_mask = rets > abs(cfg.neutral_epsilon_pct)
    bear_mask = rets < -abs(cfg.neutral_epsilon_pct)
    neutral_mask = (~bull_mask) & (~bear_mask)

    wins = rets[bull_mask]
    losses = rets[bear_mask]
    neutrals = rets[neutral_mask]

    bullish_rate = len(wins) / n if n > 0 else 0.0
    bearish_rate = len(losses) / n if n > 0 else 0.0
    neutral_rate = len(neutrals) / n if n > 0 else 0.0

    avg_return_pct = _safe_float(rets.mean())
    median_return_pct = _safe_float(rets.median())
    std_return_pct = _safe_float(rets.std(ddof=0))

    avg_win_pct = _safe_float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss_pct = _safe_float(losses.mean()) if len(losses) > 0 else 0.0

    loss_mag = abs(avg_loss_pct)
    payoff_ratio = (avg_win_pct / loss_mag) if loss_mag > 1e-12 else 0.0

    expectancy_pct = (bullish_rate * avg_win_pct) + (bearish_rate * avg_loss_pct)

    zscore_return = _compute_zscore(rets, clip=cfg.zscore_clip)

    probability_state = _probability_state(
        win_rate=bullish_rate,
        expectancy=expectancy_pct,
    )

    bias_signal, bias_label = _bias_from_score(
        bullish_rate,
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    indicator_strength = _build_strength(
        win_rate=bullish_rate,
        expectancy_pct=expectancy_pct,
        payoff_ratio=payoff_ratio,
        zscore_return=zscore_return,
        cfg=cfg,
    )

    latest_ts = work.index[-1]

    payload: Dict[str, Any] = {
        "indicator": "FORECASTER_QUANT",
        "timestamp": latest_ts.isoformat(),
        "state": probability_state,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(indicator_strength, 4),

        "sample_size": n,
        "bullish_rate": round(bullish_rate, 4),
        "bearish_rate": round(bearish_rate, 4),
        "neutral_rate": round(neutral_rate, 4),

        "avg_return_pct": round(avg_return_pct, 4),
        "median_return_pct": round(median_return_pct, 4),
        "std_return_pct": round(std_return_pct, 4),

        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "payoff_ratio": round(payoff_ratio, 4),
        "expectancy_pct": round(expectancy_pct, 4),
        "z_score_return": round(zscore_return, 4),

        "probability_state": probability_state,

        "config": asdict(cfg),
    }

    return payload


def build_quant_latest_payload(
    df: pd.DataFrame,
    config: Optional[QuantConfig] = None,
) -> Dict[str, Any]:
    """
    Alias kept to match SmartChart payload-builder naming.
    """
    return build_quant_payload(df=df, config=config)