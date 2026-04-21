from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MacroConfig:
    lookback_bars: int = 50000
    min_samples: int = 100

    atr_window: int = 14
    regime_window: int = 30
    trend_window: int = 50
    pressure_window: int = 20

    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45

    high_risk_threshold: float = 0.70
    low_risk_threshold: float = 0.30

    expansion_threshold: float = 0.65
    compression_threshold: float = 0.35


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Macro input missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Macro input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Macro input dataframe must use a DatetimeIndex")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(_clamp((value - low) / (high - low), 0.0, 1.0))


def _bias_from_score(score: float, bull_th: float, bear_th: float) -> Tuple[int, str]:
    if score >= bull_th:
        return 1, "BULLISH"
    if score <= bear_th:
        return -1, "BEARISH"
    return 0, "NEUTRAL"


def _prepare_df(df: pd.DataFrame, cfg: MacroConfig) -> pd.DataFrame:
    out = df.copy().sort_index()

    if cfg.lookback_bars > 0:
        out = out.tail(cfg.lookback_bars).copy()

    out["ret_pct"] = ((out["close"] / out["open"]) - 1.0) * 100.0
    out["hl_range"] = out["high"] - out["low"]

    prev_close = out["close"].shift(1)
    tr_components = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    out["true_range"] = tr_components.max(axis=1)
    out["atr"] = out["true_range"].rolling(
        cfg.atr_window,
        min_periods=max(5, cfg.atr_window // 2),
    ).mean()

    out["atr_pct"] = (out["atr"] / out["close"].replace(0, np.nan)) * 100.0
    out["atr_pct_mean"] = out["atr_pct"].rolling(
        cfg.regime_window,
        min_periods=max(10, cfg.regime_window // 2),
    ).mean()

    out["range_mean"] = out["hl_range"].rolling(
        cfg.regime_window,
        min_periods=max(10, cfg.regime_window // 2),
    ).mean()

    out["ema_trend"] = out["close"].ewm(span=cfg.trend_window, adjust=False).mean()
    out["trend_spread_pct"] = ((out["close"] / out["ema_trend"]) - 1.0) * 100.0

    out["pressure_mean"] = out["ret_pct"].rolling(
        cfg.pressure_window,
        min_periods=max(5, cfg.pressure_window // 2),
    ).mean()

    out["body_frac"] = (
        (out["close"] - out["open"]).abs() / out["hl_range"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def _classify_risk(risk_score: float, cfg: MacroConfig) -> str:
    if risk_score >= cfg.high_risk_threshold:
        return "HIGH_RISK"
    if risk_score <= cfg.low_risk_threshold:
        return "LOW_RISK"
    return "NORMAL_RISK"


def _classify_regime(expansion_score: float, cfg: MacroConfig) -> str:
    if expansion_score >= cfg.expansion_threshold:
        return "EXPANSION"
    if expansion_score <= cfg.compression_threshold:
        return "COMPRESSION"
    return "TRANSITION"


def _build_scores(work: pd.DataFrame, cfg: MacroConfig) -> Dict[str, float]:
    row = work.iloc[-1]

    atr_pct = _safe_float(row.get("atr_pct"))
    atr_pct_mean = _safe_float(row.get("atr_pct_mean"))
    hl_range = _safe_float(row.get("hl_range"))
    range_mean = _safe_float(row.get("range_mean"))
    trend_spread_pct = _safe_float(row.get("trend_spread_pct"))
    pressure_mean = _safe_float(row.get("pressure_mean"))
    body_frac = _safe_float(row.get("body_frac"))

    atr_ratio = atr_pct / atr_pct_mean if abs(atr_pct_mean) > 1e-12 else 1.0
    range_ratio = hl_range / range_mean if abs(range_mean) > 1e-12 else 1.0

    volatility_pressure = _normalize(atr_ratio, 0.7, 1.8)
    range_expansion_score = _normalize(range_ratio, 0.7, 1.8)
    directional_pressure = _normalize(abs(pressure_mean), 0.0, 0.20)
    trend_pressure = _normalize(abs(trend_spread_pct), 0.0, 0.50)
    conviction_score = _normalize(body_frac, 0.30, 0.90)

    expansion_score = (
        0.40 * volatility_pressure
        + 0.35 * range_expansion_score
        + 0.15 * directional_pressure
        + 0.10 * conviction_score
    )

    risk_score = (
        0.45 * volatility_pressure
        + 0.30 * range_expansion_score
        + 0.15 * directional_pressure
        + 0.10 * trend_pressure
    )

    directional_score = 0.5
    if trend_spread_pct > 0 and pressure_mean > 0:
        directional_score = 0.70 + 0.30 * min((trend_pressure + directional_pressure) / 2.0, 1.0)
    elif trend_spread_pct < 0 and pressure_mean < 0:
        directional_score = 0.30 - 0.30 * min((trend_pressure + directional_pressure) / 2.0, 1.0)

    directional_score = _clamp(directional_score, 0.0, 1.0)

    return {
        "atr_pct": round(atr_pct, 4),
        "atr_pct_mean": round(atr_pct_mean, 4),
        "atr_ratio": round(atr_ratio, 4),
        "range_ratio": round(range_ratio, 4),
        "trend_spread_pct": round(trend_spread_pct, 4),
        "pressure_mean": round(pressure_mean, 4),
        "body_frac": round(body_frac, 4),

        "volatility_pressure": round(volatility_pressure, 4),
        "range_expansion_score": round(range_expansion_score, 4),
        "directional_pressure": round(directional_pressure, 4),
        "trend_pressure": round(trend_pressure, 4),
        "conviction_score": round(conviction_score, 4),

        "expansion_score": round(_clamp(expansion_score, 0.0, 1.0), 4),
        "risk_score": round(_clamp(risk_score, 0.0, 1.0), 4),
        "directional_score": round(directional_score, 4),
    }


# =============================================================================
# ENGINE
# =============================================================================

def build_macro_payload(
    df: pd.DataFrame,
    config: Optional[MacroConfig] = None,
) -> Dict[str, Any]:
    cfg = config or MacroConfig()

    _validate_ohlc(df)
    work = _prepare_df(df, cfg)

    if len(work) < cfg.min_samples:
        raise ValueError(f"Macro requires at least {cfg.min_samples} rows, got {len(work)}")

    scores = _build_scores(work, cfg)

    bias_signal, bias_label = _bias_from_score(
        scores["directional_score"],
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    regime_state = _classify_regime(scores["expansion_score"], cfg)
    risk_state = _classify_risk(scores["risk_score"], cfg)

    state = f"{regime_state}_{risk_state}"
    indicator_strength = _clamp(
        0.45 * scores["expansion_score"]
        + 0.35 * scores["risk_score"]
        + 0.20 * abs(scores["directional_score"] - 0.5) * 2.0,
        0.0,
        1.0,
    )

    latest_ts = work.index[-1]

    payload: Dict[str, Any] = {
        "indicator": "FORECASTER_MACRO",
        "timestamp": latest_ts.isoformat(),
        "state": state,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(indicator_strength, 4),

        "macro_state": state,
        "regime_state": regime_state,
        "risk_state": risk_state,

        "volatility_pressure": scores["volatility_pressure"],
        "range_expansion_score": scores["range_expansion_score"],
        "directional_pressure": scores["directional_pressure"],
        "trend_pressure": scores["trend_pressure"],
        "conviction_score": scores["conviction_score"],

        "expansion_score": scores["expansion_score"],
        "risk_score": scores["risk_score"],
        "directional_score": scores["directional_score"],

        "atr_pct": scores["atr_pct"],
        "atr_pct_mean": scores["atr_pct_mean"],
        "atr_ratio": scores["atr_ratio"],
        "range_ratio": scores["range_ratio"],
        "trend_spread_pct": scores["trend_spread_pct"],
        "pressure_mean": scores["pressure_mean"],
        "body_frac": scores["body_frac"],

        "data_mode": "PRICE_PROXY_MACRO_V1",
        "future_mode": "EVENT_DRIVEN_MACRO_V2",

        "config": asdict(cfg),
    }

    return payload


def build_macro_latest_payload(
    df: pd.DataFrame,
    config: Optional[MacroConfig] = None,
) -> Dict[str, Any]:
    """
    Alias kept to match SmartChart payload-builder naming.
    """
    return build_macro_payload(df=df, config=config)