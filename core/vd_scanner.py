from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ScannerConfig:
    lookback_bars: int = 50000
    min_samples: int = 100

    # Signal windows
    momentum_window: int = 20
    trend_window: int = 50
    vol_window: int = 20

    # Signal thresholds
    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45
    strong_score_threshold: float = 0.70
    moderate_score_threshold: float = 0.45

    # Ranking weights
    weight_trend: float = 0.35
    weight_momentum: float = 0.30
    weight_volatility: float = 0.15
    weight_participation: float = 0.20


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scanner input missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Scanner input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Scanner input dataframe must use a DatetimeIndex")


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


def _scanner_state(score: float, bias_label: str, cfg: ScannerConfig) -> str:
    if score >= cfg.strong_score_threshold:
        return f"{bias_label}_HIGH_PRIORITY"
    if score >= cfg.moderate_score_threshold:
        return f"{bias_label}_WATCHLIST"
    return "NEUTRAL_MONITOR"


def _prepare_df(df: pd.DataFrame, cfg: ScannerConfig) -> pd.DataFrame:
    out = df.copy().sort_index()

    if cfg.lookback_bars > 0:
        out = out.tail(cfg.lookback_bars).copy()

    out["ret_pct"] = ((out["close"] / out["open"]) - 1.0) * 100.0
    out["hl_range"] = (out["high"] - out["low"]).replace(0, np.nan)

    out["ema_fast"] = out["close"].ewm(span=cfg.momentum_window, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=cfg.trend_window, adjust=False).mean()

    out["trend_spread_pct"] = ((out["ema_fast"] / out["ema_slow"]) - 1.0) * 100.0
    out["momentum_mean"] = out["ret_pct"].rolling(cfg.momentum_window, min_periods=max(5, cfg.momentum_window // 2)).mean()
    out["volatility_pct"] = out["ret_pct"].rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 2)).std(ddof=0)

    if "volume" in out.columns:
        out["volume_ma"] = out["volume"].rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 2)).mean()
        out["participation_ratio"] = out["volume"] / out["volume_ma"].replace(0, np.nan)
    else:
        # fallback participation proxy from candle range
        out["range_ma"] = out["hl_range"].rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 2)).mean()
        out["participation_ratio"] = out["hl_range"] / out["range_ma"].replace(0, np.nan)

    out["body_frac"] = ((out["close"] - out["open"]).abs() / out["hl_range"]).replace([np.inf, -np.inf], np.nan)
    out["body_frac"] = out["body_frac"].fillna(0.0)

    return out


def _latest_metrics(work: pd.DataFrame) -> Dict[str, float]:
    row = work.iloc[-1]

    return {
        "ret_pct": _safe_float(row.get("ret_pct")),
        "trend_spread_pct": _safe_float(row.get("trend_spread_pct")),
        "momentum_mean": _safe_float(row.get("momentum_mean")),
        "volatility_pct": _safe_float(row.get("volatility_pct")),
        "participation_ratio": _safe_float(row.get("participation_ratio"), 1.0),
        "body_frac": _safe_float(row.get("body_frac")),
    }


def _build_scores(metrics: Dict[str, float], cfg: ScannerConfig) -> Dict[str, float]:
    trend_spread = metrics["trend_spread_pct"]
    momentum_mean = metrics["momentum_mean"]
    volatility_pct = abs(metrics["volatility_pct"])
    participation_ratio = metrics["participation_ratio"]

    trend_score = _normalize(abs(trend_spread), 0.0, 0.50)
    momentum_score = _normalize(abs(momentum_mean), 0.0, 0.20)
    volatility_score = _normalize(volatility_pct, 0.0, 0.30)
    participation_score = _normalize(participation_ratio, 0.8, 1.8)

    composite = (
        cfg.weight_trend * trend_score
        + cfg.weight_momentum * momentum_score
        + cfg.weight_volatility * volatility_score
        + cfg.weight_participation * participation_score
    )

    directional_score = 0.5
    if trend_spread > 0 and momentum_mean > 0:
        directional_score = 0.70 + 0.30 * min((trend_score + momentum_score) / 2.0, 1.0)
    elif trend_spread < 0 and momentum_mean < 0:
        directional_score = 0.30 - 0.30 * min((trend_score + momentum_score) / 2.0, 1.0)

    directional_score = _clamp(directional_score, 0.0, 1.0)

    return {
        "trend_score": round(trend_score, 4),
        "momentum_score": round(momentum_score, 4),
        "volatility_score": round(volatility_score, 4),
        "participation_score": round(participation_score, 4),
        "composite_score": round(composite, 4),
        "directional_score": round(directional_score, 4),
    }


def _ranking_bucket(composite_score: float, bias_label: str, cfg: ScannerConfig) -> str:
    if composite_score >= cfg.strong_score_threshold:
        return f"{bias_label}_A"
    if composite_score >= cfg.moderate_score_threshold:
        return f"{bias_label}_B"
    return "NEUTRAL_C"


# =============================================================================
# ENGINE
# =============================================================================

def build_scanner_payload(
    df: pd.DataFrame,
    config: Optional[ScannerConfig] = None,
) -> Dict[str, Any]:
    cfg = config or ScannerConfig()

    _validate_ohlcv(df)
    work = _prepare_df(df, cfg)

    if len(work) < cfg.min_samples:
        raise ValueError(f"Scanner requires at least {cfg.min_samples} rows, got {len(work)}")

    metrics = _latest_metrics(work)
    scores = _build_scores(metrics, cfg)

    bias_signal, bias_label = _bias_from_score(
        scores["directional_score"],
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    composite_score = _safe_float(scores["composite_score"])
    scanner_state = _scanner_state(composite_score, bias_label, cfg)
    ranking_bucket = _ranking_bucket(composite_score, bias_label, cfg)

    latest_ts = work.index[-1]

    payload: Dict[str, Any] = {
        "indicator": "FORECASTER_SCANNER",
        "timestamp": latest_ts.isoformat(),
        "state": scanner_state,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(composite_score, 4),

        "scanner_state": scanner_state,
        "ranking_bucket": ranking_bucket,
        "opportunity_score": round(composite_score, 4),

        "trend_score": scores["trend_score"],
        "momentum_score": scores["momentum_score"],
        "volatility_score": scores["volatility_score"],
        "participation_score": scores["participation_score"],
        "directional_score": scores["directional_score"],

        "ret_pct": round(metrics["ret_pct"], 4),
        "trend_spread_pct": round(metrics["trend_spread_pct"], 4),
        "momentum_mean": round(metrics["momentum_mean"], 4),
        "volatility_pct": round(metrics["volatility_pct"], 4),
        "participation_ratio": round(metrics["participation_ratio"], 4),
        "body_frac": round(metrics["body_frac"], 4),

        "data_mode": "SINGLE_ASSET_SCANNER_V1",
        "future_mode": "MULTI_ASSET_RANKED_SCANNER",

        "config": asdict(cfg),
    }

    return payload


def build_scanner_latest_payload(
    df: pd.DataFrame,
    config: Optional[ScannerConfig] = None,
) -> Dict[str, Any]:
    """
    Alias kept to match SmartChart payload-builder naming.
    """
    return build_scanner_payload(df=df, config=config)