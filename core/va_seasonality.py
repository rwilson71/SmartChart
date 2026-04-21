from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SeasonalityConfig:
    lookback_bars: int = 50000
    min_group_samples: int = 5
    strength_clip: float = 1.0
    bullish_threshold: float = 0.55
    bearish_threshold: float = 0.45


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Seasonality input missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Seasonality input dataframe is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Seasonality input dataframe must use a DatetimeIndex")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_strength(win_rate: float, avg_return: float, clip: float = 1.0) -> float:
    """
    Convert directional edge into a 0..1 strength score.
    Combines:
      - win rate edge around 50%
      - normalized average return
    """
    wr_edge = abs(win_rate - 0.5) * 2.0
    ret_edge = min(abs(avg_return) / max(clip, 1e-9), 1.0)
    strength = (0.65 * wr_edge) + (0.35 * ret_edge)
    return float(_clamp(strength, 0.0, 1.0))


def _bias_from_score(score: float, bull_th: float, bear_th: float) -> Tuple[int, str]:
    """
    score expected in 0..1 where:
      > bull_th = bullish
      < bear_th = bearish
      otherwise neutral
    """
    if score >= bull_th:
        return 1, "BULLISH"
    if score <= bear_th:
        return -1, "BEARISH"
    return 0, "NEUTRAL"


def _prepare_df(df: pd.DataFrame, cfg: SeasonalityConfig) -> pd.DataFrame:
    out = df.copy().sort_index()

    if cfg.lookback_bars > 0:
        out = out.tail(cfg.lookback_bars).copy()

    out["ret_pct"] = ((out["close"] / out["open"]) - 1.0) * 100.0
    out["is_bull"] = (out["ret_pct"] > 0).astype(int)
    out["is_bear"] = (out["ret_pct"] < 0).astype(int)
    out["year"] = out.index.year
    out["month_num"] = out.index.month
    out["month_name"] = out.index.month_name()
    out["weekday_num"] = out.index.weekday
    out["weekday_name"] = out.index.day_name()
    out["hour"] = out.index.hour

    return out


# =============================================================================
# GROUP ANALYSIS
# =============================================================================

def _summarize_group(group: pd.DataFrame, label: str, cfg: SeasonalityConfig) -> Dict[str, Any]:
    n = int(len(group))
    bull_count = int(group["is_bull"].sum())
    bear_count = int(group["is_bear"].sum())
    neutral_count = int(n - bull_count - bear_count)

    bull_rate = bull_count / n if n > 0 else 0.0
    bear_rate = bear_count / n if n > 0 else 0.0
    neutral_rate = neutral_count / n if n > 0 else 0.0

    avg_return = _safe_float(group["ret_pct"].mean())
    median_return = _safe_float(group["ret_pct"].median())
    std_return = _safe_float(group["ret_pct"].std(ddof=0))
    cum_return = _safe_float(group["ret_pct"].sum())

    strength = _normalize_strength(
        win_rate=bull_rate,
        avg_return=avg_return,
        clip=cfg.strength_clip,
    )

    score01 = bull_rate
    bias_signal, bias_label = _bias_from_score(
        score=score01,
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    return {
        "label": label,
        "samples": n,
        "bullish_count": bull_count,
        "bearish_count": bear_count,
        "neutral_count": neutral_count,
        "bullish_rate": round(bull_rate, 4),
        "bearish_rate": round(bear_rate, 4),
        "neutral_rate": round(neutral_rate, 4),
        "avg_return_pct": round(avg_return, 4),
        "median_return_pct": round(median_return, 4),
        "std_return_pct": round(std_return, 4),
        "cumulative_return_pct": round(cum_return, 4),
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "strength": round(strength, 4),
    }


def _group_table(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    ordered_labels: Optional[list[str]],
    cfg: SeasonalityConfig,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    grouped = df.groupby(group_col, sort=True)

    for key, group in grouped:
        label = str(group[label_col].iloc[0])
        if len(group) < cfg.min_group_samples:
            continue
        result[label] = _summarize_group(group, label=label, cfg=cfg)

    if ordered_labels is not None:
        ordered_result: Dict[str, Dict[str, Any]] = {}
        for label in ordered_labels:
            if label in result:
                ordered_result[label] = result[label]
        return ordered_result

    return result


def _best_label(stats: Dict[str, Dict[str, Any]], mode: str = "bullish_rate") -> Optional[str]:
    if not stats:
        return None

    valid_items = [(k, v) for k, v in stats.items() if isinstance(v, dict)]
    if not valid_items:
        return None

    if mode == "bullish_rate":
        return max(valid_items, key=lambda kv: (kv[1].get("bullish_rate", 0.0), kv[1].get("avg_return_pct", 0.0)))[0]

    if mode == "avg_return_pct":
        return max(valid_items, key=lambda kv: kv[1].get("avg_return_pct", 0.0))[0]

    return None


def _worst_label(stats: Dict[str, Dict[str, Any]], mode: str = "avg_return_pct") -> Optional[str]:
    if not stats:
        return None

    valid_items = [(k, v) for k, v in stats.items() if isinstance(v, dict)]
    if not valid_items:
        return None

    if mode == "avg_return_pct":
        return min(valid_items, key=lambda kv: kv[1].get("avg_return_pct", 0.0))[0]

    if mode == "bullish_rate":
        return min(valid_items, key=lambda kv: kv[1].get("bullish_rate", 0.0))[0]

    return None


# =============================================================================
# ENGINE
# =============================================================================

def build_seasonality_payload(
    df: pd.DataFrame,
    config: Optional[SeasonalityConfig] = None,
) -> Dict[str, Any]:
    cfg = config or SeasonalityConfig()

    _validate_ohlc(df)
    work = _prepare_df(df, cfg)

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]

    month_stats = _group_table(
        df=work,
        group_col="month_num",
        label_col="month_name",
        ordered_labels=month_order,
        cfg=cfg,
    )

    day_stats = _group_table(
        df=work,
        group_col="weekday_num",
        label_col="weekday_name",
        ordered_labels=weekday_order,
        cfg=cfg,
    )

    hour_stats = _group_table(
        df=work,
        group_col="hour",
        label_col="hour",
        ordered_labels=None,
        cfg=cfg,
    )

    overall_return = _safe_float(work["ret_pct"].mean())
    overall_bull_rate = _safe_float(work["is_bull"].mean())
    overall_strength = _normalize_strength(
        win_rate=overall_bull_rate,
        avg_return=overall_return,
        clip=cfg.strength_clip,
    )

    bias_signal, bias_label = _bias_from_score(
        score=overall_bull_rate,
        bull_th=cfg.bullish_threshold,
        bear_th=cfg.bearish_threshold,
    )

    best_month = _best_label(month_stats, mode="avg_return_pct")
    worst_month = _worst_label(month_stats, mode="avg_return_pct")
    best_day = _best_label(day_stats, mode="avg_return_pct")
    worst_day = _worst_label(day_stats, mode="avg_return_pct")

    latest_ts = work.index[-1]

    payload: Dict[str, Any] = {
        "indicator": "SEASONALITY",
        "timestamp": latest_ts.isoformat(),
        "state": bias_label,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(overall_strength, 4),

        "sample_size": int(len(work)),
        "avg_return_pct": round(overall_return, 4),
        "bullish_rate": round(overall_bull_rate, 4),

        "best_month": best_month,
        "worst_month": worst_month,
        "best_day": best_day,
        "worst_day": worst_day,

        "month_stats": month_stats,
        "day_stats": day_stats,
        "hour_stats": hour_stats,

        "config": asdict(cfg),
    }

    return payload


def build_seasonality_latest_payload(
    df: pd.DataFrame,
    config: Optional[SeasonalityConfig] = None,
) -> Dict[str, Any]:
    """
    Alias kept to match your existing SmartChart naming pattern.
    """
    return build_seasonality_payload(df=df, config=config)