from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.cc_ema_distance_calibration import (
    EmaDistanceConfig,
    build_feature_frame,
)
# =============================================================================
# SMARTCHART • EMA DISTANCE CALIBRATION RESEARCH
# File: research/cc_ema_distance_reserch.py
# =============================================================================
# Purpose
# - Research-only calibration layer
# - Separate from live website payload flow
# - Uses first-true-bar signal triggering to reduce clustered overcounting
# =============================================================================


@dataclass
class EmaDistanceCalibrationConfig(EmaDistanceConfig):
    continuation_points: float = 60.0
    reversal_points: float = 30.0
    max_forward_bars: int = 60
    point_value: float = 1.0

    min_bucket_samples: int = 200
    zone1_min_continuation_rate: float = 0.58
    zone2_min_continuation_rate: float = 0.52
    max_zone1_reversal_rate: float = 0.34
    max_zone2_reversal_rate: float = 0.42


# =============================================================================
# HELPERS
# =============================================================================

def contiguous_ranges(sorted_indices: Iterable[int]) -> List[Tuple[int, int]]:
    idx = list(sorted(set(int(i) for i in sorted_indices)))
    if not idx:
        return []

    ranges: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((start, prev))
        start = i
        prev = i

    ranges.append((start, prev))
    return ranges


def _first_true_bar(signal: pd.Series) -> pd.Series:
    signal = signal.fillna(False).astype(bool)
    prev = signal.shift(1).fillna(False)
    return signal & (~prev)

def _forward_path_outcome(
    side: int,
    entry_price: float,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    cont_move_price: float,
    rev_move_price: float,
) -> Tuple[str, Optional[int], float, float]:
    if len(future_highs) == 0 or len(future_lows) == 0:
        return "unresolved", None, np.nan, np.nan

    if side == 1:
        favorable = future_highs - entry_price
        adverse = entry_price - future_lows
        cont_hit = favorable >= cont_move_price
        rev_hit = adverse >= rev_move_price
    else:
        favorable = entry_price - future_lows
        adverse = future_highs - entry_price
        cont_hit = favorable >= cont_move_price
        rev_hit = adverse >= rev_move_price

    mfe_price = float(np.nanmax(favorable)) if len(favorable) else np.nan
    mae_price = float(np.nanmax(adverse)) if len(adverse) else np.nan

    first_cont_idx = int(np.argmax(cont_hit)) if cont_hit.any() else None
    first_rev_idx = int(np.argmax(rev_hit)) if rev_hit.any() else None

    if first_cont_idx is None and first_rev_idx is None:
        return "unresolved", None, mfe_price, mae_price
    if first_cont_idx is not None and first_rev_idx is None:
        return "continuation", first_cont_idx + 1, mfe_price, mae_price
    if first_rev_idx is not None and first_cont_idx is None:
        return "reversal", first_rev_idx + 1, mfe_price, mae_price
    if first_cont_idx <= first_rev_idx:
        return "continuation", first_cont_idx + 1, mfe_price, mae_price
    return "reversal", first_rev_idx + 1, mfe_price, mae_price

# =============================================================================
# EVENT ENGINE
# =============================================================================

def run_forward_outcome_test(features: pd.DataFrame, cfg: EmaDistanceCalibrationConfig) -> pd.DataFrame:
    trigger_mask = _first_true_bar(features["research_signal"])
    signal_idx = np.flatnonzero(trigger_mask.to_numpy())

    close = features[cfg.close_col].to_numpy(dtype=float)
    high = features[cfg.high_col].to_numpy(dtype=float)
    low = features[cfg.low_col].to_numpy(dtype=float)

    cont_move_price = cfg.continuation_points * cfg.point_value
    rev_move_price = cfg.reversal_points * cfg.point_value

    rows: List[Dict[str, Any]] = []

    for i in signal_idx:
        if i + 1 >= len(features):
            continue

        end = min(i + 1 + cfg.max_forward_bars, len(features))
        future_highs = high[i + 1:end]
        future_lows = low[i + 1:end]
        entry_price = close[i]
        side = int(features.iloc[i]["trend_side"])

        outcome, bars_to_outcome, mfe_price, mae_price = _forward_path_outcome(
            side=side,
            entry_price=entry_price,
            future_highs=future_highs,
            future_lows=future_lows,
            cont_move_price=cont_move_price,
            rev_move_price=rev_move_price,
        )

        row: Dict[str, Any] = {
            "row_index": int(i),
            "entry_price": float(entry_price),
            "side": side,
            "side_name": "long" if side == 1 else "short",
            "outcome": outcome,
            "bars_to_outcome": bars_to_outcome,
            "mfe_price": mfe_price,
            "mae_price": mae_price,
            "mfe_points": mfe_price / cfg.point_value if pd.notna(mfe_price) else np.nan,
            "mae_points": mae_price / cfg.point_value if pd.notna(mae_price) else np.nan,
            "ema20": float(features.iloc[i]["ema20"]),
            "ema200": float(features.iloc[i]["ema200"]),
            "trend_side": int(features.iloc[i]["trend_side"]),
            "price_side_vs_ema20": int(features.iloc[i]["price_side_vs_ema20"]),
            "ema20_slope_pct": float(features.iloc[i]["ema20_slope_pct"]),
            "ema200_slope_pct": float(features.iloc[i]["ema200_slope_pct"]),
            "e20_to_e200_pct": float(features.iloc[i]["e20_to_e200_pct"]),
            "abs_e20_to_e200_pct": float(features.iloc[i]["abs_e20_to_e200_pct"]),
            "bucket_index": int(features.iloc[i]["bucket_index"]),
            "bucket_text": str(features.iloc[i]["bucket_text"]),
            "stage": int(features.iloc[i]["stage"]),
            "stage_text": str(features.iloc[i]["stage_text"]),
            "research_signal": bool(features.iloc[i]["research_signal"]),
        }

        if cfg.time_col and cfg.time_col in features.columns:
            row[cfg.time_col] = features.iloc[i][cfg.time_col]

        rows.append(row)

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    events["is_continuation"] = events["outcome"].eq("continuation")
    events["is_reversal"] = events["outcome"].eq("reversal")
    events["is_unresolved"] = events["outcome"].eq("unresolved")

    return events

# =============================================================================
# BUCKET STATISTICS
# =============================================================================

def summarize_by_bucket(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    grouped = events.groupby(["bucket_index", "bucket_text"], observed=True)

    summary = grouped.agg(
        samples=("outcome", "size"),
        continuation_count=("is_continuation", "sum"),
        reversal_count=("is_reversal", "sum"),
        unresolved_count=("is_unresolved", "sum"),
        avg_abs_distance_pct=("abs_e20_to_e200_pct", "mean"),
        min_abs_distance_pct=("abs_e20_to_e200_pct", "min"),
        max_abs_distance_pct=("abs_e20_to_e200_pct", "max"),
        avg_signed_distance_pct=("e20_to_e200_pct", "mean"),
        avg_mfe_points=("mfe_points", "mean"),
        avg_mae_points=("mae_points", "mean"),
        median_bars_to_outcome=("bars_to_outcome", "median"),
    ).reset_index()

    summary["continuation_rate"] = summary["continuation_count"] / summary["samples"]
    summary["reversal_rate"] = summary["reversal_count"] / summary["samples"]
    summary["unresolved_rate"] = summary["unresolved_count"] / summary["samples"]
    summary["edge"] = summary["continuation_rate"] - summary["reversal_rate"]
    summary["cont_rev_ratio"] = np.where(
        summary["reversal_count"] > 0,
        summary["continuation_count"] / summary["reversal_count"],
        np.nan,
    )

    return summary.sort_values("bucket_index").reset_index(drop=True)


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================

def classify_bucket_zone(row: pd.Series, cfg: EmaDistanceCalibrationConfig) -> int:
    samples = int(row["samples"])
    cont = float(row["continuation_rate"])
    rev = float(row["reversal_rate"])

    if samples < cfg.min_bucket_samples:
        return 0
    if cont >= cfg.zone1_min_continuation_rate and rev <= cfg.max_zone1_reversal_rate:
        return 1
    if cont >= cfg.zone2_min_continuation_rate and rev <= cfg.max_zone2_reversal_rate:
        return 2
    return 3


def recommend_zones(
    bucket_summary: pd.DataFrame,
    cfg: EmaDistanceCalibrationConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if bucket_summary.empty:
        return bucket_summary.copy(), {
            "zone_1": None,
            "zone_2": None,
            "zone_3": None,
            "note": "No events available for zone recommendation.",
        }

    out = bucket_summary.copy()
    out["zone"] = out.apply(lambda row: classify_bucket_zone(row, cfg), axis=1)
    out["zone_name"] = out["zone"].map({0: "Insufficient", 1: "Zone 1", 2: "Zone 2", 3: "Zone 3"})
    
    def _range_for_zone(zone_num: int) -> Optional[Dict[str, Any]]:
        sub = out.loc[out["zone"] == zone_num].sort_values("bucket_index")
        if sub.empty:
            return None

        valid_ranges = contiguous_ranges(sub["bucket_index"].tolist())
        largest = max(valid_ranges, key=lambda x: x[1] - x[0])

        selected = out[
            (out["bucket_index"] >= largest[0]) &
            (out["bucket_index"] <= largest[1])
        ].sort_values("bucket_index")

        return {
            "bucket_start": str(selected.iloc[0]["bucket_text"]),
            "bucket_end": str(selected.iloc[-1]["bucket_text"]),
            "bucket_index_start": int(selected.iloc[0]["bucket_index"]),
            "bucket_index_end": int(selected.iloc[-1]["bucket_index"]),
            "min_abs_distance_pct": float(selected["min_abs_distance_pct"].min()),
            "max_abs_distance_pct": float(selected["max_abs_distance_pct"].max()),
            "samples": int(selected["samples"].sum()),
            "avg_continuation_rate": float(selected["continuation_count"].sum() / selected["samples"].sum()),
            "avg_reversal_rate": float(selected["reversal_count"].sum() / selected["samples"].sum()),
        }

    recommendation = {
        "zone_1": _range_for_zone(1),
        "zone_2": _range_for_zone(2),
        "zone_3": _range_for_zone(3),
        "note": (
            "Zones are derived from absolute EMA20-to-EMA200 percentage distance. "
            "Use signed distance together with trend side for directional context."
        ),
        "pine_stage_thresholds": {
            "zone1_max_pct": cfg.zone1_max_pct,
            "zone2_max_pct": cfg.zone2_max_pct,
            "zone3_min_pct": cfg.zone3_min_pct,
        },
    }

    return out, recommendation

# =============================================================================
# PUBLIC RESEARCH API
# =============================================================================

def run_ema_distance_calibration(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceCalibrationConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceCalibrationConfig()

    features = build_feature_frame(df, cfg)
    events = run_forward_outcome_test(features, cfg)
    bucket_summary = summarize_by_bucket(events)
    bucket_with_zones, recommendation = recommend_zones(bucket_summary, cfg)

    if events.empty:
        headline: Dict[str, Any] = {
            "signals": 0,
            "continuation_rate": np.nan,
            "reversal_rate": np.nan,
            "unresolved_rate": np.nan,
        }
    else:
        headline = {
            "signals": int(len(events)),
            "continuation_rate": float(events["is_continuation"].mean()),
            "reversal_rate": float(events["is_reversal"].mean()),
            "unresolved_rate": float(events["is_unresolved"].mean()),
        }

    return {
        "config": asdict(cfg),
        "headline": headline,
        "events": events,
        "bucket_summary": bucket_summary,
        "bucket_with_zones": bucket_with_zones,
        "recommendation": recommendation,
    }