from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd


# =============================================================================
# SMARTCHART • EMA DISTANCE CALIBRATION PARITY
# File: cc_ema_distance_calibration.py
# =============================================================================
# Purpose
# - Strict Pine-authority parity for EMA20 vs EMA200 percentage distance
# - Preserve lower-pane research structure for future visual use
# - Add forward continuation/reversal calibration research
#
# Pine reference structure preserved:
# 1. Inputs / Config
# 2. EMA Core
# 3. Distance Engine
# 4. Signal Filter Engine
# 5. Bucket Engine
# 6. Stage Engine
# 7. Export / Status Outputs
# 8. Research Forward Test Engine
# 9. Bucket Statistics + Zone Recommendation
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class CalibrationConfig:
    # -------------------------------------------------------------------------
    # Core
    # -------------------------------------------------------------------------
    ema_fast_len: int = 20
    ema_slow_len: int = 200
    slope_len: int = 5

    # -------------------------------------------------------------------------
    # Signal filters
    # -------------------------------------------------------------------------
    require_price_on_trend_side: bool = True
    require_fast_slope_confirm: bool = True
    require_slow_slope_confirm: bool = False
    min_abs_distance_pct: float = 0.0

    # -------------------------------------------------------------------------
    # Pine bucket boundaries
    # Matches Pine script defaults exactly:
    # b1=0.00, b2=0.25, b3=0.50, b4=0.75, b5=1.00,
    # b6=1.25, b7=1.50, b8=2.00, b9=2.50, b10=3.00
    # -------------------------------------------------------------------------
    bucket_edges_pct: Tuple[float, ...] = (
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
        2.50,
        3.00,
        math.inf,
    )

    # -------------------------------------------------------------------------
    # Pine stage thresholds
    # -------------------------------------------------------------------------
    zone1_max_pct: float = 0.75
    zone2_max_pct: float = 1.50
    zone3_min_pct: float = 1.50

    # -------------------------------------------------------------------------
    # Forward test research
    # -------------------------------------------------------------------------
    continuation_points: float = 60.0
    reversal_points: float = 30.0
    max_forward_bars: int = 60

    # -------------------------------------------------------------------------
    # Instrument point conversion
    # Example:
    # - point_value=0.01 for XAUUSD if 1 point = 0.01 price units
    # - point_value=1.0  if 1 point = 1.0 price units
    # -------------------------------------------------------------------------
    point_value: float = 1.0

    # -------------------------------------------------------------------------
    # Recommendation engine
    # -------------------------------------------------------------------------
    min_bucket_samples: int = 200
    zone1_min_continuation_rate: float = 0.58
    zone2_min_continuation_rate: float = 0.52
    max_zone1_reversal_rate: float = 0.34
    max_zone2_reversal_rate: float = 0.42

    # -------------------------------------------------------------------------
    # Column names
    # -------------------------------------------------------------------------
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    time_col: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    """EMA helper with TradingView-style warmup alignment."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def pct_change_from_past(series: pd.Series, lookback: int) -> pd.Series:
    prev = series.shift(lookback)
    out = np.where(prev != 0.0, ((series - prev) / prev) * 100.0, np.nan)
    return pd.Series(out, index=series.index, dtype=float)


def validate_ohlc(df: pd.DataFrame, cfg: CalibrationConfig) -> None:
    required = [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")

    min_rows = max(cfg.ema_slow_len, cfg.max_forward_bars) + cfg.slope_len + 5
    if len(df) < min_rows:
        raise ValueError(
            f"Not enough rows for EMA warmup and forward testing. "
            f"Need at least {min_rows}, got {len(df)}."
        )


def build_bucket_labels(edges: Sequence[float]) -> List[str]:
    labels: List[str] = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if math.isinf(right):
            labels.append(f"{left:.2f}%+")
        else:
            labels.append(f"{left:.2f}-{right:.2f}%")
    return labels


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


def _bucket_index_from_abs_dist(abs_dist: pd.Series, edges: Sequence[float]) -> pd.Series:
    """
    Pine-style bucket index:
    0 = NA
    1 = [b1,b2)
    2 = [b2,b3)
    ...
    9 = [b9,b10)
    10 = [b10,inf)
    """
    values = abs_dist.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=int)

    finite_edges = list(edges[:-1])  # exclude inf for the comparisons
    # For Pine equivalence, compare against upper edges only:
    # < b2 => 1, < b3 => 2, ... else 10
    if len(finite_edges) < 10:
        raise ValueError("bucket_edges_pct must include Pine-style edges plus infinity.")

    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = 0
        elif v < finite_edges[1]:
            out[i] = 1
        elif v < finite_edges[2]:
            out[i] = 2
        elif v < finite_edges[3]:
            out[i] = 3
        elif v < finite_edges[4]:
            out[i] = 4
        elif v < finite_edges[5]:
            out[i] = 5
        elif v < finite_edges[6]:
            out[i] = 6
        elif v < finite_edges[7]:
            out[i] = 7
        elif v < finite_edges[8]:
            out[i] = 8
        elif v < finite_edges[9]:
            out[i] = 9
        else:
            out[i] = 10

    return pd.Series(out, index=abs_dist.index, dtype=int)


def _bucket_text_from_index(bucket_index: pd.Series, edges: Sequence[float]) -> pd.Series:
    labels = build_bucket_labels(edges)

    mapping: Dict[int, str] = {
        1: labels[0],
        2: labels[1],
        3: labels[2],
        4: labels[3],
        5: labels[4],
        6: labels[5],
        7: labels[6],
        8: labels[7],
        9: labels[8],
        10: labels[9],
        0: "NA",
    }
    return bucket_index.map(mapping).fillna("NA")


def _stage_from_abs_dist(abs_dist: pd.Series, cfg: CalibrationConfig) -> pd.Series:
    """
    Pine-style stage logic:
    stage =
         na(absDistPct) ? 0 :
         absDistPct < zone1MaxPct ? 1 :
         absDistPct < zone2MaxPct ? 2 :
         absDistPct >= zone3MinPct ? 3 : 0
    """
    values = abs_dist.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=int)

    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = 0
        elif v < cfg.zone1_max_pct:
            out[i] = 1
        elif v < cfg.zone2_max_pct:
            out[i] = 2
        elif v >= cfg.zone3_min_pct:
            out[i] = 3
        else:
            out[i] = 0

    return pd.Series(out, index=abs_dist.index, dtype=int)


def _stage_text(stage: pd.Series) -> pd.Series:
    mapping = {
        0: "NONE",
        1: "ZONE 1",
        2: "ZONE 2",
        3: "ZONE 3",
    }
    return stage.map(mapping).fillna("NONE")


def _direction_text(v: int) -> str:
    if v == 1:
        return "BULL"
    if v == -1:
        return "BEAR"
    return "NEUTRAL"


# =============================================================================
# EMA CORE (PINE PARITY)
# =============================================================================

def build_ema_core(df: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    validate_ohlc(df, cfg)

    out = df.copy()

    close = out[cfg.close_col].astype(float)

    out["ema20"] = ema(close, cfg.ema_fast_len)
    out["ema200"] = ema(close, cfg.ema_slow_len)

    out["ema20_slope_pct"] = pct_change_from_past(out["ema20"], cfg.slope_len)
    out["ema200_slope_pct"] = pct_change_from_past(out["ema200"], cfg.slope_len)

    out["trend_side"] = np.where(
        out["ema20"] > out["ema200"],
        1,
        np.where(out["ema20"] < out["ema200"], -1, 0),
    )

    out["price_side_vs_ema20"] = np.where(
        out[cfg.close_col] > out["ema20"],
        1,
        np.where(out[cfg.close_col] < out["ema20"], -1, 0),
    )

    return out


# =============================================================================
# DISTANCE ENGINE (PINE PARITY)
# =============================================================================

def build_distance_engine(features: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    out = features.copy()

    out["e20_to_e200_signed"] = out["ema20"] - out["ema200"]
    out["e20_to_e200_pct"] = np.where(
        out["ema200"] != 0.0,
        ((out["ema20"] - out["ema200"]) / out["ema200"]) * 100.0,
        np.nan,
    )
    out["abs_e20_to_e200_pct"] = out["e20_to_e200_pct"].abs()

    return out


# =============================================================================
# SIGNAL FILTER ENGINE (PINE PARITY)
# =============================================================================

def build_signal_engine(features: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    out = features.copy()

    base_signal = (
        (out["trend_side"] != 0)
        & out["abs_e20_to_e200_pct"].notna()
        & (out["abs_e20_to_e200_pct"] >= cfg.min_abs_distance_pct)
    )

    if cfg.require_price_on_trend_side:
        price_filter = out["price_side_vs_ema20"] == out["trend_side"]
    else:
        price_filter = pd.Series(True, index=out.index)

    if cfg.require_fast_slope_confirm:
        fast_slope_filter = np.where(
            out["trend_side"] == 1,
            out["ema20_slope_pct"] > 0,
            out["ema20_slope_pct"] < 0,
        )
        fast_slope_filter = pd.Series(fast_slope_filter, index=out.index)
    else:
        fast_slope_filter = pd.Series(True, index=out.index)

    if cfg.require_slow_slope_confirm:
        slow_slope_filter = np.where(
            out["trend_side"] == 1,
            out["ema200_slope_pct"] >= 0,
            out["ema200_slope_pct"] <= 0,
        )
        slow_slope_filter = pd.Series(slow_slope_filter, index=out.index)
    else:
        slow_slope_filter = pd.Series(True, index=out.index)

    out["base_signal"] = base_signal.fillna(False)
    out["price_filter"] = price_filter.fillna(False)
    out["fast_slope_filter"] = fast_slope_filter.fillna(False)
    out["slow_slope_filter"] = slow_slope_filter.fillna(False)

    out["research_signal"] = (
        out["base_signal"]
        & out["price_filter"]
        & out["fast_slope_filter"]
        & out["slow_slope_filter"]
        & out["ema20"].notna()
        & out["ema200"].notna()
    ).fillna(False)

    return out


# =============================================================================
# BUCKET ENGINE (PINE PARITY)
# =============================================================================

def build_bucket_engine(features: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    out = features.copy()

    out["bucket_index"] = _bucket_index_from_abs_dist(
        out["abs_e20_to_e200_pct"],
        cfg.bucket_edges_pct,
    )
    out["bucket_text"] = _bucket_text_from_index(
        out["bucket_index"],
        cfg.bucket_edges_pct,
    )

    return out


# =============================================================================
# STAGE ENGINE (PINE PARITY)
# =============================================================================

def build_stage_engine(features: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    out = features.copy()

    out["stage"] = _stage_from_abs_dist(out["abs_e20_to_e200_pct"], cfg)
    out["stage_text"] = _stage_text(out["stage"])

    return out


# =============================================================================
# FEATURE FRAME (FULL PARITY BUILD)
# =============================================================================

def build_feature_frame(df: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    out = build_ema_core(df, cfg)
    out = build_distance_engine(out, cfg)
    out = build_signal_engine(out, cfg)
    out = build_bucket_engine(out, cfg)
    out = build_stage_engine(out, cfg)

    out["trend_text"] = out["trend_side"].map(_direction_text)

    return out


# =============================================================================
# FORWARD OUTCOME ENGINE
# =============================================================================

def _forward_path_outcome(
    side: int,
    entry_price: float,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    cont_move_price: float,
    rev_move_price: float,
) -> Tuple[str, Optional[int], float, float]:
    """
    Returns:
    - outcome: continuation | reversal | unresolved
    - bars_to_outcome
    - mfe_price: max favorable excursion in price units
    - mae_price: max adverse excursion in price units
    """
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


def run_forward_outcome_test(features: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    signal_mask = features["research_signal"].fillna(False)
    signal_idx = np.flatnonzero(signal_mask.to_numpy())

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

def summarize_by_bucket(events: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
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

    summary = summary.sort_values("bucket_index").reset_index(drop=True)
    return summary


# =============================================================================
# ZONE RECOMMENDATION ENGINE
# =============================================================================

def classify_bucket_zone(row: pd.Series, cfg: CalibrationConfig) -> int:
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
    cfg: CalibrationConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if bucket_summary.empty:
        return bucket_summary.copy(), {
            "zone_1": None,
            "zone_2": None,
            "zone_3": None,
            "note": "No events available for zone recommendation.",
        }

    out = bucket_summary.copy()
    out["zone"] = out.apply(lambda row: classify_bucket_zone(row, cfg), axis=1)
    out["zone_name"] = out["zone"].map({
        0: "Insufficient",
        1: "Zone 1",
        2: "Zone 2",
        3: "Zone 3",
    })

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
            "avg_continuation_rate": float(
                selected["continuation_count"].sum() / selected["samples"].sum()
            ),
            "avg_reversal_rate": float(
                selected["reversal_count"].sum() / selected["samples"].sum()
            ),
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
    cfg: Optional[CalibrationConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or CalibrationConfig()

    features = build_feature_frame(df, cfg)
    events = run_forward_outcome_test(features, cfg)
    bucket_summary = summarize_by_bucket(events, cfg)
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

    latest: Dict[str, Any]
    if features.empty:
        latest = {}
    else:
        last = features.iloc[-1]
        latest = {
            "trend_side": int(last["trend_side"]) if pd.notna(last["trend_side"]) else 0,
            "trend_text": str(last["trend_text"]),
            "price_side_vs_ema20": int(last["price_side_vs_ema20"]) if pd.notna(last["price_side_vs_ema20"]) else 0,
            "ema20_slope_pct": float(last["ema20_slope_pct"]) if pd.notna(last["ema20_slope_pct"]) else np.nan,
            "ema200_slope_pct": float(last["ema200_slope_pct"]) if pd.notna(last["ema200_slope_pct"]) else np.nan,
            "e20_to_e200_pct": float(last["e20_to_e200_pct"]) if pd.notna(last["e20_to_e200_pct"]) else np.nan,
            "abs_e20_to_e200_pct": float(last["abs_e20_to_e200_pct"]) if pd.notna(last["abs_e20_to_e200_pct"]) else np.nan,
            "bucket_index": int(last["bucket_index"]) if pd.notna(last["bucket_index"]) else 0,
            "bucket_text": str(last["bucket_text"]),
            "stage": int(last["stage"]) if pd.notna(last["stage"]) else 0,
            "stage_text": str(last["stage_text"]),
            "research_signal": bool(last["research_signal"]),
        }

        if cfg.time_col and cfg.time_col in features.columns:
            latest[cfg.time_col] = last[cfg.time_col]

    return {
        "config": asdict(cfg),
        "headline": headline,
        "latest": latest,
        "features": features,
        "events": events,
        "bucket_summary": bucket_summary,
        "bucket_with_zones": bucket_with_zones,
        "recommendation": recommendation,
    }


# =============================================================================
# REPORTING
# =============================================================================

def format_recommendation_text(result: Dict[str, Any]) -> str:
    rec = result["recommendation"]
    headline = result["headline"]
    latest = result.get("latest", {})

    lines: List[str] = []
    lines.append("EMA20–EMA200 Distance Calibration Report")
    lines.append("=" * 44)
    lines.append(f"Signals: {headline['signals']}")

    if headline["signals"]:
        lines.append(f"Continuation rate: {headline['continuation_rate']:.2%}")
        lines.append(f"Reversal rate:    {headline['reversal_rate']:.2%}")
        lines.append(f"Unresolved rate:  {headline['unresolved_rate']:.2%}")

    if latest:
        lines.append("")
        lines.append("Latest Bar Status")
        lines.append("-" * 17)
        lines.append(f"Trend:   {latest.get('trend_text', '-')}")
        lines.append(f"Signal:  {'YES' if latest.get('research_signal', False) else 'NO'}")
        lines.append(f"Signed %: {latest.get('e20_to_e200_pct', np.nan):.4f}")
        lines.append(f"Abs %:    {latest.get('abs_e20_to_e200_pct', np.nan):.4f}")
        lines.append(f"Bucket:   {latest.get('bucket_text', '-')}")
        lines.append(f"Stage:    {latest.get('stage_text', '-')}")

    lines.append("")
    for key, title in [("zone_1", "Zone 1"), ("zone_2", "Zone 2"), ("zone_3", "Zone 3")]:
        zone = rec.get(key)
        if not zone:
            lines.append(f"{title}: no robust range found")
            continue

        lines.append(
            f"{title}: {zone['bucket_start']} -> {zone['bucket_end']} "
            f"(observed {zone['min_abs_distance_pct']:.3f}% to "
            f"{zone['max_abs_distance_pct']:.3f}%, samples={zone['samples']}, "
            f"cont={zone['avg_continuation_rate']:.2%}, rev={zone['avg_reversal_rate']:.2%})"
        )

    lines.append("")
    lines.append(str(rec.get("note", "")))
    return "\n".join(lines)


# =============================================================================
# DEMO DATA
# =============================================================================

def _make_demo_data(rows: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.35, rows)
    close = 2300 + np.cumsum(returns)
    high = close + rng.uniform(0.05, 0.45, rows)
    low = close - rng.uniform(0.05, 0.45, rows)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    idx = pd.date_range("2025-01-01", periods=rows, freq="min")
    return pd.DataFrame({
        "time": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import glob

    files = glob.glob("data/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in data/")

    df_list: List[pd.DataFrame] = []

    for f in files:
        temp = pd.read_csv(
            f,
            header=None,
            skiprows=1,
            names=["datetime", "open", "high", "low", "close", "volume"],
        )

        temp["datetime"] = (
            temp["datetime"]
            .astype(str)
            .str.replace(" UTC", "", regex=False)
            .str.strip()
        )

        temp["datetime"] = pd.to_datetime(
            temp["datetime"],
            format="%d.%m.%Y %H:%M:%S.%f",
            errors="coerce",
        )

        for col in ["open", "high", "low", "close", "volume"]:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")

        temp = temp.dropna(subset=["datetime", "open", "high", "low", "close"])
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)
    df = (
        df.sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    print("Rows loaded:", len(df))
    print(df.head())
    print(df.tail())

    cfg = CalibrationConfig(
        continuation_points=30,
        reversal_points=30,
        max_forward_bars=60,
        point_value=0.01,
        time_col="datetime",
    )

    result = run_ema_distance_calibration(df, cfg)

    print(format_recommendation_text(result))
    print("\nBucket summary preview:\n")

    if not result["bucket_with_zones"].empty:
        print(
            result["bucket_with_zones"][
                [
                    "bucket_index",
                    "bucket_text",
                    "samples",
                    "continuation_rate",
                    "reversal_rate",
                    "edge",
                    "zone_name",
                ]
            ].to_string(index=False)
        )
    else:
        print("No qualifying events found.")