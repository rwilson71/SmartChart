from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd


# =============================================================================
# EMA DISTANCE CALIBRATION RESEARCH MODULE
# =============================================================================
# Purpose:
# - Research the percentage distance between EMA20 and EMA200
# - Bucket that distance into ranges
# - Test forward continuation vs reversal outcomes after a signal
# - Recommend Zone 1 / Zone 2 / Zone 3 thresholds from observed data
#
# Designed to be instrument-agnostic:
# - Works on any instrument and timeframe
# - Uses configurable target/stop distances in price points
# - Uses configurable bucket definitions and forward horizon
#
# Pine parity anchor:
# e20ToE200Pct = ((ema20 - ema200) / ema200) * 100.0
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class CalibrationConfig:
    # EMA settings
    ema_fast_len: int = 20
    ema_slow_len: int = 200
    slope_len: int = 5

    # Signal filters
    require_price_on_trend_side: bool = True
    require_fast_slope_confirmation: bool = True
    require_slow_slope_confirmation: bool = False
    min_abs_distance_pct: float = 0.0

    # Forward test definition
    continuation_points: float = 60.0
    reversal_points: float = 30.0
    max_forward_bars: int = 60

    # Instrument scaling
    # For XAUUSD many users think in "points" as raw price units like 0.01, 0.10, 1.0 etc.
    # This parameter converts user-defined points into price units used by the dataframe.
    # Example:
    # - If 1 point = 0.01 price units, set point_value=0.01
    # - If 1 point = 1.0 price units, set point_value=1.0
    point_value: float = 1.0

    bucket_edges_pct: Tuple[float, ...] = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    math.inf,
)
    
    # Recommendation engine
    min_bucket_samples: int = 200
    zone1_min_continuation_rate: float = 0.58
    zone2_min_continuation_rate: float = 0.52
    max_zone1_reversal_rate: float = 0.34
    max_zone2_reversal_rate: float = 0.42

    # Column names
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    time_col: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    """EMA helper with min_periods aligned to TradingView-style warmup behavior."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def pct_change_from_past(series: pd.Series, lookback: int) -> pd.Series:
    prev = series.shift(lookback)
    return np.where(prev != 0.0, ((series - prev) / prev) * 100.0, np.nan)


def validate_ohlc(df: pd.DataFrame, cfg: CalibrationConfig) -> None:
    required = [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if len(df) < max(cfg.ema_slow_len, cfg.max_forward_bars) + cfg.slope_len + 5:
        raise ValueError("Not enough rows for EMA warmup and forward testing.")


def build_bucket_labels(edges: Sequence[float]) -> List[str]:
    labels: List[str] = []
    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]
        if math.isinf(right):
            labels.append(f"{left:.2f}%+")
        else:
            labels.append(f"{left:.2f}%–{right:.2f}%")
    return labels


def bucketize_abs_distance(abs_distance_pct: pd.Series, edges: Sequence[float]) -> pd.Categorical:
    labels = build_bucket_labels(edges)
    return pd.cut(
        abs_distance_pct,
        bins=list(edges),
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=True,
    )


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


# =============================================================================
# FEATURE ENGINE
# =============================================================================

def build_feature_frame(df: pd.DataFrame, cfg: CalibrationConfig) -> pd.DataFrame:
    validate_ohlc(df, cfg)

    out = df.copy()

    close = out[cfg.close_col].astype(float)

    out["ema20"] = ema(close, cfg.ema_fast_len)
    out["ema200"] = ema(close, cfg.ema_slow_len)

    out["e20_to_e200_signed"] = out["ema20"] - out["ema200"]
    out["e20_to_e200_pct"] = np.where(
        out["ema200"] != 0.0,
        ((out["ema20"] - out["ema200"]) / out["ema200"]) * 100.0,
        np.nan,
    )
    out["abs_e20_to_e200_pct"] = out["e20_to_e200_pct"].abs()

    out["ema20_slope_pct"] = pct_change_from_past(out["ema20"], cfg.slope_len)
    out["ema200_slope_pct"] = pct_change_from_past(out["ema200"], cfg.slope_len)

    out["trend_side"] = np.where(
        out["ema20"] > out["ema200"], 1,
        np.where(out["ema20"] < out["ema200"], -1, 0)
    )

    out["price_side_vs_ema20"] = np.where(
        out[cfg.close_col] > out["ema20"], 1,
        np.where(out[cfg.close_col] < out["ema20"], -1, 0)
    )

    out["distance_bucket"] = bucketize_abs_distance(out["abs_e20_to_e200_pct"], cfg.bucket_edges_pct)

    return out


# =============================================================================
# SIGNAL ENGINE
# =============================================================================

def build_signal_mask(features: pd.DataFrame, cfg: CalibrationConfig) -> pd.Series:
    mask = features["trend_side"] != 0
    mask &= features["abs_e20_to_e200_pct"] >= cfg.min_abs_distance_pct

    if cfg.require_price_on_trend_side:
        mask &= features["price_side_vs_ema20"] == features["trend_side"]

    if cfg.require_fast_slope_confirmation:
        mask &= np.where(
            features["trend_side"] == 1,
            features["ema20_slope_pct"] > 0,
            features["ema20_slope_pct"] < 0,
        )

    if cfg.require_slow_slope_confirmation:
        mask &= np.where(
            features["trend_side"] == 1,
            features["ema200_slope_pct"] >= 0,
            features["ema200_slope_pct"] <= 0,
        )

    mask &= features["distance_bucket"].notna()
    mask &= features["ema20"].notna()
    mask &= features["ema200"].notna()

    return mask.fillna(False)


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
    signal_mask = build_signal_mask(features, cfg)
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
            "e20_to_e200_pct": float(features.iloc[i]["e20_to_e200_pct"]),
            "abs_e20_to_e200_pct": float(features.iloc[i]["abs_e20_to_e200_pct"]),
            "ema20_slope_pct": float(features.iloc[i]["ema20_slope_pct"]),
            "ema200_slope_pct": float(features.iloc[i]["ema200_slope_pct"]),
            "distance_bucket": features.iloc[i]["distance_bucket"],
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

    def _first(x: pd.Series) -> Any:
        return x.iloc[0] if len(x) else np.nan

    grouped = events.groupby("distance_bucket", observed=True)

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

    summary["bucket_rank"] = np.arange(len(summary))
    return summary.sort_values("bucket_rank").reset_index(drop=True)


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


def recommend_zones(bucket_summary: pd.DataFrame, cfg: CalibrationConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    def _range_for_zone(zone_num: int) -> Optional[Dict[str, float]]:
        sub = out.loc[out["zone"] == zone_num].sort_values("bucket_rank")
        if sub.empty:
            return None
        valid_ranges = contiguous_ranges(sub["bucket_rank"].tolist())
        largest = max(valid_ranges, key=lambda x: x[1] - x[0])
        selected = out[(out["bucket_rank"] >= largest[0]) & (out["bucket_rank"] <= largest[1])]
        return {
            "bucket_start": str(selected.iloc[0]["distance_bucket"]),
            "bucket_end": str(selected.iloc[-1]["distance_bucket"]),
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
            "Use signed distance alongside trend side for directional context."
        ),
    }

    return out, recommendation


# =============================================================================
# PUBLIC RESEARCH API
# =============================================================================

def run_ema_distance_calibration(df: pd.DataFrame, cfg: Optional[CalibrationConfig] = None) -> Dict[str, Any]:
    cfg = cfg or CalibrationConfig()

    features = build_feature_frame(df, cfg)
    events = run_forward_outcome_test(features, cfg)
    bucket_summary = summarize_by_bucket(events, cfg)
    bucket_with_zones, recommendation = recommend_zones(bucket_summary, cfg)

    headline: Dict[str, Any]
    if events.empty:
        headline = {
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

    lines: List[str] = []
    lines.append("EMA20–EMA200 Distance Calibration Report")
    lines.append("=" * 44)
    lines.append(f"Signals: {headline['signals']}")

    if headline["signals"]:
        lines.append(f"Continuation rate: {headline['continuation_rate']:.2%}")
        lines.append(f"Reversal rate:    {headline['reversal_rate']:.2%}")
        lines.append(f"Unresolved rate:  {headline['unresolved_rate']:.2%}")

    lines.append("")

    for key, title in [("zone_1", "Zone 1"), ("zone_2", "Zone 2"), ("zone_3", "Zone 3")]:
        zone = rec.get(key)
        if not zone:
            lines.append(f"{title}: no robust range found")
            continue
        lines.append(
            f"{title}: {zone['bucket_start']} -> {zone['bucket_end']} "
            f"(observed {zone['min_abs_distance_pct']:.3f}% to {zone['max_abs_distance_pct']:.3f}%, "
            f"samples={zone['samples']}, cont={zone['avg_continuation_rate']:.2%}, rev={zone['avg_reversal_rate']:.2%})"
        )

    lines.append("")
    lines.append(str(rec.get("note", "")))
    return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
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

if __name__ == "__main__":
    import pandas as pd
    import glob

    files = glob.glob("data/*.csv")

    if not files:
        raise FileNotFoundError("No CSV files found in data/")

    df_list = []

    for f in files:
        temp = pd.read_csv(
            f,
            header=None,
            skiprows=1,
            names=["datetime", "open", "high", "low", "close", "volume"],
        )

        # clean datetime like: "01.03.2024 00:00:00.000 UTC"
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
                    "distance_bucket",
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