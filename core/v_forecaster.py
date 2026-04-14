from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ForecasterConfig:
    # Seasonal window
    window_days: int = 30

    # Historical buckets
    use_3y: bool = True
    use_5y: bool = True
    use_7y: bool = True
    use_10y: bool = True

    # Agreement / permission thresholds
    bull_bias_threshold: float = 0.55
    bear_bias_threshold: float = 0.55
    min_samples_per_bucket: int = 3
    strong_agreement_threshold: float = 0.70

    # Current run mode
    evaluate_last_n_rows: int = 200  # speed-safe: only compute the most recent rows


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def _year_buckets(cfg: ForecasterConfig) -> List[int]:
    out: List[int] = []
    if cfg.use_3y:
        out.append(3)
    if cfg.use_5y:
        out.append(5)
    if cfg.use_7y:
        out.append(7)
    if cfg.use_10y:
        out.append(10)
    return out


def _safe_mean_bool(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())


def _score_direction(bull_prob: float, bear_prob: float, bull_thr: float, bear_thr: float) -> int:
    if bull_prob >= bull_thr and bull_prob > bear_prob:
        return 1
    if bear_prob >= bear_thr and bear_prob > bull_prob:
        return -1
    return 0


def _daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    ).dropna()

    daily["day_return"] = (daily["close"] / daily["open"]) - 1.0
    daily["is_bull"] = (daily["day_return"] > 0).astype(int)
    daily["is_bear"] = (daily["day_return"] < 0).astype(int)
    daily["weekday"] = daily.index.weekday
    daily["month"] = daily.index.month
    daily["day"] = daily.index.day
    daily["dayofyear"] = daily.index.dayofyear
    daily["year"] = daily.index.year
    return daily


def _window_mask(dayofyear_series: pd.Series, target_dayofyear: int, window_days: int) -> pd.Series:
    """
    Wrap-safe seasonal window around day-of-year.
    """
    dist = (dayofyear_series - target_dayofyear).abs()
    wrap_dist = 366 - dist
    min_dist = np.minimum(dist, wrap_dist)
    return pd.Series(min_dist <= window_days, index=dayofyear_series.index)


def _bucket_stats(
    daily: pd.DataFrame,
    current_date: pd.Timestamp,
    years_back: int,
    cfg: ForecasterConfig,
) -> dict:
    """
    Historical weekday behavior inside seasonal window for a given lookback bucket.
    """
    start_date = current_date - pd.DateOffset(years=years_back)
    hist = daily[(daily.index < current_date) & (daily.index >= start_date)].copy()

    if hist.empty:
        return {
            "samples": 0,
            "bull_prob": 0.5,
            "bear_prob": 0.5,
            "avg_return": 0.0,
            "direction": 0,
            "valid": 0,
        }

    target_weekday = current_date.weekday()
    target_dayofyear = current_date.dayofyear

    seasonal_mask = _window_mask(hist["dayofyear"], target_dayofyear, cfg.window_days)
    weekday_mask = hist["weekday"] == target_weekday
    matched = hist[seasonal_mask & weekday_mask]

    samples = int(len(matched))
    if samples < cfg.min_samples_per_bucket:
        return {
            "samples": samples,
            "bull_prob": 0.5,
            "bear_prob": 0.5,
            "avg_return": 0.0,
            "direction": 0,
            "valid": 0,
        }

    bull_prob = _safe_mean_bool(matched["is_bull"] == 1)
    bear_prob = _safe_mean_bool(matched["is_bear"] == 1)
    avg_return = float(matched["day_return"].mean())

    direction = _score_direction(
        bull_prob=bull_prob,
        bear_prob=bear_prob,
        bull_thr=cfg.bull_bias_threshold,
        bear_thr=cfg.bear_bias_threshold,
    )

    return {
        "samples": samples,
        "bull_prob": bull_prob,
        "bear_prob": bear_prob,
        "avg_return": avg_return,
        "direction": direction,
        "valid": 1,
    }


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_forecaster(
    df: pd.DataFrame,
    config: Optional[ForecasterConfig] = None,
) -> pd.DataFrame:
    """
    SmartChart historical day-bias forecaster.

    Purpose:
    - evaluate current weekday behavior inside the current seasonal date window
    - compare 3Y / 5Y / 7Y / 10Y history
    - return bullish/bearish percentage, agreement, and final directional bias

    Expected input:
        DataFrame indexed by datetime with:
        open, high, low, close
    """
    cfg = config or ForecasterConfig()
    _validate_ohlc(df)

    base = df.copy().sort_index()
    daily = _daily_ohlc(base)

    out = pd.DataFrame(index=base.index)

    buckets = _year_buckets(cfg)

    for y in buckets:
        out[f"fc_{y}y_samples"] = 0
        out[f"fc_{y}y_bull_prob"] = 0.5
        out[f"fc_{y}y_bear_prob"] = 0.5
        out[f"fc_{y}y_avg_return"] = 0.0
        out[f"fc_{y}y_direction"] = 0
        out[f"fc_{y}y_valid"] = 0

    out["forecast_today_weekday"] = base.index.weekday
    out["forecast_today_dayofyear"] = base.index.dayofyear

    # speed-safe evaluation window
    start_idx = 0
    if cfg.evaluate_last_n_rows > 0 and len(base) > cfg.evaluate_last_n_rows:
        start_idx = len(base) - cfg.evaluate_last_n_rows

    eval_index = base.index[start_idx:]

    for ts in eval_index:
        day_ts = pd.Timestamp(ts).normalize()

        if day_ts not in daily.index:
            continue

        for y in buckets:
            stats = _bucket_stats(daily=daily, current_date=day_ts, years_back=y, cfg=cfg)
            out.at[ts, f"fc_{y}y_samples"] = stats["samples"]
            out.at[ts, f"fc_{y}y_bull_prob"] = stats["bull_prob"]
            out.at[ts, f"fc_{y}y_bear_prob"] = stats["bear_prob"]
            out.at[ts, f"fc_{y}y_avg_return"] = stats["avg_return"]
            out.at[ts, f"fc_{y}y_direction"] = stats["direction"]
            out.at[ts, f"fc_{y}y_valid"] = stats["valid"]

    # Forward-fill intraday rows on the same date after evaluation
    fill_cols = []
    for y in buckets:
        fill_cols.extend(
            [
                f"fc_{y}y_samples",
                f"fc_{y}y_bull_prob",
                f"fc_{y}y_bear_prob",
                f"fc_{y}y_avg_return",
                f"fc_{y}y_direction",
                f"fc_{y}y_valid",
            ]
        )

    if fill_cols:
        out[fill_cols] = out[fill_cols].ffill().fillna(
            {
                **{f"fc_{y}y_samples": 0 for y in buckets},
                **{f"fc_{y}y_bull_prob": 0.5 for y in buckets},
                **{f"fc_{y}y_bear_prob": 0.5 for y in buckets},
                **{f"fc_{y}y_avg_return": 0.0 for y in buckets},
                **{f"fc_{y}y_direction": 0 for y in buckets},
                **{f"fc_{y}y_valid": 0 for y in buckets},
            }
        )

    if buckets:
        bull_cols = [f"fc_{y}y_bull_prob" for y in buckets]
        bear_cols = [f"fc_{y}y_bear_prob" for y in buckets]
        ret_cols = [f"fc_{y}y_avg_return" for y in buckets]
        dir_cols = [f"fc_{y}y_direction" for y in buckets]
        sample_cols = [f"fc_{y}y_samples" for y in buckets]
        valid_cols = [f"fc_{y}y_valid" for y in buckets]

        out["forecast_bull_prob"] = out[bull_cols].mean(axis=1)
        out["forecast_bear_prob"] = out[bear_cols].mean(axis=1)
        out["forecast_avg_return"] = out[ret_cols].mean(axis=1)
        out["forecast_sample_count"] = out[sample_cols].sum(axis=1)
        out["forecast_valid_bucket_count"] = out[valid_cols].sum(axis=1)

        out["forecast_direction"] = np.where(
            (out["forecast_bull_prob"] >= cfg.bull_bias_threshold)
            & (out["forecast_bull_prob"] > out["forecast_bear_prob"]),
            1,
            np.where(
                (out["forecast_bear_prob"] >= cfg.bear_bias_threshold)
                & (out["forecast_bear_prob"] > out["forecast_bull_prob"]),
                -1,
                0,
            ),
        ).astype(int)

        valid_dir_frame = out[dir_cols].copy()
        bullish_agreement = (valid_dir_frame == 1).sum(axis=1)
        bearish_agreement = (valid_dir_frame == -1).sum(axis=1)
        total_valid_dirs = ((valid_dir_frame == 1) | (valid_dir_frame == -1)).sum(axis=1)

        out["forecast_bullish_agreement"] = np.where(
            total_valid_dirs > 0, bullish_agreement / total_valid_dirs, 0.0
        )
        out["forecast_bearish_agreement"] = np.where(
            total_valid_dirs > 0, bearish_agreement / total_valid_dirs, 0.0
        )
        out["forecast_year_agreement"] = np.where(
            out["forecast_direction"] == 1,
            out["forecast_bullish_agreement"],
            np.where(
                out["forecast_direction"] == -1,
                out["forecast_bearish_agreement"],
                0.0,
            ),
        )

        prob_edge = (out["forecast_bull_prob"] - out["forecast_bear_prob"]).abs()
        sample_strength = np.clip(out["forecast_sample_count"] / 20.0, 0.0, 1.0)

        out["forecast_confidence"] = (
            prob_edge * 0.45
            + out["forecast_year_agreement"] * 0.35
            + sample_strength * 0.20
        ).clip(0.0, 1.0)

        out["forecast_strong_bull_day"] = (
            (out["forecast_direction"] == 1)
            & (out["forecast_year_agreement"] >= cfg.strong_agreement_threshold)
        ).astype(int)

        out["forecast_strong_bear_day"] = (
            (out["forecast_direction"] == -1)
            & (out["forecast_year_agreement"] >= cfg.strong_agreement_threshold)
        ).astype(int)

    else:
        out["forecast_bull_prob"] = 0.5
        out["forecast_bear_prob"] = 0.5
        out["forecast_avg_return"] = 0.0
        out["forecast_sample_count"] = 0
        out["forecast_valid_bucket_count"] = 0
        out["forecast_direction"] = 0
        out["forecast_bullish_agreement"] = 0.0
        out["forecast_bearish_agreement"] = 0.0
        out["forecast_year_agreement"] = 0.0
        out["forecast_confidence"] = 0.0
        out["forecast_strong_bull_day"] = 0
        out["forecast_strong_bear_day"] = 0

    # -------------------------------------------------------------------------
    # SMARTCHART OUTPUT CONTRACT
    # -------------------------------------------------------------------------
    out["forecast_direction_export"] = out["forecast_direction"].astype(int)
    out["forecast_bull_prob_export"] = out["forecast_bull_prob"].astype(float)
    out["forecast_bear_prob_export"] = out["forecast_bear_prob"].astype(float)
    out["forecast_avg_return_export"] = out["forecast_avg_return"].astype(float)
    out["forecast_sample_count_export"] = out["forecast_sample_count"].astype(int)
    out["forecast_valid_bucket_count_export"] = out["forecast_valid_bucket_count"].astype(int)
    out["forecast_year_agreement_export"] = out["forecast_year_agreement"].astype(float)
    out["forecast_confidence_export"] = out["forecast_confidence"].astype(float)
    out["forecast_strong_bull_day_export"] = out["forecast_strong_bull_day"].astype(int)
    out["forecast_strong_bear_day_export"] = out["forecast_strong_bear_day"].astype(int)

    for y in buckets:
        out[f"fc_{y}y_samples_export"] = out[f"fc_{y}y_samples"].astype(int)
        out[f"fc_{y}y_bull_prob_export"] = out[f"fc_{y}y_bull_prob"].astype(float)
        out[f"fc_{y}y_bear_prob_export"] = out[f"fc_{y}y_bear_prob"].astype(float)
        out[f"fc_{y}y_avg_return_export"] = out[f"fc_{y}y_avg_return"].astype(float)
        out[f"fc_{y}y_direction_export"] = out[f"fc_{y}y_direction"].astype(int)
        out[f"fc_{y}y_valid_export"] = out[f"fc_{y}y_valid"].astype(int)

    out["forecast_signal"] = (out["forecast_direction"] != 0).astype(int)
    out["forecast_strength"] = out["forecast_confidence"].astype(float)

    # -------------------------------------------------------------------------
    # TRUTH ENGINE CONTRACT
    # -------------------------------------------------------------------------
    out["forecast_dir"] = out["forecast_direction"].fillna(0).astype(int)
    out["forecast_confidence"] = out["forecast_confidence"].fillna(0.0).astype(float)
    out["forecast_agreement"] = out["forecast_year_agreement"].fillna(0.0).astype(float)
    out["forecast_long_pct"] = (out["forecast_bull_prob"].fillna(0.5) * 100.0).astype(float)
    out["forecast_short_pct"] = (out["forecast_bear_prob"].fillna(0.5) * 100.0).astype(float)

    return out


def build_forecaster(
    df: pd.DataFrame,
    config: Optional[ForecasterConfig] = None,
) -> pd.DataFrame:
    return calculate_forecaster(df, config=config)


def run_forecaster(
    df: pd.DataFrame,
    config: Optional[ForecasterConfig] = None,
) -> pd.DataFrame:
    return calculate_forecaster(df, config=config)


# =============================================================================
# DIRECT TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    rng = pd.date_range("2016-01-01", periods=4000, freq="4h")
    np.random.seed(42)

    base_walk = np.random.normal(0, 1.0, len(rng)).cumsum()
    weekday_bias = np.array([0.30 if d == 0 else -0.20 if d == 3 else 0.0 for d in rng.weekday])
    seasonal_bias = np.sin((rng.dayofyear / 365.0) * 2.0 * np.pi) * 2.0

    close = pd.Series(1800 + base_walk + np.cumsum(weekday_bias * 0.05) + seasonal_bias, index=rng)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.random.uniform(0.2, 2.0, len(rng))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.random.uniform(0.2, 2.0, len(rng))

    test_df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        },
        index=rng,
    )

    result = calculate_forecaster(test_df)

    cols = [
        "forecast_dir",
        "forecast_confidence",
        "forecast_agreement",
        "forecast_long_pct",
        "forecast_short_pct",
        "forecast_signal",
        "forecast_strength",
    ]

    print("SmartChart Forecaster — direct test")
    print(result[cols].tail(20))