"""
SmartChart Backend — n_volatility.py

Volatility Engine v1
Backend rebuild from Pine logic.

Core features:
- Volatility regime detection:
    0 = none
    1 = compression
    2 = normal
    3 = expansion
- ATR % vs baseline volatility model
- Optional EMA spread filter
- VP-Lite context:
    - POC price
    - POC distance %
    - value area high / low
    - in-value-area state
- MTF volatility average support layer
- Final export-ready outputs
- Direct test block included

Backend only:
- no TradingView visuals
- no labels / plots / drawing objects
- clean standalone output contract
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

import math
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class VolatilityConfig:
    # Core volatility
    vol_on: bool = True
    vol_atr_len: int = 20
    vol_base_len: int = 20
    vol_compression_mult: float = 0.85
    vol_expansion_mult: float = 1.20
    vol_use_band_filter: bool = True
    vol_band_cmp_max_pct: float = 0.08
    vol_band_exp_min_pct: float = 0.18

    # EMA structure support
    ema_fast_len: int = 20
    ema_slow_len: int = 200

    # VP Lite
    vp_on: bool = True
    vp_look: int = 100
    vp_bins: int = 50
    vp_va_levels: int = 5

    # MTF
    mtf_on: bool = True
    mtf_weights: Optional[Dict[str, float]] = None
    mtf_rules: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.mtf_weights is None:
            self.mtf_weights = {
                "tf1": 1.0,
                "tf2": 1.0,
                "tf3": 1.0,
                "tf4": 1.0,
            }

        if self.mtf_rules is None:
            self.mtf_rules = {
                "tf1": "15min",
                "tf2": "1h",
                "tf3": "4h",
                "tf4": "1d",
            }


# ==============================================================================
# HELPERS
# ==============================================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.rolling(length, min_periods=1).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(tr, length)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Resampling requires DatetimeIndex.")

    out = pd.DataFrame(
        {
            "open": df["open"].resample(rule).first(),
            "high": df["high"].resample(rule).max(),
            "low": df["low"].resample(rule).min(),
            "close": df["close"].resample(rule).last(),
            "volume": df["volume"].resample(rule).sum(),
        }
    ).dropna()

    return out


def safe_pct_diff(a: pd.Series, b: pd.Series, base: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=base.index, dtype=float)
    valid = base != 0
    out.loc[valid] = ((a.loc[valid] - b.loc[valid]) / base.loc[valid]) * 100.0
    return out


def clip_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


# ==============================================================================
# OUTPUT CONTRACT
# ==============================================================================

@dataclass
class VolatilityOutput:
    sc_vol_regime: int
    sc_vol_score: float
    sc_vol_compression: bool
    sc_vol_normal: bool
    sc_vol_expansion: bool

    sc_vol_mtf_avg: float
    sc_vol_mtf_agreement: int

    atr_pct_now: Optional[float]
    atr_pct_mean: Optional[float]

    ema_band_spread_pct: Optional[float]
    abs_band_spread_pct: Optional[float]
    band_compression_ok: bool
    band_expansion_ok: bool

    sc_poc_price: Optional[float]
    sc_poc_dist_pct: Optional[float]
    sc_in_value_area: bool
    sc_va_hi: Optional[float]
    sc_va_lo: Optional[float]

    timestamp: Any


# ==============================================================================
# CORE ENGINE
# ==============================================================================

class VolatilityEngine:
    def __init__(self, config: Optional[VolatilityConfig] = None) -> None:
        self.config = config or VolatilityConfig()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("VolatilityEngine requires a DatetimeIndex.")

        c = self.config
        out = df.copy()

        # ----------------------------------------------------------------------
        # Core EMA support
        # ----------------------------------------------------------------------
        out["ema_fast"] = ema(out["close"], c.ema_fast_len)
        out["ema_slow"] = ema(out["close"], c.ema_slow_len)

        out["ema_band_spread_pct"] = safe_pct_diff(
            out["ema_fast"], out["ema_slow"], out["close"]
        )
        out["abs_band_spread_pct"] = out["ema_band_spread_pct"].abs()

        out["e_fast_slow_near"] = (
            out["abs_band_spread_pct"].notna()
            & (out["abs_band_spread_pct"] <= c.vol_band_cmp_max_pct)
        )
        out["e_fast_slow_expand"] = (
            out["abs_band_spread_pct"].notna()
            & (out["abs_band_spread_pct"] >= c.vol_band_exp_min_pct)
        )

        # ----------------------------------------------------------------------
        # Volatility engine
        # ----------------------------------------------------------------------
        out["atr_now"] = atr(out, c.vol_atr_len)
        out["atr_pct_now"] = pd.Series(np.nan, index=out.index, dtype=float)
        valid_close = out["close"] != 0
        out.loc[valid_close, "atr_pct_now"] = (
            out.loc[valid_close, "atr_now"] / out.loc[valid_close, "close"]
        ) * 100.0

        out["atr_pct_mean"] = sma(out["atr_pct_now"], c.vol_base_len)

        if c.vol_on:
            out["vol_has"] = out["atr_pct_now"].notna() & out["atr_pct_mean"].notna()
        else:
            out["vol_has"] = False

        out["raw_compression"] = (
            out["vol_has"]
            & (out["atr_pct_now"] <= out["atr_pct_mean"] * c.vol_compression_mult)
        )
        out["raw_expansion"] = (
            out["vol_has"]
            & (out["atr_pct_now"] >= out["atr_pct_mean"] * c.vol_expansion_mult)
        )

        out["band_compression_ok"] = (~pd.Series(c.vol_use_band_filter, index=out.index)) | out["e_fast_slow_near"]
        out["band_expansion_ok"] = (~pd.Series(c.vol_use_band_filter, index=out.index)) | out["e_fast_slow_expand"]

        # simpler / clearer boolean logic
        if c.vol_use_band_filter:
            out["band_compression_ok"] = out["e_fast_slow_near"]
            out["band_expansion_ok"] = out["e_fast_slow_expand"]
        else:
            out["band_compression_ok"] = True
            out["band_expansion_ok"] = True

        out["vol_compression"] = out["raw_compression"] & out["band_compression_ok"]
        out["vol_expansion"] = out["raw_expansion"] & out["band_expansion_ok"]

        if c.vol_on:
            out["vol_regime"] = np.select(
                [
                    out["vol_compression"],
                    out["vol_expansion"],
                    out["vol_has"],
                ],
                [1, 3, 2],
                default=0,
            ).astype(int)
        else:
            out["vol_regime"] = 0

        out["vol_score_raw"] = pd.Series(np.nan, index=out.index, dtype=float)
        valid_mean = out["vol_has"] & (out["atr_pct_mean"] != 0)
        out.loc[valid_mean, "vol_score_raw"] = (
            out.loc[valid_mean, "atr_pct_now"] / out.loc[valid_mean, "atr_pct_mean"]
        ) - 1.0

        out["vol_score"] = clip_series(out["vol_score_raw"], -1.0, 1.0).fillna(0.0)

        # ----------------------------------------------------------------------
        # VP-Lite
        # ----------------------------------------------------------------------
        poc_prices: List[float] = []
        poc_dists: List[float] = []
        in_value_areas: List[bool] = []
        va_his: List[float] = []
        va_los: List[float] = []

        for i in range(len(out)):
            if not c.vp_on:
                poc_prices.append(np.nan)
                poc_dists.append(np.nan)
                in_value_areas.append(False)
                va_his.append(np.nan)
                va_los.append(np.nan)
                continue

            start = max(0, i - c.vp_look + 1)
            window = out.iloc[start : i + 1]

            if len(window) < max(10, c.vp_bins):
                poc_prices.append(np.nan)
                poc_dists.append(np.nan)
                in_value_areas.append(False)
                va_his.append(np.nan)
                va_los.append(np.nan)
                continue

            max_p = float(window["high"].max())
            min_p = float(window["low"].min())
            rng_vp = max_p - min_p

            if c.vp_bins <= 0 or rng_vp <= 0:
                poc_prices.append(np.nan)
                poc_dists.append(np.nan)
                in_value_areas.append(False)
                va_his.append(np.nan)
                va_los.append(np.nan)
                continue

            bin_size = rng_vp / float(c.vp_bins)
            if not np.isfinite(bin_size) or bin_size <= 0:
                poc_prices.append(np.nan)
                poc_dists.append(np.nan)
                in_value_areas.append(False)
                va_his.append(np.nan)
                va_los.append(np.nan)
                continue

            bins = np.zeros(c.vp_bins, dtype=float)
            hlc3 = ((window["high"] + window["low"] + window["close"]) / 3.0).to_numpy(dtype=float)
            vols = window["volume"].to_numpy(dtype=float)

            for px, vv in zip(hlc3, vols):
                idx = int((px - min_p) / bin_size)
                idx = max(0, min(c.vp_bins - 1, idx))
                bins[idx] += vv

            poc_idx = int(np.argmax(bins))
            poc_price = min_p + (poc_idx + 0.5) * bin_size

            close_i = float(out["close"].iat[i])
            poc_dist = ((close_i - poc_price) / close_i) * 100.0 if close_i != 0 else np.nan

            va_hi = poc_price + bin_size * c.vp_va_levels
            va_lo = poc_price - bin_size * c.vp_va_levels
            in_va = bool(va_lo <= close_i <= va_hi)

            poc_prices.append(poc_price)
            poc_dists.append(poc_dist)
            in_value_areas.append(in_va)
            va_his.append(va_hi)
            va_los.append(va_lo)

        out["poc_price"] = pd.Series(poc_prices, index=out.index, dtype=float)
        out["poc_dist"] = pd.Series(poc_dists, index=out.index, dtype=float)
        out["in_value_area"] = pd.Series(in_value_areas, index=out.index, dtype=bool)
        out["va_hi"] = pd.Series(va_his, index=out.index, dtype=float)
        out["va_lo"] = pd.Series(va_los, index=out.index, dtype=float)

        # ----------------------------------------------------------------------
        # MTF volatility average
        # ----------------------------------------------------------------------
        out["mtf_avg_vol"] = 0.0
        out["mtf_agreement"] = 0

        if c.mtf_on:
            mtf_df = self._calculate_mtf_average(df)
            out["mtf_avg_vol"] = mtf_df["mtf_avg_vol"].reindex(out.index).fillna(0.0)

        out["mtf_agreement"] = np.select(
            [
                (out["vol_regime"] == 1) & (out["mtf_avg_vol"] <= -0.50),
                (out["vol_regime"] == 3) & (out["mtf_avg_vol"] >= 0.50),
                (out["vol_regime"] == 2) & (out["mtf_avg_vol"].abs() < 0.50),
            ],
            [1, 1, 1],
            default=0,
        ).astype(int)

        # ----------------------------------------------------------------------
        # Export-ready fields
        # ----------------------------------------------------------------------
        out["sc_vol_regime"] = out["vol_regime"]
        out["sc_vol_score"] = out["vol_score"]
        out["sc_vol_compression"] = out["vol_regime"] == 1
        out["sc_vol_normal"] = out["vol_regime"] == 2
        out["sc_vol_expansion"] = out["vol_regime"] == 3

        out["sc_vol_mtf_avg"] = out["mtf_avg_vol"]
        out["sc_vol_mtf_agreement"] = out["mtf_agreement"]

        out["sc_poc_price"] = out["poc_price"]
        out["sc_poc_dist_pct"] = out["poc_dist"]
        out["sc_in_value_area"] = out["in_value_area"]
        out["sc_va_hi"] = out["va_hi"]
        out["sc_va_lo"] = out["va_lo"]

        return out

    def _compute_vol_state_series(self, df: pd.DataFrame) -> pd.Series:
        c = self.config

        ema_fast_s = ema(df["close"], c.ema_fast_len)
        ema_slow_s = ema(df["close"], c.ema_slow_len)

        ema_band_spread_pct = safe_pct_diff(ema_fast_s, ema_slow_s, df["close"])
        abs_band_spread_pct = ema_band_spread_pct.abs()

        e_fast_slow_near = (
            abs_band_spread_pct.notna()
            & (abs_band_spread_pct <= c.vol_band_cmp_max_pct)
        )
        e_fast_slow_expand = (
            abs_band_spread_pct.notna()
            & (abs_band_spread_pct >= c.vol_band_exp_min_pct)
        )

        atr_now = atr(df, c.vol_atr_len)
        atr_pct = pd.Series(np.nan, index=df.index, dtype=float)
        valid_close = df["close"] != 0
        atr_pct.loc[valid_close] = (atr_now.loc[valid_close] / df["close"].loc[valid_close]) * 100.0

        atr_mean = sma(atr_pct, c.vol_base_len)
        vol_has = atr_pct.notna() & atr_mean.notna()

        raw_cmp = vol_has & (atr_pct <= atr_mean * c.vol_compression_mult)
        raw_exp = vol_has & (atr_pct >= atr_mean * c.vol_expansion_mult)

        if c.vol_use_band_filter:
            band_compression_ok = e_fast_slow_near
            band_expansion_ok = e_fast_slow_expand
        else:
            band_compression_ok = pd.Series(True, index=df.index)
            band_expansion_ok = pd.Series(True, index=df.index)

        vol_compression = raw_cmp & band_compression_ok
        vol_expansion = raw_exp & band_expansion_ok

        state = np.select(
            [vol_compression, vol_expansion, vol_has],
            [-1.0, 1.0, 0.0],
            default=0.0,
        )

        return pd.Series(state, index=df.index, dtype=float)

    def _calculate_mtf_average(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        weights = c.mtf_weights or {}
        rules = c.mtf_rules or {}
        total_weight = max(sum(float(v) for v in weights.values()), 0.0001)

        parts: List[pd.Series] = []

        for tf_key, rule in rules.items():
            weight = float(weights.get(tf_key, 0.0))

            if weight <= 0:
                parts.append(pd.Series(0.0, index=df.index, dtype=float))
                continue

            try:
                rs = resample_ohlcv(df, rule)
                if rs.empty:
                    parts.append(pd.Series(0.0, index=df.index, dtype=float))
                    continue

                state_series = self._compute_vol_state_series(rs)
                aligned = state_series.reindex(df.index, method="ffill").fillna(0.0)
                parts.append(aligned * weight)
            except Exception:
                parts.append(pd.Series(0.0, index=df.index, dtype=float))

        mtf_avg = sum(parts) / total_weight
        return pd.DataFrame({"mtf_avg_vol": mtf_avg}, index=df.index)

    def latest(self, df: pd.DataFrame) -> VolatilityOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        return VolatilityOutput(
            sc_vol_regime=int(row["sc_vol_regime"]),
            sc_vol_score=float(row["sc_vol_score"]),
            sc_vol_compression=bool(row["sc_vol_compression"]),
            sc_vol_normal=bool(row["sc_vol_normal"]),
            sc_vol_expansion=bool(row["sc_vol_expansion"]),

            sc_vol_mtf_avg=float(row["sc_vol_mtf_avg"]),
            sc_vol_mtf_agreement=int(row["sc_vol_mtf_agreement"]),

            atr_pct_now=None if pd.isna(row["atr_pct_now"]) else float(row["atr_pct_now"]),
            atr_pct_mean=None if pd.isna(row["atr_pct_mean"]) else float(row["atr_pct_mean"]),

            ema_band_spread_pct=None if pd.isna(row["ema_band_spread_pct"]) else float(row["ema_band_spread_pct"]),
            abs_band_spread_pct=None if pd.isna(row["abs_band_spread_pct"]) else float(row["abs_band_spread_pct"]),
            band_compression_ok=bool(row["band_compression_ok"]),
            band_expansion_ok=bool(row["band_expansion_ok"]),

            sc_poc_price=None if pd.isna(row["sc_poc_price"]) else float(row["sc_poc_price"]),
            sc_poc_dist_pct=None if pd.isna(row["sc_poc_dist_pct"]) else float(row["sc_poc_dist_pct"]),
            sc_in_value_area=bool(row["sc_in_value_area"]),
            sc_va_hi=None if pd.isna(row["sc_va_hi"]) else float(row["sc_va_hi"]),
            sc_va_lo=None if pd.isna(row["sc_va_lo"]) else float(row["sc_va_lo"]),

            timestamp=calc.index[-1],
        )


# ==============================================================================
# PUBLIC API
# ==============================================================================

def run_volatility_engine(
    df: pd.DataFrame,
    config: Optional[VolatilityConfig] = None,
) -> Dict[str, Any]:
    engine = VolatilityEngine(config=config)
    latest = engine.latest(df)
    return asdict(latest)


def run_volatility_engine_full(
    df: pd.DataFrame,
    config: Optional[VolatilityConfig] = None,
) -> pd.DataFrame:
    engine = VolatilityEngine(config=config)
    return engine.calculate(df)


# ==============================================================================
# DIRECT TEST BLOCK
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    periods = 1400
    idx = pd.date_range("2026-01-01", periods=periods, freq="5min")

    base = 100 + np.cumsum(np.random.normal(0, 0.10, periods))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])

    high = np.maximum(open_, close) + np.random.uniform(0.02, 0.20, periods)
    low = np.minimum(open_, close) - np.random.uniform(0.02, 0.20, periods)
    volume = np.random.randint(100, 1400, periods).astype(float)

    # compression windows
    for k in [220, 560, 980]:
        if k + 40 < periods:
            high.iloc[k:k + 40] = close.iloc[k:k + 40] + np.random.uniform(0.01, 0.04, 40)
            low.iloc[k:k + 40] = close.iloc[k:k + 40] - np.random.uniform(0.01, 0.04, 40)

    # expansion windows
    for k in [120, 430, 760, 1180]:
        if k + 20 < periods:
            high.iloc[k:k + 20] = np.maximum(
                high.iloc[k:k + 20],
                close.iloc[k:k + 20] + np.random.uniform(0.25, 0.60, 20),
            )
            low.iloc[k:k + 20] = np.minimum(
                low.iloc[k:k + 20],
                close.iloc[k:k + 20] - np.random.uniform(0.25, 0.60, 20),
            )

    test_df = pd.DataFrame(
        {
            "open": open_.values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": volume,
        },
        index=idx,
    )

    config = VolatilityConfig(
        vol_on=True,
        vol_atr_len=20,
        vol_base_len=20,
        vol_compression_mult=0.85,
        vol_expansion_mult=1.20,
        vol_use_band_filter=True,
        vol_band_cmp_max_pct=0.08,
        vol_band_exp_min_pct=0.18,
        ema_fast_len=20,
        ema_slow_len=200,
        vp_on=True,
        vp_look=100,
        vp_bins=50,
        vp_va_levels=5,
        mtf_on=True,
        mtf_weights={"tf1": 1.0, "tf2": 1.0, "tf3": 1.0, "tf4": 1.0},
        mtf_rules={"tf1": "15min", "tf2": "1h", "tf3": "4h", "tf4": "1d"},
    )

    engine = VolatilityEngine(config=config)
    full = engine.calculate(test_df)
    latest = engine.latest(test_df)

    print("\n=== SMARTCHART VOLATILITY ENGINE TEST ===")
    print(
        full[
            [
                "sc_vol_regime",
                "sc_vol_score",
                "sc_vol_mtf_avg",
                "sc_vol_mtf_agreement",
                "atr_pct_now",
                "atr_pct_mean",
                "ema_band_spread_pct",
                "abs_band_spread_pct",
                "sc_poc_dist_pct",
                "sc_in_value_area",
            ]
        ].tail(15)
    )

    print("\n=== LATEST OUTPUT ===")
    print(asdict(latest))