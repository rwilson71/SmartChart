from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class VolatilityConfig:
    # Core
    vol_on: bool = True
    vol_atr_len: int = 20
    vol_base_len: int = 20
    vol_compression_mult: float = 0.85
    vol_expansion_mult: float = 1.20

    # Band filter
    vol_use_band_filter: bool = True
    vol_band_cmp_max_pct: float = 0.08
    vol_band_exp_min_pct: float = 0.18
    ema_fast_len: int = 20
    ema_slow_len: int = 200

    # VP Lite
    vp_on: bool = True
    vp_look: int = 100
    vp_bins: int = 50
    vp_va_levels: int = 5

    # MTF
    mtf_on: bool = True
    tf1: str = "15min"
    tf2: str = "1h"
    tf3: str = "4h"
    tf4: str = "1d"
    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0
    w4: float = 1.0


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
    return series.rolling(window=length, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return rma(true_range(df), length)


def safe_pct_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator != 0
    out.loc[valid] = (numerator.loc[valid] / denominator.loc[valid]) * 100.0
    return out


def safe_pct_diff(a: pd.Series, b: pd.Series, base: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=base.index, dtype=float)
    valid = base != 0
    out.loc[valid] = ((a.loc[valid] - b.loc[valid]) / base.loc[valid]) * 100.0
    return out


def clip_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


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


def regime_text(v: int) -> str:
    if v == 1:
        return "COMP"
    if v == 2:
        return "NORMAL"
    if v == 3:
        return "EXPAND"
    return "NONE"


def derive_bias_signal(regime: int) -> int:
    if regime == 3:
        return 1
    if regime == 1:
        return -1
    return 0


def derive_bias_label(signal: int) -> str:
    if signal > 0:
        return "BULLISH"
    if signal < 0:
        return "BEARISH"
    return "NEUTRAL"


def derive_market_bias(signal: int) -> str:
    if signal > 0:
        return "BULLISH"
    if signal < 0:
        return "BEARISH"
    return "NEUTRAL"


def derive_indicator_strength(vol_score: float) -> float:
    strength = abs(float(vol_score)) * 100.0
    return float(np.clip(strength, 0.0, 100.0))


# ==============================================================================
# OUTPUT CONTRACT
# ==============================================================================

@dataclass
class VolatilityOutput:
    # Standard website contract
    timestamp: Any
    state: str
    bias_signal: int
    bias_label: str
    indicator_strength: float
    market_bias: str

    # Core volatility outputs
    sc_vol_regime: int
    sc_vol_regime_text: str
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

    vp_on: bool
    vol_on: bool
    mtf_on: bool


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

        # ------------------------------------------------------------------
        # EMA band layer
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Core volatility layer
        # ------------------------------------------------------------------
        out["atr_now"] = atr(out, c.vol_atr_len)
        out["atr_pct_now"] = safe_pct_ratio(out["atr_now"], out["close"])
        out["atr_pct_mean"] = sma(out["atr_pct_now"], c.vol_base_len)

        out["vol_has"] = (
            bool(c.vol_on)
            & out["atr_pct_now"].notna()
            & out["atr_pct_mean"].notna()
        )

        out["raw_compression"] = (
            out["vol_has"]
            & (out["atr_pct_now"] <= out["atr_pct_mean"] * c.vol_compression_mult)
        )
        out["raw_expansion"] = (
            out["vol_has"]
            & (out["atr_pct_now"] >= out["atr_pct_mean"] * c.vol_expansion_mult)
        )

        if c.vol_use_band_filter:
            out["band_compression_ok"] = out["e_fast_slow_near"]
            out["band_expansion_ok"] = out["e_fast_slow_expand"]
        else:
            out["band_compression_ok"] = True
            out["band_expansion_ok"] = True

        out["vol_compression"] = out["raw_compression"] & out["band_compression_ok"]
        out["vol_expansion"] = out["raw_expansion"] & out["band_expansion_ok"]

        out["vol_regime"] = np.select(
            [
                ~pd.Series(bool(c.vol_on), index=out.index),
                out["vol_compression"],
                out["vol_expansion"],
                out["vol_has"],
            ],
            [0, 1, 3, 2],
            default=0,
        ).astype(int)

        out["vol_score_raw"] = pd.Series(np.nan, index=out.index, dtype=float)
        valid_mean = out["vol_has"] & (out["atr_pct_mean"] != 0)
        out.loc[valid_mean, "vol_score_raw"] = (
            out.loc[valid_mean, "atr_pct_now"] / out.loc[valid_mean, "atr_pct_mean"]
        ) - 1.0

        out["vol_score"] = clip_series(out["vol_score_raw"].fillna(0.0), -1.0, 1.0)

        # ------------------------------------------------------------------
        # VP Lite - Pine parity scan
        # ------------------------------------------------------------------
        poc_price_list: List[float] = []
        poc_dist_pct_list: List[float] = []
        va_hi_list: List[float] = []
        va_lo_list: List[float] = []
        in_value_area_list: List[bool] = []

        min_ready_bars = max(10, int(c.vp_bins))

        hlc3_series = (out["high"] + out["low"] + out["close"]) / 3.0

        for i in range(len(out)):
            if not c.vp_on or i < min_ready_bars:
                poc_price_list.append(np.nan)
                poc_dist_pct_list.append(np.nan)
                va_hi_list.append(np.nan)
                va_lo_list.append(np.nan)
                in_value_area_list.append(False)
                continue

            start = max(0, i - c.vp_look + 1)
            window = out.iloc[start : i + 1]
            window_hlc3 = hlc3_series.iloc[start : i + 1]

            hh = float(window["high"].max())
            ll = float(window["low"].min())
            rng = hh - ll

            if c.vp_bins <= 0 or rng <= 0:
                poc_price_list.append(np.nan)
                poc_dist_pct_list.append(np.nan)
                va_hi_list.append(np.nan)
                va_lo_list.append(np.nan)
                in_value_area_list.append(False)
                continue

            bin_size = rng / float(c.vp_bins)
            if not np.isfinite(bin_size) or bin_size <= 0:
                poc_price_list.append(np.nan)
                poc_dist_pct_list.append(np.nan)
                va_hi_list.append(np.nan)
                va_lo_list.append(np.nan)
                in_value_area_list.append(False)
                continue

            best_vol = np.nan
            best_price = np.nan

            for b in range(c.vp_bins):
                lo_bin = ll + b * bin_size
                hi_bin = lo_bin + bin_size
                bucket_vol = 0.0

                for px, vv in zip(
                    window_hlc3.to_numpy(dtype=float),
                    window["volume"].to_numpy(dtype=float),
                ):
                    if b == c.vp_bins - 1:
                        in_bin = (px >= lo_bin) and (px <= hi_bin)
                    else:
                        in_bin = (px >= lo_bin) and (px < hi_bin)

                    if in_bin:
                        bucket_vol += float(vv)

                if np.isnan(best_vol) or bucket_vol > best_vol:
                    best_vol = bucket_vol
                    best_price = lo_bin + bin_size * 0.5

            close_i = float(out["close"].iat[i])

            poc_price = best_price if np.isfinite(best_price) else np.nan
            poc_dist_pct = (
                ((close_i - poc_price) / close_i) * 100.0
                if close_i != 0 and np.isfinite(poc_price)
                else np.nan
            )
            va_hi = (
                poc_price + bin_size * c.vp_va_levels
                if np.isfinite(poc_price)
                else np.nan
            )
            va_lo = (
                poc_price - bin_size * c.vp_va_levels
                if np.isfinite(poc_price)
                else np.nan
            )
            in_value_area = (
                bool(close_i >= va_lo and close_i <= va_hi)
                if np.isfinite(va_hi) and np.isfinite(va_lo)
                else False
            )

            poc_price_list.append(poc_price)
            poc_dist_pct_list.append(poc_dist_pct)
            va_hi_list.append(va_hi)
            va_lo_list.append(va_lo)
            in_value_area_list.append(in_value_area)

        out["poc_price"] = pd.Series(poc_price_list, index=out.index, dtype=float)
        out["poc_dist_pct"] = pd.Series(poc_dist_pct_list, index=out.index, dtype=float)
        out["va_hi"] = pd.Series(va_hi_list, index=out.index, dtype=float)
        out["va_lo"] = pd.Series(va_lo_list, index=out.index, dtype=float)
        out["in_value_area"] = pd.Series(in_value_area_list, index=out.index, dtype=bool)

        # ------------------------------------------------------------------
        # MTF
        # ------------------------------------------------------------------
        out["tf1_vol"] = 0.0
        out["tf2_vol"] = 0.0
        out["tf3_vol"] = 0.0
        out["tf4_vol"] = 0.0

        total_weight = max(c.w1 + c.w2 + c.w3 + c.w4, 0.0001)

        if c.mtf_on:
            if c.w1 > 0:
                out["tf1_vol"] = self._mtf_state_aligned(df, c.tf1)
            if c.w2 > 0:
                out["tf2_vol"] = self._mtf_state_aligned(df, c.tf2)
            if c.w3 > 0:
                out["tf3_vol"] = self._mtf_state_aligned(df, c.tf3)
            if c.w4 > 0:
                out["tf4_vol"] = self._mtf_state_aligned(df, c.tf4)

        out["mtf_avg_vol"] = (
            out["tf1_vol"] * c.w1
            + out["tf2_vol"] * c.w2
            + out["tf3_vol"] * c.w3
            + out["tf4_vol"] * c.w4
        ) / total_weight

        out["mtf_agreement"] = np.select(
            [
                (out["vol_regime"] == 1) & (out["mtf_avg_vol"] <= -0.50),
                (out["vol_regime"] == 3) & (out["mtf_avg_vol"] >= 0.50),
                (out["vol_regime"] == 2) & (out["mtf_avg_vol"].abs() < 0.50),
            ],
            [1, 1, 1],
            default=0,
        ).astype(int)

        # ------------------------------------------------------------------
        # Export
        # ------------------------------------------------------------------
        out["sc_vol_regime"] = out["vol_regime"]
        out["sc_vol_regime_text"] = out["sc_vol_regime"].map(regime_text)
        out["sc_vol_score"] = out["vol_score"]

        out["sc_vol_compression"] = out["vol_regime"] == 1
        out["sc_vol_normal"] = out["vol_regime"] == 2
        out["sc_vol_expansion"] = out["vol_regime"] == 3

        out["sc_vol_mtf_avg"] = out["mtf_avg_vol"]
        out["sc_vol_mtf_agreement"] = out["mtf_agreement"]

        out["sc_poc_price"] = out["poc_price"]
        out["sc_poc_dist_pct"] = out["poc_dist_pct"]
        out["sc_in_value_area"] = out["in_value_area"]
        out["sc_va_hi"] = out["va_hi"]
        out["sc_va_lo"] = out["va_lo"]

        # ------------------------------------------------------------------
        # Standard website contract
        # ------------------------------------------------------------------
        out["state"] = out["sc_vol_regime_text"]
        out["bias_signal"] = out["sc_vol_regime"].apply(derive_bias_signal)
        out["bias_label"] = out["bias_signal"].apply(derive_bias_label)
        out["market_bias"] = out["bias_signal"].apply(derive_market_bias)
        out["indicator_strength"] = out["sc_vol_score"].apply(derive_indicator_strength)

        return out

    def _vol_state_score_series(self, df: pd.DataFrame) -> pd.Series:
        c = self.config

        atr_now = atr(df, c.vol_atr_len)
        atr_pct = safe_pct_ratio(atr_now, df["close"])
        atr_mean = sma(atr_pct, c.vol_base_len)

        vol_has = atr_pct.notna() & atr_mean.notna()
        raw_cmp = vol_has & (atr_pct <= atr_mean * c.vol_compression_mult)
        raw_exp = vol_has & (atr_pct >= atr_mean * c.vol_expansion_mult)

        if c.vol_use_band_filter:
            ema_fast_s = ema(df["close"], c.ema_fast_len)
            ema_slow_s = ema(df["close"], c.ema_slow_len)
            band_spread_pct = safe_pct_diff(
                ema_fast_s, ema_slow_s, df["close"]
            ).abs()

            band_cmp_ok = band_spread_pct.notna() & (
                band_spread_pct <= c.vol_band_cmp_max_pct
            )
            band_exp_ok = band_spread_pct.notna() & (
                band_spread_pct >= c.vol_band_exp_min_pct
            )

            raw_cmp = raw_cmp & band_cmp_ok
            raw_exp = raw_exp & band_exp_ok

        state = np.select(
            [raw_cmp, raw_exp, vol_has],
            [-1.0, 1.0, 0.0],
            default=0.0,
        )
        return pd.Series(state, index=df.index, dtype=float)

    def _mtf_state_aligned(self, df: pd.DataFrame, rule: str) -> pd.Series:
        try:
            rs = resample_ohlcv(df, rule)
            if rs.empty:
                return pd.Series(0.0, index=df.index, dtype=float)

            state_series = self._vol_state_score_series(rs)
            return state_series.reindex(df.index, method="ffill").fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=df.index, dtype=float)

    def latest(self, df: pd.DataFrame) -> VolatilityOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        sc_vol_regime = int(row["sc_vol_regime"])
        sc_vol_regime_text = str(row["sc_vol_regime_text"])
        sc_vol_score = float(row["sc_vol_score"])

        bias_signal = int(row["bias_signal"])
        bias_label = str(row["bias_label"])
        market_bias = str(row["market_bias"])
        indicator_strength = float(row["indicator_strength"])
        state = str(row["state"])

        return VolatilityOutput(
            timestamp=calc.index[-1],
            state=state,
            bias_signal=bias_signal,
            bias_label=bias_label,
            indicator_strength=indicator_strength,
            market_bias=market_bias,

            sc_vol_regime=sc_vol_regime,
            sc_vol_regime_text=sc_vol_regime_text,
            sc_vol_score=sc_vol_score,
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

            vp_on=bool(self.config.vp_on),
            vol_on=bool(self.config.vol_on),
            mtf_on=bool(self.config.mtf_on),
        )


# ==============================================================================
# PAYLOAD BUILDER
# ==============================================================================

def build_volatility_latest_payload(
    df: pd.DataFrame,
    config: Optional[VolatilityConfig] = None,
) -> Dict[str, Any]:
    engine = VolatilityEngine(config=config)
    latest = engine.latest(df)
    payload = asdict(latest)

    ts = payload.get("timestamp")
    if hasattr(ts, "isoformat"):
        payload["timestamp"] = ts.isoformat()

    return payload


def run_volatility_engine(
    df: pd.DataFrame,
    config: Optional[VolatilityConfig] = None,
) -> Dict[str, Any]:
    return build_volatility_latest_payload(df=df, config=config)


def run_volatility_engine_full(
    df: pd.DataFrame,
    config: Optional[VolatilityConfig] = None,
) -> pd.DataFrame:
    engine = VolatilityEngine(config=config)
    return engine.calculate(df)