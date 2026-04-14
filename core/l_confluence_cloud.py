"""
==============================================================================
SMARTCHART BACKEND — CONFLUENCE ENGINE
File: r_confluence.py
==============================================================================

LOCKED STATE — CONFLUENCE ENGINE PARITY VERSION

This module is the backend parity implementation of the locked
SmartChart Confluence Engine v2.1.

Logic preserved from Pine:
- Trend Layer
- Location Layer
- Liquidity Layer
- Momentum Layer
- Trigger Layer
- confReady / confInZone / confValid / confActive / confTTL
- Confluence score and strength mapping
- Active zone state outputs

Important:
- Visual plotting is NOT handled here
- Pine visual/table parity is separate
- Keep this module separate from l_confluence_cloud.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class ConfluenceCloudConfig:
    min_strength: int = 4

    # weights
    w1: float = 1.0
    w5: float = 1.0
    w15: float = 1.0
    w60: float = 1.0
    w240: float = 1.0
    wD: float = 1.0

    # higher-timeframe structural zone
    h1_lookback: int = 50

    # fixed TF mapping
    tf_map: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if self.tf_map is None:
            self.tf_map = {
                "1": "1min",
                "5": "5min",
                "15": "15min",
                "60": "1h",
                "240": "4h",
                "D": "1d",
            }


# ==============================================================================
# HELPERS
# ==============================================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def session_vwap(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("VWAP requires DatetimeIndex.")

    dates = df.index.normalize()
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = hlc3 * df["volume"]

    out = []
    cur_date = None
    cum_pv = 0.0
    cum_v = 0.0

    for dt, pv_i, vol_i in zip(dates, pv, df["volume"]):
        if cur_date is None or dt != cur_date:
            cur_date = dt
            cum_pv = 0.0
            cum_v = 0.0

        cum_pv += float(pv_i)
        cum_v += float(vol_i)
        out.append(cum_pv / cum_v if cum_v > 0 else np.nan)

    return pd.Series(out, index=df.index)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Resampling requires DatetimeIndex.")

    out = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()

    return out


def trend_check_ema200(df: pd.DataFrame) -> pd.Series:
    e200 = ema(df["close"], 200)
    return pd.Series(np.where(df["close"] > e200, 1, -1), index=df.index, dtype=float)


# ==============================================================================
# OUTPUT CONTRACT
# ==============================================================================

@dataclass
class ConfluenceCloudOutput:
    cc_is_confluent: bool
    cc_in_structural_zone: bool
    cc_active_zone: bool

    cc_dir: int
    cc_strength_score: float

    bull_count: int
    bear_count: int

    vwap_confirm: bool
    rsi_confirm: bool
    macd_confirm: bool

    mtf_avg: float
    mtf_bias: str

    t1: int
    t5: int
    t15: int
    t60: int
    t240: int
    tD: int

    rsi_value: float
    vwap_value: Optional[float]
    macd_line: float
    signal_line: float

    h1_high: Optional[float]
    h1_low: Optional[float]
    fib618: Optional[float]
    fib786: Optional[float]
    zone_top: Optional[float]
    zone_bottom: Optional[float]

    timestamp: Any


# ==============================================================================
# CORE ENGINE
# ==============================================================================

class ConfluenceCloudEngine:
    def __init__(self, config: Optional[ConfluenceCloudConfig] = None) -> None:
        self.config = config or ConfluenceCloudConfig()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("ConfluenceCloudEngine requires a DatetimeIndex.")

        c = self.config
        out = df.copy()

        # ----------------------------------------------------------------------
        # Current timeframe confirmation layer
        # ----------------------------------------------------------------------
        out["rsi_value"] = rsi(out["close"], 14)
        out["vwap_value"] = session_vwap(out)
        out["macd_line"], out["signal_line"], out["macd_hist"] = macd(out["close"], 12, 26, 9)

        # ----------------------------------------------------------------------
        # MTF trend gathering
        # ----------------------------------------------------------------------
        mtf_series = self._build_mtf_trend_map(out)
        for col in ["t1", "t5", "t15", "t60", "t240", "tD"]:
            out[col] = mtf_series[col]

        out["bull_count"] = (
            (out["t1"] == 1).astype(int)
            + (out["t5"] == 1).astype(int)
            + (out["t15"] == 1).astype(int)
            + (out["t60"] == 1).astype(int)
            + (out["t240"] == 1).astype(int)
            + (out["tD"] == 1).astype(int)
        )

        out["bear_count"] = (
            (out["t1"] == -1).astype(int)
            + (out["t5"] == -1).astype(int)
            + (out["t15"] == -1).astype(int)
            + (out["t60"] == -1).astype(int)
            + (out["t240"] == -1).astype(int)
            + (out["tD"] == -1).astype(int)
        )

        # ----------------------------------------------------------------------
        # Current timeframe confirmation logic
        # ----------------------------------------------------------------------
        out["vwap_confirm"] = (
            ((out["bull_count"] > out["bear_count"]) & (out["close"] > out["vwap_value"]))
            | ((out["bear_count"] > out["bull_count"]) & (out["close"] < out["vwap_value"]))
        )

        # This follows your Pine script exactly
        out["rsi_confirm"] = (
            ((out["bull_count"] > out["bear_count"]) & (out["rsi_value"] < 50))
            | ((out["bear_count"] > out["bull_count"]) & (out["rsi_value"] > 50))
        )

        out["macd_confirm"] = (
            ((out["bull_count"] > out["bear_count"]) & (out["macd_line"] > out["signal_line"]))
            | ((out["bear_count"] > out["bull_count"]) & (out["macd_line"] < out["signal_line"]))
        )

        # ----------------------------------------------------------------------
        # Weighted MTF average
        # ----------------------------------------------------------------------
        w_sum = c.w1 + c.w5 + c.w15 + c.w60 + c.w240 + c.wD

        if w_sum > 0:
            out["mtf_avg"] = (
                out["t1"] * c.w1
                + out["t5"] * c.w5
                + out["t15"] * c.w15
                + out["t60"] * c.w60
                + out["t240"] * c.w240
                + out["tD"] * c.wD
            ) / w_sum
        else:
            out["mtf_avg"] = 0.0

        out["mtf_avg"] = out["mtf_avg"].clip(-1.0, 1.0)
        out["mtf_bias"] = out["mtf_avg"].apply(self._dir_text)

        # ----------------------------------------------------------------------
        # 1H structural fib cloud
        # ----------------------------------------------------------------------
        struct_df = self._build_h1_structure_map(out)
        for col in ["h1_high", "h1_low", "fib618", "fib786", "zone_top", "zone_bottom"]:
            out[col] = struct_df[col]

        out["in_structural_zone"] = (
            out["close"] <= out["h1_high"]
        ) & (
            out["close"] >= out["h1_low"]
        )

        # ----------------------------------------------------------------------
        # Confluence aggregator
        # ----------------------------------------------------------------------
        out["is_confluent"] = (
            ((out["bull_count"] >= c.min_strength) | (out["bear_count"] >= c.min_strength))
            & out["vwap_confirm"]
            & out["rsi_confirm"]
            & out["macd_confirm"]
        )

        out["cc_active_zone"] = out["is_confluent"] & out["in_structural_zone"]

        out["cc_dir"] = np.select(
            [
                out["bull_count"] > out["bear_count"],
                out["bear_count"] > out["bull_count"],
            ],
            [1, -1],
            default=0,
        )

        agreement_strength = np.maximum(out["bull_count"], out["bear_count"]) / 6.0
        confirm_strength = (
            out["vwap_confirm"].astype(float) * 0.34
            + out["rsi_confirm"].astype(float) * 0.33
            + out["macd_confirm"].astype(float) * 0.33
        )
        mtf_strength = out["mtf_avg"].abs()

        out["cc_strength_score"] = np.clip(
            agreement_strength * 0.50
            + confirm_strength * 0.25
            + mtf_strength * 0.25,
            0.0,
            1.0,
        )

        return out

    def _build_mtf_trend_map(self, df: pd.DataFrame) -> pd.DataFrame:
        tf_map = self.config.tf_map or {}
        result = pd.DataFrame(index=df.index)

        mapping = {
            "t1": tf_map["1"],
            "t5": tf_map["5"],
            "t15": tf_map["15"],
            "t60": tf_map["60"],
            "t240": tf_map["240"],
            "tD": tf_map["D"],
        }

        for out_col, rule in mapping.items():
            try:
                rs = resample_ohlcv(df, rule)
                state = trend_check_ema200(rs)
                aligned = state.reindex(df.index, method="ffill").fillna(-1.0)
                result[out_col] = aligned.astype(int)
            except Exception:
                result[out_col] = -1

        return result

    def _build_h1_structure_map(self, df: pd.DataFrame) -> pd.DataFrame:
        tf_map = self.config.tf_map or {}
        rule = tf_map["60"]

        rs = resample_ohlcv(df, rule)

        h1_high = rs["high"].rolling(self.config.h1_lookback, min_periods=1).max()
        h1_low = rs["low"].rolling(self.config.h1_lookback, min_periods=1).min()

        fib618 = h1_high - (h1_high - h1_low) * 0.618
        fib786 = h1_high - (h1_high - h1_low) * 0.786

        zone_top = np.maximum(fib618, fib786)
        zone_bottom = np.minimum(fib618, fib786)

        struct = pd.DataFrame({
            "h1_high": h1_high,
            "h1_low": h1_low,
            "fib618": fib618,
            "fib786": fib786,
            "zone_top": zone_top,
            "zone_bottom": zone_bottom,
        }, index=rs.index)

        return struct.reindex(df.index, method="ffill")

    def _dir_text(self, v: float) -> str:
        if v > 0.60:
            return "STRONG BULL"
        if v > 0.20:
            return "BULL"
        if v < -0.60:
            return "STRONG BEAR"
        if v < -0.20:
            return "BEAR"
        return "NEUTRAL"

    def latest(self, df: pd.DataFrame) -> ConfluenceCloudOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        return ConfluenceCloudOutput(
            cc_is_confluent=bool(row["is_confluent"]),
            cc_in_structural_zone=bool(row["in_structural_zone"]),
            cc_active_zone=bool(row["cc_active_zone"]),
            cc_dir=int(row["cc_dir"]),
            cc_strength_score=float(row["cc_strength_score"]),
            bull_count=int(row["bull_count"]),
            bear_count=int(row["bear_count"]),
            vwap_confirm=bool(row["vwap_confirm"]),
            rsi_confirm=bool(row["rsi_confirm"]),
            macd_confirm=bool(row["macd_confirm"]),
            mtf_avg=float(row["mtf_avg"]),
            mtf_bias=str(row["mtf_bias"]),
            t1=int(row["t1"]),
            t5=int(row["t5"]),
            t15=int(row["t15"]),
            t60=int(row["t60"]),
            t240=int(row["t240"]),
            tD=int(row["tD"]),
            rsi_value=float(row["rsi_value"]),
            vwap_value=None if pd.isna(row["vwap_value"]) else float(row["vwap_value"]),
            macd_line=float(row["macd_line"]),
            signal_line=float(row["signal_line"]),
            h1_high=None if pd.isna(row["h1_high"]) else float(row["h1_high"]),
            h1_low=None if pd.isna(row["h1_low"]) else float(row["h1_low"]),
            fib618=None if pd.isna(row["fib618"]) else float(row["fib618"]),
            fib786=None if pd.isna(row["fib786"]) else float(row["fib786"]),
            zone_top=None if pd.isna(row["zone_top"]) else float(row["zone_top"]),
            zone_bottom=None if pd.isna(row["zone_bottom"]) else float(row["zone_bottom"]),
            timestamp=calc.index[-1],
        )


# ==============================================================================
# PUBLIC API
# ==============================================================================

def run_confluence_cloud_engine(
    df: pd.DataFrame,
    config: Optional[ConfluenceCloudConfig] = None,
) -> Dict[str, Any]:
    engine = ConfluenceCloudEngine(config=config)
    latest = engine.latest(df)
    return asdict(latest)


# ==============================================================================
# DIRECT TEST BLOCK
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    periods = 1800
    idx = pd.date_range("2026-01-01", periods=periods, freq="1min")

    base = 100 + np.cumsum(np.random.normal(0, 0.08, periods))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + np.random.uniform(0.01, 0.12, periods)
    low = np.minimum(open_, close) - np.random.uniform(0.01, 0.12, periods)
    volume = np.random.randint(100, 1000, periods)

    for k in [250, 800, 1200]:
        if k + 80 < periods:
            close.iloc[k:k + 80] += np.linspace(0.0, 3.0, 80)
            high.iloc[k:k + 80] = np.maximum(high.iloc[k:k + 80], close.iloc[k:k + 80] + 0.05)

    for k in [500, 1450]:
        if k + 70 < periods:
            close.iloc[k:k + 70] -= np.linspace(0.0, 2.5, 70)
            low.iloc[k:k + 70] = np.minimum(low.iloc[k:k + 70], close.iloc[k:k + 70] - 0.05)

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

    config = ConfluenceCloudConfig(
        min_strength=4,
        w1=1.0,
        w5=1.0,
        w15=1.0,
        w60=1.0,
        w240=1.0,
        wD=1.0,
    )

    engine = ConfluenceCloudEngine(config=config)
    full = engine.calculate(test_df)
    latest = engine.latest(test_df)

    print("\n=== SMARTCHART CONFLUENCE CLOUD TEST ===")
    print(
        full[
            [
                "is_confluent",
                "in_structural_zone",
                "cc_active_zone",
                "cc_dir",
                "cc_strength_score",
                "bull_count",
                "bear_count",
                "mtf_avg",
                "mtf_bias",
            ]
        ].tail(15)
    )

    print("\n=== LATEST OUTPUT ===")
    print(asdict(latest))