from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class ConfluenceCloudConfig:
    min_strength: int = 4

    # MTF weights
    w1: float = 1.0
    w5: float = 1.0
    w15: float = 1.0
    w60: float = 1.0
    w240: float = 1.0
    wD: float = 1.0

    # structural zone
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


# =============================================================================
# HELPERS
# =============================================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Confluence Cloud requires a DatetimeIndex.")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if pd.isna(value):
            return default
        return bool(value)
    except Exception:
        return default


def safe_optional_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


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


def trend_check_ema200(df: pd.DataFrame) -> pd.Series:
    e200 = ema(df["close"], 200)
    return pd.Series(np.where(df["close"] > e200, 1, -1), index=df.index, dtype=float)


def dir_text(v: float) -> str:
    if v > 0.60:
        return "STRONG BULL"
    if v > 0.20:
        return "BULL"
    if v < -0.60:
        return "STRONG BEAR"
    if v < -0.20:
        return "BEAR"
    return "NEUTRAL"


def dir_color(v: float) -> str:
    if v > 0.60:
        return "lime"
    if v > 0.20:
        return "green"
    if v < -0.60:
        return "red"
    if v < -0.20:
        return "maroon"
    return "gray"


# =============================================================================
# OUTPUT CONTRACT
# =============================================================================

@dataclass
class ConfluenceCloudOutput:
    cc_is_confluent: bool
    cc_in_structural_zone: bool
    cc_active_zone: bool

    cc_dir: int
    cc_direction: str
    cc_strength_score: float

    bull_count: int
    bear_count: int

    vwap_confirm: bool
    rsi_confirm: bool
    macd_confirm: bool

    mtf_avg: float
    mtf_bias: str
    mtf_bias_color: str

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


# =============================================================================
# CORE ENGINE
# =============================================================================

class ConfluenceCloudEngine:
    def __init__(self, config: Optional[ConfluenceCloudConfig] = None) -> None:
        self.config = config or ConfluenceCloudConfig()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv(df)

        c = self.config
        out = df.copy()

        # ---------------------------------------------------------------------
        # Current timeframe confirmation layer
        # ---------------------------------------------------------------------
        out["rsi_value"] = rsi(out["close"], 14)
        out["vwap_value"] = session_vwap(out)
        out["macd_line"], out["signal_line"], out["macd_hist"] = macd(out["close"], 12, 26, 9)

        # ---------------------------------------------------------------------
        # MTF trend gathering
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Confirmation logic — kept aligned to Pine authority exactly
        # ---------------------------------------------------------------------
        out["vwap_confirm"] = (
            ((out["bull_count"] > out["bear_count"]) & (out["close"] > out["vwap_value"]))
            | ((out["bear_count"] > out["bull_count"]) & (out["close"] < out["vwap_value"]))
        )

        out["rsi_confirm"] = (
            ((out["bull_count"] > out["bear_count"]) & (out["rsi_value"] < 50))
            | ((out["bear_count"] > out["bull_count"]) & (out["rsi_value"] > 50))
        )

        out["macd_confirm"] = (
            ((out["bull_count"] > out["bear_count"]) & (out["macd_line"] > out["signal_line"]))
            | ((out["bear_count"] > out["bull_count"]) & (out["macd_line"] < out["signal_line"]))
        )

        # ---------------------------------------------------------------------
        # Weighted MTF average
        # ---------------------------------------------------------------------
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
        out["mtf_bias"] = out["mtf_avg"].apply(dir_text)
        out["mtf_bias_color"] = out["mtf_avg"].apply(dir_color)

        # ---------------------------------------------------------------------
        # 1H structural fib cloud
        # ---------------------------------------------------------------------
        struct_df = self._build_h1_structure_map(out)
        for col in ["h1_high", "h1_low", "fib618", "fib786", "zone_top", "zone_bottom"]:
            out[col] = struct_df[col]

        # Pine uses:
        # inStructuralZone = close <= h1_high and close >= h1_low
        out["in_structural_zone"] = (out["close"] <= out["h1_high"]) & (out["close"] >= out["h1_low"])

        # ---------------------------------------------------------------------
        # Confluence aggregator
        # ---------------------------------------------------------------------
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

        out["cc_direction"] = np.select(
            [
                out["cc_dir"] == 1,
                out["cc_dir"] == -1,
            ],
            ["BULL", "BEAR"],
            default="NEUTRAL",
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

        struct = pd.DataFrame(
            {
                "h1_high": h1_high,
                "h1_low": h1_low,
                "fib618": fib618,
                "fib786": fib786,
                "zone_top": zone_top,
                "zone_bottom": zone_bottom,
            },
            index=rs.index,
        )

        return struct.reindex(df.index, method="ffill")

    def latest(self, df: pd.DataFrame) -> ConfluenceCloudOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        return ConfluenceCloudOutput(
            cc_is_confluent=safe_bool(row["is_confluent"]),
            cc_in_structural_zone=safe_bool(row["in_structural_zone"]),
            cc_active_zone=safe_bool(row["cc_active_zone"]),
            cc_dir=safe_int(row["cc_dir"]),
            cc_direction=str(row["cc_direction"]),
            cc_strength_score=safe_float(row["cc_strength_score"]),
            bull_count=safe_int(row["bull_count"]),
            bear_count=safe_int(row["bear_count"]),
            vwap_confirm=safe_bool(row["vwap_confirm"]),
            rsi_confirm=safe_bool(row["rsi_confirm"]),
            macd_confirm=safe_bool(row["macd_confirm"]),
            mtf_avg=safe_float(row["mtf_avg"]),
            mtf_bias=str(row["mtf_bias"]),
            mtf_bias_color=str(row["mtf_bias_color"]),
            t1=safe_int(row["t1"]),
            t5=safe_int(row["t5"]),
            t15=safe_int(row["t15"]),
            t60=safe_int(row["t60"]),
            t240=safe_int(row["t240"]),
            tD=safe_int(row["tD"]),
            rsi_value=safe_float(row["rsi_value"], 50.0),
            vwap_value=safe_optional_float(row["vwap_value"]),
            macd_line=safe_float(row["macd_line"]),
            signal_line=safe_float(row["signal_line"]),
            h1_high=safe_optional_float(row["h1_high"]),
            h1_low=safe_optional_float(row["h1_low"]),
            fib618=safe_optional_float(row["fib618"]),
            fib786=safe_optional_float(row["fib786"]),
            zone_top=safe_optional_float(row["zone_top"]),
            zone_bottom=safe_optional_float(row["zone_bottom"]),
            timestamp=calc.index[-1],
        )


# =============================================================================
# PUBLIC ENGINE API
# =============================================================================

def run_confluence_cloud_engine(
    df: pd.DataFrame,
    config: Optional[ConfluenceCloudConfig] = None,
) -> Dict[str, Any]:
    engine = ConfluenceCloudEngine(config=config)
    latest = engine.latest(df)
    return asdict(latest)


# =============================================================================
# WEBSITE PAYLOAD BUILDER
# =============================================================================

def build_confluence_cloud_latest_payload(
    df: pd.DataFrame,
    config: Optional[ConfluenceCloudConfig] = None,
) -> Dict[str, Any]:
    engine = ConfluenceCloudEngine(config=config)
    latest = engine.latest(df)

    zone_ready = latest.zone_top is not None and latest.zone_bottom is not None

    payload = {
        "indicator": "confluence_cloud",
        "debug_version": "confluence_cloud_payload_v1",
        "timestamp": str(latest.timestamp),

        "state": {
            "is_confluent": latest.cc_is_confluent,
            "in_structural_zone": latest.cc_in_structural_zone,
            "active_zone": latest.cc_active_zone,
            "direction": latest.cc_direction,
            "dir_value": latest.cc_dir,
            "strength_score": round(latest.cc_strength_score, 4),
        },

        "mtf": {
            "average": round(latest.mtf_avg, 4),
            "bias": latest.mtf_bias,
            "bias_color": latest.mtf_bias_color,
            "bull_count": latest.bull_count,
            "bear_count": latest.bear_count,
            "t1": latest.t1,
            "t5": latest.t5,
            "t15": latest.t15,
            "t60": latest.t60,
            "t240": latest.t240,
            "tD": latest.tD,
        },

        "confirmations": {
            "vwap_confirm": latest.vwap_confirm,
            "rsi_confirm": latest.rsi_confirm,
            "macd_confirm": latest.macd_confirm,
            "rsi_value": round(latest.rsi_value, 4),
            "vwap_value": None if latest.vwap_value is None else round(latest.vwap_value, 4),
            "macd_line": round(latest.macd_line, 6),
            "signal_line": round(latest.signal_line, 6),
        },

        "structure": {
            "h1_high": None if latest.h1_high is None else round(latest.h1_high, 4),
            "h1_low": None if latest.h1_low is None else round(latest.h1_low, 4),
            "fib618": None if latest.fib618 is None else round(latest.fib618, 4),
            "fib786": None if latest.fib786 is None else round(latest.fib786, 4),
            "zone_top": None if latest.zone_top is None else round(latest.zone_top, 4),
            "zone_bottom": None if latest.zone_bottom is None else round(latest.zone_bottom, 4),
            "zone_ready": zone_ready,
        },
    }

    return payload