from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class LTFFirstCandleConfig:
    touch_ttl_bars: int = 12
    price_tolerance_pct: float = 0.0008


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def _to_native(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _bars_since(condition: pd.Series) -> pd.Series:
    out = np.full(len(condition), np.nan, dtype=float)
    last_true = -1
    vals = condition.fillna(False).astype(bool).to_numpy()

    for i, flag in enumerate(vals):
        if flag:
            last_true = i
            out[i] = 0.0
        elif last_true >= 0:
            out[i] = float(i - last_true)

    return pd.Series(out, index=condition.index)


def _ttl_flag(raw_touch: pd.Series, ttl_bars: int) -> pd.DataFrame:
    bars = _bars_since(raw_touch)
    active = bars.notna() & (bars >= 0) & (bars <= ttl_bars)
    ttl = pd.Series(
        np.where(active, np.maximum(0, ttl_bars - bars), 0),
        index=raw_touch.index,
        dtype=int,
    )
    return pd.DataFrame(
        {
            "touch_now": raw_touch.astype(int),
            "touch_active": active.astype(int),
            "touch_ttl": ttl,
        },
        index=raw_touch.index,
    )


def _relation_to_range(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            close > high,
            "above",
            np.where(close < low, "below", "inside"),
        ),
        index=close.index,
        dtype="object",
    )


def _first_bar_by_period(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    tf = df.resample(rule).agg(agg).dropna()
    period_key = tf.index.normalize()

    first_mask = ~period_key.duplicated()
    first_bars = tf.loc[first_mask, ["open", "high", "low", "close", "volume"]].copy()
    first_bars.columns = [
        f"first_{rule}_open",
        f"first_{rule}_high",
        f"first_{rule}_low",
        f"first_{rule}_close",
        f"first_{rule}_volume",
    ]

    first_bars.index = first_bars.index.normalize()
    mapped = first_bars.reindex(df.index.normalize(), method="ffill")
    mapped.index = df.index
    return mapped


# =============================================================================
# CORE
# =============================================================================

def calculate_ltf_first_candle(
    df: pd.DataFrame,
    config: Optional[LTFFirstCandleConfig] = None,
) -> pd.DataFrame:
    cfg = config or LTFFirstCandleConfig()
    _validate_ohlcv(df)

    base = df.copy().sort_index()
    out = pd.DataFrame(index=base.index)

    tol = (base["close"].abs() * cfg.price_tolerance_pct).astype(float)

    first_5m = _first_bar_by_period(base, "5min")
    first_15m = _first_bar_by_period(base, "15min")

    for tf_name, src, prefix in [
        ("5min", first_5m, "5m"),
        ("15min", first_15m, "15m"),
    ]:
        o = src[f"first_{tf_name}_open"]
        h = src[f"first_{tf_name}_high"]
        l = src[f"first_{tf_name}_low"]
        c = src[f"first_{tf_name}_close"]
        v = src[f"first_{tf_name}_volume"]

        rng = (h - l).abs()
        inside = ((base["close"] <= h) & (base["close"] >= l)).astype(int)
        rel_high = pd.Series(
            np.where(base["close"] > h, "above", np.where(base["close"] < h, "below", "at")),
            index=base.index,
            dtype="object",
        )
        rel_low = pd.Series(
            np.where(base["close"] > l, "above", np.where(base["close"] < l, "below", "at")),
            index=base.index,
            dtype="object",
        )
        range_relation = _relation_to_range(base["close"], h, l)

        touch_high_now = ((base["low"] <= (h + tol)) & (base["high"] >= (h - tol))).astype(bool)
        touch_low_now = ((base["low"] <= (l + tol)) & (base["high"] >= (l - tol))).astype(bool)

        high_ttl = _ttl_flag(touch_high_now, cfg.touch_ttl_bars)
        low_ttl = _ttl_flag(touch_low_now, cfg.touch_ttl_bars)

        out[f"{prefix}_open"] = o
        out[f"{prefix}_high"] = h
        out[f"{prefix}_low"] = l
        out[f"{prefix}_close"] = c
        out[f"{prefix}_volume"] = v
        out[f"{prefix}_range"] = rng
        out[f"{prefix}_inside_range"] = inside
        out[f"{prefix}_rel_high"] = rel_high
        out[f"{prefix}_rel_low"] = rel_low
        out[f"{prefix}_range_relation"] = range_relation

        out[f"{prefix}_touch_high_now"] = high_ttl["touch_now"]
        out[f"{prefix}_touch_high_active"] = high_ttl["touch_active"]
        out[f"{prefix}_touch_high_ttl"] = high_ttl["touch_ttl"]

        out[f"{prefix}_touch_low_now"] = low_ttl["touch_now"]
        out[f"{prefix}_touch_low_active"] = low_ttl["touch_active"]
        out[f"{prefix}_touch_low_ttl"] = low_ttl["touch_ttl"]

    out["ltf_any_touch"] = (
        (out["5m_touch_high_active"] == 1)
        | (out["5m_touch_low_active"] == 1)
        | (out["15m_touch_high_active"] == 1)
        | (out["15m_touch_low_active"] == 1)
    ).astype(int)

    return out


def build_ltf_first_candle(
    df: pd.DataFrame,
    config: Optional[LTFFirstCandleConfig] = None,
) -> pd.DataFrame:
    return calculate_ltf_first_candle(df, config=config)


# =============================================================================
# PAYLOAD
# =============================================================================

DEFAULT_LTF_FIRST_CANDLE_CONFIG: Dict[str, Any] = asdict(LTFFirstCandleConfig())


def build_ltf_first_candle_latest_payload(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg_dict = DEFAULT_LTF_FIRST_CANDLE_CONFIG.copy()
    if config:
        cfg_dict.update(config)

    cfg = LTFFirstCandleConfig(**cfg_dict)
    result = calculate_ltf_first_candle(df=df, config=cfg)

    if result.empty:
        raise ValueError("LTF first candle payload build failed: empty dataframe result")

    last_idx = result.index[-1]
    last = result.iloc[-1]

    rel_5m = str(_to_native(last.get("5m_range_relation")) or "inside")
    rel_15m = str(_to_native(last.get("15m_range_relation")) or "inside")

    if rel_5m == "above" or rel_15m == "above":
        state, bias_signal, bias_label, indicator_strength, market_bias = (
            "bullish_break", 1, "BULLISH", 0.85, "BULLISH"
        )
    elif rel_5m == "below" or rel_15m == "below":
        state, bias_signal, bias_label, indicator_strength, market_bias = (
            "bearish_break", -1, "BEARISH", 0.85, "BEARISH"
        )
    elif int(_to_native(last.get("5m_inside_range", 0)) or 0) == 1 or int(_to_native(last.get("15m_inside_range", 0)) or 0) == 1:
        state, bias_signal, bias_label, indicator_strength, market_bias = (
            "inside_first_candle", 0, "NEUTRAL", 0.50, "NEUTRAL"
        )
    else:
        state, bias_signal, bias_label, indicator_strength, market_bias = (
            "neutral", 0, "NEUTRAL", 0.25, "NEUTRAL"
        )

    payload = {
        "indicator": "ltf_first_candle",
        "debug_version": "ltf_first_candle_payload_v1",
        "timestamp": last_idx.isoformat(),

        "state": state,
        "bias_signal": bias_signal,
        "bias_label": bias_label,
        "indicator_strength": round(indicator_strength, 4),
        "market_bias": market_bias,

        "first_5m": {
            "open": _to_native(last.get("5m_open")),
            "high": _to_native(last.get("5m_high")),
            "low": _to_native(last.get("5m_low")),
            "close": _to_native(last.get("5m_close")),
            "volume": _to_native(last.get("5m_volume")),
            "range": _to_native(last.get("5m_range")),
            "inside_range": int(_to_native(last.get("5m_inside_range", 0)) or 0),
            "range_relation": _to_native(last.get("5m_range_relation")),
            "rel_high": _to_native(last.get("5m_rel_high")),
            "rel_low": _to_native(last.get("5m_rel_low")),
            "touch_high_active": int(_to_native(last.get("5m_touch_high_active", 0)) or 0),
            "touch_high_ttl": int(_to_native(last.get("5m_touch_high_ttl", 0)) or 0),
            "touch_low_active": int(_to_native(last.get("5m_touch_low_active", 0)) or 0),
            "touch_low_ttl": int(_to_native(last.get("5m_touch_low_ttl", 0)) or 0),
        },

        "first_15m": {
            "open": _to_native(last.get("15m_open")),
            "high": _to_native(last.get("15m_high")),
            "low": _to_native(last.get("15m_low")),
            "close": _to_native(last.get("15m_close")),
            "volume": _to_native(last.get("15m_volume")),
            "range": _to_native(last.get("15m_range")),
            "inside_range": int(_to_native(last.get("15m_inside_range", 0)) or 0),
            "range_relation": _to_native(last.get("15m_range_relation")),
            "rel_high": _to_native(last.get("15m_rel_high")),
            "rel_low": _to_native(last.get("15m_rel_low")),
            "touch_high_active": int(_to_native(last.get("15m_touch_high_active", 0)) or 0),
            "touch_high_ttl": int(_to_native(last.get("15m_touch_high_ttl", 0)) or 0),
            "touch_low_active": int(_to_native(last.get("15m_touch_low_active", 0)) or 0),
            "touch_low_ttl": int(_to_native(last.get("15m_touch_low_ttl", 0)) or 0),
        },

        "any_touch": int(_to_native(last.get("ltf_any_touch", 0)) or 0),

        "price": {
            "close": _to_native(df["close"].iloc[-1]),
        },

        "config": cfg_dict,
    }

    return payload