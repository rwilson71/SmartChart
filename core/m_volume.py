"""
SmartChart Backend — m_volume.py

Volume + Orderflow Engine v3
Production rebuild from locked Volume Module v2
with TradingView Orderflow authority merged into backend structure.

Core purpose:
- Preserve SmartChart volume engine logic
- Add backend-safe institutional orderflow layer
- Build clean website payload from backend truth only

Production rules:
- No frontend logic duplication
- No visual/chart objects
- No direct test block in core
- Payload builder stays inside core as single source of truth
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class VolumeConfig:
    # -------------------------------------------------------------------------
    # Volume core
    # -------------------------------------------------------------------------
    vol_len: int = 20
    vol_smooth_len: int = 8
    range_len: int = 20

    exp_mult: float = 1.20
    climax_mult: float = 1.80
    low_mult: float = 0.80

    range_expand_mult: float = 1.10
    range_climax_mult: float = 1.50
    small_body_thresh: float = 0.35

    # Memory
    confirm_bars: int = 2
    hold_bars: int = 3
    bias_refresh_bars: int = 1

    # -------------------------------------------------------------------------
    # Orderflow authority merge
    # -------------------------------------------------------------------------
    atr_len: int = 14
    atr_base_len: int = 50
    avg_vol_len: int = 20

    # Order blocks
    ob_lookback: int = 20
    ob_vol_mult: float = 1.30
    ob_atr_mult: float = 1.50
    max_obs: int = 8

    # FVG
    fvg_min_gap_pct: float = 0.30
    max_fvgs: int = 15

    # Liquidity
    liq_swing_len: int = 5
    liq_tolerance_pct: float = 0.15
    max_liq_levels: int = 20

    # Premium / discount array
    pd_array_len: int = 50

    # Volume imbalance
    imbalance_vol_mult: float = 1.50
    imbalance_body_atr_mult: float = 0.50

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


# =============================================================================
# HELPERS
# =============================================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.rolling(length, min_periods=1).mean()


def safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    fallback: float = 0.0,
) -> pd.Series:
    num = pd.Series(numerator)
    den = pd.Series(denominator).replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(fallback)


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


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return sma(true_range(df), length)


def rolling_pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    vals = high.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(left, len(vals) - right):
        center = vals[i]
        if np.isnan(center):
            continue
        window = vals[i - left : i + right + 1]
        if np.all(center >= window):
            out[i] = center
    return pd.Series(out, index=high.index)


def rolling_pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    vals = low.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)

    for i in range(left, len(vals) - right):
        center = vals[i]
        if np.isnan(center):
            continue
        window = vals[i - left : i + right + 1]
        if np.all(center <= window):
            out[i] = center
    return pd.Series(out, index=low.index)


def state_to_score(s: int) -> float:
    if s == 2:
        return 1.00
    if s == 1:
        return 0.55
    if s == 0:
        return 0.00
    if s == -1:
        return -0.55
    return 0.00


def bias_to_score(b: int) -> float:
    if b == 1:
        return 1.0
    if b == -1:
        return -1.0
    return 0.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


# =============================================================================
# OUTPUT CONTRACT
# =============================================================================

@dataclass
class VolumeOutput:
    # Volume core
    vol_state: int
    vol_bias: int
    vol_age: int

    vol_flip_export: int
    vol_expand_export: int
    vol_climax_export: int
    vol_low_export: int
    vol_normal_export: int

    entered_climax: bool
    entered_expand: bool
    entered_normal: bool
    entered_low: bool

    bull_expand_pulse: bool
    bear_expand_pulse: bool
    bull_climax_pulse: bool
    bear_climax_pulse: bool

    vol_ratio: float
    range_ratio: float
    body_frac: float
    close_pos: float
    bias_strength: float

    raw_vol_state: int
    raw_bias: int

    mtf_bias_avg: float
    mtf_state_avg: float
    mtf_bias_dir: int
    mtf_bias_text: str
    mtf_state_text: str

    # Orderflow layer
    orderflow_bias: int
    orderflow_bias_text: str
    orderflow_state: str

    volume_imbalance: bool
    volume_imbalance_bull: bool
    volume_imbalance_bear: bool

    active_order_blocks: int
    bullish_order_blocks: int
    bearish_order_blocks: int
    order_block_signal: int
    order_block_top: float
    order_block_bottom: float
    order_block_eq: float
    order_block_strength: str

    active_fvgs: int
    bullish_fvgs: int
    bearish_fvgs: int
    fvg_signal: int
    fvg_top: float
    fvg_bottom: float
    fvg_gap_pct: float

    active_liquidity_levels: int
    buy_liquidity_levels: int
    sell_liquidity_levels: int
    liquidity_bias: int
    nearest_liquidity_price: float
    nearest_liquidity_distance_pct: float

    pd_position: str
    pd_bias: int
    range_high: float
    range_low: float
    range_eq: float
    premium_50: float
    premium_75: float
    discount_50: float
    discount_75: float

    breaker_bullish: bool
    breaker_bearish: bool
    mitigation_bullish: bool
    mitigation_bearish: bool

    confluence_score: int
    confluence_level: str
    signal_quality: float

    timestamp: Any


# =============================================================================
# CORE ENGINE
# =============================================================================

class VolumeEngine:
    def __init__(self, config: Optional[VolumeConfig] = None) -> None:
        self.config = config or VolumeConfig()

    # -------------------------------------------------------------------------
    # MAIN
    # -------------------------------------------------------------------------
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("VolumeEngine requires a DatetimeIndex.")

        c = self.config
        out = df.copy()

        # =====================================================================
        # BASE SERIES
        # =====================================================================
        out["atr"] = atr(out, c.atr_len)
        out["atr_sma50"] = sma(out["atr"], c.atr_base_len)
        out["avg_vol"] = sma(out["volume"], c.avg_vol_len)

        out["vol_sm"] = ema(out["volume"], c.vol_smooth_len)
        out["vol_base"] = sma(out["volume"], c.vol_len)
        out["vol_ratio0"] = safe_ratio(out["vol_sm"], out["vol_base"], fallback=1.0).to_numpy()
        out["vol_ratio"] = pd.Series(out["vol_ratio0"], index=out.index).clip(0.0, 10.0)

        out["bar_range"] = (out["high"] - out["low"]).clip(lower=0.0)
        out["range_base"] = sma(out["bar_range"], c.range_len)
        out["range_ratio0"] = safe_ratio(out["bar_range"], out["range_base"], fallback=1.0).to_numpy()
        out["range_ratio"] = pd.Series(out["range_ratio0"], index=out.index).clip(0.0, 10.0)

        out["body_size"] = (out["close"] - out["open"]).abs()
        out["body_frac0"] = safe_ratio(out["body_size"], out["bar_range"], fallback=0.0).to_numpy()
        out["body_frac"] = pd.Series(out["body_frac0"], index=out.index).clip(0.0, 1.0)

        out["close_pos0"] = safe_ratio(out["close"] - out["low"], out["bar_range"], fallback=0.5).to_numpy()
        out["close_pos"] = pd.Series(out["close_pos0"], index=out.index).clip(0.0, 1.0)

        out["is_bull_bar"] = out["close"] > out["open"]
        out["is_bear_bar"] = out["close"] < out["open"]
        out["is_small_body"] = out["body_frac"] <= c.small_body_thresh

        # =====================================================================
        # DIRECTIONAL PARTICIPATION
        # =====================================================================
        out["bull_effort"] = np.where(out["is_bull_bar"], out["volume"] * out["body_frac"], 0.0)
        out["bear_effort"] = np.where(out["is_bear_bar"], out["volume"] * out["body_frac"], 0.0)

        out["bull_effort_sm"] = ema(pd.Series(out["bull_effort"], index=out.index), c.vol_len)
        out["bear_effort_sm"] = ema(pd.Series(out["bear_effort"], index=out.index), c.vol_len)

        out["raw_bias"] = np.select(
            [
                out["bull_effort_sm"] > out["bear_effort_sm"],
                out["bear_effort_sm"] > out["bull_effort_sm"],
            ],
            [1, -1],
            default=0,
        ).astype(int)

        denom = out["bull_effort_sm"] + out["bear_effort_sm"]
        out["bias_strength_raw"] = safe_ratio(
            (out["bull_effort_sm"] - out["bear_effort_sm"]).abs(),
            denom,
            fallback=0.0,
        ).to_numpy()
        out["bias_strength"] = pd.Series(out["bias_strength_raw"], index=out.index).clip(0.0, 1.0)

        # =====================================================================
        # RAW VOLUME CLASSIFICATION
        # =====================================================================
        out["is_climax"] = (
            ((out["vol_ratio"] >= c.climax_mult) & (out["range_ratio"] >= c.range_climax_mult))
            | ((out["vol_ratio"] >= c.climax_mult) & out["is_small_body"])
        )

        out["is_expansion"] = (
            (~out["is_climax"])
            & (out["vol_ratio"] >= c.exp_mult)
            & (out["range_ratio"] >= c.range_expand_mult)
        )

        out["is_low"] = (
            (out["vol_ratio"] <= c.low_mult)
            & (out["range_ratio"] <= 1.0)
        )

        out["is_normal"] = (~out["is_climax"]) & (~out["is_expansion"]) & (~out["is_low"])

        out["raw_vol_state"] = np.select(
            [out["is_climax"], out["is_expansion"], out["is_low"]],
            [2, 1, -1],
            default=0,
        ).astype(int)

        # =====================================================================
        # MEMORY HELPERS
        # =====================================================================
        raw_state_values = out["raw_vol_state"].to_numpy()
        raw_bias_values = out["raw_bias"].to_numpy()

        raw_stable_bars: List[int] = []
        bias_stable_bars: List[int] = []

        state_run = 0
        bias_run = 0

        for i in range(len(out)):
            if i == 0:
                state_run = 1
                bias_run = 1
            else:
                state_run = state_run + 1 if raw_state_values[i] == raw_state_values[i - 1] else 1
                bias_run = bias_run + 1 if raw_bias_values[i] == raw_bias_values[i - 1] else 1

            raw_stable_bars.append(state_run)
            bias_stable_bars.append(bias_run)

        out["raw_stable_bars"] = raw_stable_bars
        out["state_ready"] = out["raw_stable_bars"] >= c.confirm_bars

        out["bias_stable_bars"] = bias_stable_bars
        out["bias_ready"] = out["bias_stable_bars"] >= c.bias_refresh_bars

        # =====================================================================
        # MEMORY ENGINE
        # =====================================================================
        vol_state_list: List[int] = []
        vol_bias_list: List[int] = []
        vol_age_list: List[int] = []

        current_state = int(out["raw_vol_state"].iat[0]) if len(out) else 0
        current_bias = int(out["raw_bias"].iat[0]) if len(out) else 0
        current_age = 0

        for i in range(len(out)):
            raw_state_i = int(out["raw_vol_state"].iat[i])
            raw_bias_i = int(out["raw_bias"].iat[i])
            state_ready_i = bool(out["state_ready"].iat[i])
            bias_ready_i = bool(out["bias_ready"].iat[i])

            if i == 0:
                current_state = raw_state_i
                current_bias = raw_bias_i
                current_age = 0
            else:
                can_flip = current_age >= c.hold_bars
                allow_state_flip = state_ready_i and (raw_state_i != current_state) and can_flip
                allow_bias_refresh = bias_ready_i and (raw_bias_i != 0) and (raw_bias_i != current_bias)

                if allow_state_flip:
                    current_state = raw_state_i
                    current_bias = raw_bias_i
                    current_age = 0
                else:
                    current_age += 1
                    if allow_bias_refresh:
                        current_bias = raw_bias_i

            vol_state_list.append(current_state)
            vol_bias_list.append(current_bias)
            vol_age_list.append(current_age)

        out["vol_state"] = vol_state_list
        out["vol_bias"] = vol_bias_list
        out["vol_age"] = vol_age_list

        # =====================================================================
        # EVENTS
        # =====================================================================
        prev_state = out["vol_state"].shift(1).fillna(out["vol_state"])
        out["state_changed"] = out["vol_state"] != prev_state

        out["entered_climax"] = out["state_changed"] & (out["vol_state"] == 2)
        out["entered_expand"] = out["state_changed"] & (out["vol_state"] == 1)
        out["entered_normal"] = out["state_changed"] & (out["vol_state"] == 0)
        out["entered_low"] = out["state_changed"] & (out["vol_state"] == -1)

        out["bull_expand_pulse"] = out["entered_expand"] & (out["vol_bias"] == 1)
        out["bear_expand_pulse"] = out["entered_expand"] & (out["vol_bias"] == -1)
        out["bull_climax_pulse"] = out["entered_climax"] & (out["vol_bias"] == 1)
        out["bear_climax_pulse"] = out["entered_climax"] & (out["vol_bias"] == -1)

        out["vol_state_export"] = out["vol_state"]
        out["vol_bias_export"] = out["vol_bias"]
        out["vol_flip_export"] = np.where(out["state_changed"], 1, 0).astype(int)
        out["vol_expand_export"] = np.where(out["entered_expand"], 1, 0).astype(int)
        out["vol_climax_export"] = np.where(out["entered_climax"], 1, 0).astype(int)
        out["vol_low_export"] = np.where(out["entered_low"], 1, 0).astype(int)
        out["vol_normal_export"] = np.where(out["entered_normal"], 1, 0).astype(int)

        # =====================================================================
        # ORDERFLOW LAYER
        # =====================================================================
        self._calculate_volume_imbalance(out)
        self._calculate_order_blocks(out)
        self._calculate_fvgs(out)
        self._calculate_liquidity(out)
        self._calculate_premium_discount(out)
        self._calculate_breaker_mitigation(out)
        self._calculate_confluence(out)

        # =====================================================================
        # MTF
        # =====================================================================
        out["mtf_bias_avg"] = 0.0
        out["mtf_state_avg"] = 0.0

        if c.mtf_on:
            mtf_df = self._calculate_mtf_average(df)
            out["mtf_bias_avg"] = mtf_df["mtf_bias_avg"].reindex(out.index).fillna(0.0)
            out["mtf_state_avg"] = mtf_df["mtf_state_avg"].reindex(out.index).fillna(0.0)

        out["mtf_bias_dir"] = out["mtf_bias_avg"].apply(sign)

        def bias_text(v: int) -> str:
            return "BULL" if v > 0 else "BEAR" if v < 0 else "NEUTRAL"

        def state_text(v: float) -> str:
            if v > 0.75:
                return "CLIMAX"
            if v > 0.20:
                return "EXPANSION"
            if v < -0.20:
                return "LOW"
            return "NORMAL"

        out["mtf_bias_text"] = out["mtf_bias_dir"].apply(bias_text)
        out["mtf_state_text"] = out["mtf_state_avg"].apply(state_text)

        # Orderflow bias summary
        out["orderflow_bias"] = np.select(
            [
                (out["confluence_score"] >= 60) & (out["vol_bias"] > 0),
                (out["confluence_score"] >= 60) & (out["vol_bias"] < 0),
                out["vol_bias"] > 0,
                out["vol_bias"] < 0,
            ],
            [1, -1, 1, -1],
            default=0,
        ).astype(int)

        out["orderflow_bias_text"] = out["orderflow_bias"].map(
            {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
        ).fillna("NEUTRAL")

        out["orderflow_state"] = np.select(
            [
                out["confluence_score"] >= 80,
                out["confluence_score"] >= 60,
                out["confluence_score"] >= 40,
            ],
            ["EXTREME", "STRONG", "MODERATE"],
            default="WEAK",
        )

        return out

    # -------------------------------------------------------------------------
    # ORDERFLOW COMPONENTS
    # -------------------------------------------------------------------------
    def _calculate_volume_imbalance(self, out: pd.DataFrame) -> None:
        c = self.config

        out["buy_volume_proxy"] = np.where(out["close"] > out["open"], out["volume"], 0.0)
        out["sell_volume_proxy"] = np.where(out["close"] < out["open"], out["volume"], 0.0)
        out["volume_delta_proxy"] = out["buy_volume_proxy"].cumsum() - out["sell_volume_proxy"].cumsum()

        out["volume_imbalance"] = (
            (out["volume"] > out["avg_vol"] * c.imbalance_vol_mult)
            & (out["body_size"] > out["atr"] * c.imbalance_body_atr_mult)
        )

        out["volume_imbalance_bull"] = out["volume_imbalance"] & (out["close"] > out["open"])
        out["volume_imbalance_bear"] = out["volume_imbalance"] & (out["close"] < out["open"])

    def _calculate_order_blocks(self, out: pd.DataFrame) -> None:
        c = self.config
        n = len(out)

        active_obs: List[Dict[str, Any]] = []
        active_count = np.zeros(n, dtype=int)
        bull_count = np.zeros(n, dtype=int)
        bear_count = np.zeros(n, dtype=int)

        ob_signal = np.zeros(n, dtype=int)
        ob_top = np.full(n, np.nan, dtype=float)
        ob_bottom = np.full(n, np.nan, dtype=float)
        ob_eq = np.full(n, np.nan, dtype=float)
        ob_strength = np.array(["NONE"] * n, dtype=object)

        breaker_bullish = np.zeros(n, dtype=bool)
        breaker_bearish = np.zeros(n, dtype=bool)
        mitigation_bullish = np.zeros(n, dtype=bool)
        mitigation_bearish = np.zeros(n, dtype=bool)

        for i in range(1, n):
            # New bullish OB: last bearish candle before strong bullish move
            bullish_ob = (
                (out["close"].iat[i - 1] < out["open"].iat[i - 1])
                and (out["close"].iat[i] > out["open"].iat[i])
                and ((out["high"].iat[i] - out["low"].iat[i]) > out["atr"].iat[i] * c.ob_atr_mult)
                and (out["volume"].iat[i] > out["avg_vol"].iat[i] * c.ob_vol_mult)
            )

            # New bearish OB: last bullish candle before strong bearish move
            bearish_ob = (
                (out["close"].iat[i - 1] > out["open"].iat[i - 1])
                and (out["close"].iat[i] < out["open"].iat[i])
                and ((out["high"].iat[i] - out["low"].iat[i]) > out["atr"].iat[i] * c.ob_atr_mult)
                and (out["volume"].iat[i] > out["avg_vol"].iat[i] * c.ob_vol_mult)
            )

            if bullish_ob:
                top = float(out["high"].iat[i - 1])
                bottom = float(out["low"].iat[i - 1])
                eq = (top + bottom) / 2.0
                strength = "STRONG" if out["volume"].iat[i] > out["avg_vol"].iat[i] * 1.8 else "BULL"
                active_obs.append(
                    {
                        "top": top,
                        "bottom": bottom,
                        "eq": eq,
                        "is_bullish": True,
                        "strength": strength,
                    }
                )
                ob_signal[i] = 1
                ob_top[i] = top
                ob_bottom[i] = bottom
                ob_eq[i] = eq
                ob_strength[i] = strength

            if bearish_ob:
                top = float(out["high"].iat[i - 1])
                bottom = float(out["low"].iat[i - 1])
                eq = (top + bottom) / 2.0
                strength = "STRONG" if out["volume"].iat[i] > out["avg_vol"].iat[i] * 1.8 else "BEAR"
                active_obs.append(
                    {
                        "top": top,
                        "bottom": bottom,
                        "eq": eq,
                        "is_bullish": False,
                        "strength": strength,
                    }
                )
                ob_signal[i] = -1
                ob_top[i] = top
                ob_bottom[i] = bottom
                ob_eq[i] = eq
                ob_strength[i] = strength

            if len(active_obs) > c.max_obs:
                active_obs = active_obs[-c.max_obs:]

            survivors: List[Dict[str, Any]] = []
            for ob in active_obs:
                broken_bull = ob["is_bullish"] and (out["close"].iat[i] < ob["bottom"])
                broken_bear = (not ob["is_bullish"]) and (out["close"].iat[i] > ob["top"])

                if broken_bull:
                    breaker_bearish[i] = True
                    continue
                if broken_bear:
                    breaker_bullish[i] = True
                    continue

                if ob["is_bullish"]:
                    if (
                        out["low"].iat[i] <= ob["top"]
                        and out["low"].iat[i] >= ob["bottom"]
                        and out["close"].iat[i] > out["open"].iat[i]
                    ):
                        mitigation_bullish[i] = True
                else:
                    if (
                        out["high"].iat[i] >= ob["bottom"]
                        and out["high"].iat[i] <= ob["top"]
                        and out["close"].iat[i] < out["open"].iat[i]
                    ):
                        mitigation_bearish[i] = True

                survivors.append(ob)

            active_obs = survivors

            active_count[i] = len(active_obs)
            bull_count[i] = sum(1 for ob in active_obs if ob["is_bullish"])
            bear_count[i] = sum(1 for ob in active_obs if not ob["is_bullish"])

            if active_obs and ob_signal[i] == 0:
                nearest = active_obs[-1]
                ob_top[i] = nearest["top"]
                ob_bottom[i] = nearest["bottom"]
                ob_eq[i] = nearest["eq"]
                ob_strength[i] = nearest["strength"]

        out["active_order_blocks"] = active_count
        out["bullish_order_blocks"] = bull_count
        out["bearish_order_blocks"] = bear_count
        out["order_block_signal"] = ob_signal
        out["order_block_top"] = ob_top
        out["order_block_bottom"] = ob_bottom
        out["order_block_eq"] = ob_eq
        out["order_block_strength"] = ob_strength
        out["breaker_bullish"] = breaker_bullish
        out["breaker_bearish"] = breaker_bearish
        out["mitigation_bullish"] = mitigation_bullish
        out["mitigation_bearish"] = mitigation_bearish

    def _calculate_fvgs(self, out: pd.DataFrame) -> None:
        c = self.config
        n = len(out)

        active_fvgs: List[Dict[str, Any]] = []
        active_count = np.zeros(n, dtype=int)
        bull_count = np.zeros(n, dtype=int)
        bear_count = np.zeros(n, dtype=int)

        fvg_signal = np.zeros(n, dtype=int)
        fvg_top = np.full(n, np.nan, dtype=float)
        fvg_bottom = np.full(n, np.nan, dtype=float)
        fvg_gap_pct = np.zeros(n, dtype=float)

        for i in range(2, n):
            bullish_fvg = out["low"].iat[i] > out["high"].iat[i - 2]
            bearish_fvg = out["high"].iat[i] < out["low"].iat[i - 2]

            bull_size = (
                ((out["low"].iat[i] - out["high"].iat[i - 2]) / out["high"].iat[i - 2]) * 100.0
                if bullish_fvg and out["high"].iat[i - 2] != 0
                else 0.0
            )

            bear_size = (
                ((out["low"].iat[i - 2] - out["high"].iat[i]) / out["low"].iat[i - 2]) * 100.0
                if bearish_fvg and out["low"].iat[i - 2] != 0
                else 0.0
            )

            if bullish_fvg and bull_size >= c.fvg_min_gap_pct:
                top = float(out["low"].iat[i])
                bottom = float(out["high"].iat[i - 2])
                active_fvgs.append(
                    {
                        "top": top,
                        "bottom": bottom,
                        "is_bullish": True,
                        "gap_pct": bull_size,
                    }
                )
                fvg_signal[i] = 1
                fvg_top[i] = top
                fvg_bottom[i] = bottom
                fvg_gap_pct[i] = bull_size

            if bearish_fvg and bear_size >= c.fvg_min_gap_pct:
                top = float(out["low"].iat[i - 2])
                bottom = float(out["high"].iat[i])
                active_fvgs.append(
                    {
                        "top": top,
                        "bottom": bottom,
                        "is_bullish": False,
                        "gap_pct": bear_size,
                    }
                )
                fvg_signal[i] = -1
                fvg_top[i] = top
                fvg_bottom[i] = bottom
                fvg_gap_pct[i] = bear_size

            if len(active_fvgs) > c.max_fvgs:
                active_fvgs = active_fvgs[-c.max_fvgs:]

            survivors: List[Dict[str, Any]] = []
            for fvg in active_fvgs:
                filled_bull = fvg["is_bullish"] and (out["low"].iat[i] <= fvg["bottom"])
                filled_bear = (not fvg["is_bullish"]) and (out["high"].iat[i] >= fvg["top"])

                if not (filled_bull or filled_bear):
                    survivors.append(fvg)

            active_fvgs = survivors

            active_count[i] = len(active_fvgs)
            bull_count[i] = sum(1 for f in active_fvgs if f["is_bullish"])
            bear_count[i] = sum(1 for f in active_fvgs if not f["is_bullish"])

            if active_fvgs and fvg_signal[i] == 0:
                nearest = active_fvgs[-1]
                fvg_top[i] = nearest["top"]
                fvg_bottom[i] = nearest["bottom"]
                fvg_gap_pct[i] = nearest["gap_pct"]

        out["active_fvgs"] = active_count
        out["bullish_fvgs"] = bull_count
        out["bearish_fvgs"] = bear_count
        out["fvg_signal"] = fvg_signal
        out["fvg_top"] = fvg_top
        out["fvg_bottom"] = fvg_bottom
        out["fvg_gap_pct"] = fvg_gap_pct

    def _calculate_liquidity(self, out: pd.DataFrame) -> None:
        c = self.config
        n = len(out)

        ph = rolling_pivot_high(out["high"], c.liq_swing_len, c.liq_swing_len)
        pl = rolling_pivot_low(out["low"], c.liq_swing_len, c.liq_swing_len)

        out["pivot_high"] = ph
        out["pivot_low"] = pl

        active_liq: List[Dict[str, Any]] = []

        active_count = np.zeros(n, dtype=int)
        buy_count = np.zeros(n, dtype=int)
        sell_count = np.zeros(n, dtype=int)
        liq_bias = np.zeros(n, dtype=int)
        nearest_price = np.full(n, np.nan, dtype=float)
        nearest_dist_pct = np.full(n, np.nan, dtype=float)

        ph_vals = ph.to_numpy(dtype=float)
        pl_vals = pl.to_numpy(dtype=float)
        close_vals = out["close"].to_numpy(dtype=float)

        for i in range(n):
            # Equal highs
            if not np.isnan(ph_vals[i]):
                is_equal = False
                for j in range(1, 4):
                    idx = i - j * c.liq_swing_len
                    if idx >= 0 and not np.isnan(ph_vals[idx]):
                        base = ph_vals[i] if ph_vals[i] != 0 else np.nan
                        diff = abs(ph_vals[i] - ph_vals[idx]) / base * 100.0 if base and not np.isnan(base) else np.nan
                        if not np.isnan(diff) and diff <= c.liq_tolerance_pct:
                            is_equal = True
                            break
                if is_equal:
                    active_liq.append({"price": float(ph_vals[i]), "side": -1})

            # Equal lows
            if not np.isnan(pl_vals[i]):
                is_equal = False
                for j in range(1, 4):
                    idx = i - j * c.liq_swing_len
                    if idx >= 0 and not np.isnan(pl_vals[idx]):
                        base = pl_vals[i] if pl_vals[i] != 0 else np.nan
                        diff = abs(pl_vals[i] - pl_vals[idx]) / base * 100.0 if base and not np.isnan(base) else np.nan
                        if not np.isnan(diff) and diff <= c.liq_tolerance_pct:
                            is_equal = True
                            break
                if is_equal:
                    active_liq.append({"price": float(pl_vals[i]), "side": 1})

            if len(active_liq) > c.max_liq_levels:
                active_liq = active_liq[-c.max_liq_levels:]

            active_count[i] = len(active_liq)
            buy_count[i] = sum(1 for x in active_liq if x["side"] == 1)
            sell_count[i] = sum(1 for x in active_liq if x["side"] == -1)

            if buy_count[i] > sell_count[i]:
                liq_bias[i] = 1
            elif sell_count[i] > buy_count[i]:
                liq_bias[i] = -1
            else:
                liq_bias[i] = 0

            if active_liq:
                nearest = min(active_liq, key=lambda x: abs(close_vals[i] - x["price"]))
                nearest_price[i] = nearest["price"]
                if close_vals[i] != 0:
                    nearest_dist_pct[i] = abs(close_vals[i] - nearest["price"]) / close_vals[i] * 100.0

        out["active_liquidity_levels"] = active_count
        out["buy_liquidity_levels"] = buy_count
        out["sell_liquidity_levels"] = sell_count
        out["liquidity_bias"] = liq_bias
        out["nearest_liquidity_price"] = nearest_price
        out["nearest_liquidity_distance_pct"] = nearest_dist_pct

    def _calculate_premium_discount(self, out: pd.DataFrame) -> None:
        c = self.config

        out["range_high"] = out["high"].rolling(c.pd_array_len, min_periods=1).max()
        out["range_low"] = out["low"].rolling(c.pd_array_len, min_periods=1).min()
        out["range_eq"] = (out["range_high"] + out["range_low"]) / 2.0

        out["premium_75"] = out["range_eq"] + (out["range_high"] - out["range_eq"]) * 0.75
        out["premium_50"] = out["range_eq"] + (out["range_high"] - out["range_eq"]) * 0.50
        out["discount_50"] = out["range_eq"] - (out["range_eq"] - out["range_low"]) * 0.50
        out["discount_75"] = out["range_eq"] - (out["range_eq"] - out["range_low"]) * 0.75

        in_premium = (
            (out["close"] > out["range_eq"])
            & (out["close"] > (out["range_eq"] + (out["range_high"] - out["range_eq"]) * 0.5))
        )

        in_discount = (
            (out["close"] < out["range_eq"])
            & (out["close"] < (out["range_eq"] - (out["range_eq"] - out["range_low"]) * 0.5))
        )

        out["pd_position"] = np.select(
            [in_premium, in_discount],
            ["PREMIUM", "DISCOUNT"],
            default="EQUILIBRIUM",
        )

        out["pd_bias"] = np.select(
            [out["pd_position"] == "DISCOUNT", out["pd_position"] == "PREMIUM"],
            [1, -1],
            default=0,
        ).astype(int)

    def _calculate_breaker_mitigation(self, out: pd.DataFrame) -> None:
        # already populated inside order-block routine to preserve event timing
        pass

    def _calculate_confluence(self, out: pd.DataFrame) -> None:
        ob_score = np.select(
            [
                out["active_order_blocks"] > 3,
                out["active_order_blocks"] > 1,
            ],
            [25, 15],
            default=5,
        )

        fvg_score = np.select(
            [
                out["active_fvgs"] > 5,
                out["active_fvgs"] > 2,
            ],
            [25, 15],
            default=5,
        )

        liq_score = np.select(
            [
                out["active_liquidity_levels"] > 3,
                out["active_liquidity_levels"] > 1,
            ],
            [20, 10],
            default=0,
        )

        pd_score = np.where(
            (out["pd_position"] == "PREMIUM") | (out["pd_position"] == "DISCOUNT"),
            15,
            5,
        )

        atr_base = out["atr_sma50"].replace(0, np.nan)
        atr_ratio = (out["atr"] / atr_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        flow_score = np.select(
            [
                atr_ratio > 1.3,
                atr_ratio < 0.8,
            ],
            [15, 10],
            default=5,
        )

        out["confluence_score"] = (ob_score + fvg_score + liq_score + pd_score + flow_score).astype(int)

        out["confluence_level"] = np.select(
            [
                out["confluence_score"] >= 80,
                out["confluence_score"] >= 60,
                out["confluence_score"] >= 40,
            ],
            ["EXTREME", "STRONG", "MODERATE"],
            default="WEAK",
        )

        base_quality = (
            out["confluence_score"] / 100.0 * 0.60
            + out["bias_strength"].clip(0.0, 1.0) * 0.25
            + out["body_frac"].clip(0.0, 1.0) * 0.15
        )

        out["signal_quality"] = base_quality.clip(0.0, 1.0)

    # -------------------------------------------------------------------------
    # MTF
    # -------------------------------------------------------------------------
    def _compute_bias_score_series(self, df: pd.DataFrame) -> pd.Series:
        c = self.config

        bar_range = (df["high"] - df["low"]).clip(lower=0.0)
        body_frac = safe_ratio((df["close"] - df["open"]).abs(), bar_range, fallback=0.0).clip(0.0, 1.0)

        bull_effort = np.where(df["close"] > df["open"], df["volume"] * body_frac, 0.0)
        bear_effort = np.where(df["close"] < df["open"], df["volume"] * body_frac, 0.0)

        bull_effort_sm = ema(pd.Series(bull_effort, index=df.index), c.vol_len)
        bear_effort_sm = ema(pd.Series(bear_effort, index=df.index), c.vol_len)

        score = np.select(
            [
                bull_effort_sm > bear_effort_sm,
                bear_effort_sm > bull_effort_sm,
            ],
            [1.0, -1.0],
            default=0.0,
        )

        return pd.Series(score, index=df.index, dtype=float)

    def _compute_state_score_series(self, df: pd.DataFrame) -> pd.Series:
        c = self.config

        vol_sm = ema(df["volume"], c.vol_smooth_len)
        vol_base = sma(df["volume"], c.vol_len)
        vol_ratio = safe_ratio(vol_sm, vol_base, fallback=1.0)

        bar_range = (df["high"] - df["low"]).clip(lower=0.0)
        range_base = sma(bar_range, c.range_len)
        range_ratio = safe_ratio(bar_range, range_base, fallback=1.0)

        body_frac = safe_ratio((df["close"] - df["open"]).abs(), bar_range, fallback=0.0).clip(0.0, 1.0)

        is_climax = (
            ((vol_ratio >= c.climax_mult) & (range_ratio >= c.range_climax_mult))
            | ((vol_ratio >= c.climax_mult) & (body_frac <= c.small_body_thresh))
        )

        is_expansion = (
            (~is_climax)
            & (vol_ratio >= c.exp_mult)
            & (range_ratio >= c.range_expand_mult)
        )

        is_low = (
            (vol_ratio <= c.low_mult)
            & (range_ratio <= 1.0)
        )

        score = np.select(
            [is_climax, is_expansion, is_low],
            [1.0, 0.55, -0.55],
            default=0.0,
        )

        return pd.Series(score, index=df.index, dtype=float)

    def _calculate_mtf_average(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        weights = c.mtf_weights or {}
        rules = c.mtf_rules or {}

        total_weight = max(sum(float(v) for v in weights.values()), 0.0001)

        bias_parts: List[pd.Series] = []
        state_parts: List[pd.Series] = []

        for tf_key, rule in rules.items():
            weight = float(weights.get(tf_key, 0.0))

            if weight <= 0:
                bias_parts.append(pd.Series(0.0, index=df.index))
                state_parts.append(pd.Series(0.0, index=df.index))
                continue

            try:
                rs = resample_ohlcv(df, rule)

                if rs.empty:
                    bias_parts.append(pd.Series(0.0, index=df.index))
                    state_parts.append(pd.Series(0.0, index=df.index))
                    continue

                bias_score = self._compute_bias_score_series(rs)
                state_score = self._compute_state_score_series(rs)

                bias_aligned = bias_score.reindex(df.index, method="ffill").fillna(0.0)
                state_aligned = state_score.reindex(df.index, method="ffill").fillna(0.0)

                bias_parts.append(bias_aligned * weight)
                state_parts.append(state_aligned * weight)

            except Exception:
                bias_parts.append(pd.Series(0.0, index=df.index))
                state_parts.append(pd.Series(0.0, index=df.index))

        mtf_bias_avg = sum(bias_parts) / total_weight
        mtf_state_avg = sum(state_parts) / total_weight

        return pd.DataFrame(
            {
                "mtf_bias_avg": mtf_bias_avg,
                "mtf_state_avg": mtf_state_avg,
            },
            index=df.index,
        )

    # -------------------------------------------------------------------------
    # LATEST OUTPUT
    # -------------------------------------------------------------------------
    def latest(self, df: pd.DataFrame) -> VolumeOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        return VolumeOutput(
            vol_state=_safe_int(row["vol_state"]),
            vol_bias=_safe_int(row["vol_bias"]),
            vol_age=_safe_int(row["vol_age"]),

            vol_flip_export=_safe_int(row["vol_flip_export"]),
            vol_expand_export=_safe_int(row["vol_expand_export"]),
            vol_climax_export=_safe_int(row["vol_climax_export"]),
            vol_low_export=_safe_int(row["vol_low_export"]),
            vol_normal_export=_safe_int(row["vol_normal_export"]),

            entered_climax=bool(row["entered_climax"]),
            entered_expand=bool(row["entered_expand"]),
            entered_normal=bool(row["entered_normal"]),
            entered_low=bool(row["entered_low"]),

            bull_expand_pulse=bool(row["bull_expand_pulse"]),
            bear_expand_pulse=bool(row["bear_expand_pulse"]),
            bull_climax_pulse=bool(row["bull_climax_pulse"]),
            bear_climax_pulse=bool(row["bear_climax_pulse"]),

            vol_ratio=_safe_float(row["vol_ratio"]),
            range_ratio=_safe_float(row["range_ratio"]),
            body_frac=_safe_float(row["body_frac"]),
            close_pos=_safe_float(row["close_pos"]),
            bias_strength=_safe_float(row["bias_strength"]),

            raw_vol_state=_safe_int(row["raw_vol_state"]),
            raw_bias=_safe_int(row["raw_bias"]),

            mtf_bias_avg=_safe_float(row["mtf_bias_avg"]),
            mtf_state_avg=_safe_float(row["mtf_state_avg"]),
            mtf_bias_dir=_safe_int(row["mtf_bias_dir"]),
            mtf_bias_text=str(row["mtf_bias_text"]),
            mtf_state_text=str(row["mtf_state_text"]),

            orderflow_bias=_safe_int(row["orderflow_bias"]),
            orderflow_bias_text=str(row["orderflow_bias_text"]),
            orderflow_state=str(row["orderflow_state"]),

            volume_imbalance=bool(row["volume_imbalance"]),
            volume_imbalance_bull=bool(row["volume_imbalance_bull"]),
            volume_imbalance_bear=bool(row["volume_imbalance_bear"]),

            active_order_blocks=_safe_int(row["active_order_blocks"]),
            bullish_order_blocks=_safe_int(row["bullish_order_blocks"]),
            bearish_order_blocks=_safe_int(row["bearish_order_blocks"]),
            order_block_signal=_safe_int(row["order_block_signal"]),
            order_block_top=_safe_float(row["order_block_top"]),
            order_block_bottom=_safe_float(row["order_block_bottom"]),
            order_block_eq=_safe_float(row["order_block_eq"]),
            order_block_strength=str(row["order_block_strength"]),

            active_fvgs=_safe_int(row["active_fvgs"]),
            bullish_fvgs=_safe_int(row["bullish_fvgs"]),
            bearish_fvgs=_safe_int(row["bearish_fvgs"]),
            fvg_signal=_safe_int(row["fvg_signal"]),
            fvg_top=_safe_float(row["fvg_top"]),
            fvg_bottom=_safe_float(row["fvg_bottom"]),
            fvg_gap_pct=_safe_float(row["fvg_gap_pct"]),

            active_liquidity_levels=_safe_int(row["active_liquidity_levels"]),
            buy_liquidity_levels=_safe_int(row["buy_liquidity_levels"]),
            sell_liquidity_levels=_safe_int(row["sell_liquidity_levels"]),
            liquidity_bias=_safe_int(row["liquidity_bias"]),
            nearest_liquidity_price=_safe_float(row["nearest_liquidity_price"]),
            nearest_liquidity_distance_pct=_safe_float(row["nearest_liquidity_distance_pct"]),

            pd_position=str(row["pd_position"]),
            pd_bias=_safe_int(row["pd_bias"]),
            range_high=_safe_float(row["range_high"]),
            range_low=_safe_float(row["range_low"]),
            range_eq=_safe_float(row["range_eq"]),
            premium_50=_safe_float(row["premium_50"]),
            premium_75=_safe_float(row["premium_75"]),
            discount_50=_safe_float(row["discount_50"]),
            discount_75=_safe_float(row["discount_75"]),

            breaker_bullish=bool(row["breaker_bullish"]),
            breaker_bearish=bool(row["breaker_bearish"]),
            mitigation_bullish=bool(row["mitigation_bullish"]),
            mitigation_bearish=bool(row["mitigation_bearish"]),

            confluence_score=_safe_int(row["confluence_score"]),
            confluence_level=str(row["confluence_level"]),
            signal_quality=_safe_float(row["signal_quality"]),

            timestamp=calc.index[-1],
        )


# =============================================================================
# PUBLIC API
# =============================================================================

def run_volume_engine(
    df: pd.DataFrame,
    config: Optional[VolumeConfig] = None,
) -> Dict[str, Any]:
    engine = VolumeEngine(config=config)
    latest = engine.latest(df)
    return asdict(latest)


def run_volume_engine_full(
    df: pd.DataFrame,
    config: Optional[VolumeConfig] = None,
) -> pd.DataFrame:
    engine = VolumeEngine(config=config)
    return engine.calculate(df)


# =============================================================================
# WEBSITE PAYLOAD BUILDER
# =============================================================================

def build_volume_latest_payload(
    df: pd.DataFrame,
    config: Optional[VolumeConfig] = None,
) -> Dict[str, Any]:
    engine = VolumeEngine(config=config)
    latest = engine.latest(df)

    vol_state_label = {
        2: "CLIMAX",
        1: "EXPANSION",
        0: "NORMAL",
        -1: "LOW",
    }.get(latest.vol_state, "NORMAL")

    vol_bias_label = {
        1: "BULL",
        -1: "BEAR",
        0: "NEUTRAL",
    }.get(latest.vol_bias, "NEUTRAL")

    payload: Dict[str, Any] = {
        "indicator": "volume",
        "module": "m_volume",
        "version": "volume_orderflow_payload_v1",

        "timestamp": str(latest.timestamp),

        "summary": {
            "state": vol_state_label,
            "bias": vol_bias_label,
            "bias_strength": latest.bias_strength,
            "age": latest.vol_age,
            "signal_quality": latest.signal_quality,
        },

        "volume": {
            "state": latest.vol_state,
            "state_label": vol_state_label,
            "bias": latest.vol_bias,
            "bias_label": vol_bias_label,
            "age": latest.vol_age,
            "raw_state": latest.raw_vol_state,
            "raw_bias": latest.raw_bias,
            "vol_ratio": latest.vol_ratio,
            "range_ratio": latest.range_ratio,
            "body_frac": latest.body_frac,
            "close_pos": latest.close_pos,
            "bias_strength": latest.bias_strength,
            "entered_climax": latest.entered_climax,
            "entered_expand": latest.entered_expand,
            "entered_normal": latest.entered_normal,
            "entered_low": latest.entered_low,
            "bull_expand_pulse": latest.bull_expand_pulse,
            "bear_expand_pulse": latest.bear_expand_pulse,
            "bull_climax_pulse": latest.bull_climax_pulse,
            "bear_climax_pulse": latest.bear_climax_pulse,
            "exports": {
                "flip": latest.vol_flip_export,
                "expand": latest.vol_expand_export,
                "climax": latest.vol_climax_export,
                "low": latest.vol_low_export,
                "normal": latest.vol_normal_export,
            },
        },

        "orderflow": {
            "bias": latest.orderflow_bias,
            "bias_text": latest.orderflow_bias_text,
            "state": latest.orderflow_state,
            "volume_imbalance": latest.volume_imbalance,
            "volume_imbalance_bull": latest.volume_imbalance_bull,
            "volume_imbalance_bear": latest.volume_imbalance_bear,

            "order_blocks": {
                "active": latest.active_order_blocks,
                "bullish": latest.bullish_order_blocks,
                "bearish": latest.bearish_order_blocks,
                "signal": latest.order_block_signal,
                "top": latest.order_block_top,
                "bottom": latest.order_block_bottom,
                "equilibrium": latest.order_block_eq,
                "strength": latest.order_block_strength,
            },

            "fair_value_gaps": {
                "active": latest.active_fvgs,
                "bullish": latest.bullish_fvgs,
                "bearish": latest.bearish_fvgs,
                "signal": latest.fvg_signal,
                "top": latest.fvg_top,
                "bottom": latest.fvg_bottom,
                "gap_pct": latest.fvg_gap_pct,
            },

            "liquidity": {
                "active_levels": latest.active_liquidity_levels,
                "buy_levels": latest.buy_liquidity_levels,
                "sell_levels": latest.sell_liquidity_levels,
                "bias": latest.liquidity_bias,
                "nearest_price": latest.nearest_liquidity_price,
                "nearest_distance_pct": latest.nearest_liquidity_distance_pct,
            },

            "premium_discount": {
                "position": latest.pd_position,
                "bias": latest.pd_bias,
                "range_high": latest.range_high,
                "range_low": latest.range_low,
                "equilibrium": latest.range_eq,
                "premium_50": latest.premium_50,
                "premium_75": latest.premium_75,
                "discount_50": latest.discount_50,
                "discount_75": latest.discount_75,
            },

            "structure_events": {
                "breaker_bullish": latest.breaker_bullish,
                "breaker_bearish": latest.breaker_bearish,
                "mitigation_bullish": latest.mitigation_bullish,
                "mitigation_bearish": latest.mitigation_bearish,
            },

            "confluence": {
                "score": latest.confluence_score,
                "level": latest.confluence_level,
                "signal_quality": latest.signal_quality,
            },
        },

        "mtf": {
            "bias_avg": latest.mtf_bias_avg,
            "state_avg": latest.mtf_state_avg,
            "bias_dir": latest.mtf_bias_dir,
            "bias_text": latest.mtf_bias_text,
            "state_text": latest.mtf_state_text,
        },
    }

    return payload