"""
SmartChart Backend — m_volume.py

Volume Module v2 • MTF Avg
Backend parity rebuild from Pine logic.

Core features:
- Smoothed volume / baseline volume ratio
- Range participation ratio
- Candle body participation analysis
- Bull / bear directional effort
- Volume state classification:
    2  = climax
    1  = expansion
    0  = normal
   -1  = low
- Memory / hold logic for state + bias
- Expansion / climax / low event pulses
- MTF bias average
- MTF state average
- Clean export contract
- Direct test block included

Backend only:
- no TradingView visuals
- no labels / plots / alerts
- clean standalone output contract
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class VolumeConfig:
    # Core
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


def safe_ratio(numerator: pd.Series, denominator: pd.Series, fallback: float = 0.0) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    out = numerator / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(fallback)


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


# ==============================================================================
# OUTPUT CONTRACT
# ==============================================================================

@dataclass
class VolumeOutput:
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

    timestamp: Any


# ==============================================================================
# CORE ENGINE
# ==============================================================================

class VolumeEngine:
    def __init__(self, config: Optional[VolumeConfig] = None) -> None:
        self.config = config or VolumeConfig()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("VolumeEngine requires a DatetimeIndex.")

        c = self.config
        out = df.copy()

        # ----------------------------------------------------------------------
        # Core series
        # ----------------------------------------------------------------------
        out["vol_sm"] = ema(out["volume"], c.vol_smooth_len)
        out["vol_base"] = sma(out["volume"], c.vol_len)
        out["vol_ratio0"] = safe_ratio(out["vol_sm"], out["vol_base"], fallback=1.0)
        out["vol_ratio"] = out["vol_ratio0"].clip(0.0, 10.0)

        out["bar_range"] = (out["high"] - out["low"]).clip(lower=0.0)
        out["range_base"] = sma(out["bar_range"], c.range_len)
        out["range_ratio0"] = safe_ratio(out["bar_range"], out["range_base"], fallback=1.0)
        out["range_ratio"] = out["range_ratio0"].clip(0.0, 10.0)

        out["body_size"] = (out["close"] - out["open"]).abs()
        out["body_frac0"] = safe_ratio(out["body_size"], out["bar_range"], fallback=0.0)
        out["body_frac"] = out["body_frac0"].clip(0.0, 1.0)

        out["close_pos0"] = safe_ratio(out["close"] - out["low"], out["bar_range"], fallback=0.5)
        out["close_pos"] = out["close_pos0"].clip(0.0, 1.0)

        out["is_bull_bar"] = out["close"] > out["open"]
        out["is_bear_bar"] = out["close"] < out["open"]
        out["is_small_body"] = out["body_frac"] <= c.small_body_thresh

        # ----------------------------------------------------------------------
        # Directional participation
        # ----------------------------------------------------------------------
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
        )
        out["bias_strength"] = out["bias_strength_raw"].clip(0.0, 1.0)

        # ----------------------------------------------------------------------
        # Raw classification
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # Memory helpers
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # Memory engine
        # ----------------------------------------------------------------------
        vol_state_list: List[int] = []
        vol_bias_list: List[int] = []
        vol_age_list: List[int] = []

        current_state = int(out["raw_vol_state"].iat[0])
        current_bias = int(out["raw_bias"].iat[0])
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

        # ----------------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------------
        prev_state = out["vol_state"].shift(1).fillna(out["vol_state"])
        out["state_changed"] = out["vol_state"] != prev_state
        out["bias_changed"] = out["vol_bias"] != out["vol_bias"].shift(1).fillna(out["vol_bias"])

        out["entered_climax"] = out["state_changed"] & (out["vol_state"] == 2)
        out["entered_expand"] = out["state_changed"] & (out["vol_state"] == 1)
        out["entered_normal"] = out["state_changed"] & (out["vol_state"] == 0)
        out["entered_low"] = out["state_changed"] & (out["vol_state"] == -1)

        out["bull_expand_pulse"] = out["entered_expand"] & (out["vol_bias"] == 1)
        out["bear_expand_pulse"] = out["entered_expand"] & (out["vol_bias"] == -1)

        out["bull_climax_pulse"] = out["entered_climax"] & (out["vol_bias"] == 1)
        out["bear_climax_pulse"] = out["entered_climax"] & (out["vol_bias"] == -1)

        # ----------------------------------------------------------------------
        # Export-ready fields
        # ----------------------------------------------------------------------
        out["vol_state_export"] = out["vol_state"]
        out["vol_bias_export"] = out["vol_bias"]
        out["vol_flip_export"] = np.where(out["state_changed"], 1, 0).astype(int)
        out["vol_expand_export"] = np.where(out["entered_expand"], 1, 0).astype(int)
        out["vol_climax_export"] = np.where(out["entered_climax"], 1, 0).astype(int)
        out["vol_low_export"] = np.where(out["entered_low"], 1, 0).astype(int)
        out["vol_normal_export"] = np.where(out["entered_normal"], 1, 0).astype(int)

        # ----------------------------------------------------------------------
        # MTF layer
        # ----------------------------------------------------------------------
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

        return out

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

    def latest(self, df: pd.DataFrame) -> VolumeOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        return VolumeOutput(
            vol_state=int(row["vol_state"]),
            vol_bias=int(row["vol_bias"]),
            vol_age=int(row["vol_age"]),

            vol_flip_export=int(row["vol_flip_export"]),
            vol_expand_export=int(row["vol_expand_export"]),
            vol_climax_export=int(row["vol_climax_export"]),
            vol_low_export=int(row["vol_low_export"]),
            vol_normal_export=int(row["vol_normal_export"]),

            entered_climax=bool(row["entered_climax"]),
            entered_expand=bool(row["entered_expand"]),
            entered_normal=bool(row["entered_normal"]),
            entered_low=bool(row["entered_low"]),

            bull_expand_pulse=bool(row["bull_expand_pulse"]),
            bear_expand_pulse=bool(row["bear_expand_pulse"]),
            bull_climax_pulse=bool(row["bull_climax_pulse"]),
            bear_climax_pulse=bool(row["bear_climax_pulse"]),

            vol_ratio=float(row["vol_ratio"]),
            range_ratio=float(row["range_ratio"]),
            body_frac=float(row["body_frac"]),
            close_pos=float(row["close_pos"]),
            bias_strength=float(row["bias_strength"]),

            raw_vol_state=int(row["raw_vol_state"]),
            raw_bias=int(row["raw_bias"]),

            mtf_bias_avg=float(row["mtf_bias_avg"]),
            mtf_state_avg=float(row["mtf_state_avg"]),
            mtf_bias_dir=int(row["mtf_bias_dir"]),
            mtf_bias_text=str(row["mtf_bias_text"]),
            mtf_state_text=str(row["mtf_state_text"]),

            timestamp=calc.index[-1],
        )


# ==============================================================================
# PUBLIC API
# ==============================================================================

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


# ==============================================================================
# DIRECT TEST BLOCK
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    periods = 1200
    idx = pd.date_range("2026-01-01", periods=periods, freq="5min")

    base = 100 + np.cumsum(np.random.normal(0, 0.14, periods))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])

    high = np.maximum(open_, close) + np.random.uniform(0.02, 0.22, periods)
    low = np.minimum(open_, close) - np.random.uniform(0.02, 0.22, periods)
    volume = np.random.randint(100, 1200, periods).astype(float)

    # Synthetic bullish participation bursts
    for k in [120, 280, 540, 820]:
        if k + 6 < periods:
            volume[k:k + 6] *= 2.2
            close.iloc[k:k + 6] += np.linspace(0.1, 0.8, 6)
            high.iloc[k:k + 6] = np.maximum(high.iloc[k:k + 6], close.iloc[k:k + 6] + 0.08)

    # Synthetic bearish participation bursts
    for k in [210, 430, 700, 980]:
        if k + 5 < periods:
            volume[k:k + 5] *= 2.0
            close.iloc[k:k + 5] -= np.linspace(0.1, 0.7, 5)
            low.iloc[k:k + 5] = np.minimum(low.iloc[k:k + 5], close.iloc[k:k + 5] - 0.08)

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

    config = VolumeConfig(
        vol_len=20,
        vol_smooth_len=8,
        range_len=20,
        exp_mult=1.20,
        climax_mult=1.80,
        low_mult=0.80,
        range_expand_mult=1.10,
        range_climax_mult=1.50,
        small_body_thresh=0.35,
        confirm_bars=2,
        hold_bars=3,
        bias_refresh_bars=1,
        mtf_on=True,
        mtf_weights={"tf1": 1.0, "tf2": 1.0, "tf3": 1.0, "tf4": 1.0},
        mtf_rules={"tf1": "15min", "tf2": "1h", "tf3": "4h", "tf4": "1d"},
    )

    engine = VolumeEngine(config=config)
    full = engine.calculate(test_df)
    latest = engine.latest(test_df)

    print("\n=== SMARTCHART VOLUME ENGINE TEST ===")
    print(
        full[
            [
                "vol_state",
                "vol_bias",
                "vol_age",
                "entered_expand",
                "entered_climax",
                "entered_low",
                "bull_expand_pulse",
                "bear_expand_pulse",
                "bull_climax_pulse",
                "bear_climax_pulse",
                "vol_ratio",
                "range_ratio",
                "body_frac",
                "close_pos",
                "bias_strength",
                "mtf_bias_avg",
                "mtf_state_avg",
                "mtf_bias_text",
                "mtf_state_text",
            ]
        ].tail(15)
    )

    print("\n=== LATEST OUTPUT ===")
    print(asdict(latest))