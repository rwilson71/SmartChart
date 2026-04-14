"""
SmartChart Backend — j_fvg.py

FVG Engine v1.2 — Validation Rewrite
Backend rebuild from Pine logic.

Core features:
- 3-candle bullish / bearish FVG detection
- Wick-based or body-based gap mode
- Minimum gap % filter
- Impulse ATR filter
- Middle-candle body % filter
- EMA trend filter
- MTF trend average filter
- Zone storage with hold / expiry logic
- Retest / mitigation detection
- Freshness scoring
- Best active bull / bear zone scoring
- Final FVG direction, state, retest state, and strength score

Backend only:
- no TradingView visuals
- no boxes / labels / lines
- clean standalone output contract
- direct test block included
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class FVGConfig:
    # Core
    fvg_on: bool = True
    fvg_min_gap_pct: float = 0.05
    fvg_hold_bars: int = 60
    fvg_max_zones: int = 12
    fvg_use_wicks: bool = False
    fvg_touch_mode: str = "Any Touch"   # "Any Touch", "Midpoint", "Full Fill"

    # Filters
    use_impulse_filter: bool = True
    use_body_filter: bool = True
    use_trend_filter: bool = True
    use_mtf_filter: bool = True

    impulse_atr_len: int = 14
    impulse_min_atr: float = 0.60
    body_min_pct: float = 0.55

    # Retest / Freshness
    rt_score_on: bool = True
    fresh_bars_max: int = 25

    # MTF
    mtf_on: bool = True
    mtf_weights: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if self.mtf_weights is None:
            self.mtf_weights = {"tf1": 1.0, "tf2": 1.0, "tf3": 1.0}


# ==============================================================================
# HELPERS
# ==============================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


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
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def body_pct(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    rng = (h - l).clip(lower=1e-10)
    return (c - o).abs() / rng


def gap_pct(top: pd.Series, bot: pd.Series, ref: pd.Series) -> pd.Series:
    ref_safe = ref.replace(0.0, np.nan)
    out = ((top - bot).abs() / ref_safe.abs()) * 100.0
    return out.fillna(0.0)


def grade_txt(score: float) -> str:
    if score >= 0.85:
        return "A+"
    if score >= 0.65:
        return "B+"
    if score >= 0.45:
        return "B"
    if score >= 0.25:
        return "C"
    return "N"


def age_score(current_bar: int, born_bar: int, fresh_bars_max: int, hold_bars: int) -> float:
    age = current_bar - born_bar
    if age <= fresh_bars_max:
        return 1.0
    denom = max(hold_bars - fresh_bars_max, 1)
    return clamp(1.0 - ((age - fresh_bars_max) / denom), 0.0, 1.0)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("MTF resampling requires a DatetimeIndex.")

    out = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna()

    return out


def trend_dir_from_ema(df: pd.DataFrame) -> pd.Series:
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    e200 = ema(df["close"], 200)

    return pd.Series(
        np.select(
            [(e20 > e50) & (e50 > e200), (e20 < e50) & (e50 < e200)],
            [1, -1],
            default=0,
        ),
        index=df.index,
        dtype=float,
    )


# ==============================================================================
# ZONE MODEL
# ==============================================================================

@dataclass
class FVGZone:
    side: str                 # "bull" or "bear"
    top: float
    bot: float
    mid: float
    born_bar: int
    touched: bool
    active: bool
    base_strength: float


# ==============================================================================
# OUTPUT CONTRACT
# ==============================================================================

@dataclass
class FVGOutput:
    fvg_dir: int
    fvg_strength_score: float
    fvg_mtf_avg: float
    fvg_state: int
    fvg_retest_state: int

    bull_active_now: bool
    bear_active_now: bool
    bull_best_score: float
    bear_best_score: float
    bull_retest_ready: bool
    bear_retest_ready: bool

    bull_fvg_found: bool
    bear_fvg_found: bool
    bull_raw: bool
    bear_raw: bool

    bull_gap_pct: float
    bear_gap_pct: float
    mid_body_pct: float
    mid_impulse: float

    bull_base_strength: float
    bear_base_strength: float

    trend_dir: int
    timestamp: Any


# ==============================================================================
# CORE ENGINE
# ==============================================================================

class FVGEngine:
    def __init__(self, config: Optional[FVGConfig] = None) -> None:
        self.config = config or FVGConfig()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv(df)
        c = self.config

        out = df.copy().reset_index().rename(columns={"index": "_timestamp"})
        out["_bar"] = np.arange(len(out))

        # ----------------------------------------------------------------------
        # Core series
        # ----------------------------------------------------------------------
        out["ema20"] = ema(out["close"], 20)
        out["ema50"] = ema(out["close"], 50)
        out["ema200"] = ema(out["close"], 200)
        out["atr_val"] = atr(out[["open", "high", "low", "close", "volume"]], c.impulse_atr_len)

        out["trend_dir"] = np.select(
            [
                (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"]),
                (out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"]),
            ],
            [1, -1],
            default=0,
        ).astype(int)

        # ----------------------------------------------------------------------
        # MTF average
        # ----------------------------------------------------------------------
        out["fvg_mtf_avg"] = 0.0
        if c.mtf_on:
            mtf_df = self._calculate_mtf_average(df)
            out["fvg_mtf_avg"] = mtf_df["fvg_mtf_avg"].values

        out["mtf_bull_ok"] = (~c.use_mtf_filter) | (out["fvg_mtf_avg"] > 0.20)
        out["mtf_bear_ok"] = (~c.use_mtf_filter) | (out["fvg_mtf_avg"] < -0.20)

        # ----------------------------------------------------------------------
        # Detection series
        # ----------------------------------------------------------------------
        if c.fvg_use_wicks:
            out["bull_gap_top"] = out["low"]
            out["bull_gap_bot"] = out["high"].shift(2)

            out["bear_gap_top"] = out["low"].shift(2)
            out["bear_gap_bot"] = out["high"]
        else:
            out["bull_gap_top"] = np.minimum(out["open"], out["close"])
            out["bull_gap_bot"] = np.maximum(out["open"].shift(2), out["close"].shift(2))

            out["bear_gap_top"] = np.minimum(out["open"].shift(2), out["close"].shift(2))
            out["bear_gap_bot"] = np.maximum(out["open"], out["close"])

        out["bull_raw"] = c.fvg_on & (out["_bar"] >= 2) & (out["bull_gap_top"] > out["bull_gap_bot"])
        out["bear_raw"] = c.fvg_on & (out["_bar"] >= 2) & (out["bear_gap_bot"] < out["bear_gap_top"])

        out["mid_body_pct"] = body_pct(
            out["open"].shift(1),
            out["high"].shift(1),
            out["low"].shift(1),
            out["close"].shift(1),
        )

        out["mid_impulse"] = np.where(
            out["atr_val"] > 0,
            (out["close"].shift(1) - out["open"].shift(1)).abs() / out["atr_val"],
            0.0,
        )

        out["bull_gap_pct"] = np.where(
            out["bull_raw"],
            gap_pct(out["bull_gap_top"], out["bull_gap_bot"], out["close"]),
            0.0,
        )
        out["bear_gap_pct"] = np.where(
            out["bear_raw"],
            gap_pct(out["bear_gap_top"], out["bear_gap_bot"], out["close"]),
            0.0,
        )

        out["bull_gap_ok"] = out["bull_raw"] & (out["bull_gap_pct"] >= c.fvg_min_gap_pct)
        out["bear_gap_ok"] = out["bear_raw"] & (out["bear_gap_pct"] >= c.fvg_min_gap_pct)

        out["body_ok"] = (~c.use_body_filter) | (out["mid_body_pct"] >= c.body_min_pct)
        out["impulse_ok"] = (~c.use_impulse_filter) | (out["mid_impulse"] >= c.impulse_min_atr)
        out["trend_bull_ok"] = (~c.use_trend_filter) | (out["trend_dir"] == 1)
        out["trend_bear_ok"] = (~c.use_trend_filter) | (out["trend_dir"] == -1)

        out["bull_fvg_found"] = (
            out["bull_gap_ok"]
            & out["body_ok"]
            & out["impulse_ok"]
            & out["trend_bull_ok"]
            & out["mtf_bull_ok"]
        )

        out["bear_fvg_found"] = (
            out["bear_gap_ok"]
            & out["body_ok"]
            & out["impulse_ok"]
            & out["trend_bear_ok"]
            & out["mtf_bear_ok"]
        )

        # ----------------------------------------------------------------------
        # Detection strength score
        # ----------------------------------------------------------------------
        bull_gap_div = max(c.fvg_min_gap_pct * 2.5, 1e-10)
        bear_gap_div = max(c.fvg_min_gap_pct * 2.5, 1e-10)
        impulse_div = max(c.impulse_min_atr * 1.5, 1e-10)

        out["bull_gap_score"] = np.where(
            out["bull_raw"],
            np.clip(out["bull_gap_pct"] / bull_gap_div, 0.0, 1.0),
            0.0,
        )
        out["bear_gap_score"] = np.where(
            out["bear_raw"],
            np.clip(out["bear_gap_pct"] / bear_gap_div, 0.0, 1.0),
            0.0,
        )
        out["body_score"] = np.clip(out["mid_body_pct"], 0.0, 1.0)
        out["impulse_score"] = np.clip(out["mid_impulse"] / impulse_div, 0.0, 1.0)

        out["bull_trend_score"] = np.select(
            [out["trend_dir"] == 1, out["trend_dir"] == 0],
            [1.0, 0.4],
            default=0.0,
        )
        out["bear_trend_score"] = np.select(
            [out["trend_dir"] == -1, out["trend_dir"] == 0],
            [1.0, 0.4],
            default=0.0,
        )

        out["bull_mtf_score"] = np.clip((out["fvg_mtf_avg"] + 1.0) / 2.0, 0.0, 1.0)
        out["bear_mtf_score"] = np.clip((1.0 - out["fvg_mtf_avg"]) / 2.0, 0.0, 1.0)

        out["bull_base_strength"] = np.clip(
            (out["bull_gap_score"] * 0.35)
            + (out["body_score"] * 0.20)
            + (out["impulse_score"] * 0.20)
            + (out["bull_trend_score"] * 0.10)
            + (out["bull_mtf_score"] * 0.15),
            0.0,
            1.0,
        )

        out["bear_base_strength"] = np.clip(
            (out["bear_gap_score"] * 0.35)
            + (out["body_score"] * 0.20)
            + (out["impulse_score"] * 0.20)
            + (out["bear_trend_score"] * 0.10)
            + (out["bear_mtf_score"] * 0.15),
            0.0,
            1.0,
        )

        # ----------------------------------------------------------------------
        # Zone engine
        # ----------------------------------------------------------------------
        bull_active_now_list: List[bool] = []
        bear_active_now_list: List[bool] = []
        bull_best_score_list: List[float] = []
        bear_best_score_list: List[float] = []
        bull_retest_ready_list: List[bool] = []
        bear_retest_ready_list: List[bool] = []
        fvg_dir_list: List[int] = []
        fvg_strength_score_list: List[float] = []
        fvg_state_list: List[int] = []
        fvg_retest_state_list: List[int] = []

        bull_zones: List[FVGZone] = []
        bear_zones: List[FVGZone] = []

        for i in range(len(out)):
            row = out.iloc[i]

            # --------------------------------------------------------------
            # Add bull zone
            # --------------------------------------------------------------
            if bool(row["bull_fvg_found"]):
                z_top = float(row["bull_gap_top"])
                z_bot = float(row["bull_gap_bot"])
                z_mid = (z_top + z_bot) * 0.5
                z_str = float(row["bull_base_strength"])

                bull_zones.insert(
                    0,
                    FVGZone(
                        side="bull",
                        top=z_top,
                        bot=z_bot,
                        mid=z_mid,
                        born_bar=i,
                        touched=False,
                        active=True,
                        base_strength=z_str,
                    ),
                )

            # --------------------------------------------------------------
            # Add bear zone
            # --------------------------------------------------------------
            if bool(row["bear_fvg_found"]):
                z_top = float(row["bear_gap_top"])
                z_bot = float(row["bear_gap_bot"])
                z_mid = (z_top + z_bot) * 0.5
                z_str = float(row["bear_base_strength"])

                bear_zones.insert(
                    0,
                    FVGZone(
                        side="bear",
                        top=z_top,
                        bot=z_bot,
                        mid=z_mid,
                        born_bar=i,
                        touched=False,
                        active=True,
                        base_strength=z_str,
                    ),
                )

            # --------------------------------------------------------------
            # Limit arrays
            # --------------------------------------------------------------
            bull_zones = bull_zones[: c.fvg_max_zones]
            bear_zones = bear_zones[: c.fvg_max_zones]

            low_i = float(row["low"])
            high_i = float(row["high"])

            # --------------------------------------------------------------
            # Update bull zones
            # --------------------------------------------------------------
            for z in bull_zones:
                expired = (i - z.born_bar) > c.fvg_hold_bars
                hit = self._bull_touched(c.fvg_touch_mode, low_i, high_i, z.top, z.bot, z.mid)

                if z.active and hit:
                    z.touched = True
                    z.active = False

                if expired:
                    z.active = False

            # --------------------------------------------------------------
            # Update bear zones
            # --------------------------------------------------------------
            for z in bear_zones:
                expired = (i - z.born_bar) > c.fvg_hold_bars
                hit = self._bear_touched(c.fvg_touch_mode, low_i, high_i, z.top, z.bot, z.mid)

                if z.active and hit:
                    z.touched = True
                    z.active = False

                if expired:
                    z.active = False

            # --------------------------------------------------------------
            # Current state
            # --------------------------------------------------------------
            bull_active_now = False
            bear_active_now = False
            bull_best_score = 0.0
            bear_best_score = 0.0
            bull_retest_ready = False
            bear_retest_ready = False

            for z in bull_zones:
                if z.active:
                    age_sc = age_score(i, z.born_bar, c.fresh_bars_max, c.fvg_hold_bars) if c.rt_score_on else 1.0
                    final_sc = clamp((z.base_strength * 0.75) + (age_sc * 0.25), 0.0, 1.0)
                    bull_active_now = True
                    bull_best_score = max(bull_best_score, final_sc)
                    if not z.touched:
                        bull_retest_ready = True

            for z in bear_zones:
                if z.active:
                    age_sc = age_score(i, z.born_bar, c.fresh_bars_max, c.fvg_hold_bars) if c.rt_score_on else 1.0
                    final_sc = clamp((z.base_strength * 0.75) + (age_sc * 0.25), 0.0, 1.0)
                    bear_active_now = True
                    bear_best_score = max(bear_best_score, final_sc)
                    if not z.touched:
                        bear_retest_ready = True

            fvg_dir = (
                1 if bull_active_now and not bear_active_now else
                -1 if bear_active_now and not bull_active_now else
                1 if bull_best_score > bear_best_score else
                -1 if bear_best_score > bull_best_score else
                0
            )

            fvg_strength_score = max(bull_best_score, bear_best_score)

            fvg_state = (
                2 if bull_active_now and bear_active_now else
                1 if bull_active_now or bear_active_now else
                0
            )

            fvg_retest_state = (
                1 if bull_retest_ready and not bear_retest_ready else
                -1 if bear_retest_ready and not bull_retest_ready else
                2 if bull_retest_ready and bear_retest_ready else
                0
            )

            bull_active_now_list.append(bull_active_now)
            bear_active_now_list.append(bear_active_now)
            bull_best_score_list.append(bull_best_score)
            bear_best_score_list.append(bear_best_score)
            bull_retest_ready_list.append(bull_retest_ready)
            bear_retest_ready_list.append(bear_retest_ready)
            fvg_dir_list.append(fvg_dir)
            fvg_strength_score_list.append(fvg_strength_score)
            fvg_state_list.append(fvg_state)
            fvg_retest_state_list.append(fvg_retest_state)

        out["bull_active_now"] = bull_active_now_list
        out["bear_active_now"] = bear_active_now_list
        out["bull_best_score"] = bull_best_score_list
        out["bear_best_score"] = bear_best_score_list
        out["bull_retest_ready"] = bull_retest_ready_list
        out["bear_retest_ready"] = bear_retest_ready_list
        out["fvg_dir"] = fvg_dir_list
        out["fvg_strength_score"] = fvg_strength_score_list
        out["fvg_state"] = fvg_state_list
        out["fvg_retest_state"] = fvg_retest_state_list

        # ----------------------------------------------------------------------
        # SmartChart output contract
        # ----------------------------------------------------------------------
        out["sc_fvg_dir"] = out["fvg_dir"].astype(int)
        out["sc_fvg_strength_score"] = out["fvg_strength_score"].astype(float)
        out["sc_fvg_mtf_avg"] = out["fvg_mtf_avg"].astype(float)
        out["sc_fvg_state"] = out["fvg_state"].astype(int)
        out["sc_fvg_retest_state"] = out["fvg_retest_state"].astype(int)

        out["sc_fvg_bull_active_now"] = out["bull_active_now"].astype(int)
        out["sc_fvg_bear_active_now"] = out["bear_active_now"].astype(int)
        out["sc_fvg_bull_best_score"] = out["bull_best_score"].astype(float)
        out["sc_fvg_bear_best_score"] = out["bear_best_score"].astype(float)
        out["sc_fvg_bull_retest_ready"] = out["bull_retest_ready"].astype(int)
        out["sc_fvg_bear_retest_ready"] = out["bear_retest_ready"].astype(int)

        out["sc_fvg_bull_found"] = out["bull_fvg_found"].astype(int)
        out["sc_fvg_bear_found"] = out["bear_fvg_found"].astype(int)
        out["sc_fvg_bull_raw"] = out["bull_raw"].astype(int)
        out["sc_fvg_bear_raw"] = out["bear_raw"].astype(int)

        out["sc_fvg_bull_gap_pct"] = out["bull_gap_pct"].astype(float)
        out["sc_fvg_bear_gap_pct"] = out["bear_gap_pct"].astype(float)
        out["sc_fvg_mid_body_pct"] = out["mid_body_pct"].fillna(0.0).astype(float)
        out["sc_fvg_mid_impulse"] = out["mid_impulse"].fillna(0.0).astype(float)

        out["sc_fvg_bull_base_strength"] = out["bull_base_strength"].astype(float)
        out["sc_fvg_bear_base_strength"] = out["bear_base_strength"].astype(float)

        out["sc_fvg_trend_dir"] = out["trend_dir"].astype(int)

        # Truth-style convenience fields
        out["fvg_valid"] = ((out["sc_fvg_bull_active_now"] == 1) | (out["sc_fvg_bear_active_now"] == 1)).astype(int)
        out["fvg_dir_simple"] = out["sc_fvg_dir"].astype(int)

        # restore original index
        out = out.set_index("_timestamp")
        return out

    def _bull_touched(self, mode: str, low: float, high: float, top: float, bot: float, mid: float) -> bool:
        if mode == "Any Touch":
            return low <= top and high >= bot
        if mode == "Midpoint":
            return low <= mid
        if mode == "Full Fill":
            return low <= bot
        return False

    def _bear_touched(self, mode: str, low: float, high: float, top: float, bot: float, mid: float) -> bool:
        if mode == "Any Touch":
            return high >= bot and low <= top
        if mode == "Midpoint":
            return high >= mid
        if mode == "Full Fill":
            return high >= top
        return False

    def _calculate_mtf_average(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame({"fvg_mtf_avg": pd.Series(0.0, index=df.index)})

        tf_rules = {
            "tf1": "15min",
            "tf2": "1H",
            "tf3": "4H",
        }
        weights = self.config.mtf_weights or {}
        total_weight = max(sum(weights.values()), 0.0001)

        aligned_dirs: List[pd.Series] = []

        for tf_key, rule in tf_rules.items():
            w = float(weights.get(tf_key, 0.0))
            if w <= 0:
                aligned_dirs.append(pd.Series(0.0, index=df.index))
                continue

            try:
                resampled = resample_ohlcv(df, rule)
                dir_series = trend_dir_from_ema(resampled)
                aligned = dir_series.reindex(df.index, method="ffill").fillna(0.0)
                aligned_dirs.append(aligned * w)
            except Exception:
                aligned_dirs.append(pd.Series(0.0, index=df.index))

        fvg_mtf_avg = sum(aligned_dirs) / total_weight
        return pd.DataFrame({"fvg_mtf_avg": fvg_mtf_avg}, index=df.index)

    def latest(self, df: pd.DataFrame) -> FVGOutput:
        calc = self.calculate(df)
        row = calc.iloc[-1]

        return FVGOutput(
            fvg_dir=int(row["fvg_dir"]),
            fvg_strength_score=float(row["fvg_strength_score"]),
            fvg_mtf_avg=float(row["fvg_mtf_avg"]),
            fvg_state=int(row["fvg_state"]),
            fvg_retest_state=int(row["fvg_retest_state"]),
            bull_active_now=bool(row["bull_active_now"]),
            bear_active_now=bool(row["bear_active_now"]),
            bull_best_score=float(row["bull_best_score"]),
            bear_best_score=float(row["bear_best_score"]),
            bull_retest_ready=bool(row["bull_retest_ready"]),
            bear_retest_ready=bool(row["bear_retest_ready"]),
            bull_fvg_found=bool(row["bull_fvg_found"]),
            bear_fvg_found=bool(row["bear_fvg_found"]),
            bull_raw=bool(row["bull_raw"]),
            bear_raw=bool(row["bear_raw"]),
            bull_gap_pct=float(row["bull_gap_pct"]),
            bear_gap_pct=float(row["bear_gap_pct"]),
            mid_body_pct=float(row["mid_body_pct"]) if not pd.isna(row["mid_body_pct"]) else 0.0,
            mid_impulse=float(row["mid_impulse"]) if not pd.isna(row["mid_impulse"]) else 0.0,
            bull_base_strength=float(row["bull_base_strength"]),
            bear_base_strength=float(row["bear_base_strength"]),
            trend_dir=int(row["trend_dir"]),
            timestamp=calc.index[-1],
        )


# ==============================================================================
# PUBLIC API
# ==============================================================================

def run_fvg_engine(df: pd.DataFrame, config: Optional[FVGConfig] = None) -> pd.DataFrame:
    engine = FVGEngine(config=config)
    return engine.calculate(df)


# ==============================================================================
# DIRECT TEST BLOCK
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    periods = 600
    idx = pd.date_range("2026-01-01", periods=periods, freq="5min")

    base = 100 + np.cumsum(np.random.normal(0, 0.18, periods))
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])

    high = np.maximum(open_, close) + np.random.uniform(0.03, 0.28, periods)
    low = np.minimum(open_, close) - np.random.uniform(0.03, 0.28, periods)

    for k in [80, 160, 280, 420]:
        if k + 2 < periods:
            low.iloc[k + 2] = high.iloc[k] + 0.20
            open_.iloc[k + 1] = close.iloc[k] + 0.35
            close.iloc[k + 1] = open_.iloc[k + 1] + 0.45
            high.iloc[k + 1] = close.iloc[k + 1] + 0.10
            low.iloc[k + 1] = open_.iloc[k + 1] - 0.05

    for k in [120, 240, 360, 520]:
        if k + 2 < periods:
            high.iloc[k + 2] = low.iloc[k] - 0.20
            open_.iloc[k + 1] = close.iloc[k] - 0.35
            close.iloc[k + 1] = open_.iloc[k + 1] - 0.45
            low.iloc[k + 1] = close.iloc[k + 1] - 0.10
            high.iloc[k + 1] = open_.iloc[k + 1] + 0.05

    volume = np.random.randint(100, 1500, periods)

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

    config = FVGConfig(
        fvg_on=True,
        fvg_min_gap_pct=0.05,
        fvg_hold_bars=60,
        fvg_max_zones=12,
        fvg_use_wicks=False,
        fvg_touch_mode="Any Touch",
        use_impulse_filter=True,
        use_body_filter=True,
        use_trend_filter=True,
        use_mtf_filter=True,
        impulse_atr_len=14,
        impulse_min_atr=0.60,
        body_min_pct=0.55,
        rt_score_on=True,
        fresh_bars_max=25,
        mtf_on=True,
        mtf_weights={"tf1": 1.0, "tf2": 1.0, "tf3": 1.0},
    )

    engine = FVGEngine(config=config)
    full = engine.calculate(test_df)
    latest = engine.latest(test_df)

    print("\n=== SMARTCHART FVG ENGINE TEST ===")
    print(
        full[
            [
                "sc_fvg_dir",
                "sc_fvg_strength_score",
                "sc_fvg_mtf_avg",
                "sc_fvg_state",
                "sc_fvg_retest_state",
                "sc_fvg_bull_active_now",
                "sc_fvg_bear_active_now",
                "sc_fvg_bull_best_score",
                "sc_fvg_bear_best_score",
                "sc_fvg_bull_found",
                "sc_fvg_bear_found",
            ]
        ].tail(15)
    )

    print("\n=== LATEST OUTPUT ===")
    print(asdict(latest))