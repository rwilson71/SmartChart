from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np
import pandas as pd


# =============================================================================
# SMARTCHART AI RSI ENGINE v2
# LEVEL 2 CLEAN REBUILD
# PINE AUTHORITY LOGIC + WEBSITE JSON CONTRACT
# =============================================================================
# Locked build order:
# helpers -> core engine -> base state -> MTF confirm-only layer
# -> export fields -> payload builder
#
# Rules:
# - Pine logic remains the authority
# - MTF is confirm-only
# - Core file is the single source of truth
# - Payload builder lives inside this module
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class AiRsiConfig:
    rsi_len: int = 14
    sig_len: int = 20
    learn_len: int = 20

    dead_zone_on: bool = True
    dead_zone_val: float = 0.15

    mtf_on: bool = True
    tf1: str = "5"
    tf2: str = "15"
    tf3: str = "60"
    tf4: str = "240"
    tf5: str = "D"


# =============================================================================
# HELPERS
# =============================================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AI RSI engine missing required columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("AI RSI engine requires a DatetimeIndex for MTF parity.")


def _series(src: pd.Series | np.ndarray | list[float], index=None) -> pd.Series:
    s = pd.Series(src, copy=False)
    if index is not None and not s.index.equals(index):
        s = pd.Series(s.to_numpy(), index=index, dtype=float)
    return s.astype(float)


def _sma(src: pd.Series, length: int) -> pd.Series:
    src = _series(src)
    return src.rolling(length, min_periods=length).mean()


def _stdev(src: pd.Series, length: int) -> pd.Series:
    src = _series(src)
    return src.rolling(length, min_periods=length).std(ddof=0)


def _rma(src: pd.Series, length: int) -> pd.Series:
    src = _series(src)
    out = pd.Series(np.nan, index=src.index, dtype=float)

    if length <= 0 or src.empty:
        return out

    first_valid = src.first_valid_index()
    if first_valid is None:
        return out

    first_pos = src.index.get_loc(first_valid)
    seed_end = first_pos + length
    if seed_end > len(src):
        return out

    seed = src.iloc[first_pos:seed_end].mean()
    out.iloc[seed_end - 1] = seed

    alpha = 1.0 / float(length)
    for i in range(seed_end, len(src)):
        prev = out.iloc[i - 1]
        curr = src.iloc[i]
        out.iloc[i] = prev + alpha * (curr - prev)

    return out


def _rsi(close: pd.Series, length: int) -> pd.Series:
    close = _series(close)
    delta = close.diff()

    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    avg_up = _rma(up, length)
    avg_down = _rma(down, length)

    rs = avg_up / avg_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.mask((avg_down == 0) & (avg_up > 0), 100.0)
    rsi = rsi.mask((avg_up == 0) & (avg_down > 0), 0.0)
    rsi = rsi.mask((avg_up == 0) & (avg_down == 0), 0.0)

    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    high = _series(high)
    low = _series(low)
    close = _series(close)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return _rma(tr, length)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _series(a)
    b = _series(b, index=a.index)

    denom = b.fillna(a)
    out = pd.Series(0.0, index=a.index, dtype=float)

    valid = (a > 0) & (denom > 0)
    out.loc[valid] = np.log(a.loc[valid] / denom.loc[valid])
    return out


def _rolling_corr(a: pd.Series, b: pd.Series, length: int) -> pd.Series:
    a = _series(a)
    b = _series(b, index=a.index)
    return a.rolling(length, min_periods=length).corr(b)


def _zscore(src: pd.Series, length: int) -> pd.Series:
    src = _series(src)
    mean = _sma(src, length)
    std = _stdev(src, length)

    out = pd.Series(0.0, index=src.index, dtype=float)
    valid = std > 0
    out.loc[valid] = (src.loc[valid] - mean.loc[valid]) / std.loc[valid]
    return out


def _topk_indices(values: list[float], k: int = 5) -> list[int]:
    sz = len(values)
    if sz == 0:
        return []

    kk = min(k, sz)
    tmp = [0.0 if pd.isna(v) else float(v) for v in values]
    out: list[int] = []

    for _ in range(kk):
        max_i = 0
        max_v = tmp[0]

        for j in range(1, sz):
            vj = tmp[j]
            take = pd.isna(max_v) or ((not pd.isna(vj)) and (vj > max_v))
            if take:
                max_v = vj
                max_i = j

        out.append(max_i)
        tmp[max_i] = np.nan

    return out


def _pred(top_idx: list[int], coef_arr: list[float], feat_arr: list[float]) -> float:
    total = 0.0
    for idx in top_idx:
        c = 0.0 if pd.isna(coef_arr[idx]) else float(coef_arr[idx])
        z = 0.0 if pd.isna(feat_arr[idx]) else float(feat_arr[idx])
        total += c * z
    return float(total)


def _tf_to_pandas_rule(tf: str) -> str:
    tf = str(tf).strip()

    mapping = {
        "1": "1min",
        "3": "3min",
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "45": "45min",
        "60": "60min",
        "120": "120min",
        "180": "180min",
        "240": "240min",
        "D": "1D",
        "W": "1W",
        "M": "1ME",
    }

    if tf in mapping:
        return mapping[tf]

    raise ValueError(f"Unsupported timeframe for AI RSI parity mapping: {tf}")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    out = df.resample(rule, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _dir_from_signal(sig: pd.Series) -> pd.Series:
    sig = _series(sig)
    out = pd.Series(0.0, index=sig.index, dtype=float)
    out = out.mask(sig > 0, 1.0)
    out = out.mask(sig < 0, -1.0)
    return out


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


def _dir_text(v: float) -> str:
    if v > 0:
        return "BULLISH"
    if v < 0:
        return "BEARISH"
    return "NEUTRAL"


def _mtf_strength_text(v: float) -> str:
    if v > 0.6:
        return "STRONG BULL"
    if v > 0.2:
        return "BULL"
    if v < -0.6:
        return "STRONG BEAR"
    if v < -0.2:
        return "BEAR"
    return "NEUTRAL"


def _state_text(direction: float, strength: float, neutral: bool) -> str:
    if neutral:
        return "NEUTRAL"
    if direction > 0 and strength >= 0.8:
        return "STRONG BULLISH"
    if direction > 0:
        return "BULLISH"
    if direction < 0 and strength >= 0.8:
        return "STRONG BEARISH"
    if direction < 0:
        return "BEARISH"
    return "NEUTRAL"


# =============================================================================
# CORE ENGINE
# =============================================================================

def compute_ai_rsi_core(
    df: pd.DataFrame,
    config: AiRsiConfig | None = None,
) -> pd.DataFrame:
    config = config or AiRsiConfig()
    _validate_ohlcv(df)

    out = df.copy()

    close = _series(out["close"])
    high = _series(out["high"], index=close.index)
    low = _series(out["low"], index=close.index)
    vol = _series(out["volume"], index=close.index)

    ret_log = _safe_log_ratio(close, close.shift(1))
    rsi_val = _rsi(close, config.rsi_len)
    atr_pct = _atr(high, low, close, 200) / close.replace(0, np.nan)
    atr_pct = atr_pct.fillna(0.0)
    vol_log_chg = _safe_log_ratio(vol, vol.shift(1))

    out["retLog"] = ret_log
    out["rsiVal"] = rsi_val
    out["atrPct"] = atr_pct
    out["volLogChg"] = vol_log_chg

    y_rsi = rsi_val.shift(1)
    out["y_rsi"] = y_rsi

    x_ret = ret_log.shift(1).fillna(0.0)
    x_rsi = rsi_val.shift(1).fillna(0.0)
    x_atrp = atr_pct.shift(1).fillna(0.0)
    x_vchg = vol_log_chg.shift(1).fillna(0.0)
    x_vol = vol.shift(1).fillna(0.0)

    out["x_ret"] = x_ret
    out["x_rsi"] = x_rsi
    out["x_atrp"] = x_atrp
    out["x_vchg"] = x_vchg
    out["x_vol"] = x_vol

    corr_ret = _rolling_corr(y_rsi, x_ret, config.learn_len)
    corr_rsi = _rolling_corr(y_rsi, x_rsi, config.learn_len)
    corr_atrp = _rolling_corr(y_rsi, x_atrp, config.learn_len)
    corr_vchg = _rolling_corr(y_rsi, x_vchg, config.learn_len)
    corr_vol = _rolling_corr(y_rsi, x_vol, config.learn_len)

    corrs_abs_ret = corr_ret.fillna(0.0).abs()
    corrs_abs_rsi = corr_rsi.fillna(0.0).abs()
    corrs_abs_atrp = corr_atrp.fillna(0.0).abs()
    corrs_abs_vchg = corr_vchg.fillna(0.0).abs()
    corrs_abs_vol = corr_vol.fillna(0.0).abs()

    out["coef_ret_raw"] = corr_ret
    out["coef_rsi_raw"] = corr_rsi
    out["coef_atrp_raw"] = corr_atrp
    out["coef_vchg_raw"] = corr_vchg
    out["coef_vol_raw"] = corr_vol

    out["corrs_abs_ret"] = corrs_abs_ret
    out["corrs_abs_rsi"] = corrs_abs_rsi
    out["corrs_abs_atrp"] = corrs_abs_atrp
    out["corrs_abs_vchg"] = corrs_abs_vchg
    out["corrs_abs_vol"] = corrs_abs_vol

    top_idx_0: list[int] = []
    top_idx_1: list[int] = []
    top_idx_2: list[int] = []
    top_idx_3: list[int] = []
    top_idx_4: list[int] = []

    for i in range(len(out)):
        corrs_i = [
            corrs_abs_ret.iloc[i],
            corrs_abs_rsi.iloc[i],
            corrs_abs_atrp.iloc[i],
            corrs_abs_vchg.iloc[i],
            corrs_abs_vol.iloc[i],
        ]
        top_idx = _topk_indices(corrs_i, k=5)

        top_idx_0.append(top_idx[0] if len(top_idx) > 0 else 0)
        top_idx_1.append(top_idx[1] if len(top_idx) > 1 else 0)
        top_idx_2.append(top_idx[2] if len(top_idx) > 2 else 0)
        top_idx_3.append(top_idx[3] if len(top_idx) > 3 else 0)
        top_idx_4.append(top_idx[4] if len(top_idx) > 4 else 0)

    out["topIdx_0"] = top_idx_0
    out["topIdx_1"] = top_idx_1
    out["topIdx_2"] = top_idx_2
    out["topIdx_3"] = top_idx_3
    out["topIdx_4"] = top_idx_4

    xz_ret = _zscore(x_ret, config.learn_len).fillna(0.0)
    xz_rsi = _zscore(x_rsi, config.learn_len).fillna(0.0)
    xz_atrp = _zscore(x_atrp, config.learn_len).fillna(0.0)
    xz_vchg = _zscore(x_vchg, config.learn_len).fillna(0.0)
    xz_vol = _zscore(x_vol, config.learn_len).fillna(0.0)

    out["xz_ret"] = xz_ret
    out["xz_rsi"] = xz_rsi
    out["xz_atrp"] = xz_atrp
    out["xz_vchg"] = xz_vchg
    out["xz_vol"] = xz_vol

    coef_ret = corr_ret.fillna(0.0)
    coef_rsi = pd.Series(1.0, index=out.index, dtype=float)
    coef_atrp = corr_atrp.fillna(0.0)
    coef_vchg = corr_vchg.fillna(0.0)
    coef_vol = corr_vol.fillna(0.0)

    out["coef_ret"] = coef_ret
    out["coef_rsi"] = coef_rsi
    out["coef_atrp"] = coef_atrp
    out["coef_vchg"] = coef_vchg
    out["coef_vol"] = coef_vol

    pred_rsi_z: list[float] = []

    for i in range(len(out)):
        top_idx = [
            int(out["topIdx_0"].iloc[i]),
            int(out["topIdx_1"].iloc[i]),
            int(out["topIdx_2"].iloc[i]),
            int(out["topIdx_3"].iloc[i]),
            int(out["topIdx_4"].iloc[i]),
        ]

        coef_arr = [
            coef_ret.iloc[i],
            coef_rsi.iloc[i],
            coef_atrp.iloc[i],
            coef_vchg.iloc[i],
            coef_vol.iloc[i],
        ]

        feat_arr = [
            xz_ret.iloc[i],
            xz_rsi.iloc[i],
            xz_atrp.iloc[i],
            xz_vchg.iloc[i],
            xz_vol.iloc[i],
        ]

        pred_rsi_z.append(_pred(top_idx, coef_arr, feat_arr))

    pred_rsi_z = pd.Series(pred_rsi_z, index=out.index, dtype=float)
    out["pred_rsi_z"] = pred_rsi_z

    rsi_mean = _sma(y_rsi, config.learn_len)
    rsi_std = _stdev(y_rsi, config.learn_len)
    pred_rsi = rsi_mean.fillna(0.0) + rsi_std.fillna(0.0) * pred_rsi_z

    out["rsi_mean"] = rsi_mean
    out["rsi_std"] = rsi_std
    out["pred_rsi"] = pred_rsi

    rsi_weight = ((50.0 - pred_rsi.fillna(0.0)) / 50.0).clip(-2.0, 2.0) * -1.0
    ma_rsi = _sma(rsi_weight, config.sig_len)

    out["rsiWeight"] = rsi_weight
    out["ma_rsi"] = ma_rsi

    return out


# =============================================================================
# BASE STATE
# =============================================================================

def apply_ai_rsi_base_state(
    df: pd.DataFrame,
    config: AiRsiConfig | None = None,
) -> pd.DataFrame:
    config = config or AiRsiConfig()
    out = df.copy()

    ma_rsi = _series(out["ma_rsi"])

    ai_dead_zone = pd.Series(
        config.dead_zone_val if config.dead_zone_on else 0.0,
        index=out.index,
        dtype=float,
    )
    out["aiDeadZone"] = ai_dead_zone

    ai_neutral = ma_rsi.abs() < ai_dead_zone
    out["aiNeutral"] = ai_neutral

    ai_dir = pd.Series(0.0, index=out.index, dtype=float)
    ai_dir = ai_dir.mask((~ai_neutral) & (ma_rsi > 0), 1.0)
    ai_dir = ai_dir.mask((~ai_neutral) & (ma_rsi < 0), -1.0)
    out["aiDir"] = ai_dir

    out["aiBull"] = (~ai_neutral) & (ma_rsi > 0)
    out["aiBear"] = (~ai_neutral) & (ma_rsi < 0)
    out["aiStrongBull"] = ma_rsi > 0.5
    out["aiStrongBear"] = ma_rsi < -0.5

    ai_strength_score = (ma_rsi.abs() / 0.5).clip(upper=1.0)
    ai_strength_score = ai_strength_score.mask(ai_neutral, 0.0)
    out["aiStrengthScore"] = ai_strength_score.astype(float)

    return out


# =============================================================================
# MTF CONFIRM-ONLY LAYER
# =============================================================================

def _compute_single_mtf_dir(
    df: pd.DataFrame,
    tf: str,
    config: AiRsiConfig,
) -> pd.Series:
    rule = _tf_to_pandas_rule(tf)
    htf = _resample_ohlcv(df[["open", "high", "low", "close", "volume"]], rule)

    htf = compute_ai_rsi_core(htf, config=config)
    htf = apply_ai_rsi_base_state(htf, config=config)
    htf_dir = _dir_from_signal(htf["ma_rsi"])

    base_dir = htf_dir.reindex(df.index, method="ffill").fillna(0.0)
    return pd.Series(base_dir, index=df.index, dtype=float)


def apply_ai_rsi_mtf_layer(
    df: pd.DataFrame,
    config: AiRsiConfig | None = None,
) -> pd.DataFrame:
    config = config or AiRsiConfig()
    out = df.copy()

    if "aiDir" not in out.columns:
        raise ValueError("apply_ai_rsi_mtf_layer requires aiDir from base state.")

    if not config.mtf_on:
        for col in ["mtf1", "mtf2", "mtf3", "mtf4", "mtf5", "mtfSum", "mtfAvg", "aiMtfStrength"]:
            out[col] = 0.0
        for col in ["mtfStrongBull", "mtfBull", "mtfStrongBear", "mtfBear", "mtfAlignedBull", "mtfAlignedBear", "mtfConflict"]:
            out[col] = False
        return out

    mtf1 = _compute_single_mtf_dir(out, config.tf1, config)
    mtf2 = _compute_single_mtf_dir(out, config.tf2, config)
    mtf3 = _compute_single_mtf_dir(out, config.tf3, config)
    mtf4 = _compute_single_mtf_dir(out, config.tf4, config)
    mtf5 = _compute_single_mtf_dir(out, config.tf5, config)

    out["mtf1"] = mtf1
    out["mtf2"] = mtf2
    out["mtf3"] = mtf3
    out["mtf4"] = mtf4
    out["mtf5"] = mtf5

    mtf_sum = mtf1 + mtf2 + mtf3 + mtf4 + mtf5
    mtf_avg = mtf_sum / 5.0

    out["mtfSum"] = mtf_sum
    out["mtfAvg"] = mtf_avg
    out["mtfStrongBull"] = mtf_avg > 0.6
    out["mtfBull"] = mtf_avg > 0.2
    out["mtfStrongBear"] = mtf_avg < -0.6
    out["mtfBear"] = mtf_avg < -0.2
    out["aiMtfStrength"] = mtf_avg.astype(float)

    ai_dir = _series(out["aiDir"])
    out["mtfAlignedBull"] = (ai_dir > 0) & (mtf_avg > 0.2)
    out["mtfAlignedBear"] = (ai_dir < 0) & (mtf_avg < -0.2)
    out["mtfConflict"] = ((ai_dir > 0) & (mtf_avg < -0.2)) | ((ai_dir < 0) & (mtf_avg > 0.2))

    return out


# =============================================================================
# EXPORT FIELDS
# =============================================================================

def apply_ai_rsi_exports(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sc_ai_rsi_raw"] = _series(out["rsiWeight"])
    out["sc_ai_rsi_signal"] = _series(out["ma_rsi"])
    out["sc_ai_rsi_dir"] = _series(out["aiDir"])
    out["sc_ai_rsi_strength"] = _series(out["aiStrengthScore"])
    out["sc_ai_rsi_dead_zone"] = _series(out["aiDeadZone"])

    out["sc_ai_rsi_is_neutral"] = _series(out["aiNeutral"]).astype(float)
    out["sc_ai_rsi_is_bull"] = _series(out["aiBull"]).astype(float)
    out["sc_ai_rsi_is_bear"] = _series(out["aiBear"]).astype(float)
    out["sc_ai_rsi_is_strong_bull"] = _series(out["aiStrongBull"]).astype(float)
    out["sc_ai_rsi_is_strong_bear"] = _series(out["aiStrongBear"]).astype(float)

    out["sc_ai_rsi_mtf_1"] = _series(out["mtf1"])
    out["sc_ai_rsi_mtf_2"] = _series(out["mtf2"])
    out["sc_ai_rsi_mtf_3"] = _series(out["mtf3"])
    out["sc_ai_rsi_mtf_4"] = _series(out["mtf4"])
    out["sc_ai_rsi_mtf_5"] = _series(out["mtf5"])
    out["sc_ai_rsi_mtf_avg"] = _series(out["mtfAvg"])
    out["sc_ai_rsi_mtf_strength"] = _series(out["aiMtfStrength"])
    out["sc_ai_rsi_mtf_aligned"] = (out["mtfAlignedBull"] | out["mtfAlignedBear"]).astype(float)
    out["sc_ai_rsi_mtf_conflict"] = _series(out["mtfConflict"]).astype(float)

    return out


# =============================================================================
# MAIN ENGINE WRAPPER
# =============================================================================

def run_ai_rsi_engine(
    df: pd.DataFrame,
    config: AiRsiConfig | None = None,
) -> pd.DataFrame:
    config = config or AiRsiConfig()
    _validate_ohlcv(df)

    out = df.copy()
    out = compute_ai_rsi_core(out, config=config)
    out = apply_ai_rsi_base_state(out, config=config)
    out = apply_ai_rsi_mtf_layer(out, config=config)
    out = apply_ai_rsi_exports(out)

    return out


# =============================================================================
# JSON PAYLOAD BUILDER
# =============================================================================

def build_ai_rsi_latest_payload(
    df: pd.DataFrame,
    config: AiRsiConfig | None = None,
) -> Dict[str, Any]:
    config = config or AiRsiConfig()

    engine = run_ai_rsi_engine(df, config=config)
    if engine.empty:
        return {}

    row = engine.iloc[-1]

    ai_dir = _safe_float(row.get("sc_ai_rsi_dir"))
    ai_strength = _safe_float(row.get("sc_ai_rsi_strength"))
    ai_signal = _safe_float(row.get("sc_ai_rsi_signal"))
    ai_raw = _safe_float(row.get("sc_ai_rsi_raw"))
    ai_dead_zone = _safe_float(row.get("sc_ai_rsi_dead_zone"))
    mtf_avg = _safe_float(row.get("sc_ai_rsi_mtf_avg"))
    mtf_strength = _safe_float(row.get("sc_ai_rsi_mtf_strength"))
    is_neutral = bool(_safe_int(row.get("sc_ai_rsi_is_neutral")))

    payload: Dict[str, Any] = {
        # ---------------------------------------------------------------------
        # LOCKED SHARED WEBSITE CONTRACT
        # ---------------------------------------------------------------------
        "timestamp": str(engine.index[-1]),
        "state": _state_text(ai_dir, ai_strength, is_neutral),
        "bias_signal": _safe_int(ai_dir),
        "bias_label": _dir_text(ai_dir),
        "indicator_strength": round(ai_strength, 4),

        # ---------------------------------------------------------------------
        # MODULE IDENTITY
        # ---------------------------------------------------------------------
        "engine": "AI RSI",
        "module": "ai_rsi",
        "version": "v2",
        "category": "Reversal Indicators",
        "family": "Momentum / AI",

        # ---------------------------------------------------------------------
        # PRIMARY CARD FIELDS
        # ---------------------------------------------------------------------
        "signal_value": round(ai_signal, 6),
        "raw_value": round(ai_raw, 6),
        "dead_zone": round(ai_dead_zone, 6),
        "direction": _safe_int(ai_dir),
        "direction_label": _dir_text(ai_dir),

        "is_neutral": is_neutral,
        "is_bull": bool(_safe_int(row.get("sc_ai_rsi_is_bull"))),
        "is_bear": bool(_safe_int(row.get("sc_ai_rsi_is_bear"))),
        "is_strong_bull": bool(_safe_int(row.get("sc_ai_rsi_is_strong_bull"))),
        "is_strong_bear": bool(_safe_int(row.get("sc_ai_rsi_is_strong_bear"))),

        # ---------------------------------------------------------------------
        # MTF WEBSITE FIELDS
        # ---------------------------------------------------------------------
        "mtf_average": round(mtf_avg, 4),
        "mtf_strength": round(mtf_strength, 4),
        "mtf_state": _mtf_strength_text(mtf_avg),
        "mtf_aligned": bool(_safe_int(row.get("sc_ai_rsi_mtf_aligned"))),
        "mtf_conflict": bool(_safe_int(row.get("sc_ai_rsi_mtf_conflict"))),

        # ---------------------------------------------------------------------
        # MODULE-SPECIFIC DETAIL
        # ---------------------------------------------------------------------
        "prediction_rsi": round(_safe_float(row.get("pred_rsi")), 4),
        "prediction_z": round(_safe_float(row.get("pred_rsi_z")), 4),
        "rsi_value": round(_safe_float(row.get("rsiVal")), 4),
        "rsi_mean": round(_safe_float(row.get("rsi_mean")), 4),
        "rsi_std": round(_safe_float(row.get("rsi_std")), 4),
        "ret_log": round(_safe_float(row.get("retLog")), 6),
        "atr_pct": round(_safe_float(row.get("atrPct")), 6),
        "vol_log_chg": round(_safe_float(row.get("volLogChg")), 6),

        "mtf": {
            "tf1": round(_safe_float(row.get("sc_ai_rsi_mtf_1")), 4),
            "tf2": round(_safe_float(row.get("sc_ai_rsi_mtf_2")), 4),
            "tf3": round(_safe_float(row.get("sc_ai_rsi_mtf_3")), 4),
            "tf4": round(_safe_float(row.get("sc_ai_rsi_mtf_4")), 4),
            "tf5": round(_safe_float(row.get("sc_ai_rsi_mtf_5")), 4),
            "average": round(mtf_avg, 4),
            "strength": round(mtf_strength, 4),
            "state_label": _mtf_strength_text(mtf_avg),
            "aligned": bool(_safe_int(row.get("sc_ai_rsi_mtf_aligned"))),
            "conflict": bool(_safe_int(row.get("sc_ai_rsi_mtf_conflict"))),
        },

        "settings": asdict(config),

        "debug": {
            "top_idx": [
                _safe_int(row.get("topIdx_0")),
                _safe_int(row.get("topIdx_1")),
                _safe_int(row.get("topIdx_2")),
                _safe_int(row.get("topIdx_3")),
                _safe_int(row.get("topIdx_4")),
            ],
            "x_rsi": round(_safe_float(row.get("x_rsi")), 4),
            "y_rsi": round(_safe_float(row.get("y_rsi")), 4),
        },
    }

    return payload