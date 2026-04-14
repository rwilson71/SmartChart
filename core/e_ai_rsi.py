from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# =============================================================================
# HELPERS — STRICT PINE PARITY REBUILD
# =============================================================================

def _sma(src: pd.Series, length: int) -> pd.Series:
    """
    Pine parity for ta.sma(src, length)
    """
    src = pd.Series(src, copy=False).astype(float)
    return src.rolling(length, min_periods=length).mean()


def _stdev(src: pd.Series, length: int) -> pd.Series:
    """
    Pine parity for ta.stdev(src, length)

    Uses population stdev behavior for parity work.
    """
    src = pd.Series(src, copy=False).astype(float)
    return src.rolling(length, min_periods=length).std(ddof=0)


def _rma(src: pd.Series, length: int) -> pd.Series:
    """
    Pine parity for ta.rma(src, length)

    Pine RMA:
    - alpha = 1 / length
    - seed starts from SMA(length)
    - recursive smoothing after seed
    """
    src = pd.Series(src, copy=False).astype(float)
    out = pd.Series(np.nan, index=src.index, dtype=float)

    if length <= 0 or len(src) == 0:
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
    """
    Pine parity for ta.rsi(close, length)
    """
    close = pd.Series(close, copy=False).astype(float)
    delta = close.diff()

    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    avg_up = _rma(up, length)
    avg_down = _rma(down, length)

    rs = avg_up / avg_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Pine-style edge behavior
    rsi = rsi.where(avg_down != 0, 100.0)
    rsi = rsi.where(~((avg_up == 0) & (avg_down == 0)), 0.0)

    return rsi


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int,
) -> pd.Series:
    """
    Pine parity for ta.atr(length)
    """
    high = pd.Series(high, copy=False).astype(float)
    low = pd.Series(low, copy=False).astype(float)
    close = pd.Series(close, copy=False).astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _rma(tr, length)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Pine parity helper for:
        math.log(a / nz(b, a))

    Safe handling:
    - if b is NaN, replace with a
    - if denominator <= 0 or numerator <= 0, return 0.0
    """
    a = pd.Series(a, copy=False).astype(float)
    b = pd.Series(b, copy=False).astype(float)

    denom = b.fillna(a)

    out = pd.Series(0.0, index=a.index, dtype=float)
    valid = (a > 0) & (denom > 0)
    out.loc[valid] = np.log(a.loc[valid] / denom.loc[valid])

    return out


def _rolling_corr(a: pd.Series, b: pd.Series, length: int) -> pd.Series:
    """
    Pine parity for ta.correlation(a, b, length)
    """
    a = pd.Series(a, copy=False).astype(float)
    b = pd.Series(b, copy=False).astype(float)
    return a.rolling(length, min_periods=length).corr(b)


def _zscore(src: pd.Series, length: int) -> pd.Series:
    """
    Pine parity for:
        f_z(src, len) =>
            m = ta.sma(src, len)
            s = ta.stdev(src, len)
            s > 0 ? (src - m) / s : 0.0
    """
    src = pd.Series(src, copy=False).astype(float)
    mean = _sma(src, length)
    std = _stdev(src, length)

    out = pd.Series(0.0, index=src.index, dtype=float)
    valid = std > 0
    out.loc[valid] = (src.loc[valid] - mean.loc[valid]) / std.loc[valid]
    return out


def _nz(value, replacement=0.0):
    """
    Pine-style nz() helper for scalars.
    """
    if pd.isna(value):
        return replacement
    return value


def _topk_indices(values: List[float], k: int = 5) -> List[int]:
    """
    Pine parity for f_topk_indices(arr)

    Behavior:
    - size-limited to min(5, len(values))
    - NaNs converted with nz(...)->0.0 before ranking
    - stable first-max wins on ties
    - selected value then invalidated
    """
    sz = len(values)
    kk = min(k, sz)

    tmp = [0.0 if pd.isna(v) else float(v) for v in values]
    out: List[int] = []

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


def _pred(top_idx: List[int], coef_arr: List[float], feat_arr: List[float]) -> float:
    """
    Pine parity for:
        f_pred(topIdxArr, coefArr, featArr)
    """
    total = 0.0
    for idx in top_idx:
        c = coef_arr[idx]
        z = feat_arr[idx]
        total += _nz(c, 0.0) * _nz(z, 0.0)
    return float(total)


# =============================================================================
# CORE AI RSI ENGINE — PARITY LOGIC
# =============================================================================

def compute_ai_rsi_core(
    df: pd.DataFrame,
    rsi_len: int = 14,
    sig_len: int = 20,
    learn_len: int = 20,
) -> pd.DataFrame:
    """
    Strict Pine-parity rebuild of the AI RSI core block.

    Required columns:
    - close
    - high
    - low
    - volume
    """
    out = df.copy()

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    vol = out["volume"].astype(float)

    # -------------------------------------------------------------------------
    # Per-bar features
    # Pine:
    # retLog    = math.log(close / nz(close[1], close))
    # rsiVal    = ta.rsi(close, rsiLen)
    # atrPct    = ta.atr(200) / close
    # vol       = volume
    # volLogChg = math.log(vol / nz(vol[1], vol))
    # -------------------------------------------------------------------------
    ret_log = _safe_log_ratio(close, close.shift(1))
    rsi_val = _rsi(close, rsi_len)
    atr_pct = _atr(high, low, close, 200) / close
    vol_log_chg = _safe_log_ratio(vol, vol.shift(1))

    out["retLog"] = ret_log
    out["rsiVal"] = rsi_val
    out["atrPct"] = atr_pct
    out["vol"] = vol
    out["volLogChg"] = vol_log_chg

    # -------------------------------------------------------------------------
    # Target
    # Pine:
    # y_rsi = rsiVal[1]
    # -------------------------------------------------------------------------
    y_rsi = rsi_val.shift(1)
    out["y_rsi"] = y_rsi

    # -------------------------------------------------------------------------
    # Predictors
    # Pine:
    # x_ret  = nz(retLog[1])
    # x_rsi  = nz(rsiVal[1])
    # x_atrp = nz(atrPct[1])
    # x_vchg = nz(volLogChg[1])
    # x_vol  = nz(vol[1])
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Feature selection via rolling correlations
    # Pine:
    # corrs_abs_ret  = math.abs(nz(f_corr(y_rsi, x_ret,  learnLen)))
    # ...
    # corrs  = array.from(...)
    # topIdx = f_topk_indices(corrs)
    # -------------------------------------------------------------------------
    corr_ret = _rolling_corr(y_rsi, x_ret, learn_len)
    corr_rsi = _rolling_corr(y_rsi, x_rsi, learn_len)
    corr_atrp = _rolling_corr(y_rsi, x_atrp, learn_len)
    corr_vchg = _rolling_corr(y_rsi, x_vchg, learn_len)
    corr_vol = _rolling_corr(y_rsi, x_vol, learn_len)

    out["coef_ret_raw"] = corr_ret
    out["coef_rsi_raw"] = corr_rsi
    out["coef_atrp_raw"] = corr_atrp
    out["coef_vchg_raw"] = corr_vchg
    out["coef_vol_raw"] = corr_vol

    corrs_abs_ret = corr_ret.fillna(0.0).abs()
    corrs_abs_rsi = corr_rsi.fillna(0.0).abs()
    corrs_abs_atrp = corr_atrp.fillna(0.0).abs()
    corrs_abs_vchg = corr_vchg.fillna(0.0).abs()
    corrs_abs_vol = corr_vol.fillna(0.0).abs()

    out["corrs_abs_ret"] = corrs_abs_ret
    out["corrs_abs_rsi"] = corrs_abs_rsi
    out["corrs_abs_atrp"] = corrs_abs_atrp
    out["corrs_abs_vchg"] = corrs_abs_vchg
    out["corrs_abs_vol"] = corrs_abs_vol

    top_idx_0 = []
    top_idx_1 = []
    top_idx_2 = []
    top_idx_3 = []
    top_idx_4 = []

    for i in range(len(out)):
        corrs_i = [
            corrs_abs_ret.iloc[i],
            corrs_abs_rsi.iloc[i],
            corrs_abs_atrp.iloc[i],
            corrs_abs_vchg.iloc[i],
            corrs_abs_vol.iloc[i],
        ]
        top_idx = _topk_indices(corrs_i, k=5)

        top_idx_0.append(top_idx[0] if len(top_idx) > 0 else np.nan)
        top_idx_1.append(top_idx[1] if len(top_idx) > 1 else np.nan)
        top_idx_2.append(top_idx[2] if len(top_idx) > 2 else np.nan)
        top_idx_3.append(top_idx[3] if len(top_idx) > 3 else np.nan)
        top_idx_4.append(top_idx[4] if len(top_idx) > 4 else np.nan)

    out["topIdx_0"] = top_idx_0
    out["topIdx_1"] = top_idx_1
    out["topIdx_2"] = top_idx_2
    out["topIdx_3"] = top_idx_3
    out["topIdx_4"] = top_idx_4

    # -------------------------------------------------------------------------
    # Z-scored feature levels
    # Pine:
    # xz_ret  = nz(f_z(x_ret,  learnLen))
    # ...
    # featZ   = array.from(...)
    # -------------------------------------------------------------------------
    xz_ret = _zscore(x_ret, learn_len).fillna(0.0)
    xz_rsi = _zscore(x_rsi, learn_len).fillna(0.0)
    xz_atrp = _zscore(x_atrp, learn_len).fillna(0.0)
    xz_vchg = _zscore(x_vchg, learn_len).fillna(0.0)
    xz_vol = _zscore(x_vol, learn_len).fillna(0.0)

    out["xz_ret"] = xz_ret
    out["xz_rsi"] = xz_rsi
    out["xz_atrp"] = xz_atrp
    out["xz_vchg"] = xz_vchg
    out["xz_vol"] = xz_vol

    # -------------------------------------------------------------------------
    # Signed coefficients
    # Pine:
    # coef_ret  = nz(f_corr(y_rsi, x_ret,  learnLen))
    # coef_rsi  = 1.0
    # coef_atrp = nz(f_corr(y_rsi, x_atrp, learnLen))
    # coef_vchg = nz(f_corr(y_rsi, x_vchg, learnLen))
    # coef_vol  = nz(f_corr(y_rsi, x_vol,  learnLen))
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Prediction
    # Pine:
    # pred_rsi_z = f_pred(topIdx, coef, featZ)
    # -------------------------------------------------------------------------
    pred_rsi_z = []

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

    # -------------------------------------------------------------------------
    # Map prediction back into RSI space
    # Pine:
    # rsi_mean = ta.sma(y_rsi, learnLen)
    # rsi_std  = ta.stdev(y_rsi, learnLen)
    # pred_rsi = nz(rsi_mean) + nz(rsi_std) * pred_rsi_z
    # -------------------------------------------------------------------------
    rsi_mean = _sma(y_rsi, learn_len)
    rsi_std = _stdev(y_rsi, learn_len)
    pred_rsi = rsi_mean.fillna(0.0) + rsi_std.fillna(0.0) * pred_rsi_z

    out["rsi_mean"] = rsi_mean
    out["rsi_std"] = rsi_std
    out["pred_rsi"] = pred_rsi

    # -------------------------------------------------------------------------
    # Final AI-weighted RSI
    # Pine:
    # rsiWeight = math.max(-2.0, math.min(2.0, (50.0 - nz(pred_rsi)) / 50.0)) * -1.0
    # ma_rsi    = ta.sma(rsiWeight, sigLen)
    # -------------------------------------------------------------------------
    rsi_weight = ((50.0 - pred_rsi.fillna(0.0)) / 50.0).clip(-2.0, 2.0) * -1.0
    ma_rsi = _sma(rsi_weight, sig_len)

    out["rsiWeight"] = rsi_weight
    out["ma_rsi"] = ma_rsi

    return out


# =============================================================================
# BASE ENGINE STATE
# =============================================================================

def apply_ai_rsi_base_state(
    df: pd.DataFrame,
    dead_zone_on: bool = True,
    dead_zone_val: float = 0.15,
) -> pd.DataFrame:
    """
    Strict Pine-parity rebuild of the base AI RSI state block.
    """
    out = df.copy()
    ma_rsi = out["ma_rsi"].astype(float)

    # -------------------------------------------------------------------------
    # Pine:
    # aiDeadZone = deadZoneOn ? deadZoneVal : 0.0
    # -------------------------------------------------------------------------
    ai_dead_zone = pd.Series(
        dead_zone_val if dead_zone_on else 0.0,
        index=out.index,
        dtype=float,
    )
    out["aiDeadZone"] = ai_dead_zone

    # -------------------------------------------------------------------------
    # Pine:
    # aiNeutral = math.abs(ma_rsi) < aiDeadZone
    # -------------------------------------------------------------------------
    ai_neutral = ma_rsi.abs() < ai_dead_zone
    out["aiNeutral"] = ai_neutral

    # -------------------------------------------------------------------------
    # Pine:
    # aiDir =
    #      aiNeutral ? 0 :
    #      ma_rsi > 0 ? 1 :
    #      ma_rsi < 0 ? -1 : 0
    # -------------------------------------------------------------------------
    ai_dir = pd.Series(0.0, index=out.index, dtype=float)
    ai_dir = ai_dir.mask(~ai_neutral & (ma_rsi > 0), 1.0)
    ai_dir = ai_dir.mask(~ai_neutral & (ma_rsi < 0), -1.0)
    out["aiDir"] = ai_dir

    # -------------------------------------------------------------------------
    # Pine:
    # aiBull = not aiNeutral and ma_rsi > 0
    # aiBear = not aiNeutral and ma_rsi < 0
    # -------------------------------------------------------------------------
    ai_bull = (~ai_neutral) & (ma_rsi > 0)
    ai_bear = (~ai_neutral) & (ma_rsi < 0)
    out["aiBull"] = ai_bull
    out["aiBear"] = ai_bear

    # -------------------------------------------------------------------------
    # Pine:
    # aiStrongBull = ma_rsi > 0.5
    # aiStrongBear = ma_rsi < -0.5
    # -------------------------------------------------------------------------
    ai_strong_bull = ma_rsi > 0.5
    ai_strong_bear = ma_rsi < -0.5
    out["aiStrongBull"] = ai_strong_bull
    out["aiStrongBear"] = ai_strong_bear

    # -------------------------------------------------------------------------
    # Pine:
    # aiStrengthScore =
    #      aiNeutral ? 0.0 :
    #      math.min(1.0, math.abs(ma_rsi) / 0.5)
    # -------------------------------------------------------------------------
    ai_strength_score = (ma_rsi.abs() / 0.5).clip(upper=1.0)
    ai_strength_score = ai_strength_score.mask(ai_neutral, 0.0)
    out["aiStrengthScore"] = ai_strength_score.astype(float)

    return out


# =============================================================================
# MTF AGREEMENT LAYER — CONFIRM ONLY
# =============================================================================

def _tf_to_pandas_rule(tf: str) -> str:
    """
    Minimal Pine timeframe -> pandas resample rule mapper
    """
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

    raise ValueError(f"Unsupported timeframe for parity mapping: {tf}")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV data to higher timeframe.

    Required index:
    - DatetimeIndex

    Required columns:
    - open, high, low, close, volume
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for MTF parity.")

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


def _mtf_dir_from_ma_rsi(htf_ma_rsi: pd.Series) -> pd.Series:
    """
    Pine parity for:
        ma_rsi > 0 ? 1.0 : ma_rsi < 0 ? -1.0 : 0.0
    """
    out = pd.Series(0.0, index=htf_ma_rsi.index, dtype=float)
    out = out.mask(htf_ma_rsi > 0, 1.0)
    out = out.mask(htf_ma_rsi < 0, -1.0)
    return out


def _compute_single_mtf_dir(
    df: pd.DataFrame,
    tf: str,
    rsi_len: int,
    sig_len: int,
    learn_len: int,
) -> pd.Series:
    """
    Python parity replacement for Pine:
        request.security(syminfo.tickerid, tf, ma_rsi > 0 ? 1.0 : ma_rsi < 0 ? -1.0 : 0.0)

    Process:
    1. Resample base OHLCV to target TF
    2. Recompute AI RSI core on that TF
    3. Convert HTF ma_rsi to 1 / -1 / 0
    4. Forward-fill back to base timeframe
    """
    rule = _tf_to_pandas_rule(tf)
    htf = _resample_ohlcv(df, rule)

    htf = compute_ai_rsi_core(
        htf,
        rsi_len=rsi_len,
        sig_len=sig_len,
        learn_len=learn_len,
    )

    htf_dir = _mtf_dir_from_ma_rsi(htf["ma_rsi"])

    base_dir = htf_dir.reindex(df.index, method="ffill").fillna(0.0)
    return pd.Series(base_dir, index=df.index, dtype=float)


def apply_ai_rsi_mtf_layer(
    df: pd.DataFrame,
    mtf_on: bool = True,
    tf1: str = "5",
    tf2: str = "15",
    tf3: str = "60",
    tf4: str = "240",
    tf5: str = "D",
    rsi_len: int = 14,
    sig_len: int = 20,
    learn_len: int = 20,
) -> pd.DataFrame:
    """
    Strict Pine-parity rebuild of the MTF confirm-only layer.

    Required input:
    - base OHLCV with DatetimeIndex
    - aiDir already present from base state block
    """
    out = df.copy()

    if "aiDir" not in out.columns:
        raise ValueError("apply_ai_rsi_mtf_layer requires 'aiDir' from base state block.")

    if not mtf_on:
        out["mtf1"] = 0.0
        out["mtf2"] = 0.0
        out["mtf3"] = 0.0
        out["mtf4"] = 0.0
        out["mtf5"] = 0.0
        out["mtfSum"] = 0.0
        out["mtfAvg"] = 0.0
        out["mtfStrongBull"] = False
        out["mtfBull"] = False
        out["mtfStrongBear"] = False
        out["mtfBear"] = False
        out["aiMtfStrength"] = 0.0
        out["mtfAlignedBull"] = False
        out["mtfAlignedBear"] = False
        out["mtfConflict"] = False
        return out

    # -------------------------------------------------------------------------
    # Pine:
    # mtf1 = mtfOn ? f_mtf_dir(tf1) : 0.0
    # ...
    # -------------------------------------------------------------------------
    mtf1 = _compute_single_mtf_dir(out, tf1, rsi_len, sig_len, learn_len)
    mtf2 = _compute_single_mtf_dir(out, tf2, rsi_len, sig_len, learn_len)
    mtf3 = _compute_single_mtf_dir(out, tf3, rsi_len, sig_len, learn_len)
    mtf4 = _compute_single_mtf_dir(out, tf4, rsi_len, sig_len, learn_len)
    mtf5 = _compute_single_mtf_dir(out, tf5, rsi_len, sig_len, learn_len)

    out["mtf1"] = mtf1
    out["mtf2"] = mtf2
    out["mtf3"] = mtf3
    out["mtf4"] = mtf4
    out["mtf5"] = mtf5

    # -------------------------------------------------------------------------
    # Pine:
    # mtfSum = mtf1 + mtf2 + mtf3 + mtf4 + mtf5
    # mtfAvg = mtfOn ? mtfSum / 5.0 : 0.0
    # -------------------------------------------------------------------------
    mtf_sum = mtf1 + mtf2 + mtf3 + mtf4 + mtf5
    mtf_avg = mtf_sum / 5.0

    out["mtfSum"] = mtf_sum
    out["mtfAvg"] = mtf_avg

    # -------------------------------------------------------------------------
    # Pine:
    # mtfStrongBull = mtfAvg > 0.6
    # mtfBull       = mtfAvg > 0.2
    # mtfStrongBear = mtfAvg < -0.6
    # mtfBear       = mtfAvg < -0.2
    # -------------------------------------------------------------------------
    out["mtfStrongBull"] = mtf_avg > 0.6
    out["mtfBull"] = mtf_avg > 0.2
    out["mtfStrongBear"] = mtf_avg < -0.6
    out["mtfBear"] = mtf_avg < -0.2

    # -------------------------------------------------------------------------
    # Pine:
    # aiMtfStrength = mtfAvg
    # -------------------------------------------------------------------------
    out["aiMtfStrength"] = mtf_avg.astype(float)

    # -------------------------------------------------------------------------
    # Pine:
    # mtfAlignedBull = aiDir > 0 and mtfAvg > 0.2
    # mtfAlignedBear = aiDir < 0 and mtfAvg < -0.2
    # mtfConflict    = (aiDir > 0 and mtfAvg < -0.2) or (aiDir < 0 and mtfAvg > 0.2)
    # -------------------------------------------------------------------------
    ai_dir = out["aiDir"].astype(float)

    out["mtfAlignedBull"] = (ai_dir > 0) & (mtf_avg > 0.2)
    out["mtfAlignedBear"] = (ai_dir < 0) & (mtf_avg < -0.2)
    out["mtfConflict"] = ((ai_dir > 0) & (mtf_avg < -0.2)) | ((ai_dir < 0) & (mtf_avg > 0.2))

    return out


# =============================================================================
# EXPORT / PARITY VALUES
# =============================================================================

def apply_ai_rsi_exports(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final export/parity mapping block for AI RSI.

    Mirrors the Pine export fields exactly so Python columns
    can be validated directly against TradingView CSV exports.
    """
    out = df.copy()

    # -------------------------------------------------------------------------
    # Base exports
    # -------------------------------------------------------------------------
    out["sc_ai_rsi_raw"] = out["rsiWeight"].astype(float)
    out["sc_ai_rsi_signal"] = out["ma_rsi"].astype(float)
    out["sc_ai_rsi_dir"] = out["aiDir"].astype(float)
    out["sc_ai_rsi_strength"] = out["aiStrengthScore"].astype(float)
    out["sc_ai_rsi_dead_zone"] = out["aiDeadZone"].astype(float)

    out["sc_ai_rsi_is_neutral"] = out["aiNeutral"].astype(float)
    out["sc_ai_rsi_is_bull"] = out["aiBull"].astype(float)
    out["sc_ai_rsi_is_bear"] = out["aiBear"].astype(float)
    out["sc_ai_rsi_is_strong_bull"] = out["aiStrongBull"].astype(float)
    out["sc_ai_rsi_is_strong_bear"] = out["aiStrongBear"].astype(float)

    # -------------------------------------------------------------------------
    # MTF exports
    # -------------------------------------------------------------------------
    out["sc_ai_rsi_mtf_1"] = out["mtf1"].astype(float)
    out["sc_ai_rsi_mtf_2"] = out["mtf2"].astype(float)
    out["sc_ai_rsi_mtf_3"] = out["mtf3"].astype(float)
    out["sc_ai_rsi_mtf_4"] = out["mtf4"].astype(float)
    out["sc_ai_rsi_mtf_5"] = out["mtf5"].astype(float)
    out["sc_ai_rsi_mtf_avg"] = out["mtfAvg"].astype(float)
    out["sc_ai_rsi_mtf_strength"] = out["aiMtfStrength"].astype(float)
    out["sc_ai_rsi_mtf_aligned"] = (out["mtfAlignedBull"] | out["mtfAlignedBear"]).astype(float)
    out["sc_ai_rsi_mtf_conflict"] = out["mtfConflict"].astype(float)

    return out


# =============================================================================
# DEBUG / CSV PARITY EXPORTS
# =============================================================================

def apply_ai_rsi_debug_exports(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV/debug export names aligned to the TradingView hidden plot labels.
    """
    out = df.copy()

    out["AI RSI Dir CSV"] = out["sc_ai_rsi_dir"].astype(float)
    out["AI RSI Strength CSV"] = out["sc_ai_rsi_strength"].astype(float)
    out["AI RSI Neutral CSV"] = out["sc_ai_rsi_is_neutral"].astype(float)
    out["AI RSI Raw CSV"] = out["sc_ai_rsi_raw"].astype(float)
    out["AI RSI Signal CSV"] = out["sc_ai_rsi_signal"].astype(float)

    out["DBG RSI Val CSV"] = out["rsiVal"].astype(float)
    out["DBG Y RSI CSV"] = out["y_rsi"].astype(float)
    out["DBG X RSI CSV"] = out["x_rsi"].astype(float)
    out["DBG Pred Z CSV"] = out["pred_rsi_z"].astype(float)
    out["DBG RSI Mean CSV"] = out["rsi_mean"].astype(float)
    out["DBG RSI Std CSV"] = out["rsi_std"].astype(float)
    out["DBG Pred RSI CSV"] = out["pred_rsi"].astype(float)
    out["DBG Raw CSV"] = out["rsiWeight"].astype(float)
    out["DBG X RSI RAW CSV"] = out["rsiVal"].shift(1).astype(float)

    return out


# =============================================================================
# MAIN ENGINE WRAPPER
# =============================================================================

def run_ai_rsi_engine(
    df: pd.DataFrame,
    rsi_len: int = 14,
    sig_len: int = 20,
    learn_len: int = 20,
    dead_zone_on: bool = True,
    dead_zone_val: float = 0.15,
    mtf_on: bool = True,
    tf1: str = "5",
    tf2: str = "15",
    tf3: str = "60",
    tf4: str = "240",
    tf5: str = "D",
) -> pd.DataFrame:
    """
    Full AI RSI engine wrapper in Pine block order:
    core -> base state -> MTF -> exports -> debug exports
    """
    out = compute_ai_rsi_core(
        df,
        rsi_len=rsi_len,
        sig_len=sig_len,
        learn_len=learn_len,
    )

    out = apply_ai_rsi_base_state(
        out,
        dead_zone_on=dead_zone_on,
        dead_zone_val=dead_zone_val,
    )

    out = apply_ai_rsi_mtf_layer(
        out,
        mtf_on=mtf_on,
        tf1=tf1,
        tf2=tf2,
        tf3=tf3,
        tf4=tf4,
        tf5=tf5,
        rsi_len=rsi_len,
        sig_len=sig_len,
        learn_len=learn_len,
    )

    out = apply_ai_rsi_exports(out)
    out = apply_ai_rsi_debug_exports(out)

    return out