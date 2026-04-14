from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from core.truth_engine import build_truth
from core.b_trend import build_trend
from core.c_ema import build_ema
from core.playbook.w_strategy_s1 import build_strategy_s1
from core.playbook.w_strategy_s1_5 import build_strategy_s1_5


# =============================================================================
# CONFIG
# =============================================================================

APP_TITLE = "SmartChart API"
APP_VERSION = "0.4.0"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_FILE = DATA_DIR / "xauusd.csv"

# Small TTL for fast testing while still preventing repeated rebuilds
CACHE_TTL_SECONDS = 10.0

_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "raw": {},
    "truth": {},
    "trend": {},
    "ema": {},
    "s1": {},
    "s15": {},
}

_CACHE_LOCK = RLock()


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="SmartChart backend API for truth table and website consumption.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPERS
# =============================================================================

def _resolve_data_file(filename: Optional[str] = None) -> Path:
    """
    Resolve a CSV file inside the local data folder safely.
    Prevents path traversal and rejects files outside DATA_DIR.
    """
    if filename:
        requested = Path(filename).name
        path = (DATA_DIR / requested).resolve()
    else:
        path = DEFAULT_DATA_FILE.resolve()

    data_dir_resolved = DATA_DIR.resolve()

    if not str(path).startswith(str(data_dir_resolved)):
        raise FileNotFoundError("Invalid data file path.")

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.name}")

    if path.suffix.lower() != ".csv":
        raise FileNotFoundError("Only CSV files are supported.")

    return path


def _file_signature(path: Path) -> Tuple[str, int, int]:
    """
    Build a lightweight signature for cache invalidation.
    Cache is invalidated early if file name, mtime, or size changes.
    """
    stat = path.stat()
    return (path.name, int(stat.st_mtime_ns), int(stat.st_size))


def _cache_key(path: Path) -> str:
    return path.name


def _load_input_dataframe(path: Path) -> pd.DataFrame:
    """
    Load CSV data into a DataFrame and normalize timestamp columns.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    timestamp_candidates = ["timestamp", "datetime", "time", "date"]
    found_ts = next((c for c in timestamp_candidates if c in df.columns), None)

    if found_ts:
        df[found_ts] = pd.to_datetime(df[found_ts], errors="coerce")
        if found_ts != "timestamp":
            df = df.rename(columns={found_ts: "timestamp"})

    # Make DatetimeIndex available for engines that require it, while preserving
    # the timestamp column for website payloads.
    if "timestamp" in df.columns:
        dt_index = pd.to_datetime(df["timestamp"], errors="coerce")
        if dt_index.notna().any():
            df = df.copy()
            df.index = dt_index

    return df


def _clean_value(value: Any) -> Any:
    """
    Convert pandas / numpy values into JSON-safe Python values.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _records_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Faster than iterrows() for API response shaping.
    """
    records = df.to_dict(orient="records")
    cleaned: List[Dict[str, Any]] = []

    for row in records:
        cleaned.append({str(k): _clean_value(v) for k, v in row.items()})

    return cleaned


def _row_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {str(k): _clean_value(v) for k, v in row.items()}


def _get_cached(section: str, key: str, sig: Tuple[str, int, int]) -> Optional[pd.DataFrame]:
    with _CACHE_LOCK:
        entry = _CACHE.get(section, {}).get(key)
        if not entry:
            return None

        now = time.time()

        if now - entry["ts"] > CACHE_TTL_SECONDS:
            return None

        if entry.get("sig") != sig:
            return None

        return entry["value"]


def _set_cached(section: str, key: str, sig: Tuple[str, int, int], value: pd.DataFrame) -> None:
    with _CACHE_LOCK:
        _CACHE.setdefault(section, {})[key] = {
            "ts": time.time(),
            "sig": sig,
            "value": value,
        }


def _build_truth_from_path(path: Path) -> pd.DataFrame:
    key = _cache_key(path)
    sig = _file_signature(path)

    cached_truth = _get_cached("truth", key, sig)
    if cached_truth is not None:
        return cached_truth

    raw_cached = _get_cached("raw", key, sig)
    if raw_cached is not None:
        raw_df = raw_cached
    else:
        raw_df = _load_input_dataframe(path)
        _set_cached("raw", key, sig, raw_df)

    truth_df = build_truth(raw_df)

    if truth_df.empty:
        raise ValueError("Truth DataFrame is empty.")

    truth_df = truth_df.copy()
    _set_cached("truth", key, sig, truth_df)
    return truth_df


def _build_truth_from_file(filename: Optional[str] = None) -> pd.DataFrame:
    path = _resolve_data_file(filename)
    return _build_truth_from_path(path)


def _build_trend_from_file(filename: Optional[str] = None) -> pd.DataFrame:
    path = _resolve_data_file(filename)
    key = _cache_key(path)
    sig = _file_signature(path)

    cached_trend = _get_cached("trend", key, sig)
    if cached_trend is not None:
        return cached_trend

    raw_cached = _get_cached("raw", key, sig)
    if raw_cached is not None:
        raw_df = raw_cached
    else:
        raw_df = _load_input_dataframe(path)
        _set_cached("raw", key, sig, raw_df)

    trend_df = build_trend(raw_df)

    if trend_df.empty:
        raise ValueError("Trend DataFrame is empty.")

    trend_df = trend_df.copy()
    _set_cached("trend", key, sig, trend_df)
    return trend_df


def _build_ema_from_file(filename: Optional[str] = None) -> pd.DataFrame:
    path = _resolve_data_file(filename)
    key = _cache_key(path)
    sig = _file_signature(path)

    cached_ema = _get_cached("ema", key, sig)
    if cached_ema is not None:
        return cached_ema

    raw_cached = _get_cached("raw", key, sig)
    if raw_cached is not None:
        raw_df = raw_cached
    else:
        raw_df = _load_input_dataframe(path)
        _set_cached("raw", key, sig, raw_df)

    ema_df = build_ema(raw_df)

    if ema_df.empty:
        raise ValueError("EMA DataFrame is empty.")

    ema_df = ema_df.copy()
    _set_cached("ema", key, sig, ema_df)
    return ema_df


def _build_s1_from_file(filename: Optional[str] = None) -> pd.DataFrame:
    path = _resolve_data_file(filename)
    key = _cache_key(path)
    sig = _file_signature(path)

    cached_s1 = _get_cached("s1", key, sig)
    if cached_s1 is not None:
        return cached_s1

    truth_df = _build_truth_from_path(path)
    s1_df = build_strategy_s1(truth_df)

    if s1_df.empty:
        raise ValueError("S1 DataFrame is empty.")

    s1_df = s1_df.copy()
    _set_cached("s1", key, sig, s1_df)
    return s1_df


def _build_s15_from_file(filename: Optional[str] = None) -> pd.DataFrame:
    path = _resolve_data_file(filename)
    key = _cache_key(path)
    sig = _file_signature(path)

    cached_s15 = _get_cached("s15", key, sig)
    if cached_s15 is not None:
        return cached_s15

    truth_df = _build_truth_from_path(path)
    s15_df = build_strategy_s1_5(truth_df)

    if s15_df.empty:
        raise ValueError("S1.5 DataFrame is empty.")

    s15_df = s15_df.copy()
    _set_cached("s15", key, sig, s15_df)
    return s15_df


def _website_payload(strategy_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]

    return {
        "status": "ok",
        "strategy": strategy_name,
        "price": _clean_value(latest.get("close")),
        "direction": _clean_value(latest.get("direction")),
        "state": _clean_value(latest.get("state")),
        "grade": _clean_value(latest.get("grade")),
        "trade_ready": _clean_value(latest.get("trade_ready")),
        "reason": _clean_value(latest.get("reason")),
        "entry_type": _clean_value(latest.get("entry_type")),
        "timestamp": _clean_value(latest.get("timestamp")),
    }


def _website_trend_payload(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]

    trend_dir_num = int(_clean_value(latest.get("sc_trend_dir")) or 0)
    regime_text = str(_clean_value(latest.get("sc_regime_text")) or "Neutral")
    quality_text = str(_clean_value(latest.get("trend_quality_text")) or "Neutral")
    stage = int(_clean_value(latest.get("sc_trend_stage")) or 0)

    if trend_dir_num > 0:
        direction = "Bullish"
        direction_color = "green"
    elif trend_dir_num < 0:
        direction = "Bearish"
        direction_color = "red"
    else:
        direction = "Neutral"
        direction_color = "gray"

    return {
        "status": "ok",
        "indicator": "Trend",
        "price": _clean_value(latest.get("close")),
        "timestamp": _clean_value(latest.get("timestamp")),
        "direction": direction,
        "direction_color": direction_color,
        "state": _clean_value(latest.get("sc_trend_state")),
        "stage": stage,
        "strength": _clean_value(latest.get("sc_trend_strength")),
        "continuation_quality": _clean_value(latest.get("sc_continuation_quality")),
        "decay": _clean_value(latest.get("sc_trend_decay")),
        "regime": regime_text,
        "quality": quality_text,
        "mtf_bias": _clean_value(latest.get("sc_mtf_bias")),
        "mtf_alignment": _clean_value(latest.get("sc_mtf_alignment")),
        "long_signal": _clean_value(latest.get("trend_long_signal")),
        "short_signal": _clean_value(latest.get("trend_short_signal")),
        "bull_weak_signal": _clean_value(latest.get("bull_weak_signal")),
        "bear_weak_signal": _clean_value(latest.get("bear_weak_signal")),
    }


def _website_ema_payload(df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]

    ema_dir_num = int(_clean_value(latest.get("sc_ema_final_dir")) or 0)
    behavior_type = int(_clean_value(latest.get("sc_ema_behavior_type")) or 0)
    trend_quality = int(_clean_value(latest.get("sc_ema_trend_quality")) or 0)
    rt_family = int(_clean_value(latest.get("sc_ema_rt_family")) or 0)

    if ema_dir_num > 0:
        direction = "Bullish"
        direction_color = "green"
    elif ema_dir_num < 0:
        direction = "Bearish"
        direction_color = "red"
    else:
        direction = "Neutral"
        direction_color = "gray"

    behavior_map = {
        0: "Neutral",
        1: "Compression",
        2: "Expansion",
        3: "Decay",
    }

    quality_map = {
        0: "Neutral",
        1: "Clean",
        2: "Strong",
    }

    rt_family_map = {
        0: "None",
        1: "EMA 14-20",
        2: "EMA 33-50",
        3: "EMA 100-200",
    }

    return {
        "status": "ok",
        "indicator": "EMA",
        "price": _clean_value(latest.get("close")),
        "timestamp": _clean_value(latest.get("timestamp")),

        "direction": direction,
        "direction_color": direction_color,

        "local_dir": _clean_value(latest.get("sc_ema_local_dir")),
        "final_dir_score": _clean_value(latest.get("sc_ema_final_dir_score")),

        "behavior": behavior_map.get(behavior_type, "Neutral"),
        "behavior_type": behavior_type,

        "quality": quality_map.get(trend_quality, "Neutral"),
        "quality_state": trend_quality,

        "compression": _clean_value(latest.get("sc_ema_compression")),
        "fast_compression": _clean_value(latest.get("sc_ema_fast_compression")),
        "slow_compression": _clean_value(latest.get("sc_ema_slow_compression")),
        "slow_expansion": _clean_value(latest.get("sc_ema_slow_expansion")),

        "rt_any": _clean_value(latest.get("sc_ema_rt_any")),
        "rt_family": rt_family_map.get(rt_family, "None"),
        "rt_family_state": rt_family,
        "rt_dir": _clean_value(latest.get("ema_rt_dir")),

        "reclaim_20": _clean_value(latest.get("sc_ema_reclaim_20")),
        "reclaim_3350": _clean_value(latest.get("sc_ema_reclaim_3350")),
        "reclaim_100200": _clean_value(latest.get("sc_ema_reclaim_100200")),

        "price_to_ema20_pct": _clean_value(latest.get("sc_price_to_ema20_pct")),
        "ema20_to_ema200_pct": _clean_value(latest.get("sc_ema20_to_ema200_pct")),

        "mtf_avg_dir": _clean_value(latest.get("sc_ema_mtf_avg_dir")),
        "is_stretched_from20": _clean_value(latest.get("sc_is_stretched_from20")),
        "is_stretched_from200": _clean_value(latest.get("sc_is_stretched_from200")),
    }


def _table_payload(name: str, df: pd.DataFrame, limit: int) -> Dict[str, Any]:
    sliced = df.tail(limit)

    return {
        "status": "ok",
        "strategy": name,
        "row_count": int(len(df)),
        "returned_rows": int(len(sliced)),
        "columns": list(map(str, df.columns)),
        "rows": _records_from_df(sliced),
    }


def _latest_payload(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    latest = df.iloc[-1]

    return {
        "status": "ok",
        "strategy": name,
        "row_count": int(len(df)),
        "latest": _row_to_dict(latest),
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
    }


@app.get("/health")
def health(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        path = _resolve_data_file(filename)
        raw_df = _load_input_dataframe(path)
        truth_df = _build_truth_from_path(path)

        return {
            "status": "ok",
            "data_file": path.name,
            "rows_in_input": int(len(raw_df)),
            "rows_in_truth": int(len(truth_df)),
            "columns_in_truth": list(map(str, truth_df.columns)),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/cache/status")
def cache_status() -> Dict[str, Any]:
    """
    Helpful for debugging website/API refresh behavior.
    """
    with _CACHE_LOCK:
        summary: Dict[str, Any] = {}

        for section, entries in _CACHE.items():
            summary[section] = {
                "keys": list(entries.keys()),
                "count": len(entries),
            }

    return {
        "status": "ok",
        "cache": summary,
        "ttl_seconds": CACHE_TTL_SECONDS,
    }


@app.post("/cache/clear")
def clear_cache() -> Dict[str, Any]:
    with _CACHE_LOCK:
        for section in _CACHE:
            _CACHE[section].clear()

    return {
        "status": "ok",
        "message": "Cache cleared.",
    }


# =============================================================================
# TRUTH ROUTES
# =============================================================================

@app.get("/truth/latest")
def truth_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        truth = _build_truth_from_file(filename)
        latest = truth.iloc[-1]

        return {
            "status": "ok",
            "row_count": int(len(truth)),
            "latest": _row_to_dict(latest),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/truth/table")
def truth_table(
    filename: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        truth = _build_truth_from_file(filename)
        sliced = truth.tail(limit)

        return {
            "status": "ok",
            "row_count": int(len(truth)),
            "returned_rows": int(len(sliced)),
            "columns": list(map(str, truth.columns)),
            "rows": _records_from_df(sliced),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/truth/columns")
def truth_columns(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        truth = _build_truth_from_file(filename)
        return {
            "status": "ok",
            "columns": list(map(str, truth.columns)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# TREND ROUTES
# =============================================================================

@app.get("/indicator/trend/latest")
def trend_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        trend = _build_trend_from_file(filename)
        return {
            "status": "ok",
            "indicator": "Trend",
            "row_count": int(len(trend)),
            "latest": _row_to_dict(trend.iloc[-1]),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/indicator/trend/table")
def trend_table(
    filename: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        trend = _build_trend_from_file(filename)
        sliced = trend.tail(limit)

        return {
            "status": "ok",
            "indicator": "Trend",
            "row_count": int(len(trend)),
            "returned_rows": int(len(sliced)),
            "columns": list(map(str, trend.columns)),
            "rows": _records_from_df(sliced),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/indicator/trend/columns")
def trend_columns(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        trend = _build_trend_from_file(filename)
        return {
            "status": "ok",
            "indicator": "Trend",
            "columns": list(map(str, trend.columns)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/website/trend/latest")
def website_trend_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        truth = _build_truth_from_file(filename)
        latest = truth.iloc[-1]

        trend_dir_num = int(_clean_value(latest.get("trend_dir")) or 0)
        regime_text = str(_clean_value(latest.get("regime_label")) or "neutral").title()
        quality_text = str(_clean_value(latest.get("trend_label")) or "Neutral").title()

        if trend_dir_num > 0:
            direction = "Bullish"
            direction_color = "green"
        elif trend_dir_num < 0:
            direction = "Bearish"
            direction_color = "red"
        else:
            direction = "Neutral"
            direction_color = "gray"

        return {
            "status": "ok",
            "indicator": "Trend",
            "price": _clean_value(latest.get("close")),
            "timestamp": _clean_value(latest.get("timestamp")),
            "direction": direction,
            "direction_color": direction_color,
            "state": _clean_value(latest.get("trend_dir")),
            "stage": None,
            "strength": _clean_value(latest.get("trend_strength")),
            "continuation_quality": _clean_value(latest.get("continuation_quality")),
            "decay": None,
            "regime": regime_text,
            "quality": quality_text,
            "mtf_bias": _clean_value(latest.get("bias_strength")),
            "mtf_alignment": None,
            "long_signal": None,
            "short_signal": None,
            "bull_weak_signal": None,
            "bear_weak_signal": None,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# =============================================================================
# EMA ROUTES
# =============================================================================

@app.get("/indicator/ema/latest")
def ema_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        ema_df = _build_ema_from_file(filename)
        return {
            "status": "ok",
            "indicator": "EMA",
            "row_count": int(len(ema_df)),
            "latest": _row_to_dict(ema_df.iloc[-1]),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/indicator/ema/table")
def ema_table(
    filename: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        ema_df = _build_ema_from_file(filename)
        sliced = ema_df.tail(limit)

        return {
            "status": "ok",
            "indicator": "EMA",
            "row_count": int(len(ema_df)),
            "returned_rows": int(len(sliced)),
            "columns": list(map(str, ema_df.columns)),
            "rows": _records_from_df(sliced),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/indicator/ema/columns")
def ema_columns(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        ema_df = _build_ema_from_file(filename)
        return {
            "status": "ok",
            "indicator": "EMA",
            "columns": list(map(str, ema_df.columns)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/website/ema/latest")
def website_ema_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        ema_df = _build_ema_from_file(filename)
        return _website_ema_payload(ema_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# S1 STRATEGY ROUTES
# =============================================================================

@app.get("/strategy/s1/latest")
def strategy_s1_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        s1 = _build_s1_from_file(filename)
        return _latest_payload("S1", s1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/strategy/s1/table")
def strategy_s1_table(
    filename: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        s1 = _build_s1_from_file(filename)
        return _table_payload("S1", s1, limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/strategy/s1/columns")
def strategy_s1_columns(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        s1 = _build_s1_from_file(filename)
        return {
            "status": "ok",
            "strategy": "S1",
            "columns": list(map(str, s1.columns)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/website/s1/latest")
def website_s1_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        s1 = _build_s1_from_file(filename)
        return _website_payload("S1", s1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# S1.5 STRATEGY ROUTES
# =============================================================================

@app.get("/strategy/s15/latest")
def strategy_s15_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        s15 = _build_s15_from_file(filename)
        return _latest_payload("S1.5", s15)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/strategy/s15/table")
def strategy_s15_table(
    filename: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        s15 = _build_s15_from_file(filename)
        return _table_payload("S1.5", s15, limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/strategy/s15/columns")
def strategy_s15_columns(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        s15 = _build_s15_from_file(filename)
        return {
            "status": "ok",
            "strategy": "S1.5",
            "columns": list(map(str, s15.columns)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/website/s15/latest")
def website_s15_latest(filename: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    try:
        s15 = _build_s15_from_file(filename)
        return _website_payload("S1.5", s15)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.on_event("startup")
def warm_cache() -> None:
    print("Startup cache DISABLED for testing.")