
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
from core.playbook.w_strategy_s1 import build_strategy_s1
from core.playbook.w_strategy_s1_5 import build_strategy_s1_5
from routers.trend_api import router as trend_router
from routers.ema_api import router as ema_router
from routers.cc_ema_distance_router import router as ema_distance_router
from routers.momentum_api import router as momentum_router
from routers.ai_rsi_router import router as ai_rsi_router
from routers.market_structure_api import router as market_structure_router
from routers.session_daily_api import router as session_daily_router
from routers.fib_api import router as fib_router
from routers.liquidity_api import router as liquidity_router
from routers.orderflow_api import router as orderflow_router
from routers.confluence_cloud_api import router as confluence_cloud_router
from routers.volatility_api import router as volatility_router
from routers.ob_os_api import router as ob_os_router
from routers.exhaustion_api import router as exhaustion_router
from routers.mfi_api import router as mfi_router
from routers.confluence_api import router as confluence_router
from routers.macd_reversal_api import router as macd_reversal_router
from routers.pullback_retest_api import router as pullback_retest_router
from routers.regime_api import router as regime_router
from routers.fvg_api import router as fvg_router
from routers.volume_api import router as volume_router
from routers.x_1h_4h_first_candle_api import router as htf_first_candle_router
from routers.w_5m_15m_first_candle_api import router as ltf_first_candle_router
from routers.y_mtf_router import router as y_mtf_router
# =============================================================================
# CONFIG
# =============================================================================

APP_TITLE = "SmartChart API"
APP_VERSION = "0.4.0"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_FILE = DATA_DIR / "xauusd.csv"

CACHE_TTL_SECONDS = 120.0

_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "raw": {},
    "truth": {},
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
    allow_origins=[
        "https://lbwtsmartchart.co.uk",
        "https://www.lbwtsmartchart.co.uk",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ema_distance_router)
app.include_router(trend_router)
app.include_router(ema_router)
app.include_router(momentum_router)
app.include_router(ai_rsi_router)
app.include_router(market_structure_router)
app.include_router(session_daily_router)
app.include_router(fib_router)
app.include_router(liquidity_router)
app.include_router(orderflow_router)
app.include_router(confluence_cloud_router)
app.include_router(volatility_router)
app.include_router(ob_os_router)
app.include_router(exhaustion_router)
app.include_router(mfi_router)
app.include_router(confluence_router)
app.include_router(macd_reversal_router)
app.include_router(pullback_retest_router)
app.include_router(regime_router)
app.include_router(fvg_router)
app.include_router(volume_router)
app.include_router(htf_first_candle_router)
app.include_router(ltf_first_candle_router)
app.include_router(y_mtf_router)
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

    # Preserve timestamp column while also providing a DatetimeIndex
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