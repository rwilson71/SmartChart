from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from core.cc_ema_distance_calibration import (
    EmaDistanceConfig,
    build_latest_payload,
)

from research.cc_ema_distance_research import (
    EmaDistanceCalibrationConfig,
    run_ema_distance_calibration,
)

router = APIRouter(prefix="/website/ema-distance", tags=["EMA Distance"])
research_router = APIRouter(prefix="/research/ema-distance", tags=["EMA Distance Research"])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_FILE = DATA_DIR / "xauusd.csv"

# =============================================================================
# HELPERS
# =============================================================================

def _standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the loader resilient for website use.

    Supports common SmartChart / Twelve Data style variants such as:
    - datetime, open, high, low, close, volume
    - time, open, high, low, close, volume
    - Datetime / Open / High / Low / Close
    - date as the timestamp column
    """
    out = df.copy()

    # Normalize headers
    out.columns = [str(c).strip().lower() for c in out.columns]

    rename_map = {}
    if "datetime" in out.columns and "time" not in out.columns:
        rename_map["datetime"] = "time"
    elif "date" in out.columns and "time" not in out.columns:
        rename_map["date"] = "time"

    out = out.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Price file is missing required columns: {missing}. "
            f"Found columns: {list(out.columns)}"
        )

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    subset = ["open", "high", "low", "close"]
    if "time" in out.columns:
        subset.append("time")

    out = out.dropna(subset=subset).copy()

    if "time" in out.columns:
        out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")

    return out.reset_index(drop=True)


def load_price_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    file_path = Path(csv_path) if csv_path else DEFAULT_DATA_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = _standardize_price_columns(df)
    return df


# =============================================================================
# WEBSITE ROUTE
# =============================================================================

@router.get("/latest")
def get_ema_distance_latest(
    csv_path: Optional[str] = Query(default=None),
):
    try:
        df = load_price_data(csv_path)
        cfg = EmaDistanceConfig(time_col="time" if "time" in df.columns else None)
        return build_latest_payload(df, cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# RESEARCH ROUTE
# =============================================================================

@research_router.get("/calibration")
def get_ema_distance_calibration(
    csv_path: Optional[str] = Query(default=None),
    continuation_points: float = Query(default=60.0),
    reversal_points: float = Query(default=30.0),
    max_forward_bars: int = Query(default=60),
    point_value: float = Query(default=1.0),
):
    try:
        df = load_price_data(csv_path)
        cfg = EmaDistanceCalibrationConfig(
            continuation_points=continuation_points,
            reversal_points=reversal_points,
            max_forward_bars=max_forward_bars,
            point_value=point_value,
            time_col="time" if "time" in df.columns else None,
        )
        result = run_ema_distance_calibration(df, cfg)

        return {
            "headline": result["headline"],
            "recommendation": result["recommendation"],
            "bucket_summary": result["bucket_with_zones"].to_dict(orient="records")
            if not result["bucket_with_zones"].empty else [],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc