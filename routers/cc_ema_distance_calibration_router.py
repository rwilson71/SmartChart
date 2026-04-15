from __future__ import annotations

from fastapi import APIRouter, HTTPException
import pandas as pd

from core.cc_ema_distance_calibration import (
    run_ema_distance_calibration,
    CalibrationConfig,
)

router = APIRouter(prefix="/website/ema-distance", tags=["EMA Distance"])


# =============================================================================
# CONFIG (you can tune later)
# =============================================================================

DEFAULT_CFG = CalibrationConfig(
    continuation_points=30,
    reversal_points=30,
    max_forward_bars=60,
    point_value=0.01,
    time_col="datetime",
)


DATA_FILE = "data/xauusd_m1_full.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data load error: {e}")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


# =============================================================================
# ENDPOINT: LATEST (FOR WEBSITE CARD)
# =============================================================================

@router.get("/latest")
def get_latest():
    df = load_data()

    if df.empty:
        raise HTTPException(status_code=400, detail="No data available")

    result = run_ema_distance_calibration(df, DEFAULT_CFG)

    latest = result.get("latest", {})
    headline = result.get("headline", {})

    return {
        "indicator": "EMA Distance Calibration",
        "trend": latest.get("trend_text"),
        "signal": latest.get("research_signal"),
        "stage": latest.get("stage_text"),
        "bucket": latest.get("bucket_text"),
        "distance_pct": latest.get("abs_e20_to_e200_pct"),
        "signed_pct": latest.get("e20_to_e200_pct"),
        "ema20_slope": latest.get("ema20_slope_pct"),
        "ema200_slope": latest.get("ema200_slope_pct"),
        "stats": {
            "signals": headline.get("signals"),
            "continuation_rate": headline.get("continuation_rate"),
            "reversal_rate": headline.get("reversal_rate"),
        },
    }


# =============================================================================
# ENDPOINT: FULL (FOR DEBUG / TABLE / FUTURE UI)
# =============================================================================

@router.get("/full")
def get_full():
    df = load_data()

    if df.empty:
        raise HTTPException(status_code=400, detail="No data available")

    result = run_ema_distance_calibration(df, DEFAULT_CFG)

    return {
        "headline": result["headline"],
        "recommendation": result["recommendation"],
        "latest": result["latest"],
        "bucket_summary": result["bucket_summary"].to_dict(orient="records")
        if not result["bucket_summary"].empty else [],
    }