import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from core.j_fvg import build_fvg_latest_payload, FvgConfig

router = APIRouter(prefix="/website/fvg", tags=["FVG"])

DATA_PATH = Path("data/xauusd.csv")
CACHE_PATH = Path("data/cache/fvg_latest.json")


def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.dropna(subset=[col]).set_index(col)
            break

    return df


@router.get("/latest")
def get_fvg_latest():
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        df = load_price_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="FVG dataset is empty")

        return build_fvg_latest_payload(
            df,
            FvgConfig(symbol="XAUUSD", timeframe="1m"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FVG live build failed: {e}")