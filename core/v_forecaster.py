from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from core.va_seasonality import build_seasonality_latest_payload, SeasonalityConfig


def build_forecaster_latest_payload(
    df: pd.DataFrame,
    seasonality_config: Optional[SeasonalityConfig] = None,
) -> Dict[str, Any]:
    """
    Master Forecaster payload builder.

    Phase 1:
    - seasonality only

    Later phases:
    - correlation
    - quant
    - scanner
    - macro
    - pattern
    - probability fusion
    """
    seasonality = build_seasonality_latest_payload(
        df=df,
        config=seasonality_config,
    )

    payload: Dict[str, Any] = {
        "indicator": "FORECASTER",
        "timestamp": seasonality.get("timestamp"),
        "state": seasonality.get("state", "NEUTRAL"),
        "bias_signal": seasonality.get("bias_signal", 0),
        "bias_label": seasonality.get("bias_label", "NEUTRAL"),
        "indicator_strength": seasonality.get("indicator_strength", 0.0),
        "modules_loaded": {
            "seasonality": True,
            "correlation": False,
            "quant": False,
            "scanner": False,
            "macro": False,
            "pattern": False,
            "probability": False,
        },
        "seasonality": seasonality,
    }

    return payload