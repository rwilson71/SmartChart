from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from core.cc_ema_distance_calibration import (
    EmaDistanceConfig,
    build_feature_frame,
)


@dataclass
class EmaDistanceCalibrationConfig:
    forward_bars: int = 20
    continue_up_threshold_pct: float = 0.20
    continue_down_threshold_pct: float = -0.20
    reversal_up_threshold_pct: float = 0.20
    reversal_down_threshold_pct: float = -0.20
    use_research_signal_only: bool = True


def _future_return_pct(close: pd.Series, forward_bars: int) -> pd.Series:
    future_close = close.shift(-forward_bars)
    out = np.where(close != 0.0, ((future_close - close) / close) * 100.0, np.nan)
    return pd.Series(out, index=close.index, dtype=float)


def run_ema_distance_calibration(
    df: pd.DataFrame,
    cfg: Optional[EmaDistanceConfig] = None,
    research_cfg: Optional[EmaDistanceCalibrationConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or EmaDistanceConfig()
    research_cfg = research_cfg or EmaDistanceCalibrationConfig()

    features = build_feature_frame(df, cfg).copy()

    if features.empty:
        return {
            "ready": False,
            "message": "No data available for calibration.",
            "config": asdict(cfg),
            "research_config": asdict(research_cfg),
            "summary": [],
        }

    close_col = cfg.close_col
    if close_col not in features.columns:
        raise ValueError(f"Close column '{close_col}' not found in feature frame.")

    features["forward_return_pct"] = _future_return_pct(
        features[close_col].astype(float),
        research_cfg.forward_bars,
    )

    if research_cfg.use_research_signal_only:
        sample = features[features["research_signal"] == True].copy()
    else:
        sample = features.copy()

    sample = sample.dropna(subset=["forward_return_pct"]).copy()

    if sample.empty:
        return {
            "ready": False,
            "message": "No valid samples after filters.",
            "config": asdict(cfg),
            "research_config": asdict(research_cfg),
            "summary": [],
        }

    def classify_outcome(row: pd.Series) -> str:
        direction = int(row["trend_side"])
        fwd = float(row["forward_return_pct"])

        if direction == 1:
            if fwd >= research_cfg.continue_up_threshold_pct:
                return "continuation"
            if fwd <= research_cfg.reversal_down_threshold_pct:
                return "reversal"
            return "neutral"

        if direction == -1:
            if fwd <= research_cfg.continue_down_threshold_pct:
                return "continuation"
            if fwd >= research_cfg.reversal_up_threshold_pct:
                return "reversal"
            return "neutral"

        return "neutral"

    sample["outcome"] = sample.apply(classify_outcome, axis=1)

    grouped = []
    for stage in [1, 2, 3]:
        block = sample[sample["stage"] == stage].copy()

        if block.empty:
            grouped.append(
                {
                    "stage": stage,
                    "stage_text": f"ZONE {stage}",
                    "samples": 0,
                    "continuation_rate": None,
                    "reversal_rate": None,
                    "neutral_rate": None,
                    "avg_forward_return_pct": None,
                }
            )
            continue

        continuation_rate = (block["outcome"] == "continuation").mean() * 100.0
        reversal_rate = (block["outcome"] == "reversal").mean() * 100.0
        neutral_rate = (block["outcome"] == "neutral").mean() * 100.0
        avg_forward_return_pct = block["forward_return_pct"].mean()

        grouped.append(
            {
                "stage": stage,
                "stage_text": f"ZONE {stage}",
                "samples": int(len(block)),
                "continuation_rate": float(continuation_rate),
                "reversal_rate": float(reversal_rate),
                "neutral_rate": float(neutral_rate),
                "avg_forward_return_pct": float(avg_forward_return_pct),
            }
        )

    return {
        "ready": True,
        "config": asdict(cfg),
        "research_config": asdict(research_cfg),
        "total_samples": int(len(sample)),
        "summary": grouped,
    }