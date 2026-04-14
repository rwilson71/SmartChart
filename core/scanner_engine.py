# =============================================================================
# SMARTCHART — SCANNER ENGINE
# =============================================================================
# Role:
# - Consume truth_engine + playbook_engine outputs
# - Build one latest-row scanner snapshot per instrument
# - Stay backend-only / pandas-safe / website-ready
# - Never generate raw signals
# =============================================================================

from __future__ import annotations

import pandas as pd


# =============================================================================
# HELPERS
# =============================================================================

def _safe_value(df: pd.DataFrame, col: str, default=0):
    if col in df.columns and len(df) > 0:
        return df.iloc[-1][col]
    return default


def _latest_row(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="object")
    return df.iloc[-1]


# =============================================================================
# SINGLE-INSTRUMENT SCAN
# =============================================================================

def build_scanner_row(
    symbol: str,
    truth: pd.DataFrame,
    playbook: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one scanner snapshot row for a single instrument using
    the latest truth/playbook state.
    """

    t = _latest_row(truth)
    p = _latest_row(playbook)

    row = {
        "symbol": symbol,
        "timestamp": t.get("timestamp", truth.index[-1] if len(truth) else None),

        # Truth layer
        "dir": t.get("dir", 0),
        "dir_label": t.get("dir_label", "neutral"),
        "truth_ready": t.get("truth_ready", 0),
        "truth_state": t.get("truth_state", "not_ready"),
        "truth_strength": t.get("truth_strength", 0),

        # Context
        "regime": t.get("regime", 0),
        "regime_label": t.get("regime_label", "unknown"),
        "market_condition": t.get("market_condition", 0),
        "market_condition_label": t.get("market_condition_label", "unknown"),
        "volatility_state": t.get("volatility_state", 0),
        "volatility_label": t.get("volatility_label", "unknown"),
        "volume_state": t.get("volume_state", 0),
        "volume_label": t.get("volume_label", "unknown"),
        "session_active": t.get("session_active", 0),
        "session_name": t.get("session_name", "none"),

        # Structure / location
        "fib_zone": t.get("fib_zone", 0),
        "fib_zone_label": t.get("fib_zone_label", "none"),
        "location_aligned": t.get("location_aligned", 0),
        "confluence_score": t.get("confluence_score", 0),
        "confluence_active": t.get("confluence_active", 0),
        "cloud_active": t.get("cloud_active", 0),

        # Retests
        "rt_ema_1420": t.get("rt_ema_1420", 0),
        "rt_ema_3350": t.get("rt_ema_3350", 0),
        "rt_ema_100200": t.get("rt_ema_100200", 0),
        "rt_fib": t.get("rt_fib", 0),
        "rt_session": t.get("rt_session", 0),
        "rt_m5": t.get("rt_m5", 0),
        "rt_orderflow": t.get("rt_orderflow", 0),
        "rt_confluence_cloud": t.get("rt_confluence_cloud", 0),
        "has_retest": t.get("has_retest", 0),
        "retest_count": t.get("retest_count", 0),

        # Reversal / confirmation
        "macd_rev_active": t.get("macd_rev_active", 0),
        "macd_rev_dir": t.get("macd_rev_dir", 0),
        "divergence_flag": t.get("divergence_flag", 0),
        "exhaustion_signal": t.get("exhaustion_signal", 0),
        "mfi_signal": t.get("mfi_signal", 0),
        "pullback_active": t.get("pullback_active", 0),

        # Forecast
        "forecast_dir": t.get("forecast_dir", 0),
        "forecast_label": t.get("forecast_label", "neutral"),
        "forecast_confidence": t.get("forecast_confidence", 0),
        "forecast_agreement": t.get("forecast_agreement", 0),
        "forecast_permission": t.get("forecast_permission", 0),

        # Playbook
        "playbook_active": p.get("playbook_active", 0),
        "playbook_id": p.get("playbook_id", 0),
        "playbook_name": p.get("playbook_name", "NONE"),
        "playbook_dir": p.get("playbook_dir", 0),
        "playbook_label": p.get("playbook_label", "neutral"),
        "playbook_long": p.get("playbook_long", 0),
        "playbook_short": p.get("playbook_short", 0),

        # Setup blocks
        "s1_active": p.get("s1_active", 0),
        "s1_long": p.get("s1_long", 0),
        "s1_short": p.get("s1_short", 0),
        "s15_active": p.get("s15_active", 0),
        "s15_long": p.get("s15_long", 0),
        "s15_short": p.get("s15_short", 0),
    }

    return pd.DataFrame([row])


# =============================================================================
# MULTI-INSTRUMENT SCAN
# =============================================================================

def build_multi_scanner(scan_map: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Build a multi-instrument scanner table.

    Expected shape:
    scan_map = {
        "XAUUSD": {
            "truth": df_truth_xau,
            "playbook": df_playbook_xau,
        },
        "EURUSD": {
            "truth": df_truth_eur,
            "playbook": df_playbook_eur,
        },
    }
    """

    rows = []

    for symbol, payload in scan_map.items():
        truth = payload.get("truth", pd.DataFrame())
        playbook = payload.get("playbook", pd.DataFrame())

        row_df = build_scanner_row(
            symbol=symbol,
            truth=truth,
            playbook=playbook,
        )
        rows.append(row_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # Sort strongest first, then active playbooks first
    if "truth_strength" in out.columns:
        out = out.sort_values(
            by=["playbook_active", "truth_strength"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return out


# =============================================================================
# OPTIONAL TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    truth_xau = pd.DataFrame(
        [{
            "timestamp": "2026-04-07 10:00:00",
            "dir": 1,
            "dir_label": "bullish",
            "truth_ready": 1,
            "truth_state": "ready",
            "truth_strength": 8,
            "regime": 1,
            "regime_label": "trend",
            "market_condition": 1,
            "market_condition_label": "normal",
            "volatility_state": 1,
            "volatility_label": "normal",
            "volume_state": 1,
            "volume_label": "normal",
            "session_active": 1,
            "session_name": "london",
            "fib_zone": 2,
            "fib_zone_label": "50_61_5",
            "location_aligned": 1,
            "confluence_score": 8,
            "confluence_active": 1,
            "cloud_active": 1,
            "rt_ema_1420": 1,
            "rt_ema_3350": 0,
            "rt_ema_100200": 0,
            "rt_fib": 0,
            "rt_session": 0,
            "rt_m5": 0,
            "rt_orderflow": 0,
            "rt_confluence_cloud": 0,
            "has_retest": 1,
            "retest_count": 1,
            "macd_rev_active": 0,
            "macd_rev_dir": 0,
            "divergence_flag": 0,
            "exhaustion_signal": 0,
            "mfi_signal": 0,
            "pullback_active": 1,
            "forecast_dir": 1,
            "forecast_label": "bullish",
            "forecast_confidence": 78,
            "forecast_agreement": 4,
            "forecast_permission": 1,
        }]
    )

    playbook_xau = pd.DataFrame(
        [{
            "playbook_active": 1,
            "playbook_id": 1,
            "playbook_name": "S1",
            "playbook_dir": 1,
            "playbook_label": "long",
            "playbook_long": 1,
            "playbook_short": 0,
            "s1_active": 1,
            "s1_long": 1,
            "s1_short": 0,
            "s15_active": 0,
            "s15_long": 0,
            "s15_short": 0,
        }]
    )

    truth_eur = pd.DataFrame(
        [{
            "timestamp": "2026-04-07 10:00:00",
            "dir": -1,
            "dir_label": "bearish",
            "truth_ready": 1,
            "truth_state": "ready",
            "truth_strength": 6,
            "regime": 1,
            "regime_label": "trend",
            "market_condition": 1,
            "market_condition_label": "normal",
            "volatility_state": 2,
            "volatility_label": "expansion",
            "volume_state": 1,
            "volume_label": "normal",
            "session_active": 1,
            "session_name": "london",
            "fib_zone": 3,
            "fib_zone_label": "66_78",
            "location_aligned": 1,
            "confluence_score": 7,
            "confluence_active": 1,
            "cloud_active": 0,
            "rt_ema_1420": 0,
            "rt_ema_3350": 0,
            "rt_ema_100200": 1,
            "rt_fib": 1,
            "rt_session": 0,
            "rt_m5": 0,
            "rt_orderflow": 0,
            "rt_confluence_cloud": 0,
            "has_retest": 1,
            "retest_count": 2,
            "macd_rev_active": 1,
            "macd_rev_dir": -1,
            "divergence_flag": 1,
            "exhaustion_signal": 1,
            "mfi_signal": 1,
            "pullback_active": 1,
            "forecast_dir": -1,
            "forecast_label": "bearish",
            "forecast_confidence": 72,
            "forecast_agreement": 3,
            "forecast_permission": 1,
        }]
    )

    playbook_eur = pd.DataFrame(
        [{
            "playbook_active": 1,
            "playbook_id": 2,
            "playbook_name": "S1.5",
            "playbook_dir": -1,
            "playbook_label": "short",
            "playbook_long": 0,
            "playbook_short": 1,
            "s1_active": 0,
            "s1_long": 0,
            "s1_short": 0,
            "s15_active": 1,
            "s15_long": 0,
            "s15_short": 1,
        }]
    )

    scan_map = {
        "XAUUSD": {"truth": truth_xau, "playbook": playbook_xau},
        "EURUSD": {"truth": truth_eur, "playbook": playbook_eur},
    }

    scanner = build_multi_scanner(scan_map)
    print(scanner)