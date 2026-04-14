# =============================================================================
# SMARTCHART — BROKER ROUTER
# =============================================================================
# Role:
# - Consume playbook output only
# - Build execution-ready broker routing rows
# - Support multi-broker architecture
# - Never generate signals
# - Never replace truth/playbook as source
# =============================================================================

from __future__ import annotations

import pandas as pd
from typing import Any


# =============================================================================
# HELPERS
# =============================================================================

def _latest_row(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="object")
    return df.iloc[-1]


def _side_from_dir(value: Any) -> str:
    try:
        v = int(value)
    except Exception:
        v = 0

    if v > 0:
        return "buy"
    if v < 0:
        return "sell"
    return "flat"


def _default_broker_config() -> dict[str, dict[str, Any]]:
    """
    Default broker registry.
    This can later be replaced by database / env / API config.
    """
    return {
        "broker_demo_1": {
            "enabled": True,
            "mode": "demo",
            "broker_name": "broker_demo_1",
            "symbols": {},
        },
        "broker_live_1": {
            "enabled": False,
            "mode": "live",
            "broker_name": "broker_live_1",
            "symbols": {},
        },
    }


def _map_symbol(symbol: str, broker_cfg: dict[str, Any]) -> str:
    """
    Optional broker-specific symbol mapping.
    Example:
        XAUUSD -> XAUUSD.r
    """
    symbol_map = broker_cfg.get("symbols", {}) or {}
    return symbol_map.get(symbol, symbol)


# =============================================================================
# SINGLE-INSTRUMENT ROUTE
# =============================================================================

def build_broker_row(
    symbol: str,
    truth: pd.DataFrame,
    playbook: pd.DataFrame,
    broker_name: str,
    broker_cfg: dict[str, Any],
) -> pd.DataFrame:
    """
    Build one execution-ready broker route row for the latest bar.
    """

    t = _latest_row(truth)
    p = _latest_row(playbook)

    playbook_active = int(p.get("playbook_active", 0)) if len(p) > 0 else 0
    playbook_dir = int(p.get("playbook_dir", 0)) if len(p) > 0 else 0
    route_enabled = bool(broker_cfg.get("enabled", False))

    row = {
        # Instrument / broker identity
        "symbol": symbol,
        "broker_symbol": _map_symbol(symbol, broker_cfg),
        "broker_name": broker_name,
        "broker_mode": broker_cfg.get("mode", "demo"),
        "broker_enabled": int(route_enabled),

        # Time / state
        "timestamp": t.get("timestamp", truth.index[-1] if len(truth) else None),
        "dir": t.get("dir", 0),
        "dir_label": t.get("dir_label", "neutral"),
        "truth_ready": t.get("truth_ready", 0),
        "truth_strength": t.get("truth_strength", 0),

        # Playbook execution source
        "playbook_active": playbook_active,
        "playbook_id": p.get("playbook_id", 0),
        "playbook_name": p.get("playbook_name", "NONE"),
        "playbook_dir": playbook_dir,
        "playbook_label": p.get("playbook_label", "neutral"),
        "side": _side_from_dir(playbook_dir),

        # Setup support info
        "s1_active": p.get("s1_active", 0),
        "s1_long": p.get("s1_long", 0),
        "s1_short": p.get("s1_short", 0),
        "s15_active": p.get("s15_active", 0),
        "s15_long": p.get("s15_long", 0),
        "s15_short": p.get("s15_short", 0),

        # Forecast / permission
        "forecast_permission": t.get("forecast_permission", 0),

        # Route permission
        "route_ready": int(route_enabled and playbook_active == 1 and playbook_dir != 0),

        # Placeholder execution fields
        "order_type": "market" if playbook_active == 1 and playbook_dir != 0 else "none",
        "qty": 0.0,
        "risk_pct": 0.0,
        "sl_price": None,
        "tp_price": None,

        # Payload state
        "payload_ready": int(route_enabled and playbook_active == 1 and playbook_dir != 0),
        "route_status": (
            "ready"
            if route_enabled and playbook_active == 1 and playbook_dir != 0
            else "blocked"
        ),
    }

    return pd.DataFrame([row])


# =============================================================================
# MULTI-BROKER ROUTER
# =============================================================================

def build_broker_router(
    symbol: str,
    truth: pd.DataFrame,
    playbook: pd.DataFrame,
    broker_config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build a routing table for one symbol across many brokers.
    """

    if broker_config is None:
        broker_config = _default_broker_config()

    rows = []

    for broker_name, broker_cfg in broker_config.items():
        row_df = build_broker_row(
            symbol=symbol,
            truth=truth,
            playbook=playbook,
            broker_name=broker_name,
            broker_cfg=broker_cfg,
        )
        rows.append(row_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    if "route_ready" in out.columns:
        out = out.sort_values(
            by=["route_ready", "broker_enabled"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return out


# =============================================================================
# MULTI-INSTRUMENT + MULTI-BROKER ROUTER
# =============================================================================

def build_multi_broker_router(
    scan_map: dict[str, dict[str, pd.DataFrame]],
    broker_config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build full routing table across multiple instruments and multiple brokers.

    Expected:
    scan_map = {
        "XAUUSD": {"truth": df_truth_xau, "playbook": df_playbook_xau},
        "EURUSD": {"truth": df_truth_eur, "playbook": df_playbook_eur},
    }
    """

    if broker_config is None:
        broker_config = _default_broker_config()

    rows = []

    for symbol, payload in scan_map.items():
        truth = payload.get("truth", pd.DataFrame())
        playbook = payload.get("playbook", pd.DataFrame())

        routed = build_broker_router(
            symbol=symbol,
            truth=truth,
            playbook=playbook,
            broker_config=broker_config,
        )
        rows.append(routed)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    if {"route_ready", "truth_strength"}.issubset(out.columns):
        out = out.sort_values(
            by=["route_ready", "truth_strength"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return out


# =============================================================================
# OPTIONAL TEST BLOCK
# =============================================================================

if __name__ == "__main__":
    truth_xau = pd.DataFrame(
        [{
            "timestamp": "2026-04-07 11:00:00",
            "dir": 1,
            "dir_label": "bullish",
            "truth_ready": 1,
            "truth_strength": 8,
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
            "timestamp": "2026-04-07 11:00:00",
            "dir": -1,
            "dir_label": "bearish",
            "truth_ready": 1,
            "truth_strength": 6,
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
            "s1_active": 0,
            "s1_long": 0,
            "s1_short": 0,
            "s15_active": 1,
            "s15_long": 0,
            "s15_short": 1,
        }]
    )

    broker_cfg = {
        "broker_demo_1": {
            "enabled": True,
            "mode": "demo",
            "broker_name": "broker_demo_1",
            "symbols": {"XAUUSD": "XAUUSD", "EURUSD": "EURUSD"},
        },
        "broker_live_1": {
            "enabled": False,
            "mode": "live",
            "broker_name": "broker_live_1",
            "symbols": {"XAUUSD": "GOLD", "EURUSD": "EURUSD"},
        },
    }

    scan_map = {
        "XAUUSD": {"truth": truth_xau, "playbook": playbook_xau},
        "EURUSD": {"truth": truth_eur, "playbook": playbook_eur},
    }

    routed = build_multi_broker_router(scan_map, broker_cfg)
    print(routed)