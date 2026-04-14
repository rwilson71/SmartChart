# =============================================================================
# SMARTCHART — MAIN APP WIRING
# =============================================================================
# Role:
# - Load market data
# - Run SmartChart backend pipeline
# - Build truth / playbook / scanner / broker outputs
# - Keep architecture clean and modular
# =============================================================================

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Callable, Any

import pandas as pd

from core.truth_engine import build_truth
from core.playbook_engine import build_playbook
from core.scanner_engine import build_multi_scanner
from core.broker_router import build_multi_broker_router


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODULE ACTIVATION CONFIG
# =============================================================================

MODULE_SETTINGS: dict[str, dict[str, Any]] = {
    "a_indicators": {"enabled": False, "config": {}},

    "b_trend": {"enabled": True, "config": {}},
    "c_ema": {"enabled": True, "config": {}},
    "d_momentum": {"enabled": True, "config": {}},
    "e_ai_rsi": {
        "enabled": True,
        "config": {
            "rsi_len": 14,
            "sig_len": 20,
            "learn_len": 20,
            "mtf_on": True,
            "dead_zone_on": True,
            "dead_zone_val": 0.15,
        },
    },

    "f_market_structure": {"enabled": False, "config": {}},
    "g_session_daily": {"enabled": False, "config": {}},

    "h_fib": {"enabled": True, "config": {}},
    "i_liquidity": {"enabled": True, "config": {}},

    "j_fvg": {"enabled": False, "config": {}},
    "k_orderflow": {"enabled": False, "config": {}},
    "l_confluence_cloud": {"enabled": False, "config": {}},
    "m_volume": {"enabled": False, "config": {}},
    "n_volatility": {"enabled": False, "config": {}},
    "o_ob_os": {"enabled": False, "config": {}},
    "p_exhaustion": {"enabled": False, "config": {}},
    "q_mfi": {"enabled": False, "config": {}},
    "r_confluence": {"enabled": False, "config": {}},
    "s_macd_reversal": {"enabled": False, "config": {}},
    "t_pullback_retest": {"enabled": False, "config": {}},

    "u_regime": {"enabled": True, "config": {}},
    "v_forecaster": {"enabled": False, "config": {}},
}


# =============================================================================
# DATA LOADER
# =============================================================================

def load_market_data(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    df.columns = df.columns.astype(str).str.strip().str.lower()

    if "time" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"time": "timestamp"}, inplace=True)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp", drop=True)

    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required market columns: {sorted(missing)} | Found: {list(df.columns)}"
        )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_index()

    return df


# =============================================================================
# FALLBACK MODULE
# =============================================================================

def identity_module(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    return df.copy()


# =============================================================================
# MODULE WRAPPER
# =============================================================================

def _call_module(
    module_func: Callable,
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    if config is None:
        config = {}

    try:
        return module_func(df, config=config)
    except TypeError:
        try:
            return module_func(df, config)
        except TypeError:
            return module_func(df)


# =============================================================================
# DYNAMIC MODULE LOADER
# =============================================================================

def resolve_module_callable(module_path: str, preferred_names: list[str]) -> tuple[Callable, str]:
    try:
        mod = import_module(module_path)
    except Exception:
        return identity_module, "identity_module"

    for fn_name in preferred_names:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return fn, fn_name

    return identity_module, "identity_module"


def build_module_pipeline() -> list[dict[str, Any]]:
    generic_candidates = [
        "run",
        "build",
        "process",
        "apply",
        "compute",
        "calculate",
        "build_indicator",
        "build_engine",
    ]

    module_specs = [
        ("a_indicators", "core.a_indicators", ["build_indicators", "run_indicators", *generic_candidates]),
        ("b_trend", "core.b_trend", ["build_trend", "run_trend", "apply_trend", *generic_candidates]),
        ("c_ema", "core.c_ema", ["build_ema", "run_ema_engine", "run_ema", "apply_ema", *generic_candidates]),
        ("d_momentum", "core.d_momentum", ["build_momentum", "run_momentum", "apply_momentum", *generic_candidates]),
        ("e_ai_rsi", "core.e_ai_rsi", ["build_ai_rsi", "run_ai_rsi_engine", "run_ai_rsi", "apply_ai_rsi", *generic_candidates]),
        ("f_market_structure", "core.f_market_structure", ["build_market_structure", "run_market_structure", *generic_candidates]),
        ("g_session_daily", "core.g_session_daily", ["build_session_daily", "run_session_daily", *generic_candidates]),
        ("h_fib", "core.h_fib", ["build_fib", "run_fib", "apply_fib", *generic_candidates]),
        ("i_liquidity", "core.i_liquidity", ["build_liquidity", "run_liquidity", "apply_liquidity", *generic_candidates]),
        ("j_fvg", "core.j_fvg", ["build_fvg", "run_fvg", *generic_candidates]),
        ("k_orderflow", "core.k_orderflow", ["build_orderflow", "run_orderflow", *generic_candidates]),
        ("l_confluence_cloud", "core.l_confluence_cloud", ["build_confluence_cloud", "run_confluence_cloud", *generic_candidates]),
        ("m_volume", "core.m_volume", ["build_volume", "run_volume", *generic_candidates]),
        ("n_volatility", "core.n_volatility", ["build_volatility", "run_volatility", *generic_candidates]),
        ("o_ob_os", "core.o_ob_os", ["build_ob_os", "run_ob_os", *generic_candidates]),
        ("p_exhaustion", "core.p_exhaustion", ["build_exhaustion", "run_exhaustion", *generic_candidates]),
        ("q_mfi", "core.q_mfi", ["build_mfi", "run_mfi", *generic_candidates]),
        ("r_confluence", "core.r_confluence", ["build_confluence", "run_confluence", *generic_candidates]),
        ("s_macd_reversal", "core.s_macd_reversal", ["build_macd_reversal", "run_macd_reversal", *generic_candidates]),
        ("t_pullback_retest", "core.t_pullback_retest", ["build_pullback_retest", "run_pullback_retest", *generic_candidates]),
        ("u_regime", "core.u_regime", ["build_regime", "run_regime", "apply_regime", *generic_candidates]),
        ("v_forecaster", "core.v_forecaster", ["build_forecaster", "run_forecaster", "apply_forecaster", *generic_candidates]),
    ]

    pipeline: list[dict[str, Any]] = []

    for short_name, module_path, fn_candidates in module_specs:
        module_func, resolved_name = resolve_module_callable(module_path, fn_candidates)
        settings = MODULE_SETTINGS.get(short_name, {"enabled": False, "config": {}})

        pipeline.append(
            {
                "name": short_name,
                "module_path": module_path,
                "func": module_func,
                "resolved_name": resolved_name,
                "enabled": bool(settings.get("enabled", False)),
                "config": settings.get("config", {}) or {},
            }
        )

    return pipeline


# =============================================================================
# MODULE EXECUTION
# =============================================================================

def run_module_pipeline(df_raw: pd.DataFrame, pipeline: list[dict[str, Any]]) -> pd.DataFrame:
    df = df_raw.copy()

    for spec in pipeline:
        module_name = spec["name"]
        module_func = spec["func"]
        resolved_name = spec["resolved_name"]
        enabled = spec["enabled"]
        config = spec["config"]

        if not enabled:
            print(f"[SKIP] {module_name}: disabled")
            continue

        if module_func is identity_module:
            print(f"[FALLBACK] {module_name}: no real callable resolved")
            continue

        try:
            before_cols = set(df.columns)
            df = _call_module(module_func, df, config=config)

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{module_name} did not return a DataFrame")

            after_cols = set(df.columns)
            new_cols = sorted(after_cols - before_cols)

            if new_cols:
                print(f"[OK] {module_name}: {resolved_name} added {len(new_cols)} column(s) -> {new_cols[:8]}")
            else:
                print(f"[INFO] {module_name}: {resolved_name} added no new columns")

        except Exception as exc:
            raise RuntimeError(f"Module failed: {module_name}") from exc

    return df


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_activation_report(pipeline: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("SMARTCHART MODULE ACTIVATION")
    print("=" * 80)

    for spec in pipeline:
        status = "ENABLED" if spec["enabled"] else "DISABLED"
        fallback = "YES" if spec["func"] is identity_module else "NO"
        print(
            f"{spec['name']:>20} | "
            f"{status:<8} | "
            f"callable={spec['resolved_name']:<24} | "
            f"fallback={fallback}"
        )


def verify_truth_fields(symbol: str, df_truth: pd.DataFrame) -> None:
    target_groups = {
        "direction": ["direction", "dir", "bias"],
        "core_alignment": ["align", "alignment", "core"],
        "location": ["location", "fib", "liq", "liquidity", "structure"],
        "regime": ["regime"],
        "forecast_permission": ["forecast", "permission", "permit", "allow"],
    }

    truth_cols = [str(c) for c in df_truth.columns]

    print(f"\n--- {symbol} | TRUTH FIELD CHECK ---")
    for group_name, patterns in target_groups.items():
        matched = [c for c in truth_cols if any(p.lower() in c.lower() for p in patterns)]
        if matched:
            print(f"[OK] {group_name}: {matched[:8]}")
        else:
            print(f"[WARN] {group_name}: no matching columns found")


def verify_scanner_output(df_scanner: pd.DataFrame) -> None:
    print("\n--- SCANNER CHECK ---")
    print(f"shape={df_scanner.shape}")
    print(f"columns={list(df_scanner.columns[:20])}")


def verify_broker_routes(df_routes: pd.DataFrame) -> None:
    print("\n--- BROKER ROUTES CHECK ---")
    print(f"shape={df_routes.shape}")
    print(f"columns={list(df_routes.columns[:20])}")


# =============================================================================
# SINGLE-INSTRUMENT PIPELINE
# =============================================================================

def run_single_symbol_pipeline(
    symbol: str,
    df_raw: pd.DataFrame,
    pipeline: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    df_modules = run_module_pipeline(df_raw, pipeline)
    df_truth = build_truth(df_modules)
    df_playbook = build_playbook(df_truth)

    return {
        "raw": df_raw,
        "modules": df_modules,
        "truth": df_truth,
        "playbook": df_playbook,
    }


# =============================================================================
# MULTI-INSTRUMENT PIPELINE
# =============================================================================

def run_multi_symbol_pipeline(
    data_map: dict[str, pd.DataFrame],
    pipeline: list[dict[str, Any]],
) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol, df_raw in data_map.items():
        result[symbol] = run_single_symbol_pipeline(symbol, df_raw, pipeline)

    return result


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def export_dataframe(df: pd.DataFrame, filename: str) -> Path:
    path = EXPORT_DIR / filename
    df.to_csv(path, index=True)
    return path


# =============================================================================
# BROKER CONFIG
# =============================================================================

def get_default_broker_config() -> dict:
    return {
        "broker_demo_1": {
            "enabled": True,
            "mode": "demo",
            "broker_name": "broker_demo_1",
            "symbols": {
                "XAUUSD": "XAUUSD",
                "EURUSD": "EURUSD",
                "GBPUSD": "GBPUSD",
            },
        },
        "broker_live_1": {
            "enabled": False,
            "mode": "live",
            "broker_name": "broker_live_1",
            "symbols": {
                "XAUUSD": "GOLD",
                "EURUSD": "EURUSD",
                "GBPUSD": "GBPUSD",
            },
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    data_map: dict[str, pd.DataFrame] = {}

    candidate_files = {
        "XAUUSD": DATA_DIR / "xauusd.csv",
        "EURUSD": DATA_DIR / "eurusd.csv",
        "GBPUSD": DATA_DIR / "gbpusd.csv",
    }

    for symbol, filepath in candidate_files.items():
        if filepath.exists():
            data_map[symbol] = load_market_data(filepath)

    if not data_map:
        raise FileNotFoundError(
            "No market data files found in /data. "
            "Add at least one file like data/xauusd.csv"
        )

    pipeline = build_module_pipeline()
    print_activation_report(pipeline)

    pipeline_map = run_multi_symbol_pipeline(data_map, pipeline)

    scan_map = {
        symbol: {
            "truth": payload["truth"],
            "playbook": payload["playbook"],
        }
        for symbol, payload in pipeline_map.items()
    }

    df_scanner = build_multi_scanner(scan_map)

    broker_config = get_default_broker_config()
    df_routes = build_multi_broker_router(scan_map, broker_config)

    print("\n" + "=" * 80)
    print("SMARTCHART BACKEND RUN COMPLETE")
    print("=" * 80)

    for symbol, payload in pipeline_map.items():
        print(f"\n--- {symbol} | MODULE COLS SAMPLE ---")
        print(list(payload["modules"].columns[:60]))

        print(f"\n--- {symbol} | TRUTH (TRANSPOSED DEBUG) ---")
        print(payload["truth"].tail(3).T)

        print(f"\n--- {symbol} | EMA DEBUG ---")
        ema_cols = [
            "ema_dir",
            "ema_quality",
            "rt_ema_1420",
            "rt_ema_3350",
            "rt_ema_100200",
            "ema_rt_any",
            "ema_rt_family",
            "ema_rt_dir",
            "ema_slope_fast",
            "ema_slope_band",
            "ema_slope_slow",
            "ema14_slope",
            "ema20_slope",
            "ema33_slope",
            "ema50_slope",
            "ema100_slope",
            "ema200_slope",
            "price_to_ema20_pct",
            "price_to_ema20_pts",
            "ema20_to_ema200_pct",
            "ema20_to_ema200_pts",
        ]
        ema_cols = [c for c in ema_cols if c in payload["truth"].columns]

        if ema_cols:
            print(payload["truth"][ema_cols].tail(3).T)
        else:
            print("EMA fields not found in truth")

        print(f"\n--- {symbol} | PLAYBOOK ---")
        print(payload["playbook"].tail(3).T)

        verify_truth_fields(symbol, payload["truth"])

    print("\n--- MULTI SCANNER ---")
    print(df_scanner)
    verify_scanner_output(df_scanner)

    print("\n--- BROKER ROUTES ---")
    print(df_routes)
    verify_broker_routes(df_routes)

    for symbol, payload in pipeline_map.items():
        export_dataframe(payload["truth"], f"{symbol.lower()}_truth.csv")
        export_dataframe(payload["playbook"], f"{symbol.lower()}_playbook.csv")
        export_dataframe(payload["modules"], f"{symbol.lower()}_modules.csv")

    export_dataframe(df_scanner, "scanner_table.csv")
    export_dataframe(df_routes, "broker_routes.csv")

    print("\nExports saved to:", EXPORT_DIR)


if __name__ == "__main__":
    main()