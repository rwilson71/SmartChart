from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    import MetaTrader5 as mt5
except Exception as exc:
    raise RuntimeError(
        "MetaTrader5 is not installed. Run: pip install MetaTrader5"
    ) from exc


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class LiveConfig:
    symbol: str = "XAUUSD"
    timeframe_label: str = "M1"
    bars: int = 500
    auto_refresh_seconds: int = 5

    mt5_path: Optional[str] = None
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_timeout_ms: int = 10000
    mt5_portable: bool = False

    modules: List[str] = field(default_factory=lambda: [
        "core.b_trend",
        "core.b_ema",
        "core.d_momentum",
        "core.e_ai_rsi",
        "core.f_market_structure",
        "core.g_session_daily",
        "core.h_fib",
        "core.i_liquidity",
        "core.k_orderflow",
        "core.l_confluence_cloud",
        "core.m_volume",
        "core.n_volatility",
        "core.o_ob_os",
    ])


TIMEFRAME_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


# ==============================================================================
# MT5 HELPERS
# ==============================================================================

def initialize_mt5(cfg: LiveConfig) -> Tuple[bool, str]:
    kwargs: Dict[str, Any] = {
        "timeout": cfg.mt5_timeout_ms,
        "portable": cfg.mt5_portable,
    }

    if cfg.mt5_path:
        kwargs["path"] = cfg.mt5_path
    if cfg.mt5_login is not None:
        kwargs["login"] = cfg.mt5_login
    if cfg.mt5_password is not None:
        kwargs["password"] = cfg.mt5_password
    if cfg.mt5_server is not None:
        kwargs["server"] = cfg.mt5_server

    ok = mt5.initialize(**kwargs)
    if not ok:
        return False, f"MT5 initialize failed: {mt5.last_error()}"

    if not mt5.symbol_select(cfg.symbol, True):
        return False, f"MT5 connected but symbol_select failed: {mt5.last_error()}"

    return True, "MT5 connected successfully."


def shutdown_mt5() -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass


def fetch_live_candles(symbol: str, timeframe_label: str, bars: int) -> pd.DataFrame:
    timeframe = TIMEFRAME_MAP.get(timeframe_label)
    if timeframe is None:
        raise ValueError(f"Unsupported timeframe: {timeframe_label}")

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"MT5 returned no data: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError("No candles returned from MT5.")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df = df.rename(columns={
        "tick_volume": "volume",
    })

    for col in ["spread", "real_volume"]:
        if col not in df.columns:
            df[col] = 0

    return df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]].copy()


# ==============================================================================
# MODULE HELPERS
# ==============================================================================

def module_short_name(module_path: str) -> str:
    return module_path.split(".")[-1]


def pick_callable(module_obj: Any, module_path: str) -> Optional[Callable[..., Any]]:
    names = [
        "run",
        "calculate",
        "compute",
        "process",
        "compute_trend_engine",
        "build_trend",
        "run_trend_engine",
        module_short_name(module_path),
    ]
    for name in names:
        fn = getattr(module_obj, name, None)
        if callable(fn):
            return fn
    return None


def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, (pd.Timestamp, datetime)):
            out[str(k)] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def normalize_module_output(result: Any, module_name: str) -> Dict[str, Any]:
    if result is None:
        return {"module": module_name, "status": "empty"}

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return {"module": module_name, "status": "empty_dataframe"}
        row = result.iloc[-1].to_dict()
        row["module"] = module_name
        row["status"] = "ok"
        return clean_dict(row)

    if isinstance(result, dict):
        result = dict(result)
        result["module"] = module_name
        result.setdefault("status", "ok")
        return clean_dict(result)

    return {
        "module": module_name,
        "status": "unsupported_output",
        "output_type": type(result).__name__,
    }


def run_module(module_path: str, df: pd.DataFrame) -> Dict[str, Any]:
    name = module_short_name(module_path)

    try:
        module_obj = importlib.import_module(module_path)
    except Exception as exc:
        return {
            "module": name,
            "status": "import_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    fn = pick_callable(module_obj, module_path)
    if fn is None:
        return {
            "module": name,
            "status": "callable_not_found",
            "error": "No supported callable found.",
        }

    try:
        try:
            result = fn(df.copy(), config=None)
        except TypeError:
            result = fn(df.copy())
        return normalize_module_output(result, name)
    except Exception as exc:
        return {
            "module": name,
            "status": "execution_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_all_modules(df: pd.DataFrame, module_paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for module_path in module_paths:
        name = module_short_name(module_path)
        results[name] = run_module(module_path, df)
    return results


# ==============================================================================
# SNAPSHOT HELPERS
# ==============================================================================

def make_last_bar_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    return {
        "time": str(last["time"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"]),
        "spread": float(last["spread"]),
        "real_volume": float(last["real_volume"]),
        "close_delta": float(last["close"] - prev["close"]),
        "bar_range": float(last["high"] - last["low"]),
    }


def flatten_results(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for _, payload in results.items():
        rows.append(payload)
    return pd.DataFrame(rows)


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def render_header() -> None:
    st.set_page_config(page_title="SmartChart Live", layout="wide")
    st.title("SmartChart Live Dashboard")
    st.caption("MT5 live feed • SmartChart modules • last-bar display")


def render_sidebar(default_cfg: LiveConfig) -> LiveConfig:
    st.sidebar.header("Settings")

    symbol = st.sidebar.text_input("Symbol", value=default_cfg.symbol)
    timeframe_label = st.sidebar.selectbox(
        "Timeframe",
        list(TIMEFRAME_MAP.keys()),
        index=list(TIMEFRAME_MAP.keys()).index(default_cfg.timeframe_label),
    )
    bars = st.sidebar.slider("Bars", 100, 5000, default_cfg.bars, 100)
    refresh = st.sidebar.slider("Auto Refresh (seconds)", 1, 60, default_cfg.auto_refresh_seconds, 1)

    return LiveConfig(
        symbol=symbol.strip(),
        timeframe_label=timeframe_label,
        bars=int(bars),
        auto_refresh_seconds=int(refresh),
        mt5_path=default_cfg.mt5_path,
        mt5_login=default_cfg.mt5_login,
        mt5_password=default_cfg.mt5_password,
        mt5_server=default_cfg.mt5_server,
        mt5_timeout_ms=default_cfg.mt5_timeout_ms,
        mt5_portable=default_cfg.mt5_portable,
        modules=default_cfg.modules,
    )


def render_last_bar(snapshot: Dict[str, Any]) -> None:
    st.subheader("Last Bar Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close", f"{snapshot['close']:.3f}", delta=f"{snapshot['close_delta']:.3f}")
    c2.metric("High", f"{snapshot['high']:.3f}")
    c3.metric("Low", f"{snapshot['low']:.3f}")
    c4.metric("Range", f"{snapshot['bar_range']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Open", f"{snapshot['open']:.3f}")
    c6.metric("Volume", f"{snapshot['volume']:.0f}")
    c7.metric("Spread", f"{snapshot['spread']:.0f}")
    c8.metric("Real Volume", f"{snapshot['real_volume']:.0f}")

    st.write("Bar time (UTC):", snapshot["time"])


def main() -> None:
    render_header()

    cfg = render_sidebar(LiveConfig())

    if st_autorefresh is not None:
        st_autorefresh(interval=cfg.auto_refresh_seconds * 1000, key="smartchart_refresh")
    else:
        st.info("Optional: install streamlit-autorefresh for timed refresh.")

    ok, message = initialize_mt5(cfg)

    if ok:
        st.success(message)
    else:
        st.error(message)
        st.stop()

    try:
        df = fetch_live_candles(cfg.symbol, cfg.timeframe_label, cfg.bars)
        snapshot = make_last_bar_snapshot(df)
        results = run_all_modules(df, cfg.modules)
        st.write("DEBUG MODULES:", results)

        render_last_bar(snapshot)

        left, right = st.columns([1.2, 1.0])

        with left:
            st.subheader("Recent Candles")
            st.dataframe(df.tail(12), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Live Info")
            st.write(f"Symbol: **{cfg.symbol}**")
            st.write(f"Timeframe: **{cfg.timeframe_label}**")
            st.write(f"Bars loaded: **{len(df)}**")
            st.write(f"Updated UTC: **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}**")

        st.subheader("Module Execution Snapshot")
        st.dataframe(flatten_results(results), use_container_width=True, hide_index=True)

        # ✅ TREND DEBUG (FIXED POSITION)
        st.subheader("Trend Debug")

        trend_data = results.get("b_trend", {})

        st.write({
            "trend_state": trend_data.get("sc_trend_state"),
            "trend_dir": trend_data.get("sc_trend_dir"),
            "trend_strength": trend_data.get("sc_trend_strength"),
            "regime": trend_data.get("sc_regime_text"),
            "adx": trend_data.get("adx_value"),
            "mtf_alignment": trend_data.get("sc_mtf_alignment"),
            "cluster_strength": trend_data.get("sc_cluster_strength"),
        })

        # ✅ KEEP THIS BELOW
        with st.expander("Raw Module Payloads"):
            st.json(results)
            
    except Exception as exc:
        st.exception(exc)
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()