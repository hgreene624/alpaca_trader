# src/storage.py
from __future__ import annotations

import json
import logging
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from datetime import datetime
logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

# -----------------------------------------------------------------------------
# Core paths & utils
# -----------------------------------------------------------------------------

def get_data_dir() -> Path:
    """
    Base data directory. Defaults to ./data, override with env DATA_DIR.
    """
    base = os.environ.get("DATA_DIR", "data")
    p = Path(base).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> str:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return str(path)

def _clean_filename(name: str) -> str:
    keep = "-_.() "
    return "".join(ch for ch in name if ch.isalnum() or ch in keep).strip()

# -----------------------------------------------------------------------------
# Portfolio storage (unified, back-compat with legacy "storage/portfolios")
# -----------------------------------------------------------------------------

# New canonical dir
def _portfolios_dir() -> Path:
    return _ensure_dir(get_data_dir() / "portfolios")

# Legacy dir support (read/upgrade on load)
_LEGACY_PORT_ROOT = Path("storage/portfolios")

def _legacy_path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", ".")).strip()
    return _LEGACY_PORT_ROOT / f"{safe}.json"

def _legacy_txt_path_for(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", ".")).strip()
    return _LEGACY_PORT_ROOT / f"{safe}.txt"

def _canonical_portfolio_path(name: str) -> Path:
    name = _clean_filename(name)
    return _portfolios_dir() / f"{name}.json"

def list_portfolios() -> list[str]:
    """
    Return portfolio names discovered under storage/portfolios, A→Z.
    Supports legacy .parquet files for discovery only.
    """
    from pathlib import Path
    PORT_DIR = Path("storage") / "portfolios"
    PORT_DIR.mkdir(parents=True, exist_ok=True)

    names = set()
    for p in PORT_DIR.glob("*.json"):
        names.add(p.stem)
    for p in PORT_DIR.glob("*.parquet"):
        names.add(p.stem)  # legacy discovery

    return sorted(names)

def _upgrade_legacy_if_needed(name: str) -> Optional[Dict[str, Any]]:
    """
    If legacy JSON/TXT exists and new JSON does not, read legacy, normalize, and
    write to canonical location. Returns normalized payload or None.
    """
    new_path = _canonical_portfolio_path(name)
    if new_path.exists():
        return _read_json(new_path)

    # Legacy JSON
    jpath = _legacy_path_for(name)
    if jpath.exists():
        try:
            data = _read_json(jpath) or {}
            tickers = [str(t).upper() for t in (data.get("tickers") or [])]
            meta = data.get("meta") or {}
            payload = {"name": name, "tickers": sorted(set(tickers)), "meta": meta}
            _write_json_atomic(new_path, payload)
            return payload
        except Exception:
            pass

    # Legacy TXT (one symbol per line)
    tpath = _legacy_txt_path_for(name)
    if tpath.exists():
        lines = [ln.strip().upper() for ln in tpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        payload = {"name": name, "tickers": sorted(set(lines)), "meta": {"upgraded_from": "txt"}}
        _write_json_atomic(new_path, payload)
        return payload

    return None

def load_portfolio(name: str) -> dict | None:
    """
    Load a portfolio by name from storage/portfolios/<name>.json
    If a legacy .parquet exists, load it read-only (symbols from 'symbol' col or first col).
    Returns:
      {"name": str, "tickers": [str], "meta": dict, "saved_at": iso} or None
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    PORT_DIR = Path("storage") / "portfolios"
    PORT_DIR.mkdir(parents=True, exist_ok=True)

    def _norm_syms(xs):
        seen, out = set(), []
        for x in xs or []:
            s = str(x).strip().upper()
            if s and s not in seen:
                seen.add(s); out.append(s)
        return out

    json_path = PORT_DIR / f"{name}.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "name": data.get("name") or name,
            "tickers": _norm_syms(data.get("tickers", [])),
            "meta": data.get("meta") or {},
            "saved_at": data.get("saved_at") or datetime.now(timezone.utc).isoformat(),
        }

    # Legacy parquet (read-only)
    pq_path = PORT_DIR / f"{name}.parquet"
    if pq_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq_path)
            if "symbol" in df.columns:
                syms = df["symbol"].astype(str).tolist()
            else:
                syms = df.iloc[:, 0].astype(str).tolist()
            return {
                "name": name,
                "tickers": _norm_syms(syms),
                "meta": {"source": "legacy_parquet"},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            return None

    return None

def save_portfolio(name: str, tickers: list[str], meta: dict | None = None) -> dict:
    """
    Create/overwrite storage/portfolios/<name>.json with {name,tickers,meta,saved_at}.
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    PORT_DIR = Path("storage") / "portfolios"
    PORT_DIR.mkdir(parents=True, exist_ok=True)

    seen, syms = set(), []
    for t in tickers or []:
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s); syms.append(s)

    payload = {
        "name": name,
        "tickers": syms,
        "meta": meta or {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = PORT_DIR / f"{name}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return payload

def append_to_portfolio(name: str, tickers: list[str], meta_update: dict | None = None) -> dict:
    """
    Append tickers & merge meta into storage/portfolios/<name>.json (create if missing).
    De-duplicates tickers, uppercases symbols, preserves existing meta keys unless overridden.
    """
    current = load_portfolio(name) or {"name": name, "tickers": [], "meta": {}}
    base_syms = current.get("tickers", [])
    new_syms = tickers or []

    seen, merged = set(), []
    for t in list(base_syms) + list(new_syms):
        s = str(t).strip().upper()
        if s and s not in seen:
            seen.add(s); merged.append(s)

    meta = dict(current.get("meta") or {})
    if meta_update:
        meta.update(meta_update)

    return save_portfolio(name, merged, meta=meta)

# -----------------------------------------------------------------------------
# Portfolio models (per-portfolio saved models)
# -----------------------------------------------------------------------------

def _models_dir(portfolio: str) -> Path:
    portfolio = _clean_filename(portfolio)
    return _ensure_dir(get_data_dir() / "models" / portfolio)

def list_portfolio_models(portfolio: str) -> List[str]:
    d = _models_dir(portfolio)
    items = list(d.glob("*.json"))
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in items]

def save_portfolio_model(portfolio: str, model_name: str, payload: Dict[str, Any]) -> str:
    portfolio = _clean_filename(portfolio)
    model_name = _clean_filename(model_name)
    if not model_name.endswith(".json"):
        model_name += ".json"
    path = _models_dir(portfolio) / model_name

    meta = dict(payload.get("meta", {}))
    meta.setdefault("portfolio", portfolio)
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    payload["meta"] = meta

    return _write_json_atomic(path, payload)

def load_portfolio_model(portfolio: str, model_filename: str) -> Optional[Dict[str, Any]]:
    portfolio = _clean_filename(portfolio)
    path = _models_dir(portfolio) / model_filename
    return _read_json(path)

# -----------------------------------------------------------------------------
# Training logs (Base Model Lab / General Trainer)
# -----------------------------------------------------------------------------

def _logs_dir(portfolio: str) -> Path:
    portfolio = _clean_filename(portfolio)
    return _ensure_dir(get_data_dir() / "logs" / portfolio)

def save_training_log(portfolio: str, payload: Dict[str, Any], filename: Optional[str] = None) -> str:
    d = _logs_dir(portfolio)
    if filename:
        filename = _clean_filename(filename)
        if not filename.endswith(".json"):
            filename += ".json"
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"base_model_training_log_{ts}.json"
    path = d / filename
    return _write_json_atomic(path, payload)

def list_training_logs(portfolio: str) -> List[str]:
    d = _logs_dir(portfolio)
    files = list(d.glob("*.json"))
    files.sort(by=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]

def load_training_log(path_or_portfolio: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if filename is None:
        path = Path(path_or_portfolio)
    else:
        path = _logs_dir(path_or_portfolio) / filename
    return _read_json(path)

# -----------------------------------------------------------------------------
# Optional index helpers
# -----------------------------------------------------------------------------

def _index_dirs() -> List[Path]:
    """Return candidate directories that may contain index membership files."""
    dirs = [
        _ensure_dir(get_data_dir() / "indexes"),
        Path("storage") / "indexes",
        Path("storage") / "index",  # legacy typo guard
        Path("storage") / "universe",
    ]
    out = []
    for d in dirs:
        if d.exists() and d.is_dir():
            out.append(d)
    return out


def _normalize_index_symbol(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.upper().replace(".", "-")
    return s


def _normalize_index_rows(seq: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(seq, (list, tuple, set)):
        if isinstance(seq, dict):
            iterable = list(seq.values())
        else:
            return rows
    else:
        iterable = list(seq)

    seen: Dict[str, Dict[str, Any]] = {}
    for item in iterable:
        row: Dict[str, Any]
        symbol: Optional[str] = None
        if isinstance(item, dict):
            row = dict(item)
            for key in ("symbol", "ticker", "code", "secid", "Symbol", "Ticker"):
                if key in row and symbol is None:
                    symbol = _normalize_index_symbol(row[key])
            if symbol is None and len(row) == 1:
                symbol = _normalize_index_symbol(next(iter(row.values())))
        else:
            symbol = _normalize_index_symbol(item)
            row = {"symbol": symbol} if symbol else {}

        if not symbol:
            continue
        row["symbol"] = symbol
        if symbol not in seen:
            seen[symbol] = row
            rows.append(row)
    return rows


def _coerce_index_payload(payload: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    if isinstance(payload, dict):
        for key in ("members", "rows", "symbols", "tickers", "data"):
            block = payload.get(key)
            if isinstance(block, (list, tuple, set, dict)):
                rows = _normalize_index_rows(block)
                if rows:
                    break
        if not rows and all(isinstance(v, (dict, str)) for v in payload.values()):
            constructed = []
            for sym_key, val in payload.items():
                if isinstance(val, dict):
                    row = dict(val)
                    row.setdefault("symbol", sym_key)
                    constructed.append(row)
                else:
                    constructed.append({"symbol": val})
            rows = _normalize_index_rows(constructed)

        meta = {
            k: v
            for k, v in payload.items()
            if k not in {"members", "rows", "symbols", "tickers", "data"}
        }
    elif isinstance(payload, (list, tuple, set)):
        rows = _normalize_index_rows(payload)

    return rows, meta


def _register_index_entry(
    entries: Dict[str, Dict[str, Any]],
    label: str,
    info: Dict[str, Any],
) -> None:
    base = str(label).strip() or info.get("fallback_label") or "Unnamed"
    candidate = base
    counter = 2
    while candidate in entries:
        candidate = f"{base} ({counter})"
        counter += 1
    info = dict(info)
    info["label"] = candidate
    entries[candidate] = info


def _discover_index_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}

    # Aggregate files (single JSON containing many indexes)
    aggregate_files = [
        get_data_dir() / "indexes.json",
        Path("storage") / "indexes.json",
    ]
    for agg in aggregate_files:
        if not agg.exists():
            continue
        payload = _read_json(agg) or {}
        if not isinstance(payload, dict):
            continue
        raw_indexes: Dict[str, Any] = {}
        if isinstance(payload.get("indexes"), dict):
            raw_indexes = payload["indexes"]
        else:
            raw_indexes = {
                k: v
                for k, v in payload.items()
                if isinstance(v, (list, dict))
            }
        for name, raw in raw_indexes.items():
            if not isinstance(raw, (list, dict)):
                continue
            _register_index_entry(
                catalog,
                str(name),
                {
                    "type": "aggregate",
                    "path": agg,
                    "key": str(name),
                },
            )

    # Individual files per universe
    for d in _index_dirs():
        for path in sorted(d.glob("*.json")):
            payload = _read_json(path) or {}
            label = None
            if isinstance(payload, dict):
                label = (
                    payload.get("name")
                    or payload.get("label")
                    or payload.get("index")
                    or payload.get("title")
                )
            if not label:
                label = path.stem
            entry_type = "universe_cache" if d.name == "universe" else "file"
            _register_index_entry(
                catalog,
                str(label),
                {
                    "type": entry_type,
                    "path": path,
                },
            )

    return catalog


def list_index_cache() -> List[str]:
    catalog = _discover_index_catalog()
    return sorted(catalog.keys(), key=lambda s: s.lower())


def save_index_members(index_key: str, payload: Dict[str, Any]) -> str:
    index_key = _clean_filename(index_key)
    d = _ensure_dir(get_data_dir() / "indexes")
    path = d / f"{index_key}.json"
    return _write_json_atomic(path, payload)


def load_index_members(index_key: str) -> Optional[Dict[str, Any]]:
    catalog = _discover_index_catalog()
    entry = catalog.get(index_key)
    if entry is None:
        # Fallback: try to resolve by sanitized filename stem (legacy callers)
        key_stem = _clean_filename(index_key).replace(".json", "")
        for info in catalog.values():
            path = info.get("path")
            if isinstance(path, Path) and path.stem == key_stem:
                entry = info
                break
    if entry is None:
        return None

    source_type = entry.get("type")
    path = entry.get("path")
    raw_payload: Any

    if source_type == "aggregate":
        payload = _read_json(path) or {}
        raw_indexes = payload.get("indexes") if isinstance(payload, dict) else None
        if isinstance(raw_indexes, dict) and entry.get("key") in raw_indexes:
            raw_payload = raw_indexes[entry["key"]]
        elif isinstance(payload, dict) and entry.get("key") in payload:
            raw_payload = payload[entry["key"]]
        else:
            raw_payload = None
    else:
        raw_payload = _read_json(path)

    rows, meta = _coerce_index_payload(raw_payload)
    if not rows and isinstance(raw_payload, dict):
        # last resort try treating the payload itself as rows
        fallback_rows, fallback_meta = _coerce_index_payload(raw_payload.get("rows"))
        if fallback_rows:
            rows = fallback_rows
            meta.update(fallback_meta)

    symbols = [row["symbol"] for row in rows]
    meta = dict(meta or {})
    if path:
        meta.setdefault("source_path", str(path))
    if source_type:
        meta.setdefault("source_type", source_type)
    if entry.get("key"):
        meta.setdefault("source_key", entry["key"])

    return {
        "name": entry.get("label", index_key),
        "symbols": symbols,
        "members": rows,
        "meta": meta,
    }

# -----------------------------------------------------------------------------
# Legacy simulations (kept as-is)
# -----------------------------------------------------------------------------

def _sims_dir() -> Path:
    return _ensure_dir(get_data_dir() / "simulations")

def list_simulations() -> List[str]:
    d = _sims_dir()
    files = list(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]

def save_simulation(name: str, payload: Dict[str, Any]) -> str:
    name = _clean_filename(name)
    if not name.endswith(".json"):
        name += ".json"
    path = _sims_dir() / name
    meta = dict(payload.get("meta", {}))
    meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
    payload["meta"] = meta
    return _write_json_atomic(path, payload)

def load_simulation(name_or_path: str) -> Optional[Dict[str, Any]]:
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" and p.exists():
        return _read_json(p)
    name = _clean_filename(name_or_path)
    if not name.endswith(".json"):
        name += ".json"
    return _read_json(_sims_dir() / name)

# -----------------------------------------------------------------------------
# OHLCV canonical root under ./storage/data/ohlcv
# -----------------------------------------------------------------------------
from pathlib import Path

def get_ohlcv_root() -> Path:
    """
    Canonical on-disk location for OHLCV parquet shards.
    Layout:
      storage/data/ohlcv/<provider>/<SYMBOL>/<TF>/<START>__<END>.parquet
    """
    root = Path("storage") / "data" / "ohlcv"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_price_history(
    symbol: str,
    *,
    timeframe: str = "1D",
    start: "datetime | pd.Timestamp | None" = None,
    end: "datetime | pd.Timestamp | None" = None,
) -> "pd.DataFrame":
    """
    Lightweight wrapper around src.data.loader.get_ohlcv that normalizes columns.
    Defaults to roughly a 3-year window when dates are omitted.
    """
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    try:
        from src.data.loader import get_ohlcv
    except Exception as exc:  # pragma: no cover - surface loader import errors
        logger.exception("load_price_history failed to import get_ohlcv")
        raise RuntimeError("load_price_history requires src.data.loader.get_ohlcv") from exc

    def _coerce(ts: "datetime | pd.Timestamp | None") -> "datetime | None":
        if ts is None:
            return None
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        elif not isinstance(ts, datetime):
            parsed = pd.to_datetime(ts)
            if isinstance(parsed, pd.DatetimeIndex):
                parsed = parsed[0]
            ts = parsed.to_pydatetime()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    symbol_norm = (symbol or "").strip().upper()
    if not symbol_norm:
        raise ValueError("symbol is required")

    end_dt = _coerce(end)
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    start_dt = _coerce(start)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365 * 3)

    logger.debug(
        "load_price_history request",
        extra={
            "symbol": symbol_norm,
            "timeframe": timeframe,
            "start": start_dt.isoformat() if start_dt else None,
            "end": end_dt.isoformat() if end_dt else None,
        },
    )
    try:
        df = get_ohlcv(symbol_norm, start_dt, end_dt, timeframe=timeframe)
    except Exception:
        logger.exception(
            "load_price_history get_ohlcv raised",
        )
        raise
    if df is None:
        logger.warning("load_price_history returned None", extra={"symbol": symbol_norm})
        return df
    if df.empty:
        logger.info("load_price_history returned empty frame", extra={"symbol": symbol_norm})
        return df

    df = df.copy()
    cols_lower = {str(col).lower(): col for col in df.columns}

    close_col = cols_lower.get("close")
    if close_col and close_col != "close":
        df = df.rename(columns={close_col: "close"})
    elif "close" not in df.columns:
        for alias in ("c", "price"):
            if alias in cols_lower:
                df = df.rename(columns={cols_lower[alias]: "close"})
                break

    adj_col = cols_lower.get("adj_close")
    if adj_col and adj_col != "adj_close":
        df = df.rename(columns={adj_col: "adj_close"})
    elif "adj_close" not in df.columns:
        for alias in ("adjusted_close", "adjclose"):
            if alias in cols_lower:
                df = df.rename(columns={cols_lower[alias]: "adj_close"})
                break

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    logger.debug(
        "load_price_history normalized columns",
        extra={
            "symbol": symbol_norm,
            "columns": list(df.columns),
            "rows": int(len(df)),
        },
    )
    return df

# ----------------------------------------------------------------------
# Strategy parameter I/O (EA / WF)
# ----------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

def _params_dir(scope: str = "ea") -> Path:
    root = Path("storage") / "params" / scope
    root.mkdir(parents=True, exist_ok=True)
    return root

def save_strategy_params(portfolio: str, strategy: str, params: Dict[str, Any], scope: str = "ea") -> str:
    """
    Save params under storage/params/<scope>/<portfolio>__<strategy>.json
    Returns absolute path as a string.
    """
    safe_port = "".join(c for c in (portfolio or "") if c.isalnum() or c in ("-", "_", ".")).strip() or "default"
    safe_strat = (strategy or "").replace("/", ".").replace("\\", ".")
    fname = f"{safe_port}__{safe_strat}.json"
    path = _params_dir(scope) / fname
    payload = {
        "portfolio": portfolio,
        "strategy": strategy,
        "scope": scope,
        "params": params or {},
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json_atomic(path, payload)
    return str(path.resolve())

def load_strategy_params(portfolio: str, strategy: str, scope: str = "ea") -> Optional[Dict[str, Any]]:
    """
    Load params from storage/params/<scope>/<portfolio>__<strategy>.json
    Returns {'portfolio','strategy','scope','params','saved_at'} or None.
    """
    safe_port = "".join(c for c in (portfolio or "") if c.isalnum() or c in ("-", "_", ".")).strip() or "default"
    safe_strat = (strategy or "").replace("/", ".").replace("\\", ".")
    fname = f"{safe_port}__{safe_strat}.json"
    path = _params_dir(scope) / fname
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None
# Logging helper configured near module imports (see top-level logger assignment).
