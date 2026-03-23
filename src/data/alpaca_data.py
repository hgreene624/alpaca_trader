# src/data/alpaca_data.py
from __future__ import annotations

import os
from datetime import datetime, timezone
import pandas as pd

from ._tz_utils import to_utc_index


def _iso_utc(dt: datetime) -> datetime:
    """Return timezone-aware UTC datetime (alpaca-py accepts dt; no need for iso strings)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_data_client():
    """codex
    Lazy-import the modern alpaca-py Historical Data client.
    Reads ALPACA_API_KEY/ALPACA_SECRET_KEY if present; otherwise falls back to APCA_*.
    """
    from alpaca.data.historical import StockHistoricalDataClient  # lazy import
    api_key = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID"))
    secret_key = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY"))
    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY (or APCA_*) in environment")
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _timeframe_to_ap(tf: str):
    """Map our string timeframe to alpaca-py TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    tf = (tf or "1Day").lower()
    if tf in ("1d", "1day", "day", "1 day"):
        return TimeFrame.Day
    if tf in ("1min", "1m"):
        return TimeFrame.Minute
    if tf in ("5min", "5m"):
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf in ("15min", "15m"):
        return TimeFrame(15, TimeFrameUnit.Minute)
    if tf in ("1h", "1hour", "60m"):
        return TimeFrame(1, TimeFrameUnit.Hour)
    return TimeFrame.Day


def load_ohlcv(symbol: str, start: datetime, end: datetime, timeframe: str = "1Day", **kwargs) -> pd.DataFrame:
    """
    Fetch OHLCV via alpaca-py Historical Data client.
    - Uses IEX feed (works on paper/basic plans; avoids 'recent SIP' errors).
    - Returns a DataFrame indexed by UTC DatetimeIndex with columns open, high, low, close, volume.
    - Accepts **kwargs (ignored) to be tolerant with callers.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment  # <-- fixed: Adjustment comes from enums

    client = _get_data_client()

    ap_tf = _timeframe_to_ap(timeframe)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=ap_tf,
        start=_iso_utc(start),
        end=_iso_utc(end),
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,  # avoid SIP
        limit=None,
    )

    bars = client.get_stock_bars(req)
    df = bars.df

    if df is None or df.empty:
        return pd.DataFrame()

    # Drop symbol level if present
    if isinstance(df.index, pd.MultiIndex) and "symbol" in (df.index.names or []):
        try:
            df = df.xs(symbol, level="symbol")
        except Exception:
            pass

    # Normalize columns
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]

    # Ensure UTC tz-aware index and sorted
    df.index = to_utc_index(df.index)
    df.sort_index(inplace=True)

    return df[keep]
