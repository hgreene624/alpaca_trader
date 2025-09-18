"""Unit tests for src.data.loader."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from src.data import loader


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure data-provider env vars do not leak between tests."""
    for key in ["DATA_PROVIDER", "ALPACA_FEED"]:
        monkeypatch.delenv(key, raising=False)


def make_df(index: list[Any], columns: list[Any], data: list[list[Any]]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.Index(index), columns=columns)


def test_normalize_ohlcv_handles_multiindex_and_renames_columns() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="US/Eastern")
    columns = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close", "Volume"]])
    values = [
        [100, 101, 99, 100.5, 1_000_000],
        [101, 102, 100, 101.5, 1_100_000],
        [102, 103, 101, 102.5, 1_200_000],
    ]
    df = make_df(idx, columns, values)

    normalized = loader._normalize_ohlcv(df)

    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert normalized.index.tz == timezone.utc
    # Ensure we sorted ascending and preserved data
    assert normalized.iloc[0, 0] == 100


def test_widen_daily_end_adds_one_day_for_daily_bars() -> None:
    end = datetime(2024, 1, 10, tzinfo=timezone.utc)
    widened = loader._widen_daily_end(end, "1D")
    assert widened == end + timedelta(days=1)

    intraday = loader._widen_daily_end(end, "1h")
    assert intraday == end


def test_get_ohlcv_prefers_alpaca_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a minimal DataFrame that _normalize_ohlcv will accept.
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    alpaca_df = pd.DataFrame({
        "open": [1, 2],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.25, 2.25],
        "volume": [100, 200],
    }, index=idx)

    called = {"alpaca": 0, "yahoo": 0}

    def fake_alpaca(symbol: str, start: datetime, end: datetime, timeframe: str, feed: str) -> pd.DataFrame:
        called["alpaca"] += 1
        assert feed == "iex"
        return alpaca_df

    def fake_yahoo(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        called["yahoo"] += 1
        raise AssertionError("Yahoo fallback should not be used when Alpaca succeeds")

    monkeypatch.setattr(loader.A, "load_ohlcv", fake_alpaca)
    monkeypatch.setattr(loader.Y, "load_ohlcv", fake_yahoo)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    df = loader.get_ohlcv("AAPL", start, end, timeframe="1d")

    pd.testing.assert_frame_equal(df, alpaca_df)
    assert called == {"alpaca": 1, "yahoo": 0}


def test_get_ohlcv_falls_back_to_yahoo(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    yahoo_df = pd.DataFrame({
        "Open": [10, 11],
        "High": [11, 12],
        "Low": [9, 10],
        "Close": [10.5, 11.5],
        "Volume": [1_000, 1_100],
    }, index=idx)

    def fake_alpaca(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("Alpaca unavailable")

    def fake_yahoo(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        return yahoo_df

    monkeypatch.setattr(loader.A, "load_ohlcv", fake_alpaca)
    monkeypatch.setattr(loader.Y, "load_ohlcv", fake_yahoo)
    monkeypatch.setenv("DATA_PROVIDER", "auto")

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    df = loader.get_ohlcv("MSFT", start, end, timeframe="1d")

    assert set(df.columns) >= {"open", "high", "low", "close"}
    assert df.index.tz == timezone.utc


def test_get_ohlcv_respects_force_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
    yahoo_df = pd.DataFrame({
        "close": [123.0],
        "open": [120.0],
        "high": [125.0],
        "low": [119.0],
        "volume": [500],
    }, index=idx)

    monkeypatch.setattr(loader.A, "load_ohlcv", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(loader.Y, "load_ohlcv", lambda *_a, **_k: yahoo_df)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)

    df = loader.get_ohlcv("SPY", start, end, timeframe="1d", force_provider="yf")
    pd.testing.assert_frame_equal(df, yahoo_df)
