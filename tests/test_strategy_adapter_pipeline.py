"""Tests covering the strategy adapter pipeline pieces."""
from __future__ import annotations

import types
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd
import pytest

from src.models import general_trainer


@pytest.fixture
def stub_strategy_module(monkeypatch: pytest.MonkeyPatch) -> str:
    """Create a fake strategy module with a deterministic run_strategy."""
    module_name = "tests.fake_strategy"
    mod = types.ModuleType(module_name)

    def run_strategy(symbol: str, start: datetime, end: datetime, equity: float, params: Dict[str, Any]) -> Dict[str, Any]:
        # simple deterministic equity curve: start at equity, +1% per day
        idx = pd.date_range(start, end, freq="D")
        if len(idx) < 2:
            idx = pd.date_range(start, periods=2, freq="D")
        curve = pd.Series(
            [equity * (1 + 0.01 * i) for i in range(len(idx))],
            index=idx,
        )
        daily_returns = curve.pct_change().fillna(0.0)
        trades = [
            {
                "return_pct": 0.02,
                "holding_days": 2,
                "mfe": 0.03,
                "mae": -0.01,
                "side": "long",
                "entry_price": 100.0,
                "exit_price": 102.0,
                "day_low": 99.0,
                "day_high": 103.0,
                "day_low_exit": 100.0,
                "day_high_exit": 104.0,
            }
        ]
        return {
            "equity": curve,
            "daily_returns": daily_returns,
            "trades": trades,
            "meta": {"symbol": symbol, "param_keys": sorted(params.keys())},
        }

    mod.run_strategy = run_strategy  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, mod)
    return module_name


@pytest.fixture
def sample_period() -> tuple[datetime, datetime]:
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=4)
    return start, end


def test_train_general_model_aggregates_metrics(stub_strategy_module: str, sample_period: tuple[datetime, datetime]) -> None:
    start, end = sample_period
    tickers = ["AAPL", "MSFT"]
    params = {"breakout_n": 14}

    report = general_trainer.train_general_model(
        stub_strategy_module,
        tickers,
        start,
        end,
        starting_equity=10_000.0,
        base_params=params,
    )

    assert report["strategy"] == stub_strategy_module
    assert report["params"] == params
    assert set(report["period"].keys()) == {"start", "end"}
    assert len(report["results"]) == len(tickers)

    aggregate = report["aggregate"]["metrics"]
    # Each symbol contributes one trade, so aggregate trades should equal len(tickers)
    assert aggregate["trades"] == len(tickers)
    # Weighted averages should stay within reasonable bounds
    assert 0.0 <= aggregate["avg_holding_days"] <= 2.0
    assert 0.0 <= aggregate["win_rate"] <= 1.0
    assert aggregate["expectancy"] != 0.0
    # Core performance metrics should be present
    for key in ["total_return", "cagr", "max_drawdown", "calmar", "sharpe"]:
        assert key in aggregate


def test_import_callable_validates_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "tests.empty_strategy"
    empty_module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, empty_module)

    with pytest.raises(ImportError):
        general_trainer.import_callable(module_name)


def test_weighted_average_handles_empty_inputs() -> None:
    assert general_trainer._weighted_average([], []) == 0.0
    assert general_trainer._weighted_average([1.0, 2.0], [0.0, 0.0]) == 0.0


# Need sys imported for monkeypatching sys.modules
import sys  # placed at end to avoid circular import issues in fixtures
