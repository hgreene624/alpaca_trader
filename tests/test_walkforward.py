"""Tests targeting evolutionary search helpers (fitness & sampling)."""
from __future__ import annotations

import random

import pytest

from src.optimization import evolutionary


def test_random_param_and_mutate_respect_bounds() -> None:
    space = {"x": (1, 3), "y": (0.5, 1.5)}
    random.seed(123)
    params = evolutionary.random_param(space)
    assert 1 <= params["x"] <= 3
    assert 0.5 <= params["y"] <= 1.5

    random.seed(123)
    mutated = evolutionary.mutate(params, space, rate=1.0)
    # With rate=1.0 every key is resampled within bounds
    assert 1 <= mutated["x"] <= 3
    assert 0.5 <= mutated["y"] <= 1.5


def test_holding_penalty_zero_inside_band() -> None:
    assert evolutionary._holding_penalty(5.0, 3.0, 10.0) == 0.0
    assert evolutionary._holding_penalty(2.5, 3.0, 10.0) == pytest.approx(0.5)
    assert evolutionary._holding_penalty(12.0, 3.0, 10.0) == pytest.approx(2.0)


def test_rate_penalty_behaviour() -> None:
    assert evolutionary._rate_penalty(7.0, 5.0, 10.0) == 0.0
    assert evolutionary._rate_penalty(3.0, 5.0, 10.0) == pytest.approx(2.0)
    assert evolutionary._rate_penalty(12.0, 5.0, 10.0) == pytest.approx(2.0)


def make_metrics(**overrides):
    base = {
        "trades": 10,
        "avg_holding_days": 5.0,
        "cagr": 0.20,
        "calmar": 1.5,
        "sharpe": 1.0,
        "max_drawdown": -0.25,
    }
    base.update(overrides)
    return base


def compute_fitness(**metrics_overrides) -> float:
    metrics = make_metrics(**metrics_overrides)
    return evolutionary._clamped_fitness(
        metrics,
        min_trades=5,
        min_avg_holding_days_gate=2.0,
        require_hold_days=False,
        eps_mdd=1e-3,
        eps_sharpe=1e-3,
        alpha_cagr=1.0,
        beta_calmar=1.0,
        gamma_sharpe=0.5,
        min_holding_days=3.0,
        max_holding_days=10.0,
        holding_penalty_weight=0.2,
        trade_rate_min=5.0,
        trade_rate_max=15.0,
        trade_rate_penalty_weight=0.1,
        num_symbols=2,
        years=1.0,
    )


def test_clamped_fitness_applies_gates() -> None:
    assert compute_fitness(trades=0) == 0.0
    assert compute_fitness(avg_holding_days=1.0) == 0.0
    # Non-zero metrics should yield positive score
    score = compute_fitness()
    assert score > 0.0


def test_clamped_fitness_penalises_outside_preferences() -> None:
    inside = compute_fitness(avg_holding_days=5.0)
    low_hold = compute_fitness(avg_holding_days=3.5)
    high_rate = compute_fitness(trades=40)  # high trade rate for num_symbols=2, years=1 -> rate=20
    assert low_hold <= inside
    assert high_rate <= inside
