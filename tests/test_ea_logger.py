"""Tests for TrainingLogger utility used by the evolutionary search."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.training_logger import TrainingLogger


def test_training_logger_appends_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    logger = TrainingLogger(path)

    logger.log("generation_start", {"gen": 0, "pop_size": 4})
    logger.log("individual_evaluated", {"score": 1.23})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    assert records[0]["event"] == "generation_start"
    assert records[0]["payload"] == {"gen": 0, "pop_size": 4}
    assert "ts" in records[0]
    assert records[1]["event"] == "individual_evaluated"


def test_training_logger_records_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "errors.jsonl"
    logger = TrainingLogger(path)

    class BoomError(RuntimeError):
        pass

    with caplog.at_level("ERROR"):
        logger.log_error({"gen": 1}, BoomError("failed"))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["event"] == "error"
    assert record["payload"]["context"] == {"gen": 1}
    assert record["payload"]["error_type"] == "BoomError"
    assert record["payload"]["error_msg"] == "failed"

    assert any("BoomError" in msg for msg in caplog.text.splitlines())
