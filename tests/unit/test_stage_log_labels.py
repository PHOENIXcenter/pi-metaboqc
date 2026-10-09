"""Stage backend labels cover inner logs and execution timing."""

from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

import pytest
from loguru import logger

from pimqc.runtime.logging import _format_record
from pimqc.runtime.timing import log_execution_time


@pytest.mark.parametrize(
    "backend,label", [("python", "[PYTHON]"), ("r", "[R ORIGINAL]")]
)
def test_stage_logs_and_timing_share_backend(backend, label):
    """Render plain labels throughout a stage without leaking context."""
    output = StringIO()
    sink = logger.add(output, format=_format_record, colorize=False)

    class Stage:
        config = {"implementation": backend}

        @log_execution_time
        def run(self):
            logger.info("inside stage")
            return 42

    start = datetime(2026, 1, 1)
    try:
        with patch("pimqc.runtime.timing.datetime") as clock:
            clock.now.side_effect = [start, start + timedelta(seconds=2)]
            assert Stage().run() == 42
        logger.info("outside stage")
    finally:
        logger.remove(sink)

    lines = output.getvalue().splitlines()
    assert label in lines[0] and "inside stage" in lines[0]
    assert label in lines[1] and "Execution time" in lines[1]
    assert "[PYTHON]" not in lines[2]
    assert "[R ORIGINAL]" not in lines[2]
    assert "\x1b" not in output.getvalue()


def test_backend_context_is_restored_after_failure():
    """Stage errors preserve their exception and restore logging context."""
    records = []
    sink = logger.add(lambda message: records.append(message.record))

    class Stage:
        config = {"implementation": "r"}

        @log_execution_time
        def run(self):
            logger.warning("stage failure")
            raise ValueError("original error")

    try:
        with pytest.raises(ValueError, match="original error"):
            Stage().run()
        logger.info("after failure")
    finally:
        logger.remove(sink)

    assert records[0]["extra"]["implementation"] == "r"
    assert "implementation" not in records[1]["extra"]
    assert "[R ORIGINAL]" not in _format_record(records[1])
    assert "[PYTHON]" not in _format_record(records[1])
