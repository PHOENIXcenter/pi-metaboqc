"""Unit checks for the optional R backend contract."""

import runpy
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pimqc.processing import r_backend


def test_reference_collection_skips_before_r_setup_without_rpy2(
    monkeypatch,
) -> None:
    """Missing rpy2 must skip optional tests before any R discovery or
    import.
    """
    monkeypatch.setitem(sys.modules, "rpy2", None)

    def forbidden_r_lookup(*args, **kwargs):
        raise AssertionError(
            "R discovery ran without the optional rpy2 package"
        )

    monkeypatch.setattr("shutil.which", forbidden_r_lookup)
    config = Path(__file__).parents[1] / "reference" / "conftest.py"
    with pytest.raises(pytest.skip.Exception, match="rpy2 is not installed"):
        runpy.run_path(str(config))


def test_module_import_does_not_initialize_r() -> None:
    """Default Python workflows must not import the optional dependency."""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pimqc.processing.r_backend; "
            "assert not any(name.startswith('rpy2') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr


def _matrix() -> pd.DataFrame:
    """Return a small feature-by-sample matrix."""
    return pd.DataFrame(
        [[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]],
        index=["F1", "F2"],
        columns=["S1", "S2", "S3"],
    )


def test_r_method_rejects_unknown_method_without_loading_r() -> None:
    """A typo must not silently select the native Python implementation."""
    with pytest.raises(ValueError, match="No original R implementation"):
        r_backend.run_r_method("quantile", _matrix())


def test_r_method_rejects_unknown_options_without_loading_r() -> None:
    """R options are method-specific and fail before runtime initialization."""
    with pytest.raises(ValueError, match="Unsupported R BPCA options"):
        r_backend.run_r_method("BPCA", _matrix(), made_up_option=1)


def test_validate_input_rejects_infinite_and_empty_rows() -> None:
    """Invalid measurements cannot be delegated to an R package."""
    with pytest.raises(ValueError, match="infinite"):
        r_backend._validate_input(pd.DataFrame([[np.inf, 1.0]]))
    with pytest.raises(ValueError, match="completely missing feature"):
        r_backend._validate_input(pd.DataFrame([[np.nan, np.nan], [2.0, 3.0]]))


def test_validate_output_preserves_observed_imputation_entries() -> None:
    """Adapters must preserve values that were observed before imputation."""
    original = np.array([[1.0, np.nan], [2.0, 3.0]])
    result = np.array([[1.0, 4.0], [2.0, 3.0]])
    r_backend._validate_output(
        original,
        result,
        preserve_observed=True,
        allow_missing=False,
    )
    changed = result.copy()
    changed[0, 0] = 99.0
    with pytest.raises(r_backend.RBackendError, match="observations"):
        r_backend._validate_output(
            original,
            changed,
            preserve_observed=True,
            allow_missing=False,
        )


def test_all_missing_input_requires_explicit_opt_in() -> None:
    """Other adapters retain the default empty-axis preflight."""
    data = pd.DataFrame([[np.nan, np.nan], [np.nan, 3.0]])
    with pytest.raises(ValueError, match="completely missing sample"):
        r_backend._validate_input(data)
    np.testing.assert_array_equal(
        r_backend._validate_input(data, allow_all_missing_input=True),
        data.to_numpy(),
    )
    with pytest.raises(ValueError, match="infinite"):
        r_backend._validate_input(
            pd.DataFrame([[np.inf]]), allow_all_missing_input=True,
        )


def test_r_loader_rejects_worker_thread() -> None:
    """Embedded R is intentionally excluded from joblib worker threads."""
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            r_backend._load_r()
        except BaseException as exc:  # pragma: no cover - thread transport
            error.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
    assert error and isinstance(error[0], r_backend.RBackendError)


def test_console_callbacks_restore_after_nested_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoped suppression preserves custom callbacks, including on errors."""
    seen: list[str] = []
    callbacks = SimpleNamespace(
        consolewrite_print=seen.append,
        consolewrite_warnerror=seen.append,
    )
    original_print = callbacks.consolewrite_print
    original_error = callbacks.consolewrite_warnerror
    monkeypatch.setattr(
        r_backend.importlib, "import_module", lambda name: callbacks
    )
    with r_backend._R_LOCK, pytest.raises(RuntimeError, match="nested error"):
        with r_backend._quiet_r_console():
            outer_print = callbacks.consolewrite_print
            with r_backend._quiet_r_console():
                callbacks.consolewrite_print("hidden print")
                callbacks.consolewrite_warnerror("hidden stderr")
            assert callbacks.consolewrite_print is outer_print
            raise RuntimeError("nested error")
    assert callbacks.consolewrite_print is original_print
    assert callbacks.consolewrite_warnerror is original_error
    callbacks.consolewrite_print("visible print")
    callbacks.consolewrite_warnerror("visible stderr")
    assert seen == ["visible print", "visible stderr"]


def test_warning_summary_keeps_original_conditions_and_failed_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deduplicate console warnings without losing audit or failure details."""
    logs: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        r_backend.logger, "warning", lambda *args: logs.append(args)
    )
    values = {
        "value": "result",
        "warnings": ["first warning", "first warning", "second warning"],
        "error": [],
    }
    captured = SimpleNamespace(rx2=values.__getitem__)
    result, warnings = r_backend._read_r_conditions(captured, "unit-test")
    assert result == "result"
    assert warnings == values["warnings"]
    assert logs[0][1:] == (
        "unit-test", 3, "first warning; second warning"
    )
    values["error"] = ["original failure"]
    with pytest.raises(r_backend.RBackendError, match="original failure"):
        r_backend._read_r_conditions(captured, "unit-test")
    assert len(logs) == 2
