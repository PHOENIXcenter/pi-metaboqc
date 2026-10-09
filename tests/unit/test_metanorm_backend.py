"""Unit checks for explicit Metanorm-rLOESS routing and provenance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.metanorm import MetanormRLOESSCorrector


def _frame() -> pd.DataFrame:
    """Return an indexed, strictly positive feature-by-sample matrix."""
    return pd.DataFrame(
        [np.linspace(10.0, 20.0, 8), np.linspace(20.0, 10.0, 8)],
        index=["F1", "F2"],
        columns=[f"S{i}" for i in range(8)],
    )


def test_metanorm_rejects_nonpositive_values_before_r(monkeypatch) -> None:
    """The adapter must not invent a pseudocount or silently transform data."""
    called = False

    def fail(*args, **kwargs):
        """Reject any unexpected runtime call."""
        nonlocal called
        called = True
        raise AssertionError("R should not be initialized")

    monkeypatch.setattr(
        "pimqc.processing.correction.metanorm.run_r_function", fail
    )
    frame = _frame()
    frame.iloc[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive raw intensities"):
        MetanormRLOESSCorrector(random_state=1).fit_transform(
            frame,
            np.array(["B1"] * 8),
            np.array([True] * 4 + [False] * 4),
            np.arange(8, dtype=float),
        )
    assert not called


def test_metanorm_records_full_fit_and_no_oof(monkeypatch) -> None:
    """R worker output is carried as a full-fit stage with no fake OOF data."""
    seen = {}

    def fake(data, **kwargs):
        """Return unchanged log intensities and record the call contract."""
        seen.update(kwargs)
        return data, {
            "implementation": "r",
            "package": "metanorm",
            "package_version": "0.10.2",
            "function": "metanorm::metanormWorker",
            "parameters": kwargs["parameters"],
            "warnings": [],
        }

    monkeypatch.setattr(
        "pimqc.processing.correction.metanorm.run_r_function", fake
    )
    result = MetanormRLOESSCorrector(random_state=1).fit_transform(
        _frame(),
        np.array(["B1"] * 8),
        np.array([True, False, True, False, False, True, False, True]),
        np.arange(8, dtype=float),
    )
    stage, oof = result["Metanorm-rLOESS corrected"]
    assert stage.shape == (2, 8)
    assert oof is None
    assert seen["function"] == "metanorm::metanormWorker"
    assert seen["parameters"]["QConly"] is True
    assert seen["parameters"]["cv"] == "GCV"


@pytest.mark.parametrize("method", ["QC-RLSC", "QC-RFSC"])
def test_r_correction_requires_distinct_explicit_method(method) -> None:
    """An R request cannot silently execute an unrelated Python method."""
    from tests.unit.test_audit_regressions import _dataset

    processor = SignalCorrector(
        _dataset(), base_est=method, implementation="r", rlsc_robust=False
    )
    with pytest.raises(ValueError, match="supports AUTO or explicit"):
        processor.run_signal_correction()


def test_metanorm_requires_r_implementation() -> None:
    """Naming the R method with the native implementation is an error."""
    from tests.unit.test_audit_regressions import _dataset

    processor = SignalCorrector(_dataset(), base_est="Metanorm-rLOESS")
    with pytest.raises(ValueError, match="requires implementation='r'"):
        processor.run_signal_correction()


def test_metanorm_rejects_scaled_input() -> None:
    """Reject transformed input before initializing the original R fit."""
    from tests.unit.test_audit_regressions import _dataset

    data = _dataset()
    scaled = replace(data, context=replace(data.context, is_scaled=True))
    processor = SignalCorrector(
        scaled, base_est="Metanorm-rLOESS", implementation="r"
    )
    with pytest.raises(ValueError, match="unscaled positive"):
        processor.run_signal_correction()
