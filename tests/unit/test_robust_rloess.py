"""Verify source-defined rLOESS and leakage-free QC validation contracts."""

import numpy as np
import pandas as pd
import pytest
import subprocess
import sys

from pimqc.processing.correction import RegressionCorrector, SignalCorrector
from pimqc.processing.correction.rloess import RLOESSModel, RobustRLSCCorrector
from tests.unit.test_audit_regressions import _dataset


def _fixture():
    order = np.arange(30.0)
    values = np.exp2(8 + 0.01 * order + 0.1 * np.sin(order / 5))
    values[9] *= 1.8
    frame = pd.DataFrame([values, values * 2], index=["F1", "F2"])
    return frame, np.repeat("B1", len(order)), np.ones(30, bool), order


def test_native_robust_uses_source_contract_without_r(monkeypatch):
    """Apply the current rLOESS settings without initializing the R runtime."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Native rLOESS must not initialize R")

    monkeypatch.setattr(
        "pimqc.processing.correction.rloess.run_r_function", forbidden
    )
    engine = RegressionCorrector("QC-RLSC", robust=True)
    full, oof = next(iter(engine.fit_transform(*_fixture()).values()))
    assert np.isfinite(full).all().all()
    assert oof.iloc[:, [0, -1]].isna().all().all()
    assert np.isfinite(oof.iloc[:, 1:-1]).all().all()
    parameters = engine.diagnostics["parameters"]
    assert parameters["degree"] == 2
    assert parameters["family"] == "symmetric"
    assert parameters["span_selection"] == "gcv"
    assert parameters["iterations"] == 4
    assert engine.provenance["implementation"] == "python"


def test_oof_qc_does_not_enter_span_fit_or_center():
    """Exclude held-out responses from span selection, fitting, and
    centering.
    """
    frame, batch, qc, order = _fixture()
    first = RobustRLSCCorrector(random_state=123)
    _, before = next(
        iter(first.fit_transform(frame, batch, qc, order).values())
    )
    target = 12
    fold = first.diagnostics["fold_assignments"][target]
    changed = frame.copy()
    changed.iloc[:, target] *= 16
    second = RobustRLSCCorrector(random_state=123)
    _, after = next(
        iter(second.fit_transform(changed, batch, qc, order).values())
    )
    np.testing.assert_allclose(
        before.iloc[:, target] / frame.iloc[:, target],
        after.iloc[:, target] / changed.iloc[:, target],
        rtol=1e-12,
    )
    assert (
        first.diagnostics["selected_spans"][f"fold_{fold}"]
        == (second.diagnostics["selected_spans"][f"fold_{fold}"])
    )


def test_blank_values_do_not_change_nonblank_center_or_predictions():
    """Keep Blank values out of nonblank centering and fitted corrections."""
    frame, batch, qc, order = _fixture()
    qc[-1] = False
    blank = ~qc
    first = RobustRLSCCorrector()
    full, oof = next(
        iter(
            first.fit_transform(
                frame, batch, qc, order, blank_mask=blank
            ).values()
        )
    )
    changed = frame.copy()
    changed.iloc[:, -1] = -42.0
    second = RobustRLSCCorrector()
    changed_full, changed_oof = next(
        iter(
            second.fit_transform(
                changed, batch, qc, order, blank_mask=blank
            ).values()
        )
    )
    np.testing.assert_allclose(full.iloc[:, :-1], changed_full.iloc[:, :-1])
    np.testing.assert_allclose(
        oof.iloc[:, :-1], changed_oof.iloc[:, :-1], equal_nan=True
    )
    assert (changed_full.iloc[:, -1] == -42).all()


def test_sparse_qc_is_unavailable_without_native_singular_call(monkeypatch):
    """Reject underdetermined QC designs before entering the native LOESS
    kernel.
    """
    def forbidden(*args, **kwargs):
        raise AssertionError("Underdetermined fit reached native LOESS")

    monkeypatch.setattr("pimqc.processing.correction.rloess.loess", forbidden)
    with pytest.raises(ValueError, match="five QCs"):
        RLOESSModel().fit(np.arange(4.0), np.arange(4.0))
    model = RLOESSModel(span_selection="fixed", span=0.2)
    with pytest.raises(ValueError, match="five span neighbours"):
        model.fit(np.arange(8.0), np.arange(8.0))


def test_legacy_robust_iterations_is_explicitly_rejected():
    """Reject the retired iteration option in favor of the current rLOESS
    setting.
    """
    with pytest.raises(ValueError, match="rloess_iterations"):
        RegressionCorrector(
            "QC-RLSC", robust=True, robust_iterations=3
        ).fit_transform(*_fixture())


@pytest.mark.parametrize("method", ["QC-RLSC", "QC-SVR"])
def test_direct_r_request_never_runs_an_ordinary_python_regressor(method):
    """Reject unsupported direct R requests instead of silently changing
    backend.
    """
    with pytest.raises(ValueError, match="only for robust QC-RLSC"):
        RegressionCorrector(
            method, robust=False, implementation="r"
        ).fit_transform(*_fixture())


def test_robust_unavailable_oof_never_scores_full_fit():
    """Keep full-model fits from replacing unavailable held-out QC evidence."""
    processor = SignalCorrector(_dataset(), base_est="robust QC-RLSC")
    result = processor.run_signal_correction()
    selection = result.audit.metrics["selection"]
    assert selection["selected_label"] == "robust QC-RLSC"
    assert selection["validation"]["evaluation_basis"] == "unavailable"
    assert not selection["validation"]["eligible_for_auto"]


def test_r_auto_routes_robust_but_keeps_ordinary_regressions_python():
    """Route robust rLOESS to R while retaining native-only regression
    methods.
    """
    engine = SignalCorrector(_dataset(), implementation="r")
    assert engine._candidate_implementation("QC-RLSC", {"robust": True}) == "r"
    assert (
        engine._candidate_implementation("QC-RLSC", {"robust": False})
        == "python"
    )
    assert engine._candidate_implementation("QC-SVR") == "python"


def test_sparse_real_qc_regressions_survive_in_separate_process():
    """Both actual failure geometries previously corrupted the native heap."""
    code = """
import numpy as np
import pandas as pd
from pimqc.processing.correction.rloess import RobustRLSCCorrector
cases = [
    ([1, 7, 13, 19, 25],
     [14.9140966474, 15.1951918840, 15.0884844857,
      15.1207932530, 15.2031730265]),
    ([11, 46, 94, 106, 118, 169],
     [8.7531195664, 8.5337729501, 9.2262977471,
      9.5675112376, 9.5867415504, 8.5589307647]),
]
for x, y in cases:
    frame = pd.DataFrame([np.exp2(y)])
    engine = RobustRLSCCorrector(cv_folds=3)
    engine.fit_transform(frame, np.repeat("B", len(x)),
                         np.ones(len(x), bool), np.asarray(x, float))
print("SURVIVED")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SURVIVED" in result.stdout


def test_worker_failure_is_propagated_and_process_is_disposed(monkeypatch):
    """Propagate native worker failures and discard the failed process."""
    disposed = []

    class FailedWorker:
        def submit(self, *args, **kwargs):
            return self

        def result(self):
            raise RuntimeError("native worker exited unexpectedly")

        def shutdown(self, **kwargs):
            disposed.append(kwargs)

    monkeypatch.setattr(
        "pimqc.processing.correction.rloess.ProcessPoolExecutor",
        lambda **kwargs: FailedWorker(),
    )
    with pytest.raises(RuntimeError, match="native worker exited"):
        RobustRLSCCorrector().fit_transform(*_fixture())
    assert disposed == [{"wait": False, "kill_workers": True}]


def test_native_worker_does_not_replace_joblib_regression_pool():
    """AUTO runs native robust between other joblib-based candidates."""
    data = _fixture()
    params = {"n_jobs": 2, "cv_folds": 3, "regression_backend": "loky"}
    before = RegressionCorrector("QC-SVR", **params).fit_transform(*data)
    RobustRLSCCorrector(cv_folds=3).fit_transform(*data)
    after = RegressionCorrector("QC-SVR", **params).fit_transform(*data)
    for stage in before:
        for left, right in zip(before[stage], after[stage]):
            np.testing.assert_array_equal(left, right)
