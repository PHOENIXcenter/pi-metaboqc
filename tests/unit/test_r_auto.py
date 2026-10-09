"""Provider substitutions precede AUTO evaluation, never hide R failures."""

import threading

import numpy as np
import pytest

from pimqc import DataNormalizer, MissingValueImputer
from pimqc.serialization import read_audit_payload, write_audit_payload
from tests.unit.test_audit_regressions import _dataset


def _fake_r(method, frame, *, seed=None, **parameters):
    assert threading.current_thread() is threading.main_thread()
    output = (
        np.log2(frame)
        if method == "VSN"
        else frame.apply(lambda row: row.fillna(row.mean()), axis=1)
    )
    return output, {
        "implementation": "r",
        "method": method,
        "function": f"R::{method}",
        "seed": seed,
    }


def test_r_auto_imputation_keeps_methods_and_records_each_provider(
    monkeypatch,
    tmp_path,
):
    """Retain the imputation portfolio and audit the backend of each
    candidate.
    """
    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", _fake_r)
    engine = MissingValueImputer(_dataset(missing=True), implementation="r")
    result = engine.run_imputation()
    selection = result.audit.metrics["selection"]
    rows = {row["method"]: row for row in selection["candidate_results"]}
    assert set(rows) == {"KNN", "MinProb", "QRILC", "Median", "LLS", "BPCA"}
    for method, row in rows.items():
        expected = "r" if method in {"BPCA", "QRILC"} else "python"
        assert row["implementation"] == expected
        assert row["status"] == "ok"
        if expected == "r":
            assert row["implementation_provenance"]
    assert selection["requested_implementation"] == "r"
    assert (
        selection["implementation"]
        == rows[selection["selected_method"]]["implementation"]
    )
    assert engine.config["implementation"] == "r"
    assert engine.config["mar_method"] == "Auto"
    assert result.data.intensity.notna().all().all()
    path = write_audit_payload(result.audit, tmp_path / "r_auto")
    restored = read_audit_payload(path)
    assert (
        restored.metrics["selection"]["candidate_results"]
        == (selection["candidate_results"])
    )


def test_r_auto_imputation_failure_does_not_use_python_same_method(monkeypatch):
    """Keep R candidate failures visible rather than substituting native
    kernels.
    """
    def failed(*args, **kwargs):
        raise RuntimeError("R unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("No native replacement of an R candidate")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", failed)
    monkeypatch.setattr(MissingValueImputer, "impute_by_bpca", forbidden)
    monkeypatch.setattr(MissingValueImputer, "impute_by_qrilc", forbidden)
    result = MissingValueImputer(
        _dataset(missing=True), implementation="r"
    ).run_imputation()
    rows = {
        row["method"]: row
        for row in result.audit.metrics["selection"]["candidate_results"]
    }
    for method in ("BPCA", "QRILC"):
        assert rows[method]["implementation"] == "r"
        assert rows[method]["status"] == "failed"
        assert "R unavailable" in rows[method]["reason"]
    assert result.audit.selected_method not in {"BPCA", "QRILC"}


@pytest.mark.parametrize("fail_r", [False, True])
def test_r_auto_normalization_uses_existing_scoring_and_keeps_provenance(
    monkeypatch,
    tmp_path,
    fail_r,
):
    """Keep shared normalization scoring and record actual backend
    provenance.
    """
    def original(*args, **kwargs):
        if fail_r:
            raise RuntimeError("R VSN unavailable")
        return _fake_r(*args, **kwargs)

    def metrics(self, frame):
        value = 1.0 if frame.attrs["norm_method"] == "VSN" else 0.0
        return {name: value for name in self._AUTO_SCORE_COMPONENT_WEIGHTS}

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", original)
    monkeypatch.setattr(
        DataNormalizer, "_AUTO_CANDIDATES", ("ROBUST_LOG_ONLY", "VSN")
    )
    monkeypatch.setattr(
        DataNormalizer, "calc_auto_norm_candidate_results", metrics
    )
    engine = DataNormalizer(_dataset(), implementation="r")
    result = engine.run_normalization()
    selection = result.audit.selection
    rows = {row["method"]: row for row in result.audit.candidate_results}
    assert rows["VSN"]["implementation"] == "r"
    assert rows["ROBUST_LOG_ONLY"]["implementation"] == "python"
    assert selection["requested_implementation"] == "r"
    assert engine.config["implementation"] == "r"
    if fail_r:
        assert rows["VSN"]["status"] == "failed"
        assert "R VSN unavailable" in rows["VSN"]["error"]
        assert selection["implementation"] == "python"
        assert selection["selected_method"] == "ROBUST_LOG_ONLY"
        assert selection["implementation_provenance"] == []
    else:
        assert rows["VSN"]["status"] == "ok"
        assert selection["implementation"] == "r"
        assert selection["selected_method"] == "VSN"
        assert selection["implementation_provenance"][0]["function"] == "R::VSN"
        assert result.audit.output_suffix == "VSN_R"
        assert "vsn_converged" not in result.audit.metrics.get(
            "vsn_parameters", {}
        )
    restored = read_audit_payload(
        write_audit_payload(
            result.audit,
            tmp_path / "norm_auto",
        )
    )
    assert restored.selection["implementation"] == selection["implementation"]
    assert restored.selection["requested_implementation"] == "r"
