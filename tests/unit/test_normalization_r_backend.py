"""Check R normalization dispatch, audit persistence, and strict selection."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pimqc import DataNormalizer, MetaboDatasetBuilder
from pimqc.serialization import read_audit_payload, write_audit_payload


def _dataset():
    """Build a small unlogged matrix with QC, sample, and blank columns."""
    names = [f"S{i}" for i in range(10)]
    meta = pd.DataFrame(
        {
            "Sample Name": names,
            "Sample Type": ["QC"] * 3 + ["Sample"] * 6 + ["Blank"],
            "Batch": ["B1"] * 10,
            "Inject Order": range(10),
        }
    )
    values = np.random.default_rng(9).lognormal(5, 0.3, (30, 10))
    return MetaboDatasetBuilder(
        meta,
        pd.DataFrame(values, index=[f"F{i}" for i in range(30)], columns=names),
    ).run_build().data


@pytest.mark.parametrize("method", ["Quantile", "Median"])
def test_r_normalization_rejects_unsupported_methods(method, monkeypatch):
    """A fixed R request cannot silently select a Python method."""

    def fail(*args, **kwargs):
        """Fail if unsupported selection reaches the adapter."""
        raise AssertionError("R must not be invoked")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fail)
    with pytest.raises(ValueError, match="requires norm_method='VSN'"):
        DataNormalizer(
            _dataset(), norm_method=method, implementation="r"
        ).run_normalization()


def test_r_vsn_dispatches_raw_values_and_roundtrips_audit(
    monkeypatch, tmp_path
):
    """Keep VSN output unshifted, with serializable upstream provenance."""
    data = _dataset()
    calls = []

    def fake(method, matrix, *, seed=None, **parameters):
        """Return distinguishable transformed data without a second log."""
        calls.append((method, matrix.copy(), seed))
        return matrix * 0.01 - 2, {
            "implementation": "r",
            "method": "VSN",
            "package": "vsn",
            "package_version": "test-version",
            "function": "vsn::vsn2 / vsn::predict",
            "parameters": {},
            "seed": seed,
            "warnings": [],
        }

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fake)
    engine = DataNormalizer(
        data,
        pipeline_params={
            "DataNormalizer": {"norm_method": "VSN", "implementation": "r"}
        },
        implementation="python",
    )
    assert engine.config["implementation"] == "python"
    result = engine.run_normalization(implementation="R", global_seed=11)
    assert len(calls) == 1
    method, source, seed = calls[0]
    assert method == "VSN" and seed == 11
    expected = data.intensity.iloc[:, :-1]
    np.testing.assert_array_equal(source.to_numpy(), expected.to_numpy())
    np.testing.assert_array_equal(
        result.data.intensity.to_numpy(), expected.to_numpy() * 0.01 - 2
    )
    assert result.data.intensity.columns.equals(expected.columns)
    assert result.data.context.is_logged is True
    assert result.data.context.log_base == "2"
    assert result.audit.output_suffix == "VSN_R"
    assert "vsn_parameters" not in result.audit.metrics
    selection = result.audit.selection
    assert selection["implementation"] == "r"
    assert selection["is_auto"] is False
    assert selection["candidate_results"] == []
    path = tmp_path / "audit"
    write_audit_payload(result.audit, path)
    restored = read_audit_payload(path)
    assert restored.selection == selection


def test_r_vsn_dependency_failure_propagates(monkeypatch):
    """Missing R cannot trigger the native VSN implementation."""
    from pimqc.processing.r_backend import RBackendUnavailable

    def fail(*args, **kwargs):
        """Simulate a missing optional R dependency."""
        raise RBackendUnavailable("R is absent")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fail)
    with pytest.raises(RBackendUnavailable, match="R is absent"):
        DataNormalizer(
            _dataset(), norm_method="VSN", implementation="r"
        ).run_normalization()


def test_r_vsn_rejects_already_logged_input():
    """Do not feed already normalized data into a second VSN log transform."""
    data = _dataset()
    logged = replace(data, context=replace(data.context, is_logged=True))
    with pytest.raises(ValueError, match="already log-transformed"):
        DataNormalizer(
            logged, norm_method="VSN", implementation="r"
        ).run_normalization()


def test_r_vsn_rejects_already_scaled_input():
    """Scaled observations are not raw-intensity VSN input."""
    data = _dataset()
    scaled = replace(data, context=replace(data.context, is_scaled=True))
    with pytest.raises(ValueError, match="already scaled"):
        DataNormalizer(
            scaled, norm_method="VSN", implementation="r"
        ).run_normalization()
