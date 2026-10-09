"""Verify explicit R imputation dispatch without requiring an R runtime."""

import json
from dataclasses import replace

import numpy as np
import pytest

from pimqc.processing.imputation import MissingValueImputer


def _dataset_with_missing_values():
    """Reuse the public test dataset builder used by audit regressions."""
    from tests.unit.test_audit_regressions import _dataset

    return _dataset(missing=True)


def test_r_implementation_rejects_unsupported_fixed_method() -> None:
    """Fixed R requests are strict; AUTO has an explicit provider map."""
    engine = MissingValueImputer(
        _dataset_with_missing_values(), mar_method="KNN", implementation="r"
    )
    with pytest.raises(ValueError, match="requires mar_method"):
        engine.run_imputation()


def test_r_implementation_records_backend_provenance(
    monkeypatch, tmp_path
) -> None:
    """A selected R method remains auditable and preserves matrix labels."""
    data = _dataset_with_missing_values()
    features = data.feature_metadata.copy()
    features.loc["F0", "missingness_type"] = "MNAR"
    intensity = data.intensity.copy()
    intensity.loc["F1", intensity.columns[-1]] = np.nan
    data = type(data).from_tables(
        intensity,
        data.sample_metadata,
        features,
        schema=data.schema,
        context=data.context,
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run_r_method(method, frame, *, seed=None, **parameters):
        calls.append((method, tuple(frame.columns.names)))
        values = frame.copy()
        values = values.apply(lambda row: row.fillna(row.mean()), axis=1)
        return values, {
            "implementation": "r",
            "method": method,
            "package": "test-package",
            "package_version": "0.0",
            "function": "test::function",
            "parameters": parameters,
            "seed": seed,
            "warnings": [],
        }

    monkeypatch.setattr(
        "pimqc.processing.r_backend.run_r_method", fake_run_r_method
    )
    engine = MissingValueImputer(
        data,
        mar_method="BPCA",
        implementation="r",
        sim_mask_ratio=0.05,
    )
    result = engine.run_imputation()

    assert not result.data.intensity.isna().any().any()
    selection = result.audit.metrics["selection"]
    assert selection["implementation"] == "r"
    assert selection["implementation_provenance"]
    json.dumps(selection["implementation_provenance"], allow_nan=False)
    mar_records = [
        item for item in selection["implementation_provenance"]
        if item["phase"] != "final_mnar"
    ]
    assert all(len(item["sample_roles"]) == 1 for item in mar_records)
    methods = {
        item["method"] for item in selection["implementation_provenance"]
    }
    assert methods == {
        "BPCA",
        "QRILC",
    }
    assert calls
    expected_names = ("Batch", "Sample Type", "Inject Order", "Sample Name")
    assert all(name == expected_names for _, name in calls)
    from pimqc.serialization import read_audit_payload, write_audit_payload

    path = write_audit_payload(result.audit, tmp_path / "audit")

    def no_r(*args, **kwargs):
        raise AssertionError("Loading a saved audit must not initialize R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    loaded = read_audit_payload(path)
    assert (
        loaded.metrics["selection"]["implementation_provenance"]
        == selection["implementation_provenance"]
    )


@pytest.mark.parametrize("mnar_method", ["Row-wise", "Column-wise", "Global"])
def test_r_rejects_python_mnar_heuristics(mnar_method) -> None:
    """A stage-wide R request cannot secretly mix in native LOD rules."""
    engine = MissingValueImputer(
        _dataset_with_missing_values(),
        mar_method="BPCA",
        mnar_method=mnar_method,
        implementation="r",
    )
    with pytest.raises(ValueError, match="mnar_method='QRILC'"):
        engine.run_imputation()


def test_r_failure_never_falls_back_to_native_bpca(monkeypatch) -> None:
    """Explicit R execution errors remain visible to the caller."""
    from pimqc.processing.r_backend import RBackendError

    def fail(*args, **kwargs):
        raise RBackendError("upstream failure")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fail)
    engine = MissingValueImputer(
        _dataset_with_missing_values(),
        mar_method="BPCA",
        implementation="r",
    )
    with pytest.raises(RBackendError, match="upstream failure"):
        engine.run_imputation()


def test_implementation_runtime_overrides_config(monkeypatch) -> None:
    """Runtime settings take precedence over both config and constructor."""
    def fail(*args, **kwargs):
        raise AssertionError("Python execution must not load R")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fail)
    engine = MissingValueImputer(
        _dataset_with_missing_values(),
        pipeline_params={
            "MissingValueImputer": {
                "implementation": "r",
                "mar_method": "BPCA",
            }
        },
        implementation="python",
    )
    assert engine.config["implementation"] == "python"
    engine.config["implementation"] = "r"
    result = engine.run_imputation(
        implementation="python", mar_method="Median"
    )
    assert result.audit.metrics["selection"]["implementation"] == "python"
    assert result.audit.metrics["selection"]["implementation_provenance"] == []


@pytest.mark.parametrize("flag", ["is_logged", "is_scaled"])
def test_r_rejects_transformed_input_before_r_call(flag, monkeypatch) -> None:
    """Never double-transform data already logged or scaled upstream."""
    data = _dataset_with_missing_values()
    data.context = replace(data.context, **{flag: True})

    def no_r(*args, **kwargs):
        raise AssertionError("Invalid input must fail before initializing R")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", no_r)
    with pytest.raises(ValueError, match="unlogged, unscaled"):
        MissingValueImputer(
            data, mar_method="BPCA", implementation="r"
        ).run_imputation()


@pytest.mark.parametrize("value", [-1.0, 0.0, 2000.0, np.inf])
def test_r_linear_reconstruction_uses_stage_policy(value, monkeypatch) -> None:
    """Finite signed draws are repaired; nonfinite inverses remain failures."""
    def fake(method, frame, *, seed=None, **parameters):
        return frame.fillna(value), {"method": method}

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fake)
    engine = MissingValueImputer(
        _dataset_with_missing_values(),
        mar_method="BPCA",
        implementation="r",
    )
    monkeypatch.setattr(
        engine,
        "_evaluate_imputation_candidate",
        lambda *args, **kwargs: ({}, np.array([]), np.array([])),
    )
    if value > 1000:
        with pytest.raises(ValueError, match="nonfinite"):
            engine.run_imputation()
    else:
        result = engine.run_imputation()
        expected = engine.dataset.intensity.loc["F0", "S1":].min() / 2
        assert result.data.intensity.loc["F0", "S0"] == pytest.approx(expected)
        domain = result.audit.metrics["numerical_domain"]
        assert domain[-1]["repaired_count"] == 1


@pytest.mark.parametrize("value", [-2.0, np.inf])
def test_r_invalid_observation_fails_before_transformation(
    value, monkeypatch
) -> None:
    """Invalid raw observations must not be converted to fillable NaNs."""
    data = _dataset_with_missing_values()
    intensity = data.intensity.copy()
    intensity.iloc[1, 1] = value

    def no_r(*args, **kwargs):
        raise AssertionError("Invalid observations must not initialize R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    with pytest.raises(ValueError, match="nonnegative observed"):
        MissingValueImputer(
            data.with_intensity(intensity),
            mar_method="BPCA",
            implementation="r",
        ).run_imputation()


def test_complete_r_stage_skips_without_initializing_r(monkeypatch) -> None:
    """Valid R selections with no missing values need no installed R."""
    from tests.unit.test_audit_regressions import _dataset

    def fail(*args, **kwargs):
        raise AssertionError("Skipped imputation must not load R")

    monkeypatch.setattr("pimqc.processing.r_backend.run_r_method", fail)
    result = MissingValueImputer(
        _dataset(), mar_method="BPCA", implementation="r"
    ).run_imputation()
    assert result.audit.skipped
    assert result.audit.metrics["selection"]["implementation"] == "r"
    assert result.audit.metrics["selection"]["implementation_provenance"] == []


def test_native_default_matches_explicit_python(monkeypatch) -> None:
    """The added selector leaves the default Python scientific path intact."""
    import pandas as pd

    def no_r(*args, **kwargs):
        raise AssertionError("Native imputation must not initialize R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    data = _dataset_with_missing_values()
    default = MissingValueImputer(data, mar_method="Median").run_imputation()
    explicit = MissingValueImputer(
        data, mar_method="Median", implementation="python"
    ).run_imputation()
    pd.testing.assert_frame_equal(
        default.data.intensity, explicit.data.intensity
    )
    assert default.audit.metrics["selection"]["implementation"] == "python"


@pytest.mark.parametrize("value", [-1.0, 0.0, 2000.0, np.inf])
def test_native_qrilc_inverse_uses_stage_policy(value, monkeypatch) -> None:
    """Fixed MNAR production fills use the same declared output policy."""
    data = _dataset_with_missing_values()
    features = data.feature_metadata.copy()
    features["missingness_type"] = "MNAR"
    data = type(data).from_tables(
        data.intensity, data.sample_metadata, features,
        schema=data.schema, context=data.context,
    )
    monkeypatch.setattr(
        MissingValueImputer, "impute_by_qrilc",
        staticmethod(lambda df_log, **kw: df_log.fillna(value)),
    )
    engine = MissingValueImputer(data, mar_method="Median")
    if value > 1000:
        with pytest.raises(ValueError, match="Fixed S-route.*nonfinite"):
            engine.run_imputation()
    else:
        result = engine.run_imputation()
        expected = data.intensity.loc["F0", "S1":].min() / 2
        assert result.data.intensity.loc["F0", "S0"] == pytest.approx(expected)
        assert (
            result.audit.metrics["numerical_domain"][0]["repaired_count"] == 1
        )
