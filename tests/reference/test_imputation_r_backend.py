"""Exercise complete original-R imputation stages and portable audit output."""

import json

import numpy as np
import pandas as pd
import pytest

from pimqc.core import MetaboDataset
from pimqc.processing.imputation import MissingValueImputer
from pimqc.serialization import read_audit_payload, write_audit_payload

from .helpers import require_r_package


def _dataset(mixed: bool) -> MetaboDataset:
    """Provide enough features per mechanism and samples per isolated role."""
    rng = np.random.default_rng(18)
    sample_ids = pd.Index([f"S{i}" for i in range(12)], name="Sample Name")
    features = pd.Index([f"F{i}" for i in range(60)], name="Metabolite")
    log_values = rng.normal(8.0, 0.35, (60, 12))
    for col in range(12):
        log_values[col, col] = np.nan
        log_values[30 + col, col] = np.nan
    intensity = pd.DataFrame(
        np.exp2(log_values) - 1.0, index=features, columns=sample_ids
    )
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC"] * 4 + ["Sample"] * 8,
            "Batch": ["B1"] * 12,
            "Inject Order": np.arange(1, 13),
        },
        index=sample_ids,
    )
    feature_metadata = pd.DataFrame(
        {"missingness_type": ["MAR"] * 60}, index=features
    )
    if mixed:
        feature_metadata.iloc[30:, 0] = "MNAR"
    return MetaboDataset.from_tables(intensity, metadata, feature_metadata)


@pytest.mark.parametrize("mixed", [False, True], ids=["bpca", "bpca-qrilc"])
def test_original_r_imputation_stage_roundtrip(
    mixed, tmp_path, monkeypatch
) -> None:
    """Production dispatch reaches original packages and survives R-free IO."""
    require_r_package("pcaMethods")
    if mixed:
        require_r_package("imputeLCMD")
    data = _dataset(mixed)
    result = MissingValueImputer(
        data,
        mar_method="BPCA",
        implementation="r",
        bpca_max_iter=100,
        global_seed=18,
    ).run_imputation()
    assert not result.audit.skipped
    assert result.audit.selected_method == "BPCA"
    assert np.isfinite(result.data.intensity.to_numpy()).all()
    assert (result.data.intensity.to_numpy() >= 0).all()
    observed = data.intensity.notna().to_numpy()
    np.testing.assert_allclose(
        result.data.intensity.to_numpy()[observed],
        data.intensity.to_numpy()[observed],
        rtol=1e-12,
    )
    pd.testing.assert_index_equal(
        result.data.intensity.index, data.intensity.index
    )
    pd.testing.assert_index_equal(
        result.data.intensity.columns, data.intensity.columns
    )
    provenance = result.audit.metrics["selection"]["implementation_provenance"]
    assert {item["package"] for item in provenance} == (
        {"pcaMethods", "imputeLCMD"} if mixed else {"pcaMethods"}
    )
    assert all(item["package_version"] for item in provenance)
    assert all(item["seed"] == 18 for item in provenance)
    assert {item["phase"] for item in provenance}.issuperset(
        {"evaluation_mar", "final_mar"}
    )
    json.dumps(provenance, allow_nan=False)
    path = write_audit_payload(result.audit, tmp_path / "audit")

    def no_r(*args, **kwargs):
        raise AssertionError("Saved results must not need embedded R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    loaded = read_audit_payload(path)
    assert loaded.metrics["selection"]["implementation_provenance"] == (
        provenance
    )
