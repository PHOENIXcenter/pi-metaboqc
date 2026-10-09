"""Operational routes drive processing while historical data stay readable."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pimqc.config.resolution import resolve_stage_config
from pimqc.config.schema import MissingValueImputerConfig, PipelineConfig
from pimqc.core.routes import (
    R_ROUTE,
    S_ROUTE,
    normalize_route,
    routes_from_metadata,
)
from pimqc.processing.filtering import (
    FeatureMissingValueFilter,
    FeatureQualityFilter,
)
from pimqc.processing.imputation import MissingValueImputer
from pimqc.serialization import (
    read_audit_payload,
    read_metabo_dataset,
    write_audit_payload,
    write_metabo_dataset,
)
from tests.unit.test_audit_regressions import _dataset
from tests.unit.test_independent_filter_stages import _quality_dataset


@pytest.mark.parametrize(
    "value, expected",
    [
        ("MAR", R_ROUTE),
        (" mar ", R_ROUTE),
        ("r-route", R_ROUTE),
        ("MNAR", S_ROUTE),
        (" s-route ", S_ROUTE),
        ("MNAR (Group & QC)", S_ROUTE),
        ("S-route (QC)", S_ROUTE),
    ],
)
def test_route_aliases(value, expected):
    assert normalize_route(value) == expected


def test_unroutable_values_and_conflicting_columns_are_rejected():
    for value in (None, pd.NA, "MCAR", "typo"):
        with pytest.raises(ValueError, match="imputation_route"):
            normalize_route(value)
    metadata = pd.DataFrame(
        {
            "imputation_route": [R_ROUTE],
            "missingness_type": ["MNAR"],
        },
        index=["feature"],
    )
    with pytest.raises(ValueError, match="Conflicting"):
        routes_from_metadata(metadata)
    assert normalize_route("INVALID", strict=False) == "INVALID"


def test_config_aliases_preserve_serialized_names_and_priority():
    params = {
        "MissingValueImputer": {
            "r_route_method": "median",
            "s_route_method": "global",
        }
    }
    settings = PipelineConfig.model_validate(params).model_dump()
    assert settings["MissingValueImputer"]["mar_method"] == "Median"
    assert settings["MissingValueImputer"]["mnar_method"] == "Global"
    assert "r_route_method" not in settings["MissingValueImputer"]
    engine = MissingValueImputer(
        _dataset(),
        pipeline_params=params,
        r_route_method="KNN",
    )
    assert engine.config["mar_method"] == "KNN"
    engine.run_imputation(r_route_method="Median", s_route_method="Row-wise")
    assert engine.config["mar_method"] == "Median"
    assert engine.config["mnar_method"] == "Row-wise"
    settings = resolve_stage_config(
        params,
        "MissingValueImputer",
        {"mar_method": "Auto"},
        {"r_route_method": "KNN"},
    )
    assert settings["mar_method"] == "KNN"


def test_conflicting_method_aliases_fail_before_execution(tmp_path):
    with pytest.raises(ValueError, match="Conflicting"):
        MissingValueImputerConfig(r_route_method="Median", mar_method="KNN")
    with pytest.raises(ValueError, match="Conflicting"):
        MissingValueImputer(
            _dataset(), r_route_method="Median", mar_method="KNN"
        )
    engine = MissingValueImputer(_dataset())
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Conflicting"):
        engine.run_imputation(
            destination,
            r_route_method="Median",
            mar_method="KNN",
        )
    assert not destination.exists()


def _mixed_routes(canonical):
    data = _dataset(missing=True)
    intensity = data.intensity.copy()
    intensity.loc["F1", "S1"] = np.nan
    metadata = data.feature_metadata.copy()
    metadata.loc["F0", "missingness_type"] = "MNAR"
    if canonical:
        metadata["imputation_route"] = metadata.pop("missingness_type").map(
            {"MAR": R_ROUTE, "MNAR": S_ROUTE}
        )
    return replace(data, intensity=intensity, feature_metadata=metadata)


def test_canonical_routes_equal_legacy_outputs_and_drive_real_routing(tmp_path):
    historical = MissingValueImputer(
        _mixed_routes(False),
        mar_method="Median",
        mnar_method="Global",
    ).transform_imputation()
    current = MissingValueImputer(
        _mixed_routes(True),
        r_route_method="Median",
        s_route_method="Global",
    ).transform_imputation()
    pd.testing.assert_frame_equal(
        historical.data.intensity, current.data.intensity
    )
    assert current.audit.metrics["feature_distribution"]["s_route_count"] == 1
    assert current.audit.metrics["selection"]["route"] == R_ROUTE
    assert (
        current.audit.r_route_feature_count
        == historical.audit.mar_feature_count
    )
    all_reconstruction = _mixed_routes(True)
    all_reconstruction.feature_metadata["imputation_route"] = R_ROUTE
    changed = MissingValueImputer(
        all_reconstruction,
        r_route_method="Median",
        s_route_method="Global",
    ).transform_imputation()
    assert (
        current.data.intensity.loc["F0", "S0"]
        != changed.data.intensity.loc["F0", "S0"]
    )
    for name, dataset in (
        ("legacy", _mixed_routes(False)),
        ("current", current.data),
    ):
        restored = read_metabo_dataset(
            write_metabo_dataset(dataset, tmp_path / name)
        )
        pd.testing.assert_frame_equal(
            restored.feature_metadata, dataset.feature_metadata
        )
        assert (
            routes_from_metadata(restored.feature_metadata).loc["F0"] == S_ROUTE
        )
    restored_audit = read_audit_payload(
        write_audit_payload(current.audit, tmp_path / "audit")
    )
    assert (
        restored_audit.r_route_feature_count
        == current.audit.r_route_feature_count
    )


def test_feature_filter_outputs_canonical_and_legacy_route_columns():
    result = FeatureMissingValueFilter(
        _dataset(missing=True)
    ).filter_features_by_missingness()
    metadata = result.data.feature_metadata
    assert "imputation_route" in metadata
    pd.testing.assert_series_equal(
        routes_from_metadata(metadata),
        metadata["imputation_route"],
    )
    assert (
        result.audit.feature_tracking["Stage1_Status"]
        .isin(
            [
                R_ROUTE,
                "S-route (Group)",
                "S-route (QC)",
                "S-route (Group & QC)",
                "INVALID",
            ]
        )
        .all()
    )
    assert result.audit.tables["idx_r_route"].equals(
        result.audit.tables["idx_mar"]
    )


@pytest.mark.parametrize("canonical", [False, True])
def test_quality_filter_uses_route_metadata_for_unchanged_exemption(canonical):
    data = _quality_dataset()
    column = "imputation_route" if canonical else "missingness_type"
    labels = (
        [R_ROUTE, S_ROUTE, R_ROUTE] if canonical else ["MAR", "MNAR", "MAR"]
    )
    metadata = pd.DataFrame({column: labels}, index=data.feature_ids)
    result = FeatureQualityFilter(
        replace(data, feature_metadata=metadata),
        qc_rsd_tol=0.3,
    ).filter_features_by_quality()
    assert "f_unstable" in result.data.feature_ids
    assert (
        result.audit.feature_tracking.loc["f_unstable", "RSD_Check"]
        == "Exempted (S-route)"
    )
