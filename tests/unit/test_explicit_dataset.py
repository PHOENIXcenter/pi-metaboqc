"""Protect the composition-based dataset as the only public data boundary."""

from dataclasses import fields, replace
import importlib

import pandas as pd
import pytest

import pimqc
from pimqc.core import (
    DatasetSchema,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)


def _ordinary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return deliberately misordered metadata for alignment tests."""
    intensity = pd.DataFrame(
        [[20.0, 10.0], [40.0, 30.0]],
        index=pd.Index(["F2", "F1"], name="Metabolite"),
        columns=pd.Index(["S2", "S1"], name="Sample Name"),
    )
    sample_metadata = pd.DataFrame(
        {
            "Sample Type": ["Sample", "QC"],
            "Batch": ["B1", "B1"],
            "Inject Order": [1, 2],
            "Bio Group": ["Case", "Reference"],
        },
        index=pd.Index(["S1", "S2"], name="Sample Name"),
    )
    feature_metadata = pd.DataFrame(
        {"mz": [101.1, 202.2]},
        index=pd.Index(["F1", "F2"], name="Metabolite"),
    )
    return intensity, sample_metadata, feature_metadata


def _dataset() -> MetaboDataset:
    intensity, samples, features = _ordinary_tables()
    return MetaboDataset.from_tables(
        intensity,
        samples,
        features,
        context=ProcessingContext(
            acquisition_mode="POS",
            internal_standards=("F1",),
            outlier_reference_features=("F2",),
            global_seed=7,
            extra_attrs={"nested": {"thresholds": [1, 2]}},
        ),
    )


def test_dataset_field_surface_is_explicit() -> None:
    """Freeze the public fields of the sole domain boundary."""
    assert tuple(item.name for item in fields(MetaboDataset)) == (
        "intensity",
        "sample_metadata",
        "feature_metadata",
        "schema",
        "context",
    )
    assert tuple(item.name for item in fields(DatasetSchema)) == (
        "feature_id",
        "sample_id",
        "sample_type",
        "batch",
        "injection_order",
        "biological_group",
        "roles",
        "sample_metadata_order",
    )


def test_from_tables_aligns_axes_and_owns_defensive_copies() -> None:
    """Align metadata to matrix order without retaining caller aliases."""
    intensity, samples, features = _ordinary_tables()
    context = ProcessingContext(extra_attrs={"nested": {"value": [1]}})

    dataset = MetaboDataset.from_tables(
        intensity,
        samples,
        features,
        context=context,
    )

    assert dataset.sample_ids.tolist() == ["S2", "S1"]
    assert dataset.feature_ids.tolist() == ["F2", "F1"]
    assert dataset.sample_metadata.index.tolist() == ["S2", "S1"]
    assert dataset.feature_metadata.index.tolist() == ["F2", "F1"]
    assert dataset.sample_metadata.loc["S2", "Sample Type"] == "QC"
    assert dataset.feature_metadata.loc["F2", "mz"] == 202.2
    assert dataset.schema.sample_metadata_order == (
        "Sample Name",
        "Sample Type",
        "Batch",
        "Inject Order",
        "Bio Group",
    )

    intensity.loc["F2", "S2"] = -1
    samples.loc["S2", "Sample Type"] = "Changed"
    features.loc["F2", "mz"] = -1
    context.extra_attrs["nested"]["value"].append(2)

    assert dataset.intensity.loc["F2", "S2"] == 20.0
    assert dataset.sample_metadata.loc["S2", "Sample Type"] == "QC"
    assert dataset.feature_metadata.loc["F2", "mz"] == 202.2
    assert dataset.context.extra_attrs == {"nested": {"value": [1]}}

    exported_intensity, exported_samples, exported_features = (
        dataset.to_tables()
    )
    exported_intensity.loc["F2", "S2"] = -2
    exported_samples.loc["S2", "Sample Type"] = "Export changed"
    exported_features.loc["F2", "mz"] = -2

    assert dataset.intensity.loc["F2", "S2"] == 20.0
    assert dataset.sample_metadata.loc["S2", "Sample Type"] == "QC"
    assert dataset.feature_metadata.loc["F2", "mz"] == 202.2


def test_from_tables_accepts_explicit_identifier_columns() -> None:
    """Normalize conventional metadata ID columns into canonical indices."""
    intensity, samples, features = _ordinary_tables()
    dataset = MetaboDataset.from_tables(
        intensity,
        samples.reset_index(),
        features.reset_index(),
        sample_id_column="Sample Name",
        feature_id_column="Metabolite",
    )

    assert dataset.sample_metadata.index.name == "Sample Name"
    assert dataset.feature_metadata.index.name == "Metabolite"
    assert "Sample Name" not in dataset.sample_metadata.columns
    assert "Metabolite" not in dataset.feature_metadata.columns


@pytest.mark.parametrize("axis_name", ["sample", "feature"])
def test_duplicate_identifiers_are_rejected(axis_name: str) -> None:
    """Reject ambiguous sample or feature identity before processing."""
    intensity, samples, features = _ordinary_tables()
    if axis_name == "sample":
        intensity.columns = ["S1", "S1"]
    else:
        intensity.index = ["F1", "F1"]

    with pytest.raises(ValueError, match=f"Duplicate {axis_name} identifiers"):
        MetaboDataset.from_tables(intensity, samples, features)


def test_metadata_identifier_mismatch_is_rejected() -> None:
    """Reject missing and extraneous metadata entities as one clear error."""
    intensity, samples, features = _ordinary_tables()
    samples = samples.rename(index={"S1": "S9"})

    with pytest.raises(ValueError, match="identifiers do not match"):
        MetaboDataset.from_tables(intensity, samples, features)


def test_metadata_order_cannot_silently_omit_columns() -> None:
    """Require every sample metadata field in the reversible level order."""
    intensity, samples, features = _ordinary_tables()
    schema = DatasetSchema(
        sample_metadata_order=(
            "Sample Name",
            "Sample Type",
            "Batch",
            "Inject Order",
        )
    )

    with pytest.raises(ValueError, match="omits metadata fields"):
        MetaboDataset.from_tables(intensity, samples, features, schema=schema)


def test_dataset_copy_and_annotated_view_are_detached() -> None:
    """Keep pandas views local and prevent hidden cross-stage state sharing."""
    source = _dataset()
    copied = source.copy()
    annotated = source.annotated_frame()

    assert type(copied.intensity) is pd.DataFrame
    assert isinstance(annotated.columns, pd.MultiIndex)
    assert annotated.columns.names == list(source.schema.sample_metadata_order)
    copied.intensity.iloc[0, 0] = -1
    annotated.iloc[0, 0] = -2
    copied.context.extra_attrs["nested"]["thresholds"].append(3)

    assert source.intensity.iloc[0, 0] == 20.0
    assert source.context.extra_attrs["nested"] == {"thresholds": [1, 2]}


def test_with_intensity_preserves_metadata_and_updates_context_explicitly() -> (
    None
):
    """Create a stage output without carrying dataframe-subclass state."""
    source = _dataset()
    subset = source.intensity.loc[["F1"], ["S1"]] * 2
    feature_metadata = source.feature_metadata.assign(
        missingness_type=pd.Series({"F1": "MAR", "F2": "MNAR"})
    )

    result = source.with_intensity(
        subset,
        context=replace(source.context, pipeline_stage="Imputation"),
        feature_metadata=feature_metadata,
    )

    assert type(result.intensity) is pd.DataFrame
    assert result.sample_ids.tolist() == ["S1"]
    assert result.feature_ids.tolist() == ["F1"]
    assert result.feature_metadata.loc["F1", "missingness_type"] == "MAR"
    assert result.context.pipeline_stage == "Imputation"


def test_obsolete_dataframe_subclass_api_is_absent() -> None:
    """Make the Phase 7 removal an intentional breaking API guarantee."""
    assert pimqc.MetaboDataset is MetaboDataset
    assert pimqc.DatasetSchema is DatasetSchema
    assert pimqc.ProcessingContext is ProcessingContext
    assert "MetaboDataset" in pimqc.__all__
    assert not hasattr(pimqc, "build_metabo_dataset")
    assert callable(pimqc.build_dataset)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pimqc.core.model")


def test_role_labels_round_trip_mapping() -> None:
    """Keep configured role names explicit and framework independent."""
    roles = SampleRoleLabels.from_mapping(
        {
            "Actual sample": "Study",
            "Blank sample": "Solvent",
            "QC sample": "Pool",
            "Calibration sample": "Cal",
        }
    )

    assert roles.to_mapping() == {
        "Calibration sample": "Cal",
        "Actual sample": "Study",
        "Blank sample": "Solvent",
        "QC sample": "Pool",
    }
