"""Protect the Phase 3 processor-free plotting payload boundary.

Plotters may retain pandas-based scientific tables, but they must receive
detached base-model snapshots rather than live stage processor instances.
"""

from dataclasses import fields

import pandas as pd
import pytest

from pimqc.core import MetaboDataset
from pimqc.plotting.dataset import DatasetPlotter
from pimqc.plotting.payloads import (
    AssessmentPlotPayload,
    CorrectionPlotPayload,
    DatasetPlotPayload,
    FilteringPlotPayload,
    ImputationPlotPayload,
    NormalizationPlotPayload,
    snapshot_dataset,
    snapshot_plot_value,
)
from pimqc.processing.filtering import FeatureFilter


def _minimal_dataset() -> MetaboDataset:
    """Return a small explicit dataset with shared plotting metadata."""
    intensity = pd.DataFrame(
        [[10.0, 11.0], [20.0, 21.0]],
        index=pd.Index(["F1", "F2"], name="Metabolite"),
        columns=pd.Index(["S1", "S2"], name="Sample Name"),
    )
    sample_metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "Sample"],
            "Batch": ["B1", "B1"],
            "Inject Order": [1, 2],
        },
        index=pd.Index(["S1", "S2"], name="Sample Name"),
    )
    return MetaboDataset.from_tables(intensity, sample_metadata)


def _field_names(data_class: type[object]) -> tuple[str, ...]:
    """Return dataclass fields in their public constructor order."""
    return tuple(item.name for item in fields(data_class))


def test_plot_payload_field_surfaces_are_explicit() -> None:
    """Freeze the minimal plot input owned by every stage."""
    assert _field_names(DatasetPlotPayload) == ("data",)
    assert _field_names(AssessmentPlotPayload) == (
        "data",
        "is_multi_batch",
    )
    assert _field_names(FilteringPlotPayload) == (
        "data",
        "audit_tables",
        "sample_mv_tolerance",
        "active_base_tolerance",
        "mnar_group_mv_tolerance",
        "mnar_qc_mv_tolerance",
        "mnar_intensity_threshold",
        "mnar_intensity_percentile",
        "blank_qc_ratio_tolerance",
        "qc_rsd_tolerance",
        "stage_status",
        "sample_filter_status",
        "missing_values_detected",
        "biological_groups_available",
        "skip_reason",
    )
    assert _field_names(CorrectionPlotPayload) == (
        "source_data",
        "selected_stages",
        "candidate_results",
        "selected_prediction",
        "internal_standard_ids",
        "boundary_type",
    )
    assert _field_names(ImputationPlotPayload) == (
        "raw_data",
        "imputed_data",
        "global_seed",
        "sample_structure",
    )
    assert _field_names(NormalizationPlotPayload) == (
        "raw_data",
        "normalized_data",
        "selection",
        "score_component_weights",
        "sample_scale_log_ratio_tolerance",
        "sample_scale_relative_delta_tolerance",
        "global_seed",
        "sample_structure",
        "qc_diagnostics",
    )


def test_snapshot_owns_plain_tables_and_rejects_processors() -> None:
    """Detach a dataset without treating a live engine as plotting input."""
    dataset = _minimal_dataset()
    processor = FeatureFilter(dataset)
    snapshot = snapshot_dataset(dataset)

    assert type(snapshot) is MetaboDataset
    assert type(snapshot.intensity) is pd.DataFrame
    pd.testing.assert_frame_equal(snapshot.intensity, dataset.intensity)
    dataset.intensity.iloc[0, 0] = -1

    assert snapshot.intensity.iloc[0, 0] == 10.0
    with pytest.raises(TypeError, match="MetaboDataset"):
        snapshot_dataset(processor)  # type: ignore[arg-type]


def test_nested_candidate_snapshots_detach_datasets_and_frames() -> None:
    """Detach datasets and ordinary tables nested in candidate stores."""
    dataset = _minimal_dataset()
    frame = dataset.annotated_frame()

    snapshot = snapshot_plot_value(
        {"candidate": {"stage_datasets": {"Original": dataset}, "table": frame}}
    )

    nested = snapshot["candidate"]["stage_datasets"]["Original"]
    assert type(nested) is MetaboDataset
    assert nested is not dataset
    assert snapshot["candidate"]["table"] is not frame


def test_plot_payload_identity_is_versioned() -> None:
    """Expose a stable adapter-facing identity for plot input schemas."""
    payload = DatasetPlotPayload(data=_minimal_dataset())

    assert payload.contract_identity() == {
        "payload_type": "dataset_plot",
        "schema_version": "1.0",
    }


def test_plotter_rejects_raw_data_objects() -> None:
    """Prevent reintroduction of direct processor or dataframe consumption."""
    with pytest.raises(TypeError, match="DatasetPlotPayload"):
        DatasetPlotter(_minimal_dataset())
