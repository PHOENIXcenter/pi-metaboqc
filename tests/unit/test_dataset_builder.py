"""Dataset actor contracts, compatibility, and artifact lifecycle."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pimqc import MetaboDatasetBuilder
from pimqc.core import MetaboDataset
from pimqc.dataset import build_dataset
from pimqc.dataset.runner import DatasetBuildStageRunner
from pimqc.plotting.dataset import DatasetPlotter
from pimqc.processing import DatasetAuditPayload, StageResult
from pimqc.serialization import read_audit_payload, write_audit_payload


@pytest.fixture
def input_tables():
    """Include duplicates, zeros, shuffled metadata, and batch order resets."""
    metadata = pd.DataFrame(
        {
            "Sample Name": ["S1", "Q2", "Q1"],
            "Sample Type": ["Sample", "QC", "QC"],
            "Batch": ["B1", "B2", "B1"],
            "Inject Order": [2, 1, 1],
        }
    )
    intensity = pd.DataFrame(
        {"Q1": [2.0, 4.0, 0.0], "S1": [4.0, 6.0, 10.0], "Q2": [6.0, 8.0, 12.0]},
        index=[" F1", "F1 ", "F2"],
    )
    return metadata, intensity


def test_builder_result_preserves_alignment_and_preprocessing(input_tables):
    """Keep the scientific construction rules at the new stage boundary."""
    metadata, intensity = input_tables
    metadata_before = metadata.copy(deep=True)
    intensity_before = intensity.copy(deep=True)
    result = MetaboDatasetBuilder(metadata, intensity).run_build()

    assert isinstance(result, StageResult)
    assert isinstance(result.data, MetaboDataset)
    assert isinstance(result.audit, DatasetAuditPayload)
    expected = pd.DataFrame(
        [[3.0, 5.0, 7.0], [np.nan, 10.0, 12.0]],
        index=pd.Index(["F1", "F2"], name="Metabolite"),
        columns=pd.Index(["Q1", "S1", "Q2"], name="Sample Name"),
    )
    pd.testing.assert_frame_equal(result.data.intensity, expected)
    assert result.data.sample_metadata.index.tolist() == ["Q1", "S1", "Q2"]
    assert result.data.sample_metadata["Inject Order"].tolist() == [1, 2, 3]
    assert result.audit.metrics == result.data.dataset_metrics
    pd.testing.assert_frame_equal(metadata, metadata_before)
    pd.testing.assert_frame_equal(intensity, intensity_before)


def test_builder_resolves_custom_schema_and_roles(input_tables):
    """Preserve configured acquisition semantics in both data and audit."""
    metadata, intensity = input_tables
    metadata = metadata.rename(
        columns={
            "Sample Name": "ID",
            "Sample Type": "Role",
            "Batch": "Run",
            "Inject Order": "Order",
        }
    )
    metadata["Role"] = metadata["Role"].replace(
        {"QC": "Pool", "Sample": "Study"}
    )
    params = {
        "Dataset": {
            "sample_name": "ID",
            "sample_type": "Role",
            "batch": "Run",
            "inject_order": "Order",
            "bio_group": None,
            "mode": "ESI-",
            "sample_dict": {
                "QC sample": "Pool",
                "Actual sample": "Study",
                "Blank sample": "Solvent",
            },
        }
    }
    result = MetaboDatasetBuilder(metadata, intensity, params).run_build()

    assert result.data.schema.sample_id == "ID"
    assert result.data.schema.roles.qc == "Pool"
    assert result.data.context.acquisition_mode == "ESI-"
    assert result.audit.metrics["samples"]["qc"] == 2
    assert result.audit.metrics["samples"]["actual"] == 1
    assert result.audit.plot_payload.data.schema == result.data.schema


def test_builder_audit_is_detached_and_serializable(input_tables, tmp_path):
    """Keep audit redraw evidence stable after downstream data edits."""
    result = MetaboDatasetBuilder(*input_tables).run_build()
    expected = result.data.intensity.copy(deep=True)
    result.data.intensity.iloc[0, 0] = -999
    pd.testing.assert_frame_equal(
        result.audit.plot_payload.data.intensity, expected
    )

    artifact = tmp_path / "dataset-audit.pimqc"
    write_audit_payload(result.audit, artifact)
    restored = read_audit_payload(artifact)
    assert isinstance(restored, DatasetAuditPayload)
    assert restored.metrics == result.audit.metrics
    pd.testing.assert_frame_equal(
        restored.plot_payload.data.intensity, expected
    )


@pytest.mark.parametrize("action", ["compute_build", "run_build"])
def test_builder_in_memory_skips_artifact_operations(input_tables, action):
    """Return data and audit without filesystem or plot operations."""
    with (
        patch("pimqc.processing.stage.ensure_directory") as ensure_directory,
        patch.object(pd.DataFrame, "to_csv") as to_csv,
        patch.object(DatasetPlotter, "plot_dataset_dashboard") as plot,
    ):
        result = getattr(MetaboDatasetBuilder(*input_tables), action)()

    assert result.audit.plot_payload is not None
    ensure_directory.assert_not_called()
    to_csv.assert_not_called()
    plot.assert_not_called()


def test_builder_exports_csv_then_renders_saved_payload(input_tables, tmp_path):
    """Preserve artifact names and render the audit that the caller receives."""
    output_dir = tmp_path / "raw"
    observed_payloads = []
    figure = object()
    plotter = Mock()
    plotter.plot_dataset_dashboard.return_value = figure

    def create_plotter(payload):
        assert (output_dir / "Raw_Data_Intensity.csv").is_file()
        observed_payloads.append(payload)
        return plotter

    with patch("pimqc.plotting.dataset.DatasetPlotter", new=create_plotter):
        result = MetaboDatasetBuilder(*input_tables).run_build(output_dir)

    assert observed_payloads == [result.audit.plot_payload]
    csv_path = output_dir / "Raw_Data_Intensity.csv"
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert csv_path.read_text(encoding="utf-8-sig") == (
        result.data.annotated_frame().to_csv(na_rep="NA", lineterminator="\n")
    )
    plotter.save_and_show_pw.assert_called_once_with(
        pw_obj=figure,
        file_path=os.path.join(output_dir, "Global_Acquisition_Overview.svg"),
    )


def test_builder_failed_validation_creates_no_artifacts(input_tables, tmp_path):
    """Reject mismatched sample IDs before creating a stage directory."""
    metadata, intensity = input_tables
    output_dir = tmp_path / "must-not-exist"
    intensity = intensity.rename(columns={"Q1": "unknown"})
    with pytest.raises(AssertionError, match="Sample inconsistency"):
        MetaboDatasetBuilder(metadata, intensity).run_build(output_dir)
    assert not output_dir.exists()


def test_dataset_compatibility_entries_retain_return_and_export_contracts(
    input_tables, tmp_path
):
    """The old entry points still return data, with their original artifacts."""
    with patch.object(DatasetPlotter, "plot_dataset_dashboard") as plot:
        old_style = MetaboDatasetBuilder(*input_tables).execute_build(tmp_path)
        plot.assert_not_called()
    assert (tmp_path / "Raw_Data_Intensity.csv").is_file()
    functional = build_dataset(*input_tables)
    pd.testing.assert_frame_equal(functional.intensity, old_style.intensity)
    pd.testing.assert_frame_equal(
        functional.sample_metadata, old_style.sample_metadata
    )
    assert functional.context == old_style.context


def test_dataset_renderer_does_not_rerun_builder(input_tables, tmp_path):
    """A finished result can be rendered without rebuilding input tables."""
    builder = MetaboDatasetBuilder(*input_tables)
    result = builder.compute_build()
    runner = DatasetBuildStageRunner(builder, tmp_path)
    with (
        patch.object(builder, "compute_build", side_effect=AssertionError),
        patch("pimqc.dataset.runner._render_dataset_dashboard") as render,
    ):
        runner.render(result)
    render.assert_called_once_with(result.audit.plot_payload, Path(tmp_path))
