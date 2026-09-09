"""Exercise complete pipeline orchestration in memory and with artifacts.

The synthetic and bundled-data scenarios traverse all processing stages while
patching expensive file writers. These are deliberately slow integration tests
because their purpose is to validate stage composition, result contracts, and
artifact dispatch rather than individual numerical kernels.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from pimqc.core import MetaboDataset
from pimqc.pipeline import PipelineResult, run_pipeline
from pimqc.plotting.payloads import (
    AssessmentPlotPayload,
    CorrectionPlotPayload,
    DatasetPlotPayload,
    FilteringPlotPayload,
    ImputationPlotPayload,
    NormalizationPlotPayload,
)
from pimqc.processing.audit import (
    AssessQualityAuditPayload,
    CorrectionAuditPayload,
    DatasetAuditPayload,
    ImputationAuditPayload,
    MissingValueFilterAuditPayload,
    NormalizationAuditPayload,
    QualityFilterAuditPayload,
)
from pimqc.serialization import read_audit_payload, write_audit_payload

PipelineData = tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]
pytestmark = pytest.mark.slow


def test_complete_pipeline_without_missing_values(synthetic_pipeline_data):
    """Preserve the no-MV route through correction, imputation, and reports."""
    metadata, intensity, parameters = synthetic_pipeline_data
    filled = intensity.T.fillna(intensity.median(axis=1)).T.astype(float)
    parameters["SignalCorrector"]["n_jobs"] = 1
    parameters["DataNormalizer"]["n_jobs"] = 1
    result = run_pipeline(metadata, filled, parameters)
    assert (
        result.stage_results["high_mv_feature_filtering"].audit.execution_status
        == "skipped"
    )
    assert result.stage_results["missing_value_imputation"].audit.skipped
    assert not result.data.intensity.isna().any().any()
    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in result.data.intensity.dtypes
    )


@patch("pimqc.pipeline.ensure_directory")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.plotting.base.BasePlotter.save_and_close_fig")
@patch("pimqc.plotting.base.BasePlotter.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_with_artifact_dispatch(
    mock_visual_reporter: Mock,
    mock_narrative_reporter: Mock,
    mock_save_dashboard: Mock,
    mock_save_figure: Mock,
    mock_to_csv: Mock,
    mock_ensure_directory: Mock,
    synthetic_pipeline_data: PipelineData,
    tmp_path: Path,
) -> None:
    """Return structured results and dispatch every artifact category."""
    metadata, intensity, parameters = synthetic_pipeline_data
    mock_ensure_directory.side_effect = lambda path: Path(path)

    result = run_pipeline(
        meta_df=metadata,
        int_df=intensity,
        params=parameters,
        output_dir=str(tmp_path / "synthetic_pipeline"),
    )

    assert isinstance(result, PipelineResult)
    assert result.data is result.stage_tables["normalized"]
    assert result.stage_results["normalization"].data is result.data
    assert result.assessments["normalization"].audit.metrics
    assert (
        result.pipeline_metrics["normalization"]
        is result.stage_results["normalization"].audit.metrics
    )
    assert result.report_input.pipeline_metrics is result.pipeline_metrics
    assert mock_to_csv.call_count > 0
    assert mock_save_figure.call_count > 0
    assert mock_narrative_reporter.called
    assert mock_visual_reporter.called
    compile_report = mock_visual_reporter.return_value.compile_assessor_report
    report_audits = compile_report.call_args.kwargs["audits"]
    assert len(report_audits) == sum(
        not item.audit.skipped for item in result.assessments.values()
    )
    assert list(report_audits.values()) == [
        item.audit
        for item in result.assessments.values()
        if not item.audit.skipped
    ]
    assert mock_save_dashboard.call_count > 0


@patch("pimqc.pipeline.ensure_directory")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.plotting.base.BasePlotter.save_and_close_fig")
@patch("pimqc.plotting.base.BasePlotter.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_in_memory_has_no_artifact_side_effects(
    mock_visual_reporter: Mock,
    mock_narrative_reporter: Mock,
    mock_save_dashboard: Mock,
    mock_save_figure: Mock,
    mock_to_csv: Mock,
    mock_ensure_directory: Mock,
    synthetic_pipeline_data: PipelineData,
    tmp_path: Path,
) -> None:
    """Return all pipeline products without creating filesystem artifacts."""
    metadata, intensity, parameters = synthetic_pipeline_data

    result = run_pipeline(
        meta_df=metadata,
        int_df=intensity,
        params=parameters,
        output_dir=None,
    )

    assert isinstance(result, PipelineResult)
    assert set(result.stage_results) == {
        "raw_dataset",
        "sample_missing_value_filtering",
        "high_mv_feature_filtering",
        "signal_correction",
        "low_quality_feature_filtering",
        "missing_value_imputation",
        "normalization",
    }
    assert {
        "raw_dataset",
        "high_mv_feature_filtering",
        "low_quality_feature_filtering",
        "missing_value_imputation",
        "normalization",
    }.issubset(result.assessments)
    assert any(
        key.startswith("signal_correction/") for key in result.assessments
    )
    assert {
        "raw",
        "high_mv_filtered",
        "quality_filtered",
        "imputed",
        "normalized",
    }.issubset(result.stage_tables)
    assert any(key.startswith("corrected/") for key in result.stage_tables)
    assert result.output_dir is None
    assert result.report_generated is False
    assert result.data is result.stage_tables["normalized"]
    assert result.assessments["raw_dataset"].data.qc_correlation.size > 0
    expected_audits = {
        "raw_dataset": DatasetAuditPayload,
        "high_mv_feature_filtering": MissingValueFilterAuditPayload,
        "signal_correction": CorrectionAuditPayload,
        "low_quality_feature_filtering": QualityFilterAuditPayload,
        "missing_value_imputation": ImputationAuditPayload,
        "normalization": NormalizationAuditPayload,
    }
    for stage_name, audit_type in expected_audits.items():
        assert isinstance(result.stage_results[stage_name].audit, audit_type)
    assert all(
        isinstance(assessment.audit, AssessQualityAuditPayload)
        for assessment in result.assessments.values()
    )
    expected_plot_payloads = {
        "raw_dataset": DatasetPlotPayload,
        "high_mv_feature_filtering": FilteringPlotPayload,
        "signal_correction": CorrectionPlotPayload,
        "low_quality_feature_filtering": FilteringPlotPayload,
        "missing_value_imputation": ImputationPlotPayload,
        "normalization": NormalizationPlotPayload,
    }
    for stage_name, payload_type in expected_plot_payloads.items():
        audit = result.stage_results[stage_name].audit
        if isinstance(audit, ImputationAuditPayload) and audit.skipped:
            assert audit.plot_payload is None
            continue
        assert isinstance(audit.plot_payload, payload_type)
        assert type(audit.plot_payload.primary_dataset) is MetaboDataset
        assert type(audit.plot_payload.primary_data) is pd.DataFrame
    for assessment in result.assessments.values():
        assert isinstance(assessment.audit.plot_payload, AssessmentPlotPayload)
        assert (
            type(assessment.audit.plot_payload.primary_dataset) is MetaboDataset
        )
        assert type(assessment.audit.plot_payload.primary_data) is pd.DataFrame
    assert all(
        type(dataset) is MetaboDataset
        for dataset in result.stage_tables.values()
    )
    all_audits = {
        **{
            f"stage-{name}": stage_result.audit
            for name, stage_result in result.stage_results.items()
        },
        **{
            f"assessment-{name}": assessment.audit
            for name, assessment in result.assessments.items()
        },
    }
    for audit_name, audit in all_audits.items():
        artifact = tmp_path / f"{audit_name.replace('/', '__')}.pimqc"
        write_audit_payload(audit, artifact)
        restored = read_audit_payload(artifact)
        assert type(restored) is type(audit)
        assert restored.contract_identity() == audit.contract_identity()
    assert result.report_input.asset_manifest == {}
    mock_ensure_directory.assert_not_called()
    mock_to_csv.assert_not_called()
    mock_save_figure.assert_not_called()
    mock_save_dashboard.assert_not_called()
    mock_visual_reporter.assert_not_called()
    mock_narrative_reporter.assert_not_called()


@patch("pimqc.pipeline.ensure_directory")
@patch("pandas.DataFrame.to_csv")
@patch("pimqc.plotting.base.BasePlotter.save_and_close_fig")
@patch("pimqc.plotting.base.BasePlotter.save_and_show_pw")
@patch("pimqc.pipeline.ru.NarrativeStatsReporter")
@patch("pimqc.pipeline.ru.VisualAssetReporter")
def test_run_pipeline_with_bundled_project_data(
    mock_visual_reporter: Mock,
    mock_narrative_reporter: Mock,
    mock_save_dashboard: Mock,
    mock_save_figure: Mock,
    mock_to_csv: Mock,
    mock_ensure_directory: Mock,
    real_project_data: PipelineData,
    tmp_path: Path,
) -> None:
    """Confirm the complete pipeline accepts the distributed demo dataset."""
    metadata, intensity, parameters = real_project_data
    mock_ensure_directory.side_effect = lambda path: Path(path)

    result = run_pipeline(
        meta_df=metadata,
        int_df=intensity,
        params=parameters,
        output_dir=str(tmp_path / "real_pipeline"),
    )

    assert isinstance(result, PipelineResult)
    assert result.data is result.stage_tables["normalized"]
    assert mock_to_csv.call_count > 0
    assert mock_narrative_reporter.called
    assert mock_visual_reporter.called
    assert mock_save_figure.call_count > 0
    assert mock_save_dashboard.call_count > 0
    assert (
        result.pipeline_metrics["missing_value_imputation"]["selection"][
            "requested_method"
        ]
        == "Auto"
    )
    selected_method = result.pipeline_metrics["missing_value_imputation"][
        "selection"
    ]["selected_method"]
    candidate_results = result.pipeline_metrics["missing_value_imputation"][
        "selection"
    ]["candidate_results"]
    selected_result = next(
        candidate
        for candidate in candidate_results
        if candidate["method"] == selected_method
    )
    assert "jsd_total" in selected_result
    assert "wasserstein_normalized" in selected_result
