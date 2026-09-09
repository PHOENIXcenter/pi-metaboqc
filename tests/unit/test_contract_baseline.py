"""Freeze structured runtime surfaces across the staged payload refactoring.

Phase 0 captured the v1.3.1 dictionaries. Phase 2 narrows ``StageResult`` to
data and typed audit.
"""

from dataclasses import fields

from pimqc.pipeline import PipelineResult
from pimqc.processing.assessment import AssessmentDiagnostics
from pimqc.processing import DatasetAuditPayload
from pimqc.processing.stage import StageResult
from pimqc.reporting import ReportInput


def _field_names(data_class: type[object]) -> tuple[str, ...]:
    """Return dataclass fields in their public constructor order."""
    return tuple(item.name for item in fields(data_class))


def test_stage_result_phase2_field_surface_is_explicit() -> None:
    """Freeze the data-plus-typed-audit result boundary."""
    assert _field_names(StageResult) == (
        "data",
        "audit",
    )


def test_dataset_audit_mutable_defaults_are_isolated() -> None:
    """Prevent audit state from leaking between stage runs."""
    first = StageResult(data="first", audit=DatasetAuditPayload())
    second = StageResult(data="second", audit=DatasetAuditPayload())

    first.audit.metric_values["status"] = "complete"

    assert second.audit.metric_values == {}


def test_assessment_diagnostics_v131_field_surface_is_explicit() -> None:
    """Freeze every table and plot input currently returned by assessment."""
    assert _field_names(AssessmentDiagnostics) == (
        "qc_correlation",
        "batch_qc_correlation",
        "pca",
        "rsd_distribution",
        "internal_standard_evaluation",
        "outlier_reference_evaluation",
        "outliers",
        "internal_standard_flags",
        "outlier_reference_flags",
        "internal_standard_data",
        "outlier_reference_data",
    )


def test_pipeline_result_field_surface_and_primary_alias() -> None:
    """Freeze pipeline product ownership and the normalized-data shortcut."""
    assert _field_names(PipelineResult) == (
        "stage_tables",
        "stage_results",
        "assessments",
        "pipeline_metrics",
        "qa_metrics",
        "report_input",
        "output_dir",
        "report_generated",
        "filtering_results",
    )

    normalized = object()
    result = PipelineResult(
        stage_tables={"normalized": normalized},
        stage_results={},
        assessments={},
        pipeline_metrics={},
        qa_metrics={},
        report_input=ReportInput(
            pipeline_metrics={},
            qa_metrics={},
            metadata={},
            resolved_config={},
        ),
    )

    assert result.data is normalized
