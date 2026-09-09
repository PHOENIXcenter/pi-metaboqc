"""Validate Phase 2 typed audit identities and result boundaries.

These tests focus on contract shape rather than scientific calculations. Full
stage-to-audit routing is exercised by the integration pipeline tests.
"""

from dataclasses import fields

import pytest

import pimqc
from pimqc.processing.audit import (
    AssessQualityAuditPayload,
    CorrectionAuditPayload,
    DatasetAuditPayload,
    ImputationAuditPayload,
    MissingValueFilterAuditPayload,
    NormalizationAuditPayload,
    QualityFilterAuditPayload,
    SampleFilterAuditPayload,
)
from pimqc.processing.stage import StageResult


def _field_names(data_class: type[object]) -> tuple[str, ...]:
    """Return dataclass fields in their public constructor order."""
    return tuple(item.name for item in fields(data_class))


def test_typed_audit_field_surfaces_are_explicit() -> None:
    """Freeze named fields for every Phase 2 stage payload."""
    assert _field_names(DatasetAuditPayload) == (
        "metric_values",
        "plot_payload",
    )
    assert _field_names(AssessQualityAuditPayload) == (
        "metric_values",
        "outliers",
        "qc_correlation",
        "batch_qc_correlation",
        "pca_results",
        "rsd_distribution",
        "internal_standard_flags",
        "outlier_reference_flags",
        "internal_standard_data",
        "outlier_reference_data",
        "skipped",
        "sample_type_column",
        "sample_name_column",
        "batch_column",
        "injection_order_column",
        "qc_label",
        "actual_label",
        "correlation_method",
        "boundary_type",
        "qc_batches",
        "qc_correlation_mask",
        "internal_standard_ids",
        "outlier_reference_ids",
        "plot_payload",
    )
    assert _field_names(SampleFilterAuditPayload) == (
        "sample_tracking",
        "dropped_sample_ids",
        "plot_payload",
        "execution_status",
        "skip_reason",
        "missing_value_count",
        "input_sample_count",
        "output_sample_count",
    )
    assert _field_names(MissingValueFilterAuditPayload) == (
        "metric_values",
        "feature_tracking",
        "sample_filtered_data",
        "sample_tracking",
        "qc_mask",
        "valid_biological_groups",
        "plot_payload",
        "tables",
        "execution_status",
        "skip_reason",
        "missing_value_count",
    )
    assert _field_names(CorrectionAuditPayload) == (
        "metric_values",
        "requested_method",
        "selected_method",
        "selected_label",
        "is_auto",
        "sample_type_column",
        "batch_column",
        "injection_order_column",
        "qc_label",
        "actual_label",
        "plot_payload",
    )
    assert _field_names(QualityFilterAuditPayload) == (
        "metric_values",
        "feature_tracking",
        "plot_payload",
        "tables",
    )
    assert _field_names(ImputationAuditPayload) == (
        "metric_values",
        "candidate_results",
        "requested_method",
        "selected_method",
        "selected_label",
        "is_auto",
        "mar_feature_count",
        "has_candidate_cache",
        "skipped",
        "plot_payload",
    )
    assert _field_names(NormalizationAuditPayload) == (
        "metric_values",
        "output_suffix",
        "plot_payload",
    )


@pytest.mark.parametrize(
    ("payload", "audit_type"),
    [
        (DatasetAuditPayload(), "dataset"),
        (AssessQualityAuditPayload(), "quality_assessment"),
        (SampleFilterAuditPayload(), "sample_filter"),
    ],
)
def test_audit_contract_identity_is_versioned(
    payload: object,
    audit_type: str,
) -> None:
    """Expose stable identities without claiming JSON portability yet."""
    assert payload.contract_identity() == {
        "audit_type": audit_type,
        "schema_version": "1.0",
    }


def test_stage_result_accepts_and_narrows_typed_audit() -> None:
    """Make the audit type explicit to stage consumers."""
    audit = DatasetAuditPayload(metric_values={"features": 2})
    result = StageResult(data="matrix", audit=audit)

    assert result.require_audit(DatasetAuditPayload) is audit
    assert result.audit is audit
    assert result.audit.metrics == {"features": 2}
    with pytest.raises(TypeError, match="Expected AssessQualityAuditPayload"):
        result.require_audit(AssessQualityAuditPayload)


def test_stage_result_requires_a_typed_audit() -> None:
    """Reject incomplete and removed generic-result construction forms."""
    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        StageResult(data="matrix")
    with pytest.raises(
        TypeError, match="unexpected keyword argument 'metrics'"
    ):
        StageResult(data="matrix", metrics={"score": 0.8})  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="AuditPayload"):
        StageResult(data="matrix", audit=object())  # type: ignore[arg-type]


def test_public_api_exports_primary_audit_types() -> None:
    """Make adapter-facing payload classes discoverable from the package."""
    assert pimqc.DatasetAuditPayload is DatasetAuditPayload
    assert pimqc.AssessQualityAuditPayload is AssessQualityAuditPayload
    assert pimqc.CorrectionAuditPayload is CorrectionAuditPayload
    assert pimqc.ImputationAuditPayload is ImputationAuditPayload
