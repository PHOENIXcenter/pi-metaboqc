"""Typed scientific audit payloads for every pi-metaboqc stage.

The payloads replace generic result dictionaries with named, stage-specific
fields. Processors and consumers access them through ``result.audit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core import MetaboDataset
from ..plotting.payloads import (
    AssessmentPlotPayload,
    CorrectionPlotPayload,
    DatasetPlotPayload,
    FilteringPlotPayload,
    ImputationPlotPayload,
    NormalizationPlotPayload,
)


@dataclass
class AuditPayload(ABC):
    """Base contract shared by all versioned stage audit payloads."""

    audit_type: ClassVar[str] = "audit"
    schema_version: ClassVar[str] = "1.0"

    @property
    @abstractmethod
    def metrics(self) -> Mapping[str, Any]:
        """Return report-facing scientific metrics."""

    def contract_identity(self) -> dict[str, str]:
        """Return the stable audit name and schema version."""
        return {
            "audit_type": self.audit_type,
            "schema_version": self.schema_version,
        }


@dataclass
class DatasetAuditPayload(AuditPayload):
    """Audit summary emitted for the validated raw dataset."""

    metric_values: Mapping[str, Any] = field(default_factory=dict)
    plot_payload: DatasetPlotPayload | None = field(
        default=None,
        repr=False,
    )

    audit_type: ClassVar[str] = "dataset"

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return dataset construction and composition metrics."""
        return self.metric_values


@dataclass
class AssessQualityAuditPayload(AuditPayload):
    """Typed quality-assessment context and sample-level outlier audit."""

    metric_values: Mapping[str, Any] = field(default_factory=dict)
    outliers: pd.DataFrame = field(default_factory=pd.DataFrame)
    qc_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    batch_qc_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_results: Mapping[str, Any] = field(default_factory=dict)
    rsd_distribution: Mapping[str, Any] = field(default_factory=dict)
    internal_standard_flags: pd.Series | None = None
    outlier_reference_flags: pd.Series | None = None
    internal_standard_data: pd.DataFrame | None = None
    outlier_reference_data: pd.DataFrame | None = None
    skipped: bool = False
    sample_type_column: str = "Sample Type"
    sample_name_column: str = "Sample Name"
    batch_column: str = "Batch"
    injection_order_column: str = "Inject Order"
    qc_label: str = "QC"
    actual_label: str = "Sample"
    correlation_method: str = "Spearman"
    boundary_type: str = "IQR"
    qc_batches: Sequence[Any] = ()
    qc_correlation_mask: np.ndarray | None = None
    internal_standard_ids: Sequence[Any] = ()
    outlier_reference_ids: Sequence[Any] = ()
    plot_payload: AssessmentPlotPayload | None = field(
        default=None,
        repr=False,
    )

    audit_type: ClassVar[str] = "quality_assessment"

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return assessment summaries used by reports."""
        return self.metric_values


@dataclass
class SampleFilterAuditPayload(AuditPayload):
    """Typed sample-attrition audit used inside missingness filtering."""

    sample_tracking: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped_sample_ids: Sequence[Any] = ()
    plot_payload: FilteringPlotPayload | None = field(
        default=None,
        repr=False,
    )
    execution_status: str = "completed"
    skip_reason: str | None = None
    missing_value_count: int = 0
    input_sample_count: int = 0
    output_sample_count: int = 0

    audit_type: ClassVar[str] = "sample_filter"

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return sample-filter execution and retention metrics."""
        return {
            "execution_status": self.execution_status,
            "skip_reason": self.skip_reason,
            "missing_value_count": self.missing_value_count,
            "missing_values_detected": self.missing_value_count > 0,
            "thresholds": {
                "sample_mv_tol": (
                    self.plot_payload.sample_mv_tolerance
                    if self.plot_payload is not None
                    else None
                )
            },
            "sample_retention": {
                "input_count": self.input_sample_count,
                "output_count": self.output_sample_count,
                "dropped_count": len(self.dropped_sample_ids),
            },
        }


@dataclass
class MissingValueFilterAuditPayload(AuditPayload):
    """Typed high-missingness feature classification and attrition audit."""

    metric_values: Mapping[str, Any]
    feature_tracking: pd.DataFrame
    sample_filtered_data: MetaboDataset
    sample_tracking: pd.DataFrame
    qc_mask: np.ndarray
    valid_biological_groups: Sequence[Any]
    plot_payload: FilteringPlotPayload = field(repr=False)
    tables: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "completed"
    skip_reason: str | None = None
    missing_value_count: int = 0

    audit_type: ClassVar[str] = "missing_value_filter"

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return missingness and retention metrics."""
        return self.metric_values


@dataclass
class CorrectionAuditPayload(AuditPayload):
    """Typed correction selection, candidate, and fitted-baseline audit."""

    metric_values: Mapping[str, Any]
    requested_method: str
    selected_method: str
    selected_label: str
    is_auto: bool
    sample_type_column: str
    batch_column: str
    injection_order_column: str
    qc_label: str
    actual_label: str
    plot_payload: CorrectionPlotPayload = field(repr=False)

    audit_type: ClassVar[str] = "signal_correction"

    @property
    def candidate_results(self) -> Mapping[str, Any]:
        """Access the single candidate store owned by the plot payload."""
        return self.plot_payload.candidate_results

    @property
    def selected_prediction(self) -> MetaboDataset | None:
        """Access the selected fitted baseline without storing it twice."""
        return self.plot_payload.selected_prediction

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return correction and method-selection metrics."""
        return self.metric_values


@dataclass
class QualityFilterAuditPayload(AuditPayload):
    """Typed Blank/QC, QC-RSD, and feature-attrition audit."""

    metric_values: Mapping[str, Any]
    feature_tracking: pd.DataFrame
    plot_payload: FilteringPlotPayload = field(repr=False)
    tables: dict[str, Any] = field(default_factory=dict)

    audit_type: ClassVar[str] = "quality_filter"

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return low-quality filtering metrics."""
        return self.metric_values


@dataclass
class ImputationAuditPayload(AuditPayload):
    """Typed MAR/MNAR method selection and reconstruction audit."""

    metric_values: Mapping[str, Any]
    candidate_results: Mapping[str, Any]
    requested_method: str
    selected_method: str
    selected_label: str
    is_auto: bool
    mar_feature_count: int
    has_candidate_cache: bool
    skipped: bool
    plot_payload: ImputationPlotPayload | None = field(repr=False)

    audit_type: ClassVar[str] = "imputation"

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return imputation and reconstruction metrics."""
        return self.metric_values


@dataclass
class NormalizationAuditPayload(AuditPayload):
    """Typed normalization strategy and AUTO candidate passport."""

    metric_values: Mapping[str, Any]
    output_suffix: str
    plot_payload: NormalizationPlotPayload = field(repr=False)

    audit_type: ClassVar[str] = "normalization"

    @property
    def selection(self) -> Mapping[str, Any]:
        """Access the single normalization selection passport."""
        return self.plot_payload.selection

    @property
    def candidate_results(self) -> Sequence[Mapping[str, Any]] | None:
        """Expose candidate summaries from the selection passport."""
        return self.selection.get("candidate_results")

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return normalization and selection metrics."""
        return self.metric_values
