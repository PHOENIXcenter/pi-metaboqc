"""Stable wire-name registry for pi-metaboqc domain dataclasses.

Only the classes declared in this module may be reconstructed from an
artifact.  Stable names are intentionally independent from Python module paths
so internal package moves do not silently change the on-disk contract and no
artifact can request an arbitrary dynamic import.
"""

from __future__ import annotations

from typing import Any

from ..core.dataset import (
    DatasetSchema,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)
from ..plotting.payloads import (
    AssessmentPlotPayload,
    CorrectionPlotPayload,
    DatasetPlotPayload,
    FilteringPlotPayload,
    ImputationPlotPayload,
    NormalizationPlotPayload,
)
from ..processing.audit import (
    AssessQualityAuditPayload,
    CorrectionAuditPayload,
    DatasetAuditPayload,
    ImputationAuditPayload,
    MissingValueFilterAuditPayload,
    NormalizationAuditPayload,
    QualityFilterAuditPayload,
    SampleFilterAuditPayload,
)


WIRE_TYPES: dict[str, type[Any]] = {
    "core.sample_roles": SampleRoleLabels,
    "core.dataset_schema": DatasetSchema,
    "core.processing_context": ProcessingContext,
    "core.metabo_dataset": MetaboDataset,
    "audit.dataset": DatasetAuditPayload,
    "audit.quality_assessment": AssessQualityAuditPayload,
    "audit.sample_filter": SampleFilterAuditPayload,
    "audit.missing_value_filter": MissingValueFilterAuditPayload,
    "audit.signal_correction": CorrectionAuditPayload,
    "audit.quality_filter": QualityFilterAuditPayload,
    "audit.imputation": ImputationAuditPayload,
    "audit.normalization": NormalizationAuditPayload,
    "plot.dataset": DatasetPlotPayload,
    "plot.assessment": AssessmentPlotPayload,
    "plot.filtering": FilteringPlotPayload,
    "plot.correction": CorrectionPlotPayload,
    "plot.imputation": ImputationPlotPayload,
    "plot.normalization": NormalizationPlotPayload,
}

CLASS_WIRE_TYPES: dict[type[Any], str] = {
    value: key for key, value in WIRE_TYPES.items()
}


def wire_name_for(value: Any) -> str | None:
    """Return the stable registered name for an exact dataclass type."""
    return CLASS_WIRE_TYPES.get(type(value))


def class_for_wire_name(name: str) -> type[Any] | None:
    """Return the registered class for a stable wire name, if available."""
    return WIRE_TYPES.get(name)
