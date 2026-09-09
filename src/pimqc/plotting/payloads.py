"""Explicit, processor-free input payloads for every plotting suite.

Plot payloads own detached ``MetaboDataset`` snapshots, resolved thresholds, and
the small pure helpers needed by current panels. Plotters must accept one of
these payloads rather than a live assessment or processing engine.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

import pandas as pd

from ..core.dataset import MetaboDataset


def snapshot_dataset(source: MetaboDataset) -> MetaboDataset:
    """Return a fully detached dataset snapshot."""
    if not isinstance(source, MetaboDataset):
        raise TypeError("source must be a MetaboDataset instance.")
    return source.copy()


def snapshot_plot_value(value: Any) -> Any:
    """Detach dataframe values recursively while retaining scalar context."""
    if isinstance(value, MetaboDataset):
        return snapshot_dataset(value)
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, pd.Series):
        return value.copy(deep=True)
    if isinstance(value, dict):
        return {key: snapshot_plot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_plot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_plot_value(item) for item in value)
    return copy.deepcopy(value)


class PlotPayload(ABC):
    """Common plotting input contract with one primary data snapshot."""

    payload_type: ClassVar[str] = "plot"
    schema_version: ClassVar[str] = "1.0"

    @property
    @abstractmethod
    def primary_dataset(self) -> MetaboDataset:
        """Return the dataset used to initialize plotting semantics."""

    @property
    def primary_data(self) -> pd.DataFrame:
        """Return the ordinary annotated dataframe consumed by plot panels."""
        return self.primary_dataset.annotated_frame()

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Return resolved plotting attributes from the primary snapshot."""
        return self.primary_dataset.settings()

    def contract_identity(self) -> dict[str, str]:
        """Return the stable plotting payload name and schema version."""
        return {
            "payload_type": self.payload_type,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class DatasetPlotPayload(PlotPayload):
    """Dataset-construction overview inputs."""

    data: MetaboDataset

    payload_type: ClassVar[str] = "dataset_plot"

    @property
    def primary_dataset(self) -> MetaboDataset:
        """Return the validated raw-data snapshot."""
        return self.data


@dataclass(frozen=True)
class AssessmentPlotPayload(PlotPayload):
    """Quality-assessment style and acquisition context."""

    data: MetaboDataset
    is_multi_batch: bool

    payload_type: ClassVar[str] = "assessment_plot"

    @property
    def primary_dataset(self) -> MetaboDataset:
        """Return the assessed data snapshot."""
        return self.data


@dataclass(frozen=True)
class FilteringPlotPayload(PlotPayload):
    """Filtering matrix, audit tables, and resolved decision thresholds."""

    data: MetaboDataset
    audit_tables: Mapping[str, Any] = field(default_factory=dict)
    sample_mv_tolerance: float = 0.5
    active_base_tolerance: float = 0.5
    mnar_group_mv_tolerance: float = 0.8
    mnar_qc_mv_tolerance: float = 0.2
    mnar_intensity_threshold: float | None = None
    mnar_intensity_percentile: float = 0.1
    blank_qc_ratio_tolerance: float = 0.2
    qc_rsd_tolerance: float = 0.3
    stage_status: str = "completed"
    sample_filter_status: str = "not_run"
    missing_values_detected: bool = True
    biological_groups_available: bool = False
    skip_reason: str | None = None

    payload_type: ClassVar[str] = "filtering_plot"

    @property
    def primary_dataset(self) -> MetaboDataset:
        """Return the filtering-stage data snapshot."""
        return self.data


@dataclass(frozen=True)
class CorrectionPlotPayload(PlotPayload):
    """Correction source, selected outputs, candidates, and IS context."""

    source_data: MetaboDataset
    selected_stages: Mapping[str, MetaboDataset]
    candidate_results: Mapping[str, Any]
    selected_prediction: MetaboDataset | None
    internal_standard_ids: Sequence[Any]
    boundary_type: str

    payload_type: ClassVar[str] = "correction_plot"

    @property
    def primary_dataset(self) -> MetaboDataset:
        """Return the pre-correction source snapshot."""
        return self.source_data

    def extract_qc_rsd_series(self, data: pd.DataFrame) -> pd.Series:
        """Retrieve precomputed RSD for a candidate-owned stage matrix."""
        for candidate in self.candidate_results.values():
            for frames_key, values_key in (
                ("stage_dfs", "stage_qc_rsd"),
                ("stage_oof_dfs", "stage_oof_qc_rsd"),
            ):
                for label, frame in candidate.get(frames_key, {}).items():
                    if frame is data:
                        return (
                            candidate.get(values_key, {})
                            .get(label, pd.Series(dtype=float))
                            .copy()
                        )
        raise ValueError("QC RSD is unavailable for this candidate matrix.")


@dataclass(frozen=True)
class ImputationPlotPayload(PlotPayload):
    """Before/after imputation matrices and deterministic plot seed."""

    raw_data: MetaboDataset
    imputed_data: MetaboDataset
    global_seed: int
    sample_structure: Mapping[str, Any] = field(default_factory=dict)

    payload_type: ClassVar[str] = "imputation_plot"

    @property
    def primary_dataset(self) -> MetaboDataset:
        """Return the completed imputation snapshot."""
        return self.imputed_data


@dataclass(frozen=True)
class NormalizationPlotPayload(PlotPayload):
    """Before/after normalization matrices and resolved scoring constants."""

    raw_data: MetaboDataset
    normalized_data: MetaboDataset
    selection: Mapping[str, Any]
    score_component_weights: Mapping[str, float]
    sample_scale_log_ratio_tolerance: float
    sample_scale_relative_delta_tolerance: float
    global_seed: int
    sample_structure: Mapping[str, Any] = field(default_factory=dict)
    qc_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    payload_type: ClassVar[str] = "normalization_plot"

    @property
    def primary_dataset(self) -> MetaboDataset:
        """Return the normalized data snapshot."""
        return self.normalized_data
