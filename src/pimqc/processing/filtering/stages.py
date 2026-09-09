"""Independent sample, feature-missingness, and quality filter stages.

The original :class:`FeatureFilter` remains available as a native-Python
facade.  The classes in this module provide the explicit three-stage API used
by the pipeline and by the future Rachis adapter.  They exchange
``MetaboDataset`` and typed ``StageResult`` objects rather than sharing a
mutable processor or pandas attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ...core import MetaboDataset
from ...runtime import log_execution_time
from ..audit import MissingValueFilterAuditPayload
from ..stage import StageResult
from .analysis import FeatureFilter
from .runner import (
    FeatureMissingValueFilteringStageRunner,
    QualityFilteringStageRunner,
    SampleFilteringStageRunner,
)


class _FilterStageFacade:
    """Delegate shared dataset/configuration behavior to the core engine."""

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize the filter with validated stage-local settings."""
        self._engine = FeatureFilter(
            data=data,
            pipeline_params=dict(pipeline_params or {}),
            **overrides,
        )

    @property
    def dataset(self) -> MetaboDataset:
        """Return the detached source dataset used by this stage."""
        return self._engine.dataset

    @property
    def frame(self) -> pd.DataFrame:
        """Return the stage-local ordinary dataframe view."""
        return self._engine.frame

    @property
    def config(self) -> dict[str, Any]:
        """Expose resolved settings to ``StageRunner``."""
        return self._engine.config

    @property
    def stats(self) -> dict[str, Any]:
        """Expose only the stage-local runtime statistics."""
        return self._engine.stats

    def _invalidate_cached_properties(self) -> None:
        """Invalidate cached values on the delegated engine."""
        self._engine._invalidate_cached_properties()


class SampleMissingValueFilter(_FilterStageFacade):
    """Filter samples whose missing-value rate exceeds the configured limit."""

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Mapping[str, Any] | None = None,
        *,
        sample_mv_tol: float | None = None,
    ) -> None:
        """Initialize the filter with validated stage-local settings."""
        super().__init__(
            data,
            pipeline_params,
            sample_mv_tol=sample_mv_tol,
        )

    def filter_samples_by_missingness(self) -> StageResult[MetaboDataset]:
        """Return the sample-filtered table and its typed audit."""
        return self._engine.filter_samples_by_missingness()

    @log_execution_time
    def run_filter_samples_by_missingness(
        self,
        output_dir: str | Path | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Execute, export, and optionally render sample filtering."""
        runner = SampleFilteringStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset({"sample_mv_tol"}),
        )
        return runner.run()


class FeatureMissingValueFilter(_FilterStageFacade):
    """Classify and filter features on a prepared sample-filtered dataset."""

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Mapping[str, Any] | None = None,
        *,
        sample_result: StageResult[MetaboDataset] | None = None,
        mv_global_tol: float | None = None,
        mv_group_tol: float | None = None,
        mv_qc_tol: float | None = None,
        mnar_group_mv_tol: float | None = None,
        mnar_qc_mv_tol: float | None = None,
        mnar_intensity_pct: float | None = None,
    ) -> None:
        """Initialize the filter with validated stage-local settings."""
        super().__init__(
            data,
            pipeline_params,
            mv_global_tol=mv_global_tol,
            mv_group_tol=mv_group_tol,
            mv_qc_tol=mv_qc_tol,
            mnar_group_mv_tol=mnar_group_mv_tol,
            mnar_qc_mv_tol=mnar_qc_mv_tol,
            mnar_intensity_pct=mnar_intensity_pct,
        )
        self.sample_result = sample_result

    def filter_features_by_missingness(
        self,
        *,
        sample_result: StageResult[MetaboDataset] | None = None,
    ) -> StageResult[MetaboDataset]:
        """Return the feature-filtered table and missingness audit."""
        return self._engine.filter_features_by_missingness(
            sample_result=sample_result or self.sample_result
        )

    @log_execution_time
    def run_filter_features_by_missingness(
        self,
        output_dir: str | Path | None = None,
        *,
        sample_result: StageResult[MetaboDataset] | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Execute, export, and optionally render feature MV filtering."""
        if sample_result is not None:
            self.sample_result = sample_result
        runner = FeatureMissingValueFilteringStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset(
                {
                    "mv_global_tol",
                    "mv_group_tol",
                    "mv_qc_tol",
                    "mnar_group_mv_tol",
                    "mnar_qc_mv_tol",
                    "mnar_intensity_pct",
                }
            ),
        )
        return runner.run()


class FeatureQualityFilter(_FilterStageFacade):
    """Filter features by Blank/QC ratio and QC-RSD rules."""

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Mapping[str, Any] | None = None,
        *,
        missingness_metadata: pd.DataFrame | pd.Series | None = None,
        qc_rsd_tol: float | None = None,
        blank_qc_ratio_tol: float | None = None,
    ) -> None:
        """Initialize the filter with validated stage-local settings."""
        super().__init__(
            data,
            pipeline_params,
            qc_rsd_tol=qc_rsd_tol,
            blank_qc_ratio_tol=blank_qc_ratio_tol,
        )
        self.missingness_metadata = missingness_metadata

    @staticmethod
    def _indices_from_metadata(
        metadata: pd.DataFrame | pd.Series | None,
    ) -> tuple[pd.Index | None, pd.Index | None]:
        """Extract MAR/MNAR identifiers from explicit feature metadata."""
        if metadata is None:
            return None, None
        if isinstance(metadata, pd.Series):
            labels = metadata
        elif isinstance(metadata, pd.DataFrame):
            if "missingness_type" in metadata.columns:
                labels = metadata["missingness_type"]
            elif "Stage1_Status" in metadata.columns:
                labels = metadata["Stage1_Status"].map(
                    lambda value: (
                        "MNAR"
                        if "MNAR" in str(value).upper()
                        else "MAR"
                        if str(value).upper() == "MAR"
                        else "INVALID"
                    )
                )
            else:
                raise ValueError(
                    "missingness_metadata must contain a 'missingness_type' "
                    "or 'Stage1_Status' column."
                )
        else:
            raise TypeError(
                "missingness_metadata must be a pandas Series or DataFrame."
            )
        labels = labels.astype(str).str.upper()
        idx_mar = labels.index[labels == "MAR"]
        idx_mnar = labels.index[labels == "MNAR"]
        if idx_mar.empty and idx_mnar.empty:
            # An audit table without any recognized class is equivalent to an
            # absent upstream missingness result for this independent action.
            return None, None
        return idx_mar, idx_mnar

    def filter_features_by_quality(
        self,
        *,
        missingness_metadata: pd.DataFrame | pd.Series | None = None,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
        missingness_tracking: pd.DataFrame | None = None,
    ) -> StageResult[MetaboDataset]:
        """Return the quality-filtered table and quality audit."""
        metadata = (
            missingness_metadata
            if missingness_metadata is not None
            else self.missingness_metadata
        )
        if idx_mar is None or idx_mnar is None:
            metadata_mar, metadata_mnar = self._indices_from_metadata(metadata)
            idx_mar = idx_mar if idx_mar is not None else metadata_mar
            idx_mnar = idx_mnar if idx_mnar is not None else metadata_mnar
        return self._engine.filter_features_by_quality(
            idx_mar=idx_mar,
            idx_mnar=idx_mnar,
            missingness_tracking=(
                missingness_tracking
                if missingness_tracking is not None
                else (
                    metadata
                    if isinstance(metadata, pd.DataFrame)
                    and "Stage1_Status" in metadata.columns
                    else None
                )
            ),
        )

    @log_execution_time
    def run_filter_features_by_quality(
        self,
        output_dir: str | Path | None = None,
        *,
        missingness_metadata: pd.DataFrame | pd.Series | None = None,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Execute, export, and optionally render quality filtering."""
        metadata = (
            missingness_metadata
            if missingness_metadata is not None
            else self.missingness_metadata
        )
        if idx_mar is None or idx_mnar is None:
            metadata_mar, metadata_mnar = self._indices_from_metadata(metadata)
            idx_mar = idx_mar if idx_mar is not None else metadata_mar
            idx_mnar = idx_mnar if idx_mnar is not None else metadata_mnar
        runner = QualityFilteringStageRunner(
            self,
            output_dir,
            idx_mar=idx_mar,
            idx_mnar=idx_mnar,
            missingness_tracking=(
                metadata
                if isinstance(metadata, pd.DataFrame)
                and "Stage1_Status" in metadata.columns
                else None
            ),
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset(
                {"qc_rsd_tol", "blank_qc_ratio_tol"}
            ),
        )
        return runner.run()


@dataclass
class FilteringRunResult:
    """Hold the independently produced results of the three filter stages."""

    sample_result: StageResult[MetaboDataset] | None = None
    feature_result: StageResult[MetaboDataset] | None = None
    quality_result: StageResult[MetaboDataset] | None = None

    @property
    def missingness_metadata(self) -> pd.DataFrame | None:
        """Return feature labels for downstream quality filtering."""
        if self.feature_result is None:
            return None
        return self.feature_result.data.feature_metadata.copy(deep=True)

    @property
    def missingness_tracking(self) -> pd.DataFrame | None:
        """Return the complete Feature MV audit for retention reporting."""
        if self.feature_result is None:
            return None
        audit = self.feature_result.require_audit(
            MissingValueFilterAuditPayload
        )
        return audit.feature_tracking.copy(deep=True)


class FilteringOrchestrator:
    """Compose the three filter classes without sharing mutable stage state."""

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the filter with validated stage-local settings."""
        self.data = data
        self.pipeline_params = dict(pipeline_params or {})
        self.result: FilteringRunResult | None = None

    def run_missingness(
        self,
        *,
        output_dir: str | Path | None = None,
        sample_overrides: Mapping[str, object] | None = None,
        feature_overrides: Mapping[str, object] | None = None,
    ) -> FilteringRunResult:
        """Run both missingness actions in one artifact directory.

        The actions retain independent ``StageResult`` and Audit boundaries,
        while their tables and combined dashboard are exported as one
        user-facing missingness-filtering stage.
        """
        sample_filter = SampleMissingValueFilter(
            self.data,
            self.pipeline_params,
        )
        sample_result = sample_filter.run_filter_samples_by_missingness(
            output_dir=output_dir,
            **dict(sample_overrides or {}),
        )
        feature_filter = FeatureMissingValueFilter(
            sample_result.data,
            self.pipeline_params,
            sample_result=sample_result,
        )
        feature_result = feature_filter.run_filter_features_by_missingness(
            output_dir=output_dir,
            **dict(feature_overrides or {}),
        )
        self.result = FilteringRunResult(sample_result, feature_result)
        return self.result

    def run_quality(
        self,
        data: MetaboDataset | None = None,
        *,
        output_dir: str | Path | None = None,
        quality_overrides: Mapping[str, object] | None = None,
    ) -> FilteringRunResult:
        """Run quality filtering on a supplied table.

        The quality action is independent: if no missingness result has been
        produced by this orchestrator, it runs directly on ``data`` (or the
        orchestrator's source dataset) in the core's explicit
        ``quality_only`` mode.  A prior ``run_missingness`` result is used
        only when it is explicitly present.
        """
        if self.result is None:
            quality_data = data if data is not None else self.data
            quality_filter = FeatureQualityFilter(
                quality_data,
                self.pipeline_params,
            )
            quality_result = quality_filter.run_filter_features_by_quality(
                output_dir=output_dir,
                **dict(quality_overrides or {}),
            )
            self.result = FilteringRunResult(
                quality_result=quality_result,
            )
            return self.result

        quality_data = (
            data
            if data is not None
            else (
                self.result.feature_result.data
                if self.result.feature_result is not None
                else self.data
            )
        )
        quality_filter = FeatureQualityFilter(
            quality_data,
            self.pipeline_params,
            missingness_metadata=self.result.missingness_tracking,
        )
        quality_result = quality_filter.run_filter_features_by_quality(
            output_dir=output_dir,
            missingness_metadata=self.result.missingness_tracking,
            **dict(quality_overrides or {}),
        )
        self.result.quality_result = quality_result
        return self.result

    def run(
        self,
        *,
        quality_data: MetaboDataset | None = None,
        missingness_output_dir: str | Path | None = None,
        quality_output_dir: str | Path | None = None,
        sample_overrides: Mapping[str, object] | None = None,
        feature_overrides: Mapping[str, object] | None = None,
        quality_overrides: Mapping[str, object] | None = None,
    ) -> FilteringRunResult:
        """Run all three filters, optionally using post-correction data."""
        self.run_missingness(
            output_dir=missingness_output_dir,
            sample_overrides=sample_overrides,
            feature_overrides=feature_overrides,
        )
        return self.run_quality(
            quality_data,
            output_dir=quality_output_dir,
            quality_overrides=quality_overrides,
        )


__all__ = [
    "FeatureMissingValueFilter",
    "FeatureQualityFilter",
    "FilteringOrchestrator",
    "FilteringRunResult",
    "SampleMissingValueFilter",
]
