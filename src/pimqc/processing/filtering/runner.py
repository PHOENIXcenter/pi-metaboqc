"""Export and visualize completed sample and feature filtering results.

The runners keep high-missing sample removal, MAR/MNAR classification, and
low-quality feature rules inside ``FeatureFilter`` while moving CSV output and
dashboard construction into explicit post-transformation lifecycle phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import pandas as pd
from loguru import logger

from ..audit import (
    MissingValueFilterAuditPayload,
    QualityFilterAuditPayload,
    SampleFilterAuditPayload,
)
from ..stage import StageResult, StageRunner
from ...plotting.filtering import FilteringPlotter
from ...core import MetaboDataset

if TYPE_CHECKING:
    from .analysis import FeatureFilter


class SampleFilteringStageRunner(StageRunner["FeatureFilter", MetaboDataset]):
    """Run and export the sample-level high-missingness filter."""

    def compute(self) -> StageResult[MetaboDataset]:
        """Remove high-missingness analytical samples."""
        return self.processor.filter_samples_by_missingness()

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Write the retained sample matrix and its attrition table."""
        assert self.output_dir is not None
        result.data.annotated_frame().to_csv(
            self.output_dir / "Filtered_Data_High-MV_Samples.csv"
        )
        audit = result.require_audit(SampleFilterAuditPayload)
        audit.sample_tracking.to_csv(
            self.output_dir / "Filtering_Tracking_High-MV_Samples.csv"
        )

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Keep the sample action tabular; its panel belongs to Feature MV.

        A sample missingness action remains independently executable and
        produces a typed audit plus CSV artifacts. Its diagnostic is not an
        independent one-panel dashboard: when a Feature MV action receives
        this result explicitly, that action composes the sample panel into
        its multi-step filtering dashboard.
        """
        result.require_audit(SampleFilterAuditPayload)


class MissingValueFilteringStageRunner(
    StageRunner["FeatureFilter", MetaboDataset]
):
    """Run missingness classification and its feature-retention dashboard."""

    def compute(self) -> StageResult[MetaboDataset]:
        """Classify features and retain accepted MAR or MNAR features."""
        return self.processor.filter_missing_values()

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Write sample and feature filtering matrices and tracking tables."""
        assert self.output_dir is not None
        audit = result.require_audit(MissingValueFilterAuditPayload)
        # Preserve both attrition levels because the feature stage is computed
        # from the already sample-filtered matrix.
        audit.sample_filtered_data.annotated_frame().to_csv(
            self.output_dir / "Filtered_Data_High-MV_Samples.csv"
        )
        audit.sample_tracking.to_csv(
            self.output_dir / "Filtering_Tracking_High-MV_Samples.csv"
        )
        result.data.annotated_frame().to_csv(
            self.output_dir / "Filtered_Data_High-MV_Features.csv"
        )
        audit.feature_tracking.to_csv(
            self.output_dir / "Filtering_Tracking_High-MV_Features.csv"
        )

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Render the MAR/MNAR classification and retention dashboard."""
        assert self.output_dir is not None
        audit = result.require_audit(MissingValueFilterAuditPayload)
        payload = audit.plot_payload
        plotter = FilteringPlotter(payload)
        dashboard = plotter.plot_mv_filtering_dashboard(
            tracking_df=audit.feature_tracking,
            active_base_tol=payload.active_base_tolerance,
            mnar_group_mv_tol=payload.mnar_group_mv_tolerance,
            mnar_qc_mv_tol=payload.mnar_qc_mv_tolerance,
            mnar_int_threshold=payload.mnar_intensity_threshold,
            mnar_intensity_pct=payload.mnar_intensity_percentile,
        )
        if dashboard:
            path = self.output_dir / "MV_Classification_Dashboard.svg"
            plotter.save_and_show_pw(
                pw_obj=dashboard,
                file_path=str(path),
            )
            logger.info(f"High-MV Filter summary dashboard saved as: {path}")
        logger.success("High-missing value feature filtering completed.")


class FeatureMissingValueFilteringStageRunner(
    StageRunner["FeatureMissingValueFilter", MetaboDataset]
):
    """Run feature-level missingness filtering on a prepared table."""

    def compute(self) -> StageResult[MetaboDataset]:
        """Classify and filter features without re-running sample filtering."""
        return self.processor.filter_features_by_missingness()

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Write feature filtering outputs and tracking tables."""
        assert self.output_dir is not None
        audit = result.require_audit(MissingValueFilterAuditPayload)
        audit.feature_tracking.to_csv(
            self.output_dir / "Filtering_Tracking_MV_Features.csv"
        )
        result.data.annotated_frame().to_csv(
            self.output_dir / "Filtered_Data_MV_Features.csv"
        )

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Render the feature-level missingness dashboard."""
        assert self.output_dir is not None
        audit = result.require_audit(MissingValueFilterAuditPayload)
        payload = audit.plot_payload
        plotter = FilteringPlotter(payload)
        dashboard = plotter.plot_mv_filtering_dashboard(
            tracking_df=audit.feature_tracking,
            active_base_tol=payload.active_base_tolerance,
            mnar_group_mv_tol=payload.mnar_group_mv_tolerance,
            mnar_qc_mv_tol=payload.mnar_qc_mv_tolerance,
            mnar_int_threshold=payload.mnar_intensity_threshold,
            mnar_intensity_pct=payload.mnar_intensity_percentile,
        )
        if dashboard:
            path = self.output_dir / "MV_Classification_Dashboard.svg"
            plotter.save_and_show_pw(pw_obj=dashboard, file_path=str(path))
            logger.info(f"MV classification dashboard saved as: {path}")
        logger.success("Feature missing-value filtering completed.")


class QualityFilteringStageRunner(StageRunner["FeatureFilter", MetaboDataset]):
    """Run blank-ratio and QC-RSD filtering and render its dashboard."""

    def __init__(
        self,
        processor: "FeatureFilter",
        output_dir: str | None,
        *,
        idx_mar: pd.Index | list[object] | None,
        idx_mnar: pd.Index | list[object] | None,
        missingness_tracking: pd.DataFrame | None = None,
        runtime_overrides: Mapping[str, object] | None = None,
        allowed_override_keys: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Initialize the quality-filtering lifecycle.

        Args:
            processor: Filtering processor containing the source matrix.
            output_dir: Optional directory for tables and dashboards.
            idx_mar: Optional MAR feature identifiers.
            idx_mnar: Optional MNAR feature identifiers.
            missingness_tracking: Optional complete Feature MV audit table used
                to render explicit cross-stage retention history.
            runtime_overrides: Named threshold overrides for this execution.
            allowed_override_keys: Permitted runtime configuration names.
        """
        super().__init__(
            processor,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=allowed_override_keys,
        )
        self.idx_mar = idx_mar
        self.idx_mnar = idx_mnar
        self.missingness_tracking = (
            None
            if missingness_tracking is None
            else missingness_tracking.copy(deep=True)
        )

    def compute(self) -> StageResult[MetaboDataset]:
        """Apply the low-quality feature rules."""
        return self.processor.filter_features_by_quality(
            idx_mar=self.idx_mar,
            idx_mnar=self.idx_mnar,
            missingness_tracking=self.missingness_tracking,
        )

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Write the retained matrix and feature-level audit table."""
        assert self.output_dir is not None
        audit = result.require_audit(QualityFilterAuditPayload)
        path = self.output_dir / "Filtered_Data_Low-quality_Features.csv"
        result.data.annotated_frame().to_csv(
            path, encoding="utf-8-sig", na_rep="NA"
        )
        audit.feature_tracking.to_csv(
            self.output_dir / "Filtering_Tracking_Low-quality_Features.csv",
            na_rep="N/A",
        )
        logger.info(
            f"Data after low-quality features filtering saved as: {path}"
        )

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Render the low-quality filtering dashboard."""
        assert self.output_dir is not None
        audit = result.require_audit(QualityFilterAuditPayload)
        plotter = FilteringPlotter(audit.plot_payload)
        # Rendering failures should not invalidate an already completed and
        # exported filtering transformation.
        try:
            dashboard = plotter.plot_quality_filtering_dashboard()
            if dashboard:
                path = self.output_dir / "Low-quality_Filtering_Dashboard.svg"
                plotter.save_and_show_pw(
                    pw_obj=dashboard,
                    file_path=str(path),
                )
                logger.info(
                    f"Low-quality Filter summary dashboard saved as: {path}"
                )
        except Exception as error:
            logger.error(
                "Grid of low-quality features filtering generation failed: "
                f"{error}"
            )
        logger.success("Low-quality features filtering completed.")
