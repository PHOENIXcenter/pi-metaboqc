"""Missing-value triage and low-quality feature filtering calculations.

FeatureFilter removes high-missing samples, classifies features as MAR, MNAR,
or invalid, applies biological-group and QC rescue rules, then filters features
by blank-to-QC abundance and QC RSD. It stores retained indices and tracking
tables required by imputation, assessment, and reporting.
"""

import copy
import numpy as np
import pandas as pd
from functools import cached_property


from loguru import logger
from typing import Dict, Any, Optional

from ...runtime import log_execution_time
from ...core import DatasetProcessor, MetaboDataset
from ...config import resolve_stage_config
from ...plotting.payloads import FilteringPlotPayload, snapshot_dataset
from ..audit import (
    MissingValueFilterAuditPayload,
    QualityFilterAuditPayload,
    SampleFilterAuditPayload,
)
from ..stage import StageResult
from .runner import (
    MissingValueFilteringStageRunner,
    QualityFilteringStageRunner,
    SampleFilteringStageRunner,
)


class FeatureFilter(DatasetProcessor):
    """Filtering engine for metabolomics datasets with QC enforcement."""

    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "sample_mv_tol",
            "mv_global_tol",
            "mv_group_tol",
            "mv_qc_tol",
            "mnar_group_mv_tol",
            "mnar_qc_mv_tol",
            "mnar_intensity_pct",
            "qc_rsd_tol",
            "blank_qc_ratio_tol",
        }
    )

    _INVALID_STRS = {
        "unknown",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "",
        "unassigned",
        "blank",
        "blk",
        "is",
        "solvent",
        "wash",
        "sst",
        "pool",
        "invalid",
        "unvalid",
    }

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Optional[Dict[str, Any]] = None,
        sample_mv_tol: Optional[float] = None,
        mv_global_tol: Optional[float] = None,
        mv_group_tol: Optional[float] = None,
        mv_qc_tol: Optional[float] = None,
        mnar_group_mv_tol: Optional[float] = None,
        mnar_qc_mv_tol: Optional[float] = None,
        mnar_intensity_pct: Optional[float] = None,
        qc_rsd_tol: Optional[float] = None,
        blank_qc_ratio_tol: Optional[float] = None,
    ) -> None:
        """Initialize the filtering engine with QC enforcement.

        Args:
            data: Explicit dataset to filter.
            pipeline_params: Global configuration dictionary from TOML.
            sample_mv_tol: Max missing rate for sample removal.
            mv_global_tol: Max missing rate across all retained samples.
            mv_group_tol: Base missing rate tolerance within bio groups.
            mv_qc_tol: Base missing rate tolerance in QC samples.
            mnar_group_mv_tol: Max missing rate for group MNAR rescue.
            mnar_qc_mv_tol: Max missing rate for QC MNAR rescue.
            mnar_intensity_pct: Intensity percentile threshold for MNAR QC.
            qc_rsd_tol: Max relative standard deviation in QC.
            blank_qc_ratio_tol: Max allowable blank to QC intensity ratio.
        """
        super().__init__(data)
        self.stats["feature_counts"] = {}

        filter_configs = resolve_stage_config(
            pipeline_params,
            "FeatureFilter",
            {
                "sample_mv_tol": 0.5,
                "mv_global_tol": 0.7,
                "mv_group_tol": 0.5,
                "mv_qc_tol": 0.3,
                "mnar_group_mv_tol": 0.8,
                "mnar_qc_mv_tol": 0.2,
                "mnar_intensity_pct": 0.1,
                "qc_rsd_tol": 0.3,
                "blank_qc_ratio_tol": 0.2,
            },
            {
                "sample_mv_tol": sample_mv_tol,
                "mv_global_tol": mv_global_tol,
                "mv_group_tol": mv_group_tol,
                "mv_qc_tol": mv_qc_tol,
                "mnar_group_mv_tol": mnar_group_mv_tol,
                "mnar_qc_mv_tol": mnar_qc_mv_tol,
                "mnar_intensity_pct": mnar_intensity_pct,
                "qc_rsd_tol": qc_rsd_tol,
                "blank_qc_ratio_tol": blank_qc_ratio_tol,
            },
        )
        self.config.update(filter_configs)
        self.stats.update(
            {
                "mv_group_df": pd.DataFrame(),
                "mv_qc_series": pd.Series(dtype=float),
                "mv_global_series": pd.Series(dtype=float),
                "blank_mean": pd.Series(dtype=float),
                "qc_mean": pd.Series(dtype=float),
                "qc_rsd_all": pd.Series(dtype=float),
                "idx_mar": pd.Index([]),
                "idx_mnar": pd.Index([]),
                "idx_mnar_group": pd.Index([]),
                "idx_mnar_qc": pd.Index([]),
            }
        )

    def _audit_snapshot(self) -> dict[str, Any]:
        """Copy runtime audit values into the explicit stage result."""
        return copy.deepcopy(self.stats)

    # =========================================================================
    # Sample-Level Filtering
    # =========================================================================
    def filter_samples_by_missingness(self) -> StageResult[MetaboDataset]:
        """Filter high-missingness samples and retain their audit table."""
        batch = self.config.get("batch", "Batch")
        inject_order = self.config.get("inject_order", "Inject Order")
        sample_mv_tol = self.config.get("sample_mv_tol", 0.5)

        # Strictly evaluate only QC and Actual Samples via concatenation
        df_check = pd.concat([self.qc_data, self.actual_data], axis=1)
        sample_mv_rates = df_check.isna().mean(axis=0)
        missing_value_count = int(df_check.isna().sum().sum())
        execution_status = (
            "skipped" if missing_value_count == 0 else "completed"
        )
        skip_reason = (
            "No missing values detected in QC or actual samples."
            if execution_status == "skipped"
            else None
        )

        # Determine status
        bad_mask = sample_mv_rates > sample_mv_tol
        bad_samples = sample_mv_rates[bad_mask].index

        # Safe drop: Removes bad samples while retaining Blanks/Others intact
        retained_samples = self.frame.columns.difference(bad_samples)

        # Extract sample types for tracking
        sample_type = self.config.get("sample_type", "Sample Type")
        check_types = df_check.columns.get_level_values(sample_type)

        # Build tracking table for diagnostics
        track_df = pd.DataFrame(
            {
                "Sample_ID": sample_mv_rates.index.get_level_values(
                    self.dataset.schema.sample_id
                ),
                "Sample_Type": check_types,
                "MV_Rate_Pct": sample_mv_rates.values * 100,
                "Status": np.where(bad_mask, "Dropped", "Retained"),
            }
        ).set_index("Sample_ID")

        self.stats["sample_tracking"] = track_df
        self.stats["sample_dropped_idx"] = bad_samples

        if not bad_samples.empty:
            logger.warning(
                f"Dropping {len(bad_samples)} samples (MV > {sample_mv_tol})"
            )

        df_filtered = self.frame.loc[:, retained_samples].copy()

        # Physically reorder the remaining columns by injection sequence
        sort_levels = [
            lvl
            for lvl in [batch, inject_order]
            if lvl in df_filtered.columns.names
        ]
        if sort_levels:
            df_filtered = df_filtered.sort_index(axis=1, level=sort_levels)

        filtered_dataset = self._to_dataset(df_filtered)
        plot_payload = FilteringPlotPayload(
            data=snapshot_dataset(self.dataset),
            audit_tables={"sample_tracking": track_df.copy(deep=True)},
            sample_mv_tolerance=sample_mv_tol,
            stage_status=execution_status,
            sample_filter_status=execution_status,
            missing_values_detected=missing_value_count > 0,
            biological_groups_available=bool(self._get_valid_bio_groups()),
            skip_reason=skip_reason,
        )
        return StageResult(
            data=filtered_dataset,
            audit=SampleFilterAuditPayload(
                sample_tracking=track_df,
                dropped_sample_ids=bad_samples,
                plot_payload=plot_payload,
                execution_status=execution_status,
                skip_reason=skip_reason,
                missing_value_count=missing_value_count,
                input_sample_count=self.frame.shape[1],
                output_sample_count=df_filtered.shape[1],
            ),
        )

    @log_execution_time
    def run_sample_filtering(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Return the structured sample-filtering stage result.

        Named overrides take precedence over pipeline configuration and module
        defaults. This stage accepts ``sample_mv_tol``.
        """
        runner = SampleFilteringStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset({"sample_mv_tol"}),
        )
        return runner.run()

    # =========================================================================
    # Feature-Level Missing Value Classification
    # =========================================================================

    def _get_valid_bio_groups(self) -> list[object]:
        """Extract valid biological group names from the column index."""
        bio_group = self.config.get("bio_group", "Bio Group")
        sample_dict = self.config.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")

        valid_bio_groups = []
        if bio_group in self.frame.columns.names:
            raw_groups = self.frame.columns.get_level_values(bio_group).unique()
            for group in raw_groups:
                if pd.isna(group):
                    continue
                group_str = str(group).strip().lower()
                if (
                    group_str in self._INVALID_STRS
                    or group_str == str(qc_label).lower()
                ):
                    continue
                valid_bio_groups.append(group)
        return valid_bio_groups

    def classify_missing_types(self) -> tuple[pd.Index, pd.Index, pd.Index]:
        """Classifies features with strict QC enforcement and dynamic tol."""
        total_missing = self.frame.isna().sum().sum()
        empty_idx = self.frame.index[:0]
        if total_missing == 0:
            logger.info(
                "No missing values detected. Missingness classification is "
                "not required."
            )
            self.stats.update(
                {
                    "idx_mar": empty_idx,
                    "idx_mnar": empty_idx,
                    "idx_mnar_group": empty_idx,
                    "idx_mnar_qc": empty_idx,
                    "idx_dropped_stage1": empty_idx,
                    "feature_filter_status": "skipped",
                    "missing_value_count": 0,
                }
            )
            return empty_idx, empty_idx, empty_idx

        try:
            bio_group = self.config.get("bio_group", "Bio Group")
            sample_type = self.config.get("sample_type", "Sample Type")
            sample_dict = self.config.get("sample_dict", {})
            qc_label = sample_dict.get("QC sample", "QC")

            mv_group_tol = self.config.get("mv_group_tol", 0.5)
            mv_qc_tol = self.config.get("mv_qc_tol", 0.3)
            mv_global_tol = self.config.get("mv_global_tol", 0.7)
            mnar_group_mv_tol = self.config.get("mnar_group_mv_tol", 0.8)
            mnar_qc_mv_tol = self.config.get("mnar_qc_mv_tol", 0.2)
            mnar_intensity_pct = self.config.get("mnar_intensity_pct", 0.1)

            qc_mask = (
                self.frame.columns.get_level_values(sample_type) == qc_label
                if sample_type in self.frame.columns.names
                else np.zeros(self.frame.shape[1], dtype=bool)
            )

            if not qc_mask.any():
                raise ValueError("Fatal: QC samples are required.")

            idx_mnar_group = pd.Index([])
            valid_bio_groups = self._get_valid_bio_groups()
            global_mv_rate = self.stats.get("mv_global_series")
            if (
                not isinstance(global_mv_rate, pd.Series)
                or global_mv_rate.empty
            ):
                global_mv_rate = self.frame.isna().mean(axis=1)
            cond_global_valid = global_mv_rate <= mv_global_tol

            # Group Rescue (Tier 1 - Optimal)
            if valid_bio_groups:
                na_rate_group = (
                    self.frame.isna().T.groupby(level=bio_group).mean().T
                )
                na_rate_valid = na_rate_group[valid_bio_groups]
                cond_group_mnar = (na_rate_valid >= mnar_group_mv_tol).any(
                    axis=1
                ) & (na_rate_valid <= mv_group_tol).any(axis=1)
                cond_group_mnar &= cond_global_valid
                idx_mnar_group = self.frame.index[cond_group_mnar]

            # QC Rescue (Enforced QC standard)
            df_qc = self.frame.loc[:, qc_mask]
            qc_na_rate = df_qc.isna().mean(axis=1)
            qc_median = df_qc.median(axis=1)
            int_threshold = qc_median.quantile(mnar_intensity_pct)

            cond_qc_mv = qc_na_rate > mnar_qc_mv_tol
            cond_qc_int = qc_median <= int_threshold

            if valid_bio_groups:
                cond_qc_bio_valid = (na_rate_valid <= mv_group_tol).any(axis=1)
                idx_mnar_qc = self.frame.index[
                    cond_qc_mv
                    & cond_qc_int
                    & cond_qc_bio_valid
                    & cond_global_valid
                ]
            else:
                idx_mnar_qc = self.frame.index[
                    cond_qc_mv & cond_qc_int & cond_global_valid
                ]
                logger.warning("No Bio Groups. Falling back to QC rescue.")

            idx_mnar_all = idx_mnar_group.union(idx_mnar_qc)

            # Base Health (Tiered Degradation)
            if valid_bio_groups:
                cond_healthy = (na_rate_valid <= mv_group_tol).any(axis=1)
            else:
                cond_healthy = self.stats["mv_qc_series"] <= mv_qc_tol
            cond_healthy &= cond_global_valid

            idx_healthy = self.frame.index[cond_healthy]
            idx_mar = idx_healthy.difference(idx_mnar_all)
            idx_dropped = self.frame.index.difference(
                idx_mar.union(idx_mnar_all)
            )

            self.stats.update(
                {
                    "idx_mar": idx_mar,
                    "idx_mnar": idx_mnar_all,
                    "idx_mnar_group": idx_mnar_group,
                    "idx_mnar_qc": idx_mnar_qc,
                    "idx_dropped_stage1": idx_dropped,
                }
            )
            return idx_mar, idx_mnar_all, idx_dropped

        except Exception as exc:
            # A missing QC cohort is a violated stage precondition, not an
            # empty classification result. Returning empty indices here would
            # silently drop every feature downstream and make the failure
            # appear to be a valid, but empty, filtering outcome.
            raise ValueError(
                "Missing-value classification failed. Verify that the input "
                "contains QC samples and complete sample metadata."
            ) from exc

    # =========================================================================
    # Filtering Execution Flow
    # =========================================================================

    def filter_missing_values(self) -> StageResult[MetaboDataset]:
        """Run sample and feature-level missingness filtering in sequence.

        This method remains as a convenience facade for the native Python API.
        The independently runnable :class:`FeatureMissingValueFilter` calls
        :meth:`filter_features_by_missingness` directly after receiving an
        explicit sample-filtered dataset.
        """
        sample_result = self.filter_samples_by_missingness()
        return self.filter_features_by_missingness(sample_result=sample_result)

    def filter_features_by_missingness(
        self,
        *,
        sample_result: StageResult[MetaboDataset] | None = None,
    ) -> StageResult[MetaboDataset]:
        """Classify missingness on an already sample-filtered dataset.

        ``sample_result`` is optional so the feature-level class can also be
        used independently on a table that has already undergone sample
        filtering.  When supplied, the sample audit is copied into the feature
        audit explicitly; no processor-local state is shared across stages.
        """
        if sample_result is None:
            sample_filtered_dataset = self.dataset
            sample_tracking = pd.DataFrame()
            sample_mv_tolerance = self.config.get("sample_mv_tol", 0.5)
            sample_filter_status = "not_run"
        else:
            sample_filtered_dataset = sample_result.data
            sample_audit = sample_result.require_audit(SampleFilterAuditPayload)
            sample_tracking = sample_audit.sample_tracking
            sample_filter_status = sample_audit.execution_status
            sample_mv_tolerance = (
                sample_audit.plot_payload.sample_mv_tolerance
                if sample_audit.plot_payload is not None
                else self.config.get("sample_mv_tol", 0.5)
            )
        self._replace_frame(sample_filtered_dataset.annotated_frame())
        self.stats["sample_tracking"] = sample_tracking.copy(deep=True)
        self.stats["sample_mv_tolerance"] = sample_mv_tolerance

        feature_counts = self.stats["feature_counts"]
        if "raw" not in feature_counts:
            feature_counts["raw"] = self.frame.shape[0]

        sample_type = self.config.get("sample_type", "Sample Type")
        sample_dict = self.config.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        valid_groups = self._get_valid_bio_groups()

        qc_mask = (
            self.frame.columns.get_level_values(sample_type) == qc_label
            if sample_type in self.frame.columns.names
            else np.zeros(self.frame.shape[1], dtype=bool)
        )

        if qc_mask.any():
            self.stats["mv_qc_series"] = (
                self.frame.loc[:, qc_mask].isna().mean(axis=1)
            )
        self.stats["mv_global_series"] = self.frame.isna().mean(axis=1)

        if valid_groups:
            bio_group = self.config.get("bio_group", "Bio Group")
            group_na = self.frame.isna().T.groupby(level=bio_group).mean().T
            self.stats["mv_group_df"] = group_na[valid_groups]

        missing_value_count = int(self.frame.isna().sum().sum())
        missing_values_detected = missing_value_count > 0
        feature_filter_status = (
            "completed" if missing_values_detected else "skipped"
        )
        skip_reason = (
            "No missing values detected in the Feature MV input."
            if feature_filter_status == "skipped"
            else None
        )
        self.stats.update(
            {
                "feature_filter_status": feature_filter_status,
                "sample_filter_status": sample_filter_status,
                "missing_value_count": missing_value_count,
                "missing_values_detected": missing_values_detected,
                "biological_groups_available": bool(valid_groups),
                "feature_filter_skip_reason": skip_reason,
            }
        )

        if missing_values_detected:
            idx_mar, idx_mnar, idx_dropped = self.classify_missing_types()
            retained_idx = idx_mar.union(idx_mnar)
            df_final = self.frame.loc[retained_idx].copy(deep=True)

            idx_mnar_group = self.stats.get("idx_mnar_group", pd.Index([]))
            idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))
            feature_metadata = sample_filtered_dataset.feature_metadata.loc[
                list(retained_idx)
            ].copy()
            feature_metadata["missingness_type"] = "MAR"
            feature_metadata.loc[idx_mnar, "missingness_type"] = "MNAR"
            feature_metadata["mnar_group_rescued"] = (
                feature_metadata.index.isin(idx_mnar_group)
            )
            feature_metadata["mnar_qc_rescued"] = feature_metadata.index.isin(
                idx_mnar_qc
            )
        else:
            empty_idx = self.frame.index[:0]
            idx_mar = empty_idx
            idx_mnar = empty_idx
            idx_dropped = empty_idx
            retained_idx = self.frame.index
            df_final = self.frame.copy(deep=True)
            feature_metadata = sample_filtered_dataset.feature_metadata.loc[
                list(retained_idx)
            ].copy()
            self.stats.update(
                {
                    "idx_mar": empty_idx,
                    "idx_mnar": empty_idx,
                    "idx_mnar_group": empty_idx,
                    "idx_mnar_qc": empty_idx,
                    "idx_dropped_stage1": empty_idx,
                }
            )

        feature_counts["post_stage1"] = len(retained_idx)
        df_tracking = self._generate_s1_tracking_table(
            qc_mask,
            idx_mar,
            idx_mnar,
            idx_dropped,
            skipped=not missing_values_detected,
        )
        self.stats["stage1_tracking"] = df_tracking

        audit_tables = self._audit_snapshot()
        # The feature action receives the sample result explicitly. Preserve
        # that audit in its plotting snapshot so the combined missingness
        # dashboard never depends on state inherited from another processor.
        audit_tables["sample_tracking"] = sample_tracking.copy(deep=True)
        intensity_percentile = self.config.get("mnar_intensity_pct", 0.1)
        intensity_threshold = None
        if missing_values_detected and qc_mask.any():
            raw_threshold = (
                self.frame.loc[:, qc_mask]
                .median(axis=1)
                .quantile(intensity_percentile)
            )
            intensity_threshold = np.log2(raw_threshold + 1)
        active_tolerance = self.config.get(
            "mv_group_tol" if valid_groups else "mv_qc_tol",
            0.5,
        )
        result_dataset = self._to_dataset(
            df_final,
            context_updates={"pipeline_stage": "High-missing values filtering"},
            feature_metadata=feature_metadata,
        )
        plot_dataset = self._to_dataset(self.frame)
        return StageResult(
            data=result_dataset,
            audit=MissingValueFilterAuditPayload(
                metric_values=self.mv_filtering_metrics,
                feature_tracking=df_tracking,
                sample_filtered_data=sample_filtered_dataset,
                sample_tracking=sample_tracking,
                qc_mask=qc_mask,
                valid_biological_groups=valid_groups,
                plot_payload=FilteringPlotPayload(
                    data=snapshot_dataset(plot_dataset),
                    audit_tables=audit_tables,
                    sample_mv_tolerance=sample_mv_tolerance,
                    active_base_tolerance=active_tolerance,
                    mnar_group_mv_tolerance=self.config.get(
                        "mnar_group_mv_tol", 0.8
                    ),
                    mnar_qc_mv_tolerance=self.config.get("mnar_qc_mv_tol", 0.2),
                    mnar_intensity_threshold=intensity_threshold,
                    mnar_intensity_percentile=intensity_percentile,
                    stage_status=feature_filter_status,
                    sample_filter_status=sample_filter_status,
                    missing_values_detected=missing_values_detected,
                    biological_groups_available=bool(valid_groups),
                    skip_reason=skip_reason,
                ),
                tables=audit_tables,
                execution_status=feature_filter_status,
                skip_reason=skip_reason,
                missing_value_count=missing_value_count,
            ),
        )

    @log_execution_time
    def run_mv_filtering(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Return the structured missingness-filtering stage result.

        Supported settings are ``sample_mv_tol``, ``mv_global_tol``,
        ``mv_group_tol``, ``mv_qc_tol``, ``mnar_group_mv_tol``,
        ``mnar_qc_mv_tol``, and ``mnar_intensity_pct``. They override pipeline
        configuration only for the current processor instance.
        """
        runner = MissingValueFilteringStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS.difference(
                {"qc_rsd_tol", "blank_qc_ratio_tol"}
            ),
        )
        return runner.run()

    def _generate_s1_tracking_table(
        self,
        qc_mask: np.ndarray,
        idx_mar: pd.Index,
        idx_mnar: pd.Index,
        idx_dropped: pd.Index,
        *,
        skipped: bool = False,
    ) -> pd.DataFrame:
        """Builds a detailed feature status tracking DataFrame."""
        global_mv = self.stats.get("mv_global_series", pd.Series(dtype=float))
        qc_mv = self.stats.get("mv_qc_series", pd.Series(dtype=float))
        group_df = self.stats.get("mv_group_df", pd.DataFrame())

        idx_mnar_group = self.stats.get("idx_mnar_group", pd.Index([]))
        idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))

        if qc_mask.any():
            qc_median = self.frame.loc[:, qc_mask].median(axis=1)
        else:
            qc_median = pd.Series(dtype=float)

        track_data = []
        for feat in self.frame.index:
            val_global = global_mv.get(feat, np.nan) * 100
            val_qc = qc_mv.get(feat, np.nan) * 100

            val_group_max = np.nan
            val_group_min = np.nan

            if not group_df.empty and feat in group_df.index:
                val_group_max = group_df.loc[feat].max() * 100
                val_group_min = group_df.loc[feat].min() * 100

            val_intensity = qc_median.get(feat, np.nan)
            log2_int = (
                np.log2(val_intensity + 1)
                if pd.notna(val_intensity)
                else np.nan
            )

            # Use exact string matching for sample-type labels.
            # Downstream visualization relies heavily on .str.contains("Group")
            # and .str.contains("QC"). Abbreviations break the routing logic!
            if skipped:
                status, reason, sort_order = (
                    "NOT_APPLICABLE",
                    "No missing values detected",
                    0,
                )
            elif feat in idx_dropped:
                status, reason, sort_order = "INVALID", "Fail MV rules", 0
            elif feat in idx_mar:
                status, reason, sort_order = "MAR", "Health passed", 1
            elif feat in idx_mnar_group and feat in idx_mnar_qc:
                status, reason, sort_order = (
                    "MNAR (Group & QC)",
                    "Dual rescue",
                    2,
                )
            elif feat in idx_mnar_group:
                status, reason, sort_order = "MNAR (Group)", "Group pass", 3
            elif feat in idx_mnar_qc:
                status, reason, sort_order = "MNAR (QC)", "QC pass", 4
            else:
                status, reason, sort_order = "Unknown", "Logic gap", 5

            track_data.append(
                {
                    "Feature_ID": feat,
                    "Global_MV_Pct": round(val_global, 2),
                    "QC_MV_Pct": round(val_qc, 2),
                    "Min_Group_MV_Pct": round(val_group_min, 2),
                    "Max_Group_MV_Pct": round(val_group_max, 2),
                    "Log2_Intensity": round(log2_int, 4),
                    "Stage1_Status": status,
                    "Reference_Basis": reason,
                    "_sort": sort_order,
                }
            )

        df_tracking = pd.DataFrame(track_data).set_index("Feature_ID")
        return df_tracking.sort_values(by="_sort").drop(columns=["_sort"])

    @cached_property
    def mv_filtering_metrics(self) -> Dict[str, Any]:
        """
        Extracts metrics from Stage-1 missing value filtering.
        Unifies results into 'sample_wise' and 'feature_wise' dimensions.
        """
        feature_counts = self.stats.get("feature_counts", {})

        # =====================================================================
        # Sample-wise Metrics Extraction
        # =====================================================================
        track_df = self.stats.get("sample_tracking", pd.DataFrame())
        sample_metrics = {}
        if not track_df.empty:
            sample_total = len(track_df)
            sample_dropped = sum(track_df["Status"] == "Dropped")
            sample_retained = sample_total - sample_dropped
            sample_retention_rate = (
                round(sample_retained / sample_total * 100, 2)
                if sample_total
                else 0.0
            )

            sample_metrics = {
                "thresholds": {
                    "sample_mv_tol": self.stats.get(
                        "sample_mv_tolerance",
                        self.config.get("sample_mv_tol", 0.5),
                    )
                },
                "feature_retention": {  # Kept naming style consistent
                    "total_checked": sample_total,
                    "retained_count": sample_retained,
                    "dropped_count": sample_dropped,
                    "retention_rate_pct": sample_retention_rate,
                },
            }

        # =====================================================================
        # Feature-wise Metrics Extraction
        # =====================================================================
        valid_groups = self._get_valid_bio_groups()
        sample_type = self.config.get("sample_type", "Sample Type")
        sample_dict = self.config.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")

        has_qc = False
        if sample_type in self.frame.columns.names:
            has_qc = (
                self.frame.columns.get_level_values(sample_type) == qc_label
            ).any()

        if valid_groups:
            filter_level = "Group"
        elif has_qc:
            filter_level = "QC"
        else:
            filter_level = "Global"

        idx_mar = self.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.stats.get("idx_mnar", pd.Index([]))
        idx_mnar_group = self.stats.get("idx_mnar_group", pd.Index([]))
        idx_mnar_qc = self.stats.get("idx_mnar_qc", pd.Index([]))

        execution_status = self.stats.get("feature_filter_status", "completed")
        skip_reason = self.stats.get("feature_filter_skip_reason")
        missing_value_count = int(self.stats.get("missing_value_count", 0))
        retained_idx = idx_mar.union(idx_mnar)

        feature_raw_count = int(feature_counts.get("raw", 0))
        feature_retained_count = (
            int(feature_counts.get("post_stage1", feature_raw_count))
            if execution_status == "skipped"
            else len(retained_idx)
        )
        feature_dropped_count = max(
            0, feature_raw_count - feature_retained_count
        )
        feature_retention_rate = (
            round((feature_retained_count / feature_raw_count) * 100, 2)
            if (feature_raw_count > 0)
            else 0.0
        )

        feature_metrics = {
            "execution_status": execution_status,
            "skip_reason": skip_reason,
            "missing_value_count": missing_value_count,
            "filtering_level": filter_level,
            "thresholds": {
                "mv_global_tol": self.config.get("mv_global_tol", 0.7),
                "mv_group_tol": self.config.get("mv_group_tol", 0.5),
                "mv_qc_tol": self.config.get("mv_qc_tol", 0.3),
                "mnar_group_mv_tol": self.config.get("mnar_group_mv_tol", 0.8),
                "mnar_qc_mv_tol": self.config.get("mnar_qc_mv_tol", 0.2),
            },
            "missing_classification": {
                "mar_count": int(len(idx_mar)),
                "mnar_total": int(len(idx_mnar)),
                "mnar_group": int(len(idx_mnar_group)),
                "mnar_qc": int(len(idx_mnar_qc)),
            },
            "feature_retention": {
                "pre_mv_filter_count": feature_raw_count,
                "after_mv_filter_count": feature_retained_count,
                "dropped_count": feature_dropped_count,
                "retention_rate_pct": feature_retention_rate,
            },
        }

        # =====================================================================
        # Unified Metric Assembly
        # =====================================================================
        return {"sample_wise": sample_metrics, "feature_wise": feature_metrics}

    def filter_features_by_quality(
        self,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
        missingness_tracking: pd.DataFrame | None = None,
    ) -> StageResult[MetaboDataset]:
        """Apply blank-ratio and QC-RSD feature-quality rules.

        The quality stage is intentionally runnable without a preceding
        missing-value stage.  When both ``idx_mar`` and ``idx_mnar`` (or a
        ``missingness_type`` column in feature metadata) are available, the
        normal missingness-aware policy is used: MAR features are subject to
        QC-RSD and MNAR features are exempt from that check.  When the
        missingness labels are unavailable, this method enters explicit
        ``quality_only`` mode and applies Blank/QC and QC-RSD to every input
        feature.  It never re-runs ``classify_missing_types`` and therefore
        cannot manufacture MAR/MNAR labels at an independent action boundary.
        """
        feature_metadata = self.dataset.feature_metadata
        metadata_has_missingness = False
        if "missingness_type" in feature_metadata:
            normalized_missingness = (
                feature_metadata["missingness_type"].astype(str).str.upper()
            )
            metadata_has_missingness = normalized_missingness.isin(
                {"MAR", "MNAR"}
            ).any()
        if idx_mar is None and metadata_has_missingness:
            idx_mar = feature_metadata.index[
                feature_metadata["missingness_type"].astype(str).str.upper()
                == "MAR"
            ]
        if idx_mnar is None and metadata_has_missingness:
            idx_mnar = feature_metadata.index[
                feature_metadata["missingness_type"].astype(str).str.upper()
                == "MNAR"
            ]

        # A quality action must not infer upstream missingness state.  Both
        # sets are required for the missingness-aware policy; otherwise the
        # action is a self-contained quality-only check.
        quality_only = idx_mar is None or idx_mnar is None
        if quality_only:
            logger.info(
                "Missingness audit unavailable; running quality-only mode "
                "(Blank/QC and QC-RSD on all input features)."
            )
            idx_mar = pd.Index([], dtype=self.frame.index.dtype)
            idx_mnar = pd.Index([], dtype=self.frame.index.dtype)

        if not isinstance(idx_mar, pd.Index):
            idx_mar = pd.Index(idx_mar)
        if not isinstance(idx_mnar, pd.Index):
            idx_mnar = pd.Index(idx_mnar)

        if quality_only:
            # Rescue flags are meaningful only with an upstream missingness
            # classification and must not leak into a standalone audit.
            self.stats["idx_mnar_group"] = pd.Index([])
            self.stats["idx_mnar_qc"] = pd.Index([])
        else:
            self.stats["idx_mnar_group"] = feature_metadata.index[
                feature_metadata.get(
                    "mnar_group_rescued",
                    pd.Series(False, index=feature_metadata.index),
                ).astype(bool)
            ]
            self.stats["idx_mnar_qc"] = feature_metadata.index[
                feature_metadata.get(
                    "mnar_qc_rescued",
                    pd.Series(False, index=feature_metadata.index),
                ).astype(bool)
            ]
        self.stats["idx_mar"] = idx_mar
        self.stats["idx_mnar"] = idx_mnar
        self.stats["quality_filter_mode"] = (
            "quality_only" if quality_only else "missingness_aware"
        )
        self.stats["missingness_classified"] = not quality_only

        feature_counts = self.stats["feature_counts"]
        # This processor may be instantiated independently, so its input
        # count cannot rely on Feature MV processor state.
        input_feature_count = len(self.frame.index)
        if (
            missingness_tracking is not None
            and "Stage1_Status" in missingness_tracking.columns
        ):
            upstream_tracking = missingness_tracking.copy(deep=True)
            self.stats["stage1_tracking"] = upstream_tracking
            feature_counts["raw"] = len(upstream_tracking)
            feature_counts["post_stage1"] = input_feature_count
            upstream_statuses = (
                upstream_tracking["Stage1_Status"].astype(str).str.upper()
            )
            self.stats["feature_mv_status"] = (
                "skipped"
                if len(upstream_statuses) > 0
                and upstream_statuses.eq("NOT_APPLICABLE").all()
                else "completed"
            )
        else:
            feature_counts.setdefault("raw", input_feature_count)
            self.stats["feature_mv_status"] = "not_run"
        feature_counts["quality_input"] = input_feature_count

        sample_type = self.config.get("sample_type", "Sample Type")
        sample_dict = self.config.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        blank_label = sample_dict.get("Blank sample", "Blank")

        qc_rsd_tol = self.config.get("qc_rsd_tol", 0.3)
        blank_qc_ratio_tol = self.config.get("blank_qc_ratio_tol", 0.2)

        qc_mask = self.frame.columns.get_level_values(sample_type) == qc_label
        blank_mask = (
            self.frame.columns.get_level_values(sample_type) == blank_label
        )
        current_idx = self.frame.index
        logger.info(f"Features before filtering: {len(current_idx)}")

        # Blank Ratio Quality Check
        if blank_mask.any() and qc_mask.any():
            qc_mean = self.frame.loc[:, qc_mask].mean(axis=1)
            blank_mean = self.frame.loc[:, blank_mask].mean(axis=1)

            self.stats.update({"qc_mean": qc_mean, "blank_mean": blank_mean})

            blank_mean_safe = blank_mean.astype(float).fillna(0.0)
            qc_safe = qc_mean.astype(float).replace(0, np.finfo(float).eps)

            ratio_series = blank_mean_safe / qc_safe
            pass_blank = ratio_series[ratio_series <= blank_qc_ratio_tol].index

            next_idx = current_idx.intersection(pass_blank)
            self.stats["idx_dropped_blank"] = current_idx.difference(next_idx)
            current_idx = next_idx
            logger.info(f"Features after Blank/QC check: {len(current_idx)}")
        else:
            self.stats["idx_dropped_blank"] = pd.Index([])

        feature_counts["post_stage2_blank"] = len(current_idx)

        # QC RSD Quality Check
        if qc_mask.any():
            df_qc = self.frame.loc[current_idx, qc_mask]
            std_qc = df_qc.std(axis=1, ddof=1)
            mean_qc = df_qc.mean(axis=1)

            self.stats["qc_rsd_all"] = std_qc / mean_qc
            if quality_only:
                # No upstream labels: QC-RSD applies to every feature that
                # passed Blank/QC.  Do not encode these as MAR.
                pass_quality = self.stats["qc_rsd_all"].loc[current_idx]
                final_idx = pass_quality[pass_quality <= qc_rsd_tol].index
            else:
                pass_mar = self.stats["qc_rsd_all"].loc[
                    idx_mar.intersection(current_idx)
                ]
                final_idx = pass_mar[pass_mar <= qc_rsd_tol].index.union(
                    idx_mnar.intersection(current_idx)
                )

            self.stats["idx_dropped_rsd"] = current_idx.difference(final_idx)
            logger.info(f"Features after QC RSD check: {len(final_idx)}")
        else:
            final_idx = current_idx
            self.stats["idx_dropped_rsd"] = pd.Index([])

        self.stats["idx_retained_stage2"] = final_idx
        feature_counts["post_stage2_rsd"] = len(final_idx)
        df_final = self.frame.loc[final_idx].copy()

        # Generate detailed tracking table via private helper
        start_idx = (
            self.frame.index
            if quality_only
            else idx_mar.union(idx_mnar).intersection(self.frame.index)
        )
        self.stats["quality_pre_stage2_count"] = len(start_idx)
        df_tracking = self._generate_s2_tracking_table(
            start_idx=start_idx,
            idx_mar=idx_mar,
            idx_mnar=idx_mnar,
            quality_only=quality_only,
        )
        self.stats["stage2_tracking"] = df_tracking

        audit_tables = self._audit_snapshot()
        result_dataset = self._to_dataset(
            df_final,
            context_updates={"pipeline_stage": "Low-quality filtering"},
            feature_metadata=feature_metadata.loc[list(final_idx)],
        )
        return StageResult(
            data=result_dataset,
            audit=QualityFilterAuditPayload(
                metric_values=self.quality_filtering_metrics,
                feature_tracking=df_tracking,
                plot_payload=FilteringPlotPayload(
                    data=snapshot_dataset(result_dataset),
                    audit_tables=audit_tables,
                    blank_qc_ratio_tolerance=blank_qc_ratio_tol,
                    qc_rsd_tolerance=qc_rsd_tol,
                ),
                tables=audit_tables,
            ),
        )

    @log_execution_time
    def run_quality_filtering(
        self,
        idx_mar: pd.Index | list[object] | None = None,
        idx_mnar: pd.Index | list[object] | None = None,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Return the structured quality-filtering stage result.

        This stage accepts ``qc_rsd_tol`` and ``blank_qc_ratio_tol``. MAR and
        MNAR indices remain explicit data inputs rather than configuration.
        """
        runner = QualityFilteringStageRunner(
            self,
            output_dir,
            idx_mar=idx_mar,
            idx_mnar=idx_mnar,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=frozenset(
                {"qc_rsd_tol", "blank_qc_ratio_tol"}
            ),
        )
        return runner.run()

    def _generate_s2_tracking_table(
        self,
        start_idx: pd.Index,
        idx_mar: pd.Index,
        idx_mnar: pd.Index,
        *,
        quality_only: bool = False,
    ) -> pd.DataFrame:
        """Build a detailed feature status tracking table for Stage 2.

        In ``quality_only`` mode the ``Base_Type`` is deliberately reported
        as ``UNCLASSIFIED``.  This distinguishes an independently executed
        quality action from a feature that was actually classified as MAR.
        """
        idx_dropped_blank = self.stats.get("idx_dropped_blank", pd.Index([]))
        idx_dropped_rsd = self.stats.get("idx_dropped_rsd", pd.Index([]))

        blank_mean = self.stats.get("blank_mean", pd.Series(dtype=float))
        qc_mean = self.stats.get("qc_mean", pd.Series(dtype=float))
        qc_rsd = self.stats.get("qc_rsd_all", pd.Series(dtype=float))

        track_data = []
        for feat in start_idx:
            if quality_only:
                base_type = "UNCLASSIFIED"
            else:
                base_type = "MNAR" if feat in idx_mnar else "MAR"
            val_blank = blank_mean.get(feat, np.nan)
            val_qc = qc_mean.get(feat, np.nan)
            ratio_val = np.nan

            if pd.notna(val_qc):
                if val_qc > 1e-9:
                    safe_val_blank = 0.0 if pd.isna(val_blank) else val_blank
                    ratio_val = round(safe_val_blank / val_qc, 4)
                else:
                    ratio_val = "QC Mean <= 0"

            val_rsd = qc_rsd.get(feat, np.nan)
            rsd_val = round(val_rsd, 4) if pd.notna(val_rsd) else np.nan

            if feat in idx_dropped_blank:
                (
                    blank_check_status,
                    rsd_check_status,
                    stage2_status,
                    sort_order,
                ) = (
                    "Failed",
                    "Skipped",
                    "Drop",
                    0,
                )
            elif feat in idx_dropped_rsd:
                (
                    blank_check_status,
                    rsd_check_status,
                    stage2_status,
                    sort_order,
                ) = (
                    "Passed",
                    "Failed",
                    "Drop",
                    1,
                )
            else:
                (
                    blank_check_status,
                    rsd_check_status,
                    stage2_status,
                    sort_order,
                ) = (
                    "Passed",
                    "Passed",
                    "Keep",
                    2,
                )

            if base_type == "MNAR" and feat not in idx_dropped_blank:
                rsd_check_status = "Exempted (MNAR)"

            track_data.append(
                {
                    "Feature_ID": feat,
                    "Base_Type": base_type,
                    "Ratio_Value": ratio_val,
                    "Ratio_Check": blank_check_status,
                    "RSD_Value": rsd_val,
                    "RSD_Check": rsd_check_status,
                    "Stage2_Status": stage2_status,
                    "_sort": sort_order,
                }
            )

        if not track_data:
            return pd.DataFrame(
                columns=[
                    "Base_Type",
                    "Ratio_Value",
                    "Ratio_Check",
                    "RSD_Value",
                    "RSD_Check",
                    "Stage2_Status",
                ],
                index=pd.Index([], name="Feature_ID"),
            )
        df_tracking = pd.DataFrame(track_data).set_index("Feature_ID")
        df_tracking = df_tracking.sort_values(by=["_sort", "Base_Type"])
        return df_tracking.drop(columns=["_sort"])

    @cached_property
    def quality_filtering_metrics(self) -> dict:
        """Extracts metrics from Stage-2 low-quality feature filtering."""
        feature_counts = self.stats.get("feature_counts", {})
        quality_only = self.stats.get("quality_filter_mode") == "quality_only"

        idx_dropped_blank = self.stats.get("idx_dropped_blank", pd.Index([]))
        idx_dropped_rsd = self.stats.get("idx_dropped_rsd", pd.Index([]))
        idx_mar = self.stats.get("idx_mar", pd.Index([]))
        idx_mnar = self.stats.get("idx_mnar", pd.Index([]))

        blank_drop_mar = len(idx_dropped_blank.intersection(idx_mar))
        blank_drop_mnar = len(idx_dropped_blank.intersection(idx_mnar))
        rsd_drop_mar = len(idx_dropped_rsd.intersection(idx_mar))
        rsd_drop_mnar = len(idx_dropped_rsd.intersection(idx_mnar))

        pre_mar = len(idx_mar)
        pre_mnar = len(idx_mnar)
        pre_unclassified = (
            int(self.stats.get("quality_pre_stage2_count", 0))
            if quality_only
            else 0
        )

        post_blank_mar = pre_mar - blank_drop_mar
        post_blank_mnar = pre_mnar - blank_drop_mnar
        post_rsd_mar = post_blank_mar - rsd_drop_mar
        post_rsd_mnar = post_blank_mnar - rsd_drop_mnar
        post_blank_unclassified = (
            max(0, int(feature_counts.get("post_stage2_blank", 0)))
            if quality_only
            else 0
        )
        post_rsd_unclassified = (
            max(
                0,
                int(feature_counts.get("post_stage2_rsd", 0)),
            )
            if quality_only
            else 0
        )

        metrics = {
            "filtering_mode": (
                "quality_only" if quality_only else "missingness_aware"
            ),
            "missingness_classified": not quality_only,
            "thresholds": {
                "blank_qc_ratio_tol": self.config.get(
                    "blank_qc_ratio_tol", 0.2
                ),
                "qc_rsd_tol": self.config.get("qc_rsd_tol", 0.3),
            },
            "feature_retention": {
                "pre_stage2": {
                    "total": int(
                        self.stats.get(
                            "quality_pre_stage2_count",
                            feature_counts.get("post_stage1", 0),
                        )
                    ),
                    "mar_count": pre_mar,
                    "mnar_count": pre_mnar,
                    "unclassified_count": pre_unclassified,
                },
                "post_blank_check": {
                    "total": feature_counts.get("post_stage2_blank", 0),
                    "mar_count": post_blank_mar,
                    "mnar_count": post_blank_mnar,
                    "unclassified_count": post_blank_unclassified,
                },
                "post_rsd_check": {
                    "total": feature_counts.get("post_stage2_rsd", 0),
                    "mar_count": post_rsd_mar,
                    "mnar_count": post_rsd_mnar,
                    "unclassified_count": post_rsd_unclassified,
                },
            },
            "filtering_breakdown": {
                "dropped_by_blank": {
                    "total": len(idx_dropped_blank),
                    "mar_count": blank_drop_mar,
                    "mnar_count": blank_drop_mnar,
                    "unclassified_count": (
                        len(idx_dropped_blank) if quality_only else 0
                    ),
                },
                "dropped_by_rsd": {
                    "total": len(idx_dropped_rsd),
                    "mar_count": rsd_drop_mar,
                    "mnar_count": rsd_drop_mnar,
                    "unclassified_count": (
                        len(idx_dropped_rsd) if quality_only else 0
                    ),
                },
            },
        }
        return metrics
