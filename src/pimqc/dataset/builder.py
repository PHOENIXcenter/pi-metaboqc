"""Dataset construction and validation from metadata and intensity tables.

Align sample identities, validate acquisition metadata, and create the explicit
dataset and typed construction audit consumed by native processing stages.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional, Dict, Any

from ..io import ensure_directory
from ..config import validate_pipeline_params
from ..plotting.payloads import DatasetPlotPayload, snapshot_dataset
from ..processing.audit import DatasetAuditPayload
from ..processing.stage import StageResult
from ..runtime import log_execution_time
from ..core.dataset import (
    DatasetSchema,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)


class MetaboDatasetBuilder:
    """Validate input tables and run the dataset-construction stage.

    ``run_build`` returns the aligned ``MetaboDataset`` and its typed audit
    through ``StageResult``, matching the other processing actors. Supplying
    an output directory also exports the raw table and acquisition overview.
    Configuration comes from the validated pipeline ``Dataset`` section.
    """

    def __init__(
        self,
        meta_info: pd.DataFrame,
        int_df: pd.DataFrame,
        pipeline_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Copy input tables and validate configuration without file output."""
        self.metadata_dataframe = meta_info.copy()
        self.intensity_dataframe = int_df.copy()
        self.params = validate_pipeline_params(pipeline_params)

        # Extract configuration attributes smartly
        dataset_params = self.params.get("Dataset", {})

        self.dataset_params = dataset_params
        self.mode = dataset_params.get("mode", "ESI+")
        self.batch = dataset_params.get("batch", "Batch")
        self.sample_type = dataset_params.get("sample_type", "Sample Type")
        self.bio_group = dataset_params.get("bio_group", "Bio Group")
        self.sample_name = dataset_params.get("sample_name", "Sample Name")
        self.inject_order = dataset_params.get("inject_order", "Inject Order")
        self.resort_strategy = dataset_params.get("resort_inject_order", "Auto")

        self.unique_batches = []
        self.is_multi_batch = False

    def _resolve_duplicate_features(self) -> None:
        """Average intensities for duplicated metabolite names."""
        if isinstance(self.intensity_dataframe.index, pd.RangeIndex):
            first_column = self.intensity_dataframe.columns[0]
            logger.warning(
                f"RangeIndex detected. Setting '{first_column}' as "
                "feature index."
            )
            self.intensity_dataframe = self.intensity_dataframe.set_index(
                first_column
            )

        self.intensity_dataframe.index.name = "Metabolite"

        # Record original index to identify modified features
        orig_idx = self.intensity_dataframe.index.astype(str)
        new_idx = orig_idx.str.strip()

        # Calculate exactly how many features had leading/trailing whitespaces
        stripped_count = (orig_idx != new_idx).sum()
        self.intensity_dataframe.index = new_idx

        # Logger 1: Output the number of stripped features
        if stripped_count > 0:
            logger.info(
                f"Stripped whitespaces from {stripped_count} feature(s)."
            )

        if self.intensity_dataframe.index.duplicated().any():
            num_duplicates = self.intensity_dataframe.index.duplicated().sum()
            duplicated_features = (
                self.intensity_dataframe.index[
                    self.intensity_dataframe.index.duplicated()
                ]
                .unique()
                .tolist()
            )

            # Logger 2: Output the exact names of features merged by mean
            logger.info(f"Features merged via mean: {duplicated_features}")

            logged_features = (
                duplicated_features[:5] + ["..."]
                if len(duplicated_features) > 5
                else duplicated_features
            )
            logger.warning(
                f"Detected {num_duplicates} duplicate row indices "
                f"(e.g., {logged_features}). Merging their intensities "
                "by averaging (mean)."
            )
            self.intensity_dataframe = self.intensity_dataframe.groupby(
                level=0, sort=False
            ).mean()

    def _check_duplicate_samples(self) -> None:
        """Abort if intensity dataframe contains duplicated sample columns."""
        value_counts = self.intensity_dataframe.columns.value_counts()
        if value_counts.max() > 1:
            duplicated_names = value_counts[value_counts > 1].index.tolist()
            duplicated_details = []

            for name in duplicated_names:
                locations = [
                    i
                    for i, col in enumerate(self.intensity_dataframe.columns)
                    if col == name
                ]
                duplicated_details.append(f'"{name}" (indices: {locations})')

            error_message = (
                "Duplicate sample names in intensity data: "
                f"{', '.join(duplicated_details)}."
            )
            raise AssertionError(error_message)

    def _verify_sample_consistency(self) -> None:
        """Ensure sample names perfectly match between meta and intensity."""
        meta_samples = set(self.metadata_dataframe[self.sample_name])
        intensity_samples = set(self.intensity_dataframe.columns)

        if meta_samples != intensity_samples:
            only_in_meta = sorted(list(meta_samples - intensity_samples))
            only_in_intensity = sorted(list(intensity_samples - meta_samples))

            intersection_size = len(
                meta_samples.intersection(intensity_samples)
            )
            union_size = len(meta_samples.union(intensity_samples))
            jaccard_score = intersection_size / union_size

            message = [
                f"Sample inconsistency (Jaccard Score: {jaccard_score:.4f})."
            ]

            if only_in_meta:
                log_meta = (
                    only_in_meta[:5] + ["..."]
                    if len(only_in_meta) > 5
                    else only_in_meta
                )
                message.append(f"Only in Metadata: {log_meta}")

            if only_in_intensity:
                log_intensity = (
                    only_in_intensity[:5] + ["..."]
                    if len(only_in_intensity) > 5
                    else only_in_intensity
                )
                message.append(f"Only in Intensity: {log_intensity}")

            raise AssertionError(" ".join(message))

    def _verify_metadata_completeness(self) -> None:
        """Abort if critical tracking columns are missing from metadata."""
        required_cols = {
            "Batch": self.batch,
            "Sample Type": self.sample_type,
            "Sample Name": self.sample_name,
            "Inject Order": self.inject_order,
        }

        missing_cols = [
            col_name
            for label, col_name in required_cols.items()
            if col_name not in self.metadata_dataframe.columns
        ]

        if missing_cols:
            raise AssertionError(
                f"Incomplete metadata. Missing columns: {missing_cols}."
            )

        self.unique_batches = self.metadata_dataframe[self.batch].unique()
        self.is_multi_batch = len(self.unique_batches) > 1

    def _manage_injection_orders(self) -> None:
        """Align injection sequences across batches to prevent overlap."""
        if not self.is_multi_batch or (
            self.inject_order not in self.metadata_dataframe
        ):
            return

        if not self.resort_strategy:
            return

        self.metadata_dataframe[self.inject_order] = pd.to_numeric(
            self.metadata_dataframe[self.inject_order], errors="coerce"
        )
        ordered_batches = sorted(
            self.metadata_dataframe[self.batch].dropna().unique().tolist()
        )

        is_overlap = False
        current_max = -float("inf")
        for batch_id in ordered_batches:
            batch_mask = self.metadata_dataframe[self.batch] == batch_id
            if batch_mask.sum() == 0:
                continue
            batch_min = self.metadata_dataframe.loc[
                batch_mask, self.inject_order
            ].min()
            batch_max = self.metadata_dataframe.loc[
                batch_mask, self.inject_order
            ].max()

            if batch_min <= current_max:
                is_overlap = True
                break
            current_max = max(current_max, batch_max)

        trigger = False
        if self.resort_strategy == "Auto" and is_overlap:
            trigger = True
        elif str(self.resort_strategy).lower() in ["force", "true", "always"]:
            trigger = True

        if trigger:
            logger.warning(
                f"Inject orders resort triggered "
                f"(mode: {self.resort_strategy})."
                " Re-numbering sequentially to ensure global continuity."
            )
            previous_max = None
            for batch_id in ordered_batches:
                batch_mask = self.metadata_dataframe[self.batch] == batch_id
                if batch_mask.sum() == 0:
                    continue

                batch_min = self.metadata_dataframe.loc[
                    batch_mask, self.inject_order
                ].min()
                if previous_max is not None:
                    offset = previous_max - batch_min + 1
                    self.metadata_dataframe.loc[
                        batch_mask, self.inject_order
                    ] += offset
                previous_max = self.metadata_dataframe.loc[
                    batch_mask, self.inject_order
                ].max()

    def _audit_dataset_health(self, dataset: MetaboDataset) -> None:
        """
        Conducts a comprehensive health audit on the newly built dataset.
        Emits targeted warnings and infos to set user expectations regarding
        pipeline degradation and algorithmic limitations.
        """
        logger.info("Executing dataset health audit...")

        # Blank Samples Check
        if dataset.blank_data.empty:
            logger.warning(
                "[Audit] No Blank samples detected. Pipeline will skip Stage-2 "
                "Blank/QC ratio filtering and degrade to QC RSD check only."
            )

        # Biological Groups Check
        if not self.bio_group or self.bio_group not in dataset.sample_metadata:
            logger.warning(
                "[Audit] No Biological Group information detected. "
                "Missing value "
                "imputation and filtering will fall back to QC-only rescue."
            )

        # Internal Standards (IS) Check
        if not dataset.valid_internal_standards:
            logger.warning(
                "[Audit] No Internal Standards (IS) detected. "
                "Analytical outlier "
                "diagnostics will rely solely on global PCA statistics."
            )

        # Outlier Reference Features (ORF) Check
        if not dataset.valid_outlier_reference_features:
            logger.info(
                "[Audit] No Outlier Reference Features (ORF) detected. This is "
                "normal for untargeted datasets; ORF diagnostics "
                "will be skipped."
            )

        # High-throughput Cohort Check (Batch Count > 15)
        # Threshold set to 15 to match the plotter's MathText threshold
        n_batches = len(dataset.ordered_batches)
        if n_batches > 15:
            logger.info(
                f"[Audit] High batch count (n={n_batches}) detected. "
                "Visualizations will automatically switch to "
                "alphanumeric mode for readability."
            )

        # Critical QC Density Check
        if dataset.qc_data.empty:
            logger.error(
                "[Audit] FATAL: No QC samples detected! Subsequent correction "
                "and evaluation steps will inevitably fail."
            )
        else:
            qc_ids = dataset.role_sample_ids("qc")
            qc_counts = dataset.sample_metadata.loc[
                qc_ids, self.batch
            ].value_counts()

            # Warn if any batch has fewer than 3 QCs (Minimum required for
            # SVR/RFSC)
            weak_batches = qc_counts[qc_counts < 3].index.tolist()
            if weak_batches:
                logger.warning(
                    f"[Audit] Batches {weak_batches} contain fewer than 3 QCs. "
                    "Step-wise machine learning correction algorithms (e.g., "
                    "QC-SVR) may severely overfit or fail in these batches."
                )

    @log_execution_time
    def run_build(
        self,
        output_dir: str | Path | None = None,
    ) -> StageResult[MetaboDataset]:
        """Build data and audit, then optionally export and render artifacts.

        Args:
            output_dir: Directory for ``Raw_Data_Intensity.csv`` and
                ``Global_Acquisition_Overview.svg``. Omit it to run entirely
                in memory, including preparation of the audit plot payload.

        Returns:
            A stage result containing the dataset and ``DatasetAuditPayload``.
        """
        from .runner import DatasetBuildStageRunner

        return DatasetBuildStageRunner(self, output_dir).run()

    def compute_build(self) -> StageResult[MetaboDataset]:
        """Prepare the dataset and detached audit without file or plot output.

        Construction uses the same validation and alignment rules as the
        data-only ``execute_build`` interface.
        """
        dataset = self.execute_build()
        return StageResult(
            data=dataset,
            audit=DatasetAuditPayload(
                metric_values=dataset.dataset_metrics,
                plot_payload=DatasetPlotPayload(data=snapshot_dataset(dataset)),
            ),
        )

    def execute_build(
        self,
        output_dir: str | Path | None = None,
    ) -> MetaboDataset:
        """Build the dataset using the legacy data-only interface.

        An output directory enables CSV export only, preserving this method's
        original behavior. Use ``run_build`` for the typed audit and dashboard.
        """
        self._resolve_duplicate_features()
        self._check_duplicate_samples()
        self._verify_sample_consistency()
        self._verify_metadata_completeness()
        self._manage_injection_orders()

        # Keep the canonical matrix flat and align explicit sample metadata.
        self.intensity_dataframe = self.intensity_dataframe.rename_axis(
            index=["Metabolite"], columns=[self.sample_name]
        )

        has_bio_group = pd.notna(self.bio_group) and (
            self.bio_group in self.metadata_dataframe.columns
        )
        column_order = (
            [
                self.batch,
                self.sample_type,
                self.bio_group,
                self.inject_order,
                self.sample_name,
            ]
            if has_bio_group
            else [
                self.batch,
                self.sample_type,
                self.inject_order,
                self.sample_name,
            ]
        )
        column_order.extend(
            column
            for column in self.metadata_dataframe.columns
            if column not in column_order
        )

        metadata = self.metadata_dataframe.set_index(self.sample_name).loc[
            list(self.intensity_dataframe.columns)
        ]

        # =====================================================================
        # Zero-value detection and conversion
        # =====================================================================
        zero_mask = self.intensity_dataframe == 0
        zero_count = int(zero_mask.sum().sum())

        if zero_count > 0:
            logger.warning(
                f"[Dataset Builder] Detected {zero_count} explicit zero (0) "
                "values in the raw matrix. In mass spectrometry, these "
                "typically represent missing values rather than true zero "
                "intensity. They have been automatically converted to NaN to "
                "ensure pipeline safety."
            )
            # Represent explicit zeros as missing values for downstream steps.
            self.intensity_dataframe = self.intensity_dataframe.replace(
                0, np.nan
            )

        # Use float64 for stable downstream machine-learning assignments.
        self.intensity_dataframe = self.intensity_dataframe.astype(float)

        roles = SampleRoleLabels.from_mapping(
            self.dataset_params.get("sample_dict", {})
        )
        schema = DatasetSchema(
            feature_id=self.intensity_dataframe.index.name or "Metabolite",
            sample_id=self.sample_name,
            sample_type=self.sample_type,
            batch=self.batch,
            injection_order=self.inject_order,
            biological_group=(self.bio_group if has_bio_group else None),
            roles=roles,
            sample_metadata_order=tuple(column_order),
        )
        context = ProcessingContext.from_mapping(self.dataset_params)
        dataset = MetaboDataset.from_tables(
            intensity=self.intensity_dataframe,
            sample_metadata=metadata,
            schema=schema,
            context=context,
        )

        if output_dir:
            ensure_directory(output_dir)
            _export_dataset_table(dataset, output_dir)

        logger.info(
            f"MetaboDataset built: {dataset.intensity.shape[0]} metabolites, "
            f"{dataset.intensity.shape[1]} samples."
        )

        self._audit_dataset_health(dataset)

        return dataset


def build_dataset(
    meta_info: pd.DataFrame,
    int_df: pd.DataFrame,
    pipeline_params: Optional[Dict[str, Any]] = None,
    output_dir: str | Path | None = None,
) -> MetaboDataset:
    """Return only the dataset through the compatible functional interface.

    New workflows should use ``MetaboDatasetBuilder(...).run_build()`` to
    retain the typed dataset audit as well. This wrapper delegates the full
    timed build/export/render lifecycle to that same class implementation.
    """
    builder = MetaboDatasetBuilder(
        meta_info=meta_info, int_df=int_df, pipeline_params=pipeline_params
    )

    return builder.run_build(output_dir=output_dir).data


def _export_dataset_table(
    dataset: MetaboDataset,
    output_dir: str | os.PathLike[str],
) -> None:
    """Write the established raw CSV for both builder entry points."""
    output_path = os.path.join(output_dir, "Raw_Data_Intensity.csv")
    dataset.annotated_frame().to_csv(
        output_path, na_rep="NA", encoding="utf-8-sig"
    )
    logger.info(f"Raw dataset saved as: {output_path}")


def _render_dataset_dashboard(
    payload: DatasetPlotPayload,
    output_dir: str | os.PathLike[str],
) -> None:
    """Render the dataset overview from the completed audit's plot payload.

    Plotting is imported lazily so dataset validation and construction remain
    usable without importing the plotting stack. Notebook display is handled
    by ``save_and_show_pw`` when the same execution runs in Jupyter.
    """
    from ..plotting.dataset import DatasetPlotter

    try:
        plotter = DatasetPlotter(payload)
        dashboard = plotter.plot_dataset_dashboard()
        if dashboard is None:
            logger.warning(
                "Dataset dashboard was skipped because no plotting backend "
                "was available."
            )
            return

        overview_path = os.path.join(
            output_dir, "Global_Acquisition_Overview.svg"
        )
        plotter.save_and_show_pw(
            pw_obj=dashboard,
            file_path=str(overview_path),
        )
        logger.info(
            f"Global acquisition overview plot saved as: {overview_path}"
        )
    except Exception as exc:
        logger.error(f"Dataset dashboard rendering failed: {exc}")
