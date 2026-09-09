"""End-to-end orchestration from input tables to structured stage results.

``run_pipeline`` constructs the dataset, executes QA checkpoints, runs the
three explicit filtering actions, correction, imputation, and normalization,
then builds the report input. Every processing and assessment call returns a
``StageResult``; the pipeline keeps stage tables, results, assessments, and
metrics in explicit mappings. With no ``output_dir`` it performs the complete
calculation in memory and skips filesystem export, plotting, and reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from ._version import __version__
from .core import MetaboDataset
from .config import validate_pipeline_params
from .io import ensure_directory
from .runtime import log_execution_time
from .reporting import ReportInput
from .reporting import utils as ru
from .dataset import MetaboDatasetBuilder
from .processing.assessment import AssessmentDiagnostics, QualityAssessor
from .processing.filtering import FilteringOrchestrator
from .processing.correction import SignalCorrector
from .processing.imputation import MissingValueImputer
from .processing.normalization import DataNormalizer
from .processing.stage import StageResult


@dataclass
class PipelineResult:
    """Carry all in-memory products of one end-to-end pipeline execution."""

    stage_tables: dict[str, MetaboDataset]
    stage_results: dict[str, StageResult[Any]]
    assessments: dict[str, StageResult[AssessmentDiagnostics]]
    pipeline_metrics: dict[str, Any]
    qa_metrics: dict[str, Any]
    report_input: ReportInput
    output_dir: Path | None = None
    report_generated: bool = False
    filtering_results: dict[str, StageResult[Any]] = field(default_factory=dict)

    @property
    def data(self) -> MetaboDataset:
        """Return the final normalized matrix as the primary pipeline output."""
        return self.stage_tables["normalized"]


@log_execution_time
def run_pipeline(
    meta_df: pd.DataFrame,
    int_df: pd.DataFrame,
    params: dict,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """Run the complete pi-metaboqc pipeline.

    Executes dataset construction, QA checkpoints, the three explicit
    filtering actions, signal correction, missingness-aware imputation,
    normalization, and optional report generation. Omitting ``output_dir``
    still performs every calculation but skips all table, figure, and report
    side effects while returning the same structured in-memory result.

    Args:
        meta_df (pd.DataFrame): The metadata pandas DataFrame.
        int_df (pd.DataFrame): The intensity pandas DataFrame.
        params (dict): The pipeline parameters dict.
        output_dir: Optional root directory for exported artifacts.

    Returns:
        Structured stage tables, diagnostics, metrics, and report status.
    """
    logger.info(f"pi-metaboqc version: {__version__}")

    # Normalize the in-memory configuration through the same strict schema as
    # TOML/JSON loading. This makes old section names and misspelled fields
    # fail before any stage starts and guarantees one precedence surface.
    params = validate_pipeline_params(params)

    # =========================================================================
    # Step 00: Environment Initialization
    # **Purpose & Function**:
    # Creates the root output directory only when artifacts are requested.
    # These explicit mappings are the hand-off contract between stages:
    # tables carry matrices, results carry metrics/audits, and assessments
    # carry observational QA diagnostics. With exports enabled, each stage
    # owns a dedicated subdirectory and follows compute -> export -> render.
    # =========================================================================
    logger.info("Step 00: Environment Initialization...")
    root_dir = ensure_directory(output_dir) if output_dir is not None else None
    if root_dir is None:
        logger.info("Running the pipeline without filesystem artifacts.")
    else:
        logger.info(f"Pipeline workspace securely mounted at: {root_dir}")

    def stage_dir(name: str) -> Path | None:
        """Resolve a stage artifact directory without creating it eagerly."""
        return (
            Path(os.path.join(root_dir, name))
            if root_dir is not None
            else None
        )

    stage_results: dict[str, StageResult[Any]] = {}
    filtering_results: dict[str, StageResult[Any]] = {}
    assessments: dict[str, StageResult[AssessmentDiagnostics]] = {}
    stage_tables: dict[str, MetaboDataset] = {}

    # =========================================================================
    # Step 01: Dataset Construction
    # **Purpose & Function**:
    # Validates metadata and peak-table consistency, resolves duplicated
    # features and injection-order issues, aligns intensity columns to
    # metadata, and builds the composition-based `MetaboDataset` object.
    # Explicit zero intensities are converted to missing values. When an
    # output directory is supplied, the builder exports the raw matrix and
    # renders the global acquisition overview as part of this stage. The
    # builder returns its own DatasetAuditPayload with composition metrics
    # and a detached plot payload, just like the downstream processing actors.
    # =========================================================================
    logger.info("Step 01: Dataset Construction...")
    step1_dir = stage_dir("01_Raw_Data")
    dataset_builder = MetaboDatasetBuilder(
        meta_info=meta_df,
        int_df=int_df,
        pipeline_params=params,
    )
    raw_result = dataset_builder.run_build(output_dir=step1_dir)
    raw_data = raw_result.data
    stage_results["raw_dataset"] = raw_result
    stage_tables["raw"] = raw_data
    is_multi_batch_flag = raw_data.is_multi_batch

    # =========================================================================
    # QA-Step 01: Quality Assessment of Raw Data
    # **Evaluation Logic**:
    # Establishes the unprocessed baseline with the fixed QA suite: QC and
    # batch correlation, QC RSD distribution, PCA and multivariate outlier
    # diagnostics, plus optional internal-standard or outlier-reference-
    # feature evaluations and control charts. These metrics provide the
    # comparison point for all later transformations.
    # =========================================================================
    logger.info("QA-Step 01: Quality Assessment of Raw Data...")
    qa_step1_dir = stage_dir("QA_01_Raw_Data")
    qa_raw_engine = QualityAssessor(data=raw_data, pipeline_params=params)
    qa_raw_result = qa_raw_engine.run_assessment(output_dir=qa_step1_dir)
    assessments["raw_dataset"] = qa_raw_result

    # =========================================================================
    # Step 02: Missing-Value Filtering
    # **Purpose & Function**:
    # Runs two explicit but dependent actions. First, pooled-QC and Actual
    # samples are screened by sample-level missingness; Blank and other
    # non-target roles remain intact. Second, the retained sample matrix is
    # evaluated for global, pooled-QC, and biological-group missingness.
    # Feature routing applies MAR eligibility, group/QC MNAR rescue, or
    # exclusion. The independent sample and feature Audits retain counts,
    # reasons, thresholds, and plot payloads. The complete feature tracking
    # table is reused later by quality filtering and imputation.
    # =========================================================================
    logger.info("Step 02: Sample and Feature Missing-Value Filtering...")
    step2_dir = stage_dir("02_MV_Filtered")
    filtering_orchestrator = FilteringOrchestrator(
        data=raw_data,
        pipeline_params=params,
    )
    missingness_results = filtering_orchestrator.run_missingness(
        output_dir=step2_dir,
    )
    sample_mv_result = missingness_results.sample_result
    mv_filter_result = missingness_results.feature_result
    mv_filter_data = mv_filter_result.data
    filtering_results["sample_missing_value_filtering"] = sample_mv_result
    filtering_results["feature_missing_value_filtering"] = mv_filter_result
    stage_tables["sample_mv_filtered"] = sample_mv_result.data
    stage_results["sample_missing_value_filtering"] = sample_mv_result
    stage_results["high_mv_feature_filtering"] = mv_filter_result
    stage_tables["high_mv_filtered"] = mv_filter_data

    # =========================================================================
    # QA-Step 02: Quality Assessment of High-MV Filtered Data
    # **Evaluation Logic**:
    # Compares the post-filter matrix with the raw baseline to reveal the
    # effect of sample attrition and MAR/MNAR routing on QC consistency, RSD,
    # PCA geometry, multivariate outliers, and reference-feature behaviour.
    # Filtering labels and attrition reasons remain authoritative in the
    # preceding processing Audits; this checkpoint describes only the state
    # of the retained matrix.
    # =========================================================================
    logger.info("QA-Step 02: Quality Assessment of High-MV Filtered Data...")
    qa_step2_dir = stage_dir("QA_02_MV_Filtered")
    qa_mv_filter_engine = QualityAssessor(
        data=mv_filter_data, pipeline_params=params
    )
    qa_mv_filter_result = qa_mv_filter_engine.run_assessment(
        output_dir=qa_step2_dir
    )
    assessments["high_mv_feature_filtering"] = qa_mv_filter_result

    # =========================================================================
    # Step 03: Signal Drift & Batch Effect Correction
    # **Purpose & Function**:
    # Evaluates one configured QC-anchored/global method or, in `Auto` mode,
    # compares SERRF, RUV-III, WaveICA 2.0, standard/robust QC-RLSC, and
    # QC-SVR using QC-RSD and sample-structure criteria. QC-RLSC, QC-SVR,
    # and QC-RFSC fit feature-wise drift from QC samples; multi-batch runs
    # also evaluate batch alignment where applicable. Only the selected
    # method's stage matrices cross the StageResult boundary; candidate
    # metrics remain in `CorrectionAuditPayload.candidate_results`. Blanks are
    # excluded from fitting where supported and receive frozen-model
    # corrections in output.
    # =========================================================================
    logger.info("Step 03: Signal Drift & Batch Effect Correction...")
    step3_dir = stage_dir("03_Corrected_Data")
    sc_engine = SignalCorrector(data=mv_filter_data, pipeline_params=params)
    correction_result = sc_engine.run_signal_correction(output_dir=step3_dir)
    corrected_stages = correction_result.data
    stage_results["signal_correction"] = correction_result
    for stage_name, stage_data in corrected_stages.items():
        stage_tables[f"corrected/{stage_name}"] = stage_data
    final_corr_data = list(corrected_stages.values())[-1]

    # =========================================================================
    # QA-Step 03: Quality Assessment of Signal Corrected Data
    # **Evaluation Logic**:
    # Assesses every selected correction stage, rather than every candidate
    # evaluated internally. The comparison shows whether drift and batch
    # effects improved, while checking for unintended changes to sample
    # structure and reference-feature behaviour.
    # =========================================================================
    logger.info("QA-Step 03: Quality Assessment of Signal Corrected Data...")
    qa_step3_dir = stage_dir("QA_03_Corrected_Data")
    qa_results_dict = {}
    for stage_name, stage_data in corrected_stages.items():
        logger.info(
            f"QA-Step 03: Quality Assessment for corrected stage "
            f"'{stage_name}'..."
        )
        qa_corr_engine = QualityAssessor(
            data=stage_data, pipeline_params=params
        )
        qa_corr_result = qa_corr_engine.run_assessment(
            output_dir=(
                Path(os.path.join(qa_step3_dir, stage_name))
                if qa_step3_dir is not None
                else None
            )
        )
        qa_results_dict[stage_name] = qa_corr_result
        assessments[f"signal_correction/{stage_name}"] = qa_corr_result

    # =========================================================================
    # Step 04: Low-Quality Feature Filtering
    # **Purpose & Function**:
    # Uses the previous pass's explicit MAR/MNAR labels. It first removes
    # features whose mean Blank/QC ratio exceeds its tolerance, then removes
    # technically irreproducible MAR features whose pooled-QC RSD exceeds its
    # tolerance. MNAR features are exempt from the RSD screen; both checks and
    # their drop reasons are retained in the StageResult audit table.
    # =========================================================================
    logger.info("Step 04: Low-Quality Feature Filtering...")
    step4_dir = stage_dir("04_Quality_Filtered")
    filtering_orchestrator.run_quality(
        data=final_corr_data,
        output_dir=step4_dir,
    )
    assert filtering_orchestrator.result is not None
    low_quality_filter_result = filtering_orchestrator.result.quality_result
    assert low_quality_filter_result is not None
    low_quality_filter_data = low_quality_filter_result.data
    filtering_results["feature_quality_filtering"] = low_quality_filter_result
    stage_results["low_quality_feature_filtering"] = low_quality_filter_result
    stage_tables["quality_filtered"] = low_quality_filter_data

    # =========================================================================
    # QA-Step 04: Quality Assessment of Quality Filtered Data
    # **Evaluation Logic**:
    # Checks whether Blank/QC and RSD-based feature removal changed QC
    # consistency or sample structure before imputation. The retained-feature
    # set and removal reasons are documented by the preceding StageResult.
    # =========================================================================
    logger.info(
        "QA-Step 04: Quality Assessment of Low-Quality Filtered Data..."
    )
    qa_step4_dir = stage_dir("QA_04_Quality_Filtered")
    qa_low_quality_filter_engine = QualityAssessor(
        data=low_quality_filter_data, pipeline_params=params
    )
    qa_low_quality_filter_result = qa_low_quality_filter_engine.run_assessment(
        output_dir=qa_step4_dir
    )
    assessments["low_quality_feature_filtering"] = qa_low_quality_filter_result

    # =========================================================================
    # Step 05: Missing Value Imputation
    # **Purpose & Function**:
    # Applies label-aware routes to non-Blank samples. MNAR features use QRILC
    # or row-, column-, or global LOD-fraction constants. MAR features use
    # KNN, LLS, BPCA, MinProb, Median, or `Auto`; Auto benchmarks the MAR
    # candidates with stratified masking, reconstruction/distribution metrics,
    # and sample-structure preservation before applying the selected method.
    # Blank columns are preserved without imputation, and the stage records a
    # skipped result when no target values are missing.
    # =========================================================================
    logger.info("Step 05: Missing Value Imputation...")
    step5_dir = stage_dir("05_Imputation")
    imp_engine = MissingValueImputer(
        data=low_quality_filter_data, pipeline_params=params
    )
    imputation_result = imp_engine.run_imputation(output_dir=step5_dir)
    imputed_data = imputation_result.data
    stage_results["missing_value_imputation"] = imputation_result
    stage_tables["imputed"] = imputed_data

    # =========================================================================
    # QA-Step 05: Quality Assessment of Imputed Data
    # **Evaluation Logic**:
    # Checks whether completing missing values introduced distributional,
    # QC-consistency, or sample-structure shifts. The imputation StageResult,
    # rather than QA, carries the selected method and candidate benchmarks.
    # =========================================================================
    logger.info("QA-Step 05: Quality Assessment of Imputed Data...")
    qa_step5_dir = stage_dir("QA_05_Imputed_Data")
    qa_imp_engine = QualityAssessor(data=imputed_data, pipeline_params=params)
    qa_imp_result = qa_imp_engine.run_assessment(output_dir=qa_step5_dir)
    assessments["missing_value_imputation"] = qa_imp_result

    # =========================================================================
    # Step 06: Data Normalization
    # **Purpose & Function**:
    # Removes sample-wise scale effects and stabilizes intensity variance using
    # robust-log-only, TIC, Median, PQN, MDFC, Quantile, or VSN processing.
    # TIC/Median/PQN/MDFC apply robust Log2 afterward; Quantile operates on a
    # robust-logged view; VSN applies its intrinsic generalized log. `Auto`
    # scores fixed candidates with QC RLE, variance/structure diagnostics, and
    # sample-structure preservation, then selects one conservative method.
    # Blank samples are permanently excluded from the normalized output, and
    # the Auto passport is kept in the processing StageResult and optional CSV.
    # =========================================================================
    logger.info("Step 06: Data Normalization...")
    step6_dir = stage_dir("06_Normalized_Data")
    norm_engine = DataNormalizer(imputed_data, pipeline_params=params)
    normalization_result = norm_engine.run_normalization(output_dir=step6_dir)
    normalized_data = normalization_result.data
    stage_results["normalization"] = normalization_result
    stage_tables["normalized"] = normalized_data

    # =========================================================================
    # QA-Step 06: Quality Assessment of Normalized Data
    # **Evaluation Logic**:
    # Checks whether normalization reduced residual scale/batch effects while
    # preserving sample structure and reference-feature behaviour. The
    # normalized MetaboDataset carries its log/scale state explicitly, so
    # scale-sensitive QA does not infer state from a DataFrame or stage name.
    # RLE diagnostics and candidate scores belong to Step 06's processing
    # Audit rather than this observational assessment.
    # =========================================================================
    logger.info("QA-Step 06: Quality Assessment of Normalized Data...")
    qa_step6_dir = stage_dir("QA_06_Norm_Data")
    qa_norm_engine = QualityAssessor(
        data=normalized_data, pipeline_params=params
    )
    qa_norm_result = qa_norm_engine.run_assessment(output_dir=qa_step6_dir)
    assessments["normalization"] = qa_norm_result

    # =========================================================================
    # Step 07: Sequential Audit Report Compilation
    # **Purpose & Function**:
    # Draws four cross-stage QA dashboards directly from saved assessment
    # Audits, in processing order, without recalculating distances or reading
    # intermediate SVG panels. RSD, PCA, correlation, and outlier grids each
    # share their project-wide legend. ReportInput then combines stage metrics,
    # QA metrics, the validated configuration, package version, metadata, and
    # the asset manifest for comprehensive and brief Markdown reports plus the
    # configured PDF output. Without an output folder, ReportInput remains
    # available but no report files are generated.
    # =========================================================================
    # Dynamic QA Mapping
    qa_corr_metrics = {}
    stage_key_map = {
        "Intra-batch corrected": "intra_batch_correction",
        "Inter-batch corrected": "inter_batch_correction",
        "SERRF": "global_correction",
        "RUV-III": "global_correction",
        "WaveICA 2.0": "global_correction",
    }

    for stage_name, result in qa_results_dict.items():
        safe_key = stage_key_map.get(stage_name, "unknown_correction")
        qa_corr_metrics[safe_key] = result.audit.metrics

    # Assemble Final Metrics
    pipeline_metrics_objs = {
        "raw_dataset": stage_results["raw_dataset"].audit.metrics,
        "sample_missing_value_filtering": sample_mv_result.audit.metrics,
        "high_mv_feature_filtering": stage_results[
            "high_mv_feature_filtering"
        ].audit.metrics,
        "signal_correction": stage_results["signal_correction"].audit.metrics,
        "low_quality_feature_filtering": stage_results[
            "low_quality_feature_filtering"
        ].audit.metrics,
        "missing_value_imputation": stage_results[
            "missing_value_imputation"
        ].audit.metrics,
        "normalization": stage_results["normalization"].audit.metrics,
    }

    qa_metrics_objs = {
        "raw_dataset": qa_raw_result.audit.metrics,
        "high_mv_feature_filtering": qa_mv_filter_result.audit.metrics,
        **qa_corr_metrics,
        "low_quality_feature_filtering": (
            qa_low_quality_filter_result.audit.metrics
        ),
        "missing_value_imputation": qa_imp_result.audit.metrics,
        "normalization": qa_norm_result.audit.metrics,
    }

    report_input = ReportInput(
        pipeline_metrics=pipeline_metrics_objs,
        qa_metrics=qa_metrics_objs,
        metadata={
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "version": __version__,
            "mode": raw_result.audit.metrics.get("mode", "N/A"),
            "is_multi_batch": is_multi_batch_flag,
        },
        resolved_config=params,
    )
    report_generated = False
    if root_dir is not None:
        logger.info("Step 07: Sequential Audit Report Compilation...")
        report_dir = "07_Report_Summary"

        visual_rep = ru.VisualAssetReporter(base_dir=str(root_dir))
        # Preserve processing order and distinguish batch-correction stages.
        qa_audits = {
            (
                key.split("/", 1)[1]
                if key.startswith("signal_correction/")
                else (
                    result.audit.plot_payload.data.context.pipeline_stage
                    or key
                )
            ): result.audit
            for key, result in assessments.items()
            if not result.audit.skipped
        }
        asset_manifest = visual_rep.compile_assessor_report(
            audits=qa_audits,
            report_folder=report_dir,
            is_multi_batch=is_multi_batch_flag,
        )
        report_input = ReportInput(
            pipeline_metrics=pipeline_metrics_objs,
            qa_metrics=qa_metrics_objs,
            metadata=report_input.metadata,
            resolved_config=params,
            asset_manifest=asset_manifest,
        )

        logger.info(f"Initializing narrative reporter at workspace: {root_dir}")
        md_reporter = ru.NarrativeStatsReporter(base_dir=str(root_dir))
        md_reporter.generate_markdown(report_input, report_folder=report_dir)
        report_generated = bool(
            md_reporter.export_report(pdf_engine="weasyprint")
        )
    else:
        logger.info(
            "Skipping report compilation because no output directory was "
            "requested. All report inputs remain available in PipelineResult."
        )

    logger.success("PI-METABOQC PIPELINE COMPLETED SUCCESSFULLY.")
    pipeline_result = PipelineResult(
        stage_tables=stage_tables,
        stage_results=stage_results,
        assessments=assessments,
        pipeline_metrics=pipeline_metrics_objs,
        qa_metrics=qa_metrics_objs,
        report_input=report_input,
        output_dir=root_dir,
        report_generated=report_generated,
        filtering_results=filtering_results,
    )
    return pipeline_result
