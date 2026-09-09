"""Export and visualize completed normalization results.

The runner invokes the normalization transformation, records its selection
passport, writes the normalized matrix and optional AUTO summary, and delegates
the stage-specific dashboard to the normalization plotter.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from ...constants import DEFAULT_RANDOM_SEED
from ...plotting.payloads import (
    NormalizationPlotPayload,
    snapshot_dataset,
)
from ...core import MetaboDataset
from ...statistics import metrics as su
from ...statistics.normalization import calculate_rle_diagnostics
from ...statistics.sample_structure import calc_sample_structure_diagnostics
from ..audit import NormalizationAuditPayload
from ..stage import StageResult, StageRunner
from ...plotting.normalization import NormalizationPlotter


class NormalizationStageRunner(StageRunner["DataNormalizer", MetaboDataset]):
    """Run normalization without mixing calculations with artifact handling."""

    def compute(self) -> StageResult[MetaboDataset]:
        """Apply normalization and collect its candidate-selection passport."""
        requested = self.processor.config.get("norm_method", "ROBUST_LOG_ONLY")
        blank_count = len(self.processor.blank_data.columns)
        if blank_count:
            logger.info(f"Permanently dropping {blank_count} Blank samples.")
        logger.info(f"Applying Normalization | Method: {requested}")

        normalized_frame = self.processor.apply_normalization()
        normalized_frame.attrs["pipeline_stage"] = "Normalization"
        # Execution facts must not overwrite the next run's request settings.
        self.processor._normalization_output_attrs = dict(
            normalized_frame.attrs
        )
        selection = normalized_frame.attrs.get("selection", {})
        is_logged = bool(normalized_frame.attrs.get("is_logged", False))
        normalized = self.processor._to_dataset(
            normalized_frame,
            context_updates={
                "pipeline_stage": "Normalization",
                "is_logged": is_logged,
                "log_base": "2" if is_logged else "None",
                "is_scaled": False,
                "scale_method": "None",
            },
        )
        seed = int(
            self.processor.config.get("global_seed", DEFAULT_RANDOM_SEED)
        )
        raw = self.processor.dataset.annotated_frame()
        transformed = normalized.annotated_frame()
        qc_diagnostics = {}
        for label, frame in (("Before Norm", raw), ("After Norm", transformed)):
            log_data = su._extract_log2_target(frame)
            if log_data is None or log_data.empty:
                continue
            qc_cols = su._role_columns(frame, "QC sample", "QC").intersection(
                log_data.columns
            )
            variance = self.processor._calc_qc_variance_stabilization_values(
                log_data, qc_cols=qc_cols
            )
            qc_diagnostics[label] = {
                "variance": variance,
                "structure": self.processor._calc_qc_structure_values(
                    log_data, qc_cols=qc_cols, max_features=5000, seed=seed
                ),
            }
        rle = calculate_rle_diagnostics(
            [("Before Norm", raw), ("After Norm", transformed)]
        )
        for label, diagnostic in qc_diagnostics.items():
            diagnostic.update(rle.get(label, {}))
        sample_structure = calc_sample_structure_diagnostics(
            raw,
            transformed,
            seed=seed,
            scale_log_ratio_tol=self.processor._SAMPLE_SCALE_LOG_RATIO_TOL,
            scale_rel_delta_tol=self.processor._SAMPLE_SCALE_REL_DELTA_TOL,
        )
        # Keep the AUTO passport beside the matrix rather than coupling it to
        # the dashboard implementation.
        return StageResult(
            data=normalized,
            audit=NormalizationAuditPayload(
                metric_values=self.processor.normalization_metrics,
                output_suffix=self._output_suffix(normalized_frame),
                plot_payload=NormalizationPlotPayload(
                    raw_data=snapshot_dataset(self.processor.dataset),
                    normalized_data=snapshot_dataset(normalized),
                    sample_structure=sample_structure,
                    qc_diagnostics=qc_diagnostics,
                    selection=selection,
                    score_component_weights=dict(
                        self.processor._AUTO_SCORE_COMPONENT_WEIGHTS
                    ),
                    sample_scale_log_ratio_tolerance=(
                        self.processor._SAMPLE_SCALE_LOG_RATIO_TOL
                    ),
                    sample_scale_relative_delta_tolerance=(
                        self.processor._SAMPLE_SCALE_REL_DELTA_TOL
                    ),
                    global_seed=int(
                        self.processor.config.get(
                            "global_seed", DEFAULT_RANDOM_SEED
                        )
                    ),
                ),
            ),
        )

    @staticmethod
    def _output_suffix(normalized: pd.DataFrame) -> str:
        method = normalized.attrs.get("norm_method", "ROBUST_LOG_ONLY")
        parts = [method]
        if normalized.attrs.get("is_logged", False) and method.upper() not in {
            "VSN",
            "ROBUST_LOG_ONLY",
        }:
            parts.append("Log2")
        return "_".join(parts)

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Write the normalized matrix and optional AUTO summary."""
        audit = result.require_audit(NormalizationAuditPayload)
        suffix = audit.output_suffix
        result.data.annotated_frame().to_csv(
            self.output_dir / f"Normalized_Data_{suffix}.csv",
            na_rep="NA",
            encoding="utf-8-sig",
        )
        if audit.candidate_results:
            # Candidate tables are exported only for AUTO runs.
            summary_path = self.output_dir / "Normalization_Auto_Summary.csv"
            pd.DataFrame(audit.candidate_results).to_csv(
                summary_path,
                index=False,
                na_rep="NA",
                encoding="utf-8-sig",
            )
            logger.info(f"Auto normalization summary saved as: {summary_path}")

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Render the normalization dashboard from the completed result."""
        logger.info("Generating diagnostic plots for normalization...")
        audit = result.require_audit(NormalizationAuditPayload)
        plotter = NormalizationPlotter(audit.plot_payload)
        dashboard = plotter.plot_normalization_dashboard()
        if dashboard is None:
            return
        path = self.output_dir / (
            f"Normalization_Dashboard_{audit.output_suffix}.svg"
        )
        plotter.save_and_show_pw(pw_obj=dashboard, file_path=str(path))
        logger.info(f"Normalization summary dashboard saved as: {path}")
