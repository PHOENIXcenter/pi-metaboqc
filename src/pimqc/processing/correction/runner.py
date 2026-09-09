"""Export and visualize completed signal-correction results.

``CorrectionStageRunner`` delegates numerical work and candidate selection to
``SignalCorrector``, then writes only the selected matrices and renders the
stage-specific dashboard and internal-standard diagnostics.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from ...io import ensure_directory
from ..audit import CorrectionAuditPayload
from ..stage import StageResult, StageRunner
from .algorithms import (
    _format_correction_method_file_label,
    _format_correction_method_label,
)
from ...plotting.correction import CorrectionPlotter

if TYPE_CHECKING:
    from ...core import MetaboDataset

    # Resolve the quoted generic forward reference for static analyzers while
    # avoiding the runtime analysis/runner import cycle.


class CorrectionStageRunner(
    StageRunner["SignalCorrector", dict[str, "MetaboDataset"]]
):
    """Run correction while keeping artifact handling outside the processor."""

    def compute(self) -> StageResult[dict[str, MetaboDataset]]:
        """Evaluate candidates and return the selected correction stages."""
        return self.processor.transform_correction()

    def export(self, result: StageResult[dict[str, MetaboDataset]]) -> None:
        """Write selected correction stages and the fitted QC baseline."""
        assert self.output_dir is not None
        audit = result.require_audit(CorrectionAuditPayload)
        method = audit.selected_method
        file_label = _format_correction_method_file_label(method)

        # Export only the stages selected for propagation; the original matrix
        # remains available through the processor and is never duplicated.
        for stage_name, frame in result.data.items():
            clean_name = stage_name.replace("\n", " ")
            if method in {"SERRF", "RUV-III", "WaveICA 2.0"}:
                file_name = f"{method}.csv"
            else:
                prefix = clean_name.replace(" corrected", "")
                file_name = f"{prefix.replace(' ', '_')}_{file_label}.csv"
            frame.annotated_frame().to_csv(self.output_dir / file_name)

        predicted = audit.selected_prediction
        if predicted is not None:
            predicted.annotated_frame().to_csv(
                self.output_dir / f"QC_Fit_Base_{file_label}.csv"
            )

    def render(self, result: StageResult[dict[str, MetaboDataset]]) -> None:
        """Render the selected-method and candidate correction diagnostics."""
        assert self.output_dir is not None
        audit = result.require_audit(CorrectionAuditPayload)
        payload = audit.plot_payload
        plotter = CorrectionPlotter(payload)
        label = audit.selected_label
        method = audit.selected_method
        file_label = _format_correction_method_file_label(method)
        dashboard_label = file_label.replace(" ", "_")
        is_auto = audit.is_auto

        logger.info("Assembling correction diagnostic dashboard...")
        dashboard = plotter.plot_correction_dashboard(
            payload.candidate_results,
            label,
            include_auto_summary=(
                is_auto and len(payload.candidate_results) > 1
            ),
        )
        if dashboard is not None:
            path = self.output_dir / (
                f"Correction_Dashboard_{dashboard_label}.svg"
            )
            plotter.save_and_show_pw(
                pw_obj=dashboard,
                file_path=str(path),
            )

        if is_auto and len(payload.candidate_results) > 1:
            # Candidate comparison is an AUTO-only artifact; fixed-method runs
            # retain the smaller selected-method dashboard.
            candidate_dashboard = plotter.plot_correction_candidate_dashboard(
                results_store=payload.candidate_results,
                selected_method=label,
            )
            if candidate_dashboard is not None:
                path = self.output_dir / (
                    f"Correction_Candidate_Dashboard_{dashboard_label}.svg"
                )
                plotter.save_and_show_pw(
                    pw_obj=candidate_dashboard,
                    file_path=str(path),
                )
                logger.info(f"Correction candidate dashboard saved as: {path}")

        self._render_internal_standards(result, plotter, file_label)
        display_label = _format_correction_method_label(label)
        logger.success(f"Signal drift correction ({display_label}) completed.")

    def _render_internal_standards(
        self,
        result: StageResult[dict[str, MetaboDataset]],
        plotter: CorrectionPlotter,
        file_label: str,
    ) -> None:
        """Render internal-standard diagnostics using explicit stage context."""
        audit = result.require_audit(CorrectionAuditPayload)
        payload = audit.plot_payload
        if not payload.internal_standard_ids:
            return

        assert self.output_dir is not None
        method = audit.selected_method
        display_label = _format_correction_method_label(audit.selected_label)
        predicted = (
            payload.selected_prediction.annotated_frame()
            if payload.selected_prediction is not None
            else None
        )
        # Reconstruct the visual stage sequence without adding Original to the
        # exported StageResult payload.
        stage_dfs = {
            "Original": payload.source_data.annotated_frame(),
            **{
                name: dataset.annotated_frame()
                for name, dataset in payload.selected_stages.items()
            },
        }
        directory = ensure_directory(
            self.output_dir / "Internal_Standard_Scatters"
        )

        logger.info(f"Generating IS plots for {display_label}...")
        for feature, figure in plotter.plot_is_int_order_scatter(
            stage_dfs,
            predicted,
            payload.internal_standard_ids,
            audit.sample_type_column,
            audit.batch_column,
            audit.injection_order_column,
            audit.qc_label,
            audit.actual_label,
            payload.boundary_type,
        ):
            safe_feature = re.sub(r"[^a-zA-Z0-9]", "_", feature)
            path = directory / f"IS_Scatter_{safe_feature}_{file_label}.svg"
            plotter.save_and_show_pw(
                pw_obj=figure,
                file_path=str(path),
                show_plot=False,
            )

        if method in {"SERRF", "RUV-III", "WaveICA 2.0"} or predicted is None:
            logger.info(
                f"Bypassing IS baseline prediction for {display_label}."
            )
            return

        baseline = plotter.plot_pred_baseline_is(
            payload.source_data.annotated_frame(),
            predicted,
            payload.internal_standard_ids,
            audit.sample_type_column,
            audit.batch_column,
            audit.injection_order_column,
            audit.qc_label,
            audit.actual_label,
            method=method,
        )
        if baseline is not None:
            path = self.output_dir / f"Pred_Base_IS_{file_label}.svg"
            plotter.save_and_show_pw(
                pw_obj=baseline,
                file_path=str(path),
                show_plot=False,
            )
