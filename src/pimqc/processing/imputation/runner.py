"""Export and visualize completed missing-value imputation results.

The runner keeps benchmark arrays and candidate metrics in ``StageResult`` and
uses them to render the selected method or AUTO diagnostics. The processor
therefore remains responsible for transformation rather than file operations.
"""

from __future__ import annotations

from loguru import logger

from ..audit import ImputationAuditPayload
from ..stage import StageResult, StageRunner
from ...plotting.imputation import ImputationPlotter
from ...core import MetaboDataset


class ImputationStageRunner(StageRunner["MissingValueImputer", MetaboDataset]):
    """Run imputation while keeping calculation artifacts in StageResult."""

    def compute(self) -> StageResult[MetaboDataset]:
        """Compute the completed matrix and retain benchmark arrays."""
        return self.processor.transform_imputation()

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Write the completed matrix using the selected method label."""
        assert self.output_dir is not None
        audit = result.require_audit(ImputationAuditPayload)
        filename = (
            "Imputed_Data_NotRequired.csv"
            if audit.skipped
            else f"Imputed_Data_{audit.selected_label}.csv"
        )
        result.data.annotated_frame().to_csv(self.output_dir / filename)

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Render selected-method and candidate diagnostics."""
        audit = result.require_audit(ImputationAuditPayload)
        if audit.skipped or not audit.mar_feature_count:
            return
        assert self.output_dir is not None
        logger.info("Generating diagnostic plots for imputation...")
        if audit.plot_payload is None:
            raise ValueError("Imputation plot payload is unavailable.")
        plotter = ImputationPlotter(audit.plot_payload)
        selected_method = audit.selected_method
        selected_label = audit.selected_label
        renderable = {
            method: candidate
            for method, candidate in audit.candidate_results.items()
            if candidate[0].get("status", "ok") == "ok"
            and len(candidate[1]) > 0
            and len(candidate[2]) > 0
        }
        if selected_method not in renderable:
            logger.info(
                "No supported masked benchmark for the selected imputer; "
                "skipping its benchmark dashboard. See the audit instead."
            )
            return
        if audit.is_auto:
            # Failed candidates remain in the audit, not empty plot panels.
            dashboard = plotter.plot_imputation_auto_dashboard(
                renderable,
                selected_method=selected_method,
            )
        else:
            metrics, true_values, predicted_values = renderable[
                selected_method
            ]
            dashboard = plotter.plot_imputation_method_dashboard(
                metrics=metrics,
                true_vals=true_values,
                pred_vals=predicted_values,
                method_name=selected_label,
            )
        if dashboard is not None:
            path = self.output_dir / (
                f"Imputation_Dashboard_{selected_label}.svg"
            )
            plotter.save_and_show_pw(
                pw_obj=dashboard,
                file_path=str(path),
            )
            logger.info(f"Imputation dashboard saved as: {path}")

        appendix = (
            # The appendix requires the complete candidate cache and therefore
            # has no meaningful fixed-method equivalent.
            plotter.plot_imputation_nrmse_appendix_dashboard(
                renderable
            )
            if audit.is_auto and audit.has_candidate_cache
            else None
        )
        if appendix is not None:
            path = self.output_dir / (
                f"Imputation_Candidate_Dashboard_{selected_label}.svg"
            )
            plotter.save_and_show_pw(
                pw_obj=appendix,
                file_path=str(path),
            )
            logger.info(f"Imputer candidate NRMSE grid saved as: {path}")
