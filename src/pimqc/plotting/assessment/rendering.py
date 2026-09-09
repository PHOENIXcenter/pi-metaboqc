"""Render assessment audits without a live processor or runner.

The native runner and independent callers share this panel export path. Stored
diagnostics determine the figures; rendering never executes an assessment.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .plotter import AssessmentPlotter

if TYPE_CHECKING:
    from ...processing.audit import AssessQualityAuditPayload


def render_assessment(
    audit: AssessQualityAuditPayload,
    output_dir: str | Path,
    *,
    legend_mode: str = "external",
    stage_label: str | None = None,
    include_dashboard: bool = True,
    include_reference_charts: bool = True,
    show_plot: bool = False,
) -> None:
    """Save existing QA panels and the standard single-dataset dashboard.

    The native assessment runner uses this same renderer. A stage label is
    applied to a detached plotting context and never changes the input audit.
    """
    if audit.skipped:
        return
    if audit.plot_payload is None:
        raise ValueError("Assessment plot payload is unavailable.")
    if stage_label is not None:
        source = audit.plot_payload.data
        data = source.with_intensity(
            source.intensity, context_updates={"pipeline_stage": stage_label}
        )
        audit = replace(
            audit, plot_payload=replace(audit.plot_payload, data=data)
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_dir = str(destination)
    if audit.plot_payload is None:
        raise ValueError("Assessment plot payload is unavailable.")
    plotter = AssessmentPlotter(audit.plot_payload)
    legend_mode = plotter._validate_legend_mode(legend_mode)
    title_mode = "stage" if legend_mode == "external" else "full"

    plotter.save_and_close_fig(
        fig=plotter.plot_qc_corr_heatmap(
            corr_matrix=audit.qc_correlation,
            corr_mask=audit.qc_correlation_mask,
            batches=audit.qc_batches,
            method=audit.correlation_method,
            cluster="none",
            show_colorbar=legend_mode != "external",
            title_mode=title_mode,
        ),
        file_path=os.path.join(output_dir, "QC_Correlation_Heatmap"),
        save_format=plotter.QA_PANEL_SAVE_FORMAT,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plotter.save_and_close_fig(
        fig=plotter.plot_batch_corr_heatmap(
            batch_corr_matrix=audit.batch_qc_correlation,
            method=audit.correlation_method,
            show_colorbar=legend_mode != "external",
            title_mode=title_mode,
        ),
        file_path=os.path.join(output_dir, "Batch_Correlation_Heatmap"),
        save_format=plotter.QA_PANEL_SAVE_FORMAT,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plotter.save_and_close_fig(
        fig=plotter.plot_pca_scatter(
            pca_df=audit.pca_results["pca_scatter"],
            pca_var=audit.pca_results["pca_variance"],
            pca_diagnostics=audit.pca_results["diagnostics"],
            sample_type=audit.sample_type_column,
            batch=audit.batch_column,
            qc_label=audit.qc_label,
            actual_label=audit.actual_label,
            legend_mode=legend_mode,
            title_mode=title_mode,
        ),
        file_path=os.path.join(output_dir, "PCA_Scatter_QC_Sample"),
        save_format=plotter.QA_PANEL_SAVE_FORMAT,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plotter.save_and_close_fig(
        fig=plotter.plot_sd_od_scatter(
            metrics_df=audit.pca_results["metrics_df"],
            sd_limit=audit.pca_results["sd_limit"],
            od_limit=audit.pca_results["od_limit"],
            is_flags=audit.internal_standard_flags,
            orf_flags=audit.outlier_reference_flags,
            show_legend=legend_mode == "local",
            legend_mode=legend_mode,
            title_mode=title_mode,
            annotate_thresholds=legend_mode == "external",
        ),
        file_path=os.path.join(output_dir, "Outlier_Scatter"),
        save_format=plotter.QA_PANEL_SAVE_FORMAT,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    plotter.save_and_close_fig(
        fig=plotter.plot_rsd_bar(
            rsd_data=audit.rsd_distribution,
            qc_label=audit.qc_label,
            actual_label=audit.actual_label,
            legend_mode=legend_mode,
            title_mode=title_mode,
        ),
        file_path=os.path.join(output_dir, "RSD_Barplot"),
        save_format=plotter.QA_PANEL_SAVE_FORMAT,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    if legend_mode == "external":
        corr_legend_prefix = (
            "Batch_Correlation_Heatmap"
            if audit.plot_payload.is_multi_batch
            else "QC_Correlation_Heatmap"
        )
        plotter.save_and_close_fig(
            fig=plotter.plot_correlation_colorbar_legend(
                method=audit.correlation_method
            ),
            file_path=os.path.join(
                output_dir,
                f"{corr_legend_prefix}_Legend",
            ),
            save_format=plotter.QA_LEGEND_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        plotter.save_and_close_fig(
            fig=plotter.plot_rsd_standalone_legend(
                qc_label=audit.qc_label,
                actual_label=audit.actual_label,
            ),
            file_path=os.path.join(output_dir, "RSD_Barplot_Legend"),
            save_format=plotter.QA_LEGEND_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        plotter.save_and_close_fig(
            fig=plotter.plot_pca_diagnostics_legend(
                pca_df=audit.pca_results["pca_scatter"],
                sample_type=audit.sample_type_column,
                batch=audit.batch_column,
                qc_label=audit.qc_label,
                actual_label=audit.actual_label,
            ),
            file_path=os.path.join(
                output_dir,
                "PCA_Scatter_QC_Sample_Legend",
            ),
            save_format=plotter.QA_LEGEND_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        plotter.save_and_close_fig(
            fig=plotter.plot_outlier_standalone_legend(
                metrics_df=audit.pca_results["metrics_df"],
                sd_limit=audit.pca_results["sd_limit"],
                od_limit=audit.pca_results["od_limit"],
                is_flags=audit.internal_standard_flags,
                orf_flags=audit.outlier_reference_flags,
                complete_categories=True,
                include_bar_diagnostics=False,
                include_thresholds=False,
            ),
            file_path=os.path.join(
                output_dir,
                "Outlier_Scatter_Legend",
            ),
            save_format=plotter.QA_LEGEND_SAVE_FORMAT,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    if include_reference_charts and audit.internal_standard_ids:
        is_grid = plotter.plot_ref_shewhart_chart(
            ref_data=audit.internal_standard_data,
            valid_feats=audit.internal_standard_ids,
            sample_type=audit.sample_type_column,
            batch=audit.batch_column,
            inject_order=audit.injection_order_column,
            qc_label=audit.qc_label,
            actual_label=audit.actual_label,
            bound_type=audit.boundary_type,
            ref_type="IS",
        )
        plotter.save_and_show_pw(
            pw_obj=is_grid,
            show_plot=False,
            file_path=os.path.join(output_dir, "IS_Shewhart_Chart"),
        )

    if include_reference_charts and audit.outlier_reference_ids:
        orf_grid = plotter.plot_ref_shewhart_chart(
            ref_data=audit.outlier_reference_data,
            valid_feats=audit.outlier_reference_ids,
            sample_type=audit.sample_type_column,
            batch=audit.batch_column,
            inject_order=audit.injection_order_column,
            qc_label=audit.qc_label,
            actual_label=audit.actual_label,
            bound_type=audit.boundary_type,
            ref_type="ORF",
        )
        plotter.save_and_show_pw(
            pw_obj=orf_grid,
            show_plot=False,
            file_path=os.path.join(output_dir, "ORF_Shewhart_Chart"),
        )

    if include_dashboard:
        dashboard = plotter.plot_assessment_dashboard(
            pca_res=audit.pca_results,
            rsd_data=audit.rsd_distribution,
            batch_corr=audit.batch_qc_correlation,
            corr_mat=audit.qc_correlation,
            qc_mask=audit.qc_correlation_mask,
            batches=audit.qc_batches,
            method=audit.correlation_method,
            sample_type=audit.sample_type_column,
            sample_name=audit.sample_name_column,
            batch=audit.batch_column,
            qc_label=audit.qc_label,
            actual_label=audit.actual_label,
            is_flags=audit.internal_standard_flags,
            orf_flags=audit.outlier_reference_flags,
        )
        grid_path = os.path.join(output_dir, "QA_Summary_Dashboard.svg")
        plotter.save_and_show_pw(
            pw_obj=dashboard, file_path=grid_path, show_plot=show_plot
        )

        logger.info(f"Assessor summary dashboard saved as: {grid_path}")
        logger.success("Data quality assessment completed.")
