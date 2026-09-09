"""Patchworklib QA grids shared by standalone comparisons and native reports.

Each grid reuses assessment panel primitives and a project-wide legend. All
scientific values come from typed audits; no intermediate SVGs are required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from matplotlib.patches import Patch

from .. import plot_utils as pu
from ..assembly import _get_optimal_cols
from .plotter import AssessmentPlotter

if TYPE_CHECKING:
    from ...processing.audit import AssessQualityAuditPayload


def _active_audits(audits):
    from ...processing.audit import AssessQualityAuditPayload

    active = {}
    for name, audit in audits.items():
        if not isinstance(name, str):
            raise TypeError("Assessment stage labels must be strings.")
        if not isinstance(audit, AssessQualityAuditPayload):
            raise TypeError(
                "audits must contain AssessQualityAuditPayload values."
            )
        if audit.skipped:
            continue
        if audit.plot_payload is None:
            raise ValueError(f"Assessment {name!r} has no plot payload.")
        active[name] = audit
    signatures = {
        (
            a.correlation_method,
            a.qc_label,
            a.actual_label,
            a.sample_type_column,
            a.batch_column,
        )
        for a in active.values()
    }
    if len(signatures) > 1:
        raise ValueError(
            "Comparison requires consistent correlation methods and "
            "sample-role labels/columns."
        )
    return active


def _plotters(audits):
    plotters = []
    for label, audit in audits.items():
        payload = audit.plot_payload
        data = payload.data.with_intensity(
            payload.data.intensity, context_updates={"pipeline_stage": label}
        )
        plotters.append(AssessmentPlotter(replace(payload, data=data)))
    # Filtering may remove a whole batch. A project-wide marker map keeps
    # remaining batches visually identical across every panel and its legend.
    batches = sorted(set(b for p in plotters for b in p.all_batches), key=str)
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
    styles = {
        b: markers[i] if len(batches) <= 10 else f"${i + 1}$"
        for i, b in enumerate(batches)
    }
    for plotter in plotters:
        plotter.all_batches = batches
        plotter.style_map = styles
    return plotters


def _bind_legends(ax, *, center=True):
    # Keep both grouped and ordinary legends on their carrier brick so
    # patchworklib moves and exports them together.
    for legend in list(ax.figure.legends):
        ax.add_artist(legend)
    ax.figure.legends.clear()
    ax.axis("off")
    from matplotlib.legend import Legend
    from matplotlib.transforms import Bbox

    legends = [artist for artist in ax.artists if isinstance(artist, Legend)]
    if ax.get_legend() is not None and ax.get_legend() not in legends:
        legends.append(ax.get_legend())
    if legends and center:
        ax.figure.canvas.draw()
        # Rachis visualizers may provide a lightweight FigureCanvasBase
        # without ``get_renderer``.  Materialize an Agg renderer solely for
        # legend measurement so patchworklib exports remain backend agnostic.
        canvas = ax.figure.canvas
        if not hasattr(canvas, "get_renderer"):
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            canvas = FigureCanvasAgg(ax.figure)
        renderer = canvas.get_renderer()
        bounds = Bbox.union(
            [item.get_window_extent(renderer) for item in legends]
        )
        bounds = bounds.transformed(ax.transAxes.inverted())
        dx, dy = (
            0.5 - (bounds.x0 + bounds.x1) / 2,
            0.5 - (bounds.y0 + bounds.y1) / 2,
        )
        for item in legends:
            anchor = item.get_bbox_to_anchor().transformed(
                ax.transAxes.inverted()
            )
            item.set_bbox_to_anchor(
                (anchor.x0 + dx, anchor.y0 + dy, anchor.width, anchor.height),
                transform=ax.transAxes,
            )


def plot_assessment_comparison(
    audits: Mapping[str, AssessQualityAuditPayload],
    diagnostic: str,
    *,
    cols: int | str = "auto",
    is_multi_batch: bool | None = None,
):
    """Build one same-diagnostic grid, with a final independent legend brick.

    Uses the same AssessmentPlotter panel methods as single-stage QA.
    Coordinates and diagnostic values come only from completed audits.
    Returned bricks are live patchworklib objects: save before building the
    next grid, since patchworklib owns a shared figure.
    """
    import patchworklib as pw

    active = _active_audits(audits)
    if not active:
        return None
    titles = {
        "rsd": "Feature RSD Distribution",
        "pca": "Pooled QC & Sample PCA Scatter",
        "correlation": "Inter-Batch Pooled QC Correlation"
        if is_multi_batch
        else "Pooled QCs Correlation",
        "outlier": "Integrated Outlier Diagnostics",
    }
    if diagnostic not in titles:
        raise ValueError(f"Unknown QA diagnostic: {diagnostic!r}")
    if cols != "auto" and (
        isinstance(cols, bool) or not isinstance(cols, int) or cols < 1
    ):
        raise ValueError("cols must be 'auto' or a positive integer.")
    if is_multi_batch is None:
        is_multi_batch = any(
            a.plot_payload.is_multi_batch for a in active.values()
        )
    titles["correlation"] = (
        "Inter-Batch Pooled QC Correlation"
        if is_multi_batch
        else "Pooled QCs Correlation"
    )
    plotters = _plotters(active)
    entries = list(active.values())
    n = len(entries)
    columns = _get_optimal_cols(n) if cols == "auto" else min(cols, n)
    rows = (n + columns - 1) // columns
    embedded = rows * columns > n
    layout_width = 4.0 * columns + (0.0 if embedded else 2.4)
    panel_size = pu.dashboard_brick_size(4.0, 4.0, layout_width)
    pw.clear()
    panels = []
    # Equal data-brick dimensions, independent of missing groups or legend size.
    for index, (plotter, audit) in enumerate(zip(plotters, entries)):
        brick = pw.Brick(
            figsize=panel_size, label=f"{diagnostic}_stage_{index}"
        )
        pca = audit.pca_results
        if diagnostic == "rsd":
            plotter.plot_rsd_bar(
                rsd_data=audit.rsd_distribution,
                qc_label=audit.qc_label,
                actual_label=audit.actual_label,
                ax=brick,
                legend_mode="external",
                title_mode="stage",
            )
        elif diagnostic == "pca":
            plotter.plot_pca_scatter(
                pca_df=pca["pca_scatter"],
                pca_var=pca["pca_variance"],
                pca_diagnostics=pca["diagnostics"],
                sample_type=audit.sample_type_column,
                batch=audit.batch_column,
                qc_label=audit.qc_label,
                actual_label=audit.actual_label,
                ax=brick,
                legend_mode="external",
                title_mode="stage",
            )
        elif diagnostic == "outlier":
            plotter.plot_sd_od_scatter(
                metrics_df=pca["metrics_df"],
                sd_limit=pca["sd_limit"],
                od_limit=pca["od_limit"],
                is_flags=audit.internal_standard_flags,
                orf_flags=audit.outlier_reference_flags,
                ax=brick,
                show_legend=False,
                legend_mode="external",
                title_mode="stage",
                annotate_thresholds=True,
            )
        elif is_multi_batch:
            plotter.plot_batch_corr_heatmap(
                batch_corr_matrix=audit.batch_qc_correlation,
                method=audit.correlation_method,
                ax=brick,
                show_colorbar=False,
                title_mode="stage",
            )
        else:
            plotter.plot_qc_corr_heatmap(
                corr_matrix=audit.qc_correlation,
                corr_mask=audit.qc_correlation_mask,
                batches=plotters[0].all_batches,
                method=audit.correlation_method,
                cluster="none",
                ax=brick,
                show_colorbar=False,
                title_mode="stage",
                show_batch_legend=False,
            )
        panels.append(brick)

    legend = pw.Brick(
        figsize=pu.dashboard_brick_size(
            4.0 if embedded else 2.4, 4.0, layout_width
        ),
        label=f"{diagnostic}_legend",
    )
    legend.axis("off")
    plotter, audit = plotters[0], entries[0]
    if diagnostic == "rsd":
        plotter.plot_rsd_standalone_legend(
            audit.qc_label, audit.actual_label, ax=legend
        )
    elif diagnostic == "pca":
        # Include the union of batches, even if absent from the first stage.
        scores = pd.concat([a.pca_results["pca_scatter"] for a in entries])
        plotter.plot_pca_diagnostics_legend(
            scores,
            audit.sample_type_column,
            audit.batch_column,
            audit.qc_label,
            audit.actual_label,
            ax=legend,
        )
    elif diagnostic == "outlier":
        plotter.plot_outlier_standalone_legend(
            audit.pca_results["metrics_df"],
            None,
            None,
            ax=legend,
            complete_categories=True,
            include_bar_diagnostics=False,
            include_thresholds=False,
            has_is_features=any(bool(a.internal_standard_ids) for a in entries),
            has_orf_features=any(
                bool(a.outlier_reference_ids) for a in entries
            ),
        )
    else:
        plotter.plot_correlation_colorbar_legend(
            audit.correlation_method, ax=legend
        )
        if not is_multi_batch:
            colors = pu.extract_linear_cmap(
                cmap=pu.custom_linear_cmap(
                    ["white", pu.PRIMARY_ACCENT_COLOR], 100
                ),
                cmin=0.5,
                cmax=1.0,
                n_colors=len(plotter.all_batches),
            )
            legend.legend(
                handles=[
                    Patch(facecolor=color, edgecolor="k", label=str(batch))
                    for batch, color in zip(plotter.all_batches, colors)
                ],
                title="Batch",
                loc="center left",
                bbox_to_anchor=(0.5, 0.5),
                **plotter.LEGEND_KWARGS,
            )
    _bind_legends(legend, center=diagnostic != "correlation")

    if embedded:
        # The legend is the last occupied cell (including the usual 2 x 4).
        panels.append(legend)
        while len(panels) < rows * columns:
            spacer = pw.Brick(
                figsize=panel_size, label=f"{diagnostic}_space_{len(panels)}"
            )
            spacer.axis("off")
            panels.insert(-1, spacer)
    row_grids = []
    for i in range(0, len(panels), columns):
        row = panels[i]
        for panel in panels[i + 1 : i + columns]:
            row = pw.hstack(
                row, panel, margin=0.35, adjust_height=False, adjust_width=False
            )
        row_grids.append(row)
    grid = row_grids[0]
    for row in row_grids[1:]:
        grid = pw.vstack(
            grid,
            row,
            direction="b",
            margin=0.45,
            adjust_height=False,
            adjust_width=False,
        )
    if not embedded:
        grid = pw.hstack(
            grid,
            legend,
            margin=0.35,
            adjust_height=False,
            adjust_width=False,
            va="bottom",
        )
    grid.set_suptitle(
        titles[diagnostic],
        fontsize=pu.DEFAULT_TITLE_FONTSIZE,
        fontweight="bold",
    )
    return grid


def render_assessment_comparison(
    audits: Mapping[str, AssessQualityAuditPayload],
    output_dir: str | Path,
    *,
    cols: int | str = "auto",
    is_multi_batch: bool | None = None,
    show_plot: bool = False,
) -> dict[str, Path]:
    """Save four report-compatible QA grids directly from named audits.

    No intermediate SVGs are read, stitched, or deleted. A single active
    audit still produces four grids; use render_assessment for the usual
    single-stage mixed-diagnostic dashboard.
    """
    active = _active_audits(audits)
    if not active:
        return {}
    if is_multi_batch is None:
        is_multi_batch = any(
            a.plot_payload.is_multi_batch for a in active.values()
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    correlation = "Batch" if is_multi_batch else "QC"
    registry = {
        "01_QC_Sample_RSD_Dashboard": "rsd",
        "02_PCA_Scatter_Dashboard": "pca",
        f"03_{correlation}_Correlation_Dashboard": "correlation",
        "04_Outlier_Diagnosis_Dashboard": "outlier",
    }
    writer = AssessmentPlotter(next(iter(active.values())).plot_payload)
    assets = {}
    for name, diagnostic in registry.items():
        grid = plot_assessment_comparison(
            active, diagnostic, cols=cols, is_multi_batch=is_multi_batch
        )
        path = destination / f"{name}.svg"
        writer.save_and_show_pw(
            grid, file_path=str(path), save_format="svg", show_plot=show_plot
        )
        assets[name] = path
    return assets


# Legacy SVG-file compositor retained but disabled. Reports now consume audits.
# from ..assembly import stitch_svg_grids
# def compose_assessment_panels(
#     stage_directories: Mapping[str, str | Path],
#     output_dir: str | Path,
#     *,
#     is_multi_batch: bool = True,
#     cols: int | str = "auto",
#     cleanup_source_svgs: bool = False,
#     show_plot: bool = False,
# ) -> dict[str, Path]:
#     """Compose existing QA panels in mapping order using one shared registry.
#
#     This is also the report compiler's SVG path: it reuses exported panels
#     without recalculating diagnostics or rendering them a second time.
#     """
#     assets_path = Path(output_dir)
#     assets_path.mkdir(parents=True, exist_ok=True)
#     asset_manifest: dict[str, Path] = {}
#     if is_multi_batch:
#         corr_prefix = "03_Batch_Correlation_Dashboard"
#         corr_file = "Batch_Correlation_Heatmap.svg"
#         corr_title = "Inter-Batch Pooled QC Correlation"
#         logger.info("Multi-batch design detected. Assembling Batch Grid.")
#     else:
#         corr_prefix = "03_QC_Correlation_Dashboard"
#         corr_file = "QC_Correlation_Heatmap.svg"
#         corr_title = "Pooled QCs Correlation"
#         logger.info("Single-batch design detected. Assembling QC Grid.")
#
#     # Target files now expect the .svg extension natively
#     target_map = {
#         "01_QC_Sample_RSD_Dashboard": (
#             "RSD_Barplot.svg",
#             "RSD_Barplot_Legend.svg",
#             "Feature RSD Distribution",
#         ),
#         "02_PCA_Scatter_Dashboard": (
#             "PCA_Scatter_QC_Sample.svg",
#             "PCA_Scatter_QC_Sample_Legend.svg",
#             "Pooled QC & Sample PCA Scatter",
#         ),
#         corr_prefix: (
#             corr_file,
#             f"{Path(corr_file).stem}_Legend.svg",
#             corr_title,
#         ),
#         "04_Outlier_Diagnosis_Dashboard": (
#             "Outlier_Scatter.svg",
#             "Outlier_Scatter_Legend.svg",
#             "Integrated Outlier Diagnostics",
#         ),
#     }
#
#     for prefix, (target_file, legend_file, suptitle) in target_map.items():
#         # Output directly to the assets folder as an SVG grid
#         svg_out = assets_path / f"{prefix}.svg"
#         input_svgs = []
#         legend_svgs = []
#         for folder_path in stage_directories.values():
#             source_dir = Path(folder_path)
#             source_file = source_dir / target_file
#             if source_file.is_file():
#                 input_svgs.append(source_file)
#                 sidecar = source_dir / legend_file
#                 if sidecar.is_file():
#                     legend_svgs.append(sidecar)
#
#         if not input_svgs:
#             logger.warning(f"Skipped {prefix}: No source SVGs found.")
#             continue
#         # Execute SVG stitching
#         stitched = stitch_svg_grids(
#             svg_paths=input_svgs,
#             file_path=svg_out,
#             legend_paths=legend_svgs,
#             suptitle=suptitle,
#             cols=cols,
#             save_format="svg",
#             display_format="png",
#             show_plot=show_plot,
#         )
#         if stitched:
#             asset_manifest[prefix] = svg_out
#         if stitched and cleanup_source_svgs:
#             # The four QA panels and their sidecar legends are intermediate
#             # assets. Remove both vector formats after the final grid has
#             # been compiled so stale PDFs cannot remain beside the report.
#             cleanup_paths = []
#             for source_path in dict.fromkeys(input_svgs + legend_svgs):
#                 cleanup_paths.append(source_path)
#                 cleanup_paths.append(source_path.with_suffix(".pdf"))
#                 if target_file in {
#                     "QC_Correlation_Heatmap.svg",
#                     "Batch_Correlation_Heatmap.svg",
#                 }:
#                     # Assessment emits both correlation variants so the
#                     # same QA object can support single- and multi-batch
#                     # reports. Only one is stitched, but neither variant
#                     # should remain as an unreferenced intermediate.
#                     for stem in (
#                         "QC_Correlation_Heatmap",
#                         "Batch_Correlation_Heatmap",
#                     ):
#                         cleanup_paths.extend(
#                             [
#                                 source_path.parent / f"{stem}.svg",
#                                 source_path.parent / f"{stem}.pdf",
#                                 source_path.parent / f"{stem}_Legend.svg",
#                                 source_path.parent / f"{stem}_Legend.pdf",
#                             ]
#                         )
#             for source_path in dict.fromkeys(cleanup_paths):
#                 try:
#                     if source_path.exists():
#                         source_path.unlink()
#                 except OSError as exc:
#                     logger.warning(
#                         "Could not remove stitched source "
#                         f"{source_path}: {exc}"
#                     )
#     return asset_manifest
