"""Dashboard composition and experimental layouts for imputation.

Standard AUTO/fixed-method dashboards, appendix panels, standalone legends,
and retained experimental manuscript layouts are assembled here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from .. import plot_utils as pu
from ..sample_structure import plot_sample_structure_change_map
from ...processing.imputation.methods import IMPUTATION_METHODS


class ImputationDashboardMixin:
    """Assemble standard and experimental imputation dashboards."""

    def _resolve_article_benchmark_item(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        selected_method: str,
    ) -> tuple[str, tuple[dict[str, float], np.ndarray, np.ndarray]] | None:
        """
        Return only the selected successful, nonempty masked benchmark.

        A missing or failed selected benchmark is not evidence for another
        method; use the same eligibility check as ImputationStageRunner.
        """
        selected_key = IMPUTATION_METHODS.canonicalize(
            selected_method, strict=False
        )
        for method_name, item in results_dict.items():
            if (
                IMPUTATION_METHODS.canonicalize(method_name, strict=False)
                == selected_key
            ):
                metrics, true_vals, pred_vals = item
                if (
                    metrics.get("status", "ok") != "ok"
                    or true_vals is None
                    or pred_vals is None
                    or len(true_vals) == 0
                    or len(pred_vals) == 0
                ):
                    return None
                return method_name, item
        return None

    def plot_imputation_reconstruction_article_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        selected_method: str,
    ) -> object | None:
        """Experimental compact AUTO selection manuscript panel.

        This article-oriented interface is retained for manuscript revision
        workflows and is not used by the standard pipeline dashboard.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping imputation article panel."
            )
            return None

        best_item = self._resolve_article_benchmark_item(
            results_dict, selected_method
        )
        if best_item is None:
            return None

        # Failed/final-fit-failed candidates remain in the Audit. Their stale
        # benchmark scores must not be shown as eligible AUTO candidates.
        renderable = {
            method: candidate
            for method, candidate in results_dict.items()
            if candidate[0].get("status", "ok") == "ok"
            and candidate[1] is not None
            and candidate[2] is not None
            and len(candidate[1]) > 0
            and len(candidate[2]) > 0
        }
        method_name, (metrics, true_vals, pred_vals) = best_item
        pw.clear()
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = pw.Brick(
            figsize=pu.article_brick_size(2.05, panel_height),
            label="article_imputation_summary",
        )
        self.plot_imputation_score_summary(
            results_dict=renderable,
            selected_method=selected_method,
            ax=summary_ax,
            show_legend=False,
        )
        self._apply_article_panel_format(
            summary_ax,
            title="AUTO Imputation\nScore Components",
        )

        scatter_ax = pw.Brick(
            figsize=pu.article_brick_size(2.05, panel_height),
            label="article_imputation_masked_nrmse",
        )
        self._plot_nrmse_scatter(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            method_name=IMPUTATION_METHODS.display_name(method_name),
            compact_title=False,
            show_method_in_title=False,
            show_colorbar=False,
            article_compact=True,
            ax=scatter_ax,
        )
        self._apply_article_panel_format(
            scatter_ax,
            title="Masked-Value\nReconstruction",
        )
        scatter_ax.set_xlabel("Known masked intensity (log2)")
        scatter_ax.set_ylabel("Reconstructed intensity (log2)")

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_imputation_score_legend",
        )
        self.plot_imputation_article_score_legend(ax=legend_ax)
        return self._finalize_article_dashboard(
            summary_ax | scatter_ax | legend_ax
        )

    def plot_imputation_preservation_article_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        selected_method: str,
    ) -> object | None:
        """Experimental compact fidelity manuscript panel.

        This article-oriented interface is retained for manuscript revision
        workflows and is not used by the standard pipeline dashboard.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping imputation article panel."
            )
            return None

        best_item = self._resolve_article_benchmark_item(
            results_dict, selected_method
        )
        if best_item is None:
            return None

        _, (metrics, true_vals, pred_vals) = best_item
        pw.clear()
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        density_ax = pw.Brick(
            figsize=pu.article_brick_size(2.05, panel_height),
            label="article_imputation_density",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=density_ax,
            compact_title=False,
            article_compact=True,
            show_legend=False,
        )
        self._apply_article_panel_format(
            density_ax,
            title="Masked-Value\nDistribution Fidelity",
        )

        sample_ax = pw.Brick(
            figsize=pu.article_brick_size(2.05, panel_height),
            label="article_imputation_sample_structure",
        )
        plot_sample_structure_change_map(
            ax=sample_ax,
            diagnostics=self.payload.sample_structure,
            title="Full-fit Sample Structure\nChange Map",
            compact_style=True,
        )
        self._apply_article_panel_format(
            sample_ax,
            title="Full-fit Sample Structure\nChange Map",
        )

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_imputation_density_legend",
        )
        self.plot_imputation_article_density_legend(ax=legend_ax)
        return self._finalize_article_dashboard(
            density_ax | sample_ax | legend_ax
        )

    def plot_imputation_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
    ) -> plt.Axes:
        """Draw score-component and masked-density legends in one panel."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        score_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Masked reconstruction",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.75),
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Distribution fidelity",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.45),
                edgecolor="k",
                linewidth=pu.DEFAULT_AXIS_LINEWIDTH,
                label="Sample structure preservation",
            ),
        ]
        density_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                label="Known masked values",
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                linestyle="-",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                label="Reconstructed values",
            ),
        ]

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Imputation score components", score_handles),
                ("Masked-value density reference", density_handles),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
            max_item_rows=6,
            borderaxespad=0.0,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        return ax

    def plot_imputation_article_score_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw an experimental manuscript score legend.

        This revision-oriented interface is excluded from the default pipeline.
        """
        return self.plot_imputation_score_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
        )

    def plot_imputation_article_density_legend(self, ax: plt.Axes) -> plt.Axes:
        """Draw an experimental manuscript density legend.

        This revision-oriented interface is excluded from the default pipeline.
        """
        import matplotlib.lines as mlines

        ax.axis("off")
        density_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=0.75,
                label="Known masked values",
            ),
            mlines.Line2D(
                [],
                [],
                color=pu.PRIMARY_ACCENT_COLOR,
                linestyle="-",
                linewidth=pu.DEFAULT_GUIDE_LINEWIDTH,
                label="Reconstructed values",
            ),
        ]
        ax.legend(handles=density_handles)
        self._format_single_legend(
            ax=ax,
            group_title="Masked-value density reference",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            max_item_rows=6,
            borderaxespad=0.0,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            handlelength=1.0,
            handletextpad=0.3,
            labelspacing=0.25,
            borderpad=0.3,
        )
        self._apply_article_legend_style(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
        )
        return ax

    def plot_imputation_auto_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
        selected_method: str,
    ) -> object | None:
        """Create the final R-route imputation Auto-selection dashboard."""
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "Module 'patchworklib' not found. Skipping dashboard."
            )
            return None

        if not results_dict:
            return None

        g_min, g_max = float("inf"), float("-inf")
        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: (
                float(item[1][0].get("Auto_Score", np.nan)),
                -float(item[1][0].get("NRMSE_Total", np.nan)),
                IMPUTATION_METHODS.display_name(item[0]),
            ),
            reverse=True,
        )

        for _, (_, true_vals, pred_vals) in sorted_items:
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)
        selected_key = IMPUTATION_METHODS.canonicalize(
            selected_method, strict=False
        )
        best_item = next(
            (
                item
                for item in sorted_items
                if (
                    IMPUTATION_METHODS.canonicalize(item[0], strict=False)
                    == selected_key
                )
            ),
            sorted_items[0],
        )

        pw.clear()
        layout_width = 12.5
        ax_summary = pw.Brick(
            figsize=pu.dashboard_brick_size(4.8, 4.0, layout_width),
            label="imputation_score_summary",
        )
        self.plot_imputation_score_summary(
            results_dict=results_dict,
            selected_method=selected_method,
            ax=ax_summary,
        )
        ax_scorecard = pw.Brick(
            figsize=pu.dashboard_brick_size(6.0, 4.0, layout_width),
            label="imputation_scorecard",
        )
        self.plot_imputation_preservation_scorecard(
            results_dict=results_dict,
            selected_method=selected_method,
            ax=ax_scorecard,
        )
        ax_legend = pw.Brick(
            figsize=pu.dashboard_brick_size(1.7, 4.0, layout_width),
            label="imputation_dashboard_legend",
        )
        self.plot_imputation_dashboard_legend(ax=ax_legend)

        method_name, (metrics, true_vals, pred_vals) = best_item
        ax_best_scatter = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="best_nrmse_scatter",
        )
        self._plot_nrmse_scatter(
            true_vals,
            pred_vals,
            metrics,
            method_name=IMPUTATION_METHODS.display_name(method_name),
            axis_lims=shared_lims,
            compact_title=False,
            show_method_in_title=False,
            article_compact=True,
            ax=ax_best_scatter,
        )

        ax_density = pw.Brick(
            figsize=pu.dashboard_brick_size(4.25, 4.0, layout_width),
            label="imputation_masked_density",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
            article_compact=True,
        )
        ax_sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.25, 4.0, layout_width),
            label="imputation_sample_structure_preservation",
        )
        plot_sample_structure_change_map(
            ax=ax_sample_structure,
            diagnostics=self.payload.sample_structure,
            title="Sample Structure Change Map",
            compact_style=True,
        )

        top_row = ax_summary | ax_scorecard | ax_legend
        diagnostic_row = ax_best_scatter | ax_density | ax_sample_structure

        return top_row / diagnostic_row

    def plot_imputation_method_dashboard(
        self,
        metrics: dict[str, float],
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        method_name: str,
    ) -> object | None:
        """
        Create a fixed-method imputation dashboard without Auto score panels.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "Module 'patchworklib' not found. Skipping dashboard."
            )
            return None

        if true_vals is None or pred_vals is None or len(true_vals) == 0:
            return None

        d_min = min(float(np.nanmin(true_vals)), float(np.nanmin(pred_vals)))
        d_max = max(float(np.nanmax(true_vals)), float(np.nanmax(pred_vals)))
        margin = (d_max - d_min) * 0.05 if d_max > d_min else 1.0
        shared_lims = (d_min - margin, d_max + margin)

        pw.clear()
        layout_width = 12.0
        ax_scatter = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="method_nrmse_scatter",
        )
        self._plot_nrmse_scatter(
            true_vals,
            pred_vals,
            metrics,
            method_name=IMPUTATION_METHODS.display_name(method_name),
            axis_lims=shared_lims,
            compact_title=False,
            show_method_in_title=False,
            article_compact=True,
            ax=ax_scatter,
        )

        ax_density = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="method_masked_density",
        )
        self._plot_masked_distribution_fidelity(
            true_vals=true_vals,
            pred_vals=pred_vals,
            metrics=metrics,
            ax=ax_density,
            compact_title=False,
            article_compact=True,
            show_legend=True,
        )

        ax_sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
            label="method_sample_structure_preservation",
        )
        plot_sample_structure_change_map(
            ax=ax_sample_structure,
            diagnostics=self.payload.sample_structure,
            title="Sample Structure Change Map",
            compact_style=True,
        )

        return ax_scatter | ax_density | ax_sample_structure

    def plot_imputation_nrmse_appendix_dashboard(
        self,
        results_dict: dict[
            str, tuple[dict[str, float], np.ndarray, np.ndarray]
        ],
    ) -> object | None:
        """Assemble the candidate reconstruction appendix dashboard.

        Up to six candidate panels are arranged in a two-by-three layout.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "Module 'patchworklib' not found. Skipping dashboard."
            )
            return None

        if not results_dict:
            return None

        sorted_items = sorted(
            results_dict.items(),
            key=lambda item: (
                float(item[1][0].get("Auto_Score", np.nan)),
                -float(item[1][0].get("NRMSE_Total", np.nan)),
                IMPUTATION_METHODS.display_name(item[0]),
            ),
            reverse=True,
        )
        g_min, g_max = float("inf"), float("-inf")
        for _, (_, true_vals, pred_vals) in sorted_items:
            g_min = min(g_min, true_vals.min(), pred_vals.min())
            g_max = max(g_max, true_vals.max(), pred_vals.max())

        margin = (g_max - g_min) * 0.05
        shared_lims = (g_min - margin, g_max + margin)
        best_key = IMPUTATION_METHODS.canonicalize(
            sorted_items[0][0], strict=False
        )

        pw.clear()
        scatter_bricks: list[object] = []
        for idx, (method_name, (metrics, true_vals, pred_vals)) in enumerate(
            sorted_items[:6]
        ):
            ax_scatter = pw.Brick(
                figsize=pu.dashboard_brick_size(3.6, 3.6, 10.8),
                label=f"nrmse_appendix_scatter_{idx + 1}",
            )
            display_method = IMPUTATION_METHODS.display_name(method_name)
            if (
                IMPUTATION_METHODS.canonicalize(method_name, strict=False)
                == best_key
            ):
                display_method = f"* {display_method}"
            self._plot_nrmse_scatter(
                true_vals,
                pred_vals,
                metrics,
                method_name=display_method,
                axis_lims=shared_lims,
                compact_title=False,
                ax=ax_scatter,
            )
            scatter_bricks.append(ax_scatter)

        while len(scatter_bricks) < 6:
            ax_blank = pw.Brick(
                figsize=pu.dashboard_brick_size(3.6, 3.6, 10.8),
                label=f"nrmse_appendix_scatter_blank_{len(scatter_bricks)}",
            )
            ax_blank.axis("off")
            scatter_bricks.append(ax_blank)

        return (scatter_bricks[0] | scatter_bricks[1] | scatter_bricks[2]) / (
            scatter_bricks[3] | scatter_bricks[4] | scatter_bricks[5]
        )
