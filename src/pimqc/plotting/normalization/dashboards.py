"""Dashboard composition and legends for normalization.

Standard dashboards, standalone legends, and retained experimental manuscript
layouts are assembled from the diagnostic and scorecard sibling modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from loguru import logger

from .. import plot_utils as pu


class NormalizationDashboardMixin:
    """Assemble standard and experimental normalization dashboards."""

    def plot_normalization_dashboard_legend(
        self,
        ax: plt.Axes,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
        layout_cols: int = 1,
        diagnostic_panels: set[str] | None = None,
        show_score_components: bool = True,
    ) -> plt.Axes | None:
        """Draw grouped score-component and stage legends for the dashboard."""
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH
        line_width = pu.DEFAULT_GUIDE_LINEWIDTH
        marker_size = pu.DEFAULT_LEGEND_MARKER_SIZE
        score_cols, label_map, color_map = (
            self._normalization_score_component_style()
        )
        score_handles = [
            mpatches.Patch(
                facecolor=color_map[col],
                edgecolor="k",
                linewidth=legend_linewidth,
                label=label_map[col],
            )
            for col in score_cols
        ]
        rle_handles = [
            mpatches.Patch(
                facecolor=self.pal["Before Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Before Norm",
            ),
            mpatches.Patch(
                facecolor=self.pal["After Norm"],
                edgecolor="k",
                linewidth=legend_linewidth,
                label="After Norm",
            ),
        ]
        variance_handles = [
            mlines.Line2D(
                [],
                [],
                color=self.pal["Before Norm"],
                linestyle="--",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [],
                [],
                color=self.pal["After Norm"],
                linestyle="-",
                marker="o",
                linewidth=line_width,
                markersize=marker_size,
                label="After Norm",
            ),
        ]
        distance_handles = [
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["Before Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="Before Norm",
            ),
            mlines.Line2D(
                [0],
                [0],
                color=self.pal["After Norm"],
                marker="o",
                linestyle="",
                markeredgecolor="k",
                markeredgewidth=0.5 if article_compact else 0.25,
                markersize=marker_size,
                label="After Norm",
            ),
        ]

        legend_groups = []
        if show_score_components:
            legend_groups.append(
                ("Normalization score components", score_handles)
            )
        for panel, title, handles in (
            ("QC_Alignment", "QC RLE alignment stage", rle_handles),
            ("QC_Variance", "QC variance stabilization stage", variance_handles),
            ("QC_Structure", "QC structure distance stage", distance_handles),
        ):
            if diagnostic_panels is None or panel in diagnostic_panels:
                legend_groups.append((title, handles))
        if not legend_groups:
            ax.set_visible(False)
            return None

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=legend_groups,
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.035,
            layout_cols=layout_cols,
            column_gap=0.12,
            max_item_rows=6,
            borderaxespad=0.0,
            handlelength=1.0 if article_compact else 1.8,
            handletextpad=0.3 if article_compact else 0.8,
            labelspacing=0.25 if article_compact else 0.5,
            borderpad=0.3 if article_compact else 0.4,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
        )
        if article_compact:
            self._apply_article_legend_style(
                ax=ax,
                fontsize=fontsize,
                title_fontsize=title_fontsize,
            )
        return ax

    def plot_normalization_article_legend(
        self,
        ax: plt.Axes,
        diagnostic_panels: set[str] | None = None,
        show_score_components: bool = True,
    ) -> plt.Axes | None:
        """
        Draw an experimental manuscript legend for normalization. This
        revision-oriented interface is excluded from the default pipeline.
        """
        return self.plot_normalization_dashboard_legend(
            ax=ax,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
            layout_cols=1,
            diagnostic_panels=diagnostic_panels,
            show_score_components=show_score_components,
        )

    @staticmethod
    def _compose_normalization_panels(
        panels: list[object], max_columns: int
    ) -> object | None:
        """Pack available panels into rows without empty placeholder bricks."""
        if not panels:
            return None
        if len(panels) == 1:
            import patchworklib as pw

            return pw.Bricks({panels[0].get_label(): panels[0]})
        row_count = (len(panels) + max_columns - 1) // max_columns
        column_count = (len(panels) + row_count - 1) // row_count
        rows = []
        for start in range(0, len(panels), column_count):
            row = panels[start]
            for panel in panels[start + 1 : start + column_count]:
                row = row | panel
            rows.append(row)
        dashboard = rows[0]
        for row in rows[1:]:
            dashboard = dashboard / row
        return dashboard

    def plot_normalization_dashboard(self) -> object | None:
        """
        Combine score-aligned normalization diagnostics into a PW dashboard.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping dashboard.")
            return None

        pw.clear()

        auto_summary = self.payload.selection.get("candidate_results")
        is_auto = bool(auto_summary)
        layout_width = 13.7 if is_auto else 8.0
        panel_size = (
            pu.dashboard_brick_size(3.0, 4.0, layout_width)
            if is_auto
            else pu.dashboard_brick_size(
                4.0,
                4.0,
                layout_width,
                target_width=pu.TWO_BY_TWO_DASHBOARD_TARGET_WIDTH_IN,
            )
        )
        diagnostics = {}
        for label, draw in (
            ("QC_Alignment", self._plot_qc_rle_boxplot),
            ("QC_Variance", self._plot_qc_variance_stabilization),
            ("QC_Structure", self._plot_qc_structure_improvement),
            ("Sample_Structure", self._plot_sample_structure_preservation),
        ):
            panel = pw.Brick(figsize=panel_size, label=label)
            if label == "Sample_Structure":
                result = draw(ax_geom=panel, compact_style=True)
            else:
                result = draw(
                    ax=panel, show_legend=not is_auto, article_compact=True
                )
            if result is not None:
                diagnostics[label] = panel

        if not is_auto:
            return self._compose_normalization_panels(
                list(diagnostics.values()), max_columns=2
            )

        summary_panels = []
        ax_auto = self.plot_normalization_score_summary(
            auto_summary=auto_summary,
            figsize=pu.dashboard_brick_size(6.2, 4.0, layout_width),
            show_legend=False,
        )
        if ax_auto is not None:
            summary_panels.append(ax_auto)

        ax_scorecard = pw.Brick(
            figsize=pu.dashboard_brick_size(4.9, 4.0, layout_width),
            label="Norm_Preservation_Scorecard",
        )
        scorecard = self.plot_normalization_preservation_scorecard(
            auto_summary=auto_summary, ax=ax_scorecard
        )
        if scorecard is not None and ax_scorecard.axison:
            summary_panels.append(ax_scorecard)
        else:
            ax_scorecard.set_visible(False)

        ax_legend = pw.Brick(
            figsize=pu.dashboard_brick_size(2.6, 4.0, layout_width),
            label="normalization_dashboard_legend",
        )
        if self.plot_normalization_dashboard_legend(
            ax=ax_legend,
            diagnostic_panels=set(diagnostics),
            show_score_components=ax_auto is not None,
        ) is not None:
            summary_panels.append(ax_legend)

        summary_row = self._compose_normalization_panels(
            summary_panels, max_columns=3
        )
        diagnostic_row = self._compose_normalization_panels(
            list(diagnostics.values()), max_columns=4
        )
        if summary_row is None:
            return diagnostic_row
        if diagnostic_row is None:
            return summary_row
        return summary_row / diagnostic_row

    def plot_normalization_article_dashboard(self) -> object | None:
        """Show AUTO selection and available score-aligned diagnostics.

        Two rows retain readable article typography: selection, QC RLE and
        shared legends above QC variance, QC structure and sample preservation.
        All diagnostics consume saved 1.5 payloads without fitting new models.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping normalization article panel."
            )
            return None

        auto_summary = self.payload.selection.get("candidate_results")
        if not auto_summary:
            return None

        pw.clear()
        # Three columns avoid squeezing six panels into one manuscript row.
        # The shared finalizer sets the total width without shrinking fonts.
        panel_height = max(pu.ARTICLE_PANEL_HEIGHT_IN, 2.1)
        panel_size = pu.article_brick_size(1.72, panel_height)

        summary_ax = self.plot_normalization_score_summary(
            auto_summary=auto_summary,
            figsize=panel_size,
            show_legend=False,
        )
        if summary_ax is not None:
            self._apply_article_panel_format(
                summary_ax,
                title="AUTO Normalization\nMethod Selection",
            )

        diagnostics = {}
        for key, label, title, draw in (
            (
                "QC_Alignment",
                "article_normalization_rle",
                "QC RLE\nAlignment Change",
                self._plot_qc_rle_boxplot,
            ),
            (
                "QC_Variance",
                "article_normalization_variance",
                "QC Variance\nStabilization",
                self._plot_qc_variance_stabilization,
            ),
            (
                "QC_Structure",
                "article_normalization_structure",
                "QC Structure\nDistance Change",
                self._plot_qc_structure_improvement,
            ),
            (
                "Sample_Structure",
                "article_normalization_sample_structure",
                "Sample Structure\nPreservation",
                self._plot_sample_structure_preservation,
            ),
        ):
            panel = pw.Brick(figsize=panel_size, label=label)
            if key == "Sample_Structure":
                result = draw(ax_geom=panel, compact_style=True)
            else:
                result = draw(ax=panel, show_legend=False, article_compact=True)
            if result is None:
                continue
            self._apply_article_panel_format(panel, title=title)
            if key == "QC_Structure":
                panel.set_ylabel("QC distance (log scale)")
            diagnostics[key] = panel

        legend_ax = pw.Brick(
            figsize=panel_size,
            label="article_normalization_legend",
        )
        legend = self.plot_normalization_article_legend(
            ax=legend_ax,
            diagnostic_panels=set(diagnostics),
            show_score_components=summary_ax is not None,
        )
        panels = [
            panel
            for panel in (
                summary_ax,
                diagnostics.get("QC_Alignment"),
                legend,
                diagnostics.get("QC_Variance"),
                diagnostics.get("QC_Structure"),
                diagnostics.get("Sample_Structure"),
            )
            if panel is not None
        ]
        dashboard = self._compose_normalization_panels(panels, max_columns=3)
        if dashboard is None:
            return None
        return self._finalize_article_dashboard(dashboard)
