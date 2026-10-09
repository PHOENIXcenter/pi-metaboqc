"""Dashboard composition and legends for signal correction.

Standard dashboards, standalone legends, candidate appendices, and retained
experimental manuscript layouts are assembled from sibling panel modules.
"""

from __future__ import annotations

from collections.abc import Callable
from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ...processing.correction.algorithms import _format_correction_method_label
from .. import annotation_layout as al
from .. import plot_utils as pu
from ..sample_structure import plot_sample_structure_change_map

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


class CorrectionDashboardMixin:
    """Assemble standard and experimental correction dashboards."""

    def plot_correction_dashboard_legend(
        self,
        ax: plt.Axes,
        show_cv: bool = True,
        fontsize: float = pu.DEFAULT_LEGEND_FONTSIZE,
        title_fontsize: float = pu.DEFAULT_LEGEND_TITLE_FONTSIZE,
        article_compact: bool = False,
    ) -> plt.Axes:
        """Draw grouped score-component and correction-mode legends."""
        import matplotlib.patches as mpatches

        legend_linewidth = pu.DEFAULT_AXIS_LINEWIDTH

        score_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Technical precision (QC-RSD)",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=0.33
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Biological-variation preservation\n(D-ratio)",
            ),
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=0.6),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Sample-structure preservation",
            ),
        ]

        mode_handles = [
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex("tab:gray", alpha=1.0),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Baseline",
            )
        ]
        if show_cv:
            mode_handles.append(
                mpatches.Patch(
                    facecolor=pu.get_equivalent_hex(
                        pu.PRIMARY_ACCENT_COLOR, alpha=0.33
                    ),
                    edgecolor="k",
                    linewidth=legend_linewidth,
                    linestyle="--",
                    label="OOF model",
                )
            )
        mode_handles.append(
            mpatches.Patch(
                facecolor=pu.get_equivalent_hex(
                    pu.PRIMARY_ACCENT_COLOR, alpha=1.0
                ),
                edgecolor="k",
                linewidth=legend_linewidth,
                label="Full model",
            )
        )

        self._plot_grouped_standalone_legends(
            ax=ax,
            legend_groups=[
                ("Correction score components", score_handles),
                (
                    "Diagnostic evaluation stage"
                    if article_compact else "QC-RSD evaluation stage",
                    mode_handles,
                ),
            ],
            loc="upper left",
            start_bbox=(0.0, 1.0),
            row_gap=0.04,
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

    def plot_correction_article_legend(
        self,
        ax: plt.Axes,
        show_oof: bool,
    ) -> plt.Axes:
        """Draw an experimental manuscript legend for correction.

        This revision-oriented interface is excluded from the default pipeline.
        """
        return self.plot_correction_dashboard_legend(
            ax=ax,
            show_cv=show_oof,
            fontsize=pu.ARTICLE_LEGEND_FONTSIZE,
            title_fontsize=pu.ARTICLE_LEGEND_TITLE_FONTSIZE,
            article_compact=True,
        )

    def plot_correction_article_dashboard(
        self,
        results_store: dict[str, dict[str, Any]],
        selected_method: str,
    ) -> object | None:
        """
        Create a compact two-row correction manuscript dashboard.

        Selection and preservation use the saved AUTO metrics. Selected-method
        diagnostics share the standard dashboard's OOF/full-model semantics.
        This interface is excluded from the default pipeline.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping correction article panel."
            )
            return None

        results_store = self._eligible_correction_results(
            results_store, context="correction article dashboard"
        )
        if selected_method not in results_store:
            return None

        pw.clear()
        selected_result = results_store[selected_method]
        panel_height = pu.ARTICLE_PANEL_HEIGHT_IN

        summary_ax = pw.Brick(
            figsize=pu.article_brick_size(1.85, panel_height),
            label="article_correction_summary",
        )
        self.plot_correction_score_summary(
            results_store=results_store,
            selected_method=selected_method,
            ax=summary_ax,
            show_legend=False,
        )
        self._apply_article_panel_format(
            summary_ax,
            title="AUTO Correction\nScore Components",
        )

        preservation_ax = pw.Brick(
            figsize=pu.article_brick_size(2.25, panel_height),
            label="article_correction_preservation_scorecard",
        )
        self.plot_correction_preservation_scorecard(
            results_store=results_store,
            selected_method=selected_method,
            ax=preservation_ax,
        )
        self._apply_article_panel_format(
            preservation_ax,
            title="Candidate Preservation\nScorecard",
        )

        rsd_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_correction_qc_rsd",
        )
        self.plot_corr_rsd(
            stage_dfs=selected_result["stage_dfs"],
            stage_oof_dfs=selected_result.get("stage_oof_dfs", {}),
            ax=rsd_ax,
            show_legend=False,
            article_compact=True,
        )
        self._apply_article_panel_format(
            rsd_ax,
            title="QC-RSD\nDistribution",
        )

        ecdf_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_correction_featurewise",
        )
        self.plot_featurewise_qc_rsd_improvement_ecdf(
            result=selected_result,
            ax=ecdf_ax,
            article_compact=True,
        )
        self._apply_article_panel_format(
            ecdf_ax,
            title="Featurewise QC-RSD\nImprovement",
        )
        ecdf_ax.set_xlabel("QC-RSD relative\nimprovement")
        ecdf_ax.set_ylabel("Cumulative fraction")

        d_ratio_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_correction_d_ratio",
        )
        self.plot_canonical_d_ratio_distribution(
            result=selected_result,
            ax=d_ratio_ax,
            show_legend=False,
            box_width_fraction=0.38 / np.ptp(rsd_ax.get_xlim()),
        )
        self._apply_article_panel_format(
            d_ratio_ax,
            title="Canonical D-ratio\nDistribution",
        )

        sample_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_correction_sample_structure",
        )
        plot_sample_structure_change_map(
            ax=sample_ax,
            diagnostics=selected_result.get("sample_structure", {}),
            title="Sample Structure Change Map",
            compact_style=True,
        )
        self._apply_article_panel_format(
            sample_ax,
            title="Sample Structure\nChange Map",
        )
        # The four-column article row cannot fit the standard full metric
        # labels. Keep every saved value and the same font size, then place
        # the shorter note against the final (width-adjusted) axes geometry.
        structure_notes = []
        for text in sample_ax.texts:
            if any(label in text.get_text() for label in (
                "Global T(k):", "Distance-rank preservation:",
                "Distance-scale preservation:",
            )):
                text.set_text(
                    text.get_text()
                    .replace("Distance-rank preservation:", "Rank:")
                    .replace("Distance-scale preservation:", "Scale:")
                )
                structure_notes.append(text)

        legend_ax = pw.Brick(
            figsize=pu.article_brick_size(1.40, panel_height),
            label="article_correction_legend",
        )
        self.plot_correction_article_legend(
            ax=legend_ax,
            show_oof=bool(selected_result.get("stage_oof_dfs")),
        )
        selection_row = summary_ax | preservation_ax | legend_ax
        diagnostic_row = rsd_ax | ecdf_ax | d_ratio_ax | sample_ax
        dashboard = self._finalize_article_dashboard(
            selection_row / diagnostic_row
        )
        samples = selected_result.get("sample_structure", {}).get("samples")
        occupancy = (
            [samples[["scale_shift", "rank_rho"]].to_numpy()]
            if samples is not None and not samples.empty else None
        )
        for text in structure_notes:
            al.place_annotation_with_legend_awareness(
                ax=sample_ax,
                text_artist=text,
                occupancy_arrays=occupancy,
                expand_axes=False,
            )
        return dashboard

    def plot_correction_dashboard(
        self,
        results_store: dict[str, dict[str, Any]],
        selected_method: str,
        include_auto_summary: bool = True,
    ) -> object | None:
        """Combine correction selection and selected-method diagnostics."""
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        results_store = self._eligible_correction_results(
            results_store, context="correction dashboard"
        )
        if not results_store:
            return None

        row1 = None
        if include_auto_summary:
            layout_width = 12.3
            summary_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(5.0, 4.0, layout_width),
                label="correction_eval_summary",
            )
            self.plot_correction_score_summary(
                results_store=results_store,
                selected_method=selected_method,
                ax=summary_brick,
                show_legend=False,
            )
            structure_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(4.8, 4.0, layout_width),
                label="correction_preservation_scorecard",
            )
            self.plot_correction_preservation_scorecard(
                results_store=results_store,
                selected_method=selected_method,
                ax=structure_brick,
            )
            legend_brick = pw.Brick(
                figsize=pu.dashboard_brick_size(2.5, 4.0, layout_width),
                label="correction_dashboard_legend",
            )
            self.plot_correction_dashboard_legend(ax=legend_brick)
            row1 = summary_brick | structure_brick | legend_brick

        if selected_method not in results_store:
            return row1

        selected_result = results_store[selected_method]
        layout_width = 12.3 if include_auto_summary else 8.0
        target_width = (
            pu.DASHBOARD_TARGET_WIDTH_IN
            if include_auto_summary
            else pu.TWO_BY_TWO_DASHBOARD_TARGET_WIDTH_IN
        )
        panel_width = layout_width / (4 if include_auto_summary else 2)
        selected_rsd = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width, 4.0, layout_width,
                target_width=target_width,
            ),
            label="selected_correction_qc_rsd",
        )
        self.plot_corr_rsd(
            stage_dfs=selected_result["stage_dfs"],
            stage_oof_dfs=selected_result.get("stage_oof_dfs", {}),
            ax=selected_rsd,
            show_legend=not include_auto_summary,
            article_compact=True,
        )
        selected_rsd.set_title(
            "QC-RSD Distribution",
            fontsize=pu.DEFAULT_TITLE_FONTSIZE,
            fontweight="bold",
        )

        featurewise_ecdf = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width, 4.0, layout_width,
                target_width=target_width,
            ),
            label="selected_featurewise_qc_rsd_ecdf",
        )
        self.plot_featurewise_qc_rsd_improvement_ecdf(
            result=selected_result,
            ax=featurewise_ecdf,
            article_compact=True,
        )

        d_ratio = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width, 4.0, layout_width,
                target_width=target_width,
            ),
            label="selected_correction_d_ratio",
        )
        self.plot_canonical_d_ratio_distribution(
            result=selected_result,
            ax=d_ratio,
            show_legend=not include_auto_summary,
            box_width_fraction=0.38 / np.ptp(selected_rsd.get_xlim()),
        )

        sample_structure = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width, 4.0, layout_width,
                target_width=target_width,
            ),
            label="selected_correction_sample_structure",
        )
        plot_sample_structure_change_map(
            ax=sample_structure,
            diagnostics=selected_result.get("sample_structure", {}),
            title="Sample Structure Change Map",
            compact_style=True,
        )

        if row1 is not None:
            row2 = selected_rsd | featurewise_ecdf | d_ratio | sample_structure
            return row1 / row2
        return (selected_rsd | featurewise_ecdf) / (
            d_ratio | sample_structure
        )

    def plot_correction_candidate_dashboard(
        self, results_store: dict[str, dict[str, Any]], selected_method: str
    ) -> object | None:
        """Compare final-stage selection evidence for eligible candidates.

        Evaluation colors identify held-out QC (OOF) or descriptive full-model
        evidence, not two outputs to compare within a candidate. Each metric
        follows its recorded evaluation basis. Ineligible candidates are
        omitted from every panel; full-fit diagnostics never replace required
        OOF.
        """
        try:
            import patchworklib as pw
        except ImportError:
            raise ImportError("patchworklib is required for this plot.")

        pw.clear()
        results_store = self._eligible_correction_results(
            results_store, context="correction candidate dashboard"
        )
        if not results_store:
            return None

        def candidate_score(name: str) -> float:
            score = results_store[name].get("auto_score")
            try:
                numeric = float(score)
            except (TypeError, ValueError):
                return float("-inf")
            return numeric if np.isfinite(numeric) else float("-inf")

        ordered = sorted(
            results_store,
            key=lambda name: (
                -candidate_score(name),
                name,
            ),
        )
        layout_width = 12.3
        panel_width = layout_width / 2
        panel_height = 4.0
        rsd_ax = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width, panel_height, layout_width
            ),
            label="candidate_qc_rsd_comparison",
        )
        d_ratio_ax = pw.Brick(
            figsize=pu.dashboard_brick_size(
                panel_width, panel_height, layout_width
            ),
            label="candidate_d_ratio_comparison",
        )

        def draw_comparison(
            ax: plt.Axes,
            rows: list[tuple[str, Any, str]],
            *,
            is_rsd: bool,
        ) -> None:
            values = []
            positions = []
            colors = []
            styles = []
            labels = []
            for row_number, (label, raw, basis) in enumerate(rows):
                center = row_number + 1
                labels.append(label)
                array = np.asarray(
                    [] if raw is None else raw, dtype=float
                )
                array = array[np.isfinite(array)]
                if array.size:
                    values.append(array * (100.0 if is_rsd else 1.0))
                    positions.append(center)
                    colors.append(
                        pu.get_equivalent_hex(
                            "tab:gray" if row_number == 0
                            else pu.PRIMARY_ACCENT_COLOR,
                            alpha=0.33 if basis == "oof" else 1.0,
                        )
                    )
                    styles.append("--" if basis == "oof" else "-")
                else:
                    ax.text(
                        center,
                        0.98,
                        "N/A",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                    )
            if values:
                box = ax.boxplot(
                    values,
                    positions=positions,
                    widths=0.38,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch, color, style in zip(
                    box["boxes"], colors, styles
                ):
                    patch.set_facecolor(color)
                    patch.set_edgecolor("k")
                    patch.set_linestyle(style)
                    patch.set_linewidth(pu.DEFAULT_AXIS_LINEWIDTH)
                for median in box["medians"]:
                    median.set_color("k")
            ax.set_xticks(range(1, len(rows) + 1), labels)
            ax.set_xlim(0.4, len(rows) + 0.6)
            self._apply_standard_format(
                ax,
                title=(
                    "Candidate QC-RSD Distribution"
                    if is_rsd
                    else "Candidate Canonical D-ratio Distribution"
                ),
                ylabel=(
                    "QC-RSD (%)" if is_rsd
                    else "Canonical D-ratio (%)"
                ),
                append_stage=False,
            )
            ax.tick_params(axis="x", length=0)

        baseline = results_store[ordered[0]]
        rsd_rows_data = [(
            "Baseline",
            baseline.get("stage_qc_rsd", {}).get("Original"),
            "baseline",
        )]
        d_ratio_rows_data = [(
            "Baseline", baseline.get("d_ratio_baseline"), "baseline"
        )]
        for method in ordered:
            result = results_store[method]
            prefix = "* " if method == selected_method else ""
            method_label = prefix + _format_correction_method_label(method)
            full_stages = result.get("stage_qc_rsd", {})
            oof_stages = result.get("stage_oof_qc_rsd", {})
            stages = result.get("stage_dfs") or full_stages
            final_stage = next(
                (name for name in reversed(stages) if name != "Original"),
                None,
            )
            suffix = (
                "\nInter-batch"
                if final_stage is not None
                and "inter-batch" in final_stage.replace("\n", " ").lower()
                else ""
            )
            label = fill(
                method_label,
                width=12,
                break_long_words=False,
                break_on_hyphens=False,
            ) + suffix
            validation = result.get("validation") or {}
            basis = validation.get("evaluation_basis")
            if basis is None:
                # Older saved payloads lack the explicit basis. Prefer their
                # recorded summary, then the available final-stage evidence.
                if "final_rsd_oof" in result:
                    try:
                        has_oof = bool(np.isfinite(result["final_rsd_oof"]))
                    except TypeError:
                        has_oof = False
                else:
                    has_oof = final_stage in oof_stages
                basis = "oof" if has_oof else "full_model"
            d_basis = result.get("d_ratio_evaluation_basis", basis)
            unavailable = (
                basis not in ("oof", "full_model")
                or result.get("eligible_for_auto") is False
                or validation.get("eligible_for_auto") is False
                or "auto_score" in result
                and not np.isfinite(candidate_score(method))
            )
            rsd = None
            d_ratio = None
            if not unavailable:
                rsd = (oof_stages if basis == "oof" else full_stages).get(
                    final_stage
                )
                if d_basis in ("oof", "full_model"):
                    d_ratio = result.get(
                        "d_ratio_current_oof" if d_basis == "oof"
                        else "d_ratio_current_full"
                    )
            rsd_rows_data.append((label, rsd, basis))
            d_ratio_rows_data.append((
                label, d_ratio, d_basis,
            ))

        draw_comparison(rsd_ax, rsd_rows_data, is_rsd=True)
        draw_comparison(d_ratio_ax, d_ratio_rows_data, is_rsd=False)
        legend_brick = pw.Brick(
            figsize=pu.dashboard_brick_size(
                layout_width, 0.55, layout_width
            ),
            label="correction_mode_legend",
        )
        self.plot_rsd_standalone_legend(
            ax=legend_brick,
            show_cv=True,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            legend_cols=3,
        )
        previous_margin = pw.param.get("margin", 0.5)
        try:
            pw.param["margin"] = 0.10
            panels = rsd_ax | d_ratio_ax
            return panels / legend_brick
        finally:
            pw.param["margin"] = previous_margin
