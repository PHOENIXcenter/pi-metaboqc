"""Dashboard composition for quality-assessment diagnostics.

The module combines correlation, RSD, PCA, and outlier panels while adapting
the layout to the diagnostics available for each stage snapshot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from .. import plot_utils as pu


class AssessmentDashboardMixin:
    """Assemble the complete assessment dashboard."""

    def plot_assessment_dashboard(
        self,
        pca_res: dict[str, Any],
        rsd_data: dict[str, dict[str, int]],
        batch_corr: pd.DataFrame,
        corr_mat: pd.DataFrame,
        qc_mask: np.ndarray | None,
        batches: list[object] | pd.Index | np.ndarray,
        method: str,
        sample_type: str,
        batch: str,
        qc_label: str,
        actual_label: str,
        is_flags: pd.Series | None = None,
        orf_flags: pd.Series | None = None,
        sample_name: str = "Sample Name",
        target_param: str = "both",
    ) -> object | None:
        """Assemble the standard assessment dashboard.

        The panel set and legends adapt to available batch, reference-sample,
        and internal-standard metadata.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        has_outlier_bar = self._has_stat_outlier_bar_data(
            pca_res["outliers"],
            sample_type,
            actual_label,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )
        # Every compact data axes is four source units wide. The wider parent
        # bricks reserve local side lanes for heatmap/PCA/outlier guides while
        # preserving equal analytical-panel widths across both rows.
        compact_left_width = 4.8
        compact_right_width = 6.3
        layout_width = (
            14.0
            if has_outlier_bar
            else compact_left_width + compact_right_width
        )
        target_width = (
            pu.DASHBOARD_TARGET_WIDTH_IN
            if has_outlier_bar
            else pu.TWO_BY_TWO_DASHBOARD_TARGET_WIDTH_IN
        )

        def _brick_size(width: float, height: float) -> tuple[float, float]:
            return pu.dashboard_brick_size(
                width,
                height,
                layout_width,
                target_width=target_width,
            )

        def _content_axes(brick: object, width: float) -> plt.Axes:
            """Reserve a right-hand legend lane inside one parent brick."""
            brick.axis("off")
            return brick.inset_axes([0.0, 0.0, width, 1.0])

        def _bind_legends_to_axes(ax: plt.Axes | None) -> None:
            if ax is not None and hasattr(ax.figure, "legends"):
                axis_legend = ax.get_legend()
                artist_ids = {id(artist) for artist in ax.artists}
                for leg in list(ax.figure.legends):
                    if leg is axis_legend or id(leg) in artist_ids:
                        continue
                    ax.add_artist(leg)
                ax.figure.legends.clear()

        # Row 1 Assembly
        top_widths = (
            (4.8, 4.0, 5.2)
            if has_outlier_bar
            else (
                compact_left_width,
                compact_left_width,
                compact_right_width,
            )
        )
        ax1 = pw.Brick(
            figsize=_brick_size(top_widths[0], 4.0),
            label="correlation_heatmap",
        )
        ax_corr = _content_axes(
            ax1,
            width=(
                0.83
                if has_outlier_bar
                else 4.0 / compact_left_width
            ),
        )

        n_batches = batch_corr.shape[0] if batch_corr is not None else 0
        if n_batches <= 1:
            self.plot_qc_corr_heatmap(
                corr_matrix=corr_mat,
                corr_mask=qc_mask,
                batches=batches,
                method=method,
                cluster="none",
                ax=ax_corr,
            )
        else:
            self.plot_batch_corr_heatmap(
                batch_corr_matrix=batch_corr, method=method, ax=ax_corr
            )
        _bind_legends_to_axes(ax_corr)

        ax2 = pw.Brick(
            figsize=_brick_size(top_widths[1], 4.0),
            label="rsd_bar",
        )
        if has_outlier_bar:
            ax_rsd = ax2
        else:
            ax_rsd = _content_axes(
                ax2,
                width=4.0 / compact_left_width,
            )
        self.plot_rsd_bar(
            rsd_data=rsd_data,
            qc_label=qc_label,
            actual_label=actual_label,
            ax=ax_rsd,
        )
        _bind_legends_to_axes(ax_rsd)

        ax3 = pw.Brick(
            figsize=_brick_size(top_widths[2], 4.0),
            label="pca_scatter",
        )
        ax_pca = _content_axes(
            ax3,
            width=(
                0.77
                if has_outlier_bar
                else 4.0 / compact_right_width
            ),
        )

        self.plot_pca_scatter(
            pca_df=pca_res["pca_scatter"],
            pca_var=pca_res["pca_variance"],
            pca_diagnostics=pca_res["diagnostics"],
            sample_type=sample_type,
            batch=batch,
            qc_label=qc_label,
            actual_label=actual_label,
            ax=ax_pca,
        )
        _bind_legends_to_axes(ax_pca)

        # Row 2 Assembly
        ax4 = pw.Brick(
            figsize=_brick_size(
                4.0 if has_outlier_bar else compact_right_width,
                4.0,
            ),
            label="outlier_scatter",
        )
        if has_outlier_bar:
            ax_outlier = ax4
        else:
            ax4.axis("off")
            ax_outlier = ax4.inset_axes(
                [0.0, 0.0, 4.0 / compact_right_width, 1.0]
            )
        self.plot_sd_od_scatter(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax_outlier,
            show_legend=not has_outlier_bar,
            legend_mode="local" if not has_outlier_bar else "none",
            complete_legend_categories=not has_outlier_bar,
            annotate_thresholds=True,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )
        _bind_legends_to_axes(ax_outlier)

        if not has_outlier_bar:
            top_row = pw.hstack(
                ax1,
                ax3,
                margin=0.35,
                adjust_height=False,
                adjust_width=False,
            )
            bottom_row = pw.hstack(
                ax2,
                ax4,
                margin=0.35,
                adjust_height=False,
                adjust_width=False,
            )
            return pw.vstack(
                top_row,
                bottom_row,
                direction="b",
                margin=0.45,
                adjust_height=False,
                adjust_width=False,
            )

        ax5 = pw.Brick(
            figsize=_brick_size(8.8, 4.0),
            label="outlier_bar",
        )
        ax5.axis("off")
        ax5_top = ax5.inset_axes([0.0, 0.52, 1.0, 0.48])
        ax5_bot = ax5.inset_axes(
            [0.0, 0.0, 1.0, 0.48], sharex=ax5_top
        )
        self._plot_stat_outliers_bar(
            outliers_df=pca_res["outliers"],
            sample_type=sample_type,
            batch=batch,
            sample_name=sample_name,
            actual_label=actual_label,
            target_param=target_param,
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax1=ax5_top,
            ax2=ax5_bot,
            show_legend=False,
            is_flags=is_flags,
            orf_flags=orf_flags,
        )

        ax6 = pw.Brick(figsize=_brick_size(1.2, 4.0))
        self.plot_outlier_standalone_legend(
            metrics_df=pca_res["metrics_df"],
            sd_limit=pca_res["sd_limit"],
            od_limit=pca_res["od_limit"],
            ax=ax6,
            is_flags=is_flags,
            orf_flags=orf_flags,
            include_bar_diagnostics=True,
            include_thresholds=False,
        )
        _bind_legends_to_axes(ax6)

        return (ax1 | ax2 | ax3) / (ax4 | ax5 | ax6)
