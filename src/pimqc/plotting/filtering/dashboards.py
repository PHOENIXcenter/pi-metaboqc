"""Dashboard composition for both filtering stages.

Standard and experimental manuscript layouts are assembled here from the
diagnostic panels and filtering flowchart defined in sibling modules.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from .. import plot_utils as pu


class FilteringDashboardMixin:
    """Assemble standard and experimental filtering dashboards."""

    def plot_mv_filtering_dashboard(
        self,
        tracking_df: pd.DataFrame,
        active_base_tol: float,
        mnar_group_mv_tol: float | None = None,
        mnar_qc_mv_tol: float = 0.2,
        mnar_int_threshold: float | None = None,
        mnar_intensity_pct: float = 0.1,
    ) -> object | None:
        """Assemble the high-missing-value filtering dashboard.

        The layout adapts to biological-group metadata and combines sample
        filtering, rescue diagnostics, MAR eligibility, and the stage
        decision flowchart.
        """
        try:
            import patchworklib as pw
        except ImportError:
            return None

        pw.clear()

        # Initialize data copy and evaluate the three orthogonal layout states.
        df_curr = tracking_df.copy()
        has_group_info = self.payload.biological_groups_available and (
            "Max_Group_MV_Pct" in df_curr.columns
        )

        sample_track = self.audit_tables.get("sample_tracking", pd.DataFrame())
        sample_filter_status = self.payload.sample_filter_status
        feature_mv_skipped = (
            self.payload.stage_status == "skipped"
            or not self.payload.missing_values_detected
        )

        if feature_mv_skipped:
            logger.info(
                "Feature MV filtering was skipped; no dashboard is produced."
            )
            return None

        has_sample_diagnostic = (
            sample_filter_status == "completed" and not sample_track.empty
        )
        layout_width = 12.0 if has_group_info or has_sample_diagnostic else 8.0
        ax_sample = None
        if has_sample_diagnostic:
            ax_sample = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="sample_mv",
            )
            self._plot_sample_mv_stripplot(
                sample_track,
                self.payload.sample_mv_tolerance,
                ax=ax_sample,
                article_compact=True,
            )

        # Dynamic layout assembly based on biological grouping
        if has_group_info:
            # Layout A: With Groups (1+2 Top, 1+1+1 Bottom)

            # Flowchart ratio is 2 units wide to match the bottom 2 plots
            flow_width = 8.0 if ax_sample is not None else 12.0
            ax_flow = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    flow_width,
                    4.0 if ax_sample is not None else 2.65,
                    layout_width,
                ),
                label="flowchart",
            )
            self._plot_mv_filtering_flowchart(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=mnar_group_mv_tol,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=True,
                mnar_intensity_pct=mnar_intensity_pct,
                compact=True,
                margin_right=0.0,
            )

            # Subplot S1: Group Rescue Scatter
            ax_group_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s1",
            )
            self._plot_group_rescue_scatter(
                df_curr,
                "Max_Group_MV_Pct",
                "Min_Group_MV_Pct",
                mnar_group_mv_tol,
                active_base_tol,
                ax_group_rescue,
                "Group-level MNAR Rescue",
                article_compact=True,
            )
            # Cascade remaining features downward
            mask_group = df_curr["Stage1_Status"].str.contains("Group")
            df_curr = df_curr[~mask_group]

            # Subplot S2: QC Rescue Scatter
            ax_qc_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s2",
            )
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
                mnar_intensity_pct=mnar_intensity_pct,
                article_compact=True,
            )
            # Cascade remaining features downward
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram
            ax_base_check = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="s3",
            )
            self._plot_cutoff_histogram(
                df_curr,
                "Min_Group_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
                ["MAR", "INVALID"],
                ax_base_check,
                ("MAR Eligibility Check"),
                "Min Group-level MV (%)",
                article_compact=True,
            )

            # Column-first topology binding to enforce strict vertical alignment
            # Prevents width stretching caused by the axis-off flowchart
            if ax_sample is not None:
                col_left = ax_sample / ax_group_rescue
                col_right = ax_flow / (ax_qc_rescue | ax_base_check)
                return col_left | col_right
            return ax_flow / (ax_group_rescue | ax_qc_rescue | ax_base_check)

        else:
            # Layout B: No Groups. Keep the QC-only flowchart at its canonical
            # two-column width; do not stretch it when a Sample MV panel is
            # available.
            diagnostic_width = 6.0 if ax_sample is not None else 4.0

            # The flowchart spans the available feature-only or combined row.
            ax_flow = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    8.0,
                    4.0 if ax_sample is not None else 2.65,
                    layout_width,
                ),
                label="flowchart",
            )
            self._plot_mv_filtering_flowchart(
                df=df_curr,
                ax=ax_flow,
                mnar_group_mv_tol=None,
                mnar_qc_mv_tol=mnar_qc_mv_tol,
                active_base_tol=active_base_tol,
                has_group_info=False,
                mnar_intensity_pct=mnar_intensity_pct,
                compact=True,
            )

            # Subplot S2: QC Rescue Scatter (Acts as Step 1 here)
            ax_qc_rescue = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    diagnostic_width,
                    4.0,
                    layout_width,
                ),
                label="s2",
            )
            self._plot_qc_rescue_scatter(
                df_curr,
                mnar_qc_mv_tol,
                mnar_int_threshold,
                ax_qc_rescue,
                "QC-level MNAR Rescue",
                mnar_intensity_pct=mnar_intensity_pct,
                article_compact=True,
            )
            mask_qc = df_curr["Stage1_Status"].str.contains("QC")
            df_curr = df_curr[~mask_qc]

            # Subplot S3: Base Threshold Check Histogram (Acts as Step 2 here)
            ax_base_check = pw.Brick(
                figsize=pu.dashboard_brick_size(
                    diagnostic_width,
                    4.0,
                    layout_width,
                ),
                label="s3",
            )
            self._plot_cutoff_histogram(
                df_curr,
                "QC_MV_Pct",
                "Stage1_Status",
                active_base_tol,
                {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
                ["MAR", "INVALID"],
                ax_base_check,
                "QC-level MV Check",
                "QC-level MV (%)",
                article_compact=True,
                threshold_label="QC MV",
            )

            if ax_sample is not None:
                row_top = ax_sample | ax_flow
                return row_top / (ax_qc_rescue | ax_base_check)
            return ax_flow / (ax_qc_rescue | ax_base_check)

    # Manuscript-Only Filtering Dashboards
    def plot_high_mv_filter_article_dashboard(self) -> object | None:
        """Create a compact three-panel summary of high-MV feature screening.

        Experimental: The manuscript layout retains three decision diagnostics
        to classify group-rescued MNAR, QC-rescued MNAR, and MAR features. It
        is deliberately independent of the full Stage 1 dashboard so the
        standard report layout and its typography remain unchanged.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping article dashboard."
            )
            return None

        tracking_df = self.audit_tables.get("stage1_tracking", pd.DataFrame())
        if tracking_df.empty:
            logger.warning(
                "Stage 1 tracking data are unavailable for article export."
            )
            return None

        has_group_info = (
            "Max_Group_MV_Pct" in tracking_df.columns
            and tracking_df["Max_Group_MV_Pct"].notna().any()
        )
        if not has_group_info:
            logger.warning(
                "Group-level MNAR rescue is unavailable; skipping high-MV "
                "article dashboard."
            )
            return None

        mnar_int_threshold = self.payload.mnar_intensity_threshold
        active_base_tol = self.payload.active_base_tolerance
        mnar_group_mv_tol = self.payload.mnar_group_mv_tolerance
        mnar_qc_mv_tol = self.payload.mnar_qc_mv_tolerance
        mnar_intensity_pct = self.payload.mnar_intensity_percentile

        pw.clear()
        # Patchworklib adds fixed label/legend padding. This width yields an
        # approximately 17.7 cm export, within the ACS double-column limit.
        panel_size = pu.article_brick_size(1.72, 1.72)
        ax_group = pw.Brick(figsize=panel_size, label="article_group_rescue")
        ax_qc = pw.Brick(figsize=panel_size, label="article_qc_rescue")
        ax_mar = pw.Brick(figsize=panel_size, label="article_mar_eligibility")

        self._plot_group_rescue_scatter(
            tracking_df,
            "Max_Group_MV_Pct",
            "Min_Group_MV_Pct",
            mnar_group_mv_tol,
            active_base_tol,
            ax_group,
            "Group-level MNAR Rescue",
            article_compact=True,
        )
        self._apply_article_panel_format(ax_group, "Group-level MNAR Rescue")

        after_group = tracking_df[
            ~tracking_df["Stage1_Status"].str.contains("Group", na=False)
        ]
        self._plot_qc_rescue_scatter(
            after_group,
            mnar_qc_mv_tol,
            mnar_int_threshold,
            ax_qc,
            "QC-level MNAR Rescue",
            mnar_intensity_pct=mnar_intensity_pct,
            article_compact=True,
        )
        self._apply_article_panel_format(ax_qc, "QC-level MNAR Rescue")

        after_qc = after_group[
            ~after_group["Stage1_Status"].str.contains("QC", na=False)
        ]
        self._plot_cutoff_histogram(
            after_qc,
            "Min_Group_MV_Pct",
            "Stage1_Status",
            active_base_tol,
            {"MAR": pu.PRIMARY_ACCENT_COLOR, "INVALID": pu.NEUTRAL_COLOR},
            ["MAR", "INVALID"],
            ax_mar,
            "MAR Eligibility Check",
            "Min group MV (%)",
            article_compact=True,
        )
        self._apply_article_panel_format(ax_mar, "MAR Eligibility Check")

        return ax_group | ax_qc | ax_mar

    def plot_quality_filtering_dashboard(self) -> object | None:
        """Assemble the low-quality feature-filtering dashboard.

        The dashboard contains three panels when blank samples are available
        and omits the blank/QC diagnostic otherwise.
        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning("patchworklib not found. Skipping dashboard.")
            return None

        pw.clear()

        # Detect if Blank data exists to determine the topology
        blank_mean = self.audit_tables.get("blank_mean")
        has_blanks = blank_mean is not None and not blank_mean.empty

        idx_mar = self.audit_tables.get("idx_mar", pd.Index([]))
        if self.audit_tables.get("quality_filter_mode") == "quality_only":
            # Quality-only runs have no MAR labels.  The QC-RSD diagnostic
            # still covers every feature that reached the quality stage.
            qc_rsd_all = self.audit_tables.get("qc_rsd_all")
            if qc_rsd_all is not None:
                idx_mar = qc_rsd_all.index

        # Topology A: 1x3 Grid (Blank samples exist)
        if has_blanks:
            layout_width = 12.0
            ax1 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_blank",
            )
            ax2 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_rsd",
            )
            ax3 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="retention",
            )

            self._plot_qc_blank_scatter(ax=ax1, article_compact=True)
            self._plot_rsd_dist(idx_mar=idx_mar, ax=ax2, article_compact=True)
            self._plot_retained_count_steps(ax=ax3, article_compact=True)

            return ax1 | ax2 | ax3

        # Topology B: 1x2 Grid (No Blank samples)
        else:
            layout_width = 8.0
            ax2 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="qc_rsd",
            )
            ax3 = pw.Brick(
                figsize=pu.dashboard_brick_size(4.0, 4.0, layout_width),
                label="retention",
            )

            self._plot_rsd_dist(idx_mar=idx_mar, ax=ax2, article_compact=True)
            self._plot_retained_count_steps(ax=ax3, article_compact=True)

            return ax2 | ax3

    def plot_low_quality_filter_article_dashboard(self) -> object | None:
        """
        Create a compact three-panel summary of low-quality feature filtering.

        Experimental: The QC-RSD panel reuses the MAR-only distribution used by
        the filtering engine. MNAR features remain absent from this diagnostic
        because they are exempt from the QC-RSD reproducibility filter.

        """
        try:
            import patchworklib as pw
        except ImportError:
            logger.warning(
                "patchworklib not found. Skipping article dashboard."
            )
            return None

        blank_mean = self.audit_tables.get("blank_mean")
        idx_mar = self.audit_tables.get("idx_mar", pd.Index([]))
        if self.audit_tables.get("quality_filter_mode") == "quality_only":
            qc_rsd_all = self.audit_tables.get("qc_rsd_all")
            if qc_rsd_all is not None:
                idx_mar = qc_rsd_all.index
        if blank_mean is None or blank_mean.empty or len(idx_mar) == 0:
            logger.warning(
                "Blank/QC and MAR QC-RSD inputs are required for the "
                "article dashboard."
            )
            return None

        pw.clear()
        # Compensate for the smaller low-quality layout margin so the exported
        # dashboard matches the high-MV article dashboard at approximately 17.7
        # cm.
        panel_size = pu.article_brick_size(1.72, 1.72)
        ax_blank = pw.Brick(figsize=panel_size, label="article_blank_qc")
        ax_rsd = pw.Brick(figsize=panel_size, label="article_qc_rsd")
        ax_retention = pw.Brick(
            figsize=panel_size, label="article_feature_retention"
        )

        self._plot_qc_blank_scatter(
            ax=ax_blank,
            article_compact=True,
            legend_inside=True,
        )
        self._apply_article_panel_format(ax_blank, "Blank/QC Check")

        self._plot_rsd_dist(
            idx_mar=idx_mar,
            ax=ax_rsd,
            article_compact=True,
        )
        self._apply_article_panel_format(ax_rsd, "QC-RSD Check")

        self._plot_retained_count_steps(
            ax=ax_retention,
            article_compact=True,
        )
        self._apply_article_panel_format(
            ax_retention, "Feature Retention Across Filtering Steps"
        )

        return ax_blank | ax_rsd | ax_retention
