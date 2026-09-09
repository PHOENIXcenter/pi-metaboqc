"""Responsive missing-value filtering decision flowcharts.

The flowchart topology is selected independently from the outer dashboard
layout.  Nodes and edges share one normalized coordinate system so changing a
Patchworklib Brick's physical width changes spacing without detaching arrows
from their node boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pandas as pd

from .. import plot_utils as pu


@dataclass(frozen=True)
class _FlowNode:
    """Describe one node in normalized flowchart coordinates."""

    key: str
    x: float
    y: float
    text: str
    color: str
    width: float
    height: float
    body_fontsize: float | None = None
    line_step: float | None = None


@dataclass(frozen=True)
class _FlowEdge:
    """Describe one directed connection between named nodes."""

    source: str
    target: str
    style: str = "horizontal"


class FilteringFlowchartMixin:
    """Render group-aware and QC-only filtering decision topologies."""

    def _plot_mv_filtering_flowchart(
        self,
        df: pd.DataFrame,
        ax: plt.Axes,
        mnar_group_mv_tol: float | None,
        mnar_qc_mv_tol: float,
        active_base_tol: float,
        has_group_info: bool,
        mnar_intensity_pct: float = 0.1,
        margin_left: float = 0.0,
        margin_right: float = 0.0,
        margin_top: float = 0.0,
        margin_bottom: float = 0.0,
        compact: bool = False,
    ) -> None:
        """Draw a responsive missingness-classification flowchart.

        ``has_group_info`` selects a grouped or QC-only topology.  Geometry is
        expressed on a 0-1 canvas and every arrow endpoint is derived from the
        final node boundary.  The outer dashboard can therefore use either a
        compact or full-width Brick without maintaining a second set of edge
        coordinates.
        """
        statuses = df["Stage1_Status"].astype(str)
        total = len(df)
        group_mask = statuses.str.contains("Group", na=False)
        count_group = int(group_mask.sum())
        after_group = df.loc[~group_mask]
        qc_mask = (
            after_group["Stage1_Status"]
            .astype(str)
            .str.contains("QC", na=False)
        )
        count_qc = int(qc_mask.sum())
        after_qc = after_group.loc[~qc_mask]
        count_mar = int((after_qc["Stage1_Status"] == "MAR").sum())
        count_inv = int((after_qc["Stage1_Status"] == "INVALID").sum())

        ax.axis("off")
        ax.set_xlim(0.0 - margin_left, 1.0 + margin_right)
        ax.set_ylim(0.0 - margin_bottom, 1.0 + margin_top)

        color_mar = pu.PRIMARY_ACCENT_COLOR
        color_mnar = pu.get_equivalent_hex(pu.PRIMARY_ACCENT_COLOR, alpha=0.5)
        color_inv = "tab:gray"
        color_pass = "white"
        node_fontsize = 7.0 if compact else (12.0 if has_group_info else 14.0)
        node_body_fontsize = 5.5 if compact else 10.0
        flow_linewidth = pu.DEFAULT_AXIS_LINEWIDTH if compact else 1.2
        arrow_linewidth = pu.DEFAULT_GUIDE_LINEWIDTH if compact else 2.0
        intensity_label = (
            f"QC intensity <= {pu.format_percentile_label(mnar_intensity_pct)}"
        )

        if has_group_info:
            group_tol = (
                active_base_tol
                if mnar_group_mv_tol is None
                else mnar_group_mv_tol
            )
            nodes = [
                _FlowNode(
                    "raw",
                    0.08,
                    0.50,
                    f"Raw Features\n(n={total})",
                    color_pass,
                    0.14,
                    0.17,
                ),
                _FlowNode(
                    "group_check",
                    0.29,
                    0.50,
                    "Group Rescue\n"
                    f"Max MV >= {group_tol * 100:.0f}%\n"
                    f"Min MV <= {active_base_tol * 100:.0f}%",
                    color_pass,
                    0.17,
                    0.22,
                ),
                _FlowNode(
                    "group_mnar",
                    0.29,
                    0.85,
                    f"MNAR Group\n(n={count_group})",
                    color_mnar,
                    0.15,
                    0.14,
                ),
                _FlowNode(
                    "qc_check",
                    0.50,
                    0.50,
                    "QC Rescue\n"
                    f"QC MV > {mnar_qc_mv_tol * 100:.0f}%\n"
                    f"{intensity_label}\n"
                    f"Min group MV <= {active_base_tol * 100:.0f}%",
                    color_pass,
                    0.18,
                    0.28,
                    body_fontsize=(
                        pu.DEFAULT_ANNOTATION_FONTSIZE if compact else 10.0
                    ),
                    line_step=0.047,
                ),
                _FlowNode(
                    "qc_mnar",
                    0.50,
                    0.85,
                    f"MNAR QC\n(n={count_qc})",
                    color_mnar,
                    0.15,
                    0.14,
                ),
                _FlowNode(
                    "eligibility",
                    0.72,
                    0.50,
                    "MAR Eligibility\n"
                    f"Min group MV <= {active_base_tol * 100:.0f}%",
                    color_pass,
                    0.17,
                    0.20,
                    body_fontsize=(
                        pu.DEFAULT_ANNOTATION_FONTSIZE if compact else 10.0
                    ),
                    line_step=0.047,
                ),
                _FlowNode(
                    "mar",
                    0.94,
                    0.73,
                    f"MAR\n(n={count_mar})",
                    color_mar,
                    0.12,
                    0.14,
                ),
                _FlowNode(
                    "invalid",
                    0.94,
                    0.27,
                    f"INVALID\n(n={count_inv})",
                    color_inv,
                    0.12,
                    0.14,
                ),
            ]
            edges = [
                _FlowEdge("raw", "group_check"),
                _FlowEdge("group_check", "qc_check"),
                _FlowEdge("group_check", "group_mnar", "vertical"),
                _FlowEdge("qc_check", "eligibility"),
                _FlowEdge("qc_check", "qc_mnar", "vertical"),
                _FlowEdge("eligibility", "mar", "step_h"),
                _FlowEdge("eligibility", "invalid", "step_h"),
            ]
        else:
            nodes = [
                _FlowNode(
                    "raw",
                    0.10,
                    0.50,
                    f"Raw Features\n(n={total})",
                    color_pass,
                    0.16,
                    0.17,
                ),
                _FlowNode(
                    "qc_check",
                    0.37,
                    0.50,
                    "QC Rescue\n"
                    f"QC MV > {mnar_qc_mv_tol * 100:.0f}%\n"
                    f"{intensity_label}",
                    color_pass,
                    0.20,
                    0.22,
                ),
                _FlowNode(
                    "qc_mnar",
                    0.37,
                    0.85,
                    f"MNAR QC\n(n={count_qc})",
                    color_mnar,
                    0.17,
                    0.14,
                ),
                _FlowNode(
                    "eligibility",
                    0.66,
                    0.50,
                    f"QC MV Check\nQC MV <= {active_base_tol * 100:.0f}%",
                    color_pass,
                    0.19,
                    0.18,
                ),
                _FlowNode(
                    "mar",
                    0.92,
                    0.73,
                    f"MAR\n(n={count_mar})",
                    color_mar,
                    0.14,
                    0.14,
                ),
                _FlowNode(
                    "invalid",
                    0.92,
                    0.27,
                    f"INVALID\n(n={count_inv})",
                    color_inv,
                    0.14,
                    0.14,
                ),
            ]
            edges = [
                _FlowEdge("raw", "qc_check"),
                _FlowEdge("qc_check", "eligibility"),
                _FlowEdge("qc_check", "qc_mnar", "vertical"),
                _FlowEdge("eligibility", "mar", "step_h"),
                _FlowEdge("eligibility", "invalid", "step_h"),
            ]

        rendered_nodes: dict[str, _FlowNode] = {}
        box_style = "round,pad=0.012,rounding_size=0.018"
        for node in nodes:
            patch = mpatches.FancyBboxPatch(
                (node.x - node.width / 2, node.y - node.height / 2),
                node.width,
                node.height,
                boxstyle=box_style,
                facecolor=node.color,
                edgecolor="k",
                linewidth=flow_linewidth,
                zorder=3,
                clip_on=False,
            )
            ax.add_patch(patch)
            rendered_nodes[node.key] = node

            lines = node.text.splitlines()
            line_step = node.line_step or (0.055 if has_group_info else 0.060)
            start_y = node.y + (len(lines) - 1) * line_step / 2
            text_color = pu.get_contrast_color(node.color)
            for line_index, line_text in enumerate(lines):
                is_title = line_index == 0
                ax.text(
                    node.x,
                    start_y - line_index * line_step,
                    line_text,
                    ha="center",
                    va="center",
                    multialignment="center",
                    fontsize=(
                        node_fontsize
                        if is_title
                        else (node.body_fontsize or node_body_fontsize)
                    ),
                    fontweight="semibold" if is_title else "normal",
                    color=text_color,
                    zorder=4,
                )

        def _anchor(node: _FlowNode, side: str) -> tuple[float, float]:
            if side == "left":
                return (node.x - node.width / 2, node.y)
            if side == "right":
                return (node.x + node.width / 2, node.y)
            if side == "top":
                return (node.x, node.y + node.height / 2)
            if side == "bottom":
                return (node.x, node.y - node.height / 2)
            return (node.x, node.y)

        arrow_kwargs = {
            "arrowstyle": "-|>",
            "color": "gray",
            "lw": arrow_linewidth,
            "mutation_scale": 8 if compact else 15,
            "zorder": 2,
            "shrinkA": 0,
            "shrinkB": 0,
            "clip_on": False,
        }
        for edge in edges:
            source = rendered_nodes[edge.source]
            target = rendered_nodes[edge.target]
            if edge.style == "vertical":
                if target.y >= source.y:
                    start = _anchor(source, "top")
                    end = _anchor(target, "bottom")
                else:
                    start = _anchor(source, "bottom")
                    end = _anchor(target, "top")
                arrow = mpatches.FancyArrowPatch(
                    posA=start,
                    posB=end,
                    **arrow_kwargs,
                )
            elif edge.style == "step_h":
                start = _anchor(source, "right")
                end = _anchor(target, "left")
                mid_x = (start[0] + end[0]) / 2
                path = mpath.Path(
                    [start, (mid_x, start[1]), (mid_x, end[1]), end],
                    [
                        mpath.Path.MOVETO,
                        mpath.Path.LINETO,
                        mpath.Path.LINETO,
                        mpath.Path.LINETO,
                    ],
                )
                arrow = mpatches.FancyArrowPatch(path=path, **arrow_kwargs)
            else:
                arrow = mpatches.FancyArrowPatch(
                    posA=_anchor(source, "right"),
                    posB=_anchor(target, "left"),
                    **arrow_kwargs,
                )
            ax.add_patch(arrow)
