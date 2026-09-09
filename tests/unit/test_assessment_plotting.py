"""Regression tests for assessment visualization annotations."""

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from pimqc.core import MetaboDataset
from pimqc.plotting.assessment import AssessmentPlotter
from pimqc.plotting import plot_utils as pu
from pimqc.plotting.payloads import AssessmentPlotPayload


def _assert_standard_heatmap_colorbar_edge(colorbar_ax) -> None:
    """Check the visible colorbar outline against the heatmap grid style."""
    outline = colorbar_ax.spines["outline"]
    assert outline.get_visible()
    np.testing.assert_allclose(outline.get_edgecolor(), mcolors.to_rgba("k"))
    assert outline.get_linewidth() == pu.DEFAULT_HEATMAP_CELL_LINEWIDTH


def test_single_batch_pca_annotation_omits_batch_silhouette() -> None:
    """Hide batch silhouette when the PCA data contain only one batch."""
    annotation = AssessmentPlotter._format_pca_diagnostics_annotation(
        pca_diagnostics={
            "relative_dispersion": 0.25,
            "batch_silhouette": np.nan,
            "centrality_shift": 0.1,
        },
        batches=pd.Series(["Batch 1", "Batch 1"]),
    )

    assert annotation.splitlines() == [
        "Relative Dispersion: 0.2500",
        "Centrality Shift: 0.1000",
    ]


def test_multi_batch_pca_annotation_keeps_batch_silhouette() -> None:
    """Keep batch silhouette for multi-batch PCA data, including N/A."""
    annotation = AssessmentPlotter._format_pca_diagnostics_annotation(
        pca_diagnostics={
            "relative_dispersion": 0.25,
            "batch_silhouette": np.nan,
            "centrality_shift": 0.1,
        },
        batches=pd.Series(["Batch 1", "Batch 2"]),
    )

    assert annotation.splitlines() == [
        "Relative Dispersion: 0.2500",
        "Batch Silhouette: N/A",
        "Centrality Shift: 0.1000",
    ]


def test_correlation_colorbar_legend_uses_half_brick_height() -> None:
    """Keep the pooled-QC colorbar sidecar at about half a report brick."""
    columns = pd.Index(["Q1", "Q2"], name="Sample Name")
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "QC"],
            "Batch": ["B1", "B1"],
            "Inject Order": [1, 2],
        },
        index=columns,
    )
    dataset = MetaboDataset.from_tables(
        pd.DataFrame([[1.0, 2.0]], index=["F1"], columns=columns),
        metadata,
        pd.DataFrame(index=["F1"]),
    )
    plotter = AssessmentPlotter(AssessmentPlotPayload(dataset, False))

    figure = plotter.plot_correlation_colorbar_legend("Spearman")
    try:
        assert tuple(figure.get_size_inches()) == (
            pu.CORRELATION_COLORBAR_LEGEND_WIDTH_IN,
            pu.CORRELATION_COLORBAR_LEGEND_HEIGHT_IN,
        )
        _assert_standard_heatmap_colorbar_edge(figure.axes[0])
    finally:
        plt.close(figure)


def test_dashboard_batch_heatmap_colorbar_has_visible_cell_grid_edge() -> None:
    """Keep the embedded QA colorbar edge consistent with heatmap cells."""
    columns = pd.Index(["Q1", "Q2", "Q3"], name="Sample Name")
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "QC", "QC"],
            "Batch": ["B1", "B2", "B3"],
            "Inject Order": [1, 2, 3],
        },
        index=columns,
    )
    dataset = MetaboDataset.from_tables(
        pd.DataFrame([[1.0, 2.0, 3.0]], index=["F1"], columns=columns),
        metadata,
        pd.DataFrame(index=["F1"]),
    )
    plotter = AssessmentPlotter(AssessmentPlotPayload(dataset, True))
    correlation = pd.DataFrame(
        [
            [0.973, 0.930, 0.921],
            [0.930, 0.980, 0.955],
            [0.921, 0.955, 0.976],
        ],
        index=["B1", "B2", "B3"],
        columns=["B1", "B2", "B3"],
    )

    figure, parent_ax = plt.subplots()
    parent_ax.axis("off")
    heatmap_ax = parent_ax.inset_axes([0.0, 0.0, 0.83, 1.0])
    plotter.plot_batch_corr_heatmap(
        correlation,
        "spearman",
        ax=heatmap_ax,
    )
    try:
        colorbar_ax = heatmap_ax.child_axes[0]
        _assert_standard_heatmap_colorbar_edge(colorbar_ax)
        x_labels = heatmap_ax.get_xticklabels()
        assert all(label.get_rotation() == 0.0 for label in x_labels)
        assert all(label.get_ha() == "center" for label in x_labels)
    finally:
        plt.close(figure)


def test_heatmap_tick_wrapping_preserves_identifier_tokens() -> None:
    """Wrap long IDs at separators without splitting alphanumeric tokens."""
    figure = plt.figure(figsize=(4.0, 4.0), dpi=100)
    try:
        label = "B1-QC-1-KFI2600-25min_R11"
        wrapped = pu.wrap_tick_label(
            label,
            figure=figure,
            max_width_pixels=50.0,
            fontsize=6.0,
        )
        assert "\n" in wrapped
        assert "KFI2600" in wrapped
        assert "25min_R11" in wrapped
        assert "KFI2\n600" not in wrapped
    finally:
        plt.close(figure)


def test_empty_outlier_bar_data_is_detected_before_layout() -> None:
    """Do not reserve a barplot brick when no outlier data are actionable."""
    columns = pd.MultiIndex.from_tuples(
        [("S1", "Sample", "B1", 1), ("S2", "Sample", "B1", 2)],
        names=["Sample Name", "Sample Type", "Batch", "Inject Order"],
    )
    metadata = columns.to_frame(index=False).set_index("Sample Name")
    dataset = MetaboDataset.from_tables(
        pd.DataFrame([[1.0, 2.0]], index=["F1"], columns=["S1", "S2"]),
        metadata,
        pd.DataFrame(index=["F1"]),
    )
    outliers = pd.DataFrame(
        {
            ("SPE-DModX", "Outliers (SPE-DModX)"): [False, False],
            ("HT2", "Outliers (HT2)"): [False, False],
        },
        index=columns,
    )
    plotter = AssessmentPlotter(AssessmentPlotPayload(dataset, False))
    assert not plotter._has_stat_outlier_bar_data(
        outliers,
        "Sample Type",
        "Sample",
    )


def test_sample_heatmap_places_batch_strips_between_cells_and_ticks() -> None:
    """Keep left/bottom annotation strips in stable physical lanes."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("B1", "QC", 1, "Long-QC-Sample-1"),
            ("B1", "QC", 2, "Long-QC-Sample-2"),
            ("B1", "QC", 3, "Long-QC-Sample-3"),
        ],
        names=["Batch", "Sample Type", "Inject Order", "Sample Name"],
    )
    metadata = columns.to_frame(index=False).set_index("Sample Name")
    dataset = MetaboDataset.from_tables(
        pd.DataFrame(
            [[1.0, 2.0, 3.0]],
            index=["F1"],
            columns=[
                "Long-QC-Sample-1",
                "Long-QC-Sample-2",
                "Long-QC-Sample-3",
            ],
        ),
        metadata,
        pd.DataFrame(index=["F1"]),
    )
    plotter = AssessmentPlotter(AssessmentPlotPayload(dataset, False))
    correlation = pd.DataFrame(np.eye(3), index=columns, columns=columns)
    figure = plotter.plot_qc_corr_heatmap(
        correlation,
        None,
        ["B1"],
        cluster="none",
        show_colorbar=False,
    )
    try:
        axis = figure.axes[0]
        bottom = [
            patch
            for patch in axis.patches
            if patch.get_gid() == "batch-annotation-bottom"
        ]
        left = [
            patch
            for patch in axis.patches
            if patch.get_gid() == "batch-annotation-left"
        ]
        assert len(bottom) == len(left) == 3
        assert all(
            np.isclose(patch.get_y() + patch.get_height(), 0.0)
            for patch in bottom
        )
        assert all(
            np.isclose(patch.get_x() + patch.get_width(), 0.0) for patch in left
        )
        assert axis.xaxis.majorTicks[0].get_pad() > 5.0
        assert axis.yaxis.majorTicks[0].get_pad() > 5.0
        assert axis.xaxis.majorTicks[0].tick1line.get_markersize() == 0.0
        assert axis.yaxis.majorTicks[0].tick1line.get_markersize() == 0.0
        assert all(
            label.get_ha() == "center" for label in axis.get_xticklabels()
        )
        assert [
            label.get_text().replace("\n", "")
            for label in axis.get_xticklabels()
        ] == [
            "Long-QC-Sample-1",
            "Long-QC-Sample-2",
            "Long-QC-Sample-3",
        ]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bottom_box = bottom[0].get_window_extent(renderer)
        left_box = left[0].get_window_extent(renderer)
        assert (
            max(
                label.get_window_extent(renderer).y1
                for label in axis.get_xticklabels()
            )
            < bottom_box.y0
        )
        assert (
            max(
                label.get_window_extent(renderer).x1
                for label in axis.get_yticklabels()
            )
            < left_box.x0
        )
    finally:
        plt.close(figure)


def test_batch_heatmap_wraps_long_ticks_without_splitting_words() -> None:
    """Use the same renderer-aware tick policy for batch heatmaps."""
    columns = pd.Index(["Q1", "Q2", "Q3"], name="Sample Name")
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "QC", "QC"],
            "Batch": ["B1", "B2", "B3"],
            "Inject Order": [1, 2, 3],
        },
        index=columns,
    )
    dataset = MetaboDataset.from_tables(
        pd.DataFrame([[1.0, 2.0, 3.0]], index=["F1"], columns=columns),
        metadata,
        pd.DataFrame(index=["F1"]),
    )
    plotter = AssessmentPlotter(AssessmentPlotPayload(dataset, True))
    labels = [
        "Batch-Alpha-Cohort",
        "Batch-Beta-Cohort",
        "Batch-Gamma-Cohort",
    ]
    correlation = pd.DataFrame(np.eye(3), index=labels, columns=labels)
    figure = plotter.plot_batch_corr_heatmap(
        correlation,
        "spearman",
        show_colorbar=False,
    )
    try:
        displayed = [
            label.get_text() for label in figure.axes[0].get_xticklabels()
        ]
        assert all("\n" in label for label in displayed)
        assert all("Cohort" in label for label in displayed)
        assert all("Co\nhort" not in label for label in displayed)
        assert all(
            label.get_rotation() == 0.0
            for label in figure.axes[0].get_xticklabels()
        )
        assert all(
            label.get_ha() == "center"
            for label in figure.axes[0].get_xticklabels()
        )
    finally:
        plt.close(figure)
