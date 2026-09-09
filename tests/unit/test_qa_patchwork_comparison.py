"""Shared QA bricks, project-wide legends, and report grid geometry."""

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pytest

from pimqc import QualityAssessor
from pimqc.core import MetaboDataset
from pimqc.plotting.assessment import (
    plot_assessment_comparison,
    AssessmentPlotter,
)
from pimqc.plotting.base import BasePlotter
from test_saved_diagnostics import _dataset


@pytest.fixture(scope="module")
def project_audit():
    """
    Prepare a project with valid reference features but no flagged samples.
    """
    data = _dataset()
    data = data.with_intensity(
        data.intensity,
        context_updates={
            "internal_standards": ("F0",),
            "outlier_reference_features": ("F1",),
        },
    )
    audit = QualityAssessor(data).run_assessment().audit
    # No plotted abnormal categories or reference flags: the project legend
    # must still advertise the available diagnostics, not just observed hits.
    metrics = audit.pca_results["metrics_df"].copy()
    metrics["Category"] = "Normal"
    return replace(
        audit,
        pca_results={**audit.pca_results, "metrics_df": metrics},
        internal_standard_flags=audit.internal_standard_flags * False,
        outlier_reference_flags=audit.outlier_reference_flags * False,
    )


def _legend_texts(brick):
    legends = [a for a in brick.artists if isinstance(a, Legend)]
    if brick.get_legend() is not None:
        legends.append(brick.get_legend())
    return {t.get_text() for legend in legends for t in legend.get_texts()}


@pytest.mark.parametrize("count", [6, 7])
@pytest.mark.parametrize("diagnostic", ["rsd", "pca", "outlier", "correlation"])
def test_report_grid_has_shared_panels_and_final_legend(
    project_audit, count, diagnostic, tmp_path
):
    """
    Check equal data bricks, full shared legends, and half-height colorbars.
    """
    audits = {f"Stage {i + 1}": project_audit for i in range(count)}
    grid = plot_assessment_comparison(audits, diagnostic)
    bricks = grid.bricks_dict
    assert list(bricks)[-1] == f"{diagnostic}_legend"
    assert len(bricks) == count + 1
    panels = [bricks[f"{diagnostic}_stage_{i}"] for i in range(count)]
    for panel in panels:
        assert panel.get_legend() is None
    np.testing.assert_allclose(
        [[b.get_position().width, b.get_position().height] for b in panels],
        np.tile(
            [panels[0].get_position().width, panels[0].get_position().height],
            (count, 1),
        ),
        rtol=1e-5,
    )
    legend = bricks[f"{diagnostic}_legend"]
    if diagnostic != "correlation":
        from matplotlib.transforms import Bbox

        legend.figure.canvas.draw()
        legends = [a for a in legend.artists if isinstance(a, Legend)]
        if legend.get_legend() is not None:
            legends.append(legend.get_legend())
        box = Bbox.union(
            [item.get_window_extent() for item in legends]
        ).transformed(legend.transAxes.inverted())
        assert (box.x0 + box.x1) / 2 == pytest.approx(0.5)
        assert (box.y0 + box.y1) / 2 == pytest.approx(0.5)
    if diagnostic == "outlier":
        assert _legend_texts(legend) == {
            "Extreme Outlier",
            "Orthogonal Outlier",
            "Strong Outlier",
            "Normal",
            "IS Outlier",
            "ORF Outlier",
        }
    if diagnostic == "correlation":
        assert len(legend.child_axes) == 1
        colorbar = legend.child_axes[0]
        legend.figure.canvas.draw()
        assert (
            colorbar.get_position().height / legend.get_position().height
            == pytest.approx(0.5)
        )
        assert all(not panel.child_axes for panel in panels)
    writer = AssessmentPlotter(project_audit.plot_payload)
    writer.save_and_show_pw(
        grid,
        file_path=str(tmp_path / f"{diagnostic}-{count}.svg"),
        save_format="svg",
        show_plot=False,
    )
    # A preview for manual layout verification; no changes to scientific data.
    grid.savefig(str(tmp_path / f"{diagnostic}-{count}.png"), dpi=100)
    plt.close("all")


def test_reference_legend_uses_all_stage_availability(project_audit):
    """Keep valid reference channels even when absent from the first stage."""
    absent = replace(
        project_audit, internal_standard_ids=(), outlier_reference_ids=()
    )
    grid = plot_assessment_comparison(
        {"First": absent, "Last": project_audit}, "outlier"
    )
    assert {"IS Outlier", "ORF Outlier"} <= _legend_texts(
        grid.bricks_dict["outlier_legend"]
    )
    grid = plot_assessment_comparison({"Absent": absent}, "outlier")
    assert not (
        {"IS Outlier", "ORF Outlier"}
        & _legend_texts(grid.bricks_dict["outlier_legend"])
    )
    plt.close("all")


def test_summary_dashboard_omits_empty_outlier_bar(project_audit, tmp_path):
    """Use aligned heatmap/PCA over RSD/outlier in a compact 2 x 2 grid."""
    plotter = AssessmentPlotter(project_audit.plot_payload)
    audit = project_audit
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

    assert "outlier_scatter" in dashboard.bricks_dict
    assert "outlier_bar" not in dashboard.bricks_dict
    assert len(dashboard.bricks_dict) == 4
    # The outlier legend is mounted inside the scatter brick.  patchworklib
    # may expose that internal legend lane as an additional grid column on
    # different versions; assert the semantic four-brick composition below
    # rather than a renderer-specific internal shape.
    outlier_legend_texts = {
        text.get_text()
        for legend in dashboard.bricks_dict["outlier_scatter"].findobj(Legend)
        for text in legend.get_texts()
    }
    assert {
        "Extreme Outlier",
        "Orthogonal Outlier",
        "Strong Outlier",
        "Normal",
        "IS Outlier",
        "ORF Outlier",
    } <= outlier_legend_texts
    bricks = list(dashboard.bricks_dict.values())
    top_y = sorted({brick.get_position().y0 for brick in bricks}, reverse=True)
    assert len(top_y) == 2
    assert dashboard.bricks_dict["outlier_scatter"].get_position().y0 < top_y[0]
    heatmap_brick = dashboard.bricks_dict["correlation_heatmap"]
    rsd_brick = dashboard.bricks_dict["rsd_bar"]
    pca_brick = dashboard.bricks_dict["pca_scatter"]
    outlier_brick = dashboard.bricks_dict["outlier_scatter"]
    assert len(outlier_brick.child_axes) == 1
    heatmap_ax = heatmap_brick.child_axes[0]
    rsd_ax = rsd_brick.child_axes[0]
    pca_ax = pca_brick.child_axes[0]
    outlier_ax = outlier_brick.child_axes[0]
    dashboard.savefig(tmp_path / "qa-compact.png", dpi=100)
    renderer = outlier_brick.figure.canvas.get_renderer()
    heatmap_box = heatmap_ax.get_window_extent(renderer)
    rsd_box = rsd_ax.get_window_extent(renderer)
    pca_box = pca_ax.get_window_extent(renderer)
    outlier_box = outlier_ax.get_window_extent(renderer)
    assert heatmap_box.width == pytest.approx(heatmap_box.height, rel=0.02)
    assert rsd_box.width == pytest.approx(heatmap_box.width, rel=0.02)
    assert pca_box.width == pytest.approx(outlier_box.width, rel=0.02)
    legend_box = max(
        (
            legend.get_window_extent(renderer)
            for legend in outlier_brick.findobj(Legend)
        ),
        key=lambda box: box.x1,
    )
    assert (
        legend_box.x1 / outlier_brick.figure.dpi
        <= dashboard.get_outer_corner()[1] + 0.1
    )
    plt.close("all")


@pytest.mark.parametrize("cols", [0, -1, True, "3"])
def test_invalid_columns_are_rejected(project_audit, cols):
    """Reject invalid layout requests before constructing a patchwork figure."""
    with pytest.raises(ValueError, match="cols"):
        plot_assessment_comparison({"Stage": project_audit}, "rsd", cols=cols)


def test_batch_styles_and_legend_survive_batch_filtering(monkeypatch, tmp_path):
    """
    Share a union marker map even when the first audit lacks a later batch.
    """
    data = _dataset()
    metadata = data.sample_metadata.copy()
    metadata.loc[metadata.index[8:], "Batch"] = "B2"
    both = MetaboDataset.from_tables(data.intensity, metadata)
    only_b2 = MetaboDataset.from_tables(
        data.intensity.iloc[:, 8:], metadata.iloc[8:]
    )
    audits = {
        "Filtered B2": QualityAssessor(only_b2).run_assessment().audit,
        "Both batches": QualityAssessor(both).run_assessment().audit,
    }
    styles = []
    original = AssessmentPlotter.plot_pca_scatter

    def capture(self, *args, **kwargs):
        styles.append(self.style_map.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AssessmentPlotter, "plot_pca_scatter", capture)
    grid = plot_assessment_comparison(audits, "pca")
    assert styles == [{"B1": "o", "B2": "s"}] * 2
    assert {"B1", "B2", "QC", "Sample"} <= _legend_texts(
        grid.bricks_dict["pca_legend"]
    )
    grid = plot_assessment_comparison(audits, "correlation")
    assert (
        grid.bricks_dict["correlation_legend"].child_axes[0].get_ylabel()
        == "Spearman Correlation"
    )
    writer = AssessmentPlotter(audits["Both batches"].plot_payload)
    writer.save_and_show_pw(
        grid,
        file_path=str(tmp_path / "batch-correlation.svg"),
        save_format="svg",
        show_plot=False,
    )
    assert (tmp_path / "batch-correlation.svg").is_file()
    plt.close("all")
