"""Regression tests for normalization plotter initialization contracts.

Normalization plotting is assembled from focused plotting components, but the
public plotter must still initialize the shared before/after palette. This test
protects the constructor contract that previously regressed during module
splitting without coupling the suite to private mixin decorators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest

from pimqc.core import MetaboDataset
from pimqc.plotting.base import BasePlotter
from pimqc.plotting import plot_utils as pu
from pimqc.plotting.normalization import NormalizationPlotter
from pimqc.plotting.payloads import (
    NormalizationPlotPayload,
    snapshot_dataset,
)


def _minimal_normalization_dataset() -> MetaboDataset:
    """Return a dataset with metadata required by the plotter."""
    intensity = pd.DataFrame(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [20.0, 22.0, 21.0, 25.0, 24.0],
            [30.0, 29.0, 31.0, 35.0, 34.0],
        ],
        index=pd.Index(
            ["Feature 1", "Feature 2", "Feature 3"], name="Metabolite"
        ),
        columns=pd.Index(["S1", "S2", "S3", "S4", "S5"], name="Sample Name"),
    )
    sample_metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "QC", "QC", "Sample", "Sample"],
            "Batch": ["Batch 1"] * 5,
            "Inject Order": [1, 2, 3, 4, 5],
            "Bio Group": [
                "Group 1",
                "Group 1",
                "Group 1",
                "Group 1",
                "Group 2",
            ],
        },
        index=pd.Index(["S1", "S2", "S3", "S4", "S5"], name="Sample Name"),
    )
    return MetaboDataset.from_tables(intensity, sample_metadata)


def _auto_selection() -> dict[str, object]:
    """Return an AUTO passport deliberately absent from dataset attributes."""
    candidate_results = [
        {
            "method": "PQN",
            "status": "ok",
            "selected": True,
            "overall_score": 0.75,
            "rle_alignment_change_score": 0.8,
            "variance_stabilization_score": 0.7,
            "qc_structure_change_score": 0.8,
            "sample_structure_score": 0.6,
            "sample_structure_trustworthiness": 0.95,
            "sample_structure_rank_preservation": 0.92,
            "sample_structure_scale_preservation": 0.88,
        },
        {
            "method": "TIC",
            "status": "ok",
            "selected": False,
            "overall_score": 0.65,
            "rle_alignment_change_score": 0.7,
            "variance_stabilization_score": 0.6,
            "qc_structure_change_score": 0.7,
            "sample_structure_score": 0.5,
            "sample_structure_trustworthiness": 0.9,
            "sample_structure_rank_preservation": 0.87,
            "sample_structure_scale_preservation": 0.83,
        },
    ]
    return {
        "requested_method": "Auto",
        "selected_method": "PQN",
        "is_auto": True,
        "candidate_results": candidate_results,
    }


def _saved_diagnostics() -> dict[str, object]:
    """Small saved plot inputs; no normalization or statistics are rerun."""
    qc = {}
    for stage in ("Before Norm", "After Norm"):
        qc[stage] = {
            "rle_records": [
                {"Stage": stage, "Metric": metric, "Value": 0.2,
                 "Q25": 0.1, "Q75": 0.3}
                for metric in ("RLE center offset", "RLE spread")
            ],
            "variance": {
                "feature_stats": pd.DataFrame({"mean_intensity": [1., 2., 3.]}),
                "trend": pd.DataFrame({
                    "mean_intensity": [1., 2., 3.],
                    "dispersion_median": [0.2, 0.3, 0.4],
                    "dispersion_q25": [0.1, 0.2, 0.3],
                    "dispersion_q75": [0.3, 0.4, 0.5],
                }),
                "mean_variance_abs_rho": 0.5,
                "mean_variance_abs_slope": 0.1,
                "qc_dispersion_median": 0.3,
            },
            "structure": {
                "qc_centroid_distance": pd.Series([0.2, 0.3, 0.4]),
                "qc_pairwise_distance": pd.Series([0.4, 0.5, 0.6]),
            },
        }
    return {
        "qc_diagnostics": qc,
        "sample_structure": {
            "samples": pd.DataFrame({
                "scale_shift": [-0.05, 0.05],
                "rank_rho": [0.92, 0.96],
                "local_trust": [0.95, 0.96],
            }),
            "metrics": {
                "sample_structure_trustworthiness": 0.95,
                "sample_structure_rank_preservation": 0.94,
                "sample_structure_scale_preservation": 0.9,
            },
        },
    }


def test_normalization_plotter_uses_shared_palette_constants() -> None:
    """Initialize split plotting components with the public palette values."""
    raw = snapshot_dataset(_minimal_normalization_dataset())
    normalized = snapshot_dataset(_minimal_normalization_dataset())
    payload = NormalizationPlotPayload(
        raw_data=raw,
        normalized_data=normalized,
        selection={},
        score_component_weights={},
        sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25,
        global_seed=42,
    )
    plotter = NormalizationPlotter(payload)

    assert plotter.pal == {
        "Before Norm": pu.NEUTRAL_COLOR,
        "After Norm": pu.PRIMARY_ACCENT_COLOR,
    }


def test_standard_format_sizes_special_axis_text_consistently() -> None:
    """Size scientific offsets and logarithmic ticks like ordinary ticks."""
    raw = snapshot_dataset(_minimal_normalization_dataset())
    payload = NormalizationPlotPayload(
        raw_data=raw,
        normalized_data=raw,
        selection={},
        score_component_weights={},
        sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25,
        global_seed=42,
    )
    plotter = NormalizationPlotter(payload)
    figure, axes = plt.subplots(1, 2)
    scientific_ax, logarithmic_ax = axes
    scientific_ax.plot([1.0, 2.0], [1e-15, 8e-15])
    logarithmic_ax.plot([1.0, 2.0], [1e-5, 1e5])
    logarithmic_ax.set_yscale("log")
    tick_size = 4.5

    for axis in axes:
        plotter._apply_standard_format(
            axis,
            tick_fontsize=tick_size,
            append_stage=False,
        )
    figure.canvas.draw()

    offset = scientific_ax.yaxis.get_offset_text()
    assert offset.get_text()
    assert offset.get_fontsize() == tick_size
    assert offset.get_weight() == pu.DEFAULT_AXIS_TICK_WEIGHT
    assert all(
        label.get_fontsize() == tick_size
        for label in logarithmic_ax.get_yticklabels()
    )
    plt.close(figure)


def test_normalization_auto_dashboard_reads_selection_from_payload() -> None:
    """AUTO dashboard remains available after dataframe attrs are discarded."""
    raw = snapshot_dataset(_minimal_normalization_dataset())
    normalized = snapshot_dataset(_minimal_normalization_dataset())
    selection = _auto_selection()
    payload = NormalizationPlotPayload(
        raw_data=raw,
        normalized_data=normalized,
        selection=selection,
        score_component_weights={
            "rle_alignment_change_score": 3.0,
            "variance_stabilization_score": 3.0,
            "qc_structure_change_score": 2.0,
            "sample_structure_score": 2.0,
        },
        sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25,
        global_seed=42,
    )
    plotter = NormalizationPlotter(payload)

    dashboard = plotter.plot_normalization_dashboard()
    labels = set(dashboard.bricks_dict)

    assert "auto_norm_stacked_bar" in labels
    assert "Norm_Preservation_Scorecard" in labels
    plt.close("all")


def test_fixed_normalization_dashboard_uses_compact_target(
    monkeypatch,
) -> None:
    """Scale fixed-method 2 x 2 exports to two thirds of AUTO width."""
    raw = snapshot_dataset(_minimal_normalization_dataset())
    normalized = snapshot_dataset(_minimal_normalization_dataset())
    payload = NormalizationPlotPayload(
        raw_data=raw,
        normalized_data=normalized,
        selection={
            "requested_method": "VSN",
            "selected_method": "VSN",
            "is_auto": False,
            "candidate_results": [],
        },
        score_component_weights={},
        sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25,
        global_seed=42,
    )
    plotter = NormalizationPlotter(payload)
    for method_name in (
        "_plot_qc_rle_boxplot",
        "_plot_qc_variance_stabilization",
        "_plot_qc_structure_improvement",
        "_plot_sample_structure_preservation",
    ):
        monkeypatch.setattr(
            plotter, method_name, lambda *a, **k: k.get("ax", k.get("ax_geom"))
        )

    dashboard = plotter.plot_normalization_dashboard()

    assert dashboard is not None
    assert BasePlotter._dashboard_grid_shape(dashboard) == (2, 2)
    assert set(dashboard.bricks_dict) == {
        "QC_Alignment",
        "QC_Variance",
        "QC_Structure",
        "Sample_Structure",
    }
    expected_width = pu.TWO_BY_TWO_DASHBOARD_TARGET_WIDTH_IN / 2.0
    for brick in dashboard.bricks_dict.values():
        assert np.isclose(brick.get_position().width, expected_width)
    plt.close("all")


def test_decorated_fixed_normalization_dashboard_uses_two_by_two_preview() -> (
    None
):
    """Ignore small legend-induced offsets when resolving preview width."""
    raw = snapshot_dataset(_minimal_normalization_dataset())
    normalized = snapshot_dataset(_minimal_normalization_dataset())
    payload = NormalizationPlotPayload(
        raw_data=raw,
        normalized_data=normalized,
        selection={
            "requested_method": "VSN",
            "selected_method": "VSN",
            "is_auto": False,
            "candidate_results": [],
        },
        score_component_weights={},
        sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25,
        global_seed=42,
        **_saved_diagnostics(),
    )

    dashboard = NormalizationPlotter(payload).plot_normalization_dashboard()

    assert dashboard is not None
    assert BasePlotter._dashboard_grid_shape(dashboard) == (2, 2)
    assert (
        BasePlotter.resolve_dashboard_display_width(dashboard)
        == pu.TWO_BY_TWO_DASHBOARD_DISPLAY_WIDTH
    )
    plt.close("all")


@pytest.mark.parametrize("mode", ["auto", "fixed", "article"])
@pytest.mark.parametrize("trend_missing", ["empty", "nonfinite"])
def test_normalization_omits_unavailable_variance_panel_and_stage_legend(
    mode, trend_missing,
):
    """A valid scalar score does not require a drawable binned trend."""
    data = snapshot_dataset(_minimal_normalization_dataset())
    diagnostic = _saved_diagnostics()
    for stage in diagnostic["qc_diagnostics"].values():
        trend = stage["variance"]["trend"]
        stage["variance"]["trend"] = (
            trend.iloc[:0] if trend_missing == "empty" else trend * np.nan
        )
    selection = {} if mode == "fixed" else _auto_selection()
    plotter = NormalizationPlotter(NormalizationPlotPayload(
        raw_data=data, normalized_data=data, selection=selection,
        score_component_weights={
            "rle_alignment_change_score": 0.3,
            "variance_stabilization_score": 0.3,
            "qc_structure_change_score": 0.2,
            "sample_structure_score": 0.2,
        },
        sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25,
        global_seed=42, **diagnostic,
    ))
    dashboard = (
        plotter.plot_normalization_article_dashboard() if mode == "article"
        else plotter.plot_normalization_dashboard()
    )
    assert dashboard is not None
    assert all("variance" not in label.lower() for label in dashboard.bricks_dict)
    assert all(ax.get_visible() for ax in dashboard.bricks_dict.values())
    from matplotlib.text import Text
    text = [
        artist.get_text() for ax in dashboard.bricks_dict.values()
        for artist in ax.findobj(Text)
    ]
    assert not any("Insufficient" in t for t in text)
    assert "QC variance stabilization stage" not in text
    if mode != "fixed":
        assert "QC variance stabilization" in text  # Score legend stays valid.
        assert selection["candidate_results"][0]["overall_score"] == 0.75
    plt.close("all")


@pytest.mark.parametrize("panel", [
    "_plot_qc_rle_boxplot", "_plot_qc_variance_stabilization",
    "_plot_qc_structure_improvement", "_plot_sample_structure_preservation",
])
def test_empty_normalization_diagnostics_return_none_without_placeholder(panel):
    data = snapshot_dataset(_minimal_normalization_dataset())
    plotter = NormalizationPlotter(NormalizationPlotPayload(
        raw_data=data, normalized_data=data, selection={},
        score_component_weights={}, sample_scale_log_ratio_tolerance=0.25,
        sample_scale_relative_delta_tolerance=0.25, global_seed=42,
    ))
    assert getattr(plotter, panel)() is None
    assert plotter.plot_normalization_dashboard() is None
    plt.close("all")
