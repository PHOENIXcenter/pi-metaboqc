"""Regression tests for imputation dashboard visualization geometry."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pimqc.core import MetaboDataset
from pimqc.plotting.base import BasePlotter
from pimqc.plotting.imputation import ImputationPlotter
from pimqc.plotting import plot_utils as pu
from pimqc.plotting.payloads import (
    ImputationPlotPayload,
    snapshot_dataset,
)


def _minimal_imputation_payload() -> ImputationPlotPayload:
    """Return a processor-free payload for dashboard geometry tests."""
    intensity = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["F1", "F2"],
        columns=["S1", "S2"],
    )
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "Sample"],
            "Batch": ["Batch 1", "Batch 1"],
            "Inject Order": [1, 2],
        },
        index=["S1", "S2"],
    )
    dataset = MetaboDataset.from_tables(intensity, metadata)
    snapshot = snapshot_dataset(dataset)
    return ImputationPlotPayload(
        raw_data=snapshot,
        imputed_data=snapshot,
        global_seed=42,
    )


def test_fixed_imputation_dashboard_uses_three_panels_with_density_legend(
    monkeypatch,
) -> None:
    """Place the fixed-method density legend inside its diagnostic panel."""
    from pimqc.plotting.imputation import dashboards as dashboard_module

    plotter = ImputationPlotter(_minimal_imputation_payload())
    density_kwargs = {}
    monkeypatch.setattr(plotter, "_plot_nrmse_scatter", lambda *a, **k: None)
    monkeypatch.setattr(
        plotter,
        "_plot_masked_distribution_fidelity",
        lambda *a, **kwargs: density_kwargs.update(kwargs),
    )
    monkeypatch.setattr(
        dashboard_module,
        "plot_sample_structure_change_map",
        lambda *a, **k: None,
    )

    dashboard = plotter.plot_imputation_method_dashboard(
        metrics={},
        true_vals=np.array([1.0, 2.0]),
        pred_vals=np.array([1.1, 1.9]),
        method_name="KNN",
    )

    assert dashboard is not None
    assert BasePlotter._dashboard_grid_shape(dashboard) == (1, 3)
    assert set(dashboard.bricks_dict) == {
        "method_nrmse_scatter",
        "method_masked_density",
        "method_sample_structure_preservation",
    }
    assert density_kwargs["show_legend"] is True
    expected_width = pu.DASHBOARD_TARGET_WIDTH_IN / 3.0
    for brick in dashboard.bricks_dict.values():
        assert np.isclose(brick.get_position().width, expected_width)
    plt.close("all")


def test_auto_imputation_dashboard_keeps_combined_legend(monkeypatch) -> None:
    """Keep AUTO density keys in its existing shared legend brick."""
    from pimqc.plotting.imputation import dashboards as dashboard_module

    plotter = ImputationPlotter(_minimal_imputation_payload())
    density_kwargs = {}
    legend_calls = []
    monkeypatch.setattr(
        plotter,
        "plot_imputation_score_summary",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        plotter,
        "plot_imputation_preservation_scorecard",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        plotter,
        "plot_imputation_dashboard_legend",
        lambda *a, **kwargs: legend_calls.append(kwargs),
    )
    monkeypatch.setattr(plotter, "_plot_nrmse_scatter", lambda *a, **k: None)
    monkeypatch.setattr(
        plotter,
        "_plot_masked_distribution_fidelity",
        lambda *a, **kwargs: density_kwargs.update(kwargs),
    )
    monkeypatch.setattr(
        dashboard_module,
        "plot_sample_structure_change_map",
        lambda *a, **k: None,
    )
    values = np.array([1.0, 2.0])
    results = {
        "KNN": (
            {"Auto_Score": 0.9, "NRMSE_Total": 0.1},
            values,
            values,
        )
    }

    dashboard = plotter.plot_imputation_auto_dashboard(
        results,
        selected_method="KNN",
    )

    assert dashboard is not None
    assert "imputation_dashboard_legend" in dashboard.bricks_dict
    assert len(legend_calls) == 1
    assert density_kwargs.get("show_legend", False) is False
    plt.close("all")


def test_masked_density_legend_uses_best_location() -> None:
    """Let Matplotlib select the least obstructed in-axes legend position."""
    plotter = ImputationPlotter(_minimal_imputation_payload())
    figure, axis = plt.subplots()

    plotter._plot_masked_distribution_fidelity(
        true_vals=np.array([8.0, 10.0, 12.0, 14.0]),
        pred_vals=np.array([8.2, 9.8, 11.5, 14.4]),
        metrics={"JSD_Total": 0.1, "Wasserstein_Normalized": 0.2},
        ax=axis,
        show_legend=True,
    )

    legend = axis.get_legend()
    assert legend is not None
    assert legend._loc == 0
    plt.close(figure)


def test_scorecard_cell_borders_do_not_cross_grouped_header() -> None:
    """Keep matrix column borders out of the semantic group-header row."""
    metric_values = {
        "KNN": (0.87, 0.96, 0.98, 0.96, 0.89),
        "LLS": (0.86, 0.96, 0.98, 0.95, 0.96),
    }
    results = {
        method: (
            {
                "JSD_Score": values[0],
                "Wasserstein_Score": values[1],
                "Trustworthiness": values[2],
                "Distance_Rank_Preservation": values[3],
                "Distance_Scale_Preservation": values[4],
                "Auto_Score": float(np.mean(values)),
            },
            np.array([]),
            np.array([]),
        )
        for method, values in metric_values.items()
    }

    figure, axis = plt.subplots()
    visualizer = object.__new__(ImputationPlotter)
    visualizer.runtime_font_fallbacks = visualizer.VECTOR_FONT_FALLBACKS
    visualizer.plot_imputation_preservation_scorecard(
        results,
        selected_method="KNN",
        ax=axis,
    )

    header_center = -0.86
    vertical_borders = []
    for line in axis.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        if x_data.size == 2 and np.isclose(x_data[0], x_data[1]):
            vertical_borders.append(line)
            assert not (
                float(np.min(y_data)) <= header_center <= float(np.max(y_data))
            )

    assert len(vertical_borders) == 6
    assert [patch.get_width() for patch in axis.patches] == [2.0, 3.0]
    plt.close(figure)


def test_nrmse_scatter_feature_strata_without_point_cutoff_lines() -> (
    None
):
    """Describe the feature-median split without implying point-level limits."""
    true_values = np.array([8.0, 10.0, 12.0, 14.0])
    predicted_values = np.array([8.2, 9.8, 11.5, 14.4])
    metrics = {
        "NRMSE_Total": 0.10,
        "NRMSE_Low": 0.12,
        "NRMSE_High": 0.08,
        "Threshold": 10.5,
        "Threshold_Quantile": 0.25,
    }
    figure, axis = plt.subplots()
    visualizer = object.__new__(ImputationPlotter)

    visualizer._plot_nrmse_scatter(
        true_vals=true_values,
        pred_vals=predicted_values,
        metrics=metrics,
        show_colorbar=False,
        ax=axis,
    )

    annotation_text = "\n".join(text.get_text() for text in axis.texts)
    assert annotation_text.splitlines() == [
        "NRMSE (total): 0.1000",
        "NRMSE (low): 0.1200",
    ]
    assert len(axis.lines) == 1
    plt.close(figure)
