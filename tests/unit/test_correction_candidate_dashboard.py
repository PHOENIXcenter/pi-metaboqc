"""Candidate comparisons show the final distributions used in scoring."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from pimqc.core import MetaboDataset
from pimqc.plotting import plot_utils as pu
from pimqc.plotting.correction import CorrectionPlotter
from pimqc.plotting.payloads import CorrectionPlotPayload


RSD_PANEL = "candidate_qc_rsd_comparison"
D_RATIO_PANEL = "candidate_d_ratio_comparison"
METHODS = (
    "SERRF",
    "QC-SVR",
    "QC-RLSC",
    "robust QC-RLSC",
    "RUV-III",
    "WaveICA 2.0",
)


@pytest.fixture
def candidate_dashboard(monkeypatch):
    """Build saved diagnostics directly, without fitting correction models."""
    calls = {}
    original_boxplot = Axes.boxplot

    def record_boxplot(ax, values, *args, **kwargs):
        artists = original_boxplot(ax, values, *args, **kwargs)
        if ax.get_label() in (RSD_PANEL, D_RATIO_PANEL):
            calls[ax.get_label()] = {
                "values": [np.asarray(value).copy() for value in values],
                "positions": np.asarray(kwargs["positions"]),
                "artists": artists,
            }
        return artists

    monkeypatch.setattr(Axes, "boxplot", record_boxplot)

    def build(*, multi_batch=True):
        samples = pd.Index(["Q1", "Q2", "S1", "S2"], name="Sample Name")
        features = pd.Index(["F1", "F2", "F3"], name="Metabolite")
        metadata = pd.DataFrame(
            {
                "Sample Type": ["QC", "QC", "Sample", "Sample"],
                "Batch": ["B1", "B2", "B1", "B2"]
                if multi_batch else ["B1"] * 4,
                "Inject Order": [1, 2, 3, 4],
            },
            index=samples,
        )
        source = MetaboDataset.from_tables(
            pd.DataFrame(100.0, index=features, columns=samples), metadata
        )
        frame = source.annotated_frame()
        results = {}
        for index, method in enumerate(METHODS):
            regression = method in {"QC-SVR", "QC-RLSC"}
            stages = ["Intra-batch corrected"] if regression else [
                f"{method} corrected"
            ]
            if regression and multi_batch:
                stages.append("Inter-batch corrected")
            full_only = method in {"RUV-III", "WaveICA 2.0"}
            final_full = pd.Series([0.08, 0.10, 0.12]) + index * 0.01
            final_oof = pd.Series([0.18, 0.20, 0.22]) + index * 0.01
            full_rsd = {"Original": pd.Series([0.4, 0.5, 0.6])}
            oof_rsd = {}
            for stage in stages:
                full_rsd[stage] = final_full if stage == stages[-1] else (
                    pd.Series([0.61, 0.62, 0.63])
                )
                oof_rsd[stage] = final_oof if stage == stages[-1] else (
                    pd.Series([0.71, 0.72, 0.73])
                )
            results[method] = {
                "auto_score": 1.0 - index * 0.1,
                "implementation": "python",
                "candidate_params": {"robust": method == "robust QC-RLSC"},
                "stage_dfs": {stage: frame for stage in ["Original", *stages]},
                "stage_oof_dfs": {stage: frame for stage in stages},
                "stage_qc_rsd": full_rsd,
                "stage_oof_qc_rsd": oof_rsd,
                "final_rsd_full": float(final_full.median()),
                "final_rsd_oof": float(final_oof.median()),
                "eval_rsd": float(
                    (final_full if full_only else final_oof).median()
                ),
                "validation": {
                    "evaluation_basis": "full_model" if full_only else "oof",
                    "eligible_for_auto": True,
                },
                "d_ratio_baseline": pd.Series([45.0, 50.0, 55.0]),
                "d_ratio_current_full": pd.Series([10.0, 15.0, 20.0]) + index,
                "d_ratio_current_oof": pd.Series([25.0, 30.0, 35.0]) + index,
                "d_ratio_evaluation_basis": (
                    "full_model" if full_only else "oof"
                ),
            }
        plotter = CorrectionPlotter(
            CorrectionPlotPayload(
                source_data=source,
                selected_stages={},
                candidate_results=results,
                selected_prediction=None,
                internal_standard_ids=(),
                boundary_type="IQR",
            )
        )
        return plotter, results, calls

    yield build
    plt.close("all")


def _values_at(call, position):
    matches = np.flatnonzero(np.isclose(call["positions"], position))
    assert len(matches) == 1, "Each evaluated candidate needs one centered box"
    return call["values"][matches[0]]


def test_candidate_dashboard_uses_final_scoring_basis_and_shared_positions(
    candidate_dashboard,
):
    plotter, results, calls = candidate_dashboard()
    dashboard = plotter.plot_correction_candidate_dashboard(results, "SERRF")
    assert set(dashboard.bricks_dict) == {
        RSD_PANEL, D_RATIO_PANEL, "correction_mode_legend"
    }
    rsd_ax = dashboard.bricks_dict[RSD_PANEL]
    d_ratio_ax = dashboard.bricks_dict[D_RATIO_PANEL]
    positions = rsd_ax.get_xticks()
    assert len(positions) == 7
    np.testing.assert_allclose(positions, d_ratio_ax.get_xticks())
    for name in (RSD_PANEL, D_RATIO_PANEL):
        np.testing.assert_allclose(calls[name]["positions"], positions)
        assert dashboard.bricks_dict[name].get_legend() is None
    np.testing.assert_allclose(
        _values_at(calls[RSD_PANEL], positions[0]), [40.0, 50.0, 60.0]
    )
    np.testing.assert_allclose(
        _values_at(calls[D_RATIO_PANEL], positions[0]), [45.0, 50.0, 55.0]
    )
    for index, method in enumerate(METHODS, start=1):
        result = results[method]
        final_stage = next(reversed(result["stage_dfs"]))
        full = result["validation"]["evaluation_basis"] == "full_model"
        expected_rsd = result[
            "stage_qc_rsd" if full else "stage_oof_qc_rsd"
        ][final_stage]
        np.testing.assert_allclose(
            _values_at(calls[RSD_PANEL], positions[index]), expected_rsd * 100
        )
        np.testing.assert_allclose(
            _values_at(calls[D_RATIO_PANEL], positions[index]),
            result["d_ratio_current_full" if full else "d_ratio_current_oof"],
        )
    labels = [tick.get_text() for tick in rsd_ax.get_xticklabels()]
    assert all("OOF" not in label and "Full" not in label for label in labels)
    assert all("Intra" not in label for label in labels)
    assert sum("Inter-batch" in label for label in labels) == 2
    assert [label for label in labels if label.startswith("* ")] == [
        "* SERRF"
    ]
    # Both limits are driven by the displayed observations, not fixed 0/100.
    assert 0 < d_ratio_ax.get_ylim()[0] < 14
    assert 55 < d_ratio_ax.get_ylim()[1] < 100
    assert labels == [tick.get_text() for tick in d_ratio_ax.get_xticklabels()]
    rsd_bounds = rsd_ax.get_position()
    d_ratio_bounds = d_ratio_ax.get_position()
    assert rsd_bounds.x1 < d_ratio_bounds.x0
    assert rsd_bounds.y0 == pytest.approx(d_ratio_bounds.y0)
    assert rsd_bounds.height == pytest.approx(d_ratio_bounds.height)
    legend_ax = dashboard.bricks_dict["correction_mode_legend"]
    assert legend_ax.get_position().y1 < rsd_bounds.y0
    assert legend_ax.get_legend() is not None
    colors = ["tab:gray", *[pu.PRIMARY_ACCENT_COLOR] * len(METHODS)]
    for name in (RSD_PANEL, D_RATIO_PANEL):
        boxes = calls[name]["artists"]["boxes"]
        for index, (patch, color) in enumerate(zip(boxes, colors)):
            oof = 0 < index < 5
            np.testing.assert_allclose(
                patch.get_facecolor(),
                to_rgba(pu.get_equivalent_hex(
                    color, alpha=0.33 if oof else 1.0
                )),
            )
            assert patch.get_linestyle() == ("--" if oof else "-")


def test_candidate_dashboard_respects_independent_metric_basis(
    candidate_dashboard,
):
    plotter, results, calls = candidate_dashboard(multi_batch=False)
    result = results["QC-SVR"]
    result["d_ratio_evaluation_basis"] = "full_model"
    dashboard = plotter.plot_correction_candidate_dashboard(results, "SERRF")
    position = dashboard.bricks_dict[RSD_PANEL].get_xticks()[2]
    np.testing.assert_allclose(
        _values_at(calls[RSD_PANEL], position),
        result["stage_oof_qc_rsd"]["Intra-batch corrected"] * 100,
    )
    np.testing.assert_allclose(
        _values_at(calls[D_RATIO_PANEL], position),
        result["d_ratio_current_full"],
    )
    assert calls[RSD_PANEL]["artists"]["boxes"][2].get_linestyle() == "--"
    assert calls[D_RATIO_PANEL]["artists"]["boxes"][2].get_linestyle() == "-"
    for name in (RSD_PANEL, D_RATIO_PANEL):
        labels = [
            tick.get_text()
            for tick in dashboard.bricks_dict[name].get_xticklabels()
        ]
        assert all(
            "Inter" not in label and "Intra" not in label for label in labels
        )


def test_candidate_dashboard_marks_selected_method_not_first_rank(
    candidate_dashboard,
):
    plotter, results, calls = candidate_dashboard()
    dashboard = plotter.plot_correction_candidate_dashboard(
        results, "RUV-III"
    )
    for name in (RSD_PANEL, D_RATIO_PANEL):
        marked = [
            tick.get_text()
            for tick in dashboard.bricks_dict[name].get_xticklabels()
            if tick.get_text().startswith("* ")
        ]
        assert marked == ["* RUV-III"]


@pytest.mark.parametrize("failure", ["required_oof", "score_none", "score_nan"])
def test_candidate_dashboard_omits_ineligible_and_marks_unscored_as_na(
    candidate_dashboard, failure, monkeypatch,
):
    from pimqc.plotting.correction import scorecards

    warnings = []
    monkeypatch.setattr(
        scorecards.logger, "warning", lambda *args: warnings.append(args)
    )
    plotter, results, calls = candidate_dashboard()
    method = "robust QC-RLSC"
    result = results[method]
    if failure == "required_oof":
        result["validation"].update(
            evaluation_basis="unavailable", eligible_for_auto=False
        )
        result["stage_oof_qc_rsd"] = {}
        result["final_rsd_oof"] = None
        # This descriptive full-fit D-ratio must not stand in for a score.
        result["d_ratio_evaluation_basis"] = "full_model"
    else:
        result["auto_score"] = None if failure == "score_none" else float("nan")
    dashboard = plotter.plot_correction_candidate_dashboard(results, "SERRF")
    for name in (RSD_PANEL, D_RATIO_PANEL):
        axis = dashboard.bricks_dict[name]
        labels = [tick.get_text() for tick in axis.get_xticklabels()]
        if failure == "required_oof":
            assert all("robust" not in label for label in labels)
            assert len(axis.get_xticks()) == 6
            # Plot filtering must preserve the original audit result.
            assert method in results
            continue
        candidate_index = next(
            index for index, label in enumerate(labels) if "robust" in label
        )
        position = axis.get_xticks()[candidate_index]
        assert len(axis.get_xticks()) == 7
        assert not np.isclose(calls[name]["positions"], position).any()
        assert any(
            text.get_text() == "N/A"
            and np.isclose(text.get_position()[0], position)
            for text in axis.texts
        )

    if failure == "required_oof":
        # Descriptive full-fit preservation must not make the omitted
        # candidate reappear in either scorecard.
        for candidate in results.values():
            candidate.update(
                technical_precision_score=0.6,
                d_ratio_preservation_score=0.7,
                sample_structure_score=0.8,
                sample_structure_metrics={
                    "sample_structure_trustworthiness": 0.8,
                    "sample_structure_rank_preservation": 0.8,
                    "sample_structure_scale_preservation": 0.8,
                },
            )
        for render in (
            plotter.plot_correction_score_summary,
            plotter.plot_correction_preservation_scorecard,
        ):
            _, ax = plt.subplots()
            render(results, "SERRF", ax=ax)
            assert all("robust" not in t.get_text() for t in ax.get_yticklabels())
            _, empty_ax = plt.subplots()
            render({method: result}, "SERRF", ax=empty_ax)
            assert not empty_ax.axison
        assert len(warnings) == 1
        assert "robust QC-RLSC" in str(warnings[0])
        assert plotter.plot_correction_candidate_dashboard(
            {method: result}, "SERRF"
        ) is None


def test_candidate_dashboard_does_not_substitute_full_for_missing_oof(
    candidate_dashboard,
):
    plotter, results, calls = candidate_dashboard()
    result = results["QC-SVR"]
    result["stage_oof_qc_rsd"]["Inter-batch corrected"] = pd.Series(dtype=float)
    result["d_ratio_current_oof"] = pd.Series([np.nan])
    dashboard = plotter.plot_correction_candidate_dashboard(results, "SERRF")
    position = dashboard.bricks_dict[RSD_PANEL].get_xticks()[2]
    for name in (RSD_PANEL, D_RATIO_PANEL):
        assert not np.isclose(calls[name]["positions"], position).any()
        assert any(
            text.get_text() == "N/A"
            and np.isclose(text.get_position()[0], position)
            for text in dashboard.bricks_dict[name].texts
        )
