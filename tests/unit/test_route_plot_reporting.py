"""Operational routes remain renderable for current and saved legacy audits."""

import re

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pimqc.plotting.filtering import FilteringPlotter
from pimqc.plotting.payloads import FilteringPlotPayload
from pimqc.reporting import NarrativeStatsReporter
from tests.unit.test_independent_filter_stages import _quality_dataset
from tests.unit.test_reporting_contract import _report_input


def _plotter() -> FilteringPlotter:
    """Return a plotter without running filtering or model fitting."""
    return FilteringPlotter(FilteringPlotPayload(data=_quality_dataset()))


def test_route_plotter_accepts_canonical_only_feature_indices():
    """Canonical audit names and legacy rendering access share one index."""
    indices = {
        "idx_r_route": pd.Index(["f_stable", "f_other"]),
        "idx_s_route": pd.Index(["f_unstable"]),
    }
    plotter = FilteringPlotter(FilteringPlotPayload(
        data=_quality_dataset(), audit_tables=indices,
    ))
    pd.testing.assert_index_equal(
        plotter.audit_tables["idx_mar"], indices["idx_r_route"],
    )
    pd.testing.assert_index_equal(
        plotter.audit_tables["idx_mnar"], indices["idx_s_route"],
    )
    assert set(indices) == {"idx_r_route", "idx_s_route"}


@pytest.mark.parametrize("legacy", [True, False])
def test_route_flowchart_counts_current_and_saved_labels(legacy):
    """Translating display labels must neither drop nor reassign features."""
    statuses = ["MAR", "MNAR (Group)", "MNAR (QC)", "INVALID"]
    tracking = pd.DataFrame({"Stage1_Status": statuses})
    if not legacy:
        tracking = FilteringPlotter._route_tracking(tracking)
    original = tracking.copy(deep=True)
    figure, axis = plt.subplots()
    try:
        _plotter()._plot_mv_filtering_flowchart(
            tracking, axis, 0.8, 0.2, 0.5, has_group_info=True,
        )
        labels = [artist.get_text() for artist in axis.texts]
        text = "\n".join(labels)
        assert "R-route Eligibility" in text
        assert "R-route" in labels
        assert "S-route Group" in labels
        assert "S-route QC" in labels
        assert labels.count("(n=1)") == 4
        assert not re.search(r"\b(?:MAR|MNAR)\b", text)
        pd.testing.assert_frame_equal(tracking, original)
    finally:
        plt.close(figure)


def test_route_histogram_preserves_legacy_feature_counts():
    """Old status data and palette render the same histogram as new ones."""
    plotter = _plotter()
    tracking = pd.DataFrame({
        "Stage1_Status": ["MAR", "MAR", "INVALID"],
        "MV": [10.0, 20.0, 90.0],
    })
    figure, axes = plt.subplots(1, 2)
    try:
        for axis, source, label in [
            (axes[0], tracking, "MAR"),
            (axes[1], plotter._route_tracking(tracking), "R-route"),
        ]:
            plotter._plot_cutoff_histogram(
                source, "MV", "Stage1_Status", 0.5,
                {label: "#8ebad9", "INVALID": "#808080"},
                [label, "INVALID"], axis, "Eligibility", "Missingness (%)",
            )
        heights = [[p.get_height() for p in axis.patches] for axis in axes]
        assert heights[0] == heights[1]
        assert sum(heights[0]) == 3
        for axis in axes:
            labels = "\n".join(
                artist.get_text() for artist in axis.findobj()
                if hasattr(artist, "get_text")
            )
            assert "R-route" in labels
            assert not re.search(r"\b(?:MAR|MNAR)\b", labels)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("legacy", [True, False])
def test_route_reports_accept_canonical_and_legacy_imputation_stats(
    tmp_path, legacy,
):
    """Old audit keys remain readable without exposing mechanism labels."""
    report = _report_input()
    if not legacy:
        feature_filter = report.pipeline_metrics["high_mv_feature_filtering"]
        feature_filter["feature_wise"]["missing_classification"] = {
            "r_route_count": 1, "s_route_count": 1,
        }
        imputation = report.pipeline_metrics["missing_value_imputation"]
        imputation["feature_distribution"] = {
            "r_route_count": 1, "s_route_count": 1,
        }
        imputation["strategies"] = {
            "s_route_method": "row-wise", "s_route_fraction": 0.5,
        }
    NarrativeStatsReporter(base_dir=str(tmp_path)).generate_markdown(
        report, report_folder="report",
    )
    for name in ["Report_Comprehensive.md", "Report_Brief.md"]:
        text = (tmp_path / "report" / name).read_text(encoding="utf-8")
        assert "R-route" in text and "S-route" in text
        assert "reconstruction" in text
        assert "special handling" in text or "special-handling" in text
        assert "**row-wise**" in text
        assert "**KNN**" in text
        assert not re.search(r"\b(?:MAR|MNAR)\b", text)
        assert "missing-at-random" not in text
        assert "missing-not-at-random" not in text
