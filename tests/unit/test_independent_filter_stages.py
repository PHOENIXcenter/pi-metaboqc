"""Regression tests for independently runnable filtering actions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pimqc.core import MetaboDataset
from pimqc.plotting.filtering import FilteringPlotter
from pimqc.processing.filtering import (
    FeatureMissingValueFilter,
    FeatureQualityFilter,
    FilteringOrchestrator,
    SampleMissingValueFilter,
)
from pimqc.processing.imputation import MissingValueImputer


def _quality_dataset() -> MetaboDataset:
    """Create a small table with one QC-RSD failure and no MV labels."""
    samples = pd.DataFrame(
        {
            "Sample Type": [
                "Sample",
                "Sample",
                "Blank",
                "Blank",
                "QC",
                "QC",
                "QC",
            ],
            "Batch": ["B1"] * 7,
            "Inject Order": list(range(1, 8)),
            "Bio Group": ["G", "G", np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=["S1", "S2", "B1", "B2", "Q1", "Q2", "Q3"],
    )
    intensity = pd.DataFrame(
        {
            "S1": [10.0, 10.0, 10.0],
            "S2": [10.0, 10.0, 10.0],
            "B1": [1.0, 1.0, 1.0],
            "B2": [1.0, 1.0, 1.0],
            "Q1": [10.0, 10.0, 10.0],
            "Q2": [10.0, 20.0, 10.0],
            "Q3": [10.0, 10.0, 10.0],
        },
        index=["f_stable", "f_unstable", "f_other"],
    )
    return MetaboDataset.from_tables(
        intensity,
        samples,
        pd.DataFrame({"mz": [100.0, 101.0, 102.0]}, index=intensity.index),
    )


def _missingness_layout_dataset(
    *,
    has_missing_values: bool,
    has_biological_groups: bool,
) -> MetaboDataset:
    """Derive one deterministic layout fixture from the quality table."""
    source = _quality_dataset()
    intensity = source.intensity.copy(deep=True)
    if has_missing_values:
        intensity.loc["f_stable", "S1"] = np.nan
    sample_metadata = source.sample_metadata.copy(deep=True)
    if not has_biological_groups:
        sample_metadata = sample_metadata.drop(columns="Bio Group")
    return MetaboDataset.from_tables(
        intensity,
        sample_metadata,
        source.feature_metadata,
    )


def test_quality_filter_without_upstream_result_is_quality_only(
    monkeypatch,
) -> None:
    """Do not infer MAR/MNAR when a quality action runs by itself."""
    dataset = _quality_dataset()

    def fail_if_classified(self):  # pragma: no cover - failure path only
        raise AssertionError("quality-only mode must not classify missingness")

    monkeypatch.setattr(
        "pimqc.processing.filtering.analysis.FeatureFilter.classify_missing_types",
        fail_if_classified,
    )

    result = FeatureQualityFilter(
        dataset,
        qc_rsd_tol=0.3,
        blank_qc_ratio_tol=0.2,
    ).filter_features_by_quality()

    assert result.audit.metrics["filtering_mode"] == "quality_only"
    assert result.audit.metrics["missingness_classified"] is False
    assert result.audit.metrics["feature_retention"]["pre_stage2"] == {
        "total": 3,
        "mar_count": 0,
        "mnar_count": 0,
        "unclassified_count": 3,
    }
    assert result.data.feature_ids.tolist() == ["f_stable", "f_other"]
    assert set(result.audit.feature_tracking["Base_Type"]) == {"UNCLASSIFIED"}
    assert (
        result.audit.feature_tracking.loc["f_unstable", "RSD_Check"] == "Failed"
    )


def test_sample_filter_exposes_standalone_plot_payload() -> None:
    """Keep sample attrition data and its resolved threshold renderable."""
    result = SampleMissingValueFilter(
        _quality_dataset(),
        sample_mv_tol=0.4,
    ).filter_samples_by_missingness()

    payload = result.audit.plot_payload
    assert payload is not None
    assert payload.sample_mv_tolerance == 0.4
    pd.testing.assert_frame_equal(
        payload.audit_tables["sample_tracking"],
        result.audit.sample_tracking,
    )


def test_quality_dashboard_uses_explicit_feature_mv_history() -> None:
    """Render Raw and High-MV counts when the upstream audit is supplied."""
    tracking = pd.DataFrame(
        {
            "Stage1_Status": ["MAR", "MNAR (QC)", "MAR", "INVALID"],
        },
        index=["f_stable", "f_unstable", "f_other", "f_invalid"],
    )
    result = FeatureQualityFilter(
        _quality_dataset(),
        missingness_metadata=tracking,
        qc_rsd_tol=0.3,
        blank_qc_ratio_tol=0.2,
    ).filter_features_by_quality()

    counts = result.audit.plot_payload.audit_tables["feature_counts"]
    assert counts["raw"] == 4
    assert counts["post_stage1"] == 3

    figure = FilteringPlotter(
        result.audit.plot_payload
    )._plot_retained_count_steps()
    axis = figure.axes[0]
    assert [label.get_text() for label in axis.get_xticklabels()] == [
        "Raw\nData",
        "High-MV\nCheck",
        "QC/Blank\nCheck",
        "QC RSD\nCheck",
    ]


def test_quality_dashboard_marks_audited_feature_mv_noop() -> None:
    """Distinguish a skipped Feature MV action from an unexecuted action."""
    source = _missingness_layout_dataset(
        has_missing_values=False,
        has_biological_groups=True,
    )
    feature_result = FeatureMissingValueFilter(
        source
    ).filter_features_by_missingness()
    quality_result = FeatureQualityFilter(
        feature_result.data,
        missingness_metadata=feature_result.audit.feature_tracking,
    ).filter_features_by_quality()

    payload = quality_result.audit.plot_payload
    assert payload.audit_tables["feature_mv_status"] == "skipped"
    counts = payload.audit_tables["feature_counts"]
    assert counts["raw"] == counts["post_stage1"] == 3

    figure = FilteringPlotter(payload)._plot_retained_count_steps()
    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert labels[:2] == ["Raw\nData", "High-MV\nSkipped"]


def test_blank_qc_plot_resolves_x_and_y_limits_independently() -> None:
    """A large Blank value must not expand the QC x-axis range."""
    result = FeatureQualityFilter(
        _quality_dataset(),
        qc_rsd_tol=0.3,
        blank_qc_ratio_tol=0.2,
    ).filter_features_by_quality()
    payload = result.audit.plot_payload
    tables = dict(payload.audit_tables)
    tables["blank_mean"] = pd.Series(
        [1000.0, 0.0, 800.0],
        index=["f_stable", "f_unstable", "f_other"],
    )
    tables["qc_mean"] = pd.Series(
        [0.0, 2.0, 3.0],
        index=["f_stable", "f_unstable", "f_other"],
    )
    payload = replace(payload, audit_tables=tables)
    figure = FilteringPlotter(payload)._plot_qc_blank_scatter()
    try:
        axis = figure.axes[0]
        x_lower, x_upper = axis.get_xlim()
        y_lower, y_upper = axis.get_ylim()
        assert x_lower < 0.0
        assert y_lower < 0.0
        assert x_upper < y_upper
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_feature_mv_without_sample_result_omits_sample_panel(
    monkeypatch,
) -> None:
    """Do not draw a blank sample panel for an independent feature action."""
    source = _quality_dataset()
    intensity = source.intensity.copy()
    intensity.loc["f_stable", "S1"] = np.nan
    ungrouped = MetaboDataset.from_tables(
        intensity,
        source.sample_metadata.drop(columns="Bio Group"),
        source.feature_metadata,
    )
    result = FeatureMissingValueFilter(
        ungrouped
    ).filter_features_by_missingness()
    audit = result.audit

    assert not audit.valid_biological_groups
    assert audit.sample_tracking.empty
    assert audit.tables["idx_mnar_group"].empty
    assert audit.feature_tracking["Max_Group_MV_Pct"].isna().all()

    plotter = FilteringPlotter(audit.plot_payload)

    def fail_if_sample_panel_is_drawn(*args, **kwargs):
        raise AssertionError("independent Feature MV must omit sample panel")

    monkeypatch.setattr(
        plotter,
        "_plot_sample_mv_stripplot",
        fail_if_sample_panel_is_drawn,
    )
    payload = audit.plot_payload
    dashboard = plotter.plot_mv_filtering_dashboard(
        tracking_df=audit.feature_tracking,
        active_base_tol=payload.active_base_tolerance,
        mnar_group_mv_tol=payload.mnar_group_mv_tolerance,
        mnar_qc_mv_tol=payload.mnar_qc_mv_tolerance,
        mnar_int_threshold=payload.mnar_intensity_threshold,
        mnar_intensity_pct=payload.mnar_intensity_percentile,
    )
    assert dashboard is not None


@pytest.mark.parametrize("sample_filter_run", [False, True])
@pytest.mark.parametrize("has_missing_values", [False, True])
@pytest.mark.parametrize("has_biological_groups", [False, True])
def test_feature_mv_dashboard_composes_all_degradation_states(
    sample_filter_run: bool,
    has_missing_values: bool,
    has_biological_groups: bool,
) -> None:
    """Compose executed states and omit dashboards for skipped MV actions."""
    source = _missingness_layout_dataset(
        has_missing_values=has_missing_values,
        has_biological_groups=has_biological_groups,
    )
    sample_result = (
        SampleMissingValueFilter(
            source,
            sample_mv_tol=0.9,
        ).filter_samples_by_missingness()
        if sample_filter_run
        else None
    )
    feature_input = sample_result.data if sample_result is not None else source
    result = FeatureMissingValueFilter(
        feature_input,
        sample_result=sample_result,
    ).filter_features_by_missingness()
    payload = result.audit.plot_payload
    dashboard = FilteringPlotter(payload).plot_mv_filtering_dashboard(
        tracking_df=result.audit.feature_tracking,
        active_base_tol=payload.active_base_tolerance,
        mnar_group_mv_tol=payload.mnar_group_mv_tolerance,
        mnar_qc_mv_tol=payload.mnar_qc_mv_tolerance,
        mnar_int_threshold=payload.mnar_intensity_threshold,
        mnar_intensity_pct=payload.mnar_intensity_percentile,
    )

    if not has_missing_values:
        assert result.audit.execution_status == "skipped"
        assert result.audit.missing_value_count == 0
        assert dashboard is None
        assert set(result.audit.feature_tracking["Stage1_Status"]) == {
            "NOT_APPLICABLE"
        }
        assert result.data.feature_ids.equals(source.feature_ids)
        return

    labels = (
        set(dashboard.bricks_dict)
        if hasattr(dashboard, "bricks_dict")
        else {dashboard.get_label()}
    )
    visible_labels = {label for label in labels if not label.startswith("ax_")}

    expected = {"flowchart", "s2", "s3"}
    if sample_filter_run:
        expected.add("sample_mv")
    if has_biological_groups:
        expected.add("s1")
    assert result.audit.execution_status == "completed"
    assert visible_labels == expected

    flowchart = dashboard.bricks_dict["flowchart"]
    flow_text = "\n".join(text.get_text() for text in flowchart.texts)
    if has_biological_groups:
        assert "Group Rescue" in flow_text
        assert "MAR Eligibility" in flow_text
    else:
        assert "Group Rescue" not in flow_text
        assert "QC MV Check" in flow_text
        assert "QC MV <=" in flow_text
        qc_rescue_text = "\n".join(
            artist.get_text()
            for artist in dashboard.bricks_dict["s2"].findobj()
            if hasattr(artist, "get_text")
        )
        qc_check_text = "\n".join(
            artist.get_text()
            for artist in dashboard.bricks_dict["s3"].findobj()
            if hasattr(artist, "get_text")
        )
        assert "Min Group MV" not in qc_rescue_text
        assert "Min group MV" not in qc_check_text
        assert "QC MV = 30%" in qc_check_text


def test_sample_mv_action_exports_audit_without_a_standalone_dashboard(
    tmp_path,
) -> None:
    """Sample MV output is tabular; its panel is composed by Feature MV."""
    result = SampleMissingValueFilter(
        _missingness_layout_dataset(
            has_missing_values=True,
            has_biological_groups=True,
        ),
        sample_mv_tol=0.4,
    ).run_filter_samples_by_missingness(output_dir=tmp_path)

    assert result.audit.execution_status == "completed"
    assert (tmp_path / "Filtered_Data_High-MV_Samples.csv").is_file()
    assert (tmp_path / "Filtering_Tracking_High-MV_Samples.csv").is_file()
    assert not (tmp_path / "Sample_MV_Filtering_Dashboard.svg").exists()


def test_orchestrator_groups_missingness_artifacts_in_one_directory(
    tmp_path,
    monkeypatch,
) -> None:
    """Share one Step 02 directory without merging the two stage results."""
    output_dir = tmp_path / "02_MV_Filtered"
    monkeypatch.setattr(
        "pimqc.processing.filtering.runner."
        "FeatureMissingValueFilteringStageRunner.render",
        lambda self, result: None,
    )

    result = FilteringOrchestrator(
        _missingness_layout_dataset(
            has_missing_values=True,
            has_biological_groups=True,
        )
    ).run_missingness(
        output_dir=output_dir,
        sample_overrides={"sample_mv_tol": 0.9},
    )

    assert result.sample_result is not None
    assert result.feature_result is not None
    assert result.sample_result is not result.feature_result
    assert (output_dir / "Filtered_Data_High-MV_Samples.csv").is_file()
    assert (output_dir / "Filtering_Tracking_High-MV_Samples.csv").is_file()
    assert (output_dir / "Filtered_Data_MV_Features.csv").is_file()
    assert (output_dir / "Filtering_Tracking_MV_Features.csv").is_file()
    assert {path.name for path in tmp_path.iterdir()} == {"02_MV_Filtered"}


def test_feature_mv_action_uses_shared_dashboard_name(
    tmp_path,
    monkeypatch,
) -> None:
    """Use the report-facing filename for the combined MV dashboard."""
    saved_paths = []
    monkeypatch.setattr(
        FilteringPlotter,
        "plot_mv_filtering_dashboard",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(
        FilteringPlotter,
        "save_and_show_pw",
        lambda self, pw_obj, file_path: saved_paths.append(file_path),
    )

    FeatureMissingValueFilter(
        _missingness_layout_dataset(
            has_missing_values=True,
            has_biological_groups=True,
        )
    ).run_filter_features_by_missingness(output_dir=tmp_path)

    assert len(saved_paths) == 1
    assert saved_paths[0].endswith("MV_Classification_Dashboard.svg")


def test_imputation_without_missing_values_is_skipped() -> None:
    """Propagate complete target data without selecting an imputer."""
    dataset = _quality_dataset()
    result = MissingValueImputer(dataset).transform_imputation()

    assert result.audit.skipped is True
    assert result.audit.selected_method == "Not required"
    assert result.audit.plot_payload is None
    pd.testing.assert_frame_equal(result.data.intensity, dataset.intensity)


def test_quality_filter_uses_explicit_missingness_labels_when_present() -> None:
    """MNAR labels supplied by the previous action retain RSD exemption."""
    dataset = _quality_dataset()
    dataset = dataset.with_intensity(
        dataset.intensity,
        feature_metadata=pd.DataFrame(
            {
                "missingness_type": ["MAR", "MNAR", "MAR"],
            },
            index=["f_stable", "f_unstable", "f_other"],
        ),
    )

    result = FeatureQualityFilter(
        dataset,
        qc_rsd_tol=0.3,
        blank_qc_ratio_tol=0.2,
    ).filter_features_by_quality()

    assert result.audit.metrics["filtering_mode"] == "missingness_aware"
    assert set(result.data.feature_ids) == {
        "f_stable",
        "f_unstable",
        "f_other",
    }
    assert (
        result.audit.feature_tracking.loc["f_unstable", "RSD_Check"]
        == "Exempted (MNAR)"
    )


def test_orchestrator_quality_action_does_not_auto_run_missingness() -> None:
    """Calling only ``run_quality`` produces a quality-only result."""
    result = FilteringOrchestrator(_quality_dataset()).run_quality()

    assert result.sample_result is None
    assert result.feature_result is None
    assert result.quality_result is not None
    assert (
        result.quality_result.audit.metrics["filtering_mode"] == "quality_only"
    )


def test_feature_mv_filter_reports_missing_qc_as_precondition_error() -> None:
    """Do not silently convert a missing-QC input into an empty result."""
    source = _quality_dataset()
    keep_samples = source.sample_metadata["Sample Type"] != "QC"
    metadata = source.sample_metadata.loc[keep_samples].copy()
    intensity = source.intensity.loc[:, metadata.index]
    intensity.iloc[0, 0] = np.nan
    dataset = MetaboDataset.from_tables(
        intensity,
        metadata,
        source.feature_metadata,
    )

    with pytest.raises(ValueError, match="Missing-value classification failed"):
        FeatureMissingValueFilter(dataset).filter_features_by_missingness()
