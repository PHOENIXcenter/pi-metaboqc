"""Regression tests for side-effect-free, composition-based runtime behavior."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pimqc.constants import DEFAULT_RANDOM_SEED
from pimqc.core import MetaboDataset
from pimqc.dataset import MetaboDatasetBuilder
from pimqc.pipeline import run_pipeline
from pimqc.plotting.base import BasePlotter
from pimqc.plotting import plot_utils as pu
from pimqc.plotting.payloads import AssessmentPlotPayload, snapshot_dataset
from pimqc.processing.assessment import QualityAssessor
from pimqc.processing.correction import SignalCorrector
from pimqc.processing.filtering import (
    FeatureFilter,
    FeatureMissingValueFilter,
    FeatureQualityFilter,
    SampleMissingValueFilter,
)
from pimqc.processing.imputation import MissingValueImputer
from pimqc.processing.normalization import DataNormalizer
from pimqc.processing import DatasetAuditPayload
from pimqc.processing.stage import StageResult


def _minimal_dataset(*, missing: bool = False) -> MetaboDataset:
    intensity = pd.DataFrame(
        [
            [np.nan if missing else 1.0, 2.0, 2.0, 4.0],
            [4.0, 4.0, 4.0, 4.0],
        ],
        index=pd.Index(["F1", "F2"], name="Metabolite"),
        columns=pd.Index(["QC1", "QC2", "S1", "S2"], name="Sample Name"),
    )
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "QC", "Sample", "Sample"],
            "Batch": ["B1"] * 4,
            "Inject Order": [1, 2, 3, 4],
        },
        index=intensity.columns,
    )
    feature_metadata = pd.DataFrame(
        {"missingness_type": ["MAR", "MAR"]},
        index=intensity.index,
    )
    return MetaboDataset.from_tables(intensity, metadata, feature_metadata)


def test_dataset_construction_does_not_reset_numpy_global_rng() -> None:
    """Constructing the domain model must not alter application RNG state."""
    np.random.seed(DEFAULT_RANDOM_SEED)
    expected_first = np.random.random()
    expected_second = np.random.random()

    np.random.seed(DEFAULT_RANDOM_SEED)
    assert np.random.random() == expected_first
    _minimal_dataset()
    assert np.random.random() == expected_second


def test_visualizer_construction_does_not_patch_matplotlib_globals() -> None:
    """Visualizers must leave Axes construction and rcParams untouched."""
    from pimqc.plotting.assessment import AssessmentPlotter

    original_axes_init = matplotlib.axes.Axes.__init__
    original_font_type = mpl.rcParams["pdf.fonttype"]
    dataset = _minimal_dataset()
    AssessmentPlotter(
        AssessmentPlotPayload(
            data=snapshot_dataset(dataset),
            is_multi_batch=False,
        )
    )

    assert matplotlib.axes.Axes.__init__ is original_axes_init
    assert mpl.rcParams["pdf.fonttype"] == original_font_type


def test_vector_save_keeps_svg_text_editable(tmp_path: Path) -> None:
    """Save SVG labels as text while restoring host rcParams afterwards."""
    from pimqc.plotting.assessment import AssessmentPlotter

    original_svg_fonttype = mpl.rcParams["svg.fonttype"]
    visualizer = AssessmentPlotter(
        AssessmentPlotPayload(snapshot_dataset(_minimal_dataset()), False)
    )
    figure, axis = plt.subplots()
    axis.set_title("Editable title")
    axis.set_xlabel("Selectable x label")
    axis.set_ylabel("Selectable y label")
    axis.plot([0.0, 1.0], [0.0, 1.0])
    output_path = tmp_path / "editable_vector.svg"

    visualizer.save_and_close_fig(
        figure,
        file_path=str(output_path),
        save_format="svg",
    )

    svg_text = output_path.read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert "Editable title" in svg_text
    assert "Selectable x label" in svg_text
    assert "Selectable y label" in svg_text
    assert mpl.rcParams["svg.fonttype"] == original_svg_fonttype


def test_patchwork_save_keeps_svg_and_pdf_text_editable(tmp_path: Path) -> None:
    """Preserve text elements and Unicode maps in dashboard vector exports."""
    import patchworklib as pw

    from pimqc.plotting.assessment import AssessmentPlotter

    pw.clear()
    visualizer = AssessmentPlotter(
        AssessmentPlotPayload(snapshot_dataset(_minimal_dataset()), False)
    )
    brick = pw.Brick(figsize=(3.0, 2.0), label="editable_vector_brick")
    brick.set_title("Editable dashboard title")
    brick.set_xlabel("Selectable dashboard label")
    brick.plot([0.0, 1.0], [0.0, 1.0])
    output_base = tmp_path / "editable_dashboard"

    visualizer.save_and_show_pw(
        brick,
        file_path=str(output_base),
        show_plot=False,
        save_format=["svg", "pdf"],
    )

    svg_text = output_base.with_suffix(".svg").read_text(encoding="utf-8")
    pdf_bytes = output_base.with_suffix(".pdf").read_bytes()
    assert "<text" in svg_text
    assert "Editable dashboard title" in svg_text
    assert "Selectable dashboard label" in svg_text
    assert b"/Subtype /CIDFontType2" in pdf_bytes
    assert b"/ToUnicode" in pdf_bytes


def test_dashboard_display_width_follows_grid_shape() -> None:
    """Use 60/40/20% inline widths without changing export dimensions."""
    import patchworklib as pw

    def make_bricks(count: int) -> list[object]:
        return [
            pw.Brick(figsize=(2.0, 2.0), label=f"width_{count}_{index}")
            for index in range(count)
        ]

    pw.clear()
    single = make_bricks(1)[0]
    assert BasePlotter._dashboard_grid_shape(single) == (1, 1)
    assert (
        BasePlotter.resolve_dashboard_display_width(single)
        == pu.SINGLE_PANEL_DASHBOARD_DISPLAY_WIDTH
    )

    pw.clear()
    two_by_two_bricks = make_bricks(4)
    two_by_two = (two_by_two_bricks[0] | two_by_two_bricks[1]) / (
        two_by_two_bricks[2] | two_by_two_bricks[3]
    )
    assert BasePlotter._dashboard_grid_shape(two_by_two) == (2, 2)
    assert (
        BasePlotter.resolve_dashboard_display_width(two_by_two)
        == pu.TWO_BY_TWO_DASHBOARD_DISPLAY_WIDTH
    )

    pw.clear()
    three_columns = make_bricks(3)
    three_column_dashboard = (
        three_columns[0] | three_columns[1] | three_columns[2]
    )
    assert BasePlotter._dashboard_grid_shape(three_column_dashboard) == (1, 3)
    assert (
        BasePlotter.resolve_dashboard_display_width(three_column_dashboard)
        == pu.DEFAULT_DASHBOARD_DISPLAY_WIDTH
    )
    assert (
        BasePlotter.resolve_dashboard_display_width(two_by_two, width="75%")
        == "75%"
    )


def test_save_and_show_pw_passes_resolved_width_to_notebook_renderer() -> None:
    """Apply the grid policy at the actual Patchwork display boundary."""
    import patchworklib as pw
    from pimqc.plotting.assessment import AssessmentPlotter

    pw.clear()
    bricks = [
        pw.Brick(figsize=(2.0, 2.0), label=f"render_width_{index}")
        for index in range(4)
    ]
    dashboard = (bricks[0] | bricks[1]) / (bricks[2] | bricks[3])
    visualizer = AssessmentPlotter(
        AssessmentPlotPayload(snapshot_dataset(_minimal_dataset()), False)
    )

    with (
        patch("pimqc.plotting.base.is_jupyter", return_value=True),
        patch.object(visualizer, "_render_jupyter_display") as render,
    ):
        visualizer.save_and_show_pw(dashboard, file_path=None)

    assert render.call_args.kwargs["width"] == "40%"


def test_stage_processors_use_composition_not_dataframe_inheritance() -> None:
    """Keep every numerical engine outside the pandas type hierarchy."""
    source = _minimal_dataset()
    stage_types = (
        QualityAssessor,
        SignalCorrector,
        FeatureFilter,
        MissingValueImputer,
        DataNormalizer,
    )

    for stage_type in stage_types:
        stage = stage_type(source)
        assert not isinstance(stage, pd.DataFrame)
        assert pd.DataFrame not in stage_type.__mro__
        assert type(stage.frame) is pd.DataFrame
        assert type(stage.dataset) is MetaboDataset
        stage.frame.iloc[0, 0] = -1
        assert source.intensity.iloc[0, 0] == 1.0


def test_complete_stage_actions_apply_execution_timing() -> None:
    """Time every complete lifecycle without wrapping calculation helpers."""
    stage_actions = (
        MetaboDatasetBuilder.run_build,
        QualityAssessor.run_assessment,
        SampleMissingValueFilter.run_filter_samples_by_missingness,
        FeatureMissingValueFilter.run_filter_features_by_missingness,
        FeatureQualityFilter.run_filter_features_by_quality,
        SignalCorrector.run_signal_correction,
        MissingValueImputer.run_imputation,
        DataNormalizer.run_normalization,
        run_pipeline,
    )

    assert all(hasattr(action, "__wrapped__") for action in stage_actions)
    assert not hasattr(FeatureFilter.classify_missing_types, "__wrapped__")


def test_correction_helpers_read_composed_dataset_roles() -> None:
    """Resolve QC and control features without subclass-only attributes."""
    processor = SignalCorrector(_minimal_dataset())

    correlation = processor._prepare_serrf_correlation_matrix()
    controls = processor._prepare_ruv_control_features()

    assert correlation is not None
    assert correlation.shape == (2, 2)
    assert set(controls) == {"F1", "F2"}


def test_stage_result_keeps_data_and_typed_audit_separate() -> None:
    """The stage contract separates transformation output from audit data."""
    dataset = _minimal_dataset()
    result = StageResult(
        data=dataset,
        audit=DatasetAuditPayload(metric_values={"score": 0.8}),
    )
    assert result.data is dataset
    assert result.audit.metrics["score"] == 0.8


def test_auto_imputation_keeps_request_and_selection_separate() -> None:
    """Preserve AUTO in report metrics after selecting a MAR candidate."""
    imputer = MissingValueImputer(
        _minimal_dataset(missing=True),
        mar_method="Auto",
        mnar_method="Row-wise",
    )
    selected_metrics = {
        "NRMSE_Low": 0.1,
        "NRMSE_High": 0.2,
        "NRMSE_Total": 0.15,
        "JSD_Total": 0.1,
        "Wasserstein_Total": 0.2,
        "Wasserstein_Normalized": 0.1,
    }
    candidate_cache = {
        "LLS": (selected_metrics, np.array([1.0]), np.array([1.0]))
    }

    with (
        patch.object(
            imputer,
            "_select_best_imputation_method",
            return_value=("LLS", candidate_cache),
        ),
        patch.object(
            imputer,
            "_apply_isolated",
            side_effect=lambda frame, _method, **_kwargs: frame.fillna(1.0),
        ),
    ):
        result = imputer.transform_imputation()

    selection = result.audit.metrics["selection"]
    assert selection["requested_method"] == "Auto"
    assert selection["selected_method"] == "LLS"
    assert selection["is_auto"] is True
    candidate_result = next(
        candidate
        for candidate in selection["candidate_results"]
        if candidate["method"] == "LLS"
    )
    assert candidate_result["jsd_total"] == 0.1
    assert candidate_result["wasserstein_normalized"] == 0.1
    assert result.audit.metrics["feature_distribution"] == {
        "mar_count": 2,
        "mnar_count": 0,
    }
    assert result.audit.requested_method == "Auto"
    assert result.audit.is_auto is True
    assert type(result.data) is MetaboDataset


def test_import_does_not_replace_subprocess_popen() -> None:
    """The public package must not monkey-patch the host subprocess module."""
    import pimqc

    assert pimqc is not None
    assert subprocess.Popen.__module__ == "subprocess"


def test_import_does_not_load_optional_r_bridge() -> None:
    """Keep R and rpy2 outside the normal Python package import path."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pimqc; print('rpy2' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
