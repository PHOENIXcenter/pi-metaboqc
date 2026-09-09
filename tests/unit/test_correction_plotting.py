"""Regression tests for correction diagnostic layout decisions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pimqc.core import MetaboDataset
from pimqc.plotting import annotation_layout as al
from pimqc.plotting.correction import CorrectionPlotter
from pimqc.plotting.payloads import CorrectionPlotPayload
from pimqc.processing.correction import SignalCorrector


def _correction_frames() -> tuple[MetaboDataset, dict[str, pd.DataFrame]]:
    """Build deterministic frames with enough QC variation for boxplots."""
    columns = pd.Index(["Q1", "Q2", "S1", "S2"], name="Sample Name")
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "QC", "Sample", "Sample"],
            "Batch": ["B1"] * 4,
            "Inject Order": [1, 2, 3, 4],
        },
        index=columns,
    )
    features = pd.Index([f"F{index}" for index in range(60)], name="Metabolite")
    rng = np.random.default_rng(73)
    source = MetaboDataset.from_tables(
        pd.DataFrame(
            rng.uniform(0.8, 1.2, size=(len(features), len(columns))),
            index=features,
            columns=columns,
        ),
        metadata,
        pd.DataFrame(index=features),
    )
    frames = {
        name: MetaboDataset.from_tables(
            source.intensity * multiplier,
            metadata,
            source.feature_metadata,
        ).annotated_frame()
        for name, multiplier in {
            "Original": 1.0,
            "Intra-batch corrected": 0.9,
            "Inter-batch corrected": 0.85,
        }.items()
    }
    return source, frames


def test_corr_rsd_uses_dynamic_legend_and_non_overlapping_annotation() -> None:
    """
    Choose the legend from plotted occupancy before placing the annotation.
    """
    import patchworklib as pw

    source, frames = _correction_frames()
    plotter = CorrectionPlotter(
        CorrectionPlotPayload(
            source_data=source,
            selected_stages={},
            candidate_results={
                "QC-RLSC": {
                    "stage_dfs": frames,
                    "stage_qc_rsd": {
                        label: SignalCorrector.extract_qc_rsd_series(frame)
                        for label, frame in frames.items()
                    },
                }
            },
            selected_prediction=None,
            internal_standard_ids=(),
            boundary_type="IQR",
        )
    )
    pw.clear()
    axis = pw.Brick(figsize=(7.0, 4.0), label="correction_rsd_layout")
    plotter.plot_corr_rsd(frames, {}, ax=axis, show_legend=True)

    legend = axis.get_legend()
    assert legend is not None
    # Matplotlib's ``best`` location is encoded as 0. A fixed lower-right
    # location (4) would defeat the occupancy-aware legend/annotation pair.
    assert legend._loc == 0
    annotation = axis.texts[-1]
    axis.figure.canvas.draw()
    annotation_box = (
        annotation.get_bbox_patch()
        .get_window_extent(axis.figure.canvas.get_renderer())
        .transformed(axis.transAxes.inverted())
    )
    legend_box = al.legend_bboxes_in_axes(axis, legend=legend)[0]
    overlap_width = max(
        0.0,
        min(annotation_box.x1, legend_box[2])
        - max(annotation_box.x0, legend_box[0]),
    )
    overlap_height = max(
        0.0,
        min(annotation_box.y1, legend_box[3])
        - max(annotation_box.y0, legend_box[1]),
    )
    assert overlap_width * overlap_height == 0.0
    plt.close(axis.figure)
