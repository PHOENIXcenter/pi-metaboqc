"""Plotting infrastructure and stage-specific plotters for pi-metaboqc.

The package owns the shared figure lifecycle, reusable plotting primitives,
collision-aware annotation layout, dataset diagnostics, and plotters for every
processing stage. Numerical stage logic remains under
:mod:`pimqc.processing`.
"""

from .base import BasePlotter
from .payloads import (
    AssessmentPlotPayload,
    CorrectionPlotPayload,
    DatasetPlotPayload,
    FilteringPlotPayload,
    ImputationPlotPayload,
    NormalizationPlotPayload,
    PlotPayload,
    snapshot_dataset,
    snapshot_plot_value,
)
from .sample_structure import plot_sample_structure_change_map

__all__ = [
    "AssessmentPlotPayload",
    "BasePlotter",
    "CorrectionPlotPayload",
    "DatasetPlotPayload",
    "FilteringPlotPayload",
    "ImputationPlotPayload",
    "NormalizationPlotPayload",
    "PlotPayload",
    "plot_sample_structure_change_map",
    "snapshot_dataset",
    "snapshot_plot_value",
]
