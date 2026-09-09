"""Public plotter for quality-assessment diagnostics.

The concrete panels are grouped by purpose in sibling modules, while
``AssessmentPlotter`` owns their shared payload state and public API.
"""

from __future__ import annotations

from ..base import BasePlotter
from ..payloads import AssessmentPlotPayload
from .control_charts import AssessmentControlChartMixin
from .dashboards import AssessmentDashboardMixin
from .heatmaps import AssessmentHeatmapMixin
from .outliers import AssessmentOutlierMixin
from .pca import AssessmentPcaMixin


class AssessmentPlotter(
    AssessmentHeatmapMixin,
    AssessmentPcaMixin,
    AssessmentOutlierMixin,
    AssessmentControlChartMixin,
    AssessmentDashboardMixin,
    BasePlotter,
):
    """Plotting suite for metabolomics data quality assessment."""

    def __init__(self, payload: AssessmentPlotPayload) -> None:
        """Initialize from processor-free assessment plot inputs."""
        if not isinstance(payload, AssessmentPlotPayload):
            raise TypeError("AssessmentPlotter requires AssessmentPlotPayload.")
        super().__init__(payload=payload)
