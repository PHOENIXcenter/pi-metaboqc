"""Public plotter for signal-correction results.

Diagnostics, scorecards, internal-standard panels, and dashboard composition
live in separate modules and share state through ``CorrectionPlotter``.
"""

from __future__ import annotations

from ..base import BasePlotter
from ..payloads import CorrectionPlotPayload
from .dashboards import CorrectionDashboardMixin
from .diagnostics import CorrectionDiagnosticsMixin
from .internal_standards import CorrectionInternalStandardMixin
from .scorecards import CorrectionScorecardMixin


class CorrectionPlotter(
    CorrectionDiagnosticsMixin,
    CorrectionScorecardMixin,
    CorrectionDashboardMixin,
    CorrectionInternalStandardMixin,
    BasePlotter,
):
    """Plotting suite for correction evaluation and diagnostics."""

    def __init__(self, payload: CorrectionPlotPayload) -> None:
        """Initialize from processor-free correction plot inputs."""
        if not isinstance(payload, CorrectionPlotPayload):
            raise TypeError("CorrectionPlotter requires CorrectionPlotPayload.")
        super().__init__(payload=payload)
        self.corr = payload.primary_data
