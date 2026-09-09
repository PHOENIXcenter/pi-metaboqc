"""Public plotter for missing-value imputation results.

Diagnostics, scorecards, and dashboard composition live in separate modules
and share state through ``ImputationPlotter``.
"""

from __future__ import annotations

from ..base import BasePlotter
from ..payloads import ImputationPlotPayload
from .dashboards import ImputationDashboardMixin
from .diagnostics import ImputationDiagnosticsMixin
from .scorecards import ImputationScorecardMixin


class ImputationPlotter(
    ImputationDiagnosticsMixin,
    ImputationScorecardMixin,
    ImputationDashboardMixin,
    BasePlotter,
):
    """Plotting suite for imputation accuracy and method selection."""

    def __init__(self, payload: ImputationPlotPayload) -> None:
        """Initialize from processor-free imputation plot inputs."""
        if not isinstance(payload, ImputationPlotPayload):
            raise TypeError("ImputationPlotter requires ImputationPlotPayload.")
        super().__init__(payload=payload)
        self.raw_obj = (
            payload.raw_data.annotated_frame()
            .astype(float)
            .replace({0: float("nan")})
        )
        self.imp_obj = payload.imputed_data.annotated_frame().astype(float)
