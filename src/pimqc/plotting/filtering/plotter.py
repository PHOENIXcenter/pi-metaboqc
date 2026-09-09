"""Public plotter for sample and feature filtering.

Diagnostic panels, the filtering flowchart, and dashboard composition live in
separate modules and share state through ``FilteringPlotter``.
"""

from __future__ import annotations

from ..base import BasePlotter
from ..payloads import FilteringPlotPayload
from .dashboards import FilteringDashboardMixin
from .diagnostics import FilteringDiagnosticsMixin
from .flowchart import FilteringFlowchartMixin


class FilteringPlotter(
    FilteringDiagnosticsMixin,
    FilteringFlowchartMixin,
    FilteringDashboardMixin,
    BasePlotter,
):
    """Plotting suite for sample and feature filtering outcomes."""

    def __init__(
        self,
        payload: FilteringPlotPayload,
    ) -> None:
        """Initialize from processor-free filtering plot inputs."""
        if not isinstance(payload, FilteringPlotPayload):
            raise TypeError("FilteringPlotter requires FilteringPlotPayload.")
        super().__init__(payload=payload)
        self.engine = payload.primary_data
        self.audit_tables = dict(payload.audit_tables)
