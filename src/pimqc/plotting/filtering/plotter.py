"""Public plotter for sample and feature filtering.

Diagnostic panels, the filtering flowchart, and dashboard composition live in
separate modules and share state through ``FilteringPlotter``.
"""

from __future__ import annotations

import pandas as pd

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
        for current, legacy in (
            ("idx_r_route", "idx_mar"),
            ("idx_s_route", "idx_mnar"),
        ):
            if current in self.audit_tables:
                self.audit_tables[legacy] = self.audit_tables[current]

    @staticmethod
    def _route_tracking(frame: pd.DataFrame) -> pd.DataFrame:
        """Display saved legacy audits using operational route labels.

        Only the status column is translated; values, ordering and the input
        audit remain unchanged. Unclassified and invalid statuses are kept.
        """
        result = frame.copy()
        if "Stage1_Status" in result:
            result["Stage1_Status"] = result["Stage1_Status"].replace({
                "MAR": "R-route",
                "MNAR": "S-route",
                "MNAR (Group)": "S-route (Group)",
                "MNAR (QC)": "S-route (QC)",
                "MNAR (Group & QC)": "S-route (Group & QC)",
            })
        return result
