"""Public plotting API for quality-assessment results.

The package composes heatmaps, PCA panels, outlier diagnostics, control charts,
and assessment dashboards into :class:`AssessmentPlotter`. Computation remains
owned by :mod:`pimqc.processing.assessment`.
"""

from .plotter import AssessmentPlotter
from .rendering import render_assessment
from .comparison import plot_assessment_comparison, render_assessment_comparison

__all__ = [
    "AssessmentPlotter",
    "render_assessment",
    "render_assessment_comparison",
    "plot_assessment_comparison",
]
