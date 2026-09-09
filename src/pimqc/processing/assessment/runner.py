"""Export and render completed quality-assessment diagnostics.

The runner keeps assessment calculations side-effect free and preserves the
existing QA table, panel, legend, control-chart, and dashboard artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping


from ..audit import AssessQualityAuditPayload
from ..stage import StageResult, StageRunner
from .analysis import AssessmentDiagnostics
from ...plotting.assessment.rendering import render_assessment

if TYPE_CHECKING:
    from .analysis import QualityAssessor


class AssessmentStageRunner(
    StageRunner["QualityAssessor", AssessmentDiagnostics]
):
    """Run assessment while separating computation from artifact generation."""

    def __init__(
        self,
        processor: "QualityAssessor",
        output_dir: str | Path | None,
        *,
        legend_mode: str = "external",
        runtime_overrides: Mapping[str, object] | None = None,
        allowed_override_keys: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Initialize the assessment lifecycle and legend strategy."""
        super().__init__(
            processor,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=allowed_override_keys,
        )
        self.legend_mode = legend_mode

    def compute(self) -> StageResult[AssessmentDiagnostics]:
        """
        Compute all tables and plot inputs without filesystem side effects.
        """
        return self.processor.compute_assessment()

    def export(self, result: StageResult[AssessmentDiagnostics]) -> None:
        """Write the combined sample-level QA diagnostic table."""
        audit = result.require_audit(AssessQualityAuditPayload)
        if audit.skipped:
            return
        assert self.output_dir is not None
        audit.outliers.to_csv(
            self.output_dir / "QA_Diagnostics_Outliers.csv",
            encoding="utf-8-sig",
            na_rep="NA",
        )

    def render(self, result: StageResult[AssessmentDiagnostics]) -> None:
        """Render through the same public audit renderer used by comparisons."""
        assert self.output_dir is not None
        render_assessment(
            result.require_audit(AssessQualityAuditPayload),
            self.output_dir,
            legend_mode=self.legend_mode,
            show_plot=True,
        )
