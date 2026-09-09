"""Execute dataset construction through the common stage lifecycle.

Keep the calculation-only dataset build separate from optional CSV export and
acquisition dashboard rendering using its detached plotting payload.
"""

from __future__ import annotations

from ..core import MetaboDataset
from ..processing.audit import DatasetAuditPayload
from ..processing.stage import StageResult, StageRunner
from .builder import (
    MetaboDatasetBuilder,
    _export_dataset_table,
    _render_dataset_dashboard,
)


class DatasetBuildStageRunner(
    StageRunner[MetaboDatasetBuilder, MetaboDataset]
):
    """Compute the dataset audit before creating optional artifacts."""

    def compute(self) -> StageResult[MetaboDataset]:
        """Build the data product and audit in memory."""
        return self.processor.compute_build()

    def export(self, result: StageResult[MetaboDataset]) -> None:
        """Preserve the raw CSV filename, layout, and encoding."""
        assert self.output_dir is not None
        _export_dataset_table(result.data, self.output_dir)

    def render(self, result: StageResult[MetaboDataset]) -> None:
        """Render the acquisition overview from the saved plot payload."""
        assert self.output_dir is not None
        audit = result.require_audit(DatasetAuditPayload)
        if audit.plot_payload is None:
            raise ValueError("Dataset construction requires a plot payload.")
        _render_dataset_dashboard(audit.plot_payload, self.output_dir)
