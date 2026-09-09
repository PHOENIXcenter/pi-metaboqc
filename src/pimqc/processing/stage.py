"""Define the common compute, export, and render stage lifecycle.

``StageResult`` carries computed data and one typed audit. ``StageRunner``
executes calculations first and performs filesystem and payload-driven
visualization work only when an output directory is supplied.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar, cast

from ..io import ensure_directory
from ..config.resolution import validate_stage_values
from .audit import AuditPayload

DataT = TypeVar("DataT")
ProcessorT = TypeVar("ProcessorT")
AuditT = TypeVar("AuditT", bound=AuditPayload)


def validate_runtime_overrides(
    processor: Any,
    runtime_overrides: Mapping[str, object] | None,
    allowed_override_keys: frozenset[str] | set[str] | None = None,
) -> dict[str, object]:
    """Validate and normalize optional call-time settings for a processor.

    Validation happens before a runner creates an output directory, so a
    misspelled notebook option cannot leave behind partial artifacts.
    """
    overrides = {
        key: value
        for key, value in (runtime_overrides or {}).items()
        if value is not None
    }
    if not overrides:
        return {}

    if allowed_override_keys is not None:
        unknown = sorted(set(overrides).difference(allowed_override_keys))
        if unknown:
            supported = ", ".join(sorted(allowed_override_keys))
            unknown_text = ", ".join(unknown)
            raise TypeError(
                f"Unsupported runtime override(s) for "
                f"{type(processor).__name__}: {unknown_text}. "
                f"Supported options: {supported}."
            )
    engine = getattr(processor, "_engine", processor)
    section = type(engine).__name__
    merged = validate_stage_values(
        section, {**getattr(processor, "config", {}), **overrides}
    )
    return {key: merged[key] for key in overrides}


def apply_runtime_overrides(
    processor: Any,
    runtime_overrides: Mapping[str, object] | None,
    allowed_override_keys: frozenset[str] | set[str] | None = None,
) -> dict[str, object]:
    """Apply validated execution-time settings to a stage processor.

    Execution-time keyword arguments have the highest configuration priority:
    module defaults are resolved first, then pipeline settings and optional
    constructor values are applied. This function applies the final notebook
    or API overrides immediately before a stage performs its computation.

    Cached calculations are cleared when settings change, preventing metrics
    or plots from reusing values computed under an earlier configuration.
    """
    overrides = validate_runtime_overrides(
        processor,
        runtime_overrides,
        allowed_override_keys,
    )
    if not overrides:
        return {}

    config = getattr(processor, "config", None)
    if config is None:
        raise TypeError(
            "Stage processors must expose a 'config' mapping for runtime "
            "configuration."
        )
    config.update(overrides)

    invalidate = getattr(processor, "_invalidate_cached_properties", None)
    if callable(invalidate):
        invalidate()
    else:
        for cls in type(processor).__mro__:
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, cached_property):
                    processor.__dict__.pop(name, None)
    return overrides


@dataclass
class StageResult(Generic[DataT]):
    """Carry one data result and one typed audit between stage phases.

    Plotting inputs belong to the typed audit payload and never reference the
    live processor.
    """

    data: DataT
    audit: AuditPayload

    def __post_init__(self) -> None:
        """Reject values that do not implement the typed audit contract."""
        if not isinstance(self.audit, AuditPayload):
            raise TypeError("audit must be an AuditPayload instance.")

    def require_audit(self, audit_type: type[AuditT]) -> AuditT:
        """Return the audit narrowed to an expected stage-specific type."""
        if not isinstance(self.audit, audit_type):
            raise TypeError(
                f"Expected {audit_type.__name__}, received "
                f"{type(self.audit).__name__}."
            )
        return cast(AuditT, self.audit)


class StageRunner(ABC, Generic[ProcessorT, DataT]):
    """Separate computation, artifact export, and visualization phases.

    ``compute`` is intentionally broader than ``transform``: processing
    stages may replace an intensity matrix, while observational stages such
    as quality assessment derive diagnostics without modifying their input.
    """

    def __init__(
        self,
        processor: ProcessorT,
        output_dir: str | Path | None,
        runtime_overrides: Mapping[str, object] | None = None,
        allowed_override_keys: frozenset[str] | set[str] | None = None,
    ) -> None:
        """Initialize a stage lifecycle.

        Args:
            processor: Processing object that performs the computation.
            output_dir: Optional artifact directory. When omitted, the runner
                performs only the computation phase.
            runtime_overrides: Named values supplied at execution time.
            allowed_override_keys: Configuration names accepted by this stage.
        """
        self.processor = processor
        self.runtime_overrides = validate_runtime_overrides(
            processor,
            runtime_overrides,
            allowed_override_keys,
        )
        self.allowed_override_keys = allowed_override_keys
        # Defer directory creation until computation succeeds. Besides keeping
        # output-free execution genuinely side-effect free, this avoids empty
        # artifact directories when a stage fails during calculation.
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def run(self) -> StageResult[DataT]:
        """Execute the stage lifecycle in a fixed, auditable order."""
        apply_runtime_overrides(
            self.processor,
            self.runtime_overrides,
            allowed_override_keys=None,
        )
        # Complete all calculations before permitting filesystem or plotting
        # side effects, so callers can run a computation-only lifecycle.
        result = self.compute()
        if self.output_dir is not None:
            self.output_dir = ensure_directory(self.output_dir)
            # Export before rendering because dashboards may refer to the
            # finalized filenames or audit tables.
            self.export(result)
            self.render(result)
        return result

    @abstractmethod
    def compute(self) -> StageResult[DataT]:
        """Perform calculations without writing files or rendering figures."""

    @abstractmethod
    def export(self, result: StageResult[DataT]) -> None:
        """Write stage data and tabular audit artifacts."""

    @abstractmethod
    def render(self, result: StageResult[DataT]) -> None:
        """Render visual artifacts from the completed stage result."""
