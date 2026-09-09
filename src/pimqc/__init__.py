"""Expose the public pi-metaboqc API and explicit runtime initialization.

The package root re-exports the core data model, processing stages, dataset
builder, and pipeline entry point. Importing it is side-effect free; optional
logging, progress, and hardware diagnostics are enabled only through ``init``.
"""

import multiprocessing

from loguru import logger

from ._version import __version__
from .constants import DEFAULT_RANDOM_SEED

# Core Data Structures
from .core import (
    DatasetSchema,
    DatasetProcessor,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)

# Data Ingestion & Pipeline Management
from .dataset.builder import MetaboDatasetBuilder, build_dataset
from .pipeline import PipelineResult, run_pipeline
from .processing.audit import (
    AssessQualityAuditPayload,
    AuditPayload,
    CorrectionAuditPayload,
    DatasetAuditPayload,
    ImputationAuditPayload,
    MissingValueFilterAuditPayload,
    NormalizationAuditPayload,
    QualityFilterAuditPayload,
)

# Processing Modules (Actors)
from .processing.assessment import QualityAssessor
from .processing.correction import SignalCorrector
from .processing.filtering import (
    FeatureFilter,
    FeatureMissingValueFilter,
    FeatureQualityFilter,
    FilteringOrchestrator,
    FilteringRunResult,
    SampleMissingValueFilter,
)
from .processing.imputation import MissingValueImputer
from .processing.normalization import DataNormalizer
from .processing.stage import StageResult
from .runtime import (
    configure_joblib_cpu_limit,
    configure_logging,
    print_hardware_diagnostics,
    set_progress_enabled,
)

# Define public API
__all__ = [
    "MetaboDataset",
    "DatasetProcessor",
    "DatasetSchema",
    "ProcessingContext",
    "SampleRoleLabels",
    "AuditPayload",
    "DatasetAuditPayload",
    "AssessQualityAuditPayload",
    "MissingValueFilterAuditPayload",
    "CorrectionAuditPayload",
    "QualityFilterAuditPayload",
    "ImputationAuditPayload",
    "NormalizationAuditPayload",
    "StageResult",
    "QualityAssessor",
    "SignalCorrector",
    "MissingValueImputer",
    "DataNormalizer",
    "FeatureFilter",
    "SampleMissingValueFilter",
    "FeatureMissingValueFilter",
    "FeatureQualityFilter",
    "FilteringOrchestrator",
    "FilteringRunResult",
    "MetaboDatasetBuilder",
    "build_dataset",
    "PipelineResult",
    "run_pipeline",
    "DEFAULT_RANDOM_SEED",
    "__version__",
]

_IS_INITIALIZED = False


def init(
    check_hardware: bool = True,
    log_level: str = "DEBUG",
    show_progress: bool = True,
    preserve_existing_sinks: bool = False,
) -> None:
    """Explicitly initialize the pi-metaboqc runtime environment.

    Usage:
        pimqc.init(
            check_hardware=False,
            log_level="DEBUG",
            show_progress=False,
            preserve_existing_sinks=False,
        )

    Args:
        check_hardware: Emit hardware diagnostics in the main process.
        log_level: Minimum severity emitted by the package console sink.
        show_progress: Enable progress indicators for long calculations.
        preserve_existing_sinks: Retain Loguru sinks set by a host framework.
            The default removes the default sink so notebooks and scripts emit
            each package log record only once.
    """
    global _IS_INITIALIZED

    # Guard: Prevent redundant initializations within the same process.
    if _IS_INITIALIZED:
        logger.debug("pi-metaboqc already initialized. Skipping init().")
        return

    configure_logging(
        level=log_level,
        preserve_existing_sinks=preserve_existing_sinks,
    )
    configure_joblib_cpu_limit()
    set_progress_enabled(show_progress)

    # Guard: Hardware diagnostics can be time-consuming due to system
    # register probes. Execute only in MainProcess if permitted by user.
    if multiprocessing.current_process().name == "MainProcess":
        if check_hardware:
            print_hardware_diagnostics()

    # Mark the global initialization state as complete.
    _IS_INITIALIZED = True
