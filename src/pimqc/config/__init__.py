"""Configuration package exports for validated pipeline settings.

The package exposes the Pydantic schema and shared stage-resolution helper used
to merge defaults, TOML sections, and explicit runtime overrides. Keeping these
exports together gives processing stages one stable configuration entry point.
"""

from .schema import (
    DataNormalizerConfig,
    DatasetConfig,
    FeatureFilterConfig,
    MissingValueImputerConfig,
    PipelineConfig,
    QualityAssessorConfig,
    SignalCorrectorConfig,
)
from .resolution import resolve_stage_config, validate_pipeline_params

__all__ = [
    "PipelineConfig",
    "DatasetConfig",
    "QualityAssessorConfig",
    "FeatureFilterConfig",
    "SignalCorrectorConfig",
    "DataNormalizerConfig",
    "MissingValueImputerConfig",
    "resolve_stage_config",
    "validate_pipeline_params",
]
