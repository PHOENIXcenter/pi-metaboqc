"""Core composition-based data models and processing primitives.

Expose independent intensity/metadata containers, their schema and context,
and the shared processor base used by scientific stages.
"""

from .dataset import (
    DatasetSchema,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)
from .processor import DatasetProcessor

__all__ = [
    "DatasetSchema",
    "MetaboDataset",
    "DatasetProcessor",
    "ProcessingContext",
    "SampleRoleLabels",
]
