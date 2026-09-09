"""Public dataset-construction and input-validation API.

Expose the dataset builder and convenience construction function without
coupling input validation to downstream correction or visualization.
"""

from .builder import MetaboDatasetBuilder, build_dataset

__all__ = ["MetaboDatasetBuilder", "build_dataset"]
