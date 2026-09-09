"""Composition-based base class for scientific stage calculation engines.

Keep dataset context, local numerical work views, and cached diagnostics
separate while materializing explicit dataset values at stage boundaries.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import cached_property
from typing import Any, Mapping

import pandas as pd

from .dataset import MetaboDataset, ProcessingContext


class DatasetProcessor:
    """Hold one explicit dataset and a local ordinary-dataframe work view.

    The annotated frame is deliberately local to a calculation engine. Stage
    boundaries always receive and return :class:`MetaboDataset`, so pandas
    metadata propagation cannot determine scientific state.
    """

    def __init__(self, data: MetaboDataset) -> None:
        """Detach the input dataset and initialize local calculation state."""
        if not isinstance(data, MetaboDataset):
            raise TypeError("data must be a MetaboDataset instance.")
        self.dataset = data.copy()
        self.frame = self.dataset.annotated_frame()
        self.config = self.dataset.settings()
        self.stats: dict[str, Any] = {}

    def _invalidate_cached_properties(self) -> None:
        """Clear calculations cached on the processor instance."""
        for cls in type(self).__mro__:
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, cached_property):
                    self.__dict__.pop(name, None)

    def _replace_frame(self, frame: pd.DataFrame) -> None:
        """Replace the local work matrix and invalidate derived values."""
        self.frame = pd.DataFrame(frame).copy(deep=True)
        self.frame.attrs.update(copy.deepcopy(self.config))
        self._invalidate_cached_properties()

    def _context_from_config(
        self,
        *,
        updates: Mapping[str, Any] | None = None,
    ) -> ProcessingContext:
        """Build output context from explicit stage state only."""
        values = dict(updates or {})
        values.setdefault(
            "global_seed",
            self.config.get("global_seed", self.dataset.context.global_seed),
        )
        context = self.dataset.context
        allowed = {
            "acquisition_mode",
            "internal_standards",
            "outlier_reference_features",
            "global_seed",
            "pipeline_stage",
            "is_logged",
            "log_base",
            "is_scaled",
            "scale_method",
            "extra_attrs",
        }
        unknown = set(values).difference(allowed)
        if unknown:
            raise TypeError(
                f"Unknown processing context fields: {sorted(unknown)}"
            )
        return replace(context, **values)

    def _to_dataset(
        self,
        frame: pd.DataFrame,
        *,
        context_updates: Mapping[str, Any] | None = None,
        feature_metadata: pd.DataFrame | None = None,
    ) -> MetaboDataset:
        """Materialize a stage result from the local ordinary dataframe."""
        return self.dataset.with_intensity(
            frame,
            context=self._context_from_config(updates=context_updates),
            feature_metadata=feature_metadata,
        )

    @property
    def qc_data(self) -> pd.DataFrame:
        """Return the local annotated QC matrix."""
        sample_type = self.dataset.schema.sample_type
        label = self.dataset.schema.roles.qc
        mask = self.frame.columns.get_level_values(sample_type) == label
        return self.frame.loc[:, mask]

    @property
    def blank_data(self) -> pd.DataFrame:
        """Return the local annotated blank matrix."""
        sample_type = self.dataset.schema.sample_type
        label = self.dataset.schema.roles.blank
        mask = self.frame.columns.get_level_values(sample_type) == label
        return self.frame.loc[:, mask]

    @property
    def actual_data(self) -> pd.DataFrame:
        """Return the local annotated biological-sample matrix."""
        sample_type = self.dataset.schema.sample_type
        label = self.dataset.schema.roles.actual
        mask = self.frame.columns.get_level_values(sample_type) == label
        return self.frame.loc[:, mask]

    @property
    def valid_internal_standards(self) -> list[Any]:
        """Return configured internal standards present in the work matrix."""
        targets = {
            str(item).lower()
            for item in self.dataset.context.internal_standards
        }
        return [
            item for item in self.frame.index if str(item).lower() in targets
        ]

    @property
    def valid_outlier_reference_features(self) -> list[Any]:
        """Return configured outlier references present in the work matrix."""
        targets = set(self.dataset.context.outlier_reference_features)
        return [item for item in self.frame.index if item in targets]

    def intensity_order_info(self, feature_type: str = "IS") -> pd.DataFrame:
        """Return annotated reference-feature values in injection order."""
        if feature_type.lower() in {"internal_standard", "is"}:
            features = self.valid_internal_standards
        elif feature_type.lower() in {"outlier_ref_feat", "orf"}:
            features = self.valid_outlier_reference_features
        else:
            features = []

        schema = self.dataset.schema
        values = self.frame.loc[features].transpose()
        valid_labels = {schema.roles.actual, schema.roles.qc}
        mask = values.index.get_level_values(schema.sample_type).isin(
            valid_labels
        )
        values = values.loc[mask].reset_index(
            [schema.sample_type, schema.injection_order]
        )
        values[schema.injection_order] = values[schema.injection_order].astype(
            int
        )
        return values.sort_values([schema.sample_type, schema.injection_order])

    @staticmethod
    def calculate_boundaries(
        values: Any,
        boundary_type: str = "IQR",
    ) -> tuple[float, float, float]:
        """Delegate robust reference limits to the dataset domain model."""
        return MetaboDataset.calculate_boundaries(values, boundary_type)
