"""Explicit, composition-based metabolomics dataset domain model.

``MetaboDataset`` is the only in-memory data boundary used by pi-metaboqc.
The intensity matrix, sample metadata, feature metadata, schema, and
processing context remain separate values; no pandas subclass or hidden
``DataFrame.attrs`` state is required to move data between stages.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Hashable, Mapping, Optional

import pandas as pd

from ..constants import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class SampleRoleLabels:
    """Values used to identify the three supported analytical sample roles."""

    actual: str = "Sample"
    blank: str = "Blank"
    qc: str = "QC"
    additional: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = {"Actual sample", "Blank sample", "QC sample"}
        overlap = reserved.intersection(self.additional)
        if overlap:
            raise ValueError(
                "Additional role mappings cannot redefine standard roles: "
                f"{sorted(overlap)}."
            )
        object.__setattr__(
            self,
            "additional",
            copy.deepcopy(dict(self.additional)),
        )

    @classmethod
    def from_mapping(cls, sample_dict: Mapping[str, str]) -> "SampleRoleLabels":
        """Build role labels from a configured sample-role mapping."""
        standard_keys = {"Actual sample", "Blank sample", "QC sample"}
        return cls(
            actual=sample_dict.get("Actual sample", "Sample"),
            blank=sample_dict.get("Blank sample", "Blank"),
            qc=sample_dict.get("QC sample", "QC"),
            additional={
                key: copy.deepcopy(value)
                for key, value in sample_dict.items()
                if key not in standard_keys
            },
        )

    def to_mapping(self) -> dict[str, str]:
        """Return the configured sample-role mapping."""
        mapping = copy.deepcopy(dict(self.additional))
        mapping.update(
            {
                "Actual sample": self.actual,
                "Blank sample": self.blank,
                "QC sample": self.qc,
            }
        )
        return mapping


@dataclass(frozen=True)
class DatasetSchema:
    """Declare identifier fields and structural sample-metadata semantics."""

    feature_id: str = "Metabolite"
    sample_id: str = "Sample Name"
    sample_type: str = "Sample Type"
    batch: str = "Batch"
    injection_order: str = "Inject Order"
    biological_group: Optional[str] = "Bio Group"
    roles: SampleRoleLabels = field(default_factory=SampleRoleLabels)
    sample_metadata_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required_names = {
            "feature_id": self.feature_id,
            "sample_id": self.sample_id,
            "sample_type": self.sample_type,
            "batch": self.batch,
            "injection_order": self.injection_order,
        }
        empty = [name for name, value in required_names.items() if not value]
        if empty:
            raise ValueError(f"Schema fields cannot be empty: {empty}.")

        order = self.sample_metadata_order
        if order:
            if len(order) != len(set(order)):
                raise ValueError("sample_metadata_order contains duplicates.")
            if self.sample_id not in order:
                raise ValueError(
                    "sample_metadata_order must include the sample ID field "
                    f"'{self.sample_id}'."
                )

    @property
    def required_sample_metadata(self) -> tuple[str, ...]:
        """Metadata columns required by current pi-metaboqc processors."""
        fields = [self.sample_type, self.batch, self.injection_order]
        if (
            self.biological_group
            and self.biological_group in self.sample_metadata_order
        ):
            fields.append(self.biological_group)
        return tuple(fields)


@dataclass(frozen=True)
class ProcessingContext:
    """Framework-independent acquisition and transformation context.

    ``extra_attrs`` is reserved for framework-neutral acquisition annotations
    that are not yet represented by first-class fields. Scientific results
    and stage decisions belong in typed audit payloads instead.
    """

    acquisition_mode: str = "ESI+"
    internal_standards: tuple[Hashable, ...] = ()
    outlier_reference_features: tuple[Hashable, ...] = ()
    global_seed: int = DEFAULT_RANDOM_SEED
    pipeline_stage: str = "Raw data"
    is_logged: bool = False
    log_base: str = "None"
    is_scaled: bool = False
    scale_method: str = "None"
    extra_attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "internal_standards", tuple(self.internal_standards)
        )
        object.__setattr__(
            self,
            "outlier_reference_features",
            tuple(self.outlier_reference_features),
        )
        object.__setattr__(
            self,
            "extra_attrs",
            copy.deepcopy(dict(self.extra_attrs)),
        )

    @staticmethod
    def _as_sequence(value: Any) -> tuple[Hashable, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    @classmethod
    def from_mapping(cls, attrs: Mapping[str, Any]) -> "ProcessingContext":
        """Extract explicit context fields from a settings mapping."""
        known = {
            "mode",
            "sample_name",
            "sample_type",
            "bio_group",
            "batch",
            "inject_order",
            "sample_dict",
            "internal_standard",
            "outlier_ref_feat",
            "global_seed",
            "pipeline_stage",
            "is_logged",
            "log_base",
            "is_scaled",
            "scale_method",
        }
        return cls(
            acquisition_mode=attrs.get("mode", "ESI+"),
            internal_standards=cls._as_sequence(
                attrs.get("internal_standard", ())
            ),
            outlier_reference_features=cls._as_sequence(
                attrs.get("outlier_ref_feat", ())
            ),
            global_seed=attrs.get("global_seed", DEFAULT_RANDOM_SEED),
            pipeline_stage=attrs.get("pipeline_stage", "Raw data"),
            is_logged=bool(attrs.get("is_logged", False)),
            log_base=attrs.get("log_base", "None"),
            is_scaled=bool(attrs.get("is_scaled", False)),
            scale_method=attrs.get("scale_method", "None"),
            extra_attrs={
                key: copy.deepcopy(value)
                for key, value in attrs.items()
                if key not in known
            },
        )

    def to_settings(self, schema: DatasetSchema) -> dict[str, Any]:
        """Return a flat settings view for calculations and plotters."""
        attrs = copy.deepcopy(dict(self.extra_attrs))
        attrs.update(
            {
                "mode": self.acquisition_mode,
                "sample_name": schema.sample_id,
                "sample_type": schema.sample_type,
                "bio_group": schema.biological_group,
                "batch": schema.batch,
                "inject_order": schema.injection_order,
                "sample_dict": schema.roles.to_mapping(),
                "internal_standard": list(self.internal_standards),
                "outlier_ref_feat": list(self.outlier_reference_features),
                "global_seed": self.global_seed,
                "pipeline_stage": self.pipeline_stage,
                "is_logged": self.is_logged,
                "log_base": self.log_base,
                "is_scaled": self.is_scaled,
                "scale_method": self.scale_method,
            }
        )
        return attrs


@dataclass
class MetaboDataset:
    """Composition-based metabolomics data object with aligned table axes.

    The canonical orientation is features by samples. Sample identifiers are
    stored in ``intensity.columns`` and ``sample_metadata.index``; feature
    identifiers are stored in ``intensity.index`` and
    ``feature_metadata.index``. All tables are defensively copied and metadata
    is reordered to match the intensity matrix.

    Stage calculations may create local caches, but those caches do not travel
    with the dataset. Cross-stage scientific state is represented explicitly
    by context fields, feature metadata, or typed audit payloads.
    """

    intensity: pd.DataFrame
    sample_metadata: pd.DataFrame
    feature_metadata: pd.DataFrame
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    context: ProcessingContext = field(default_factory=ProcessingContext)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be a DatasetSchema instance.")
        if not isinstance(self.context, ProcessingContext):
            raise TypeError("context must be a ProcessingContext instance.")

        self.intensity = self._copy_plain_dataframe(
            self.intensity, table_name="intensity"
        )
        self.sample_metadata = self._copy_plain_dataframe(
            self.sample_metadata, table_name="sample_metadata"
        )
        self.feature_metadata = self._copy_plain_dataframe(
            self.feature_metadata, table_name="feature_metadata"
        )
        self.context = copy.deepcopy(self.context)

        if isinstance(self.intensity.columns, pd.MultiIndex):
            raise ValueError(
                "MetaboDataset intensity columns must contain sample IDs, not "
                "a MultiIndex. Keep sample annotations in sample_metadata."
            )
        if isinstance(self.intensity.index, pd.MultiIndex):
            raise ValueError(
                "MetaboDataset feature IDs cannot be a MultiIndex."
            )

        self._validate_axis(self.intensity.index, "feature")
        self._validate_axis(self.intensity.columns, "sample")

        schema = self.schema
        if not schema.sample_metadata_order:
            schema = replace(
                schema,
                sample_metadata_order=(
                    schema.sample_id,
                    *tuple(self.sample_metadata.columns),
                ),
            )
            self.schema = schema

        self._validate_metadata_columns()
        self.sample_metadata = self._align_metadata(
            self.sample_metadata,
            self.intensity.columns,
            axis_name="sample",
            index_name=self.schema.sample_id,
        )
        self.feature_metadata = self._align_metadata(
            self.feature_metadata,
            self.intensity.index,
            axis_name="feature",
            index_name=self.schema.feature_id,
        )
        self.intensity.index = self.intensity.index.rename(
            self.schema.feature_id
        )
        self.intensity.columns = self.intensity.columns.rename(
            self.schema.sample_id
        )

    @staticmethod
    def _copy_plain_dataframe(
        table: pd.DataFrame, *, table_name: str
    ) -> pd.DataFrame:
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{table_name} must be a pandas DataFrame.")
        return pd.DataFrame(table).copy(deep=True)

    @staticmethod
    def _validate_axis(axis: pd.Index, axis_name: str) -> None:
        if not axis.is_unique:
            duplicates = axis[axis.duplicated()].unique().tolist()
            raise ValueError(
                f"Duplicate {axis_name} identifiers are not allowed: "
                f"{duplicates}."
            )
        if axis.hasnans:
            raise ValueError(
                f"Missing {axis_name} identifiers are not allowed."
            )

    def _validate_metadata_columns(self) -> None:
        missing = [
            column
            for column in self.schema.required_sample_metadata
            if column not in self.sample_metadata.columns
        ]
        if missing:
            raise ValueError(
                f"Sample metadata is missing schema fields: {missing}."
            )

        expected_order = set(self.schema.sample_metadata_order)
        available = {self.schema.sample_id, *self.sample_metadata.columns}
        missing_order = expected_order - available
        unlisted = available - expected_order
        if missing_order:
            raise ValueError(
                "sample_metadata_order references unavailable fields: "
                f"{sorted(missing_order)}."
            )
        if unlisted:
            raise ValueError(
                "sample_metadata_order omits metadata fields: "
                f"{sorted(unlisted)}."
            )

    @classmethod
    def _align_metadata(
        cls,
        metadata: pd.DataFrame,
        expected_ids: pd.Index,
        *,
        axis_name: str,
        index_name: str,
    ) -> pd.DataFrame:
        if isinstance(metadata.index, pd.MultiIndex):
            raise ValueError(
                f"{axis_name.title()} metadata index must be flat."
            )
        cls._validate_axis(metadata.index, f"{axis_name} metadata")

        expected = set(expected_ids)
        observed = set(metadata.index)
        if expected != observed:
            missing = [item for item in expected_ids if item not in observed]
            extra = [item for item in metadata.index if item not in expected]
            raise ValueError(
                f"{axis_name.title()} metadata identifiers do not match the "
                f"intensity matrix; missing={missing}, extra={extra}."
            )

        aligned = metadata.loc[list(expected_ids)].copy(deep=True)
        aligned.index = aligned.index.rename(index_name)
        return aligned

    @classmethod
    def from_tables(
        cls,
        intensity: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        feature_metadata: Optional[pd.DataFrame] = None,
        *,
        schema: Optional[DatasetSchema] = None,
        context: Optional[ProcessingContext] = None,
        sample_id_column: Optional[str] = None,
        feature_id_column: Optional[str] = None,
    ) -> "MetaboDataset":
        """Construct a dataset from ordinary, framework-neutral tables.

        Identifier columns can be supplied explicitly. Otherwise metadata
        indices are treated as identifiers. Intensity orientation must already
        be features by samples.
        """
        resolved_schema = schema or DatasetSchema()
        samples = pd.DataFrame(sample_metadata).copy(deep=True)
        sample_id = sample_id_column
        if sample_id is not None:
            if sample_id not in samples.columns:
                raise ValueError(
                    f"Sample ID column '{sample_id}' is not present in "
                    "metadata."
                )
            if sample_id != resolved_schema.sample_id:
                raise ValueError(
                    "sample_id_column must match DatasetSchema.sample_id "
                    f"('{resolved_schema.sample_id}')."
                )
            samples = samples.set_index(sample_id, drop=True)

        features: pd.DataFrame
        if feature_metadata is None:
            features = pd.DataFrame(index=pd.Index(intensity.index))
        else:
            features = pd.DataFrame(feature_metadata).copy(deep=True)
            feature_id = feature_id_column
            if feature_id is not None:
                if feature_id not in features.columns:
                    raise ValueError(
                        f"Feature ID column '{feature_id}' is not present in "
                        "metadata."
                    )
                if feature_id != resolved_schema.feature_id:
                    raise ValueError(
                        "feature_id_column must match DatasetSchema.feature_id "
                        f"('{resolved_schema.feature_id}')."
                    )
                features = features.set_index(feature_id, drop=True)

        return cls(
            intensity=intensity,
            sample_metadata=samples,
            feature_metadata=features,
            schema=resolved_schema,
            context=context or ProcessingContext(),
        )

    def copy(self) -> "MetaboDataset":
        """Return a fully detached dataset copy."""
        return type(self)(
            intensity=self.intensity,
            sample_metadata=self.sample_metadata,
            feature_metadata=self.feature_metadata,
            schema=copy.deepcopy(self.schema),
            context=copy.deepcopy(self.context),
        )

    def settings(self) -> dict[str, Any]:
        """Return calculation settings derived from schema and context."""
        values = self.context.to_settings(self.schema)
        values["is_multi_batch"] = self.is_multi_batch
        values["batch_list"] = self.ordered_batches
        return values

    def annotated_frame(self) -> pd.DataFrame:
        """Return an ordinary dataframe with metadata MultiIndex columns.

        The annotated frame is a calculation and plotting view only. The
        canonical intensity matrix continues to use flat sample identifiers.
        """
        metadata = self.sample_metadata.reset_index()
        columns = pd.MultiIndex.from_frame(
            metadata.loc[:, list(self.schema.sample_metadata_order)]
        )
        frame = pd.DataFrame(
            self.intensity.to_numpy(copy=True),
            index=self.intensity.index.copy(),
            columns=columns,
        )
        frame.attrs.update(self.settings())
        return frame

    def with_intensity(
        self,
        intensity: pd.DataFrame,
        *,
        context: Optional[ProcessingContext] = None,
        context_updates: Optional[Mapping[str, Any]] = None,
        feature_metadata: Optional[pd.DataFrame] = None,
    ) -> "MetaboDataset":
        """Return a dataset aligned to a replacement intensity matrix."""
        frame = pd.DataFrame(intensity).copy(deep=True)
        if isinstance(frame.columns, pd.MultiIndex):
            sample_ids = frame.columns.get_level_values(self.schema.sample_id)
            frame.columns = pd.Index(sample_ids, name=self.schema.sample_id)
        else:
            frame.columns = frame.columns.rename(self.schema.sample_id)
        frame.index = frame.index.rename(self.schema.feature_id)

        sample_metadata = self.sample_metadata.loc[list(frame.columns)]
        if feature_metadata is None:
            feature_metadata = self.feature_metadata.loc[list(frame.index)]
        else:
            feature_metadata = pd.DataFrame(feature_metadata).loc[
                list(frame.index)
            ]

        resolved_context = copy.deepcopy(context or self.context)
        if context_updates:
            resolved_context = replace(
                resolved_context, **dict(context_updates)
            )
        return type(self)(
            intensity=frame,
            sample_metadata=sample_metadata,
            feature_metadata=feature_metadata,
            schema=copy.deepcopy(self.schema),
            context=resolved_context,
        )

    @property
    def is_multi_batch(self) -> bool:
        """Return whether multiple analytical batches are represented."""
        return self.sample_metadata[self.schema.batch].nunique(dropna=True) > 1

    @property
    def ordered_batches(self) -> list[Any]:
        """Return batches ordered by their first injection position."""
        metadata = self.sample_metadata
        starts = metadata.groupby(self.schema.batch)[
            self.schema.injection_order
        ].min()
        return starts.sort_values().index.tolist()

    def role_sample_ids(self, role: str) -> pd.Index:
        """Return sample IDs assigned to one standard analytical role."""
        labels = self.schema.roles
        try:
            label = {
                "actual": labels.actual,
                "blank": labels.blank,
                "qc": labels.qc,
            }[role.lower()]
        except KeyError as exc:
            raise ValueError(
                "role must be 'actual', 'blank', or 'qc'."
            ) from exc
        mask = self.sample_metadata[self.schema.sample_type] == label
        return self.sample_metadata.index[mask]

    @property
    def qc_data(self) -> pd.DataFrame:
        """Return the QC intensity matrix with flat sample IDs."""
        return self.intensity.loc[:, self.role_sample_ids("qc")]

    @property
    def blank_data(self) -> pd.DataFrame:
        """Return the blank intensity matrix with flat sample IDs."""
        return self.intensity.loc[:, self.role_sample_ids("blank")]

    @property
    def actual_data(self) -> pd.DataFrame:
        """Return the biological-sample matrix with flat sample IDs."""
        return self.intensity.loc[:, self.role_sample_ids("actual")]

    @property
    def valid_internal_standards(self) -> list[Hashable]:
        """Return configured internal standards found in the feature axis."""
        targets = {
            str(item).lower() for item in self.context.internal_standards
        }
        return [
            item for item in self.feature_ids if str(item).lower() in targets
        ]

    @property
    def valid_outlier_reference_features(self) -> list[Hashable]:
        """Return configured outlier references found in the feature axis."""
        targets = set(self.context.outlier_reference_features)
        return [item for item in self.feature_ids if item in targets]

    def intensity_order_info(self, feature_type: str = "IS") -> pd.DataFrame:
        """Return reference-feature intensities indexed by sample metadata."""
        if feature_type.lower() in {"internal_standard", "is"}:
            features = self.valid_internal_standards
        elif feature_type.lower() in {"outlier_ref_feat", "orf"}:
            features = self.valid_outlier_reference_features
        else:
            features = []

        frame = self.annotated_frame().loc[features].transpose()
        valid_labels = {self.schema.roles.actual, self.schema.roles.qc}
        mask = frame.index.get_level_values(self.schema.sample_type).isin(
            valid_labels
        )
        frame = frame.loc[mask].reset_index(
            [self.schema.sample_type, self.schema.injection_order]
        )
        frame[self.schema.injection_order] = frame[
            self.schema.injection_order
        ].astype(int)
        return frame.sort_values(
            [self.schema.sample_type, self.schema.injection_order]
        )

    @staticmethod
    def calculate_boundaries(
        values: Any, boundary_type: str = "IQR"
    ) -> tuple[float, float, float]:
        """Calculate a center and robust lower/upper reference limits."""
        import numpy as np

        array = np.asarray(values, dtype=float)
        if boundary_type in {"mean-std", "sigma"}:
            center = float(np.nanmean(array))
            spread = float(np.nanstd(array, ddof=1))
            return center, center - 3 * spread, center + 3 * spread
        if boundary_type == "IQR":
            center = float(np.nanmedian(array))
            q1 = float(np.nanquantile(array, 0.25))
            q3 = float(np.nanquantile(array, 0.75))
            return center, q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return 0.0, 0.0, 0.0

    @property
    def dataset_metrics(self) -> dict[str, Any]:
        """Return structural metrics used by reports and dashboards."""
        try:
            from .. import __version__ as package_version
        except ImportError:
            package_version = "0+unknown"

        metadata = self.sample_metadata
        distribution: dict[str, Any] = {}
        for batch in self.ordered_batches:
            batch_metadata = metadata.loc[metadata[self.schema.batch] == batch]
            type_counts = batch_metadata[self.schema.sample_type].value_counts()
            orders = batch_metadata[self.schema.injection_order].astype(int)
            distribution[str(batch)] = {
                "Total": int(len(batch_metadata)),
                "QC": int(type_counts.get(self.schema.roles.qc, 0)),
                "Blank": int(type_counts.get(self.schema.roles.blank, 0)),
                "Sample": int(type_counts.get(self.schema.roles.actual, 0)),
                "Inject Order": f"{orders.min()} ~ {orders.max()}",
            }
        return {
            "mode": self.context.acquisition_mode,
            "pi-metaboqc_version": package_version,
            "features": {
                "total": self.intensity.shape[0],
                "internal_standards": self.valid_internal_standards,
                "internal_standards_count": len(self.valid_internal_standards),
            },
            "samples": {
                "total": self.intensity.shape[1],
                "qc": len(self.role_sample_ids("qc")),
                "blank": len(self.role_sample_ids("blank")),
                "actual": len(self.role_sample_ids("actual")),
            },
            "batches": {
                "batch_count": len(self.ordered_batches),
                "ordered_batches": self.ordered_batches,
                "batch_distribution": distribution,
            },
        }

    def to_tables(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return defensive copies of intensity, sample, and feature tables."""
        return (
            self.intensity.copy(deep=True),
            self.sample_metadata.copy(deep=True),
            self.feature_metadata.copy(deep=True),
        )

    @property
    def sample_ids(self) -> pd.Index:
        """Return sample identifiers in canonical matrix order."""
        return self.intensity.columns.copy()

    @property
    def feature_ids(self) -> pd.Index:
        """Return feature identifiers in canonical matrix order."""
        return self.intensity.index.copy()
