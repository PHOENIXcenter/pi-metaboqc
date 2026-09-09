"""Verify the stable, non-pickle pi-metaboqc directory artifact format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pimqc.core import (
    DatasetSchema,
    MetaboDataset,
    ProcessingContext,
    SampleRoleLabels,
)
from pimqc.plotting import (
    AssessmentPlotPayload,
    CorrectionPlotPayload,
    DatasetPlotPayload,
    FilteringPlotPayload,
    ImputationPlotPayload,
    NormalizationPlotPayload,
)
from pimqc.processing import (
    AssessQualityAuditPayload,
    CorrectionAuditPayload,
    DatasetAuditPayload,
    ImputationAuditPayload,
    MissingValueFilterAuditPayload,
    NormalizationAuditPayload,
    QualityFilterAuditPayload,
    SampleFilterAuditPayload,
)
from pimqc.serialization import (
    ArtifactExistsError,
    ArtifactValidationError,
    UnknownWireTypeError,
    UnsupportedFormatVersionError,
    UnsupportedSerializationTypeError,
    read_audit_payload,
    read_metabo_dataset,
    read_plot_payload,
    validate_artifact,
    write_audit_payload,
    write_metabo_dataset,
    write_plot_payload,
)


def _sample_dataset() -> MetaboDataset:
    intensity = pd.DataFrame(
        [[1.0, 2.0, np.nan], [3.0, np.inf, 5.0]],
        index=pd.Index(["F1", "F2"], name="Metabolite"),
        columns=pd.Index(["S1", "S2", "Q1"], name="Sample Name"),
    )
    sample_metadata = pd.DataFrame(
        {
            "Sample Type": pd.Categorical(
                ["Sample", "Sample", "QC"],
                categories=["Sample", "QC", "Blank"],
            ),
            "Batch": pd.array([1, 1, 1], dtype="Int64"),
            "Inject Order": [1, 2, 3],
            "Bio Group": pd.array(["A", "B", pd.NA], dtype="string"),
            "Acquired": pd.to_datetime(
                ["2025-01-01", "2025-01-02", None], utc=True
            ),
        },
        index=pd.Index(["S1", "S2", "Q1"], name="Sample Name"),
    )
    feature_metadata = pd.DataFrame(
        {
            "Compound": ["Alpha", "Beta"],
            "Confirmed": pd.array([True, pd.NA], dtype="boolean"),
        },
        index=pd.Index(["F1", "F2"], name="Metabolite"),
    )
    schema = DatasetSchema(
        roles=SampleRoleLabels(
            actual="Sample",
            blank="Blank",
            qc="QC",
            additional={"Reference sample": "Reference"},
        ),
        sample_metadata_order=(
            "Sample Name",
            "Sample Type",
            "Batch",
            "Inject Order",
            "Bio Group",
            "Acquired",
        ),
    )
    context = ProcessingContext(
        acquisition_mode="ESI-",
        internal_standards=("F1",),
        pipeline_stage="Filtered data",
        extra_attrs={
            "threshold": np.float64(0.2),
            "complex_weight": np.complex64(1 + 2j),
            "timestamp": np.datetime64("2025-03-04T05:06:07", "ns"),
            "token": b"metaboqc",
            "source_path": Path("inputs/source.csv"),
        },
    )
    return MetaboDataset(
        intensity=intensity,
        sample_metadata=sample_metadata,
        feature_metadata=feature_metadata,
        schema=schema,
        context=context,
    )


def _assert_dataset_equal(
    left: MetaboDataset,
    right: MetaboDataset,
) -> None:
    assert type(right) is MetaboDataset
    assert_frame_equal(left.intensity, right.intensity)
    assert_frame_equal(left.sample_metadata, right.sample_metadata)
    assert_frame_equal(left.feature_metadata, right.feature_metadata)
    assert left.schema == right.schema
    assert left.context.acquisition_mode == right.context.acquisition_mode
    assert left.context.internal_standards == right.context.internal_standards
    assert left.context.pipeline_stage == right.context.pipeline_stage
    assert left.context.extra_attrs.keys() == right.context.extra_attrs.keys()


def _rehash_payload(artifact: Path) -> None:
    payload = artifact / "payload.json"
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["files"] if item["path"] == "payload.json"
    )
    content = payload.read_bytes()
    entry["size"] = len(content)
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_metabo_dataset_round_trip_preserves_tables_and_context(
    tmp_path: Path,
) -> None:
    """Preserve all explicit tables, extension dtypes, and context."""
    source = _sample_dataset()
    target = write_metabo_dataset(source, tmp_path / "dataset.pimqc")

    manifest = validate_artifact(target)
    restored = read_metabo_dataset(target)

    assert manifest.artifact_kind == "dataset"
    assert manifest.artifact_type == "metabo_dataset"
    assert manifest.schema_version == "1.0"
    assert_frame_equal(source.intensity, restored.intensity)
    assert_frame_equal(source.sample_metadata, restored.sample_metadata)
    assert_frame_equal(source.feature_metadata, restored.feature_metadata)
    assert source.schema == restored.schema
    assert source.context.acquisition_mode == restored.context.acquisition_mode
    assert (
        source.context.internal_standards == restored.context.internal_standards
    )
    assert source.context.extra_attrs["threshold"] == pytest.approx(
        restored.context.extra_attrs["threshold"]
    )
    assert (
        source.context.extra_attrs["complex_weight"]
        == restored.context.extra_attrs["complex_weight"]
    )
    assert (
        source.context.extra_attrs["timestamp"]
        == restored.context.extra_attrs["timestamp"]
    )
    assert restored.context.extra_attrs["token"] == b"metaboqc"


def test_all_plot_payload_types_round_trip(tmp_path: Path) -> None:
    """Round-trip every Phase 3 processor-free plot contract."""
    raw = _sample_dataset()
    normalized = _sample_dataset()
    payloads = [
        DatasetPlotPayload(raw),
        AssessmentPlotPayload(raw, is_multi_batch=False),
        FilteringPlotPayload(
            raw,
            audit_tables={"missing": pd.Series([0.1, 0.2], name="rate")},
        ),
        CorrectionPlotPayload(
            source_data=raw,
            selected_stages={"batch_1": normalized},
            candidate_results={
                "LOWESS": {
                    "pred_df": pd.DataFrame([[1.0, 2.0]]),
                    "score": np.float64(0.9),
                }
            },
            selected_prediction=normalized,
            internal_standard_ids=("F1",),
            boundary_type="IQR",
        ),
        ImputationPlotPayload(raw, normalized, global_seed=7),
        NormalizationPlotPayload(
            raw_data=raw,
            normalized_data=normalized,
            selection={"method": "PQN"},
            score_component_weights={"rsd": 0.5},
            sample_scale_log_ratio_tolerance=0.1,
            sample_scale_relative_delta_tolerance=0.2,
            global_seed=7,
        ),
    ]
    for position, payload in enumerate(payloads):
        artifact = tmp_path / f"plot-{position}.pimqc"
        write_plot_payload(payload, artifact)
        restored = read_plot_payload(artifact)
        assert type(restored) is type(payload)
        assert restored.contract_identity() == payload.contract_identity()
        _assert_dataset_equal(
            payload.primary_dataset,
            restored.primary_dataset,
        )
        assert type(restored.primary_data) is pd.DataFrame


def test_all_audit_payload_types_round_trip(tmp_path: Path) -> None:
    """Round-trip every typed Phase 2 audit including nested plot contracts."""
    raw = _sample_dataset()
    dataset_plot = DatasetPlotPayload(raw)
    assessment_plot = AssessmentPlotPayload(raw, False)
    filtering_plot = FilteringPlotPayload(raw)
    correction_plot = CorrectionPlotPayload(
        raw,
        {"corrected": raw},
        {"LOESS": {"stage_dfs": {"final": raw.annotated_frame()}}},
        raw,
        ("F1",),
        "IQR",
    )
    imputation_plot = ImputationPlotPayload(raw, raw, 0)
    normalization_plot = NormalizationPlotPayload(
        raw,
        raw,
        {
            "selected_method": "PQN",
            "candidate_results": [{"method": "PQN", "score": 0.8}],
        },
        {"rsd": 1.0},
        0.1,
        0.1,
        0,
    )
    audits = [
        DatasetAuditPayload({"features": 2}, dataset_plot),
        AssessQualityAuditPayload(
            metric_values={"median_rsd": np.float64(0.2)},
            outliers=pd.DataFrame({"sample": ["S2"]}),
            qc_correlation_mask=np.array([True, False, True]),
            plot_payload=assessment_plot,
        ),
        SampleFilterAuditPayload(
            pd.DataFrame({"retained": [True, False]}),
            dropped_sample_ids=("S2",),
            plot_payload=filtering_plot,
        ),
        MissingValueFilterAuditPayload(
            metric_values={"retained": 2},
            feature_tracking=pd.DataFrame({"keep": [True, False]}),
            sample_filtered_data=raw.copy(),
            sample_tracking=pd.DataFrame({"keep": [True, True, True]}),
            qc_mask=np.array([False, False, True]),
            valid_biological_groups=("A", "B"),
            plot_payload=filtering_plot,
            tables={"ids": pd.Index(["F1"])},
        ),
        CorrectionAuditPayload(
            metric_values={"score": 0.9},
            requested_method="AUTO",
            selected_method="LOESS",
            selected_label="LOESS-default",
            is_auto=True,
            sample_type_column="Sample Type",
            batch_column="Batch",
            injection_order_column="Inject Order",
            qc_label="QC",
            actual_label="Sample",
            plot_payload=correction_plot,
        ),
        QualityFilterAuditPayload(
            {"retained": 1},
            pd.DataFrame({"keep": [True, False]}),
            filtering_plot,
            {"dropped": pd.Series(["F2"])},
        ),
        ImputationAuditPayload(
            metric_values={"rmse": 0.1},
            candidate_results={
                "KNN": (
                    {"rmse": 0.1},
                    np.array([1.0, 2.0]),
                    np.array([1.1, 1.9]),
                )
            },
            requested_method="AUTO",
            selected_method="KNN",
            selected_label="KNN",
            is_auto=True,
            mar_feature_count=2,
            has_candidate_cache=True,
            skipped=False,
            plot_payload=imputation_plot,
        ),
        NormalizationAuditPayload(
            metric_values={"score": 0.8},
            output_suffix="_normalized",
            plot_payload=normalization_plot,
        ),
    ]
    for position, audit in enumerate(audits):
        artifact = tmp_path / f"audit-{position}.pimqc"
        write_audit_payload(audit, artifact)
        restored = read_audit_payload(artifact)
        assert type(restored) is type(audit)
        assert restored.contract_identity() == audit.contract_identity()


def test_duplicate_dataframe_labels_and_object_array_round_trip(
    tmp_path: Path,
) -> None:
    """Preserve positional columns and recursive object-array values."""
    frame = pd.DataFrame([[1, "x"], [2, None]], columns=["same", "same"])
    objects = np.empty(2, dtype=object)
    objects[:] = [("F1", 1), ("F2", 2)]
    payload = FilteringPlotPayload(
        _sample_dataset(),
        audit_tables={
            "duplicates": frame,
            "objects": objects,
        },
    )
    write_plot_payload(payload, tmp_path / "duplicates.pimqc")

    restored = read_plot_payload(tmp_path / "duplicates.pimqc")

    assert_frame_equal(restored.audit_tables["duplicates"], frame)
    np.testing.assert_equal(
        restored.audit_tables["objects"],
        payload.audit_tables["objects"],
    )


def test_write_is_deterministic_and_never_uses_pickle(tmp_path: Path) -> None:
    """
    Produce byte-identical members for identical object graphs without PKL.
    """
    source = _sample_dataset()
    first = write_metabo_dataset(source, tmp_path / "first.pimqc")
    second = write_metabo_dataset(source, tmp_path / "second.pimqc")

    first_files = {
        item.relative_to(first).as_posix(): item.read_bytes()
        for item in first.rglob("*")
        if item.is_file()
    }
    second_files = {
        item.relative_to(second).as_posix(): item.read_bytes()
        for item in second.rglob("*")
        if item.is_file()
    }
    assert first_files == second_files
    assert not any(path.endswith((".pkl", ".pickle")) for path in first_files)


def test_existing_destination_is_not_overwritten(tmp_path: Path) -> None:
    """Reject repeated writes before changing an existing artifact."""
    target = write_metabo_dataset(_sample_dataset(), tmp_path / "dataset.pimqc")
    original = (target / "manifest.json").read_bytes()

    with pytest.raises(ArtifactExistsError):
        write_metabo_dataset(_sample_dataset(), target)

    assert (target / "manifest.json").read_bytes() == original


def test_checksum_tampering_and_missing_member_are_rejected(
    tmp_path: Path,
) -> None:
    """
    Detect modified payload bytes and absent binary members before decoding.
    """
    changed = write_metabo_dataset(
        _sample_dataset(), tmp_path / "changed.pimqc"
    )
    payload = changed / "payload.json"
    content = payload.read_bytes()
    payload.write_bytes(b"[" + content[1:])
    with pytest.raises(ArtifactValidationError, match="checksum mismatch"):
        validate_artifact(changed)

    resized = write_metabo_dataset(
        _sample_dataset(), tmp_path / "resized.pimqc"
    )
    resized_payload = resized / "payload.json"
    resized_payload.write_bytes(resized_payload.read_bytes() + b" ")
    with pytest.raises(ArtifactValidationError, match="size mismatch"):
        validate_artifact(resized)

    missing = write_metabo_dataset(
        _sample_dataset(), tmp_path / "missing.pimqc"
    )
    array = next((missing / "arrays").glob("*.npy"))
    array.unlink()
    with pytest.raises(ArtifactValidationError, match="missing"):
        validate_artifact(missing)


def test_unknown_version_wire_type_and_path_traversal_are_rejected(
    tmp_path: Path,
) -> None:
    """Reject incompatible manifests, unknown tags, and unsafe member paths."""
    versioned = write_metabo_dataset(
        _sample_dataset(), tmp_path / "version.pimqc"
    )
    manifest_path = versioned / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "99.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(UnsupportedFormatVersionError):
        validate_artifact(versioned)

    unknown = write_plot_payload(
        DatasetPlotPayload(_sample_dataset()),
        tmp_path / "unknown.pimqc",
    )
    payload_path = unknown / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["$type"] = "obsolete_data_model"
    payload_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash_payload(unknown)
    with pytest.raises(UnknownWireTypeError):
        read_plot_payload(unknown)

    unsafe = write_metabo_dataset(_sample_dataset(), tmp_path / "unsafe.pimqc")
    unsafe_manifest_path = unsafe / "manifest.json"
    unsafe_manifest = json.loads(
        unsafe_manifest_path.read_text(encoding="utf-8")
    )
    unsafe_manifest["files"][0]["path"] = "../payload.json"
    unsafe_manifest_path.write_text(
        json.dumps(unsafe_manifest), encoding="utf-8"
    )
    with pytest.raises(ArtifactValidationError, match="Unsafe"):
        validate_artifact(unsafe)


def test_strict_validation_rejects_unlisted_files(tmp_path: Path) -> None:
    """Allow callers to choose whether unlisted sidecar files are tolerated."""
    artifact = write_metabo_dataset(
        _sample_dataset(), tmp_path / "dataset.pimqc"
    )
    (artifact / "notes.txt").write_text("local note", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="unlisted"):
        validate_artifact(artifact)
    assert validate_artifact(artifact, strict=False).artifact_kind == "dataset"


def test_unsupported_root_and_nested_runtime_object_are_rejected(
    tmp_path: Path,
) -> None:
    """
    Fail explicitly for arbitrary roots and unregistered execution objects.
    """
    with pytest.raises(TypeError):
        write_metabo_dataset(pd.DataFrame(), tmp_path / "wrong.pimqc")  # type: ignore[arg-type]

    payload = FilteringPlotPayload(
        _sample_dataset(),
        audit_tables={"callable": lambda: None},
    )
    with pytest.raises(UnsupportedSerializationTypeError, match="function"):
        write_plot_payload(payload, tmp_path / "unsupported.pimqc")
    assert not (tmp_path / "unsupported.pimqc").exists()
