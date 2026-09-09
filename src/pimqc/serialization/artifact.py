"""Versioned directory artifacts for pi-metaboqc scientific state.

Each artifact contains canonical UTF-8 JSON, external non-pickle NPY members,
and a manifest with byte sizes and SHA-256 checksums.  Writes are staged in a
temporary sibling directory and published by rename, while reads validate the
entire manifest before decoding any scientific object.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Mapping

from ..core.dataset import MetaboDataset
from ..plotting.payloads import PlotPayload
from ..processing.audit import AuditPayload
from .codec import ArtifactDecoder, ArtifactEncoder
from .errors import (
    ArtifactExistsError,
    ArtifactValidationError,
    UnsupportedFormatVersionError,
    UnsupportedSerializationTypeError,
)


@dataclass(frozen=True)
class ArtifactFile:
    """Manifest metadata for one checksummed artifact member file."""

    path: str
    media_type: str
    size: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactFile":
        """Parse one member entry from an untrusted JSON mapping."""
        if not isinstance(value, Mapping):
            raise ArtifactValidationError(
                "Manifest file entry must be an object."
            )
        expected = {"path", "media_type", "size", "sha256"}
        if set(value) != expected:
            raise ArtifactValidationError("Invalid manifest file entry fields.")
        path = value["path"]
        media_type = value["media_type"]
        size = value["size"]
        sha256 = value["sha256"]
        if not isinstance(path, str) or not _is_safe_relative_path(path):
            raise ArtifactValidationError(f"Unsafe manifest path: {path!r}.")
        if not isinstance(media_type, str) or not media_type:
            raise ArtifactValidationError("Manifest media type must be text.")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArtifactValidationError("Manifest member size is invalid.")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256)
        ):
            raise ArtifactValidationError("Manifest SHA-256 value is invalid.")
        return cls(path, media_type, size, sha256)

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical JSON-ready member representation."""
        return {
            "path": self.path,
            "media_type": self.media_type,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """Validated top-level contract for a pi-metaboqc directory artifact."""

    format: str
    format_version: str
    artifact_kind: str
    artifact_type: str
    schema_version: str
    root: str
    files: tuple[ArtifactFile, ...]

    format_name: ClassVar[str] = "pi-metaboqc-artifact"
    current_version: ClassVar[str] = "1.0"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactManifest":
        """Parse and structurally validate an untrusted manifest mapping."""
        expected = {
            "format",
            "format_version",
            "artifact_kind",
            "artifact_type",
            "schema_version",
            "root",
            "files",
        }
        if set(value) != expected:
            raise ArtifactValidationError("Invalid manifest field set.")
        if value["format"] != cls.format_name:
            raise ArtifactValidationError("Not a pi-metaboqc artifact.")
        if value["format_version"] != cls.current_version:
            raise UnsupportedFormatVersionError(
                "Unsupported pi-metaboqc artifact format version: "
                f"{value['format_version']!r}."
            )
        textual = (
            "artifact_kind",
            "artifact_type",
            "schema_version",
            "root",
        )
        if any(
            not isinstance(value[key], str) or not value[key] for key in textual
        ):
            raise ArtifactValidationError(
                "Manifest identity fields must be text."
            )
        if not _is_safe_relative_path(value["root"]):
            raise ArtifactValidationError("Manifest root path is unsafe.")
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            raise ArtifactValidationError("Manifest files must be a list.")
        files = tuple(ArtifactFile.from_mapping(item) for item in raw_files)
        paths = [item.path for item in files]
        if len(paths) != len(set(paths)):
            raise ArtifactValidationError(
                "Manifest contains duplicate members."
            )
        if value["root"] not in paths:
            raise ArtifactValidationError(
                "Manifest root is not a listed member."
            )
        return cls(
            format=value["format"],
            format_version=value["format_version"],
            artifact_kind=value["artifact_kind"],
            artifact_type=value["artifact_type"],
            schema_version=value["schema_version"],
            root=value["root"],
            files=files,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical JSON-ready manifest representation."""
        return {
            "format": self.format,
            "format_version": self.format_version,
            "artifact_kind": self.artifact_kind,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "root": self.root,
            "files": [item.to_mapping() for item in self.files],
        }


def write_artifact(value: Any, path: str | Path) -> Path:
    """Write a supported dataset, Audit, or PlotPayload directory artifact.

    The destination must not exist.  This default prevents accidental loss of
    previous scientific results and keeps publication atomic from the reader's
    perspective.
    """
    kind, artifact_type, schema_version = _artifact_identity(value)
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise ArtifactExistsError(
            f"Artifact destination already exists: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.writing-",
            dir=destination.parent,
        )
    )
    try:
        encoder = ArtifactEncoder(temporary)
        payload = encoder.encode(value)
        payload_path = temporary / "payload.json"
        _write_canonical_json(payload_path, payload)
        members = [payload_path, *encoder.generated_files]
        file_entries = tuple(
            _describe_member(temporary, member)
            for member in sorted(
                members,
                key=lambda item: item.relative_to(temporary).as_posix(),
            )
        )
        manifest = ArtifactManifest(
            format=ArtifactManifest.format_name,
            format_version=ArtifactManifest.current_version,
            artifact_kind=kind,
            artifact_type=artifact_type,
            schema_version=schema_version,
            root="payload.json",
            files=file_entries,
        )
        _write_canonical_json(
            temporary / "manifest.json", manifest.to_mapping()
        )
        _publish_staged_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _publish_staged_directory(staged: Path, destination: Path) -> None:
    """Atomically publish a staged directory with bounded Windows retries.

    Windows virus scanners and indexers can briefly retain a handle to a newly
    created NPY or JSON member.  Retrying only the sharing-related
    ``PermissionError`` preserves the atomic rename contract without masking
    other publication failures.
    """
    attempts = 6 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.replace(staged, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (2**attempt))


def read_artifact(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    strict: bool = True,
) -> Any:
    """Validate and decode one artifact, optionally checking its broad kind."""
    root = Path(path).expanduser().resolve()
    manifest = validate_artifact(root, strict=strict)
    if expected_kind is not None and manifest.artifact_kind != expected_kind:
        raise ArtifactValidationError(
            f"Expected artifact kind {expected_kind!r}, found "
            f"{manifest.artifact_kind!r}."
        )
    payload_path = root / Path(*PurePosixPath(manifest.root).parts)
    payload = _read_json(payload_path)
    decoder = ArtifactDecoder(root, {item.path for item in manifest.files})
    value = decoder.decode(payload)
    kind, artifact_type, schema_version = _artifact_identity(value)
    actual_identity = (kind, artifact_type, schema_version)
    manifest_identity = (
        manifest.artifact_kind,
        manifest.artifact_type,
        manifest.schema_version,
    )
    if actual_identity != manifest_identity:
        raise ArtifactValidationError(
            "Decoded payload identity does not match its manifest."
        )
    return value


def validate_artifact(
    path: str | Path,
    *,
    strict: bool = True,
) -> ArtifactManifest:
    """Validate format identity, paths, sizes, checksums, and extra members."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactValidationError(
            f"Artifact path is not a directory: {root}."
        )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError("Artifact is missing manifest.json.")
    raw_manifest = _read_json(manifest_path)
    if not isinstance(raw_manifest, dict):
        raise ArtifactValidationError(
            "Artifact manifest must be a JSON object."
        )
    manifest = ArtifactManifest.from_mapping(raw_manifest)
    for member in manifest.files:
        member_path = root / Path(*PurePosixPath(member.path).parts)
        if member_path.is_symlink():
            raise ArtifactValidationError(
                f"Artifact member cannot be a symbolic link: {member.path!r}."
            )
        if not member_path.is_file():
            raise ArtifactValidationError(
                f"Artifact member is missing: {member.path!r}."
            )
        if member_path.stat().st_size != member.size:
            raise ArtifactValidationError(
                f"Artifact member size mismatch: {member.path!r}."
            )
        if _sha256(member_path) != member.sha256:
            raise ArtifactValidationError(
                f"Artifact member checksum mismatch: {member.path!r}."
            )
    if strict:
        expected = {"manifest.json", *(item.path for item in manifest.files)}
        actual = {
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file()
        }
        extras = sorted(actual - expected)
        if extras:
            raise ArtifactValidationError(
                f"Artifact contains unlisted files: {extras}."
            )
    return manifest


def write_metabo_dataset(dataset: MetaboDataset, path: str | Path) -> Path:
    """Write a composition-based metabolomics dataset artifact."""
    if not isinstance(dataset, MetaboDataset):
        raise TypeError("dataset must be a MetaboDataset instance.")
    return write_artifact(dataset, path)


def read_metabo_dataset(
    path: str | Path, *, strict: bool = True
) -> MetaboDataset:
    """Read and validate a composition-based metabolomics dataset artifact."""
    value = read_artifact(path, expected_kind="dataset", strict=strict)
    if not isinstance(value, MetaboDataset):
        raise ArtifactValidationError("Artifact did not contain MetaboDataset.")
    return value


def write_audit_payload(payload: AuditPayload, path: str | Path) -> Path:
    """Write one typed processing AuditPayload artifact."""
    if not isinstance(payload, AuditPayload):
        raise TypeError("payload must be an AuditPayload instance.")
    return write_artifact(payload, path)


def read_audit_payload(
    path: str | Path, *, strict: bool = True
) -> AuditPayload:
    """Read and validate one typed processing AuditPayload artifact."""
    value = read_artifact(path, expected_kind="audit", strict=strict)
    if not isinstance(value, AuditPayload):
        raise ArtifactValidationError("Artifact did not contain AuditPayload.")
    return value


def write_plot_payload(payload: PlotPayload, path: str | Path) -> Path:
    """Write one processor-free PlotPayload artifact."""
    if not isinstance(payload, PlotPayload):
        raise TypeError("payload must be a PlotPayload instance.")
    return write_artifact(payload, path)


def read_plot_payload(path: str | Path, *, strict: bool = True) -> PlotPayload:
    """Read and validate one processor-free PlotPayload artifact."""
    value = read_artifact(path, expected_kind="plot_payload", strict=strict)
    if not isinstance(value, PlotPayload):
        raise ArtifactValidationError("Artifact did not contain PlotPayload.")
    return value


def _artifact_identity(value: Any) -> tuple[str, str, str]:
    if isinstance(value, MetaboDataset):
        return "dataset", "metabo_dataset", "1.0"
    if isinstance(value, AuditPayload):
        return "audit", value.audit_type, value.schema_version
    if isinstance(value, PlotPayload):
        return "plot_payload", value.payload_type, value.schema_version
    raise UnsupportedSerializationTypeError(
        "Artifact roots must be MetaboDataset, AuditPayload, or PlotPayload; "
        f"received {type(value).__module__}.{type(value).__qualname__}."
    )


def _is_safe_relative_path(value: str) -> bool:
    pure = PurePosixPath(value)
    return (
        bool(value)
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "\\" not in value
        and ":" not in value
    )


def _write_canonical_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.write_text(serialized + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Cannot read valid JSON from {path.name}."
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_member(root: Path, path: Path) -> ArtifactFile:
    relative = path.relative_to(root).as_posix()
    media_type = (
        "application/json" if path.suffix == ".json" else "application/x-npy"
    )
    return ArtifactFile(
        path=relative,
        media_type=media_type,
        size=path.stat().st_size,
        sha256=_sha256(path),
    )
