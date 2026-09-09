"""Public stable disk-serialization API for pi-metaboqc artifacts.

The package exposes typed helpers for datasets, processing audits, and plotting
payloads plus validation primitives for host integrations.  The v1 format is a
checksummed directory of canonical JSON and non-pickle NPY members.
"""

from .artifact import (
    ArtifactFile,
    ArtifactManifest,
    read_artifact,
    read_audit_payload,
    read_metabo_dataset,
    read_plot_payload,
    validate_artifact,
    write_artifact,
    write_audit_payload,
    write_metabo_dataset,
    write_plot_payload,
)
from .errors import (
    ArtifactExistsError,
    ArtifactValidationError,
    SerializationError,
    UnknownWireTypeError,
    UnsupportedFormatVersionError,
    UnsupportedSerializationTypeError,
)

__all__ = [
    "ArtifactExistsError",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactValidationError",
    "SerializationError",
    "UnknownWireTypeError",
    "UnsupportedFormatVersionError",
    "UnsupportedSerializationTypeError",
    "read_artifact",
    "read_audit_payload",
    "read_metabo_dataset",
    "read_plot_payload",
    "validate_artifact",
    "write_artifact",
    "write_audit_payload",
    "write_metabo_dataset",
    "write_plot_payload",
]
