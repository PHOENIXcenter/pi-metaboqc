"""Exceptions raised by the portable pi-metaboqc artifact format.

The serialization layer deliberately rejects unknown Python objects and
malformed artifacts instead of falling back to pickle.  These exception types
let callers distinguish unsupported in-memory state from damaged or
incompatible files on disk.
"""


class SerializationError(Exception):
    """Base class for all pi-metaboqc serialization failures."""


class UnsupportedSerializationTypeError(SerializationError, TypeError):
    """Raised when a value has no stable, explicitly registered wire form."""


class ArtifactExistsError(SerializationError, FileExistsError):
    """Raised when a write would replace an existing artifact directory."""


class ArtifactValidationError(SerializationError, ValueError):
    """Raised when an artifact manifest or one of its files is invalid."""


class UnsupportedFormatVersionError(ArtifactValidationError):
    """Raised when the artifact uses an unknown disk-format version."""


class UnknownWireTypeError(ArtifactValidationError):
    """Raised when encoded JSON names an unregistered tagged value type."""
