"""Portable JSON and NPY codec for scientific pi-metaboqc object graphs.

The codec preserves pandas axes and dtypes, NumPy arrays, non-finite scalars,
registered domain dataclasses, and ordinary pandas objects. Binary
arrays use NumPy's non-object NPY representation with pickle explicitly
disabled; object arrays and extension dtypes are represented recursively in
tagged JSON.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import base64
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .errors import (
    ArtifactValidationError,
    UnknownWireTypeError,
    UnsupportedSerializationTypeError,
)
from .registry import class_for_wire_name, wire_name_for


class ArtifactEncoder:
    """
    Encode a supported object graph into JSON nodes and external NPY files.
    """

    def __init__(self, root: Path) -> None:
        """Initialize an encoder writing binary members below ``root``."""
        self.root = root
        self.array_directory = root / "arrays"
        self.array_count = 0
        self.generated_files: list[Path] = []

    def encode(self, value: Any) -> Any:
        """Return a JSON-compatible tagged representation of ``value``."""
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            return self._encode_float(value)
        if isinstance(value, complex):
            return {
                "$type": "complex",
                "real": self._encode_float(value.real),
                "imag": self._encode_float(value.imag),
            }
        if isinstance(value, bytes):
            return {
                "$type": "bytes",
                "encoding": "base64",
                "value": base64.b64encode(value).decode("ascii"),
            }
        if value is pd.NA:
            return {"$type": "pandas_na"}
        if value is pd.NaT:
            return {"$type": "pandas_nat"}
        if isinstance(value, np.generic):
            return self._encode_numpy_scalar(value)
        if isinstance(value, pd.DataFrame):
            return self._encode_dataframe(value)
        if isinstance(value, pd.Series):
            return self._encode_series(value)
        if isinstance(value, pd.Index):
            return self._encode_index(value)
        if isinstance(value, np.ndarray):
            return self._encode_ndarray(value)
        if isinstance(value, Mapping):
            return {
                "$type": "mapping",
                "items": [
                    [self.encode(key), self.encode(item)]
                    for key, item in value.items()
                ],
            }
        if isinstance(value, list):
            return {"$type": "list", "items": [self.encode(x) for x in value]}
        if isinstance(value, tuple):
            return {
                "$type": "tuple",
                "items": [self.encode(x) for x in value],
            }
        if isinstance(value, (set, frozenset)):
            nodes = [self.encode(x) for x in value]
            nodes.sort(key=_canonical_node_key)
            return {
                "$type": "frozenset" if isinstance(value, frozenset) else "set",
                "items": nodes,
            }
        if isinstance(value, Path):
            return {"$type": "path", "value": str(value)}
        if isinstance(value, dt.datetime):
            return {"$type": "datetime", "value": value.isoformat()}
        if isinstance(value, dt.date):
            return {"$type": "date", "value": value.isoformat()}
        if isinstance(value, dt.time):
            return {"$type": "time", "value": value.isoformat()}
        if isinstance(value, dt.timedelta):
            return {
                "$type": "timedelta",
                "days": value.days,
                "seconds": value.seconds,
                "microseconds": value.microseconds,
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._encode_dataclass(value)
        raise UnsupportedSerializationTypeError(
            "No stable serialization is registered for "
            f"{type(value).__module__}.{type(value).__qualname__}."
        )

    def _encode_float(self, value: float) -> Any:
        if math.isnan(value):
            return {"$type": "float", "value": "nan"}
        if math.isinf(value):
            return {
                "$type": "float",
                "value": "inf" if value > 0 else "-inf",
            }
        return value

    def _encode_numpy_scalar(self, value: np.generic) -> dict[str, Any]:
        array = np.asarray(value)
        if array.dtype.hasobject:
            return {
                "$type": "numpy_scalar_object",
                "dtype": array.dtype.str,
                "value": self.encode(value.item()),
            }
        if array.dtype.fields is None and array.dtype.subdtype is None:
            return {
                "$type": "numpy_scalar_json",
                "dtype": array.dtype.str,
                "value": self.encode(value.item()),
            }
        node = self._encode_ndarray(array)
        node["$type"] = "numpy_scalar"
        return node

    def _encode_ndarray(self, value: np.ndarray) -> dict[str, Any]:
        array = np.asarray(value)
        if array.dtype.hasobject:
            return {
                "$type": "ndarray_object",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "items": [self.encode(x) for x in array.ravel(order="C")],
            }
        self.array_count += 1
        relative = PurePosixPath("arrays") / f"{self.array_count:06d}.npy"
        destination = self.root / Path(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, array, allow_pickle=False)
        self.generated_files.append(destination)
        return {
            "$type": "ndarray",
            "path": relative.as_posix(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }

    def _encode_values(self, values: Any) -> dict[str, Any]:
        dtype = values.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            categorical = pd.Categorical(values)
            return {
                "storage": "categorical",
                "categories": self._encode_index(
                    pd.Index(categorical.categories)
                ),
                "ordered": categorical.ordered,
                "codes": self._encode_ndarray(
                    np.asarray(categorical.codes, dtype=np.int64)
                ),
            }
        if isinstance(dtype, pd.DatetimeTZDtype):
            array = pd.array(values, dtype=dtype)
            return {
                "storage": "datetime_tz",
                "dtype": str(dtype),
                "values": self._encode_ndarray(array.asi8),
            }
        if isinstance(dtype, pd.PeriodDtype):
            array = pd.array(values, dtype=dtype)
            return {
                "storage": "period",
                "dtype": str(dtype),
                "values": self._encode_ndarray(array.asi8),
            }
        if isinstance(dtype, pd.IntervalDtype):
            array = pd.arrays.IntervalArray(values)
            return {
                "storage": "interval",
                "closed": array.closed,
                "left": self._encode_values(array.left.array),
                "right": self._encode_values(array.right.array),
            }
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            return {
                "storage": "extension",
                "dtype": str(dtype),
                "items": [self.encode(x) for x in values.tolist()],
            }
        array = np.asarray(values)
        return {"storage": "numpy", "values": self._encode_ndarray(array)}

    def _encode_index(self, value: pd.Index) -> dict[str, Any]:
        if isinstance(value, pd.RangeIndex):
            return {
                "$type": "range_index",
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
                "name": self.encode(value.name),
            }
        if isinstance(value, pd.MultiIndex):
            return {
                "$type": "multi_index",
                "levels": [self._encode_index(level) for level in value.levels],
                "codes": [
                    self._encode_ndarray(np.asarray(code, dtype=np.int64))
                    for code in value.codes
                ],
                "names": [self.encode(name) for name in value.names],
                "sortorder": value.sortorder,
            }
        return {
            "$type": "index",
            "name": self.encode(value.name),
            "values": self._encode_values(value.array),
        }

    def _encode_dataframe(self, value: pd.DataFrame) -> dict[str, Any]:
        return {
            "$type": "dataframe",
            "index": self._encode_index(value.index),
            "columns": self._encode_index(value.columns),
            "data": [
                self._encode_values(value.iloc[:, position].array)
                for position in range(value.shape[1])
            ],
            "attrs": self.encode(dict(value.attrs)),
        }

    def _encode_series(self, value: pd.Series) -> dict[str, Any]:
        return {
            "$type": "series",
            "index": self._encode_index(value.index),
            "name": self.encode(value.name),
            "values": self._encode_values(value.array),
            "attrs": self.encode(dict(value.attrs)),
        }

    def _encode_dataclass(self, value: Any) -> dict[str, Any]:
        wire_name = wire_name_for(value)
        if wire_name is None:
            raise UnsupportedSerializationTypeError(
                "Dataclass is not present in the stable wire registry: "
                f"{type(value).__module__}.{type(value).__qualname__}."
            )
        return {
            "$type": "dataclass",
            "class": wire_name,
            "fields": {
                item.name: self.encode(getattr(value, item.name))
                for item in dataclasses.fields(value)
            },
        }


class ArtifactDecoder:
    """Decode a validated tagged object graph from an artifact directory."""

    def __init__(self, root: Path, allowed_files: set[str]) -> None:
        """Initialize a decoder restricted to manifest-listed member files."""
        self.root = root
        self.allowed_files = allowed_files

    def decode(self, node: Any) -> Any:
        """Reconstruct a supported value from one JSON-compatible node."""
        if node is None or isinstance(node, (bool, int, float, str)):
            return node
        if not isinstance(node, dict) or "$type" not in node:
            raise ArtifactValidationError("Malformed tagged payload node.")
        tag = node["$type"]
        handlers = {
            "float": self._decode_float,
            "complex": self._decode_complex,
            "bytes": self._decode_bytes,
            "pandas_na": lambda _: pd.NA,
            "pandas_nat": lambda _: pd.NaT,
            "numpy_scalar": self._decode_numpy_scalar,
            "numpy_scalar_json": self._decode_numpy_scalar_json,
            "numpy_scalar_object": self._decode_numpy_scalar_object,
            "ndarray": self._decode_ndarray,
            "ndarray_object": self._decode_object_ndarray,
            "mapping": self._decode_mapping,
            "list": lambda x: [self.decode(v) for v in x["items"]],
            "tuple": lambda x: tuple(self.decode(v) for v in x["items"]),
            "set": lambda x: {self.decode(v) for v in x["items"]},
            "frozenset": lambda x: frozenset(
                self.decode(v) for v in x["items"]
            ),
            "path": lambda x: Path(x["value"]),
            "datetime": lambda x: dt.datetime.fromisoformat(x["value"]),
            "date": lambda x: dt.date.fromisoformat(x["value"]),
            "time": lambda x: dt.time.fromisoformat(x["value"]),
            "timedelta": self._decode_timedelta,
            "range_index": self._decode_range_index,
            "multi_index": self._decode_multi_index,
            "index": self._decode_index,
            "dataframe": self._decode_dataframe,
            "series": self._decode_series,
            "dataclass": self._decode_dataclass,
        }
        handler = handlers.get(tag)
        if handler is None:
            raise UnknownWireTypeError(f"Unknown payload wire type: {tag!r}.")
        try:
            return handler(node)
        except (ArtifactValidationError, UnknownWireTypeError):
            raise
        except Exception as exc:
            raise ArtifactValidationError(
                f"Invalid {tag!r} payload node: {exc}"
            ) from exc

    def _decode_float(self, node: dict[str, Any]) -> float:
        values = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
        try:
            return values[node["value"]]
        except KeyError as exc:
            raise ArtifactValidationError(
                "Invalid non-finite float tag."
            ) from exc

    def _decode_complex(self, node: dict[str, Any]) -> complex:
        real = self.decode(node["real"])
        imag = self.decode(node["imag"])
        return complex(real, imag)

    def _decode_bytes(self, node: dict[str, Any]) -> bytes:
        if node.get("encoding") != "base64":
            raise ArtifactValidationError("Unknown bytes encoding.")
        try:
            return base64.b64decode(node["value"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                "Invalid base64 bytes value."
            ) from exc

    def _member_path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in relative
            or ":" in relative
            or relative not in self.allowed_files
        ):
            raise ArtifactValidationError(
                "Payload references an unsafe or unlisted member: "
                f"{relative!r}."
            )
        return self.root / Path(*pure.parts)

    def _decode_ndarray(self, node: dict[str, Any]) -> np.ndarray:
        source = self._member_path(node["path"])
        array = np.load(source, allow_pickle=False)
        expected_shape = tuple(node["shape"])
        if array.shape != expected_shape or array.dtype.str != node["dtype"]:
            raise ArtifactValidationError(
                f"Array metadata does not match {node['path']!r}."
            )
        return array

    def _decode_object_ndarray(self, node: dict[str, Any]) -> np.ndarray:
        items = [self.decode(item) for item in node["items"]]
        shape = tuple(node["shape"])
        expected_size = math.prod(shape)
        if len(items) != expected_size:
            raise ArtifactValidationError(
                "Object array shape does not match items."
            )
        dtype = np.dtype(node["dtype"])
        array = np.empty(len(items), dtype=dtype)
        array[:] = items
        return array.reshape(shape)

    def _decode_numpy_scalar(self, node: dict[str, Any]) -> np.generic:
        array_node = dict(node)
        array_node["$type"] = "ndarray"
        array = self._decode_ndarray(array_node)
        if array.shape != ():
            raise ArtifactValidationError(
                "NumPy scalar member is not zero-dimensional."
            )
        return array[()]

    def _decode_numpy_scalar_json(self, node: dict[str, Any]) -> np.generic:
        value = self.decode(node["value"])
        return np.asarray(value, dtype=np.dtype(node["dtype"]))[()]

    def _decode_numpy_scalar_object(self, node: dict[str, Any]) -> np.generic:
        array = np.asarray(self.decode(node["value"]), dtype=node["dtype"])
        return array[()]

    def _decode_mapping(self, node: dict[str, Any]) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for pair in node["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ArtifactValidationError(
                    "Mapping item is not a key-value pair."
                )
            key = self.decode(pair[0])
            try:
                result[key] = self.decode(pair[1])
            except TypeError as exc:
                raise ArtifactValidationError(
                    "Decoded mapping key is not hashable."
                ) from exc
        return result

    def _decode_timedelta(self, node: dict[str, Any]) -> dt.timedelta:
        return dt.timedelta(
            days=node["days"],
            seconds=node["seconds"],
            microseconds=node["microseconds"],
        )

    def _decode_values(self, node: dict[str, Any]) -> Any:
        storage = node.get("storage")
        if storage == "numpy":
            return self.decode(node["values"])
        if storage == "categorical":
            categories = self.decode(node["categories"])
            codes = self.decode(node["codes"])
            return pd.Categorical.from_codes(
                codes,
                categories=categories,
                ordered=node["ordered"],
            )
        if storage == "datetime_tz":
            values = self.decode(node["values"])
            dtype = pd.api.types.pandas_dtype(node["dtype"])
            unit = dtype.unit
            return pd.array(values.view(f"datetime64[{unit}]"), dtype=dtype)
        if storage == "period":
            values = self.decode(node["values"])
            dtype = pd.api.types.pandas_dtype(node["dtype"])
            return pd.arrays.PeriodArray(values, dtype=dtype)
        if storage == "interval":
            left = self._decode_values(node["left"])
            right = self._decode_values(node["right"])
            return pd.arrays.IntervalArray.from_arrays(
                left, right, closed=node["closed"]
            )
        if storage == "extension":
            items = [self.decode(item) for item in node["items"]]
            dtype = pd.api.types.pandas_dtype(node["dtype"])
            return pd.array(items, dtype=dtype)
        raise ArtifactValidationError(f"Unknown pandas storage: {storage!r}.")

    def _decode_range_index(self, node: dict[str, Any]) -> pd.RangeIndex:
        return pd.RangeIndex(
            start=node["start"],
            stop=node["stop"],
            step=node["step"],
            name=self.decode(node["name"]),
        )

    def _decode_multi_index(self, node: dict[str, Any]) -> pd.MultiIndex:
        levels = [self.decode(level) for level in node["levels"]]
        codes = [self.decode(code) for code in node["codes"]]
        names = [self.decode(name) for name in node["names"]]
        return pd.MultiIndex(
            levels=levels,
            codes=codes,
            names=names,
            sortorder=node.get("sortorder"),
            verify_integrity=True,
        )

    def _decode_index(self, node: dict[str, Any]) -> pd.Index:
        values = self._decode_values(node["values"])
        return pd.Index(values, name=self.decode(node["name"]), copy=False)

    def _decode_dataframe(self, node: dict[str, Any]) -> pd.DataFrame:
        index = self.decode(node["index"])
        columns = self.decode(node["columns"])
        encoded_data = node["data"]
        if len(encoded_data) != len(columns):
            raise ArtifactValidationError(
                "Dataframe column axis does not match encoded data."
            )
        if encoded_data:
            series = [
                pd.Series(self._decode_values(item), index=index, copy=False)
                for item in encoded_data
            ]
            result = pd.concat(series, axis=1)
            result.columns = columns
        else:
            result = pd.DataFrame(index=index, columns=columns)
        result.attrs = self.decode(node["attrs"])
        return result

    def _decode_series(self, node: dict[str, Any]) -> pd.Series:
        result = pd.Series(
            self._decode_values(node["values"]),
            index=self.decode(node["index"]),
            name=self.decode(node["name"]),
            copy=False,
        )
        result.attrs = self.decode(node["attrs"])
        return result

    def _decode_dataclass(self, node: dict[str, Any]) -> Any:
        cls = class_for_wire_name(node.get("class", ""))
        if cls is None:
            raise UnknownWireTypeError(
                f"Unknown registered dataclass: {node.get('class')!r}."
            )
        fields = node.get("fields")
        if not isinstance(fields, dict):
            raise ArtifactValidationError("Dataclass fields must be an object.")
        expected = {item.name for item in dataclasses.fields(cls)}
        if set(fields) != expected:
            raise ArtifactValidationError(
                f"Dataclass field set does not match {node['class']!r}."
            )
        return cls(**{key: self.decode(value) for key, value in fields.items()})


def _canonical_node_key(node: Any) -> str:
    """Return a deterministic sort key for already encoded set members."""
    import json

    return json.dumps(
        node,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
