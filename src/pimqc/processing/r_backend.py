"""Execute optional, original R implementations without implicit fallback.

R is imported only when requested. Calls run serially on the main Python
thread, preserve R's random state, and return ordinary dataframes and JSON-safe
provenance. Neither saved results nor the default Python workflow require R.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_R_LOCK = threading.RLock()


class RBackendError(RuntimeError):
    """The explicitly requested R computation could not be completed."""


class RBackendUnavailable(RBackendError):
    """R, rpy2, or a required original R package is unavailable."""


@contextmanager
def _quiet_r_console() -> Iterator[None]:
    """Temporarily silence R console callbacks while holding ``_R_LOCK``.

    R condition handlers collect warnings separately. Callback replacement
    also covers native R output and ``cat(..., file=stderr())``, without
    changing a user's sinks, R options, or Python/loguru output streams.
    """
    callbacks = importlib.import_module("rpy2.rinterface_lib.callbacks")
    originals = {
        name: getattr(callbacks, name)
        for name in ("consolewrite_print", "consolewrite_warnerror")
    }

    def discard(text: str) -> None:
        pass

    try:
        for name in originals:
            setattr(callbacks, name, discard)
        yield
    finally:
        for name, callback in originals.items():
            setattr(callbacks, name, callback)


def _load_r() -> Any:
    """Import the embedded runtime lazily, only from the main thread."""
    if threading.current_thread() is not threading.main_thread():
        raise RBackendError(
            "R implementations must run on the main Python thread; "
            "disable threaded candidate execution."
        )
    try:
        _configure_r_environment()
        with _quiet_r_console():
            return importlib.import_module("rpy2.robjects")
    except RBackendUnavailable:
        raise
    except Exception as exc:
        raise RBackendUnavailable(
            "Cannot initialize R/rpy2. Install R and pi-metaboqc[r], "
            "verify R_HOME, and run python -m rpy2.situation. "
            f"Original error: {exc}"
        ) from exc


def _configure_r_environment() -> None:
    """Set R_HOME and native DLL paths before rpy2 initializes R.

    ``rpy2`` reads these values at import time. This is intentionally done only
    when the optional backend is requested, so importing ``pimqc`` remains
    independent of an R installation.
    """
    if os.environ.get("R_HOME"):
        r_home = os.environ["R_HOME"]
    elif sys.platform == "win32":
        r_home = None
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R"
            ) as key:
                r_home = winreg.QueryValueEx(key, "InstallPath")[0]
        except (FileNotFoundError, OSError):
            r_home = None
        if r_home is None:
            executable = shutil.which("R")
            if executable is not None:
                result = subprocess.run(
                    [executable, "RHOME"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                r_home = result.stdout.strip()
    else:
        executable = shutil.which("R")
        r_home = None
        if executable is not None:
            result = subprocess.run(
                [executable, "RHOME"],
                check=True,
                capture_output=True,
                text=True,
            )
            r_home = result.stdout.strip()
    if not r_home:
        raise RBackendUnavailable(
            "R was not found. Install R, set R_HOME, and install "
            "pi-metaboqc[r]."
        )
    os.environ["R_HOME"] = str(r_home)
    if sys.platform == "win32":
        paths = [
            os.path.join(str(r_home), "bin"),
            os.path.join(str(r_home), "bin", "x64"),
        ]
        current = os.environ.get("PATH", "").split(os.pathsep)
        os.environ["PATH"] = os.pathsep.join(
            [path for path in paths if path not in current] + current
        )
        if os.path.isdir(os.path.join(str(r_home), "bin", "x64")):
            os.environ.setdefault("R_ARCH", "/x64")


def _rpy2_version() -> str:
    """Return installed distribution metadata without importing R."""
    try:
        return importlib.metadata.version("rpy2")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_r_environment(packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Report runtime/package availability without installing anything.

    This initializes embedded R on the main thread. ``available`` is true
    only if R starts and all explicitly requested packages can be loaded.
    """
    report: dict[str, Any] = {
        "available": False,
        "rpy2_version": _rpy2_version(),
        "packages": {},
    }
    with _R_LOCK:
        try:
            ro = _load_r()
            with _quiet_r_console():
                report["r_version"] = str(ro.r("R.version.string")[0])
                report["r_home"] = str(ro.r("R.home()")[0])
                inspect = ro.r(
                    "function(pkg) { "
                    "if (!requireNamespace(pkg, quietly=TRUE)) return(''); "
                    "as.character(utils::packageVersion(pkg)) }"
                )
                for package in packages:
                    captured = ro.r(_CAPTURE_CONDITIONS)(inspect, package)
                    value, warnings = _read_r_conditions(
                        captured, f"package inspection ({package})"
                    )
                    version = str(value[0])
                    report["packages"][package] = {
                        "available": bool(version),
                        "version": version or None,
                        "warnings": warnings,
                    }
            report["available"] = all(
                item["available"] for item in report["packages"].values()
            )
        except Exception as exc:
            report["error"] = str(exc)
    return report


def _as_r_value(ro: Any, value: Any) -> Any:
    """Convert explicitly JSON-compatible options, not dataframe labels."""
    if value is None:
        return ro.NULL
    if isinstance(value, Mapping):
        return ro.ListVector(
            {str(key): _as_r_value(ro, item) for key, item in value.items()}
        )
    if isinstance(value, bool):
        return ro.BoolVector([value])
    if isinstance(value, int):
        return ro.IntVector([value])
    if isinstance(value, float):
        return ro.FloatVector([value])
    if isinstance(value, str):
        return ro.StrVector([value])
    if isinstance(value, list):
        if not value:
            return ro.ListVector({})
        if all(isinstance(item, bool) for item in value):
            return ro.BoolVector(value)
        if all(isinstance(item, str) for item in value):
            return ro.StrVector(value)
        if all(isinstance(item, (int, float)) for item in value):
            return ro.FloatVector(value)
    raise TypeError("R parameters must be JSON scalars or homogeneous lists.")


_CAPTURE_CONDITIONS = """
function(fun, ...) {
    warnings <- character()
    error <- character()
    value <- tryCatch(withCallingHandlers(fun(...),
        warning=function(w) {
            warnings <<- c(warnings, conditionMessage(w))
            invokeRestart("muffleWarning")
        },
        message=function(m) invokeRestart("muffleMessage")
    ), error=function(e) {
        error <<- conditionMessage(e)
        NULL
    })
    list(value=value, warnings=warnings, error=error)
}
"""


def _read_r_conditions(
    captured: Any, context: str
) -> tuple[Any, list[str]]:
    """Keep original warnings in provenance and report them through loguru."""
    warnings = list(captured.rx2("warnings"))
    if warnings:
        logger.warning(
            "R {} emitted {} warning(s): {}",
            context,
            len(warnings),
            "; ".join(dict.fromkeys(warnings)),
        )
    errors = list(captured.rx2("error"))
    if errors:
        raise RBackendError(f"R {context} failed: {errors[0]}")
    return captured.rx2("value"), warnings


_EXECUTE = """
function(fun, mat, params, seed, package) {
    had_seed <- exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE)
    old_seed <- if (had_seed) get(".Random.seed", envir=.GlobalEnv) else NULL
    old_kind <- RNGkind()
    on.exit({
        do.call(RNGkind, as.list(old_kind))
        if (had_seed) {
            assign(".Random.seed", old_seed, envir=.GlobalEnv)
        } else if (exists(".Random.seed", envir=.GlobalEnv, inherits=FALSE)) {
            rm(".Random.seed", envir=.GlobalEnv)
        }
    }, add=TRUE)
    if (!requireNamespace(package, quietly=TRUE)) {
        stop(paste("Required R package is unavailable:", package))
    }
    if (!is.null(seed)) {
        RNGkind("Mersenne-Twister", "Inversion", "Rejection")
        set.seed(as.integer(seed))
    }
    mat[is.nan(mat)] <- NA_real_
    result <- fun(mat, params)
    if (!is.matrix(result) || !is.numeric(result)) {
        stop("R adapter must return a numeric matrix")
    }
    description <- utils::packageDescription(package)
    source_fields <- c("Repository", "RemoteType", "RemoteHost", "RemoteRepo",
                       "RemoteUsername", "RemoteRef", "RemoteSha")
    source <- lapply(source_fields, function(field) description[[field]])
    names(source) <- source_fields
    source <- source[!vapply(source, is.null, logical(1))]
    list(data=result, diagnostics=attr(result, "pimqc_diagnostics"),
         rng_kind=RNGkind(),
         package_source=source,
         package_version=as.character(utils::packageVersion(package)))
}
"""


def _validate_input(
    data: pd.DataFrame, *, allow_all_missing_input: bool = False
) -> np.ndarray:
    """Require a nonempty feature-by-sample matrix with finite observations."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("R implementations require a pandas DataFrame.")
    if data.empty:
        raise ValueError("R implementations require a nonempty matrix.")
    values = data.to_numpy(dtype=float, copy=True)
    if np.isinf(values).any():
        raise ValueError("R input contains infinite measurements.")
    if not allow_all_missing_input and np.isnan(values).all(axis=0).any():
        raise ValueError("R input contains a completely missing sample.")
    if not allow_all_missing_input and np.isnan(values).all(axis=1).any():
        raise ValueError("R input contains a completely missing feature.")
    return values


def _validate_output(
    original: np.ndarray,
    result: np.ndarray,
    *,
    preserve_observed: bool,
    allow_missing: bool,
    defer_output_domain: bool = False,
) -> None:
    """Reject corrupted/incomplete R results instead of substituting Python."""
    if result.shape != original.shape:
        raise RBackendError(
            f"R output shape {result.shape} differs from {original.shape}."
        )
    observed = np.isfinite(original)
    required = observed if allow_missing else np.ones_like(observed)
    if not defer_output_domain and (
        np.isinf(result).any() or not np.isfinite(result[required]).all()
    ):
        raise RBackendError("R returned nonfinite values at required entries.")
    if preserve_observed and not np.array_equal(
        result[observed], original[observed]
    ):
        raise RBackendError("R imputation unexpectedly changed observations.")


def _diagnostics_to_python(value: Any) -> Any:
    """Convert optional adapter diagnostics to JSON-compatible audit data."""
    from rpy2 import robjects as ro
    from rpy2.robjects.vectors import ListVector

    if value is ro.NULL:
        return None
    if isinstance(value, ListVector):
        items = [_diagnostics_to_python(item) for item in value]
        if value.names is not ro.NULL:
            return dict(zip(map(str, value.names), items))
        return items
    items = list(value)
    return items[0] if len(items) == 1 else items


def run_r_function(
    data: pd.DataFrame,
    *,
    method: str,
    package: str,
    function: str,
    code: str,
    parameters: dict[str, Any],
    seed: int | None = None,
    transforms: tuple[str, ...] = (),
    preserve_observed: bool = False,
    allow_missing: bool = False,
    allow_all_missing_input: bool = False,
    defer_output_domain: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run an internal ``function(mat, params)`` R adapter under one lock.

    The matrix is features by samples; labels never pass through R dataframe
    name repair. ``allow_missing`` permits NA only where the input is missing.
    ``allow_all_missing_input`` explicitly delegates completely missing axes
    to an adapter whose original implementation handles or rejects them.
    Correction adapters defer result-domain handling to their stage policy;
    shape and observed-value contracts are never deferred. This helper is for
    trusted package adapters, not arbitrary user R code.
    Package console chatter is suppressed; warnings remain in provenance and
    are summarized through loguru. Existing console callbacks are restored
    even when loading or executing R fails.
    """
    values = _validate_input(
        data, allow_all_missing_input=allow_all_missing_input
    )
    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed <= 2**31 - 1
    ):
        raise ValueError("R seed must be an integer from 0 to 2**31 - 1.")
    seed = None if seed is None else int(seed)
    # Reject non-JSON values before initializing the optional R runtime.
    effective = json.loads(json.dumps(parameters, allow_nan=False))
    started = perf_counter()
    logger.debug("Starting R {} ({})", method, function)
    with _R_LOCK:
        ro = _load_r()
        with _quiet_r_console():
            try:
                matrix = ro.r["matrix"](
                    ro.FloatVector(values.ravel(order="F")),
                    nrow=values.shape[0],
                    ncol=values.shape[1],
                )
                captured = ro.r(_CAPTURE_CONDITIONS)(
                    ro.r(_EXECUTE),
                    ro.r(code),
                    matrix,
                    _as_r_value(ro, effective),
                    ro.NULL if seed is None else ro.IntVector([seed]),
                    package,
                )
                output, warnings = _read_r_conditions(
                    captured, f"{method} ({function})"
                )
                result = np.asarray(output.rx2("data"), dtype=float).copy()
                source = output.rx2("package_source")
                provenance: dict[str, Any] = {
                    "implementation": "r",
                    "method": method,
                    "r_version": str(ro.r("R.version.string")[0]),
                    "rpy2_version": _rpy2_version(),
                    "package": package,
                    "package_version": str(output.rx2("package_version")[0]),
                    "package_source": {
                        str(name): str(value[0])
                        for name, value in zip(source.names, source)
                    },
                    "function": function,
                    "parameters": effective,
                    "seed": seed,
                    "rng_kind": list(output.rx2("rng_kind")),
                    "warnings": warnings,
                    "transforms": list(transforms),
                }
                diagnostics = _diagnostics_to_python(
                    output.rx2("diagnostics")
                )
                if diagnostics is not None:
                    provenance["diagnostics"] = diagnostics
            except RBackendError:
                raise
            except Exception as exc:
                raise RBackendError(
                    f"R {method} ({function}) failed: {exc}"
                ) from exc
    _validate_output(
        values,
        result,
        preserve_observed=preserve_observed,
        allow_missing=allow_missing,
        defer_output_domain=defer_output_domain,
    )
    frame = pd.DataFrame(
        result, index=data.index.copy(), columns=data.columns.copy()
    )
    frame.attrs = copy.deepcopy(data.attrs)
    logger.debug(
        "Completed R {} ({}) in {:.3f}s",
        method,
        function,
        perf_counter() - started,
    )
    return frame, provenance


_METHODS: dict[str, dict[str, Any]] = {
    "BPCA": {
        "package": "pcaMethods",
        "function": "pcaMethods::pca / pcaMethods::completeObs",
        "parameters": {
            "n_components": 2,
            "max_iter": 100,
            "threshold": 1e-4,
        },
        "code": """
        function(mat, params) {
            fit <- pcaMethods::pca(
                t(mat), method="bpca", center=TRUE, scale="none",
                nPcs=as.integer(params$n_components),
                maxSteps=as.integer(params$max_iter),
                threshold=params$threshold)
            t(pcaMethods::completeObs(fit))
        }
        """,
        "transforms": (
            "transpose to samples by features for pcaMethods::pca",
            "transpose completeObs back to features by samples",
            "center=TRUE; scale=none; no external scale transformation",
        ),
        "preserve_observed": True,
    },
    "QRILC": {
        "package": "imputeLCMD",
        "function": "imputeLCMD::impute.QRILC",
        "parameters": {"tune_sigma": 1.0},
        "code": """
        function(mat, params) {
            imputeLCMD::impute.QRILC(
                mat, tune.sigma=params$tune_sigma)[[1]]
        }
        """,
        "transforms": ("input scale preserved; no clipping",),
        "preserve_observed": True,
    },
    "VSN": {
        "package": "vsn",
        "function": "vsn::vsn2 / vsn::predict",
        "parameters": {},
        "code": """
        function(mat, params) {
            fit <- vsn::vsn2(mat, verbose=FALSE)
            vsn::predict(fit, newdata=mat)
        }
        """,
        "transforms": ("original VSN generalized-log2 output; no extra log",),
        "allow_missing": True,
    },
}


def run_r_method(
    method: str,
    data: pd.DataFrame,
    *,
    seed: int | None = None,
    **parameters: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call original BPCA, QRILC, or VSN, with explicit supported options.

    BPCA and QRILC use the scale supplied by their caller. VSN takes unlogged
    intensities and returns the original package's generalized-log2 scale.
    Unsupported methods/options fail rather than silently using Python.
    """
    canonical = method.upper()
    if canonical not in _METHODS:
        raise ValueError(f"No original R implementation for {method!r}.")
    specification = dict(_METHODS[canonical])
    effective = dict(specification.pop("parameters"))
    unknown = parameters.keys() - effective.keys()
    if unknown:
        raise ValueError(
            f"Unsupported R {canonical} options: {sorted(unknown)}"
        )
    effective.update(parameters)
    for name, value in effective.items():
        integer = name in {"n_components", "max_iter"}
        valid_type = (
            isinstance(value, (int, np.integer))
            if integer
            else (isinstance(value, (int, float, np.number)))
        )
        if (
            isinstance(value, (bool, np.bool_))
            or not valid_type
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"R option {name!r} must be a positive number.")
        effective[name] = int(value) if integer else float(value)
    result, provenance = run_r_function(
        data,
        method=canonical,
        seed=seed,
        parameters=effective,
        **specification,
    )
    if canonical == "BPCA":
        provenance["r_arguments"] = {
            "method": "bpca",
            "center": True,
            "scale": "none",
            "nPcs": effective["n_components"],
            "maxSteps": effective["max_iter"],
            "threshold": effective["threshold"],
        }
    elif canonical == "QRILC":
        provenance["r_arguments"] = {"tune.sigma": effective["tune_sigma"]}
    else:
        provenance["r_arguments"] = {"verbose": False}
    return result, provenance
