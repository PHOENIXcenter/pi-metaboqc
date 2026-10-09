"""Direct production-adapter checks against the original R packages."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from loguru import logger
from rpy2.rinterface_lib import callbacks

from pimqc.processing.r_backend import (
    RBackendError,
    run_r_function,
    run_r_method,
)

from .helpers import require_r_package


@pytest.fixture(autouse=True)
def restore_r_random_state() -> Iterator[None]:
    """Keep test-only direct R calls from altering other reference tests."""
    had_seed = bool(ro.r('exists(".Random.seed", envir=.GlobalEnv)')[0])
    seed = ro.globalenv[".Random.seed"] if had_seed else ro.NULL
    kind = ro.r("RNGkind()")
    try:
        yield
    finally:
        ro.r(
            """
            function(kind, seed) {
                do.call(RNGkind, as.list(kind))
                if (!is.null(seed)) {
                    assign(".Random.seed", seed, envir=.GlobalEnv)
                } else if (exists(".Random.seed", envir=.GlobalEnv)) {
                    rm(".Random.seed", envir=.GlobalEnv)
                }
            }
            """
        )(kind, seed)


def _direct_r(method: str, source: pd.DataFrame) -> np.ndarray:
    """Execute the same original package call outside the adapter."""
    values = source.to_numpy(dtype=float)
    matrix = ro.r["matrix"](
        ro.FloatVector(values.ravel(order="F")),
        nrow=values.shape[0],
        ncol=values.shape[1],
    )
    if method == "BPCA":
        direct = ro.r(
            """
            function(mat) {
                mat[is.nan(mat)] <- NA_real_
                fit <- pcaMethods::pca(
                    t(mat), method="bpca", center=TRUE, scale="none",
                    nPcs=2, maxSteps=100, threshold=1e-4)
                t(pcaMethods::completeObs(fit))
            }
            """
        )(matrix)
    elif method == "QRILC":
        ro.r(
            'RNGkind("Mersenne-Twister", "Inversion", "Rejection"); '
            "set.seed(17)"
        )
        direct = ro.r(
            "function(mat) imputeLCMD::impute.QRILC(mat, tune.sigma=1.0)[[1]]"
        )(matrix)
    elif method == "VSN":
        direct = ro.r(
            """
            function(mat) {
                mat[is.nan(mat)] <- NA_real_
                fit <- vsn::vsn2(mat, verbose=FALSE)
                vsn::predict(fit, newdata=mat)
            }
            """
        )(matrix)
    else:  # pragma: no cover
        raise AssertionError(method)
    return np.asarray(direct, dtype=float)


def _data() -> pd.DataFrame:
    """Return a deterministic positive matrix with missing values."""
    rng = np.random.default_rng(17)
    values = rng.normal(8.0, 0.7, size=(60, 12))
    values += np.linspace(-0.5, 0.5, 12)
    values[0, 2] = np.nan
    values[2, 1] = np.nan
    values[4, 3] = np.nan
    return pd.DataFrame(
        values,
        index=[f"F{i}" for i in range(values.shape[0])],
        columns=[f"S{i}" for i in range(values.shape[1])],
    )


@pytest.mark.parametrize("package, method", [("pcaMethods", "BPCA")])
def test_bpca_production_adapter(package: str, method: str) -> None:
    """The production path calls pcaMethods and keeps observed entries."""
    require_r_package(package)
    source = _data()
    result, provenance = run_r_method(
        method,
        source,
        seed=17,
        n_components=2,
        max_iter=100,
        threshold=1e-4,
    )
    np.testing.assert_allclose(
        result.to_numpy(), _direct_r(method, source), rtol=1e-12, atol=1e-12
    )
    assert result.shape == _data().shape
    assert not result.isna().any().any()
    assert provenance["implementation"] == "r"
    assert provenance["package"] == package
    assert provenance["function"].startswith("pcaMethods::")
    assert provenance["parameters"]["n_components"] == 2
    assert provenance["r_arguments"] == {
        "method": "bpca",
        "center": True,
        "scale": "none",
        "nPcs": 2,
        "maxSteps": 100,
        "threshold": 1e-4,
    }
    assert isinstance(provenance["package_source"], dict)
    json.dumps(provenance, allow_nan=False)


def test_qrilc_production_adapter() -> None:
    """The production path calls imputeLCMD without a Python fallback."""
    require_r_package("imputeLCMD")
    source = _data()
    result, provenance = run_r_method("QRILC", source, seed=17, tune_sigma=1.0)
    np.testing.assert_allclose(
        result.to_numpy(), _direct_r("QRILC", source), rtol=1e-12, atol=1e-12
    )
    assert result.shape == source.shape
    assert not result.isna().any().any()
    assert provenance["package"] == "imputeLCMD"
    assert provenance["function"] == "imputeLCMD::impute.QRILC"


def test_vsn_production_adapter() -> None:
    """The production path calls vsn2/predict and preserves matrix labels."""
    require_r_package("vsn")
    source = _data()
    result, provenance = run_r_method("VSN", source)
    np.testing.assert_allclose(
        result.to_numpy(), _direct_r("VSN", source), rtol=1e-12, atol=1e-12
    )
    assert result.shape == source.shape
    assert result.index.equals(source.index)
    assert result.columns.equals(source.columns)
    assert provenance["package"] == "vsn"
    assert provenance["function"] == "vsn::vsn2 / vsn::predict"


def test_production_initialization_in_fresh_process() -> None:
    """R package loading must work without reference conftest's setup."""
    packages = ("pcaMethods", "imputeLCMD", "vsn")
    for package in packages:
        require_r_package(package)
    env = dict(os.environ)
    # Keep explicit R_HOME for custom installations, but remove the Windows
    # runtime preparation performed by conftest; the production adapter must
    # establish these paths independently in a new interpreter.
    r_home = env.get("R_HOME", "")
    env.pop("R_ARCH", None)
    env.pop("RPY2_CFFI_MODE", None)
    if r_home:
        base = os.path.normcase(os.path.normpath(r_home))
        env["PATH"] = os.pathsep.join(
            item
            for item in env.get("PATH", "").split(os.pathsep)
            if not os.path.normcase(os.path.normpath(item)).startswith(base)
        )
    code = (
        "import json; "
        "from pimqc.processing.r_backend import check_r_environment; "
        f"print(json.dumps(check_r_environment({packages!r})))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    report = next(
        json.loads(line)
        for line in process.stdout.splitlines()
        if line.startswith('{"available":')
    )
    assert report["available"], report


@pytest.mark.parametrize("existing_seed", [True, False])
def test_r_random_state_is_preserved(existing_seed: bool) -> None:
    """A seeded adapter must neither overwrite nor create global R RNG state."""
    if existing_seed:
        ro.r('RNGkind("L\'Ecuyer-CMRG"); set.seed(919)')
        before = np.array(ro.globalenv[".Random.seed"], copy=True)
    else:
        ro.r(
            'if (exists(".Random.seed", envir=.GlobalEnv)) '
            'rm(".Random.seed", envir=.GlobalEnv)'
        )
    before_kind = tuple(ro.r("RNGkind()"))
    output, provenance = run_r_function(
        _data().fillna(8.0),
        method="rng-test",
        package="stats",
        function="stats::runif",
        code="function(mat, params) {runif(3); mat}",
        parameters={},
        seed=18,
    )
    assert output.shape == _data().shape
    assert provenance["seed"] == 18
    assert tuple(ro.r("RNGkind()")) == before_kind
    if existing_seed:
        np.testing.assert_array_equal(ro.globalenv[".Random.seed"], before)
    else:
        assert not bool(ro.r('exists(".Random.seed", envir=.GlobalEnv)')[0])


def test_warnings_and_package_failure_are_explicit() -> None:
    """Capture warnings in audit data and propagate missing dependencies."""
    _, provenance = run_r_function(
        _data().fillna(8.0),
        method="warning-test",
        package="stats",
        function="base::warning",
        code='function(mat, params) {warning("adapter warning"); mat}',
        parameters={},
    )
    assert provenance["warnings"] == ["adapter warning"]
    with pytest.raises(RBackendError, match="Required R package"):
        run_r_function(
            _data(),
            method="absent-package",
            package="pimqcNonexistentTestPackage",
            function="unused",
            code="function(mat, params) mat",
            parameters={},
        )


@pytest.mark.parametrize("fail", [False, True])
def test_r_console_is_quiet_and_restored_with_conditions_preserved(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Real R chatter is quiet; loguru, warnings, RNG and callbacks survive."""
    console: list[str] = []
    records = []
    monkeypatch.setattr(callbacks, "consolewrite_print", console.append)
    monkeypatch.setattr(callbacks, "consolewrite_warnerror", console.append)
    print_callback = callbacks.consolewrite_print
    error_callback = callbacks.consolewrite_warnerror
    sink_state = 'c(sink.number(), sink.number(type="message"))'
    option_state = 'c(getOption("warn"), getOption("show.error.messages"))'
    before_sinks = tuple(ro.r(sink_state))
    before_options = tuple(ro.r(option_state))
    ro.r('RNGkind("L\'Ecuyer-CMRG"); set.seed(731)')
    before_seed = np.array(ro.globalenv[".Random.seed"], copy=True)
    before_kind = tuple(ro.r("RNGkind()"))
    source = _data().fillna(8.0)
    code = """
        function(mat, params) {
            print("hidden print")
            cat("hidden cat\\n")
            cat("hidden stderr\\n", file=stderr())
            message("hidden message")
            packageStartupMessage("hidden package startup")
            warning("retained warning")
            warning("retained warning")
            runif(3)
            if (params$fail) stop("retained failure")
            mat
        }
    """
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    try:
        arguments = dict(
            method="console-test",
            package="stats",
            function="base::identity",
            code=code,
            parameters={"fail": fail},
            seed=18,
        )
        if fail:
            with pytest.raises(RBackendError, match="retained failure"):
                run_r_function(source, **arguments)
        else:
            result, provenance = run_r_function(source, **arguments)
            pd.testing.assert_frame_equal(result, source)
            assert provenance["warnings"] == ["retained warning"] * 2
            assert any(
                "Completed R console-test" in str(item) for item in records
            )
        logger.info("existing Python stage log")
    finally:
        logger.remove(sink_id)
    assert console == []
    warning_records = [
        item for item in records if item.record["level"].name == "WARNING"
    ]
    assert len(warning_records) == 1
    assert "2 warning(s): retained warning" in str(warning_records[0])
    assert any("Starting R console-test" in str(item) for item in records)
    assert any("existing Python stage log" in str(item) for item in records)
    assert callbacks.consolewrite_print is print_callback
    assert callbacks.consolewrite_warnerror is error_callback
    assert tuple(ro.r(sink_state)) == before_sinks
    assert tuple(ro.r(option_state)) == before_options
    assert tuple(ro.r("RNGkind()")) == before_kind
    np.testing.assert_array_equal(ro.globalenv[".Random.seed"], before_seed)
    ro.r(
        'cat("visible after adapter\\n"); '
        'message("visible message after adapter")'
    )
    assert "visible after adapter" in "".join(console)
    assert "visible message after adapter" in "".join(console)


def test_r_console_restores_after_adapter_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error before the R condition wrapper still restores callbacks."""
    console: list[str] = []
    monkeypatch.setattr(callbacks, "consolewrite_print", console.append)
    monkeypatch.setattr(callbacks, "consolewrite_warnerror", console.append)
    print_callback = callbacks.consolewrite_print
    error_callback = callbacks.consolewrite_warnerror
    with pytest.raises(RBackendError, match="parse-test"):
        run_r_function(
            _data().fillna(8.0),
            method="parse-test",
            package="stats",
            function="unused",
            code="function(mat, params) {",
            parameters={},
        )
    assert console == []
    assert callbacks.consolewrite_print is print_callback
    assert callbacks.consolewrite_warnerror is error_callback
