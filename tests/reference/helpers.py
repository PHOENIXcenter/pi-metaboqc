"""Check optional original-package dependencies for production R tests."""

import pytest
import rpy2.robjects as ro


def require_r_package(package: str) -> None:
    """Skip a backend test when an optional R package is unavailable."""
    require_namespace = ro.r(
        "function(pkg) requireNamespace(pkg, quietly=TRUE)"
    )
    if not bool(require_namespace(package)[0]):
        pytest.skip(f"System R package '{package}' is not installed.")
