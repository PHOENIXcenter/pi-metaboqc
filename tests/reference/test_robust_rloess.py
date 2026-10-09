"""Compare native rLOESS against original MetaNorm and independent R fits."""

import numpy as np
import pytest

from pimqc.processing.correction.metanorm import MetanormRLOESSCorrector
from pimqc.processing.correction.rloess import RobustRLSCCorrector
from pimqc.processing.r_backend import run_r_function
from tests.unit.test_robust_rloess import _fixture

from .helpers import require_r_package


@pytest.mark.parametrize("selection", ["fixed", "gcv"])
def test_native_and_r_rloess_full_and_held_out_agree(selection):
    """Compare full-fit and held-out predictions across both rLOESS backends."""
    require_r_package("stats")
    data = _fixture()
    engines = [
        RobustRLSCCorrector(
            implementation=provider,
            span_selection=selection,
            span=0.75,
            random_state=123,
        )
        for provider in ("python", "r")
    ]
    outputs = [
        next(iter(engine.fit_transform(*data).values())) for engine in engines
    ]
    # Fixed-span fits share Cleveland's numerical kernel to roundoff. GCV
    # optimizers can select different intervals on a piecewise-flat objective.
    tolerance = 1e-10 if selection == "fixed" else 0.02
    for native, reference in zip(*outputs):
        np.testing.assert_array_equal(
            np.isfinite(native), np.isfinite(reference)
        )
        np.testing.assert_allclose(
            native, reference, rtol=tolerance, atol=1e-10, equal_nan=True
        )
    assert engines[1].diagnostics["selected_spans"]["full"]
    assert (
        engines[0].diagnostics["fold_assignments"]
        == (engines[1].diagnostics["fold_assignments"])
    )


def test_full_native_matches_unmodified_metanorm_worker():
    """Check the native full fit against the independently called MetaNorm
    worker.
    """
    require_r_package("metanorm")
    data = _fixture()
    native = next(
        iter(RobustRLSCCorrector(cv_folds=0).fit_transform(*data).values())
    )[0]
    original = next(
        iter(
            MetanormRLOESSCorrector(random_state=123)
            .fit_transform(*data)
            .values()
        )
    )[0]
    # No Blank or missing data: the nonblank adapter's center equals the
    # original full-input mean, so this checks the complete source workflow.
    np.testing.assert_allclose(native, original, rtol=0.01, atol=1e-10)


def test_both_backends_reject_insufficient_quadratic_span():
    """Reject a quadratic span with too few neighbors in either backend."""
    require_r_package("stats")
    frame, batch, qc, order = _fixture()
    for provider in ("python", "r"):
        engine = RobustRLSCCorrector(
            implementation=provider,
            span_selection="fixed",
            span=0.1,
        )
        full, oof = next(
            iter(
                engine.fit_transform(
                    frame.iloc[:, :8], batch[:8], qc[:8], order[:8]
                ).values()
            )
        )
        assert full.isna().all().all()
        assert oof.isna().all().all()
        assert engine.diagnostics["fit_warnings"]["full"]


def test_explicit_native_irls_matches_unmodified_r_symmetric():
    """Check lowesw iteration semantics against R, not our adapted R loop."""
    require_r_package("stats")
    frame, batch, qc, order = _fixture()
    logged = np.log2(frame)
    fitted, _ = run_r_function(
        logged,
        method="independent symmetric reference",
        package="stats",
        function="stats::loess",
        parameters={"order": order.tolist()},
        code="""function(mat, params) {
            result <- matrix(NA_real_, nrow(mat), ncol(mat))
            x <- params$order
            for (i in seq_len(nrow(mat))) {
                y <- mat[i, ]
                fit <- stats::loess(y ~ x, span=.75, degree=2,
                    family="symmetric",
                    control=stats::loess.control(iterations=4))
                result[i, ] <- predict(fit, data.frame(x=x))
            }
            result
        }""",
    )
    native = next(
        iter(
            RobustRLSCCorrector(
                span_selection="fixed",
                span=0.75,
                cv_folds=0,
            )
            .fit_transform(frame, batch, qc, order)
            .values()
        )
    )[0]
    expected = np.exp2(logged.subtract(fitted).add(logged.mean(axis=1), axis=0))
    np.testing.assert_allclose(native, expected, rtol=1e-11, atol=1e-10)
