"""Regression tests for BPCA centering, reconstruction, and edge contracts."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.imputation import BayesianPCAImputer, MissingValueImputer
from pimqc.processing.imputation.analysis import (
    BayesianPCAImputer as AnalysisBPCA,
)


def _incomplete_matrix() -> np.ndarray:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(24, 8)) + np.arange(8) * 5.0
    values[rng.random(values.shape) < 0.2] = np.nan
    return values


def test_public_bpca_import_is_preserved() -> None:
    """Keep the historical public BPCA import bound to the current estimator."""
    assert AnalysisBPCA is BayesianPCAImputer


@pytest.mark.parametrize("steps", [1, 10, 100])
def test_imputation_is_equivariant_to_feature_offsets(steps: int) -> None:
    """Feature offsets must not become apparent covariance at missing cells."""
    values = _incomplete_matrix()
    offsets = np.linspace(-100, 100, values.shape[1])
    base = BayesianPCAImputer(max_iter=steps).fit_transform(values)
    shifted = BayesianPCAImputer(max_iter=steps).fit_transform(values + offsets)
    np.testing.assert_allclose(shifted, base + offsets, atol=1e-10)
    np.testing.assert_array_equal(base[~np.isnan(values)],
                                  values[~np.isnan(values)])


def test_wrapper_preserves_signed_values_and_dataframe_identities() -> None:
    """Preserve signed observations and labels without mutating BPCA input."""
    values = _incomplete_matrix() - 50
    frame = pd.DataFrame(values.T, index=list("abcdefgh"))
    before = frame.copy(deep=True)
    result = MissingValueImputer.impute_by_bpca(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert result.index.equals(frame.index)
    assert result.columns.equals(frame.columns)
    observed = frame.notna().to_numpy()
    np.testing.assert_array_equal(result.to_numpy()[observed],
                                  frame.to_numpy()[observed])
    assert np.isfinite(result).all().all()
    assert (result.to_numpy()[~observed] < 0).any()


def test_final_reconstruction_uses_updated_loadings() -> None:
    """E-step yest is stale after the last loading update."""
    values = _incomplete_matrix()
    estimator = BayesianPCAImputer(max_iter=1)
    center = np.nanmean(values, axis=0)
    centered = values - center
    state = estimator._initialize_model(centered)
    state = estimator._do_step(state, centered)
    expected = state["scores"] @ state["pa"].T + center
    missing = np.isnan(values)
    assert np.max(np.abs(expected - (state["yest"] + center))) > 1e-3
    result = estimator.fit_transform(values)
    np.testing.assert_allclose(result[missing], expected[missing], atol=1e-12)
    assert estimator.n_iter_ == 1
    assert not estimator.converged_


def test_convergence_is_checked_every_ten_steps() -> None:
    """Check convergence on ten-step boundaries and honor the iteration
    limit.
    """
    estimator = BayesianPCAImputer(max_iter=50, threshold=100)
    estimator.fit_transform(_incomplete_matrix())
    assert estimator.n_iter_ == 10
    assert estimator.converged_
    estimator.threshold = 0
    estimator.max_iter = 23
    estimator.fit_transform(_incomplete_matrix())
    assert estimator.n_iter_ == 23
    assert not estimator.converged_


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[np.nan, np.nan], [np.nan, np.nan]], [[0, 0], [0, 0]]),
        ([[1, np.nan], [3, np.nan]], [[1, 2], [3, 2]]),
        ([[1], [np.nan], [5]], [[1], [3], [5]]),
        ([[1, np.nan, 5]], [[1, 3, 5]]),
        ([[2, -3], [np.nan, -3], [2, np.nan]], [[2, -3]] * 3),
    ],
)
def test_unidentifiable_and_constant_inputs_have_explicit_fallbacks(
    values: list, expected: list,
) -> None:
    """Verify defined fallback values when a BPCA model cannot be identified."""
    result = BayesianPCAImputer().fit_transform(np.array(values, dtype=float))
    np.testing.assert_array_equal(result, expected)


def test_entirely_missing_row_receives_observed_column_means() -> None:
    """Use observed column means for a row with no information."""
    values = _incomplete_matrix()
    values[0] = np.nan
    result = BayesianPCAImputer().fit_transform(values)
    np.testing.assert_allclose(result[0], np.nanmean(values, axis=0))


def test_all_missing_variable_does_not_change_other_predictions() -> None:
    """Keep an unobserved variable from perturbing estimable BPCA
    predictions.
    """
    values = _incomplete_matrix()
    extended = np.column_stack((values, np.full(len(values), np.nan)))
    estimator = BayesianPCAImputer()
    base = estimator.fit_transform(values)
    result = estimator.fit_transform(extended)
    np.testing.assert_array_equal(result[:, :-1], base)
    np.testing.assert_allclose(result[:, -1], np.nanmean(values))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 1)])
def test_complete_and_empty_inputs_return_a_copy(shape: tuple) -> None:
    """Return an independent unchanged array when no imputation is needed."""
    values = np.ones(shape)
    result = BayesianPCAImputer().fit_transform(values)
    np.testing.assert_array_equal(result, values)
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize("values", [[[1, np.inf]], [[-np.inf, np.nan]]])
def test_infinite_input_is_rejected(values: list) -> None:
    """Reject infinity rather than interpreting it as an imputable missing
    value.
    """
    with pytest.raises(ValueError, match="finite values or NaN"):
        BayesianPCAImputer().fit_transform(values)


def test_nonmatrix_input_is_rejected() -> None:
    """Require a two-dimensional BPCA input matrix."""
    with pytest.raises(ValueError, match="2D matrix"):
        BayesianPCAImputer().fit_transform(np.ones(3))
