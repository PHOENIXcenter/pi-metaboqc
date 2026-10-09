"""Verify QRILC's sampling contract without requiring R or RNG equivalence."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pimqc.processing.imputation import MissingValueImputer
from pimqc.processing.imputation.qrilc import fit_qrilc_tail, impute_qrilc


@pytest.mark.parametrize("tune_sigma", [0.25, 1.0, 2.0])
def test_tail_uses_linear_quantiles_and_sample_missing_fraction(
    tune_sigma,
) -> None:
    """Fit a normal-score OLS with the reference probability grids."""
    observed = np.array([-2.0, 0.2, 1.3, 3.8, 8.2, 11.0, 15.0])
    fraction = 0.3
    probabilities = np.linspace(0.001, 0.991, 100)
    scores = stats.norm.ppf(np.linspace(fraction + 0.001, 0.991, 100))
    empirical = np.quantile(observed, probabilities, method="linear")
    design = np.column_stack([np.ones(100), scores])
    location, scale = np.linalg.lstsq(design, empirical, rcond=None)[0]
    tail = fit_qrilc_tail(observed, fraction, tune_sigma)
    assert tail.location == pytest.approx(location, rel=1e-12)
    assert tail.fitted_scale == pytest.approx(scale, rel=1e-12)
    assert tail.sampling_sd == pytest.approx(scale * tune_sigma)
    assert tail.upper == pytest.approx(
        location + scale * stats.norm.ppf(fraction + 0.001)
    )


def test_tuning_changes_sd_but_not_fitted_center_or_upper_bound() -> None:
    """The truncation boundary uses fitted scale, not tuned sampling SD."""
    observed = np.linspace(-3.0, 12.0, 80)
    first = fit_qrilc_tail(observed, 0.2, tune_sigma=0.5)
    second = fit_qrilc_tail(observed, 0.2, tune_sigma=2.0)
    assert first.location == second.location
    assert first.fitted_scale == second.fitted_scale
    assert first.upper == second.upper
    assert second.sampling_sd == 4.0 * first.sampling_sd


def test_imputation_preserves_signed_values_labels_and_observations() -> None:
    """Do not silently clip negative values on the caller's analysis scale."""
    values = np.linspace(-5.0, -1.0, 120)
    values[:30] = np.nan
    frame = pd.DataFrame(
        {"sample": values, "complete": np.linspace(-9.0, 2.0, 120)},
        index=pd.Index([f"F{i}" for i in range(120)], name="feature"),
    )
    before = frame.copy(deep=True)
    output = impute_qrilc(frame, tune_sigma=1.0, seed=12)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_index_equal(output.index, frame.index)
    pd.testing.assert_index_equal(output.columns, frame.columns)
    missing = frame.isna().to_numpy()
    np.testing.assert_array_equal(
        output.to_numpy()[~missing], frame.to_numpy()[~missing]
    )
    draws = output.to_numpy()[missing]
    tail = fit_qrilc_tail(values[30:], 0.25)
    assert np.isfinite(draws).all()
    assert (draws < 0.0).all()
    assert (draws <= tail.upper).all()


def test_imputation_seed_is_reproducible_and_public_method_delegates() -> None:
    """Guarantee native reproducibility, not identical R random draws."""
    frame = pd.DataFrame({"sample": np.r_[np.full(25, np.nan), np.arange(75)]})
    first = impute_qrilc(frame, tune_sigma=0.5, seed=42)
    second = impute_qrilc(frame, tune_sigma=0.5, seed=42)
    other = impute_qrilc(frame, tune_sigma=0.5, seed=43)
    public = MissingValueImputer.impute_by_qrilc(
        frame, tune_sigma=0.5, global_seed=42
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, public)
    assert not np.array_equal(first.iloc[:25], other.iloc[:25])


@pytest.mark.parametrize("tune_sigma", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_tuning_is_rejected_even_without_missing_values(
    tune_sigma,
) -> None:
    """Do not repair invalid tuning with an implicit epsilon floor."""
    with pytest.raises(ValueError, match="finite and positive"):
        fit_qrilc_tail(np.arange(10.0), 0.2, tune_sigma)
    with pytest.raises(ValueError, match="finite and positive"):
        impute_qrilc(pd.DataFrame([1.0, 2.0]), tune_sigma=tune_sigma, seed=0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.99, 1.0, np.nan])
def test_invalid_missing_fraction_fails_explicitly(fraction) -> None:
    """The reference quantile grid requires a valid nondegenerate tail."""
    with pytest.raises(ValueError, match="missing fraction"):
        fit_qrilc_tail(np.arange(10.0), fraction)


@pytest.mark.parametrize("observed", [[1.0], [[1.0, 2.0]]])
def test_invalid_fit_shape_is_rejected(observed) -> None:
    """Fitting accepts only a vector with sufficient observations."""
    with pytest.raises(ValueError, match="at least two observations"):
        fit_qrilc_tail(np.asarray(observed), 0.2)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_observations(invalid) -> None:
    """Only the imputation entry point interprets NaN as missing."""
    with pytest.raises(ValueError, match="observations must be finite"):
        fit_qrilc_tail(np.array([1.0, 2.0, invalid]), 0.2)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_imputation_rejects_infinite_observations(invalid) -> None:
    """Infinite input must not reach fitting or a constant-fill fallback."""
    with pytest.raises(ValueError, match="finite or missing"):
        impute_qrilc(
            pd.DataFrame([1.0, invalid, np.nan]), tune_sigma=1.0, seed=0
        )


@pytest.mark.parametrize(
    ("values", "fill"),
    [
        ([np.nan, np.nan, np.nan, np.nan], 0.0),
        ([2.0, np.nan, np.nan, np.nan], 2.0),
        ([2.0, 3.0, np.nan, np.nan], 2.0),
        ([-2.0, -2.0, -2.0, np.nan], -2.0),
    ],
)
def test_constant_fill_extension_is_explicitly_warned(
    values, fill, monkeypatch
) -> None:
    """Sparse and constant columns are an extension, not a successful fit."""
    frame = pd.DataFrame({"sparse": values})
    messages = []
    monkeypatch.setattr(
        "pimqc.processing.imputation.qrilc.logger.warning",
        lambda message, *args: messages.append(message.format(*args)),
    )
    result = impute_qrilc(frame, tune_sigma=1.0, seed=0)
    assert len(messages) == 1
    assert "native constant-fill extension" in messages[0]
    assert "sparse" in messages[0]
    missing = frame.isna().to_numpy()
    np.testing.assert_array_equal(result.to_numpy()[missing], fill)
    np.testing.assert_array_equal(
        result.to_numpy()[~missing], frame.to_numpy()[~missing]
    )


def test_sparse_column_warnings_are_aggregated(monkeypatch, recwarn):
    """Each call logs one diagnostic with every fallback column identity."""
    frame = pd.DataFrame({
        "empty": [np.nan] * 4,
        "sparse": [2.0, 3.0, np.nan, np.nan],
        "constant": [-2.0, -2.0, -2.0, np.nan],
        "complete": [1.0, 2.0, 3.0, 4.0],
    })
    messages = []
    monkeypatch.setattr(
        "pimqc.processing.imputation.qrilc.logger.warning",
        lambda message, *args: messages.append(message.format(*args)),
    )
    result = impute_qrilc(frame, tune_sigma=1.0, seed=0)
    assert len(messages) == 1
    assert "3 columns" in messages[0]
    for column in ["empty", "sparse", "constant"]:
        assert repr(column) in messages[0]
        assert result.loc[frame[column].isna(), column].eq(
            frame[column].min() if frame[column].notna().any() else 0.0
        ).all()
    assert not recwarn
