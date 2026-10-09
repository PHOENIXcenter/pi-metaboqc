"""Small contracts for common calculation-only missing-value adaptation."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction.missing_input import prepare_median_input


@pytest.mark.parametrize("scale", ["raw", "log1p"])
def test_median_uses_only_observed_fit_values_without_mutation(scale):
    """Compute placeholders from allowed observations on the requested scale."""
    data = pd.DataFrame([[1.0, np.nan, 9.0, 1e12]])
    original = data.copy()
    filled, mask, audit = prepare_median_input(
        data, np.array([True, True, True, False]), scale=scale
    )
    observed = np.array([1.0, 9.0])
    if scale == "log1p":
        observed = np.log1p(observed)
    assert filled.iloc[0, 1] == pytest.approx(np.median(observed))
    assert mask.iloc[0, 1]
    assert audit["temporary_filled_cells"] == 1
    assert audit["reference"] == "non_blank_observed"
    pd.testing.assert_frame_equal(data, original)


def test_no_reference_is_not_replaced_by_blank_or_zero():
    """Fail when no fitting observations support a feature's placeholder."""
    data = pd.DataFrame([[np.nan, np.nan, 100.0]], index=["F0"])
    with pytest.raises(ValueError, match="fully missing features"):
        prepare_median_input(data, np.array([True, True, False]))


def test_blank_missingness_does_not_claim_fitting_imputation():
    """Distinguish placeholders outside the fit from actual fitting repairs."""
    filled, mask, audit = prepare_median_input(
        pd.DataFrame([[1.0, 3.0, np.nan]]),
        np.array([True, True, False]),
    )
    assert filled.iloc[0, 2] == 2.0
    assert mask.iloc[0, 2]
    assert not audit["applied"]


def test_infinity_is_not_silently_imputed():
    """Reject infinite correction inputs before temporary median filling."""
    with pytest.raises(ValueError, match="Infinite"):
        prepare_median_input(
            pd.DataFrame([[1.0, np.inf]]), np.array([True, True])
        )
