"""Numeric policies distinguish raw intensities from signed coordinates."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.numeric_domain import (
    apply_correction_domain,
    repair_imputed_log_values,
)
from pimqc.statistics.value_scale import positive_view_rsd


def test_correction_never_restores_raw_or_invents_positive_values():
    """Report invalid corrections as missing rather than fabricated
    positives.
    """
    reference = pd.DataFrame([[10, 20, 30, 40, 50, np.nan]])
    result, report = apply_correction_domain(
        pd.DataFrame([[-1, 0, np.inf, -np.inf, 25, np.nan]]), reference
    )
    assert result.iloc[0, :4].isna().all()
    assert result.iloc[0, 4] == 25
    assert report["new_missing_count"] == 4
    assert report["observed_coverage"] == 0.2
    assert report["input_missing_count"] == 1


def test_imputation_floor_is_role_local_raw_half_min():
    """Derive imputation repair floors from observed values in the same role."""
    training = pd.DataFrame([[10, np.nan, 100, np.nan]], dtype=float)
    logged = np.log2(training + 1)
    output = logged.fillna(-2)
    result, report = repair_imputed_log_values(
        output, logged, role_labels=np.array(["QC", "QC", "S", "S"])
    )
    np.testing.assert_allclose(np.expm1(result * np.log(2)), [[10, 5, 100, 50]])
    assert report["repaired_count"] == 2
    assert report["role_global_fallback_count"] == 0


def test_positive_low_predictions_are_not_raised_to_floor():
    """Keep valid positive predictions even when below a repair floor."""
    logged = np.log2(pd.DataFrame([[100, np.nan]]) + 1)
    output = logged.fillna(np.log2(1.001))
    result, report = repair_imputed_log_values(
        output, logged, role_labels=np.array(["S", "S"])
    )
    pd.testing.assert_frame_equal(result, output)
    assert report["repaired_count"] == 0


def test_partial_targets_do_not_use_other_imputations_as_floor_evidence():
    """Exclude previously imputed values from evidence used to set repair
    floors.
    """
    logged = np.log2(pd.DataFrame([[100, np.nan, np.nan]]) + 1)
    output = logged.copy()
    output.iloc[0, 1:] = [np.log2(1.01), -1]
    result, report = repair_imputed_log_values(
        output, logged, role_labels=np.array(["S"] * 3),
        target_mask=np.array([[False, False, True]]),
    )
    assert np.expm1(result.iloc[0, 2] * np.log(2)) == pytest.approx(50)
    assert result.iloc[0, 1] == output.iloc[0, 1]
    assert report["target_count"] == 1


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 2000.0])
def test_nonfinite_imputation_or_inverse_is_not_floor_filled(invalid):
    """Reject nonfinite predictions and inverse transforms instead of
    masking them.
    """
    training = pd.DataFrame([[3, np.nan]])
    output = pd.DataFrame([[3, invalid]])
    with pytest.raises(ValueError, match="nonfinite"):
        repair_imputed_log_values(
            output, training, role_labels=np.array(["S", "S"])
        )


def test_imputation_global_fallback_stays_inside_role():
    """Use within-role observed evidence when a feature-specific floor is
    absent.
    """
    training = pd.DataFrame([[np.nan, np.nan], [np.log2(9), np.log2(17)]])
    result, report = repair_imputed_log_values(
        training.fillna(0), training, role_labels=np.array(["S", "S"])
    )
    np.testing.assert_allclose(result.iloc[0], np.log2(5))
    assert report["role_global_fallback_count"] == 2
    with pytest.raises(ValueError, match="No positive training evidence"):
        repair_imputed_log_values(
            pd.DataFrame([[3, -1]]), pd.DataFrame([[3, np.nan]]),
            role_labels=np.array(["QC", "Sample"]),
        )


@pytest.mark.parametrize("scale", ["log2", "log2p1"])
def test_qa_inverse_rsd_matches_raw(scale):
    """Recover raw-scale RSD from supported invertible log transformations."""
    raw = pd.DataFrame([[0.2, 0.4, 0.8], [10, 20, 60]])
    transformed = np.log2(raw + (1 if scale == "log2p1" else 0))
    rsd, report = positive_view_rsd(transformed, {"value_scale": scale})
    np.testing.assert_allclose(rsd, raw.std(axis=1) / raw.mean(axis=1))
    assert report["is_raw_intensity_inverse"]


def test_vsn_negative_values_do_not_become_artificial_zero_rsd():
    """Calculate signed VSN dispersion without zero clipping or raw-inverse
    claims.
    """
    signed = pd.DataFrame([[-2.0, -1.0], [2000.0, 2001.0]])
    rsd, report = positive_view_rsd(signed, {"value_scale": "vsn_glog"})
    np.testing.assert_allclose(rsd, [np.sqrt(2) / 3] * 2)
    assert not report["is_raw_intensity_inverse"]
    assert report["valid_feature_count"] == 2


def test_scaled_data_and_insufficient_support_have_no_invented_rsd():
    """Leave RSD unavailable for scaled data or insufficient positive
    support.
    """
    values = pd.DataFrame([[1, -1], [np.nan, 10]])
    rsd, report = positive_view_rsd(values, {"is_scaled": True})
    assert rsd.isna().all()
    assert report["status"] == "unavailable"
    rsd, report = positive_view_rsd(values, {})
    assert rsd.isna().all()
    assert report["excluded_nonpositive_count"] == 1
