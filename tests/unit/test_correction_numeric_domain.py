"""Correction output, provenance, and fixed-support scoring contracts."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.algorithms import (
    fit_predict_intra_batch_safely,
)
from pimqc.processing.r_backend import RBackendError, _validate_output
from tests.unit.test_audit_regressions import _dataset


def test_full_and_oof_domain_policy_precedes_metrics(monkeypatch, tmp_path):
    """Invalid results become missing with plain IDs, not changed MV labels."""
    from pimqc.processing.correction import analysis
    from pimqc.serialization import read_metabo_dataset, write_metabo_dataset

    class Provider:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, intensity_df, **kwargs):
            result = intensity_df.copy()
            result.iloc[0, 0] = -1
            result.iloc[0, 3] = 0
            result.iloc[1, 4] = np.inf
            result.iloc[2, 5] = -np.inf
            return {"Changed": (result, result.copy())}

    monkeypatch.setattr(analysis, "RegressionCorrector", Provider)
    source = _dataset(mechanism="MNAR")
    source.feature_metadata["correction_missing_samples"] = pd.Series(
        [("earlier",)] + [()] * 19,
        index=source.feature_metadata.index,
        dtype=object,
    )
    stage = SignalCorrector(source, base_est="QC-SVR").run_signal_correction()
    result = stage.data["Changed"]
    assert result.intensity.isna().sum().sum() == 4
    assert result.feature_metadata.loc["F0", "missingness_type"] == "MNAR"
    assert result.feature_metadata.loc["F0", "correction_missing_samples"] == (
        "earlier", "Q0", "S0"
    )
    assert result.context.extra_attrs["value_scale"] == "raw_positive"
    assert source.intensity.notna().all().all()
    selected = stage.audit.metrics["selection"]
    domain = selected["numeric_domain"]["Changed"]
    assert domain["full"]["new_missing_count"] == 4
    assert domain["oof"]["new_missing_count"] == 4
    validation = selected["validation"]
    assert validation["score_qc_observed_coverage"] < 1
    assert validation["score_sample_observed_coverage"] < 1
    path = write_metabo_dataset(result, tmp_path / "corrected")
    restored = read_metabo_dataset(path)
    assert tuple(restored.feature_metadata.loc[
        "F0", "correction_missing_samples"
    ]) == ("earlier", "Q0", "S0")
    assert restored.context.extra_attrs["value_scale"] == "raw_positive"


def test_r_serrf_restored_na_reaches_imputation_and_observed_scoring(
    monkeypatch,
):
    """Stage scoring ignores original gaps without consuming later
    imputation.
    """
    from pimqc.processing.correction import serrf_r, serrf_r_validation
    from pimqc.processing.imputation import MissingValueImputer
    from tests.unit.test_serrf_r_backend import _mock_source

    _mock_source(monkeypatch)
    source = _dataset(missing=True)

    def run(frame, **kwargs):
        return (frame * 2).fillna(999.0), {
            "implementation": "r", "parameters": {},
        }

    monkeypatch.setattr(serrf_r, "run_r_function", run)
    monkeypatch.setattr(serrf_r_validation, "run_r_function", run)
    result = SignalCorrector(
        source, base_est="SERRF", implementation="r",
        serrf_r_source="app.R", serrf_corr_features=4, cv_folds=3,
    ).run_signal_correction()
    corrected = list(result.data.values())[-1]
    assert np.isnan(corrected.intensity.loc["F0", "S0"])
    selection = result.audit.metrics["selection"]
    provenance = selection["implementation_provenance"][0]
    assert provenance["restored_missing_count"] == 1
    assert selection["validation"]["score_sample_observed_coverage"] == 1.0
    assert selection["validation"]["score_qc_observed_coverage"] == 1.0
    domain = selection["numeric_domain"]["SERRF corrected"]
    assert domain["full"]["new_missing_count"] == 0
    assert domain["oof"]["new_missing_count"] == 0

    imputer = MissingValueImputer(corrected, mar_method="Median")
    monkeypatch.setattr(
        imputer, "_evaluate_imputation_candidate",
        lambda *args, **kwargs: ({}, np.array([]), np.array([])),
    )
    imputed = imputer.transform_imputation()
    assert not imputed.audit.skipped
    assert imputed.data.intensity.loc["F0", "S0"] > 0
    observed = corrected.intensity.notna().to_numpy()
    np.testing.assert_array_equal(
        imputed.data.intensity.to_numpy()[observed],
        corrected.intensity.to_numpy()[observed],
    )


def test_explicit_all_invalid_output_is_audited_not_stopped(monkeypatch):
    """Expose invalid explicit-method output in the audit without inventing
    data.
    """
    from pimqc.processing.correction import analysis

    class Provider:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, intensity_df, **kwargs):
            return {"Invalid": (intensity_df * 0 - 1, None)}

    monkeypatch.setattr(analysis, "RegressionCorrector", Provider)
    result = SignalCorrector(
        _dataset(), base_est="QC-SVR"
    ).run_signal_correction()
    assert result.data["Invalid"].intensity.isna().all().all()
    validation = result.audit.metrics["selection"]["validation"]
    assert validation["eligible_for_auto"] is False


def test_candidate_missing_metric_never_renormalizes_weights():
    """Missing a candidate metric yields zero, not a better partial score."""
    score = SignalCorrector._fixed_metric_score
    assert score((1, 1, np.nan), (True, True, True)) == pytest.approx(0.7)
    assert score((1, 1, 0.5), (True, True, True)) == pytest.approx(0.85)
    # A genuinely absent input structure metric is excluded for everyone.
    assert score((1, 1, np.nan), (True, True, False)) == pytest.approx(1)


def test_correction_auto_score_combines_precision_d_ratio_and_structure():
    """AUTO uses three fixed-support preservation components."""
    technical = SignalCorrector._technical_precision_score(
        1.0, 0.5, (True, True)
    )
    assert technical == pytest.approx(0.75)
    assert SignalCorrector._technical_precision_score(
        -0.2, 0.5, (True, True)
    ) == pytest.approx(0.25)
    assert SignalCorrector._correction_auto_score(
        technical, 0.25, 0.5, (True, True, True)
    ) == pytest.approx(0.5)
    # An input-unsupported component is excluded, while a failed candidate
    # component remains a zero contribution under the planned denominator.
    assert SignalCorrector._correction_auto_score(
        technical, np.nan, 0.5, (True, False, True)
    ) == pytest.approx((0.35 * 0.75 + 0.30 * 0.5) / 0.65)


def test_auto_excludes_candidates_without_usable_evidence():
    """Require usable validation evidence before a candidate can win
    selection.
    """
    engine = SignalCorrector(_dataset())
    candidate = {
        "auto_score": 1.0,
        "eval_rsd": 0.0,
        "validation": {"eligible_for_auto": False},
    }
    with pytest.raises(ValueError, match="No correction AUTO candidate"):
        engine._select_best_correction_method({"QC-SVR": candidate})
    valid = {
        "auto_score": 0.0,
        "eval_rsd": 1.0,
        "validation": {"eligible_for_auto": True},
    }
    assert engine._select_best_correction_method({
        "QC-SVR": candidate, "SERRF": valid,
    }) == "SERRF"


def test_feature_score_retains_input_support_after_candidate_loss():
    """Keep the input feature denominator when a candidate loses QC coverage."""
    source = _dataset().annotated_frame()
    target = source.copy()
    qc = target.columns.get_level_values("Sample Type") == "QC"
    target.loc[:, qc] = np.repeat(
        source.loc[:, qc].mean(axis=1).to_numpy()[:, None], qc.sum(), axis=1
    )
    complete = SignalCorrector.calculate_featurewise_qc_rsd_improvement(
        source, target
    )
    target.iloc[0, 0] = np.nan
    partial = SignalCorrector.calculate_featurewise_qc_rsd_improvement(
        source, target
    )
    assert complete["support_count"] == partial["support_count"] == 20
    assert complete["evaluated_count"] == 20
    assert partial["evaluated_count"] == 19
    assert partial["score"] == pytest.approx(complete["score"] * 19 / 20)


def test_canonical_d_ratio_uses_fixed_positive_support():
    """D-ratio compares QC SD with combined QC/sample dispersion."""
    source = _dataset().annotated_frame()
    target = source.copy()
    qc = target.columns.get_level_values("Sample Type") == "QC"
    target.loc[:, qc] = np.repeat(
        source.loc[:, qc].mean(axis=1).to_numpy()[:, None], qc.sum(), axis=1
    )
    diagnostics = SignalCorrector.calculate_featurewise_d_ratio(
        source, target
    )
    assert diagnostics["support_count"] == 20
    assert diagnostics["evaluated_count"] == 20
    assert diagnostics["current_median"] <= diagnostics["baseline_median"]
    assert 0.0 <= diagnostics["score"] <= 1.0

    target.iloc[0, 0] = np.nan
    partial = SignalCorrector.calculate_featurewise_d_ratio(source, target)
    assert partial["support_count"] == diagnostics["support_count"]
    assert partial["evaluated_count"] == 19


def test_regression_prediction_does_not_hide_nonpositive_values():
    """Preserve nonpositive predictions for downstream domain validation."""
    def negative_model(x_train, y_train, x_test):
        return np.full(len(x_test), -2.0)

    x = np.arange(6, dtype=float).reshape(-1, 1)
    full, oof = fit_predict_intra_batch_safely(
        negative_model, x, np.ones(6), x, cv_folds=3
    )
    assert (full == -2).all()
    assert (oof == -2).all()


def test_r_domain_deferral_keeps_shape_and_observation_contracts():
    """Defer numeric-domain decisions without weakening structural
    validation.
    """
    original = np.ones((2, 2))
    values = np.array([[np.inf, np.nan], [-1, 0]])
    options = {"preserve_observed": False, "allow_missing": False}
    with pytest.raises(RBackendError, match="nonfinite"):
        _validate_output(original, values, **options)
    _validate_output(original, values, defer_output_domain=True, **options)
    with pytest.raises(RBackendError, match="shape"):
        _validate_output(
            original, values[:1], defer_output_domain=True, **options
        )
    with pytest.raises(RBackendError, match="changed observations"):
        _validate_output(
            original, values, defer_output_domain=True,
            preserve_observed=True, allow_missing=False,
        )
