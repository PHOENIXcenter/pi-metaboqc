"""Regression guards for native SERRF's batch and validation boundaries."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from pimqc.processing.correction.serrf import SERRFCorrector


def _data():
    rng = np.random.default_rng(91)
    batches = np.repeat(["B1", "B2"], 30)
    orders = np.arange(60, dtype=float)
    qc = np.tile(np.arange(30) % 3 == 0, 2)
    drift = 1 + 0.3 * np.sin(orders / 9)
    scale = np.where(batches == "B2", 1.5, 1)
    values = rng.uniform(100, 500, (8, 1)) * drift * scale
    values *= rng.normal(1, 0.02, values.shape)
    values[:, ~qc] *= rng.lognormal(0.1, 0.06, (8, (~qc).sum()))
    return pd.DataFrame(values), batches, qc, orders


def _engine(**kwargs):
    return SERRFCorrector(
        n_estimators=25,
        n_corr_features=3,
        cv_folds=3,
        random_state=7,
        n_jobs=1,
        **kwargs,
    )


def test_held_out_response_does_not_change_its_correction_factor():
    """Selection and all scales must be fitted without held-out response y."""
    data, batches, qc, orders = _data()
    held_out = []
    for batch in np.unique(batches):
        indices = np.flatnonzero(qc & (batches == batch))
        splitter = KFold(n_splits=3, shuffle=True, random_state=7)
        _, test = next(splitter.split(indices))
        held_out.extend(indices[test])
    changed = data.copy()
    changed.iloc[0, held_out] *= np.linspace(0.01, 100, len(held_out))
    before = _engine().fit_transform(data, batches, qc, orders)["SERRF"]
    after = _engine().fit_transform(changed, batches, qc, orders)["SERRF"]
    np.testing.assert_allclose(
        before[1].iloc[0, held_out] / data.iloc[0, held_out],
        after[1].iloc[0, held_out] / changed.iloc[0, held_out],
        rtol=1e-12,
    )
    assert not np.allclose(
        before[0].iloc[0, held_out] / data.iloc[0, held_out],
        after[0].iloc[0, held_out] / changed.iloc[0, held_out],
    )


def test_batch_role_medians_align_and_input_is_not_mutated():
    """Align batch medians within sample roles without changing the input
    matrix.
    """
    data, batches, qc, orders = _data()
    original = data.copy(deep=True)
    engine = _engine()
    full, oof = engine.fit_transform(data, batches, qc, orders)["SERRF"]
    pd.testing.assert_frame_equal(data, original)
    for role in [qc, ~qc]:
        np.testing.assert_allclose(
            full.loc[:, role & (batches == "B1")].median(axis=1),
            full.loc[:, role & (batches == "B2")].median(axis=1),
        )
    assert np.isfinite(oof.to_numpy()).all()
    assert engine.diagnostics["effective_folds_by_batch"] == {
        "B1": 3,
        "B2": 3,
    }
    assert "transductive" in engine.diagnostics["validation"]
    assert engine.diagnostics["fit_counts"]["full"] == {
        "forest_feature_batch_fits": 16,
        "location_only_feature_batch_fits": 0,
        "unsupported_feature_batch_fits": 0,
        "invalid_baseline_predictions": 0,
        "invalid_qc_baseline_predictions": 0,
        "invalid_sample_baseline_predictions": 0,
    }
    full_rsd = full.loc[:, qc].std(axis=1) / full.loc[:, qc].mean(axis=1)
    oof_rsd = oof.loc[:, qc].std(axis=1) / oof.loc[:, qc].mean(axis=1)
    assert oof_rsd.median() > full_rsd.median()


def test_missing_responses_and_unsupported_batches_have_no_invented_oof():
    """Leave unsupported responses missing in held-out QC output."""
    data, batches, qc, orders = _data()
    first_qc = np.flatnonzero(qc)[0]
    data.iloc[0, first_qc] = np.nan
    data.iloc[1, first_qc] = 0
    data.iloc[2, first_qc] = -1
    data.iloc[3, qc & (batches == "B1")] = np.nan
    full, oof = _engine().fit_transform(data, batches, qc, orders)["SERRF"]
    assert np.isnan(full.iloc[0, first_qc])
    assert np.isnan(full.iloc[1, first_qc])
    assert np.isnan(full.iloc[2, first_qc])
    assert oof.iloc[:3, first_qc].isna().all()
    assert oof.loc[3, qc & (batches == "B1")].isna().all()
    assert np.isfinite(oof.loc[3, qc & (batches == "B2")]).all()


def test_predictor_selection_excludes_self_even_for_oversized_request():
    """Exclude the response feature when requesting more predictors than
    exist.
    """
    engine = SERRFCorrector(n_corr_features=100, n_jobs=1)
    corr = np.array([[1, 0.9, 0.4], [0.9, 1, -0.7], [0.4, -0.7, 1]])
    selected = engine._select_predictors(0, corr, corr)
    np.testing.assert_array_equal(selected, [1, 2])


def test_invalid_baselines_become_missing_and_are_reported(monkeypatch):
    """Invalid corrections cannot masquerade as unchanged observations."""
    from pimqc.processing.correction import serrf

    class BadForest:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.repeat(-1e9, len(x))

    monkeypatch.setattr(serrf, "RandomForestRegressor", BadForest)
    data, batches, qc, orders = _data()
    engine = _engine()
    full, oof = engine.fit_transform(data, batches, qc, orders)["SERRF"]
    assert full.loc[:, qc].isna().all().all()
    assert oof.loc[:, qc].isna().all().all()
    assert engine.diagnostics["qc_oof_value_coverage"] == 0
    assert engine.diagnostics["status"] == "degraded"
    assert (
        engine.diagnostics["fit_counts"]["full"]["invalid_baseline_predictions"]
        == data.loc[:, qc].size
    )


def test_failed_sample_baselines_are_visible_and_precede_qc_mapping(
    monkeypatch,
):
    """Expose invalid study-sample baselines before QC reference mapping."""
    from pimqc.processing.correction import serrf

    class BadSampleForest:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            if len(x) == 20:
                return np.tile([-1e9, 1e9], 10)
            return np.zeros(len(x))

    monkeypatch.setattr(serrf, "RandomForestRegressor", BadSampleForest)
    data, batches, qc, orders = _data()
    engine = _engine()
    full, _ = engine.fit_transform(data, batches, qc, orders)["SERRF"]
    assert engine.diagnostics["status"] == "degraded"
    cells = engine.diagnostics["invalid_sample_baseline_cells"]
    assert len(cells) == 160
    for cell in cells:
        row, col = int(cell["feature"]), int(cell["sample"])
        assert np.isnan(full.loc[row, col])
    supported_raw = data.loc[:, ~qc].where(full.loc[:, ~qc].notna())
    expected_qc = full.loc[:, ~qc].median(axis=1) + (
        data.loc[:, qc].median(axis=1) - supported_raw.median(axis=1)
    ) / supported_raw.std(axis=1) * full.loc[:, ~qc].std(axis=1)
    expected_qc = expected_qc.where(
        expected_qc > 0, data.loc[:, qc].median(axis=1)
    )
    np.testing.assert_allclose(full.loc[:, qc].median(axis=1), expected_qc)


@pytest.mark.parametrize("mask", [[1] * 60, [True] * 59])
def test_invalid_qc_masks_fail_before_fitting(mask):
    """Reject nonboolean or misaligned QC masks before model fitting."""
    data, batches, _, orders = _data()
    with pytest.raises(ValueError, match="boolean|sample count"):
        _engine().fit_transform(data, batches, np.asarray(mask), orders)
