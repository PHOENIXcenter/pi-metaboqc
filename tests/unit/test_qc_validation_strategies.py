"""Random default regression and optional blocked SERRF OOF contracts."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from pimqc.processing.correction import serrf_r, serrf_r_validation
from pimqc.processing.correction.qc_validation import assign_qc_folds
from pimqc.processing.correction.serrf import SERRFCorrector
from tests.unit.test_serrf_native import _data
from tests.unit.test_serrf_r_backend import _inputs, _mock_source


def test_random_default_exactly_matches_original_kfold():
    """Preserve the existing seeded, batchwise random QC fold assignment."""
    _, batches, qc, orders = _data()
    expected = np.full(len(qc), -1, dtype=int)
    for batch in pd.unique(batches):
        indices = np.flatnonzero((batches == batch) & qc)
        split = KFold(n_splits=3, shuffle=True, random_state=13)
        for fold, (_, held_out) in enumerate(split.split(indices)):
            expected[indices[held_out]] = fold
    np.testing.assert_array_equal(
        assign_qc_folds(batches, qc, 3, 13), expected
    )
    np.testing.assert_array_equal(
        assign_qc_folds(
            batches, qc, 3, 13, strategy="random", order_array=orders[::-1]
        ), expected,
    )


def test_blocked_is_chronological_and_leaves_non_qc_unassigned():
    """Assign chronological QC blocks without allocating study samples to
    folds.
    """
    _, batches, qc, orders = _data()
    orders = orders[::-1]
    folds = assign_qc_folds(
        batches, qc, 3, 13, strategy="blocked", order_array=orders
    )
    assert (folds[~qc] == -1).all()
    for batch in pd.unique(batches):
        indices = np.flatnonzero((batches == batch) & qc)
        indices = indices[np.argsort(orders[indices])]
        assert (np.diff(folds[indices]) >= 0).all()
        assert set(folds[indices]) == {0, 1, 2}
    np.testing.assert_array_equal(
        folds,
        assign_qc_folds(
            batches, qc, 3, 99, strategy="blocked", order_array=orders
        ),
    )


def test_blocked_requires_aligned_finite_orders():
    """Require aligned, finite injection orders for blocked QC validation."""
    _, batches, qc, orders = _data()
    with pytest.raises(ValueError, match="aligned orders"):
        assign_qc_folds(batches, qc, 3, 0, strategy="blocked")
    orders[np.flatnonzero(qc)[0]] = np.nan
    with pytest.raises(ValueError, match="finite QC orders"):
        assign_qc_folds(
            batches, qc, 3, 0, strategy="blocked", order_array=orders
        )
    with pytest.raises(ValueError, match="strategy"):
        assign_qc_folds(batches, qc, 3, 0, strategy="unknown")


def test_native_blocked_oof_really_refits_without_held_out_responses():
    """Ensure held-out responses cannot alter their fitted correction
    factors.
    """
    data, batches, qc, orders = _data()
    config = dict(
        n_estimators=5, cv_folds=3, n_corr_features=2,
        random_state=13, n_jobs=1, cv_strategy="blocked",
    )
    engine = SERRFCorrector(**config)
    before = engine.fit_transform(data, batches, qc, orders)["SERRF"]
    folds = np.array(engine.diagnostics["fold_assignments"])
    changed = data.copy()
    changed.loc[0, folds == 0] *= np.linspace(0.1, 10, (folds == 0).sum())
    after = SERRFCorrector(**config).fit_transform(
        changed, batches, qc, orders
    )["SERRF"]
    np.testing.assert_allclose(
        before[1].loc[0, folds == 0] / data.loc[0, folds == 0],
        after[1].loc[0, folds == 0] / changed.loc[0, folds == 0],
        rtol=1e-12,
    )
    assert engine.diagnostics["cv_strategy"] == "blocked"
    random_full = SERRFCorrector(
        **{**config, "cv_strategy": "random"}
    ).fit_transform(data, batches, qc, orders)["SERRF"][0]
    np.testing.assert_allclose(before[0], random_full, rtol=0, atol=0)


def test_r_blocked_plumbing_uses_order_and_separate_predictions(monkeypatch):
    """Pass chronological folds and distinct held-out predictions through R
    routing.
    """
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    order = order[::-1]
    seen = []

    def full(frame, **kwargs):
        return frame * 2, {"package": "ranger"}

    def validation(frame, **kwargs):
        params = kwargs["parameters"]
        np.testing.assert_array_equal(params["order"], order[~blank])
        held_out = np.asarray(params["held_out_qc"])
        training = np.asarray(params["training_qc"])
        assert not (held_out & training).any()
        seen.append(held_out)
        output = frame * np.nan
        output.loc[:, held_out] = frame.loc[:, held_out] * 3
        return output, {}

    monkeypatch.setattr(serrf_r, "run_r_function", full)
    monkeypatch.setattr(serrf_r_validation, "run_r_function", validation)
    engine = serrf_r.SERRFRCorrector(
        source_path="app.R", random_state=13,
        n_correlated_features=4, cv_folds=3, cv_strategy="blocked",
    )
    full_output, oof = engine.fit_transform(
        data, batches, qc, order, blank
    )["SERRF corrected"]
    np.testing.assert_allclose(oof.loc[:, qc], data.loc[:, qc] * 3)
    np.testing.assert_allclose(full_output.loc[:, qc], data.loc[:, qc] * 2)
    np.testing.assert_array_equal(
        engine.diagnostics["fold_assignments"],
        assign_qc_folds(
            batches, qc, 3, 13, strategy="blocked", order_array=order
        ),
    )
    assert engine.provenance["validation"]["cv_strategy"] == "blocked"
    assert len(seen) == 3
