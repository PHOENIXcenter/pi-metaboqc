"""Guard R held-out QC routing, provenance and unsupported-fold evidence."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from pimqc.processing.correction import serrf_r, serrf_r_validation
from pimqc.processing.correction.qc_validation import assign_qc_folds
from pimqc.processing.r_backend import RBackendError
from tests.unit.test_serrf_r_backend import _inputs, _mock_source


def test_native_and_r_share_original_within_batch_assignments():
    """QC fold identities are comparable without relying on R's RNG."""
    _, batches, qc, _, _ = _inputs()
    expected = np.full(len(qc), -1, dtype=int)
    for batch in pd.unique(batches):
        indices = np.flatnonzero((batches == batch) & qc)
        splitter = KFold(n_splits=3, shuffle=True, random_state=37)
        for fold, (_, held_out) in enumerate(splitter.split(indices)):
            expected[indices[held_out]] = fold
    np.testing.assert_array_equal(
        assign_qc_folds(batches, qc, 3, 37), expected
    )


def test_r_oof_has_separate_qc_roles_and_keeps_non_qc_full_fit(monkeypatch):
    """Leave-out QCs never become Samples; every QC is queried once."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    seen = []

    def full(frame, **kwargs):
        assert kwargs["function"] == "Shiny-SERRF::serrfR (source adapter)"
        assert kwargs["parameters"]["effective_model_seed"] == 37
        return frame * 2, {"package": "ranger"}

    def validation(frame, **kwargs):
        params = kwargs["parameters"]
        assert params["original_model_seed"] == 1
        assert params["effective_model_seed"] == kwargs["seed"] == 37
        assert "params$effective_model_seed)" in kwargs["code"]
        train = np.array(params["training_qc"])
        test = np.array(params["held_out_qc"])
        roles = np.array(params["qc"])
        np.testing.assert_array_equal(roles, qc[~blank])
        np.testing.assert_array_equal(train | test, roles)
        assert not (train & test).any()
        assert not test[~roles].any()
        seen.append(test)
        output = frame * np.nan
        output.loc[:, test] = frame.loc[:, test] * 3
        return output, {"function": kwargs["function"]}

    monkeypatch.setattr(serrf_r, "run_r_function", full)
    monkeypatch.setattr(serrf_r_validation, "run_r_function", validation)
    engine = serrf_r.SERRFRCorrector(
        source_path="app.R", random_state=37,
        n_correlated_features=4, cv_folds=3,
    )
    full_output, oof = engine.fit_transform(
        data, batches, qc, order, blank
    )["SERRF corrected"]
    np.testing.assert_array_equal(np.sum(seen, axis=0), qc[~blank].astype(int))
    np.testing.assert_allclose(oof.loc[:, qc], data.loc[:, qc] * 3)
    pd.testing.assert_frame_equal(oof.loc[:, ~qc], full_output.loc[:, ~qc])
    np.testing.assert_array_equal(
        engine.diagnostics["fold_assignments"],
        assign_qc_folds(batches, qc, 3, 37),
    )
    assert engine.diagnostics["qc_oof_value_coverage"] == 1.0
    assert "not upstream API" in engine.provenance["evaluation_basis"]
    assert len(engine.provenance["validation"]["fold_provenance"]) == 3
    assert engine.diagnostics["timing_seconds"]["oof"] >= 0
    assert engine.diagnostics["runtime_seed_adaptation"]["effective_model_seed"] == 37


def test_failed_r_fold_is_missing_not_replaced_by_full_fit(monkeypatch):
    """A fold failure is not hidden by fitted or uncorrected QC values."""
    data, batches, qc, _, blank = _inputs()
    active = ~blank
    calls = []

    def run(frame, **kwargs):
        fold = kwargs["parameters"]["validation_fold"]
        calls.append(fold)
        if fold == 0:
            raise RBackendError("deliberate fold failure")
        return frame * 3, {}

    monkeypatch.setattr(serrf_r_validation, "run_r_function", run)
    oof, diagnostics, _ = serrf_r_validation.validate_serrf_qcs(
        data.loc[:, active], data.loc[:, active] * 2,
        source="verified source", extractor="verified extractor",
        parameters={"n_correlated_features": 4},
        batches=batches[active], qc=qc[active],
        cv_folds=3, random_state=37,
    )
    folds = np.array(diagnostics["fold_assignments"])
    assert oof.loc[:, folds == 0].isna().all().all()
    assert oof.loc[:, folds > 0].notna().all().all()
    assert calls == [0, 1, 2]
    assert diagnostics["status"] == "degraded"
    assert diagnostics["folds"][0]["status"] == "failed"
    assert "deliberate" in diagnostics["folds"][0]["error"]


def test_missing_held_out_response_is_not_repaired_as_validation_truth(
    monkeypatch,
):
    """A missing response remains unsupported even if a backend returns it."""
    data, batches, qc, _, blank = _inputs()
    data.iloc[0, 0] = np.nan
    data.iloc[0, 1] = np.nan
    active = ~blank

    def run(frame, **kwargs):
        assert kwargs["allow_all_missing_input"] is True
        return frame.fillna(999.0), {}

    monkeypatch.setattr(serrf_r_validation, "run_r_function", run)
    oof, diagnostics, _ = serrf_r_validation.validate_serrf_qcs(
        data.loc[:, active], data.loc[:, active].fillna(999.0),
        source="verified source", extractor="verified extractor",
        parameters={"n_correlated_features": 4},
        batches=batches[active], qc=qc[active],
        cv_folds=3, random_state=37,
    )
    assert np.isnan(oof.iloc[0, 0])
    assert np.isnan(oof.iloc[0, 1])
    assert diagnostics["qc_oof_value_coverage"] < 1.0
    assert diagnostics["qc_oof_eligible_value_coverage"] == 1.0
    assert diagnostics["status"] == "ok"
    assert all(row["status"] == "completed" for row in diagnostics["folds"])


def test_public_full_and_oof_restore_original_na(monkeypatch):
    """Internal R fills never consume the next stage's missing positions."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    data.iloc[0, 0] = np.nan  # QC
    data.iloc[1, 1] = np.nan  # Sample
    data.iloc[2, -1] = np.nan  # Blank
    before = data.copy(deep=True)

    def run(frame, **kwargs):
        return (frame * 2).fillna(999.0), {"package": "ranger"}

    monkeypatch.setattr(serrf_r, "run_r_function", run)
    monkeypatch.setattr(serrf_r_validation, "run_r_function", run)
    engine = serrf_r.SERRFRCorrector(
        source_path="app.R", random_state=37,
        n_correlated_features=4, cv_folds=3,
    )
    full, oof = engine.fit_transform(
        data, batches, qc, order, blank
    )["SERRF corrected"]
    expected = data * 2
    expected.loc[:, blank] = data.loc[:, blank]
    pd.testing.assert_frame_equal(full, expected)
    pd.testing.assert_frame_equal(oof, expected)
    pd.testing.assert_frame_equal(data, before)
    assert engine.provenance["restored_missing_count"] == 2
    assert engine.diagnostics["restored_missing_count"] == 2
    assert oof.attrs["serrf_diagnostics"] == engine.diagnostics


def test_r_fold_feature_support_cannot_use_held_out_qc(monkeypatch):
    """A held-out observation cannot rescue an empty fitting batch."""
    data, batches, qc, _, blank = _inputs()
    active = ~blank
    data = data.loc[:, active].copy()
    batches, qc = batches[active], qc[active]
    folds = assign_qc_folds(batches, qc, 3, 37)
    missing = (batches == "B1") & (folds != 0)
    data.loc[0, missing] = np.nan
    called = []

    def run(frame, **kwargs):
        fold = kwargs["parameters"]["validation_fold"]
        called.append(fold)
        assert (0 in frame.index) == (fold != 0)
        # IDs, not positions, must survive fold-specific feature exclusion.
        result = frame.copy()
        for name in result.index:
            result.loc[name] = float(name + 100)
        return result, {}

    monkeypatch.setattr(serrf_r_validation, "run_r_function", run)
    oof, diagnostics, provenance = serrf_r_validation.validate_serrf_qcs(
        data, data.copy(), source="source", extractor="extractor",
        parameters={"n_correlated_features": 4}, batches=batches, qc=qc,
        cv_folds=3, random_state=37,
    )
    assert called == [0, 1, 2]
    assert oof.loc[0, folds == 0].isna().all()
    assert (oof.loc[1, qc] == 101.0).all()
    assert diagnostics["folds"][0]["unsupported_feature_ids"] == ["0"]
    assert provenance[0]["input_support"]["unsupported_feature_ids"] == ["0"]


@pytest.mark.parametrize("folds", [-1, 1, 2.1, True])
def test_invalid_validation_fold_counts_are_rejected(folds):
    """Reject invalid R SERRF validation-fold settings at construction."""
    with pytest.raises(ValueError, match="cv_folds"):
        serrf_r.SERRFRCorrector(
            source_path="app.R", random_state=1, cv_folds=folds
        )
