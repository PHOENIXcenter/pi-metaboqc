"""Guard signed normalization and fair missing-metric AUTO evaluation."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.normalization import DataNormalizer
from pimqc.statistics.metrics import fixed_weighted_mean_score


def test_signed_glog_is_valid_but_new_missing_output_is_not():
    """Accept signed VSN values while detecting lost observed-value coverage."""
    raw = pd.DataFrame([[10, 20, 30]])
    metadata = {"value_scale": "vsn_glog"}
    result = pd.DataFrame([[-2.0, 0, 1]])
    report = DataNormalizer._validate_candidate_domain(result, raw, metadata)
    assert report["negative_count"] == 1
    assert report["zero_count"] == 1
    assert report["observed_coverage"] == 1
    with pytest.raises(ValueError, match="nonpositive"):
        DataNormalizer._validate_candidate_domain(
            result, raw, {"value_scale": "raw_positive"}
        )
    result.iloc[0, 1] = np.nan
    with pytest.raises(ValueError, match="lost observed"):
        DataNormalizer._validate_candidate_domain(result, raw, metadata)
    assert metadata["numerical_domain"]["new_missing_count"] == 1


def test_existing_missing_values_are_not_presented_as_fitted_zeros():
    """Restore pre-existing missingness after a normalization provider call."""
    raw = pd.DataFrame([[10, np.nan, 30]])
    result = pd.DataFrame([[2, 0, 4]], dtype=float)
    DataNormalizer._validate_candidate_domain(
        result, raw, {"value_scale": "log2"}
    )
    assert np.isnan(result.iloc[0, 1])


def test_linear_failure_cannot_be_hidden_by_robust_log_floor(monkeypatch):
    """Expose invalid linear normalization before applying a log display
    floor.
    """
    from tests.unit.test_normalization_r_backend import _dataset

    def invalid_tic(df):
        output = df.copy()
        output.iloc[0, 0] = -1
        return output

    monkeypatch.setattr(
        DataNormalizer, "calc_tic_normalization", staticmethod(invalid_tic)
    )
    with pytest.raises(ValueError, match="nonpositive"):
        DataNormalizer(_dataset(), norm_method="TIC").run_normalization()


def test_missing_score_component_does_not_increase_remaining_weights():
    """Keep planned weights fixed when a normalization score is unavailable."""
    weights = DataNormalizer._AUTO_SCORE_COMPONENT_WEIGHTS
    baseline = {"method": "ROBUST_LOG_ONLY", "status": "ok"}
    baseline.update({key: 0.0 for key in weights})
    complete = {"method": "TIC", "status": "ok"}
    complete.update({key: 1.0 for key in weights})
    incomplete = {**complete, "method": "PQN"}
    incomplete["qc_structure_change_score"] = np.nan
    scored = DataNormalizer._score_normalization_candidates(
        [baseline, complete, incomplete]
    ).set_index("method")
    assert scored.loc["TIC", "auto_score"] > scored.loc["PQN", "auto_score"]
    assert scored.loc["PQN", "planned_metric_weight"] == 1
    assert scored.loc["PQN", "available_metric_weight"] == pytest.approx(0.8)


def test_nested_auto_component_keeps_planned_denominator():
    """A missing submetric contributes zero instead of reweighting siblings."""
    assert fixed_weighted_mean_score([(1.0, 1.0), (np.nan, 1.0)]) == 0.5
    assert np.isnan(fixed_weighted_mean_score([(np.nan, 1.0)]))


def test_lost_signed_submetric_cannot_become_neutral_improvement():
    """Do not convert a lost baseline-supported score into neutral
    improvement.
    """
    score = DataNormalizer._supported_change_score
    assert np.isnan(score([(-1, 1), (np.nan, 1)], [1, 1]))
    assert score([(-1, 1), (np.nan, 1)], [1, np.nan]) == -1


def test_auto_excludes_nonconverged_vsn_but_explicit_marks_degraded(
    monkeypatch,
):
    """Exclude unconverged VSN from selection and audit explicit degraded
    use.
    """
    from tests.unit.test_normalization_r_backend import _dataset

    def fake_vsn(df, **kwargs):
        return np.log2(df), {"vsn_converged": False}

    monkeypatch.setattr(
        DataNormalizer, "calc_vsn_normalization", staticmethod(fake_vsn)
    )
    monkeypatch.setattr(
        DataNormalizer, "_AUTO_CANDIDATES", ("ROBUST_LOG_ONLY", "VSN")
    )
    data = _dataset()
    auto = DataNormalizer(data, norm_method="Auto").run_normalization()
    candidate = next(
        row for row in auto.audit.candidate_results if row["method"] == "VSN"
    )
    assert candidate["status"] == "failed"
    assert "converge" in candidate["error"]
    assert auto.audit.selection["selected_method"] == "ROBUST_LOG_ONLY"
    explicit = DataNormalizer(data, norm_method="VSN").run_normalization()
    assert explicit.audit.selection["status"] == "degraded"
    assert explicit.data.context.extra_attrs["value_scale"] == "vsn_glog"
