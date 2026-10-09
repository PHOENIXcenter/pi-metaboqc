"""Production output policies stay separate from raw imputation kernels."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.imputation import MissingValueImputer
from tests.unit.test_audit_regressions import _dataset


def _skip_evaluation(engine, monkeypatch):
    monkeypatch.setattr(
        engine,
        "_evaluate_imputation_candidate",
        lambda *args, **kwargs: ({}, np.array([]), np.array([])),
    )


def test_zero_input_is_imputed_and_observations_preserved(monkeypatch):
    """Zero is missing before the skipped-stage check, not after it."""
    data = _dataset()
    intensity = data.intensity.copy()
    intensity.loc["F0", "S0"] = 0
    engine = MissingValueImputer(
        data.with_intensity(intensity), mar_method="Median"
    )
    _skip_evaluation(engine, monkeypatch)
    result = engine.transform_imputation()
    assert not result.audit.skipped
    assert result.data.intensity.loc["F0", "S0"] > 0
    observed = intensity.to_numpy() > 0
    np.testing.assert_array_equal(
        result.data.intensity.to_numpy()[observed],
        intensity.to_numpy()[observed],
    )
    assert result.data.context.extra_attrs["value_scale"] == "raw_positive"


def test_evaluation_floor_uses_only_unmasked_same_role(monkeypatch):
    """An artificially hidden tiny value cannot define its own substitute."""
    engine = MissingValueImputer(_dataset(), mar_method="BPCA")
    training = np.log1p(engine.frame) / np.log(2)
    training.loc["F0", engine.actual_data.columns[0]] = np.log2(1.01)
    mask = pd.DataFrame(False, index=training.index, columns=training.columns)
    mask.loc["F0", engine.actual_data.columns[0]] = True
    engine._evaluation_mask = mask
    monkeypatch.setattr(
        engine, "_run_mar_method",
        lambda frame, method, seed: frame.fillna(-3.0),
    )
    metrics, _, predicted = engine._evaluate_imputation_candidate(
        training, training.index, training.columns, "BPCA"
    )
    observed = engine.frame.loc["F0", engine.actual_data.columns[1:]]
    expected = np.log1p(observed.min() / 2) / np.log(2)
    assert predicted[0] == pytest.approx(expected)
    assert metrics["numerical_domain"]["repaired_count"] == 1


def test_correction_gaps_do_not_relabel_or_use_mnar_fills(monkeypatch):
    """Two mechanisms can route separately within one feature's old label."""
    data = _dataset(missing=True, mechanism="MNAR")
    intensity = data.intensity.copy()
    intensity.loc["F0", "S1"] = np.nan
    features = data.feature_metadata.copy()
    features["correction_missing_samples"] = [()] * len(features)
    features.at["F0", "correction_missing_samples"] = ("S1",)
    data = data.with_intensity(intensity, feature_metadata=features)
    engine = MissingValueImputer(data, mar_method="Median")
    _skip_evaluation(engine, monkeypatch)
    monkeypatch.setattr(
        MissingValueImputer, "impute_by_qrilc",
        staticmethod(lambda *args, **kwargs: kwargs["df_log"].fillna(1.0)),
    )

    def reconstruct(frame, method, seed):
        # Fixed MNAR fills must never become MAR training evidence.
        assert frame.loc["F0", engine.actual_data.columns[0]] != 1.0
        assert pd.isna(frame.loc["F0", engine.actual_data.columns[0]])
        return frame.fillna(2.0)

    monkeypatch.setattr(engine, "_run_mar_method", reconstruct)
    result = engine.transform_imputation()
    assert result.data.intensity.loc["F0", "S0"] == pytest.approx(1)
    assert result.data.intensity.loc["F0", "S1"] == pytest.approx(3)
    assert result.data.feature_metadata["missingness_type"].eq("MNAR").all()
    assert result.audit.metrics["missingness_routing"][
        "correction_reconstruction_count"
    ] == 1


@pytest.mark.parametrize("implementation", ["python", "r"])
def test_qrilc_fits_all_original_features_and_writes_only_mnar(
    implementation, monkeypatch
):
    """Both providers get full observations and preserve MAR routing."""
    data = _dataset(missing=True)
    intensity = data.intensity.copy()
    intensity.loc["F0", "S1"] = np.nan
    intensity.loc["F1", "S2"] = np.nan
    features = data.feature_metadata.copy()
    features.loc["F0", "missingness_type"] = "MNAR"
    features["correction_missing_samples"] = [()] * len(features)
    features.at["F0", "correction_missing_samples"] = ("S1",)
    data = data.with_intensity(intensity, feature_metadata=features)
    engine = MissingValueImputer(
        data, mar_method="BPCA", implementation=implementation
    )
    _skip_evaluation(engine, monkeypatch)
    expected = np.log1p(engine.frame) / np.log(2)
    calls = []

    def qrilc(frame, **kwargs):
        pd.testing.assert_frame_equal(frame, expected)
        calls.append(frame.copy())
        # Propose fills for every gap: only true MNAR gaps may commit.
        return frame.fillna(1.0)

    if implementation == "r":
        monkeypatch.setattr(engine, "_impute_by_r", qrilc)
    else:
        monkeypatch.setattr(
            MissingValueImputer, "impute_by_qrilc",
            staticmethod(lambda df_log, **kwargs: qrilc(df_log, **kwargs)),
        )

    def reconstruct(frame, method, seed):
        pd.testing.assert_frame_equal(frame, expected.loc[frame.index])
        return frame.fillna(2.0)

    monkeypatch.setattr(engine, "_run_mar_method", reconstruct)
    result = engine.transform_imputation()
    assert len(calls) == 1
    assert result.data.intensity.loc["F0", "S0"] == pytest.approx(1.0)
    assert result.data.intensity.loc["F0", "S1"] == pytest.approx(3.0)
    assert result.data.intensity.loc["F1", "S2"] == pytest.approx(3.0)
    observed = intensity.notna().to_numpy()
    np.testing.assert_array_equal(
        result.data.intensity.to_numpy()[observed],
        intensity.to_numpy()[observed],
    )


def test_auto_refit_failure_tries_next_eligible_method(monkeypatch):
    """A finite benchmark is not a guarantee that the final fit succeeds."""
    engine = MissingValueImputer(_dataset(missing=True))
    cache = {
        "BPCA": ({"Auto_Score": 0.9}, np.array([]), np.array([])),
        "Median": ({"Auto_Score": 0.8}, np.array([]), np.array([])),
    }

    def select(*args, **kwargs):
        engine._ranked_imputation_methods = ["BPCA", "Median"]
        return "BPCA", cache

    monkeypatch.setattr(engine, "_select_best_imputation_method", select)
    monkeypatch.setattr(
        engine, "_run_mar_method",
        lambda frame, method, seed: frame.fillna(
            np.inf if method == "BPCA" else 2.0
        ),
    )
    result = engine.transform_imputation()
    assert result.audit.selected_method == "Median"
    records = result.audit.metrics["selection"]["candidate_results"]
    assert records[0]["status"] == "final_fit_failed"
    assert records[1]["selected"]


def test_candidate_missing_metric_does_not_increase_other_weights():
    """Missing candidate metrics earn zero, not a larger remaining weight."""
    reference = {
        "NRMSE_Total": 1, "NRMSE_Low": 1, "JSD_Total": 1,
        "Wasserstein_Normalized": 1, "Sample_Structure_Score": 1,
    }
    metrics = {key: 0.5 for key in reference}
    metrics["Sample_Structure_Score"] = 1
    incomplete = {**metrics, "NRMSE_Low": np.nan}
    result = MissingValueImputer._score_imputation_candidates(
        {
            "complete": (metrics, [], []),
            "incomplete": (incomplete, [], []),
            "failed": ({"status": "failed"}, [], []),
        },
        reference,
    ).set_index("method")
    assert "failed" not in result.index
    assert result.loc["incomplete", "auto_score"] == pytest.approx(
        result.loc["complete", "auto_score"] - 0.65 * 0.3 * 0.5
    )


def test_auto_candidate_failure_does_not_abort_others(monkeypatch):
    """All candidate benchmarks share one mask and isolate model failures."""
    engine = MissingValueImputer(_dataset(missing=True))
    training = np.log1p(engine.frame) / np.log(2)
    mask_ids = []

    def evaluate(*args, **kwargs):
        mask_ids.append(id(engine._evaluation_mask))
        if kwargs["method"] == "BPCA":
            raise ValueError("synthetic nonfinite failure")
        return {
            "NRMSE_Total": 0.3, "NRMSE_Low": 0.3,
            "JSD_Total": 0.2, "Wasserstein_Normalized": 0.2,
            "Sample_Structure_Score": 0.9,
        }, np.array([]), np.array([])

    monkeypatch.setattr(engine, "_evaluate_imputation_candidate", evaluate)
    selected, cache = engine._select_best_imputation_method(
        training, training.index, training.columns
    )
    assert selected != "BPCA"
    assert cache["BPCA"][0]["status"] == "failed"
    assert len(set(mask_ids)) == 1


def test_minprob_kernel_preserves_signed_draws():
    """Implementation comparisons see model draws before stage repair."""
    frame = pd.DataFrame({"sample": [0.01, 0.02, 10, np.nan]})
    result = MissingValueImputer.impute_by_minprob(frame, global_seed=0)
    assert result.iloc[-1, 0] < 0


def test_runner_does_not_send_failed_arrays_to_dashboard(
    monkeypatch, tmp_path
):
    """A failed AUTO candidate stays in the audit, not an empty scatter."""
    from pimqc.processing.imputation.runner import ImputationStageRunner

    engine = MissingValueImputer(_dataset(missing=True))
    cache = {
        "Median": ({"Auto_Score": 0.8}, np.array([1.]), np.array([1.])),
        "BPCA": ({"status": "failed"}, np.array([]), np.array([])),
    }
    monkeypatch.setattr(
        engine, "_select_best_imputation_method",
        lambda *args, **kwargs: ("Median", cache),
    )
    result = engine.transform_imputation()
    rendered = []

    def plot(self, results_dict, **kwargs):
        rendered.append(set(results_dict))
        return None

    monkeypatch.setattr(
        "pimqc.processing.imputation.runner.ImputationPlotter."
        "plot_imputation_auto_dashboard", plot,
    )
    monkeypatch.setattr(
        "pimqc.processing.imputation.runner.ImputationPlotter."
        "plot_imputation_nrmse_appendix_dashboard", plot,
    )
    ImputationStageRunner(engine, tmp_path).render(result)
    assert rendered == [{"Median"}, {"Median"}]
    assert "BPCA" in result.audit.candidate_results
