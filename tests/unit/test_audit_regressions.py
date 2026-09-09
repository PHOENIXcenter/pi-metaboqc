"""Regression coverage for release audits of scientific and state contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pimqc import MetaboDatasetBuilder
from pimqc.core import MetaboDataset
from pimqc.plotting.payloads import snapshot_plot_value
from pimqc.processing.assessment import QualityAssessor
from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.algorithms import (
    fit_predict_intra_batch_safely,
)
from pimqc.processing.filtering import (
    FeatureMissingValueFilter,
    SampleMissingValueFilter,
)
from pimqc.processing.imputation import MissingValueImputer
from pimqc.processing.normalization import DataNormalizer
from pimqc.reporting import ReportInput


def _dataset(missing=False, mechanism="MAR"):
    """Build data through the public builder, preserving real column order."""
    rng = np.random.default_rng(23)
    samples = [f"Q{i}" for i in range(3)] + [f"S{i}" for i in range(6)]
    metadata = pd.DataFrame(
        {
            "Sample Name": samples,
            "Sample Type": ["QC"] * 3 + ["Sample"] * 6,
            "Batch": ["B1"] * 9,
            "Inject Order": range(1, 10),
        }
    )
    intensity = pd.DataFrame(
        rng.lognormal(5, 0.15, (20, 9)),
        index=[f"F{i}" for i in range(20)],
        columns=samples,
    )
    if missing:
        intensity.loc["F0", "S0"] = np.nan
    data = MetaboDatasetBuilder(metadata, intensity).run_build().data
    features = data.feature_metadata.copy()
    features["missingness_type"] = mechanism
    return MetaboDataset.from_tables(
        data.intensity,
        data.sample_metadata,
        features,
        schema=data.schema,
        context=data.context,
    )


def test_sample_ids_and_threshold_follow_upstream_audit():
    """Track actual identities and the executed sample tolerance."""
    data = _dataset(missing=True)
    sample = SampleMissingValueFilter(
        data, sample_mv_tol=0.23
    ).run_filter_samples_by_missingness()
    assert set(sample.audit.sample_tracking.index) == set(
        data.sample_metadata.index
    )
    feature = FeatureMissingValueFilter(
        sample.data, sample_result=sample
    ).run_filter_features_by_missingness()
    assert feature.audit.metrics["sample_wise"]["thresholds"] == {
        "sample_mv_tol": 0.23
    }


@pytest.mark.parametrize("method", ["QC-SVR", "QC-RLSC"])
def test_no_missing_correction_to_normalization_is_numeric(method):
    """Float output remains usable when imputation legitimately skips."""
    corrected = SignalCorrector(
        _dataset(), base_est=method, n_jobs=1
    ).run_signal_correction()
    final = list(corrected.data.values())[-1]
    assert all(pd.api.types.is_numeric_dtype(t) for t in final.intensity.dtypes)
    imputed = MissingValueImputer(final).run_imputation()
    assert imputed.audit.skipped
    normalized = DataNormalizer(
        imputed.data, norm_method="Median", n_jobs=1
    ).run_normalization()
    assert np.isfinite(normalized.data.intensity.to_numpy()).all()
    validation = corrected.audit.metrics["selection"]["validation"]
    assert validation["effective_folds_by_batch"] == [3]
    assert validation["evaluation_basis"] == "oof"


def test_too_few_folds_do_not_report_training_predictions_as_oof():
    """Two requested folds are unavailable under the three-fold policy."""
    from sklearn.linear_model import LinearRegression

    x = np.arange(3).reshape(-1, 1)
    full, oof = fit_predict_intra_batch_safely(
        LinearRegression(), x, np.arange(3) + 1.0, x, cv_folds=2
    )
    assert np.isfinite(full).all()
    assert np.isnan(oof).all()


def test_qa_context_update_preserves_scientific_settings():
    """Updating a stage label must not reset configured QA choices."""
    qa = QualityAssessor(
        _dataset(),
        corr_method="Pearson",
        scaling_method="None",
        is_outlier_threshold=2,
    )
    qa.run_assessment(context_updates={"pipeline_stage": "Review"})
    assert qa.config["corr_method"] == "Pearson"
    assert qa.config["scaling_method"] == "None"
    assert qa.config["is_outlier_threshold"] == 2


def test_normalization_auto_is_repeatable_and_metrics_are_current():
    """Reading metrics early or running twice must not change AUTO intent."""
    engine = DataNormalizer(
        _dataset(), norm_method="Auto", n_jobs=1, global_seed=123
    )
    _ = engine.normalization_metrics
    first = engine.run_normalization()
    second = engine.run_normalization()
    assert engine.config["norm_method"] == "Auto"
    for result in (first, second):
        assert result.audit.selection["is_auto"]
        assert result.audit.metrics["strategies"]["log_transform_active"]
        assert result.data.context.global_seed == 123
    pd.testing.assert_frame_equal(first.data.intensity, second.data.intensity)


@pytest.mark.parametrize("label", ["mar", "MNAR"])
def test_imputation_labels_and_metrics_reflect_execution(label):
    """Case-normalize labels and distinguish MNAR-only from MAR selection."""
    engine = MissingValueImputer(
        _dataset(missing=True, mechanism=label), mar_method="Median"
    )
    _ = engine.imputation_metrics
    result = engine.run_imputation()
    assert result.audit.metrics["imputation_status"] == "Completed"
    assert not result.data.intensity.isna().any().any()
    if label == "MNAR":
        assert result.audit.selected_method == "Not required"
        assert not result.audit.is_auto
        assert result.audit.candidate_results == {}


def test_unknown_missingness_fails_before_silent_incomplete_output():
    """Unknown mechanism labels cannot be silently ignored."""
    with pytest.raises(ValueError, match="missingness_type"):
        MissingValueImputer(_dataset(True, "typo")).run_imputation()


def test_constructor_and_runtime_validate_the_same_threshold(tmp_path):
    """Invalid runtime overrides fail before making artifact directories."""
    data = _dataset()
    with pytest.raises(ValueError):
        SampleMissingValueFilter(data, sample_mv_tol=-1)
    target = tmp_path / "invalid"
    with pytest.raises(ValueError):
        SampleMissingValueFilter(data).run_filter_samples_by_missingness(
            target, sample_mv_tol=-1
        )
    assert not target.exists()
    with pytest.raises(ValueError):
        DataNormalizer(data, n_jobs=0)
    with pytest.raises(ValueError):
        SignalCorrector(data, svr_gamma=-1)


def test_array_snapshots_are_detached_and_report_serializable():
    """Nested arrays must not alias the live computation buffers."""
    source = np.arange(6).reshape(2, 3)
    snap = snapshot_plot_value({"array": source})
    source[:] = -1
    assert snap["array"][0, 0] == 0
    report = ReportInput({}, {}, {"array": snap["array"]}, {})
    assert report.to_dict()["metadata"]["array"] == [[0, 1, 2], [3, 4, 5]]


def test_pipeline_result_filter_mapping_survives_dataclass_replace():
    """All public state must be explicit dataclass fields."""
    from pimqc.pipeline import PipelineResult

    stage = SampleMissingValueFilter(
        _dataset()
    ).run_filter_samples_by_missingness()
    result = PipelineResult(
        {},
        {},
        {},
        {},
        {},
        ReportInput({}, {}, {}, {}),
        filtering_results={"sample": stage},
    )
    assert replace(result).filtering_results["sample"] is stage


def test_auto_correction_retains_failures_and_auto_request(monkeypatch):
    """An inapplicable AUTO candidate cannot abort other viable candidates."""
    engine = SignalCorrector(_dataset(), base_est="Auto", n_jobs=1)
    original = engine._evaluate_correction_candidates

    def evaluate(methods_to_run, **kwargs):
        candidate = methods_to_run[0]
        if candidate == "QC-SVR":
            return original(methods_to_run, **kwargs)
        raise ValueError("Unavailable test candidate")

    monkeypatch.setattr(engine, "_evaluate_correction_candidates", evaluate)
    for _ in range(2):
        result = engine.run_signal_correction()
        selection = result.audit.metrics["selection"]
        assert selection["is_auto"]
        assert selection["selected_method"] == "QC-SVR"
        assert len(selection["failed_candidates"]) == 5
        assert engine.config["base_est"].upper() == "AUTO"


def test_skip_does_not_erase_imputation_request_parameters():
    """Skipped runs retain valid MNAR settings for the processor instance."""
    engine = MissingValueImputer(_dataset(), mnar_method="Row-wise")
    for _ in range(2):
        assert engine.run_imputation().audit.skipped
        assert engine.config["mnar_method"] == "Row-wise"
        assert engine.config["mnar_fraction"] == 0.5


def test_file_and_direct_auto_choices_have_identical_semantics():
    """AUTO casing must be accepted consistently in files and constructors."""
    from pimqc.config import validate_pipeline_params

    params = validate_pipeline_params(
        {
            "DataNormalizer": {"norm_method": "AUTO"},
            "MissingValueImputer": {"mar_method": "AUTO"},
        }
    )
    data = _dataset()
    assert (
        DataNormalizer(data, pipeline_params=params).config["norm_method"]
        == DataNormalizer(data, norm_method="AUTO").config["norm_method"]
    )
    assert (
        MissingValueImputer(data, pipeline_params=params).config["mar_method"]
        == MissingValueImputer(data, mar_method="AUTO").config["mar_method"]
    )
