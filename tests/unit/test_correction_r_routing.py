"""Keep explicit correction R routing and provenance independent of R installs.

The original numerical calls are separately tested in the optional reference
suite. These checks ensure normal configuration, typed stages, and reports do
not silently substitute native algorithms or expose ineffective parameters.
"""

from dataclasses import replace

import numpy as np
import pytest

from pimqc.constants import DEFAULT_RANDOM_SEED
from pimqc.processing.correction import SignalCorrector
from tests.unit.test_audit_regressions import _dataset


@pytest.mark.parametrize(
    "method,engine_name,options,stage",
    [
        ("WaveICA 2.0", "WaveICARCorrector", {}, "WaveICA corrected"),
        (
            "RUV-III",
            "RUVIIIRCorrector",
            {"ruv_control_features": ["F0", "F1"], "ruv_k": 1},
            "RUV corrected",
        ),
        (
            "SERRF",
            "SERRFRCorrector",
            {"serrf_r_source": "author.R"},
            "SERRF corrected",
        ),
    ],
)
def test_r_routing_uses_effective_provider_only(
    monkeypatch, method, engine_name, options, stage
):
    """R routing records actual parameters and never invokes native kernels."""
    from pimqc.processing.correction import analysis

    calls = {}

    class Provider:
        """Stand in for the verified original-R numerical boundary."""

        def __init__(self, **kwargs):
            calls["constructor"] = kwargs
            self.provenance = {
                "implementation": "r",
                "method": method,
                "parameters": {"actual_R_option": 7},
            }

        def fit_transform(self, frame, *args, **kwargs):
            """Return an unchanged complete matrix without OOF predictions."""
            calls["fit"] = kwargs
            return {stage: (frame.copy(), None)}

    def native_failure(*args, **kwargs):
        """Fail immediately if a native candidate is constructed."""
        raise AssertionError("R request used a Python kernel")

    monkeypatch.setattr(analysis, engine_name, Provider)
    for native in ("WaveICA2Corrector", "RUVCorrector", "SERRFCorrector"):
        monkeypatch.setattr(analysis, native, native_failure)
    result = SignalCorrector(
        _dataset(), base_est=method, implementation="r", **options
    ).run_signal_correction()
    metrics = result.audit.metrics
    selection = metrics["selection"]
    assert selection["implementation"] == "r"
    assert selection["requested_method"] == method
    assert not selection["is_auto"]
    assert selection["selected_score"] is None
    assert selection["validation"]["status"] == "unavailable"
    assert selection["validation"]["requested_folds"] == (
        5 if method == "SERRF" else None
    )
    if method == "SERRF":
        assert calls["constructor"]["cv_folds"] == 5
        assert not selection["validation"]["eligible_for_auto"]
        assert selection["validation"]["evaluation_basis"] == "unavailable"
    assert metrics["overall_performance"]["median_qc_rsd_current_oof"] is None
    assert metrics["stages_executed"][0]["parameters"] == {"actual_R_option": 7}
    assert stage in result.data
    assert calls["constructor"]["random_state"] == DEFAULT_RANDOM_SEED


@pytest.mark.parametrize("method", ["SERRF", "RUV-III", "WaveICA 2.0"])
@pytest.mark.parametrize("flag", ["is_logged", "is_scaled"])
def test_r_correction_rejects_transformed_context(method, flag, monkeypatch):
    """R routes reject already transformed datasets before loading R."""
    data = _dataset()
    data = replace(data, context=replace(data.context, **{flag: True}))

    def no_r():
        """Reject scale errors before initializing the optional R runtime."""
        raise AssertionError("Transformed input must not initialize R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    with pytest.raises(ValueError, match="unlogged, unscaled"):
        SignalCorrector(
            data, base_est=method, implementation="r"
        ).run_signal_correction()


@pytest.mark.parametrize(
    "method,options,message",
    [
        ("RUV-III", {}, "explicit ruv_control_features"),
        ("SERRF", {}, "serrf_r_source"),
        (
            "WaveICA 2.0",
            {"waveica_levels": 2},
            "Native Python options do not tune R",
        ),
        (
            "SERRF",
            {"serrf_r_source": "author.R", "serrf_n_tree": 200},
            "Native Python options do not tune R",
        ),
    ],
)
def test_r_correction_rejects_missing_or_ineffective_options(
    method, options, message
):
    """Do not hide missing scientific choices or ignored native settings."""
    with pytest.raises(ValueError, match=message):
        SignalCorrector(
            _dataset(), base_est=method, implementation="r", **options
        ).run_signal_correction()


def test_r_specific_options_do_not_silently_tune_native():
    """Custom technical-repeat design remains an R-only option."""
    with pytest.raises(ValueError, match="R-specific options"):
        SignalCorrector(
            _dataset(), base_est="RUV-III", ruv_replicate_column="Repeat"
        ).run_signal_correction()


def test_explicit_ruv_controls_reject_unrelated_native_method():
    """Reject RUV-specific controls for unrelated native correction methods."""
    with pytest.raises(ValueError, match="require method='RUV-III'"):
        SignalCorrector(
            _dataset(), base_est="WaveICA 2.0", ruv_control_features=["F0"]
        ).run_signal_correction()


def test_native_serrf_records_real_fold_and_fallback_diagnostics(tmp_path):
    """Small-QC degradation remains explicit in the serialized stage audit."""
    from pimqc.serialization import read_audit_payload, write_audit_payload

    result = SignalCorrector(
        _dataset(), base_est="SERRF", serrf_n_tree=10,
        serrf_corr_features=2, cv_folds=3, n_jobs=1,
    ).run_signal_correction()
    selection = result.audit.metrics["selection"]
    diagnostics = selection["native_diagnostics"]
    assert diagnostics["effective_folds_by_batch"] == {"B1": 3}
    assert selection["validation"]["effective_folds_by_batch"] == [3]
    assert "transductive" in selection["validation"]["scope"]
    # Two training QCs per fold cannot support a 3-observation correlation.
    assert diagnostics["fit_counts"]["oof"][
        "location_only_feature_batch_fits"
    ] > 0
    path = write_audit_payload(result.audit, tmp_path / "serrf-audit")
    restored = read_audit_payload(path)
    assert restored.metrics["selection"]["native_diagnostics"] == diagnostics


def test_r_options_follow_config_constructor_runtime_precedence(monkeypatch):
    """All added fields honor runtime > constructor > configuration."""
    from pimqc.processing.correction import analysis

    seen = {}

    class Provider:
        """Capture effective configuration without an R dependency."""

        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.provenance = {"parameters": kwargs}

        def fit_transform(self, frame, order, blank_mask):
            """Return a full-fit stage for dispatcher-only checks."""
            return {"WaveICA corrected": (frame.copy(), None)}

    monkeypatch.setattr(analysis, "WaveICARCorrector", Provider)
    processor = SignalCorrector(
        _dataset(),
        pipeline_params={
            "SignalCorrector": {
                "base_est": "WaveICA 2.0",
                "implementation": "r",
                "waveica_alpha": 0.1,
            },
        },
        waveica_alpha=0.2,
    )
    processor.run_signal_correction(waveica_alpha=0.3)
    assert seen["alpha"] == 0.3


def _auto_providers(monkeypatch, *, serrf_oof=True):
    """Exercise mixed AUTO scheduling without optional R dependencies."""
    from pimqc.processing.correction import analysis

    calls = []

    def provider(name):
        class Provider:
            def __init__(self, **kwargs):
                self.name = name
                calls.append((name, kwargs))
                self.provenance = {
                    "implementation": "r", "method": name,
                    "parameters": {"provider": name},
                }
                self.diagnostics = {
                    "effective_folds_by_batch": {"B1": 3},
                    "validation": "held-out QC predictions",
                }

            def fit_transform(self, frame, *args, **kwargs):
                full = frame.copy()
                qc = full.columns.get_level_values("Sample Type") == "QC"
                full.loc[:, qc] = np.repeat(
                    full.loc[:, qc].mean(axis=1).to_numpy()[:, None],
                    qc.sum(), axis=1,
                )
                oof = full.copy() if name == "SERRF" and serrf_oof else None
                if oof is not None and serrf_oof == "invalid":
                    oof.loc[:, qc] = np.nan
                if oof is not None and serrf_oof == "unchanged":
                    oof = frame.copy()
                return {name: (full, oof)}

        return Provider

    class Regression:
        def __init__(self, method, **kwargs):
            self.name = method
            calls.append((method, kwargs))
            expected = "r" if method == "QC-RLSC" and kwargs.get(
                "robust"
            ) else "python"
            assert kwargs["implementation"] == expected
            self.provenance = (
                {"implementation": "r", "parameters": {}}
                if expected == "r" else {}
            )

        def fit_transform(self, intensity_df, **kwargs):
            return {self.name: (intensity_df.copy(), intensity_df.copy())}

    def no_native(*args, **kwargs):
        raise AssertionError("An R candidate fell back to native execution")

    for method, r_name, native in (
        ("SERRF", "SERRFRCorrector", "SERRFCorrector"),
        ("RUV-III", "RUVIIIRCorrector", "RUVCorrector"),
        ("WaveICA 2.0", "WaveICARCorrector", "WaveICA2Corrector"),
    ):
        monkeypatch.setattr(analysis, r_name, provider(method))
        monkeypatch.setattr(analysis, native, no_native)
    monkeypatch.setattr(analysis, "RegressionCorrector", Regression)
    monkeypatch.setattr(analysis, "MetanormRLOESSCorrector", no_native)
    return calls


def test_r_auto_resolves_each_backend_and_preserves_repeat_intent(
    monkeypatch, tmp_path,
):
    """R replaces supported candidates, not the existing AUTO portfolio."""
    from pimqc.serialization import read_audit_payload, write_audit_payload

    calls = _auto_providers(monkeypatch)
    engine = SignalCorrector(
        _dataset(), implementation="r", base_est="Auto", cv_folds=3,
        serrf_r_source="author.R", ruv_control_features=["F0", "F1"],
    )
    expected = {
        "SERRF": "r", "RUV-III": "r", "WaveICA 2.0": "r",
        "QC-RLSC": "python", "robust QC-RLSC": "r", "QC-SVR": "python",
    }
    for _ in range(2):
        result = engine.run_signal_correction()
        selection = result.audit.metrics["selection"]
        assert selection["requested_implementation"] == "r"
        assert selection["implementation"] == "r"
        assert selection["candidate_implementations"] == expected
        assert not selection["failed_candidates"]
        candidates = {
            row["method"]: row for row in selection["candidate_results"]
        }
        assert set(candidates) == set(expected)
        for method, backend in expected.items():
            assert candidates[method]["implementation"] == backend
            assert np.isfinite(candidates[method]["auto_score"])
            assert bool(candidates[method]["implementation_provenance"]) == (
                backend == "r"
            )
        assert candidates["SERRF"]["validation"]["evaluation_basis"] == "oof"
        assert candidates["SERRF"]["validation"][
            "effective_folds_by_batch"
        ] == [3]
        assert candidates["RUV-III"]["validation"][
            "evaluation_basis"
        ] == "full_model"
        assert engine.config["implementation"] == "r"
        assert engine.config["base_est"].upper() == "AUTO"
    assert len(calls) == 12
    assert calls[0][1]["cv_folds"] == 3
    restored = read_audit_payload(write_audit_payload(
        result.audit, tmp_path / "mixed-audit"
    ))
    assert restored.metrics["selection"] == selection


def test_r_auto_isolates_missing_options_and_dependency_failures(monkeypatch):
    """Failure never replaces an R algorithm with its native namesake."""
    from pimqc.processing.correction import analysis

    calls = _auto_providers(monkeypatch)

    def unavailable(**kwargs):
        raise ImportError("WaveICA package is unavailable")

    monkeypatch.setattr(analysis, "WaveICARCorrector", unavailable)
    engine = SignalCorrector(_dataset(), implementation="r", base_est="Auto")
    selection = engine.run_signal_correction().audit.metrics["selection"]
    assert selection["requested_implementation"] == "r"
    assert selection["implementation"] == "python"
    assert selection["implementation_provenance"] == []
    assert {row["method"] for row in selection["failed_candidates"]} == {
        "SERRF", "RUV-III", "WaveICA 2.0",
    }
    assert all(
        row["implementation"] == "r"
        for row in selection["failed_candidates"]
    )
    assert {name for name, _ in calls} == {"QC-RLSC", "QC-SVR"}
    assert engine.config["implementation"] == "r"


@pytest.mark.parametrize("serrf_oof", [False, "invalid"])
def test_r_serrf_cannot_replace_missing_oof_with_full_fit(
    monkeypatch, serrf_oof,
):
    """An excellent training fit is ineligible without held-out evidence."""
    _auto_providers(monkeypatch, serrf_oof=serrf_oof)
    result = SignalCorrector(
        _dataset(), implementation="r", base_est="Auto", cv_folds=3,
        serrf_r_source="author.R", ruv_control_features=["F0", "F1"],
    ).run_signal_correction()
    selection = result.audit.metrics["selection"]
    candidate = next(
        row for row in selection["candidate_results"]
        if row["method"] == "SERRF"
    )
    assert candidate["status"] == "ineligible"
    assert candidate["auto_score"] is None
    assert candidate["eval_rsd"] is None
    assert candidate["final_rsd_full"] == pytest.approx(0)
    assert candidate["validation"]["evaluation_basis"] == "unavailable"
    assert selection["selected_method"] != "SERRF"


def test_r_serrf_auto_uses_held_out_not_training_improvement(monkeypatch):
    """A perfect full fit cannot hide unimproved held-out QC predictions."""
    _auto_providers(monkeypatch, serrf_oof="unchanged")
    result = SignalCorrector(
        _dataset(), implementation="r", base_est="Auto", cv_folds=3,
        serrf_r_source="author.R", ruv_control_features=["F0", "F1"],
    ).run_signal_correction()
    candidates = {
        row["method"]: row
        for row in result.audit.metrics["selection"]["candidate_results"]
    }
    serrf = candidates["SERRF"]
    assert serrf["validation"]["eligible_for_auto"]
    assert serrf["final_rsd_full"] == pytest.approx(0)
    assert serrf["eval_rsd"] == serrf["final_rsd_oof"] > 0
    assert serrf["median_qc_rsd_improvement_score"] == pytest.approx(0)
    assert serrf["featurewise_qc_rsd_improvement_score"] == pytest.approx(0)
    assert serrf["auto_score"] < candidates["RUV-III"]["auto_score"]
