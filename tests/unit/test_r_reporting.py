"""Reports describe actually executed R providers and reference R assets."""

import copy

import pytest

from pimqc.reporting import NarrativeStatsReporter

from tests.unit.test_reporting_contract import _report_input


def _provenance(method, package, function):
    """Create representative JSON-native provenance returned by the bridge."""
    return {
        "implementation": "r",
        "method": method,
        "package": package,
        "package_version": "2.1-test",
        "function": function,
        "r_version": "R version 4.5.2 (test)",
        "rpy2_version": "3.6.7-test",
        "seed": 37,
        "parameters": {"test_param": 0.125, "sample_vector": [1, 2, 3]},
        "warnings": ["Example upstream warning"],
    }


def _r_report_input():
    """Populate all three R-capable stages using their real report nesting."""
    report = _report_input()
    metrics = report.pipeline_metrics
    correction = metrics["signal_correction"]
    correction["stages_executed"][0]["algorithm"] = "Metanorm-rLOESS"
    correction["selection"] = {
        "requested_method": "Metanorm-rLOESS",
        "selected_method": "Metanorm-rLOESS",
        "is_auto": False,
        "implementation": "r",
        "implementation_provenance": [
            _provenance(
                "Metanorm-rLOESS", "metanorm", "metanorm::metanormWorker"
            )
        ],
    }
    imputation = metrics["missing_value_imputation"]
    bpca = _provenance(
        "BPCA", "pcaMethods", "pcaMethods::pca / pcaMethods::completeObs"
    )
    imputation["strategies"] = {"mnar_method": "QRILC", "mnar_fraction": None}
    imputation["selection"].update(
        {
            "requested_method": "BPCA",
            "selected_method": "BPCA",
            "selected_label": "BPCA",
            "implementation": "r",
            "implementation_provenance": [
                {**bpca, "phase": "evaluation_mar"},
                {**bpca, "phase": "final_mar"},
                _provenance("QRILC", "imputeLCMD", "imputeLCMD::impute.QRILC"),
            ],
        }
    )
    normalization = metrics["normalization"]
    normalization["strategies"]["normalization_method"] = "VSN"
    normalization["selection"].update(
        {
            "requested_method": "VSN",
            "selected_method": "VSN",
            "selected_label": "VSN",
            "implementation": "r",
            "implementation_provenance": [
                _provenance("VSN", "vsn", "vsn::vsn2 / vsn::predict")
            ],
        }
    )
    return report


@pytest.mark.parametrize("kind", ["Comprehensive", "Brief"])
def test_reports_include_actual_r_providers(kind, tmp_path):
    """Report functions, versions, seeds and warnings without duplicate fits."""
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(_r_report_input(), report_folder="report")
    text = (tmp_path / "report" / f"Report_{kind}.md").read_text(
        encoding="utf-8"
    )
    for function in (
        "metanorm::metanormWorker",
        "pcaMethods::pca / pcaMethods::completeObs",
        "imputeLCMD::impute.QRILC",
        "vsn::vsn2 / vsn::predict",
    ):
        assert text.count(function) == 1
    assert "R version 4.5.2 (test)" in text
    assert "rpy2 3.6.7-test" in text
    assert "random seed: 37" in text
    assert "test_param=0.125" in text
    assert "sample_vector=" not in text
    assert "Example upstream warning" in text
    assert "structural scale factor" not in text
    assert "generalized baseline shift" not in text
    if kind == "Comprehensive":
        assert "Normalization_Dashboard_VSN_R.svg" in text
        assert "Normalization_Dashboard_VSN.svg" not in text


@pytest.mark.parametrize("kind", ["Comprehensive", "Brief"])
def test_native_report_retains_existing_behavior(kind, tmp_path):
    """No R provider or R suffix is fabricated for a native execution."""
    report = _report_input()
    original = copy.deepcopy(report.to_dict())
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report, report_folder="report")
    text = (tmp_path / "report" / f"Report_{kind}.md").read_text(
        encoding="utf-8"
    )
    assert "Original R implementation" not in text
    assert "no original-package call" not in text
    if kind == "Comprehensive":
        assert "Normalization_Dashboard_QUANTILE_Log2.svg" in text
    for stage, values in original["pipeline_metrics"].items():
        assert report.pipeline_metrics[stage] == values


def test_skipped_r_request_does_not_claim_package_was_executed(tmp_path):
    """No-data skips may request R without initializing any original package."""
    report = _r_report_input()
    imputation = report.pipeline_metrics["missing_value_imputation"]
    imputation["imputation_status"] = "Skipped"
    imputation["selection"]["implementation_provenance"] = []
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report, report_folder="report")
    for kind in ("Comprehensive", "Brief"):
        text = (tmp_path / "report" / f"Report_{kind}.md").read_text(
            encoding="utf-8"
        )
        assert "no original-package call was recorded" in text
        assert "pcaMethods::pca" not in text


@pytest.mark.parametrize("kind", ["Comprehensive", "Brief"])
def test_r_auto_reports_evaluated_provider_when_python_wins(kind, tmp_path):
    """R evaluation provenance remains visible even for a native winner."""
    report = _report_input()
    selection = report.pipeline_metrics["normalization"]["selection"]
    selection.update(
        requested_method="Auto",
        is_auto=True,
        requested_implementation="r",
        implementation="python",
        selected_method="ROBUST_LOG_ONLY",
        selected_label="ROBUST_LOG_ONLY",
        implementation_provenance=[],
        candidate_results=[
            {
                "method": "VSN",
                "implementation": "r",
                "implementation_provenance": [
                    _provenance("VSN", "vsn", "vsn::vsn2 / vsn::predict")
                ],
            },
            {
                "method": "ROBUST_LOG_ONLY",
                "implementation": "python",
                "selected": True,
            },
        ],
    )
    NarrativeStatsReporter(base_dir=str(tmp_path)).generate_markdown(
        report, report_folder="report"
    )
    text = (tmp_path / "report" / f"Report_{kind}.md").read_text("utf-8")
    assert text.count("vsn::vsn2 / vsn::predict") == 1
    assert "recorded implementation is **python**" in text
    assert "AUTO may also evaluate Python-only candidates" in text
    if kind == "Comprehensive":
        assert "Implementation" in text


@pytest.mark.parametrize("converged", [True, False, None])
def test_native_vsn_reports_real_model_and_convergence(tmp_path, converged):
    """No shared-scale description or fabricated convergence for affine VSN."""
    report = _report_input()
    normalization = report.pipeline_metrics["normalization"]
    normalization["strategies"]["normalization_method"] = "VSN"
    normalization["selection"].update(
        implementation="python", selected_method="VSN", selected_label="VSN"
    )
    normalization["vsn_parameters"] = {
        "vsn_scale": 0.01, "vsn_shift": 3.0, "vsn_converged": converged,
    }
    NarrativeStatsReporter(base_dir=str(tmp_path)).generate_markdown(
        report, report_folder="report"
    )
    text = (tmp_path / "report" / "Report_Comprehensive.md").read_text("utf-8")
    assert "sample-specific affine calibrations" in text
    assert "geometric mean of sample slopes" in text
    assert "structural scale factor" not in text
    if converged is True:
        assert "All recorded optimization rounds converged" in text
    elif converged is False:
        assert "At least one optimization round did not converge" in text
    else:
        assert "Optimization convergence was not recorded" in text
