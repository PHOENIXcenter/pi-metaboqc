"""Scientific diagnostics are computed once, then survive portable rendering."""

from dataclasses import fields, replace
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pimqc import (
    DataNormalizer,
    MissingValueImputer,
    QualityAssessor,
    SignalCorrector,
)
from pimqc.core import MetaboDataset
from pimqc.plotting.assessment import render_assessment_comparison
from pimqc.plotting.correction import CorrectionPlotter
from pimqc.plotting.imputation import ImputationPlotter
from pimqc.plotting.normalization import NormalizationPlotter
from pimqc.plotting.sample_structure import plot_sample_structure_change_map
from pimqc.serialization import read_audit_payload, write_audit_payload
from pimqc.statistics import sample_structure


def _dataset():
    rng = np.random.default_rng(73)
    names = pd.Index([f"S{i}" for i in range(16)], name="Sample Name")
    features = pd.Index([f"F{i}" for i in range(30)], name="Metabolite")
    data = pd.DataFrame(
        rng.lognormal(5, 0.3, (30, 16)), index=features, columns=names
    )
    metadata = pd.DataFrame(
        {
            "Sample Type": ["QC", "Sample"] * 8,
            "Batch": ["B1"] * 16,
            "Inject Order": np.arange(1, 17),
        },
        index=names,
    )
    return MetaboDataset.from_tables(data, metadata)


def _forbid(*args, **kwargs):
    raise AssertionError("Rendering must not recompute scientific diagnostics")


@pytest.mark.parametrize("stage", ["normalization", "imputation", "correction"])
def test_saved_audit_renders_without_distance_or_qc_recalculation(
    stage, tmp_path, monkeypatch
):
    """
    Round-trip real stage evidence and forbid scientific work during drawing.
    """
    data = _dataset()
    if stage == "normalization":
        result = DataNormalizer(
            data, norm_method="PQN", n_jobs=1
        ).run_normalization()
    elif stage == "imputation":
        missing = data.intensity.copy()
        missing.iloc[0, 3] = np.nan
        result = MissingValueImputer(
            data.with_intensity(missing), mar_method="Median"
        ).run_imputation()
    else:
        result = SignalCorrector(
            data,
            base_est="QC-RLSC",
            rlsc_min_qc=3,
            rlsc_robust=False,
            cv_folds=2,
            n_jobs=1,
        ).run_signal_correction()
    artifact = write_audit_payload(result.audit, tmp_path / "audit")
    audit = read_audit_payload(artifact)
    payload = audit.plot_payload
    diagnostic = (
        audit.candidate_results[audit.selected_label]["sample_structure"]
        if stage == "correction"
        else payload.sample_structure
    )
    assert list(diagnostic["samples"].columns) == [
        "scale_shift",
        "rank_rho",
        "local_trust",
    ]
    assert len(diagnostic["samples"]) == 8
    assert np.isfinite(
        diagnostic["metrics"]["sample_structure_composite_preservation"]
    )
    # Neither live processor methods nor pairwise geometry may run after
    # loading.
    monkeypatch.setattr(
        sample_structure, "calc_sample_structure_arrays", _forbid
    )
    monkeypatch.setattr(
        DataNormalizer, "_calc_qc_variance_stabilization_values", _forbid
    )
    monkeypatch.setattr(DataNormalizer, "_calc_qc_structure_values", _forbid)
    monkeypatch.setattr(SignalCorrector, "extract_qc_rsd_series", _forbid)
    monkeypatch.setattr("scipy.stats.wilcoxon", _forbid)
    monkeypatch.setattr(
        "pimqc.statistics.normalization.calculate_rle_diagnostics", _forbid
    )
    if stage == "normalization":
        assert payload.qc_diagnostics.keys() == {"Before Norm", "After Norm"}
        assert audit.selection is payload.selection
        assert audit.candidate_results is payload.selection.get(
            "candidate_results"
        )
        plotter = NormalizationPlotter(payload)
        dashboard = plotter.plot_normalization_dashboard()
    elif stage == "imputation":
        plotter = ImputationPlotter(payload)
        metrics, true, pred = audit.candidate_results[audit.selected_method]
        dashboard = plotter.plot_imputation_method_dashboard(
            metrics=metrics,
            true_vals=true,
            pred_vals=pred,
            method_name=audit.selected_label,
        )
    else:
        assert audit.candidate_results is payload.candidate_results
        assert audit.selected_prediction is payload.selected_prediction
        plotter = CorrectionPlotter(payload)
        dashboard = plotter.plot_correction_dashboard(
            audit.candidate_results, audit.selected_label
        )
    assert dashboard is not None
    plotter.save_and_show_pw(
        dashboard, file_path=str(tmp_path / f"{stage}.svg"), show_plot=False
    )
    assert (tmp_path / f"{stage}.svg").stat().st_size > 1000
    if stage != "imputation":
        names = {item.name for item in fields(audit)}
        assert "candidate_results" not in names
        assert "selection" not in names
        assert "selected_prediction" not in names
        wire = json.loads(
            (artifact / "payload.json").read_text(encoding="utf-8")
        )
        assert "candidate_results" not in wire["fields"]
    plt.close("all")


def test_sample_structure_scores_and_plot_coordinates_share_one_geometry(
    monkeypatch,
):
    """
    Keep plot coordinates and unchanged scores aligned to one distance pass.
    """
    data = _dataset().annotated_frame()
    calls = []
    original = sample_structure.calc_sample_structure_arrays

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(
        sample_structure, "calc_sample_structure_arrays", counted
    )
    diagnostic = sample_structure.calc_sample_structure_diagnostics(data, data)
    assert len(calls) == 1
    assert diagnostic["metrics"][
        "sample_structure_composite_preservation"
    ] == pytest.approx(1)
    np.testing.assert_allclose(
        diagnostic["samples"]["scale_shift"], 0, atol=1e-10
    )
    np.testing.assert_allclose(diagnostic["samples"]["rank_rho"], 1)
    monkeypatch.setattr(
        sample_structure, "calc_sample_structure_arrays", _forbid
    )
    fig, ax = plt.subplots()
    plot_sample_structure_change_map(ax, diagnostic)
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        diagnostic["samples"][["scale_shift", "rank_rho"]],
    )
    plt.close(fig)


def test_saved_qa_comparison_and_validation(tmp_path, monkeypatch):
    """Draw restored audits in stage order without a live quality assessor."""
    data = _dataset()
    before = QualityAssessor(data).run_assessment().audit
    normalized = (
        DataNormalizer(data, norm_method="PQN").run_normalization().data
    )
    after = QualityAssessor(normalized).run_assessment().audit
    audits = {}
    for label, audit in (("Before", before), ("After", after)):
        path = write_audit_payload(audit, tmp_path / label)
        audits[label] = read_audit_payload(path)
    original_stage = audits["Before"].plot_payload.data.context.pipeline_stage
    monkeypatch.setattr(QualityAssessor, "run_assessment", _forbid)
    assets = render_assessment_comparison(audits, tmp_path / "comparison")
    assert set(assets) == {
        "01_QC_Sample_RSD_Dashboard",
        "02_PCA_Scatter_Dashboard",
        "03_QC_Correlation_Dashboard",
        "04_Outlier_Diagnosis_Dashboard",
    }
    for path in assets.values():
        text = path.read_text(encoding="utf-8")
        assert "Before" in text and "After" in text
        assert text.index("Before") < text.index("After")
    assert (
        audits["Before"].plot_payload.data.context.pipeline_stage
        == original_stage
    )
    single = render_assessment_comparison(
        {"Only": audits["Before"]}, tmp_path / "single"
    )
    assert len(single) == 4
    assert all(path.is_file() for path in single.values())
    skipped = replace(before, skipped=True, plot_payload=None)
    assert (
        render_assessment_comparison({"Skipped": skipped}, tmp_path / "skipped")
        == {}
    )
    assert not (tmp_path / "skipped").exists()
    with pytest.raises(ValueError, match="consistent"):
        render_assessment_comparison(
            {
                "Before": before,
                "After": replace(after, correlation_method="Different"),
            },
            tmp_path / "invalid",
        )
    plt.close("all")


def test_native_report_and_saved_qa_share_compositor(tmp_path, monkeypatch):
    """
    Require reports and independent comparisons to use one drawing entry point.
    """
    from pimqc.plotting.assessment import comparison
    from pimqc.reporting import utils

    assert (
        utils.render_assessment_comparison
        is comparison.render_assessment_comparison
    )
    from pimqc.plotting import assembly

    assert not hasattr(assembly, "stitch_svg_grids")
    reporter = utils.VisualAssetReporter(tmp_path)
    audit = QualityAssessor(_dataset()).run_assessment().audit
    audits = {"Before": audit, "After": audit}
    calls = []

    def compose(directories, output_dir, **kwargs):
        calls.append((directories, kwargs))
        return {
            "02_PCA_Scatter_Dashboard": output_dir
            / "02_PCA_Scatter_Dashboard.svg"
        }

    monkeypatch.setattr(utils, "render_assessment_comparison", compose)
    assets = reporter.compile_assessor_report(audits, is_multi_batch=False)
    assert len(calls) == 1
    assert len(calls[0][0]) == 2
    assert "02_PCA_Scatter_Dashboard" in assets
