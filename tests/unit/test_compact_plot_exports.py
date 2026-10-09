"""Regress compact plotting APIs, export geometry and selected-method safety."""

import io
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pimqc import (
    DataNormalizer,
    FeatureMissingValueFilter,
    FeatureQualityFilter,
    MissingValueImputer,
    SignalCorrector,
)
from pimqc.core import MetaboDataset
from pimqc.plotting import plot_utils as pu
from pimqc.plotting.base import BasePlotter
from pimqc.plotting.correction import CorrectionPlotter
from pimqc.plotting.filtering import FilteringPlotter
from pimqc.plotting.imputation import ImputationPlotter
from pimqc.plotting.normalization import NormalizationPlotter


@pytest.fixture(scope="module")
def article_audits():
    """Small real 1.5.0 stage results; no R or full tutorial computation."""
    rng = np.random.default_rng(42)
    names = pd.Index([f"S{i}" for i in range(26)], name="Sample Name")
    features = pd.Index([f"F{i}" for i in range(30)], name="Metabolite")
    intensity = pd.DataFrame(
        rng.lognormal(5, 0.12, (30, 26)), index=features, columns=names
    )
    intensity.iloc[:, -2:] *= 0.02
    metadata = pd.DataFrame({
        "Sample Type": ["QC", "Sample"] * 12 + ["Blank"] * 2,
        "Batch": ["B1"] * 12 + ["B2"] * 12 + ["B1", "B2"],
        "Inject Order": np.arange(1, 27),
        "Bio Group": ["G1"] * 12 + ["G2"] * 12 + [None, None],
    }, index=names)
    complete = MetaboDataset.from_tables(intensity, metadata)
    missing = intensity.copy()
    missing.iloc[:8, 3] = np.nan
    mv = FeatureMissingValueFilter(
        complete.with_intensity(missing)
    ).run_filter_features_by_missingness()
    quality = FeatureQualityFilter(
        mv.data, missingness_metadata=mv.audit.feature_tracking
    ).run_filter_features_by_quality()
    correction = SignalCorrector(
        mv.data, base_est="QC-RLSC", rlsc_min_qc=3, rlsc_robust=False,
        cv_folds=2, n_jobs=1,
    ).run_signal_correction()
    imputation = MissingValueImputer(
        mv.data, mar_method="Median"
    ).run_imputation()
    normalization = DataNormalizer(
        complete, norm_method="Auto", n_jobs=1
    ).run_normalization()
    return {
        "missingness": mv.audit, "quality": quality.audit,
        "correction": correction.audit, "imputation": imputation.audit,
        "normalization": normalization.audit,
    }


def build_article(name, audits):
    """Build a compact dashboard from the matching stage audit."""
    if name in {"missingness", "quality"}:
        plotter = FilteringPlotter(audits[name].plot_payload)
        builder = (
            plotter.plot_high_mv_filter_article_dashboard
            if name == "missingness"
            else plotter.plot_low_quality_filter_article_dashboard
        )
        return plotter, builder()
    if name == "correction":
        audit = audits[name]
        plotter = CorrectionPlotter(audit.plot_payload)
        return plotter, plotter.plot_correction_article_dashboard(
            audit.plot_payload.candidate_results, audit.selected_label
        )
    if name.startswith("imputation"):
        audit = audits["imputation"]
        plotter = ImputationPlotter(audit.plot_payload)
        builder = (
            plotter.plot_imputation_reconstruction_article_dashboard
            if name.endswith("reconstruction")
            else plotter.plot_imputation_preservation_article_dashboard
        )
        return plotter, builder(audit.candidate_results, audit.selected_method)
    plotter = NormalizationPlotter(audits["normalization"].plot_payload)
    return plotter, plotter.plot_normalization_article_dashboard()


@pytest.mark.parametrize("name", [
    "missingness", "quality", "correction", "imputation_reconstruction",
    "imputation_preservation", "normalization",
])
def test_article_current_payload_and_export_width(
    name, article_audits, tmp_path
):
    """Preserve exported geometry and diagnostics for every compact builder."""
    plotter, dashboard = build_article(name, article_audits)
    assert dashboard is not None
    path = tmp_path / f"{name}.svg"
    plotter.save_and_show_pw(
        dashboard, file_path=str(path), save_format="svg", show_plot=False
    )
    svg = ET.parse(path).getroot()
    assert float(svg.attrib["width"].removesuffix("pt")) == pytest.approx(
        pu.ARTICLE_TARGET_WIDTH_IN * 72, abs=1e-5
    )
    assert plotter.resolve_dashboard_display_width(dashboard) == "60%"
    for ax in dashboard.bricks_dict.values():
        if ax.axison:
            for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
                assert label.get_fontsize() == pu.ARTICLE_AXIS_TICK_FONTSIZE
    if name == "correction":
        assert any("d_ratio" in key for key in dashboard.bricks_dict)
        assert any("sample_structure" in key for key in dashboard.bricks_dict)
    if name == "normalization":
        assert any("sample_structure" in key for key in dashboard.bricks_dict)
    plt.close("all")


def test_article_width_adjustment_does_not_rescale_fonts():
    """Change panel geometry without rescaling the configured text artists."""
    import patchworklib as pw

    pw.clear()
    bricks = [
        pw.Brick(figsize=(2.3, 1.5), label=f"width_{i}") for i in range(3)
    ]
    artists = []
    for ax in bricks:
        artists.extend([
            ax.set_title("Article title", fontsize=pu.ARTICLE_TITLE_FONTSIZE),
            ax.set_xlabel("Intensity", fontsize=pu.ARTICLE_AXIS_LABEL_FONTSIZE),
            ax.text(
                0.3, 0.4, "Median: 12.3",
                fontsize=pu.ARTICLE_ANNOTATION_FONTSIZE,
            ),
        ])
    before = [(t.get_fontsize(), t.get_fontfamily()) for t in artists]
    dashboard = BasePlotter._finalize_article_dashboard(
        bricks[0] | bricks[1] | bricks[2]
    )
    BasePlotter._save_patchwork(dashboard, io.BytesIO(), format="svg")
    assert before == [(t.get_fontsize(), t.get_fontfamily()) for t in artists]
    plt.close("all")


def test_article_missing_selected_imputer_does_not_plot_other_method(
    article_audits,
):
    """Return no plot when the selected method has no matching evidence."""
    audit = article_audits["imputation"]
    plotter = ImputationPlotter(audit.plot_payload)
    assert plotter.plot_imputation_reconstruction_article_dashboard(
        audit.candidate_results, "BPCA"
    ) is None
    assert plotter.plot_imputation_preservation_article_dashboard(
        audit.candidate_results, "BPCA"
    ) is None
