"""Compare the production adapter with the installed author's WaveICA R code."""

import numpy as np
import pandas as pd
import rpy2.robjects as ro

from pimqc import MetaboDatasetBuilder
from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.waveica_r import WaveICARCorrector

from .helpers import require_r_package


def _waveica_study():
    """Return an unsorted finite study with Blanks and stable identities."""
    rng = np.random.default_rng(632)
    n_samples = 43
    order = rng.permutation(np.arange(1.0, n_samples + 1))
    values = rng.lognormal(5.0, 0.18, (8, n_samples))
    values *= 1 + 0.1 * np.sin(order / 8)
    frame = pd.DataFrame(
        values,
        index=[f"Metabolite {i}" for i in range(8)],
        columns=[f"Injection {i}" for i in range(n_samples)],
    )
    blanks = np.zeros(n_samples, dtype=bool)
    blanks[[3, 8, 27]] = True
    frame.iloc[0, 3] = np.nan
    return frame, order, blanks


def _direct_original_package(frame, order, *, components=3):
    """Call the installed package without the production helper."""
    values = frame.to_numpy(dtype=float)
    matrix = ro.r["matrix"](
        ro.FloatVector(values.ravel(order="F")),
        nrow=values.shape[0],
        ncol=values.shape[1],
    )
    direct = ro.r(
        """
        function(mat, injection_order, components) {
            previous <- options(mc.cores=1L)
            on.exit(options(previous), add=TRUE)
            input <- t(mat)
            colnames(input) <- paste0("feature_", seq_len(ncol(input)))
            rownames(input) <- paste0("sample_", seq_len(nrow(input)))
            t(WaveICA2.0::WaveICA_2.0(
                data=input, wf="haar", Injection_Order=injection_order,
                alpha=0, Cutoff=0.1, K=as.integer(components)
            )$data_wave)
        }
        """
    )
    return np.asarray(
        direct(matrix, ro.FloatVector(order.tolist()), components), dtype=float
    )


def test_waveica_r_matches_installed_package_and_restores_order():
    """Compare identical package calls; retain Blank values and provenance."""
    for package in ("WaveICA2.0", "JADE", "corpcor"):
        require_r_package(package)
    frame, order, blanks = _waveica_study()
    original = frame.copy(deep=True)
    engine = WaveICARCorrector(n_components=3, random_state=88)
    result, oof = engine.fit_transform(frame, order, blanks)[
        "WaveICA corrected"
    ]
    indices = np.flatnonzero(~blanks)
    sorted_indices = indices[np.argsort(order[indices], kind="stable")]
    expected = _direct_original_package(
        frame.iloc[:, sorted_indices], order[sorted_indices]
    )
    np.testing.assert_allclose(
        result.iloc[:, sorted_indices], expected, rtol=1e-11, atol=1e-9
    )
    np.testing.assert_array_equal(result.iloc[:, blanks], frame.iloc[:, blanks])
    assert oof is None
    assert result.index.equals(frame.index)
    assert result.columns.equals(frame.columns)
    pd.testing.assert_frame_equal(frame, original)
    assert engine.provenance["package"] == "WaveICA2.0"
    assert engine.provenance["package_version"]
    assert engine.provenance["function"] == "WaveICA2.0::WaveICA_2.0"
    assert engine.provenance["input_policy"]["blank_output"] == "unchanged"


def test_waveica_r_restores_parallel_option_and_rng():
    """The adapter must not persist its serial option or R seed changes."""
    require_r_package("WaveICA2.0")
    frame, order, blanks = _waveica_study()
    original_option = ro.r("options('mc.cores')")
    ro.r("options(mc.cores=3L); set.seed(951)")
    seed_before = np.asarray(ro.r(".Random.seed"), dtype=int).copy()
    try:
        WaveICARCorrector(n_components=3, random_state=42).fit_transform(
            frame, order, blanks
        )
        assert int(ro.r("getOption('mc.cores')")[0]) == 3
        np.testing.assert_array_equal(
            np.asarray(ro.r(".Random.seed"), dtype=int), seed_before
        )
    finally:
        ro.r["options"](original_option)


def test_waveica_r_missing_matches_manual_median_and_original_package():
    """Compare temporary filling with a separate, direct original R call."""
    for package in ("WaveICA2.0", "JADE", "corpcor"):
        require_r_package(package)
    frame, order, blanks = _waveica_study()
    frame = frame.iloc[:4, :].copy()
    frame.iloc[0, 0] = np.nan
    frame.iloc[2, 5] = np.nan
    original = frame.copy(deep=True)
    positions = np.flatnonzero(~blanks)
    positions = positions[np.argsort(order[positions], kind="stable")]
    manual = frame.iloc[:, positions].copy()
    manual = manual.T.fillna(manual.median(axis=1)).T
    expected = _direct_original_package(manual, order[positions])
    expected[np.isnan(frame.iloc[:, positions].to_numpy())] = np.nan
    engine = WaveICARCorrector(n_components=3, random_state=88)
    result, _ = engine.fit_transform(frame, order, blanks)[
        "WaveICA corrected"
    ]
    np.testing.assert_allclose(
        result.iloc[:, positions], expected, rtol=1e-11, atol=1e-9,
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(result.iloc[:, blanks], frame.iloc[:, blanks])
    pd.testing.assert_frame_equal(frame, original)
    assert "missing_input_adapter" in engine.provenance


def test_waveica_r_stage_exports_full_fit_dashboard(tmp_path):
    """Run the real R path through its public processor and report payload."""
    require_r_package("WaveICA2.0")
    frame, order, blanks = _waveica_study()
    sample_type = np.where(order % 3 == 0, "QC", "Sample")
    sample_type[blanks] = "Blank"
    metadata = pd.DataFrame(
        {
            "Sample Name": frame.columns,
            "Sample Type": sample_type,
            "Batch": np.where(order < 22, "Batch one", "Batch two"),
            "Inject Order": order,
        }
    )
    dataset = MetaboDatasetBuilder(metadata, frame).run_build().data
    output_dir = tmp_path / "waveica"
    result = SignalCorrector(
        dataset,
        base_est="WaveICA 2.0",
        implementation="r",
        waveica_components=3,
    ).run_signal_correction(output_dir=str(output_dir))
    selection = result.audit.metrics["selection"]
    assert selection["implementation"] == "r"
    assert selection["selected_method"] == "WaveICA 2.0"
    assert selection["validation"]["status"] == "unavailable"
    assert selection["validation"]["evaluation_basis"] == "full_model"
    assert selection["selected_score"] is None
    provenance = selection["implementation_provenance"][0]
    assert provenance["function"] == "WaveICA2.0::WaveICA_2.0"
    installed_sha = ro.r(
        "utils::packageDescription('WaveICA2.0')$RemoteSha"
    )
    if len(installed_sha):
        assert provenance["package_source"]["RemoteSha"] == str(
            installed_sha[0]
        )
    final = result.data["WaveICA corrected"]
    expected_blank_columns = dataset.sample_metadata.index[
        dataset.sample_metadata["Sample Type"] == "Blank"
    ]
    pd.testing.assert_frame_equal(
        final.intensity.loc[:, expected_blank_columns],
        dataset.intensity.loc[:, expected_blank_columns],
    )
    assert list(output_dir.glob("Correction_Dashboard*.svg"))
