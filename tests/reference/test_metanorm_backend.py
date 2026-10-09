"""Validate the production Metanorm adapter against the installed R package."""

import numpy as np
import pandas as pd

from pimqc import MetaboDatasetBuilder
from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.metanorm import MetanormRLOESSCorrector

from .helpers import require_r_package
from .metanorm_reference import _run_metanorm_rloess


def test_production_metanorm_adapter_uses_original_worker() -> None:
    """Compare the production path with independently called original R code."""
    require_r_package("metanorm")
    samples = 20
    values = np.vstack(
        [
            np.linspace(10.0, 20.0, samples),
            np.linspace(20.0, 10.0, samples),
        ]
    )
    frame = pd.DataFrame(
        values,
        index=["F1", "F2"],
        columns=[f"S{i}" for i in range(samples)],
    )
    engine = MetanormRLOESSCorrector(random_state=123)
    corrected = engine.fit_transform(
        frame,
        batch_array=np.array(["B1"] * samples),
        qc_mask=np.ones(samples, dtype=bool),
        order_array=np.arange(samples, dtype=float),
    )
    output, oof = corrected["Metanorm-rLOESS corrected"]
    assert oof is None
    assert output.shape == frame.shape
    assert np.isfinite(output.to_numpy(dtype=float)).all()
    expected_log = _run_metanorm_rloess(
        np.log2(frame),
        np.arange(samples, dtype=float),
        np.array(["B1"] * samples),
        np.array(["QC"] * samples),
        qc_only=True,
    )
    np.testing.assert_allclose(output, np.exp2(expected_log), rtol=1e-12)
    assert output.index.equals(frame.index)
    assert output.columns.equals(frame.columns)
    assert engine.provenance["package"] == "metanorm"
    assert engine.provenance["package_version"]
    assert engine.provenance["function"] == "metanorm::metanormWorker"


def _mixed_batch_data():
    """Create two runs bracketed by QC samples, with biologicals and a Blank."""
    rng = np.random.default_rng(89)
    count = 40
    order = np.tile(np.arange(1.0, count + 1), 2)
    batch = np.repeat(["Run B", "Run A"], count)
    qc = (order % 2 == 1) | (order == count)
    sample_type = np.where(qc, "Pooled QC", "Biological")
    sample_type[3] = "Blank"
    values = (
        np.arange(100.0, 900.0, 100.0)[:, None]
        * (1.0 + 0.003 * order + 0.03 * np.sin(order / 7))
        * np.where(batch == "Run A", 1.12, 1.0)
        * rng.lognormal(0, 0.015, (8, count * 2))
    )
    values[0, 5] = np.nan
    values[2, 8] = np.nan
    values[3, 47] = np.nan
    frame = pd.DataFrame(
        values,
        index=[f"Compound {i}" for i in range(8)],
        columns=[f"Injection {i}" for i in range(count * 2)],
    )
    return frame, order, batch, qc, sample_type


def test_mixed_batch_adapter_matches_original_and_preserves_missingness():
    """Preserve orientation, labels and existing NAs on a mixed study design."""
    require_r_package("metanorm")
    frame, order, batch, qc, sample_type = _mixed_batch_data()
    engine = MetanormRLOESSCorrector(random_state=123)
    corrected = engine.fit_transform(
        frame, batch, qc, order, sample_type_array=sample_type
    )["Metanorm-rLOESS corrected"][0]
    expected_log = _run_metanorm_rloess(
        np.log2(frame),
        order,
        batch,
        np.where(qc, "QC", sample_type),
        qc_only=True,
    )
    np.testing.assert_allclose(
        corrected, np.exp2(expected_log), rtol=1e-12, equal_nan=True
    )
    assert corrected.index.equals(frame.index)
    assert corrected.columns.equals(frame.columns)
    np.testing.assert_array_equal(corrected.isna(), frame.isna())
    assert "Blank" in engine.provenance["parameters"]["type"]
    assert "Pooled QC" in engine.provenance["parameters"]["input_type"]


def test_full_signal_corrector_r_stage_records_unavailable_oof(tmp_path):
    """Export and plot real R correction without interpreting fits as OOF."""
    require_r_package("metanorm")
    frame, order, batch, qc, sample_type = _mixed_batch_data()
    metadata = pd.DataFrame(
        {
            "Sample Name": frame.columns,
            "Sample Type": np.where(qc, "QC", sample_type),
            "Batch": batch,
            "Inject Order": order,
        }
    )
    metadata["Sample Type"] = metadata["Sample Type"].replace(
        {"Biological": "Sample"}
    )
    dataset = MetaboDatasetBuilder(metadata, frame).run_build().data
    result = SignalCorrector(
        dataset,
        base_est="Metanorm-rLOESS",
        implementation="r",
    ).run_signal_correction(output_dir=str(tmp_path / "correction"))
    selection = result.audit.metrics["selection"]
    assert selection["implementation"] == "r"
    assert not selection["is_auto"]
    assert selection["selected_score"] is None
    assert selection["validation"]["status"] == "unavailable"
    assert selection["validation"]["evaluation_basis"] == "full_model"
    assert selection["validation"]["qc_value_coverage"] == 0.0
    candidate = result.audit.candidate_results["Metanorm-rLOESS"]
    assert not candidate["stage_oof_dfs"]
    assert candidate["final_rsd_oof"] is None
    assert np.isfinite(candidate["auto_score"])
    final = result.data["Metanorm-rLOESS corrected"]
    assert final.intensity.index.equals(dataset.intensity.index)
    assert final.intensity.columns.equals(dataset.intensity.columns)
    assert list((tmp_path / "correction").glob("Correction_Dashboard*.svg"))
