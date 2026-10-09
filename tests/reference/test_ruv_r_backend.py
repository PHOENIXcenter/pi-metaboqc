"""Compare the production RUV-III adapter with independent official calls."""

import json
import threading

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pimqc import MetaboDatasetBuilder
from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.ruv_r import RUVIIIRCorrector
from pimqc.processing.r_backend import RBackendError
from pimqc.serialization import read_audit_payload, write_audit_payload

from .helpers import require_r_package


def _data():
    """Create identifiable unwanted variation and a held-out Blank."""
    rng = np.random.default_rng(35)
    count, features = 12, 20
    drift = np.linspace(-0.2, 0.2, count)
    loading = np.tile([-1.0, 1.0], features // 2)
    logged = 8.0 + loading[:, None] * drift
    logged += rng.normal(0, 0.003, logged.shape)
    logged[6:, 6:] += 0.3
    frame = pd.DataFrame(
        np.exp2(logged) - 1.0,
        index=[f"F{i}" for i in range(features)],
        columns=[f"S{i}" for i in range(count)],
    )
    frame["Blank"] = np.nan
    qc = np.array([True] * 6 + [False] * 7)
    blank = np.array([False] * count + [True])
    return frame, qc, blank, pd.Index([f"F{i}" for i in range(6)])


def _direct(frame, codes, controls):
    """Execute the exported original function without the production adapter."""
    values = np.log2(frame.to_numpy(dtype=float) + 1.0).T
    matrix = ro.r["matrix"](
        ro.FloatVector(values.ravel(order="F")),
        nrow=values.shape[0],
        ncol=values.shape[1],
    )
    expected = ro.r(
        """
        function(Y, groups, ctl) {
            M <- matrix(0, nrow=length(groups), ncol=max(groups) + 1)
            M[cbind(seq_along(groups), groups + 1L)] <- 1
            ruv::RUVIII(Y, M, ctl, k=1, eta=NULL,
                        include.intercept=TRUE, average=FALSE,
                        fullalpha=NULL, return.info=FALSE, inputcheck=TRUE)
        }
        """
    )(matrix, ro.IntVector(codes), ro.BoolVector(controls))
    return np.exp2(np.asarray(expected, dtype=float).T) - 1.0


@pytest.mark.parametrize("design", ["pooled_qc", "explicit", "all_qc"])
def test_ruv_r_matches_original_and_keeps_blank(design):
    """Keep direct R outputs, orientation and declared replicate mapping."""
    require_r_package("ruv")
    frame, qc, blank, controls = _data()
    groups = (
        np.repeat(np.arange(4), 3).tolist() + [None]
        if design == "explicit"
        else None
    )
    if design == "all_qc":
        qc[:-1] = True
    engine = RUVIIIRCorrector(k=1, random_state=71)
    output, oof = engine.fit_transform(frame, qc, controls, blank, groups)[
        "RUV corrected"
    ]
    expected = _direct(
        frame.iloc[:, :-1],
        engine.provenance["parameters"]["replicate_codes"],
        frame.index.isin(controls).tolist(),
    )
    np.testing.assert_allclose(output.iloc[:, :-1], expected, rtol=1e-12)
    pd.testing.assert_frame_equal(output.iloc[:, -1:], frame.iloc[:, -1:])
    assert output.index.equals(frame.index)
    assert output.columns.equals(frame.columns)
    assert oof is None
    assert engine.provenance["function"] == "ruv::RUVIII"
    assert engine.provenance["package_version"]
    json.dumps(engine.provenance, allow_nan=False)


def test_ruv_r_missing_matches_independent_prefilled_original():
    """The unmodified R function receives temporary observed log medians."""
    require_r_package("ruv")
    frame, qc, blank, controls = _data()
    frame = frame.iloc[:8, list(range(8)) + [12]].copy()
    qc = qc[list(range(8)) + [12]]
    blank = blank[list(range(8)) + [12]]
    frame.iloc[0, 1] = np.nan
    frame.iloc[6, 7] = np.nan
    frame.iloc[1, -1] = 1e12
    original = frame.copy(deep=True)
    engine = RUVIIIRCorrector(k=1)
    output, _ = engine.fit_transform(frame, qc, controls, blank)[
        "RUV corrected"
    ]
    logged = np.log1p(frame.loc[:, ~blank]) / np.log(2.0)
    logged = logged.T.fillna(logged.median(axis=1)).T
    prefilled = np.exp2(logged) - 1.0
    expected = _direct(
        prefilled,
        engine.provenance["parameters"]["replicate_codes"],
        frame.index.isin(controls).tolist(),
    )
    expected[frame.loc[:, ~blank].isna().to_numpy()] = np.nan
    np.testing.assert_allclose(
        output.loc[:, ~blank], expected,
        rtol=1e-10, atol=1e-10, equal_nan=True,
    )
    pd.testing.assert_frame_equal(output.loc[:, blank], frame.loc[:, blank])
    pd.testing.assert_frame_equal(frame, original)
    json.dumps(engine.provenance, allow_nan=False)


def test_ruv_r_rejects_worker_thread():
    """The production adapter uses the main-thread R execution guard."""
    require_r_package("ruv")
    frame, qc, blank, controls = _data()
    errors = []

    def run():
        try:
            RUVIIIRCorrector(k=1).fit_transform(frame, qc, controls, blank)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RBackendError)
    assert "main Python thread" in str(errors[0])


def test_original_ruv_respects_supplied_log_scale():
    """Changing log base is a rescaling, not another internal log step."""
    require_r_package("ruv")
    frame, qc, blank, controls = _data()
    values = np.log2(frame.loc[:, ~blank].to_numpy(dtype=float) + 1).T
    groups = np.arange(len(values), dtype=int)
    groups[qc[~blank]] = -1
    codes, _ = pd.factorize(groups, sort=False)
    direct = ro.r(
        """
        function(Y, groups, ctl) {
            M <- 1 * outer(groups, unique(groups), "==")
            ruv::RUVIII(Y, M, ctl, k=1, eta=NULL,
                        include.intercept=TRUE, average=FALSE,
                        fullalpha=NULL, return.info=FALSE, inputcheck=TRUE)
        }
        """
    )

    def calculate(matrix):
        """Call the installed function on the supplied numerical scale."""
        r_matrix = ro.r["matrix"](
            ro.FloatVector(matrix.ravel(order="F")),
            nrow=matrix.shape[0], ncol=matrix.shape[1],
        )
        return np.asarray(
            direct(
                r_matrix, ro.IntVector(codes),
                ro.BoolVector(frame.index.isin(controls)),
            ),
            dtype=float,
        )

    log2_result = calculate(values)
    natural_log_result = calculate(values * np.log(2))
    np.testing.assert_allclose(
        natural_log_result / np.log(2), log2_result,
        rtol=1e-8, atol=1e-8,
    )


def test_ruv_r_stage_audit_roundtrip(tmp_path, monkeypatch):
    """Preserve original provider/design across the stage boundary."""
    require_r_package("ruv")
    frame, qc, blank, controls = _data()
    metadata = pd.DataFrame(
        {
            "Sample Name": frame.columns,
            "Sample Type": np.where(
                blank, "Blank", np.where(qc, "QC", "Sample")
            ),
            "Batch": ["B1"] * len(frame.columns),
            "Inject Order": np.arange(len(frame.columns)),
            "Technical replicate": ["Pooled"] * 6 + list("ABCDEF") + [None],
        }
    )
    dataset = MetaboDatasetBuilder(metadata, frame).run_build().data
    processor = SignalCorrector(
        dataset,
        implementation="r",
        base_est="RUV-III",
        ruv_k=1,
        ruv_control_features=controls.tolist(),
        ruv_replicate_column="Technical replicate",
    )
    result = processor.run_signal_correction(output_dir=str(tmp_path / "plots"))
    repeated = processor.run_signal_correction()
    np.testing.assert_allclose(
        result.data["RUV corrected"].intensity,
        repeated.data["RUV corrected"].intensity,
        equal_nan=True,
    )
    assert processor.config["base_est"] == "RUV-III"
    assert processor.config["implementation"] == "r"
    figures = list((tmp_path / "plots").glob("Correction_Dashboard*.svg"))
    assert len(figures) == 1
    assert "<svg" in figures[0].read_text(encoding="utf-8")
    selection = result.audit.metrics["selection"]
    assert selection["implementation"] == "r"
    assert selection["selected_score"] is None
    assert selection["validation"]["status"] == "unavailable"
    assert selection["validation"]["evaluation_basis"] == "full_model"
    provenance = selection["implementation_provenance"][0]
    assert provenance["replicate_design"].startswith("explicit_")
    assert provenance["function"] == "ruv::RUVIII"
    assert result.audit.selected_prediction is None
    output = next(reversed(result.data.values()))
    assert output.intensity.columns.equals(dataset.intensity.columns)
    path = write_audit_payload(result.audit, tmp_path / "ruv-audit")

    def no_r():
        raise AssertionError("Stored RUV audits must not need R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    loaded = read_audit_payload(path)
    assert loaded.metrics["selection"] == selection
