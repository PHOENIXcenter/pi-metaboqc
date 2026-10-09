"""Validate strict inputs and original-R routing for RUV-III."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction.ruv_r import RUVIIIRCorrector
from pimqc.processing.r_backend import RBackendUnavailable


def _input():
    """Generate complete measurements with informative QC residuals."""
    frame = pd.DataFrame(
        np.random.default_rng(14).lognormal(4, 0.2, (8, 10)),
        index=[f"F{i}" for i in range(8)],
        columns=[f"S{i}" for i in range(10)],
    )
    qc = np.array([True] * 4 + [False] * 6)
    blank = np.array([False] * 9 + [True])
    return frame, qc, blank, pd.Index(["F0", "F1", "F2"])


def test_ruv_r_module_import_does_not_initialize_r():
    """Optional adapters must leave Python-only installations usable."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pimqc.processing.correction.ruv_r; "
            "assert not any(key.startswith('rpy2') for key in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_ruv_r_dependency_error_propagates_without_fallback(monkeypatch):
    """Unavailable optional R packages cannot choose the native algorithm."""
    frame, qc, blank, controls = _input()

    def fail(*args, **kwargs):
        raise RBackendUnavailable("ruv is absent")

    monkeypatch.setattr(
        "pimqc.processing.correction.ruv_r.run_r_function", fail
    )
    with pytest.raises(RBackendUnavailable, match="ruv is absent"):
        RUVIIIRCorrector(k=1).fit_transform(frame, qc, controls, blank)


def test_ruv_r_calls_original_and_preserves_blanks(monkeypatch):
    """No native correction, pseudoprediction or Blank adjustment is used."""
    frame, qc, blank, controls = _input()
    frame.iloc[0, -1] = np.nan
    calls = []

    def fake(data, **kwargs):
        calls.append((data.copy(), kwargs))
        return data, {"implementation": "r", "parameters": kwargs["parameters"]}

    monkeypatch.setattr(
        "pimqc.processing.correction.ruv_r.run_r_function", fake
    )
    engine = RUVIIIRCorrector(k=2, random_state=19)
    result, oof = engine.fit_transform(frame, qc, controls, blank)[
        "RUV corrected"
    ]
    assert oof is None
    pd.testing.assert_frame_equal(result.iloc[:, -1:], frame.iloc[:, -1:])
    np.testing.assert_allclose(result.iloc[:, :-1], frame.iloc[:, :-1])
    source, call = calls[0]
    np.testing.assert_allclose(source, np.log2(frame.iloc[:, :-1] + 1))
    assert call["function"] == "ruv::RUVIII"
    assert call["parameters"]["replicate_codes"] == [0, 0, 0, 0, 1, 2, 3, 4, 5]
    assert call["parameters"]["controls"] == [True] * 3 + [False] * 5
    assert call["parameters"]["average"] is False
    assert call["seed"] == 19
    assert engine.provenance["blank_sample_ids"] == ["S9"]
    json.dumps(engine.provenance, allow_nan=False)


@pytest.mark.parametrize("value", [np.inf, -np.inf, -1.0])
def test_ruv_r_rejects_invalid_fitting_measurements(value):
    """Infinity and negative input cannot enter the temporary adapter."""
    frame, qc, blank, controls = _input()
    frame.iloc[0, 1] = value
    with pytest.raises(ValueError, match="nonnegative"):
        RUVIIIRCorrector(k=1).fit_transform(frame, qc, controls, blank)


@pytest.mark.parametrize("k", [0, -1, 1.2, True, 4])
def test_ruv_r_rejects_invalid_k(k):
    """Do not silently cap k or use an unsupported maximum-rank default."""
    frame, qc, blank, controls = _input()
    with pytest.raises(ValueError, match="k"):
        RUVIIIRCorrector(k=k).fit_transform(frame, qc, controls, blank)


def test_ruv_r_rejects_degenerate_controls_and_unknown_ids():
    """Controls must be real features with informative replicate contrasts."""
    frame, qc, blank, controls = _input()
    with pytest.raises(ValueError, match="Unknown"):
        RUVIIIRCorrector(k=1).fit_transform(
            frame, qc, pd.Index(["not-a-feature"]), blank
        )
    frame.loc[controls] = 100.0
    with pytest.raises(ValueError, match="support"):
        RUVIIIRCorrector(k=1).fit_transform(frame, qc, controls, blank)


def test_ruv_r_temporarily_fills_log_medians_and_restores_nan(monkeypatch):
    """Real non-Blank observations define medians on the fitted log scale."""
    frame, qc, blank, controls = _input()
    frame.iloc[0, 1] = np.nan
    frame.iloc[3, 5] = np.nan
    frame.iloc[0, -1] = 1e12
    frame.iloc[2, -1] = np.nan
    original = frame.copy(deep=True)
    seen = []

    def fake(data, **kwargs):
        seen.append(data.copy())
        return data + 0.1, {}

    monkeypatch.setattr(
        "pimqc.processing.correction.ruv_r.run_r_function", fake
    )
    engine = RUVIIIRCorrector(k=1)
    output, _ = engine.fit_transform(frame, qc, controls, blank)[
        "RUV corrected"
    ]
    expected = np.log1p(frame.loc[:, ~blank]) / np.log(2.0)
    expected = expected.T.fillna(expected.median(axis=1)).T
    np.testing.assert_allclose(seen[0], expected)
    np.testing.assert_array_equal(output.isna(), frame.isna())
    observed = ~frame.loc[:, ~blank].isna().to_numpy()
    corrected = np.exp2(expected.to_numpy() + 0.1) - 1.0
    np.testing.assert_allclose(
        output.loc[:, ~blank].to_numpy()[observed], corrected[observed]
    )
    pd.testing.assert_frame_equal(output.loc[:, blank], frame.loc[:, blank])
    pd.testing.assert_frame_equal(frame, original)
    assert engine.provenance["missing_input_adapter"][
        "temporary_filled_cells"
    ] == 2
    json.dumps(engine.provenance, allow_nan=False)


def test_ruv_r_missing_controls_cannot_fake_replicate_support(monkeypatch):
    """Biological observations and their medians cannot create QC contrasts."""
    frame, qc, blank, controls = _input()
    frame.loc[controls, frame.columns[qc]] = np.nan
    frame.loc[controls, frame.columns[0]] = [10.0, 20.0, 30.0]
    with pytest.raises(ValueError, match="observed.*support"):
        RUVIIIRCorrector(k=1).fit_transform(frame, qc, controls, blank)


@pytest.mark.parametrize(
    "groups, message",
    [
        (["A"] * 3, "cover all samples"),
        ([None] + ["A"] * 9, "missing"),
        ([1] + ["A"] * 9, "mixed types"),
        ([" "] + ["A"] * 9, "must not be empty"),
        ([np.inf] + [1.0] * 9, "finite"),
        (list(range(10)), "degrees of freedom"),
    ],
)
def test_ruv_r_validates_explicit_replicate_design(groups, message):
    """Malformed or non-replicated designs cannot reach RUVIII."""
    frame, qc, blank, controls = _input()
    with pytest.raises(ValueError, match=message):
        RUVIIIRCorrector(k=1).fit_transform(
            frame, qc, controls, blank, replicate_groups=groups
        )


def test_ruv_r_explicit_design_is_used_and_series_alignment_checked(
    monkeypatch,
):
    """Real technical replicate pairs replace the pooled-QC assumption."""
    frame, qc, blank, controls = _input()
    groups = pd.Series(
        ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + [None], index=frame.columns
    )
    with pytest.raises(ValueError, match="index must match"):
        RUVIIIRCorrector(k=1).fit_transform(
            frame, qc, controls, blank, groups.iloc[::-1]
        )
    seen = {}

    def fake(data, **kwargs):
        seen.update(kwargs["parameters"])
        return data, {}

    monkeypatch.setattr(
        "pimqc.processing.correction.ruv_r.run_r_function", fake
    )
    engine = RUVIIIRCorrector(k=1)
    engine.fit_transform(frame, qc, controls, blank, groups)
    assert seen["replicate_codes"] == [0] * 3 + [1] * 3 + [2] * 3
    assert engine.provenance["replicate_design"].startswith("explicit_")


@pytest.mark.parametrize("log_value", [-1.0, 2000.0])
def test_ruv_r_defers_invalid_back_transform(monkeypatch, log_value):
    """Keep adapter output inspectable; the stage applies missing policy."""
    frame, qc, blank, controls = _input()

    def fake(data, **kwargs):
        assert kwargs["defer_output_domain"] is True
        return data * 0 + log_value, {}

    monkeypatch.setattr(
        "pimqc.processing.correction.ruv_r.run_r_function", fake
    )
    output, _ = RUVIIIRCorrector(k=1).fit_transform(
        frame, qc, controls, blank
    )["RUV corrected"]
    if log_value < 0:
        assert (output.loc[:, ~blank] < 0).all().all()
    else:
        assert np.isinf(output.loc[:, ~blank]).all().all()
