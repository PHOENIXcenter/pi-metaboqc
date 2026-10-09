"""Check the optional original WaveICA adapter without loading R."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction.waveica_r import WaveICARCorrector


def _inputs():
    """Create unsorted, named samples with one excluded Blank."""
    order = np.array([6, 2, 9, 1, 3, 12, 4, 5, 10, 7, 11, 8], dtype=float)
    frame = pd.DataFrame(
        np.arange(36, dtype=float).reshape(3, 12) + 10,
        index=["F 1", "F-2", "F_3"],
        columns=[f"Sample {i}" for i in range(12)],
    )
    blanks = np.zeros(12, dtype=bool)
    blanks[4] = True
    return frame, order, blanks


def test_waveica_r_sort_restore_blank_and_provenance(monkeypatch):
    """Exclude Blanks and restore positions without pretending to predict."""
    frame, order, blanks = _inputs()
    frame.iloc[0, 4] = np.nan
    captured = {}

    def fake(data, **kwargs):
        """Mimic numeric package output and expose the boundary contract."""
        captured.update(kwargs)
        captured["data"] = data.copy()
        return data + 5, {"package": "WaveICA2.0", "parameters": {}}

    monkeypatch.setattr(
        "pimqc.processing.correction.waveica_r.run_r_function", fake
    )
    engine = WaveICARCorrector(n_components=20, cutoff=0.2, alpha=0.3)
    stage, oof = engine.fit_transform(frame, order, blanks)[
        "WaveICA corrected"
    ]
    assert oof is None
    np.testing.assert_array_equal(stage.iloc[:, blanks], frame.iloc[:, blanks])
    np.testing.assert_array_equal(
        stage.iloc[:, ~blanks], frame.iloc[:, ~blanks] + 5
    )
    assert stage.index.equals(frame.index)
    assert stage.columns.equals(frame.columns)
    assert np.isnan(frame.iloc[0, 4])
    assert captured["parameters"]["Injection_Order"] == sorted(
        order[~blanks].tolist()
    )
    assert captured["parameters"]["K"] == 20
    assert captured["parameters"]["alpha"] == 0.3
    assert captured["function"] == "WaveICA2.0::WaveICA_2.0"
    policy = engine.provenance["input_policy"]
    assert policy["effective_component_limit"] == 3
    assert policy["excluded_blank_positions"] == [4]
    assert policy["oof"] == "unavailable"


@pytest.mark.parametrize("value", [True, 1, 0, -2, 2.5])
def test_waveica_r_rejects_invalid_components(value):
    """Reject unsupported K=1 and nonintegral component limits."""
    with pytest.raises(ValueError, match="integer >= 2"):
        WaveICARCorrector(n_components=value)


@pytest.mark.parametrize("name", ["cutoff", "alpha"])
@pytest.mark.parametrize("value", [True, -0.1, 1.1, np.nan, np.inf])
def test_waveica_r_rejects_invalid_fractions(name, value):
    """Reject values outside original package parameters' stated domain."""
    with pytest.raises(ValueError, match=name):
        WaveICARCorrector(**{name: value})


def test_waveica_r_temporarily_fills_and_restores_missing(monkeypatch):
    """Only non-Blank observations define the temporary raw-scale median."""
    frame, order, blanks = _inputs()
    frame.iloc[0, 0] = np.nan
    frame.iloc[0, 4] = 100000
    original = frame.copy(deep=True)
    expected_median = frame.iloc[0, ~blanks].median()
    captured = {}

    def fake(data, **kwargs):
        captured["data"] = data.copy()
        return data + 5, {}

    monkeypatch.setattr(
        "pimqc.processing.correction.waveica_r.run_r_function", fake
    )
    engine = WaveICARCorrector()
    result, _ = engine.fit_transform(frame, order, blanks)[
        "WaveICA corrected"
    ]
    assert captured["data"].loc["F 1", "Sample 0"] == expected_median
    assert np.isfinite(captured["data"].to_numpy()).all()
    assert np.isnan(result.iloc[0, 0])
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result.iloc[:, blanks], frame.iloc[:, blanks])
    assert "missing_input_adapter" in engine.provenance


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_waveica_r_rejects_infinite_fitting_values_before_r(
    monkeypatch, value
):
    """Infinite measurements have no temporary missing-value interpretation."""
    frame, order, blanks = _inputs()
    frame.iloc[0, 0] = value

    def fail(*args, **kwargs):
        """Prevent optional R initialization for invalid data."""
        pytest.fail("R should not be initialized")

    monkeypatch.setattr(
        "pimqc.processing.correction.waveica_r.run_r_function", fail
    )
    with pytest.raises(ValueError):
        WaveICARCorrector().fit_transform(frame, order, blanks)


def test_waveica_r_rejects_batch_local_order_resets():
    """Ambiguous chronology must not be resolved using arbitrary row order."""
    frame, order, blanks = _inputs()
    order[1] = order[0]
    with pytest.raises(ValueError, match="globally unique"):
        WaveICARCorrector().fit_transform(frame, order, blanks)


def test_waveica_r_rejects_insufficient_samples():
    """The unchanged R GAM basis requires at least ten retained samples."""
    frame, order, blanks = _inputs()
    blanks[:4] = True
    with pytest.raises(ValueError, match="at least ten"):
        WaveICARCorrector().fit_transform(frame, order, blanks)


@pytest.mark.parametrize("mask", [[0] * 12, np.zeros(11, dtype=bool)])
def test_waveica_r_rejects_nonboolean_or_misaligned_mask(mask):
    """Invalid Blank metadata cannot silently select a different sample set."""
    frame, order, _ = _inputs()
    with pytest.raises(ValueError, match="aligned boolean"):
        WaveICARCorrector().fit_transform(frame, order, mask)
