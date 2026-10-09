"""Check Python WaveICA's temporary input and restored missing positions."""

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction.waveica import WaveICA2Corrector


def test_waveica_missing_median_excludes_blank_proxy(monkeypatch):
    """A Blank interpolation must not change the non-Blank median."""
    frame = pd.DataFrame(
        [[1, np.nan, 9, 9000, 11, 13], [2, 4, 6, np.nan, 8, 10]],
        columns=list("abcdef"), dtype=float,
    )
    original = frame.copy(deep=True)
    order = np.array([6, 2, 5, 1, 4, 3], dtype=float)
    blanks = np.array([False, False, False, True, False, False])
    engine = WaveICA2Corrector(n_levels=1, cutoff=1)
    captured = {}
    decompose = engine._decompose

    def capture(values):
        captured["values"] = values.copy()
        return decompose(values)

    monkeypatch.setattr(engine, "_decompose", capture)
    result, _ = engine.fit_transform(frame, order, blank_mask=blanks)[
        "WaveICA 2.0"
    ]
    sorted_positions = np.argsort(order, kind="stable")
    missing_position = int(np.flatnonzero(sorted_positions == 1)[0])
    assert captured["values"][missing_position, 0] == 10
    assert np.isfinite(captured["values"]).all()
    np.testing.assert_array_equal(result.isna(), frame.isna())
    pd.testing.assert_frame_equal(frame, original)
    assert "missing_input_adapter" in engine.diagnostics


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_waveica_rejects_unsupported_feature_before_decomposition(value):
    """No zero fallback is permitted for a feature with no fit reference."""
    frame = pd.DataFrame([[value] * 12, list(range(12))], dtype=float)
    with pytest.raises(ValueError):
        WaveICA2Corrector().fit_transform(frame, np.arange(12))
