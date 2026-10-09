"""Guard the original SERRF source boundary without an installed R runtime."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pimqc.processing.correction import serrf_r


def _inputs():
    """Provide complete variable measurements and explicit sample roles."""
    rng = np.random.default_rng(30)
    data = pd.DataFrame(rng.lognormal(7, 0.15, (12, 17)))
    batches = np.array(["B1"] * 8 + ["B2"] * 8 + ["B1"])
    qc = np.array([True, False] * 8 + [False])
    blank = np.array([False] * 16 + [True])
    order = np.arange(17)
    return data, batches, qc, order, blank


def _mock_source(monkeypatch):
    """Isolate source validation from dispatch tests that do not load R."""
    monkeypatch.setattr(
        serrf_r,
        "_canonical_source",
        lambda path: ("serrfR <- function() NULL", "verified", Path(path)),
    )


def test_unverified_source_fails_before_r_initialization(tmp_path, monkeypatch):
    """Untrusted code cannot reach R parsing or execution."""
    source = tmp_path / "app.R"
    source.write_text("stop('must not execute')", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified source must not start R")

    monkeypatch.setattr(serrf_r, "_load_r", forbidden)
    with pytest.raises(ValueError, match="checksum"):
        serrf_r.prepare_serrf_source(source, tmp_path / "serrfR.R")
    assert not (tmp_path / "serrfR.R").exists()


def test_extraction_refuses_to_overwrite_any_source(tmp_path, monkeypatch):
    """The extraction helper must preserve existing local files."""
    _mock_source(monkeypatch)
    output = tmp_path / "existing.R"
    output.write_text("user content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        serrf_r.prepare_serrf_source("app.R", output)
    assert output.read_text(encoding="utf-8") == "user content"


def test_dispatch_preserves_blank_and_records_original_identity(monkeypatch):
    """Retain Blank measurements and distinguish original source from ranger."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    data.iloc[:, -1] = np.nan
    captured = {}

    def run(frame, **kwargs):
        captured.update(kwargs)
        assert frame.shape == (12, 16)
        return frame * 2, {"package": "ranger", "package_version": "test"}

    monkeypatch.setattr(serrf_r, "run_r_function", run)
    engine = serrf_r.SERRFRCorrector(
        source_path="app.R", random_state=13, n_correlated_features=4,
        cv_folds=0,
    )
    result, oof = engine.fit_transform(
        data, batches, qc, order, blank
    )["SERRF corrected"]
    assert oof is None
    pd.testing.assert_frame_equal(result.iloc[:, :-1], data.iloc[:, :-1] * 2)
    pd.testing.assert_series_equal(result.iloc[:, -1], data.iloc[:, -1])
    assert captured["parameters"]["original_model_seed"] == 1
    assert captured["parameters"]["effective_model_seed"] == 13
    assert "runtime AST" in captured["parameters"]["model_seed_policy"]
    assert captured["seed"] == 13
    assert captured["function"] == "Shiny-SERRF::serrfR (source adapter)"
    assert engine.provenance["upstream_source"]["commit"] == (
        serrf_r.SERRF_UPSTREAM_COMMIT
    )
    assert "no OOF" in engine.provenance["evaluation_basis"]
    adaptation = engine.provenance["runtime_seed_adaptation"]
    assert adaptation["original_model_seed"] == 1
    assert adaptation["effective_model_seed"] == 13
    assert adaptation["source_file_modified"] is False
    assert "extract(source_text, params$effective_model_seed)" in captured["code"]
    assert "source(" not in captured["code"]
    assert "library(" not in captured["code"]
    assert "install.packages" not in captured["code"]


@pytest.mark.parametrize("invalid", [np.inf, -np.inf, -1.0])
def test_original_serrf_rejects_invalid_active_data(invalid, monkeypatch):
    """Do not make implicit repairs to unsupported raw intensities."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    data.iloc[0, 0] = invalid
    engine = serrf_r.SERRFRCorrector(source_path="app.R", random_state=13)
    with pytest.raises(ValueError, match="nonnegative"):
        engine.fit_transform(data, batches, qc, order, blank)


@pytest.mark.parametrize(
    "kind", ["missing", "zero", "missing_sample", "zero_role"]
)
def test_original_serrf_passes_missing_and_zero_unchanged(kind, monkeypatch):
    """Only upstream repairs input; public output restores NA but not zero."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    if kind == "missing":
        data.iloc[0, 1] = np.nan
    elif kind == "zero":
        data.iloc[0, 1] = 0.0
    elif kind == "missing_sample":
        data.iloc[:, 1] = np.nan
    else:
        data.loc[0, qc] = 0.0
    before = data.copy(deep=True)
    called = []

    def run(frame, **kwargs):
        pd.testing.assert_frame_equal(frame, data.loc[:, ~blank])
        assert kwargs["allow_all_missing_input"] is True
        called.append(True)
        repaired = (frame * 2).fillna(999.0).mask(frame == 0, 123.0)
        # A new failure must not revert to the original observation either.
        repaired.iloc[1, 2] = np.nan
        return repaired, {"package": "ranger"}

    monkeypatch.setattr(serrf_r, "run_r_function", run)
    engine = serrf_r.SERRFRCorrector(
        source_path="app.R", random_state=13, cv_folds=0,
    )
    output, _ = engine.fit_transform(
        data, batches, qc, order, blank
    )["SERRF corrected"]
    expected = (data * 2).mask(data == 0, 123.0)
    expected.iloc[1, 2] = np.nan
    expected.loc[:, blank] = data.loc[:, blank]
    pd.testing.assert_frame_equal(output, expected)
    pd.testing.assert_frame_equal(data, before)
    assert called == [True]
    assert engine.provenance["input_missing_count"] == int(
        data.loc[:, ~blank].isna().sum().sum()
    )
    assert engine.provenance["input_zero_count"] == int(
        (data.loc[:, ~blank] == 0).sum().sum()
    )
    assert "original SERRF random repairs" in engine.provenance[
        "input_repair_policy"
    ]
    assert "NA-safe runtime subscripts" in engine.provenance[
        "input_repair_policy"
    ]
    assert engine.provenance["runtime_fixes"]
    assert engine.provenance["upstream_source"]["source_file_modified"] is False
    assert engine.provenance["output_missing_policy"] == (
        "restore_original_missing_positions"
    )
    assert engine.diagnostics["restored_missing_count"] == (
        engine.provenance["input_missing_count"]
    )
    assert output.attrs["serrf_diagnostics"] == engine.diagnostics


@pytest.mark.parametrize("value", [0, 1, True, 2.5])
def test_original_serrf_rejects_unsupported_predictor_counts(value):
    """Predictor counts must retain dimensions in original R indexing."""
    with pytest.raises(ValueError, match="at least 2"):
        serrf_r.SERRFRCorrector(
            source_path="app.R", random_state=1, n_correlated_features=value
        )


def test_original_serrf_guards_unbounded_predictor_search(monkeypatch):
    """Prevent the upstream loop from requesting too many predictors."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    engine = serrf_r.SERRFRCorrector(
        source_path="app.R", random_state=1, n_correlated_features=12
    )
    with pytest.raises(ValueError, match="more features"):
        engine.fit_transform(data, batches, qc, order, blank)


def test_original_serrf_rejects_constant_role_features(monkeypatch):
    """Prevent undefined rank correlations and original scaling errors."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    data.loc[0, qc] = 1.0
    engine = serrf_r.SERRFRCorrector(source_path="app.R", random_state=1)
    with pytest.raises(ValueError, match="constant QC"):
        engine.fit_transform(data, batches, qc, order, blank)


@pytest.mark.parametrize("invalid_mask", ["qc", "blank"])
def test_original_serrf_rejects_numeric_masks(invalid_mask, monkeypatch):
    """Prevent silent conversion of role labels or numbers to booleans."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    if invalid_mask == "qc":
        qc = qc.astype(int)
    else:
        blank = blank.astype(int)
    with pytest.raises(ValueError, match="booleans"):
        serrf_r.SERRFRCorrector(
            source_path="app.R", random_state=1
        ).fit_transform(data, batches, qc, order, blank)


def test_original_serrf_rejects_ambiguous_batch_labels(monkeypatch):
    """Distinct Python batch IDs must not become one R factor level."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    batches = np.array([1] * 8 + ["1"] * 9, dtype=object)
    with pytest.raises(ValueError, match="collide"):
        serrf_r.SERRFRCorrector(
            source_path="app.R", random_state=1
        ).fit_transform(data, batches, qc, order, blank)


def test_failure_clears_prior_provenance(monkeypatch):
    """A failed rerun must not expose an earlier call's source record."""
    _mock_source(monkeypatch)
    data, batches, qc, order, blank = _inputs()
    engine = serrf_r.SERRFRCorrector(source_path="app.R", random_state=1)
    engine.provenance = {"old": "stale"}
    data.iloc[0, 0] = -1.0
    with pytest.raises(ValueError):
        engine.fit_transform(data, batches, qc, order, blank)
    assert engine.provenance == {}
