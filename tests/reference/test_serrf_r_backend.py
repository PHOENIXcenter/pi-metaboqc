"""Compare the runtime-adapted source with independent pinned serrfR calls.

Set PIMQC_SERRF_R_SOURCE to independently obtained pinned app.R or its verified
standalone extraction. The test does not download or redistribute that source
and never evaluates the Shiny application. A source-free CI run skips this
optional reference comparison.
"""

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from pimqc import MetaboDatasetBuilder
from pimqc.processing.correction import SignalCorrector
from pimqc.processing.correction.serrf_r import (
    SERRF_EXTRACTED_SOURCE_SHA256,
    SERRFRCorrector,
    _EXTRACT_FUNCTION,
    _canonical_source,
    prepare_serrf_source,
)

from .helpers import require_r_package


@pytest.fixture
def source_path():
    """Use an explicitly provided source, never an implicit download."""
    path = os.environ.get("PIMQC_SERRF_R_SOURCE")
    if not path:
        pytest.skip("Set PIMQC_SERRF_R_SOURCE to pinned upstream app.R.")
    require_r_package("ranger")
    _canonical_source(path)
    return path


def _data():
    """Create two complete batches and an unchanged Blank column."""
    rng = np.random.default_rng(73)
    data = pd.DataFrame(
        rng.lognormal(7, 0.20, (12, 17)),
        index=[f"F{i}" for i in range(12)],
        columns=[f"S{i}" for i in range(17)],
    )
    batches = np.array(["B1"] * 8 + ["B2"] * 8 + ["B1"])
    qc = np.array([True, False] * 8 + [False])
    blank = np.array([False] * 16 + [True])
    return data, batches, qc, np.arange(17), blank


def _direct_original(
    source, data, batches, qc, order, *, extracted=False, fix_missing=False,
    seed=37,
):
    """Independently adapt only the seed and optional missing subscripts."""
    if not extracted:
        source = "\n".join(source.splitlines()[400:699])
        extracted = True
    source, seed_count = re.subn(
        r"set\.seed\(1\)", f"set.seed({seed}L)", source,
    )
    assert seed_count == 1
    if fix_missing:
        # Independent, explicit repair, not the production AST patcher.
        block = "all[j, batch. %in% current_batch]"
        source, count = re.subn(
            r"all\[j,\s*batch\.\s*%in%\s*current_batch\]\s*==\s*0",
            f"(!is.na({block}) & {block} == 0)",
            source,
        )
        assert count == 2
    direct = ro.r("""
    function(source, mat, batch, qc, order, extracted, seed) {
        old_threads <- Sys.getenv("R_RANGER_NUM_THREADS", unset=NA_character_)
        old_options <- options(ranger.num.threads=1L)
        Sys.setenv(R_RANGER_NUM_THREADS="1")
        on.exit({
            options(old_options)
            if (is.na(old_threads)) Sys.unsetenv("R_RANGER_NUM_THREADS")
            else Sys.setenv(R_RANGER_NUM_THREADS=old_threads)
        }, add=TRUE)
        lines <- strsplit(source, "\n", fixed=TRUE)[[1]]
        # The standalone verified file contains only comments and one
        # assignment. The app requires its fixed original function boundary.
        definition <- if (extracted) parse(text=source) else
            parse(text=paste(lines[401:699], collapse="\n"))
        stopifnot(length(definition) == 1L)
        assignment <- definition[[1]]
        stopifnot(is.call(assignment), length(assignment) == 3L,
                  as.character(assignment[[1]]) %in% c("<-", "="),
                  identical(assignment[[2]], as.name("serrfR")),
                  identical(assignment[[3]][[1]], as.name("function")))
        env <- new.env(parent=asNamespace("stats"))
        env$ranger <- ranger::ranger
        env$boxplot.stats <- grDevices::boxplot.stats
        env$withProgress <- function(message, value, expr, ...) {
            eval(substitute(expr), envir=parent.frame())
        }
        env$incProgress <- function(...) invisible(NULL)
        eval(definition, envir=env)
        arrangement <- c(which(qc), which(!qc))
        mat[is.nan(mat)] <- NA_real_
        RNGkind("Mersenne-Twister", "Inversion", "Rejection")
        set.seed(seed)
        result <- env$serrfR(
            train=mat[, qc], target=mat[, !qc], num=4,
            batch.=factor(batch[arrangement]), time.=order[arrangement],
            sampleType.=ifelse(qc[arrangement], "qc", "sample"),
            minus=FALSE, cl=NULL)
        cbind(result$normed_train, result$normed_target)[,order(arrangement)]
    }
    """)
    matrix = ro.r["matrix"](
        ro.FloatVector(data.to_numpy().ravel(order="F")),
        nrow=len(data), ncol=data.shape[1],
    )
    return np.asarray(
        direct(
            source,
            matrix,
            ro.StrVector(batches.tolist()),
            ro.BoolVector(qc.tolist()),
            ro.FloatVector(order),
            extracted,
            ro.IntVector([seed]),
        )
    )


@pytest.mark.parametrize("seed", [1, 123])
def test_original_serrf_matches_direct_function(source_path, tmp_path, seed):
    """Match independent seed-adapted calls and retain immutable source bytes."""
    data, batches, qc, order, blank = _data()
    source_bytes = Path(source_path).read_bytes()
    source, source_digest, _ = _canonical_source(source_path)
    expected = _direct_original(
        source, data.iloc[:, :-1], batches[:-1], qc[:-1], order[:-1],
        extracted=source_digest == SERRF_EXTRACTED_SOURCE_SHA256,
        seed=seed,
    )
    extracted = prepare_serrf_source(source_path, tmp_path / "serrfR.R")
    for path in (source_path, extracted):
        engine = SERRFRCorrector(
            source_path=path, random_state=seed, n_correlated_features=4,
            cv_folds=0,
        )
        output, oof = engine.fit_transform(
            data, batches, qc, order, blank
        )["SERRF corrected"]
        np.testing.assert_allclose(
            output.iloc[:, :-1].to_numpy(), expected, rtol=1e-12, atol=1e-12
        )
        pd.testing.assert_series_equal(output.iloc[:, -1], data.iloc[:, -1])
        assert oof is None
        assert engine.provenance["package"] == "ranger"
        assert engine.provenance["parameters"]["original_model_seed"] == 1
        assert engine.provenance["parameters"]["effective_model_seed"] == seed
        assert engine.provenance["runtime_seed_adaptation"][
            "effective_model_seed"
        ] == seed
        upstream = engine.provenance["upstream_source"]
        assert upstream["upstream_app_sha256"]
        assert upstream["loaded_source_sha256"] in {
            upstream["upstream_app_sha256"],
            SERRF_EXTRACTED_SOURCE_SHA256,
        }
        assert "original_sha256" not in upstream
        assert "loaded_sha256" not in upstream
        assert "shiny" not in list(ro.r("loadedNamespaces()"))
    assert Path(source_path).read_bytes() == source_bytes
    assert _canonical_source(extracted)[1] == SERRF_EXTRACTED_SOURCE_SHA256
    if seed != 1:
        seed_one = _direct_original(
            source, data.iloc[:, :-1], batches[:-1], qc[:-1], order[:-1],
            extracted=source_digest == SERRF_EXTRACTED_SOURCE_SHA256, seed=1,
        )
        assert not np.allclose(expected, seed_one)


def test_seed_ast_adaptation_changes_only_the_verified_constant():
    """No source evaluation, RNG calls, defaults, or other AST nodes change."""
    check = ro.r("""
    function(extract) {
        source <- paste0(
            "stop('top-level code must not run'); ",
            "serrfR <- function(train, target, num, batch., time., ",
            "sampleType., minus, cl) {set.seed(1); ranger(y ~ ., data=train)}")
        original <- extract(source)
        expected <- original
        expected[[3]][[2]][[2]] <- 123L
        stopifnot(identical(extract(source, 123L), expected))
        stopifnot(identical(extract(source), original))
        TRUE
    }
    """)
    assert bool(check(ro.r(_EXTRACT_FUNCTION))[0])


@pytest.mark.parametrize("body", [
    "NULL", "set.seed(1); set.seed(1)", "set.seed(2)",
    "set.seed(seed=1)", "set.seed(1, kind='Mersenne-Twister')",
])
def test_seed_ast_adaptation_rejects_unexpected_calls(body):
    """An altered seed call must fail instead of broad textual rewriting."""
    source = (
        "serrfR <- function(train, target, num, batch., time., "
        f"sampleType., minus, cl) {{{body}}}"
    )
    with pytest.raises(Exception, match="seed"):
        ro.r(_EXTRACT_FUNCTION)(source, 123)


def test_extraction_does_not_evaluate_application(source_path, tmp_path):
    """Extraction neither sources the app nor pollutes the R global scope."""
    before = set(ro.r("ls(envir=.GlobalEnv, all.names=TRUE)"))
    target = prepare_serrf_source(source_path, tmp_path / "serrfR.R")
    assert set(ro.r("ls(envir=.GlobalEnv, all.names=TRUE)")) == before
    assert "library(shiny)" not in target.read_text(encoding="utf-8")
    assert "shinyApp(" not in target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_serrf_source(source_path, target)


@pytest.mark.parametrize(
    "kind", ["missing", "zero", "missing_sample", "many_na", "na_and_zero"],
)
def test_original_serrf_repair_matches_direct_function(
    source_path, kind, monkeypatch,
):
    """Match an independently repaired R call, then verify NA restoration."""
    from pimqc.processing.correction import serrf_r

    data, batches, qc, order, blank = _data()
    if kind == "missing_sample":
        data.iloc[:, 1] = np.nan
    elif kind in {"many_na", "na_and_zero"}:
        data.iloc[0, [1, 3]] = np.nan
        if kind == "na_and_zero":
            data.iloc[0, 5] = 0.0
    else:
        data.iloc[0, 1] = np.nan if kind == "missing" else 0.0
    source, digest, _ = _canonical_source(source_path)
    expected = _direct_original(
        source, data.iloc[:, :-1], batches[:-1], qc[:-1], order[:-1],
        extracted=digest == SERRF_EXTRACTED_SOURCE_SHA256,
        fix_missing=True,
    )
    raw_outputs = []
    original_runner = serrf_r.run_r_function

    def capture_raw(*args, **kwargs):
        frame, provenance = original_runner(*args, **kwargs)
        raw_outputs.append(frame.to_numpy(copy=True))
        return frame, provenance

    monkeypatch.setattr(serrf_r, "run_r_function", capture_raw)
    engine = SERRFRCorrector(
        source_path=source_path, random_state=37,
        n_correlated_features=4, cv_folds=0,
    )
    output, _ = engine.fit_transform(
        data, batches, qc, order, blank,
    )["SERRF corrected"]
    np.testing.assert_allclose(
        raw_outputs[0], expected, rtol=1e-12, atol=1e-12
    )
    missing = data.iloc[:, :-1].isna().to_numpy()
    expected = expected.copy()
    expected[missing] = np.nan
    np.testing.assert_allclose(
        output.iloc[:, :-1], expected, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_series_equal(output.iloc[:, -1], data.iloc[:, -1])
    assert engine.provenance["restored_missing_count"] == int(missing.sum())


def test_serrf_missing_batch_is_unavailable_not_fabricated(source_path):
    """A feature without a batch reference cannot borrow another batch."""
    data, batches, qc, order, blank = _data()
    data.loc["F0", (batches == "B1") & ~blank] = np.nan
    original = data.copy(deep=True)
    engine = SERRFRCorrector(
        source_path=source_path, random_state=37,
        n_correlated_features=4, cv_folds=3,
    )
    full, oof = engine.fit_transform(
        data, batches, qc, order, blank,
    )["SERRF corrected"]
    assert full.loc["F0", ~blank].isna().all()
    assert oof.loc["F0", ~blank].isna().all()
    assert np.isfinite(full.loc[data.index != "F0", ~blank]).all().all()
    assert np.isfinite(oof.loc[data.index != "F0", qc]).all().all()
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(full.loc[:, blank], data.loc[:, blank])
    assert engine.diagnostics["status"] == "degraded"


def test_missing_qc_response_remains_missing_in_real_oof(source_path):
    """Original full-fit repair does not create held-out validation truth."""
    from pimqc.processing.correction.qc_validation import assign_qc_folds

    data, batches, qc, order, blank = _data()
    folds = assign_qc_folds(batches, qc, 3, 37)
    query = np.flatnonzero((folds == 0) & (batches == "B1"))
    assert query.size == 2
    data.iloc[0, query[0]] = np.nan
    data.iloc[1, query[1]] = np.nan
    engine = SERRFRCorrector(
        source_path=source_path, random_state=37,
        n_correlated_features=2, cv_folds=3,
    )
    full, oof = engine.fit_transform(
        data, batches, qc, order, blank
    )["SERRF corrected"]
    assert np.isnan(full.iloc[0, query[0]])
    assert np.isnan(full.iloc[1, query[1]])
    assert np.isnan(oof.iloc[0, query[0]])
    assert np.isnan(oof.iloc[1, query[1]])
    # Feature 0's invalid query cannot remove support for a later feature.
    assert np.isfinite(oof.iloc[1, query[0]])
    assert np.isfinite(oof.iloc[3, query]).all()
    assert engine.diagnostics["qc_oof_value_coverage"] < 1.0


def test_r_oof_excludes_held_out_response_from_fitted_parameters(source_path):
    """Perturbing a held-out response cannot change its correction factor."""
    from pimqc.processing.correction.qc_validation import assign_qc_folds

    data, batches, qc, order, blank = _data()
    folds = assign_qc_folds(batches, qc, 3, 123)
    held_out = folds == 0
    changed = data.copy(deep=True)
    changed.loc["F0", held_out] *= np.linspace(0.1, 5, held_out.sum())
    outputs = []
    for values in (data, changed):
        engine = SERRFRCorrector(
            source_path=source_path, random_state=123,
            n_correlated_features=4, cv_folds=3,
        )
        full, oof = engine.fit_transform(
            values, batches, qc, order, blank
        )["SERRF corrected"]
        assert engine.diagnostics["qc_oof_value_coverage"] == 1.0
        assert len(engine.provenance["validation"]["fold_provenance"]) == 3
        assert engine.provenance["parameters"]["effective_model_seed"] == 123
        for fold in engine.provenance["validation"]["fold_provenance"]:
            assert fold["seed"] == 123
            assert fold["parameters"]["effective_model_seed"] == 123
        np.testing.assert_array_equal(
            engine.diagnostics["fold_assignments"], folds
        )
        np.testing.assert_allclose(oof.loc[:, ~qc], full.loc[:, ~qc])
        outputs.append((full, oof))
    np.testing.assert_allclose(
        outputs[0][1].loc["F0", held_out] / data.loc["F0", held_out],
        outputs[1][1].loc["F0", held_out] / changed.loc["F0", held_out],
        rtol=1e-12, atol=1e-12,
    )
    assert not np.allclose(
        outputs[0][0].loc["F0", held_out] / data.loc["F0", held_out],
        outputs[1][0].loc["F0", held_out] / changed.loc["F0", held_out],
    )


def test_original_serrf_stage_exports_roundtrip_and_report(
    source_path, tmp_path, monkeypatch
):
    """Execute the R stage with SVG, portable audit and truthful report."""
    from pimqc.reporting import NarrativeStatsReporter
    from pimqc.serialization import read_audit_payload, write_audit_payload
    from tests.unit.test_reporting_contract import _report_input

    data, batches, qc, order, blank = _data()
    metadata = pd.DataFrame(
        {
            "Sample Name": data.columns,
            "Sample Type": np.where(
                blank, "Blank", np.where(qc, "QC", "Sample")
            ),
            "Batch": batches,
            "Inject Order": order,
        }
    )
    dataset = MetaboDatasetBuilder(metadata, data).run_build().data
    result = SignalCorrector(
        dataset,
        base_est="SERRF",
        implementation="r",
        serrf_r_source=source_path,
        serrf_corr_features=4,
        cv_folds=4,
    ).run_signal_correction(output_dir=str(tmp_path / "correction"))
    selection = result.audit.metrics["selection"]
    assert selection["implementation"] == "r"
    assert selection["selected_score"] is None
    assert selection["validation"]["status"] in {"available", "partial"}
    assert selection["validation"]["evaluation_basis"] == "oof"
    assert selection["validation"]["qc_value_coverage"] > 0
    assert result.audit.candidate_results["SERRF"]["stage_oof_dfs"]
    assert list((tmp_path / "correction").glob("Correction_Dashboard*.svg"))
    path = write_audit_payload(result.audit, tmp_path / "audit")

    def no_r(*args, **kwargs):
        raise AssertionError("Saved SERRF evidence must not require R")

    monkeypatch.setattr("pimqc.processing.r_backend._load_r", no_r)
    loaded = read_audit_payload(path)
    assert loaded.metrics["selection"]["implementation_provenance"] == (
        selection["implementation_provenance"]
    )
    report = _report_input()
    report.pipeline_metrics["signal_correction"] = dict(loaded.metrics)
    reporter = NarrativeStatsReporter(base_dir=str(tmp_path))
    reporter.generate_markdown(report, report_folder="report")
    text = (tmp_path / "report" / "Report_Comprehensive.md").read_text(
        encoding="utf-8"
    )
    assert "Shiny-SERRF::serrfR" in text
    assert "ranger" in text
    assert "CV/OOF evaluated:" in text


def test_original_serrf_enters_real_auto_with_held_out_evidence(source_path):
    """The original kernel's validation results enter the real AUTO scorer."""
    require_r_package("WaveICA2.0")
    require_r_package("ruv")
    data, batches, qc, order, blank = _data()
    metadata = pd.DataFrame(
        {
            "Sample Name": data.columns,
            "Sample Type": np.where(
                blank, "Blank", np.where(qc, "QC", "Sample")
            ),
            "Batch": batches,
            "Inject Order": order,
        }
    )
    dataset = MetaboDatasetBuilder(metadata, data).run_build().data
    result = SignalCorrector(
        dataset, base_est="Auto", implementation="r",
        serrf_r_source=source_path, serrf_corr_features=4,
        ruv_control_features=["F0", "F1", "F2", "F3"], ruv_k=1,
        waveica_components=2, cv_folds=3, global_seed=37, n_jobs=1,
    ).run_signal_correction()
    selection = result.audit.metrics["selection"]
    rows = {row["method"]: row for row in selection["candidate_results"]}
    serrf = rows["SERRF"]
    assert serrf["status"] == "ok", serrf
    assert serrf["implementation"] == "r"
    assert serrf["implementation_provenance"][0]["package"] == "ranger"
    assert serrf["validation"]["evaluation_basis"] == "oof"
    assert serrf["validation"]["qc_value_coverage"] == 1.0
    assert np.isfinite(serrf["auto_score"])
    assert selection["requested_implementation"] == "r"
    assert selection["implementation"] == (
        rows[selection["selected_label"]]["implementation"]
    )
