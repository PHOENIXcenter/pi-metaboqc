"""Use a checksum-pinned, user-supplied Shiny-SERRF function with adapters.

The upstream repository does not declare a redistribution license. No upstream
function body is bundled here. Parsing extracts only the original ``serrfR``
definition; the Shiny application and its top-level IO are never evaluated.
Execution adapts its fixed per-forest seed and NA-unsafe subscripts in memory;
the verified external source and the standalone extraction stay unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..r_backend import _R_LOCK, _load_r, run_r_function
from .serrf_r_missing import (
    NA_SAFE_SUBSCRIPTS,
    RUNTIME_FIXES,
    supported_serrf_features,
)
from .serrf_r_validation import validate_serrf_qcs


SERRF_UPSTREAM_COMMIT = "fb19f2a80a742dcdd469a9d9bcc1ecef8ee585a2"
SERRF_UPSTREAM_URL = (
    "https://github.com/slfan2013/Shiny-SERRF/blob/"
    f"{SERRF_UPSTREAM_COMMIT}/app.R"
)
SERRF_UPSTREAM_APP_SHA256 = (
    "3e37953afa70edfc4310afebcd58e6cac83f64cc2cbc9686bcf84ae4a87b8551"
)
# This digest covers only the locally generated extraction, not redistributed
# upstream code. It prevents edited standalone sources claiming false identity.
SERRF_EXTRACTED_SOURCE_SHA256 = (
    "a98885117a1f1c374531da5ef175697ac261f4c899523810d0b8c8e2c1705e70"
)

_EXTRACT_FUNCTION = """
function(text, model_seed=NULL) {
    nodes <- parse(text=text, keep.source=FALSE)
    matches <- list()
    walk <- function(node) {
        if (!is.call(node) && !is.expression(node)) return(invisible(NULL))
        if (is.call(node) && length(node) == 3L &&
            (identical(node[[1]], as.name("=")) ||
             identical(node[[1]], as.name("<-"))) &&
            identical(node[[2]], as.name("serrfR"))) {
            definition <- node[[3]]
            if (!is.call(definition) ||
                !identical(definition[[1]], as.name("function"))) {
                stop("serrfR must be a function definition")
            }
            matches[[length(matches) + 1L]] <<- definition
            return(invisible(NULL))
        }
        for (i in seq_along(node)) {
            if (!identical(node[[i]], quote(expr=))) walk(node[[i]])
        }
        invisible(NULL)
    }
    walk(nodes)
    if (length(matches) != 1L) stop("Expected exactly one serrfR definition")
    definition <- matches[[1]]
    expected <- c("train", "target", "num", "batch.", "time.",
                  "sampleType.", "minus", "cl")
    if (!identical(names(definition[[2]]), expected)) {
        stop("Unexpected original serrfR interface")
    }
    if (!is.null(model_seed)) {
        if (length(model_seed) != 1L || !is.numeric(model_seed) ||
            is.na(model_seed) || !is.finite(model_seed) ||
            model_seed < 0 || model_seed > .Machine$integer.max ||
            model_seed != floor(model_seed)) {
            stop("SERRF model seed must be a nonnegative R integer")
        }
        changed <- 0L
        adapt_seed <- function(node) {
            if (!is.call(node) && !is.expression(node)) return(node)
            if (is.call(node) && identical(node[[1]], as.name("set.seed"))) {
                if (!identical(node, quote(set.seed(1)))) {
                    stop("Unexpected SERRF per-forest seed expression")
                }
                changed <<- changed + 1L
                node[[2]] <- as.integer(model_seed)
                return(node)
            }
            for (i in seq_along(node)) {
                if (!identical(node[[i]], quote(expr=)) && !is.null(node[[i]])) {
                    node[[i]] <- adapt_seed(node[[i]])
                }
            }
            node
        }
        definition <- adapt_seed(definition)
        if (changed != 1L) stop("Expected exactly one SERRF per-forest seed")
    }
    definition
}
"""

_RUN_SOURCE = """
function(mat, params) {
    old_threads <- Sys.getenv("R_RANGER_NUM_THREADS", unset=NA_character_)
    old_options <- options(ranger.num.threads=1L)
    Sys.setenv(R_RANGER_NUM_THREADS="1")
    on.exit({
        options(old_options)
        if (is.na(old_threads)) Sys.unsetenv("R_RANGER_NUM_THREADS")
        else Sys.setenv(R_RANGER_NUM_THREADS=old_threads)
    }, add=TRUE)
    source_text <- __SOURCE_TEXT__
    extract <- __EXTRACT_FUNCTION__
    definition <- extract(source_text, params$effective_model_seed)
    repair_subscripts <- __NA_SAFE_SUBSCRIPTS__
    definition <- repair_subscripts(definition)
    # Only function creation is evaluated, never any application expression.
    env <- new.env(parent=asNamespace("stats"))
    env$ranger <- ranger::ranger
    env$boxplot.stats <- grDevices::boxplot.stats
    env$withProgress <- function(message, value, expr, ...) {
        eval(substitute(expr), envir=parent.frame())
    }
    env$incProgress <- function(...) invisible(NULL)
    adapted <- eval(definition, envir=env)
    qc <- as.logical(params$qc)
    arrangement <- c(which(qc), which(!qc))
    result <- adapted(
        train=mat[, qc, drop=FALSE], target=mat[, !qc, drop=FALSE],
        num=as.integer(params$n_correlated_features),
        batch.=factor(params$batch[arrangement]),
        time.=params$order[arrangement],
        sampleType.=ifelse(qc[arrangement], "qc", "sample"),
        minus=FALSE, cl=NULL
    )
    corrected <- cbind(result$normed_train, result$normed_target)
    corrected[, order(arrangement), drop=FALSE]
}
"""


def _canonical_source(source_path: str | Path) -> tuple[str, str, Path]:
    """Accept only the pinned original or our exact local extraction."""
    path = Path(source_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("SERRF source_path must refer to a regular R file.")
    content = path.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(content).hexdigest()
    if digest not in {
        SERRF_UPSTREAM_APP_SHA256,
        SERRF_EXTRACTED_SOURCE_SHA256,
    }:
        raise ValueError(
            "Unsupported SERRF source checksum. Supply app.R from "
            f"Shiny-SERRF commit {SERRF_UPSTREAM_COMMIT}, or the unchanged "
            "file produced by prepare_serrf_source. No source was executed."
        )
    return content.decode("utf-8"), digest, path


def prepare_serrf_source(
    source_path: str | Path, output_path: str | Path
) -> Path:
    """Extract pinned ``serrfR`` to a new local file without running Shiny.

    Obtain the upstream source independently after reviewing its usage terms.
    This helper downloads/installs nothing, does not overwrite existing files,
    and writes the unmodified body. Execution adapters separately repair
    NA-unsafe subscripts, adapt the per-forest seed to ``random_state``, and
    replace presentation callbacks at runtime.
    R/rpy2 is needed to parse the source; ranger is only needed to execute it.
    """
    source, _, original = _canonical_source(source_path)
    target = Path(output_path).expanduser().resolve()
    if target == original or target.exists():
        raise FileExistsError(f"Refusing to overwrite SERRF source: {target}")
    with _R_LOCK:
        ro = _load_r()
        definition = ro.r(_EXTRACT_FUNCTION)(source)
        lines = list(ro.r["deparse"](definition, width_cutoff=80))
    output = (
        f"# Local extraction from {SERRF_UPSTREAM_URL}\n"
        f"# Upstream app SHA-256: {SERRF_UPSTREAM_APP_SHA256}\n"
        "# No redistribution license has been established.\n"
        "serrfR <- " + "\n".join(lines) + "\n"
    )
    if hashlib.sha256(output.encode("utf-8")).hexdigest() != (
        SERRF_EXTRACTED_SOURCE_SHA256
    ):
        raise ValueError(
            "This R version produced an unsupported extraction format. "
            "Use the verified original app.R directly; no file was written."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of another source.
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(output)
    return target


class SERRFRCorrector:
    """Apply NA-safe Shiny-SERRF, restoring NAs after internal repairs.

    This source variant uses correlated feature predictors and ranger defaults;
    it is not the native Python SERRF implementation. A separate validation
    adapter refits the source-adapted function and predicts held-out QCs in R.
    Blank columns are excluded from fitting and returned unchanged.
    Internal random fills can influence fitting but are not public imputations.
    Runtime indexing/seed adaptations leave the pinned source file unchanged.
    """

    def __init__(
        self,
        *,
        source_path: str | Path,
        random_state: int,
        n_correlated_features: int = 10,
        cv_folds: int = 5,
        cv_strategy: str = "random",
    ) -> None:
        """Record explicit source and settings without initializing R.

        ``cv_strategy`` only changes optional held-out-QC validation; the
        same runtime seed/indexing adaptations apply to full fit and folds.
        """
        if (
            isinstance(n_correlated_features, bool)
            or not isinstance(n_correlated_features, (int, np.integer))
            or n_correlated_features < 2
        ):
            raise ValueError("Original SERRF requires at least 2 predictors.")
        if (
            isinstance(cv_folds, bool)
            or not isinstance(cv_folds, (int, np.integer))
            or (cv_folds != 0 and cv_folds < 2)
        ):
            raise ValueError("SERRF cv_folds must be 0 or an integer >= 2.")
        if cv_strategy not in {"random", "blocked"}:
            raise ValueError("cv_strategy must be random or blocked.")
        self.source_path = source_path
        self.random_state = random_state
        self.n_correlated_features = int(n_correlated_features)
        self.cv_folds = int(cv_folds)
        self.cv_strategy = cv_strategy
        self.provenance: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        blank_mask: np.ndarray | None = None,
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame | None]]:
        """Run the source-adapted function with explicit, audited boundaries."""
        self.provenance = {}
        self.diagnostics = {}
        source, loaded_source_sha256, source_path = _canonical_source(
            self.source_path
        )
        if not isinstance(intensity_df, pd.DataFrame) or intensity_df.empty:
            raise ValueError("Original SERRF requires a nonempty DataFrame.")
        if not intensity_df.index.is_unique or not (
            intensity_df.columns.is_unique
        ):
            raise ValueError("SERRF feature and sample IDs must be unique.")
        values = intensity_df.to_numpy(dtype=float)
        sample_count = values.shape[1]
        batches = np.asarray(batch_array, dtype=object)
        qc = np.asarray(qc_mask)
        order = np.asarray(order_array, dtype=float)
        blank = (
            np.zeros(sample_count, dtype=bool)
            if blank_mask is None else np.asarray(blank_mask)
        )
        if any(
            item.shape != (sample_count,)
            for item in (batches, qc, order, blank)
        ):
            raise ValueError("SERRF metadata must align with every sample.")
        if qc.dtype.kind != "b" or blank.dtype.kind != "b":
            raise ValueError("SERRF QC and Blank masks must contain booleans.")
        if np.any(qc & blank):
            raise ValueError("SERRF QC and Blank masks cannot overlap.")
        active = ~blank
        if not active.any():
            raise ValueError("Original SERRF requires non-Blank samples.")
        selected = values[:, active]
        if np.isinf(selected).any() or (selected < 0).any():
            raise ValueError(
                "Original SERRF requires nonnegative non-Blank intensities "
                "without infinite measurements."
            )
        active_data = intensity_df.loc[:, active]
        input_missing_mask = active_data.isna()
        input_missing_count = int(input_missing_mask.to_numpy().sum())
        input_zero_count = int((selected == 0).sum())
        if input_missing_count or input_zero_count:
            logger.info(
                "Delegating {} missing and {} zero non-Blank measurements "
                "to SERRF internal repair with NA-safe subscript handling",
                input_missing_count, input_zero_count,
            )
        if selected.shape[0] <= self.n_correlated_features:
            raise ValueError(
                "Original SERRF requires more features than "
                "n_correlated_features to avoid its open-ended search."
            )
        if pd.isna(batches[active]).any() or not np.isfinite(
            order[active]
        ).all():
            raise ValueError("SERRF requires finite orders and batch labels.")
        distinct_batches = pd.unique(batches[active])
        if len(set(map(str, distinct_batches))) != len(distinct_batches):
            raise ValueError("SERRF batch labels collide after R conversion.")
        supported, input_support = supported_serrf_features(
            active_data, batches[active],
            np.ones(int(active.sum()), dtype=bool),
            self.n_correlated_features,
        )
        if input_support["unsupported_feature_count"]:
            logger.warning(
                "R SERRF leaves {} unsupported features missing: at least "
                "one fit batch has no observed non-Blank values; no "
                "cross-batch reference is invented",
                input_support["unsupported_feature_count"],
            )
        for batch in distinct_batches:
            for role, mask in (("QC", qc), ("Sample", ~qc)):
                subset = active & (batches == batch) & mask
                if subset.sum() < 3:
                    raise ValueError(
                        f"Original SERRF batch {batch!r} requires at least "
                        f"3 {role} samples."
                    )
                role_values = values[supported][:, subset]
                positive_finite = (
                    np.isfinite(role_values).all(axis=1)
                    & (role_values > 0).all(axis=1)
                )
                if (positive_finite & (np.std(role_values, axis=1) == 0)).any():
                    raise ValueError(
                        f"Original SERRF batch {batch!r} contains constant "
                        f"{role} features; original scaling is undefined."
                    )
                # R order() keeps NA correlations last by default. Each
                # ranking is still a full permutation, so the original
                # search terminates with n_features-1 available predictors.
        parameters = {
            "qc": qc[active].tolist(),
            "batch": batches[active].astype(str).tolist(),
            "order": order[active].tolist(),
            "n_correlated_features": self.n_correlated_features,
            "minus": False,
            "original_model_seed": 1,
            "effective_model_seed": self.random_state,
            "model_seed_policy": "runtime AST adaptation to random_state",
            "ranger_parameters": "unmodified upstream defaults",
            "ranger_num_threads": 1,
        }
        code = _RUN_SOURCE.replace(
            "__SOURCE_TEXT__", json.dumps(source, ensure_ascii=True)
        ).replace("__EXTRACT_FUNCTION__", _EXTRACT_FUNCTION).replace(
            "__NA_SAFE_SUBSCRIPTS__", NA_SAFE_SUBSCRIPTS
        )
        full_started = perf_counter()
        corrected, provenance = run_r_function(
            active_data.loc[supported],
            method="SERRF",
            package="ranger",
            function="Shiny-SERRF::serrfR (source adapter)",
            code=code,
            parameters=parameters,
            seed=self.random_state,
            allow_all_missing_input=True,
            defer_output_domain=True,
            transforms=(
                "raw nonnegative intensities including missing values; "
                "no external log transformation",
                "original QC/sample centering, scaling and batch alignment",
                "original SERRF random repairs with NA-safe subscripts",
                "per-forest set.seed(1) adapted in memory to random_state",
                "unsupported all-missing fit-batch features return missing",
                "adapter restores original NA positions after correction",
                "Shiny progress callbacks replaced with headless callbacks",
                "Blank samples excluded and returned unchanged",
            ),
        )
        full_seconds = perf_counter() - full_started
        # Internal repair is needed for fitting, not a final imputation stage.
        # Keep new failures missing too; never restore observed input values.
        corrected = corrected.reindex(index=active_data.index)
        corrected = corrected.mask(input_missing_mask)
        output_contract = {
            "output_missing_policy": "restore_original_missing_positions",
            "restored_missing_count": input_missing_count,
            **input_support,
            "runtime_fixes": list(RUNTIME_FIXES),
            "runtime_seed_adaptation": {
                "original_model_seed": 1,
                "effective_model_seed": self.random_state,
                "policy": "replace the sole verified set.seed(1) AST constant",
                "source_file_modified": False,
            },
        }
        if input_missing_count:
            logger.info(
                "Restored {} original non-Blank NA positions after R SERRF; "
                "internal random fills are not final imputations",
                input_missing_count,
            )
        self.provenance = {
            **provenance,
            **output_contract,
            "upstream_source": {
                "url": SERRF_UPSTREAM_URL,
                "commit": SERRF_UPSTREAM_COMMIT,
                "upstream_app_sha256": SERRF_UPSTREAM_APP_SHA256,
                "loaded_source_sha256": loaded_source_sha256,
                "local_path": str(source_path),
                "license": "not declared; original source not distributed",
                "extraction": "R AST; only serrfR definition evaluated",
                "source_file_modified": False,
            },
            "evaluation_basis": "full_fit_only; no OOF predictions",
            "input_missing_count": input_missing_count,
            "input_zero_count": input_zero_count,
            "input_repair_policy": (
                "original SERRF random repairs; NA-safe runtime subscripts; "
                "missing and zero inputs passed unchanged; "
                "unsupported features excluded; no adapter imputation"
            ),
        }
        frame = intensity_df.copy(deep=True)
        frame.loc[:, active] = corrected.to_numpy(dtype=float)
        if self.cv_folds == 0:
            self.diagnostics = {
                **output_contract,
                "timing_seconds": {"full_fit": full_seconds, "oof": 0.0},
                "validation": "disabled; source-adapted full fit only",
                "cv_strategy": self.cv_strategy,
                "status": (
                    "degraded"
                    if input_support["unsupported_feature_count"] else "ok"
                ),
            }
            frame.attrs["serrf_diagnostics"] = self.diagnostics
            return {"SERRF corrected": (frame, None)}
        oof_started = perf_counter()
        oof_active, diagnostics, fold_provenance = validate_serrf_qcs(
            intensity_df.loc[:, active],
            corrected,
            source=source,
            extractor=_EXTRACT_FUNCTION,
            parameters=parameters,
            batches=batches[active],
            qc=qc[active],
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            cv_strategy=self.cv_strategy,
        )
        oof_active = oof_active.mask(input_missing_mask)
        diagnostics.update(output_contract)
        # Public assignments and fold indices refer to original input order.
        positions = np.flatnonzero(active)
        assignments = np.full(sample_count, -1, dtype=int)
        assignments[active] = diagnostics["fold_assignments"]
        diagnostics["fold_assignments"] = assignments.tolist()
        for record in diagnostics["folds"]:
            for key in ("training_qc_indices", "held_out_qc_indices"):
                record[key] = positions[record[key]].tolist()
        oof = frame.copy(deep=True)
        oof.loc[:, active] = oof_active.to_numpy(dtype=float)
        self.diagnostics = diagnostics
        diagnostics["timing_seconds"] = {
            "full_fit": full_seconds,
            "oof": perf_counter() - oof_started,
        }
        self.provenance["evaluation_basis"] = (
            "source-adapted full fit plus independently refitted held-out QC "
            "validation; OOF is a pi-metaboqc adapter, not upstream API"
        )
        self.provenance["validation"] = {
            "adapter": diagnostics["validation_backend"],
            "cv_folds": self.cv_folds,
            "cv_strategy": self.cv_strategy,
            "fold_provenance": fold_provenance,
        }
        frame.attrs["serrf_diagnostics"] = diagnostics
        oof.attrs["serrf_diagnostics"] = diagnostics
        return {"SERRF corrected": (frame, oof)}
