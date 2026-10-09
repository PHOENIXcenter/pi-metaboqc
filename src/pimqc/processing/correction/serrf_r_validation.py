"""Held-out QC validation around the pinned, user-supplied SERRF source.

The upstream function has no prediction API. A ranger hook records its fitted
models and QC predictor scales without changing any returned model. Validation
QCs are never supplied to the upstream fit, correlations, or alignment. Their
correction uses R prediction and the fold's fitted QC alignment factor. This
is a pi-metaboqc validation adapter, not an upstream SERRF OOF implementation.
No upstream function body is redistributed in this module.
Shared runtime adapters repair NA-unsafe subscripts and adapt the per-forest
seed to random_state without editing the verified source.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..r_backend import RBackendError, run_r_function
from .qc_validation import assign_qc_folds
from .serrf_r_missing import (
    NA_SAFE_SUBSCRIPTS,
    RUNTIME_FIXES,
    supported_serrf_features,
)


_VALIDATE_SOURCE = """
function(mat, params) {
    old_threads <- Sys.getenv("R_RANGER_NUM_THREADS", unset=NA_character_)
    old_options <- options(ranger.num.threads=1L)
    Sys.setenv(R_RANGER_NUM_THREADS="1")
    on.exit({
        options(old_options)
        if (is.na(old_threads)) Sys.unsetenv("R_RANGER_NUM_THREADS")
        else Sys.setenv(R_RANGER_NUM_THREADS=old_threads)
    }, add=TRUE)
    extract <- __EXTRACT_FUNCTION__
    definition <- extract(__SOURCE_TEXT__, params$effective_model_seed)
    repair_subscripts <- __NA_SAFE_SUBSCRIPTS__
    definition <- repair_subscripts(definition)
    records <- new.env(parent=emptyenv())
    env <- new.env(parent=asNamespace("stats"))
    env$boxplot.stats <- grDevices::boxplot.stats
    env$withProgress <- function(message, value, expr, ...) {
        eval(substitute(expr), envir=parent.frame())
    }
    env$incProgress <- function(...) invisible(NULL)
    env$ranger <- function(...) {
        state <- parent.frame()
        model <- ranger::ranger(...)
        selected <- state$sel_var[!state$train_NA_index]
        selected <- selected[state$good_column]
        qc <- state$train.index_current_batch == "qc"
        predictors <- state$e_current_batch[selected, qc, drop=FALSE]
        response <- state$e_current_batch[state$j, qc]
        key <- paste(state$j, state$current_batch, sep="::")
        records[[key]] <- list(
            model=model, selected=selected,
            centers=rowMeans(predictors),
            scales=apply(predictors, 1L, stats::sd),
            response=response, response_center=mean(response),
            training_data=state$train_data,
            columns=colnames(state$train_data)[-1L]
        )
        model
    }
    adapted <- eval(definition, envir=env)
    training <- as.logical(params$training_qc)
    held_out <- as.logical(params$held_out_qc)
    sample <- !as.logical(params$qc)
    arrangement <- c(which(training), which(sample))
    fitted <- adapted(
        train=mat[, training, drop=FALSE],
        target=mat[, sample, drop=FALSE],
        num=as.integer(params$n_correlated_features),
        batch.=factor(params$batch[arrangement]),
        time.=params$order[arrangement],
        sampleType.=ifelse(training[arrangement], "qc", "sample"),
        minus=FALSE, cl=NULL
    )
    output <- matrix(NA_real_, nrow=nrow(mat), ncol=ncol(mat))
    training_batches <- params$batch[training]
    for (batch in unique(params$batch[held_out])) {
        query <- which(held_out & params$batch == batch)
        fitted_indices <- which(training_batches == batch)
        for (feature in seq_len(nrow(mat))) {
            key <- paste(feature, batch, sep="::")
            if (!exists(key, envir=records, inherits=FALSE)) next
            fit <- records[[key]]
            # Extra predictions run after the upstream fit finishes,
            # so they cannot advance RNG before upstream random repairs.
            baseline_train <- as.numeric(stats::predict(
                fit$model, data=fit$training_data)$predictions)
            baseline_train <- baseline_train + fit$response_center
            ratio <- fit$response / baseline_train
            aligned <- fitted$normed_train[feature, fitted_indices]
            good <- is.finite(baseline_train) & baseline_train > 0 &
                is.finite(ratio) & ratio > 0 &
                is.finite(aligned) & aligned > 0
            if (sum(good) < 2L) next
            # All un-repaired training QCs share one alignment scalar.
            # Reject random upstream repairs instead of learning from them.
            factors <- aligned[good] / ratio[good]
            factor <- stats::median(factors)
            if (!is.finite(factor) || factor <= 0 ||
                any(abs(factors / factor - 1) > 1e-7)) next
            predictors <- t(mat[fit$selected, query, drop=FALSE])
            predictors <- sweep(predictors, 2L, fit$centers, "-")
            predictors <- sweep(predictors, 2L, fit$scales, "/")
            valid_query <- apply(is.finite(predictors), 1L, all) &
                is.finite(mat[feature, query])
            if (!any(valid_query)) next
            valid_indices <- query[valid_query]
            newdata <- as.data.frame(predictors[valid_query, , drop=FALSE])
            colnames(newdata) <- fit$columns
            baseline <- as.numeric(stats::predict(fit$model, data=newdata)$
                predictions) + fit$response_center
            valid <- is.finite(baseline) & baseline > 0
            output[feature, valid_indices[valid]] <-
                mat[feature, valid_indices[valid]] / baseline[valid] * factor
        }
    }
    output
}
"""


def validate_serrf_qcs(
    data: pd.DataFrame,
    full: pd.DataFrame,
    *,
    source: str,
    extractor: str,
    parameters: dict[str, Any],
    batches: np.ndarray,
    qc: np.ndarray,
    cv_folds: int,
    random_state: int,
    cv_strategy: str = "random",
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Refit the seed/index-adapted R kernel, retaining failed-fold evidence."""
    folds = assign_qc_folds(
        batches, qc, cv_folds, random_state,
        strategy=cv_strategy, order_array=parameters.get("order"),
    )
    oof = full.mask(data.isna())
    oof.loc[:, qc] = np.nan
    code = _VALIDATE_SOURCE.replace(
        "__EXTRACT_FUNCTION__", extractor
    ).replace("__SOURCE_TEXT__", json.dumps(source, ensure_ascii=True)).replace(
        "__NA_SAFE_SUBSCRIPTS__", NA_SAFE_SUBSCRIPTS
    )
    fold_records: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    values = data.to_numpy(dtype=float)
    for fold in range(int(folds.max()) + 1):
        held_out = qc & (folds == fold)
        training = qc & ~held_out
        record: dict[str, Any] = {
            "fold": fold,
            "training_qc_indices": np.flatnonzero(training).tolist(),
            "held_out_qc_indices": np.flatnonzero(held_out).tolist(),
        }
        try:
            supported, support = supported_serrf_features(
                data, batches, training | ~qc,
                parameters["n_correlated_features"],
            )
            record.update(support)
            for batch in pd.unique(batches):
                subset = (batches == batch) & training
                if subset.sum() < 2:
                    raise ValueError(
                        f"Batch {batch!r} has fewer than two training QCs."
                    )
                role_values = values[supported][:, subset]
                positive_finite = (
                    np.isfinite(role_values).all(axis=1)
                    & (role_values > 0).all(axis=1)
                )
                if (positive_finite & (np.std(role_values, axis=1) == 0)).any():
                    raise ValueError(
                        f"Batch {batch!r} has constant training QC features."
                    )
            prediction, provenance = run_r_function(
                data.loc[supported],
                method="SERRF",
                package="ranger",
                function="pi-metaboqc::serrfR held-out QC adapter",
                code=code,
                parameters={
                    **parameters,
                    "original_model_seed": 1,
                    "effective_model_seed": random_state,
                    "model_seed_policy": "runtime AST adaptation to random_state",
                    "training_qc": training.tolist(),
                    "held_out_qc": held_out.tolist(),
                    "validation_fold": fold,
                },
                seed=random_state,
                allow_all_missing_input=True,
                defer_output_domain=True,
                transforms=(
                    "source-adapted serrfR fit on training QCs and real Samples",
                    "per-forest set.seed(1) adapted in memory to random_state",
                    "NA-safe runtime subscripts; unsupported features omitted",
                    "held-out QCs excluded from all fit/alignment inputs",
                    "ranger model prediction with training QC scales",
                    "QC alignment derived only from fitted training output",
                    "invalid validation denominators remain missing",
                    "missing held-out responses or predictors remain missing; "
                    "no validation repair",
                ),
            )
            prediction = prediction.reindex(index=data.index)
            provenance["input_support"] = support
            provenance["runtime_fixes"] = list(RUNTIME_FIXES)
            held_out_prediction = prediction.loc[:, held_out].to_numpy().copy()
            held_out_prediction[~np.isfinite(values[:, held_out])] = np.nan
            oof.loc[:, held_out] = held_out_prediction
            finite = np.isfinite(held_out_prediction)
            eligible = (
                np.isfinite(values[:, held_out]) & (values[:, held_out] > 0)
            )
            retained = finite & (held_out_prediction > 0) & eligible
            record.update(
                status=(
                    "completed"
                    if eligible.any() and retained.sum() == eligible.sum()
                    else "degraded"
                ),
                finite_held_out_cells=int(finite.sum()),
                total_held_out_cells=int(finite.size),
                eligible_held_out_cells=int(eligible.sum()),
                retained_eligible_cells=int(retained.sum()),
            )
            provenance_records.append(provenance)
        except (RBackendError, ValueError) as exc:
            record.update(status="failed", error=str(exc))
        fold_records.append(record)
    held_out_values = oof.loc[:, qc].to_numpy(dtype=float)
    coverage = float(np.isfinite(held_out_values).sum()) / max(
        1, held_out_values.size
    )
    eligible = np.isfinite(values[:, qc]) & (values[:, qc] > 0)
    retained = (
        np.isfinite(held_out_values) & (held_out_values > 0) & eligible
    )
    eligible_coverage = float(retained.sum()) / max(1, int(eligible.sum()))
    diagnostics = {
        "validation_backend": "pi-metaboqc R/ranger held-out-QC adapter",
        "cv_strategy": cv_strategy,
        "validation": (
            "within-batch held-out QCs; seed/index-adapted R source refitted without "
            "held-out QCs; training-only QC selection, scaling, response "
            "and alignment; biological cohort remains transductive"
        ),
        "fold_assignments": folds.tolist(),
        "effective_folds_by_batch": {
            str(batch): int(
                np.unique(folds[(batches == batch) & (folds >= 0)]).size
            )
            for batch in pd.unique(batches)
        },
        "qc_oof_value_coverage": coverage,
        "qc_oof_eligible_value_coverage": eligible_coverage,
        "qc_oof_coverage_denominator": "all non-Blank QC intensity cells",
        "qc_oof_eligible_coverage_denominator": (
            "original positive finite non-Blank QC intensity cells"
        ),
        "folds": fold_records,
        "status": "ok" if eligible_coverage == 1.0 else "degraded",
        "differences_from_author": [
            "per-forest set.seed(1) adapted at runtime to random_state",
            "upstream has no held-out QC prediction API",
            "fitted ranger models/scales captured without altering fit",
            "validation does not apply random repairs to held-out output",
        ],
    }
    return oof, diagnostics, provenance_records
