"""QC-only MetaNorm rLOESS semantics with native and R LOESS kernels.

Both implementations use log2 intensities, batchwise quadratic symmetric
LOESS, the fANCOVA GCV criterion and continuous span optimization. The stage
adapter excludes Blank from centering and preserves its input values. Held-out
QC values never enter span selection, fitting, or centering. Predictions
outside training-QC support remain missing, including boundary-QC OOF values.
"""

from __future__ import annotations

import importlib.metadata
import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib.externals.loky import ProcessPoolExecutor
from scipy.optimize import minimize_scalar
from skmisc.loess import loess

from ...constants import DEFAULT_RANDOM_SEED
from ..r_backend import run_r_function
from .qc_validation import assign_qc_folds


_OPTIMIZE_TOL = np.finfo(float).eps ** 0.25


class RLOESSNativeError(RuntimeError):
    """A native-kernel error requiring disposal of its isolated process."""


class RLOESSModel:
    """Native LOESS fit/predict with training-only fANCOVA span selection."""

    def __init__(
        self,
        *,
        span_selection: str = "gcv",
        span: float = 0.75,
        iterations: int = 4,
    ) -> None:
        """Store the span-selection and robust-iteration settings for
        fitting.
        """
        self.span_selection = span_selection
        self.span = span
        self.iterations = iterations

    def _geometry(self, span: float) -> np.ndarray:
        """Build all possible one-dimensional interpolation-node designs."""
        neighbours = int(len(self.x_) * span)
        if neighbours < 5:
            raise ValueError("Quadratic LOESS needs five span neighbours.")
        if neighbours not in self._designs:
            unique = np.unique(self.x_)
            width = np.ptp(unique)
            padding = 0.005 * max(width, 1e-10 * np.max(np.abs(unique)) + 1e-30)
            nodes = np.r_[
                unique,
                (unique[:-1] + unique[1:]) / 2,
                unique[0] - padding,
                unique[-1] + padding,
            ]
            delta = self.x_[None, :] - nodes[:, None]
            radius = np.partition(np.abs(delta), neighbours - 1, axis=1)[
                :, neighbours - 1
            ]
            if (radius <= 0).any():
                raise ValueError("LOESS neighbourhood has zero order range.")
            scaled = delta / radius[:, None]
            distance = np.minimum(np.abs(scaled), 1.0)
            tricube = (1 - distance**3) ** 3
            design = np.stack([np.ones_like(scaled), scaled, scaled**2], axis=2)
            self._designs[neighbours] = design * np.sqrt(tricube[:, :, None])
        return self._designs[neighbours]

    def _fit_span(self, span: float) -> tuple[Any, float]:
        """Fit source symmetric iterations with a preflight for every fit.

        scikit-misc can corrupt its heap when an internal symmetric iteration
        reaches a singular design. Running the original bisquare iterations
        explicitly permits rank checks before each native Gaussian fit.
        """
        design = self._geometry(span)
        robust = np.ones(len(self.x_))
        initial_trace = float("nan")
        for iteration in range(self.iterations):
            weighted = design * np.sqrt(robust[None, :, None])
            singular = np.linalg.svd(weighted, compute_uv=False)
            if np.any(singular[:, -1] <= 1e-10 * singular[:, 0]):
                raise ValueError("LOESS weighted quadratic design is singular.")
            try:
                model = loess(
                    self.x_,
                    self.y_,
                    weights=robust,
                    span=span,
                    degree=2,
                    family="gaussian",
                    normalize=True,
                    surface="interpolate",
                    statistics="approximate",
                    trace_hat="exact",
                    cell=0.2,
                )
                model.fit()
            except ValueError as exc:
                raise RLOESSNativeError(
                    f"Native LOESS fit failed after design checks: {exc}"
                ) from exc
            fitted = np.asarray(model.outputs.fitted_values)
            if not np.isfinite(fitted).all():
                raise ValueError("LOESS produced nonfinite fitted values.")
            if iteration == 0:
                initial_trace = float(model.outputs.trace_hat)
            # Exact lowesw rule: six times median absolute residual, with
            # the source's 0.001/0.999 cutoffs and near-zero-MAD handling.
            residual = np.abs(self.y_ - fitted)
            cmad = 6 * np.median(residual)
            if cmad < np.finfo(float).tiny:
                robust = np.ones(len(self.x_))
            else:
                robust = np.where(
                    residual > 0.999 * cmad,
                    0.0,
                    np.where(
                        residual <= 0.001 * cmad,
                        1.0,
                        (1 - (residual / cmad) ** 2) ** 2,
                    ),
                )
        if (
            not np.isfinite(initial_trace)
            or initial_trace >= len(self.x_) - 1e-8
        ):
            raise ValueError("LOESS has no residual degrees of freedom.")
        return model, initial_trace

    def fit(self, x: np.ndarray, y: np.ndarray) -> RLOESSModel:
        """Fit finite training QCs, rejecting nonestimable span candidates."""
        valid = np.isfinite(x) & np.isfinite(y)
        self.x_ = np.asarray(x, dtype=float)[valid]
        self.y_ = np.asarray(y, dtype=float)[valid]
        self._designs: dict[int, np.ndarray] = {}
        if len(self.x_) < 5 or np.unique(self.x_).size < 4:
            raise ValueError("rLOESS needs five QCs at four distinct orders.")

        def objective(span: float) -> float:
            try:
                model, trace = self._fit_span(span)
                residual = np.asarray(model.outputs.fitted_residuals)
                n = len(self.x_)
                sigma2 = np.sum(residual**2) / (n - 1)
                return float(n * sigma2 / (n - trace) ** 2)
            except ValueError:
                return float("inf")

        if self.span_selection == "gcv":
            # scipy's bounded optimizer and R optimize use Brent's method;
            # use R's default tolerance rather than scipy's default 1e-5.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = minimize_scalar(
                    objective,
                    bounds=(0.05, 0.95),
                    method="bounded",
                    options={"xatol": _OPTIMIZE_TOL},
                )
            if not result.success or not np.isfinite(result.fun):
                raise ValueError("No estimable rLOESS GCV span was found.")
            self.span_ = float(result.x)
        else:
            self.span_ = float(self.span)
        self.model_, self.trace_hat_ = self._fit_span(self.span_)
        self.gcv_ = objective(self.span_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict on training support only; never extrapolate or clip."""
        targets = np.asarray(x, dtype=float)
        result = np.full(targets.shape, np.nan)
        supported = (targets >= self.x_.min()) & (targets <= self.x_.max())
        if supported.any():
            try:
                result[supported] = self.model_.predict(
                    targets[supported]
                ).values
            except ValueError as exc:
                raise RLOESSNativeError(
                    f"Native LOESS prediction failed: {exc}"
                ) from exc
        return result


_R_PREDICT = """
function(mat, params) {
    result <- matrix(NA_real_, nrow(mat), ncol(mat))
    selected <- list()
    for (b in unique(params$batch)) {
        targets <- which(params$batch == b & !params$blank)
        training <- which(params$batch == b & params$train_qc)
        for (i in seq_len(nrow(mat))) {
            observed <- training[is.finite(mat[i, training])]
            x <- params$order[observed]
            y <- mat[i, observed]
            if (length(y) < 5 || length(unique(x)) < 4) {
                warning(sprintf("feature %d, batch %s: insufficient QCs", i,b))
                next
            }
            fit_span <- function(span) {
                neighbours <- floor(length(y) * span)
                if (neighbours < 5) stop("too few neighbours")
                unique_x <- sort(unique(x))
                width <- diff(range(x))
                padding <- 0.005 * max(width, 1e-10 * max(abs(x)) + 1e-30)
                nodes <- c(unique_x,
                    (head(unique_x, -1) + tail(unique_x, -1)) / 2,
                    min(x) - padding, max(x) + padding)
                designs <- lapply(nodes, function(node) {
                    delta <- x - node
                    radius <- sort(abs(delta))[neighbours]
                    if (radius <= 0) stop("zero neighbourhood range")
                    scaled <- delta / radius
                    weights <- (1 - pmin(abs(scaled), 1)^3)^3
                    cbind(1, scaled, scaled^2) * sqrt(weights)
                })
                robust <- rep(1, length(y))
                initial_trace <- NA_real_
                for (iteration in seq_len(params$iterations)) {
                    for (design in designs) {
                        singular <- svd(design * sqrt(robust), nu=0, nv=0)$d
                        if (min(singular) <= 1e-10 * max(singular))
                            stop("weighted quadratic design is singular")
                    }
                    fit <- withCallingHandlers(stats::loess(
                        y ~ x, weights=robust, degree=2, span=span,
                        family="gaussian", control=stats::loess.control(
                            surface="interpolate", statistics="approximate",
                            trace.hat="exact", cell=0.2)
                    ), warning=function(w) stop(conditionMessage(w)))
                    if (iteration == 1) initial_trace <- fit$trace.hat
                    residual <- abs(y - fit$fitted)
                    cmad <- 6 * median(residual)
                    if (cmad < .Machine$double.xmin) {
                        robust <- rep(1, length(y))
                    } else {
                        robust <- ifelse(residual > 0.999 * cmad, 0,
                            ifelse(residual <= 0.001 * cmad, 1,
                                (1 - (residual / cmad)^2)^2))
                    }
                }
                fit$trace.hat <- initial_trace
                if (any(!is.finite(fit$fitted)) ||
                    !is.finite(fit$trace.hat) ||
                    fit$trace.hat >= length(y) - 1e-8)
                    stop("nonestimable LOESS fit")
                fit
            }
            objective <- function(span) {
                tryCatch({
                    fit <- fit_span(span)
                    n <- length(y)
                    sigma2 <- sum(fit$residuals^2) / (n - 1)
                    n * sigma2 / (n - fit$trace.hat)^2
                }, error=function(e) Inf)
            }
            tryCatch({
                span <- params$span
                if (params$span_selection == "gcv") {
                    optimum <- suppressWarnings(stats::optimize(
                        objective, c(0.05, 0.95), tol=.Machine$double.eps^0.25))
                    if (!is.finite(optimum$objective))
                        stop("no estimable rLOESS GCV span")
                    span <- optimum$minimum
                }
                fit <- fit_span(span)
                selected[[length(selected) + 1L]] <- list(
                    feature=as.integer(i), batch=as.character(b),
                    span=span, gcv=objective(span))
                supported <- targets[params$order[targets] >= min(x) &
                                     params$order[targets] <= max(x)]
                if (length(supported)) result[i, supported] <- as.numeric(
                    stats::predict(fit,
                        newdata=data.frame(x=params$order[supported])))
            }, error=function(e) warning(sprintf(
                "feature %d, batch %s: %s", i, b, conditionMessage(e))))
        }
    }
    attr(result, "pimqc_diagnostics") <- list(selected_spans=selected)
    result
}
"""


class RobustRLSCCorrector:
    """MetaNorm rLOESS correction with leakage-free held-out QC validation.

    MetaNorm's full-input log-mean is adapted to nonblank input. An OOF fold
    additionally removes its held-out QCs from this centering reference.
    Both backends reject singular LOESS fits instead of accepting an R-only
    pseudoinverse fallback. The original Metanorm worker remains separately
    available for unadapted full-fit reference comparisons.
    """

    def __init__(
        self,
        *,
        implementation: str = "python",
        span_selection: str = "gcv",
        span: float = 0.75,
        iterations: int = 4,
        cv_folds: int = 5,
        random_state: int = DEFAULT_RANDOM_SEED,
        validation_strategy: str = "random",
    ) -> None:
        """Validate shared fitting controls and configure the QC backend."""
        if implementation not in {"python", "r"}:
            raise ValueError("rLOESS implementation must be python or r.")
        if span_selection not in {"gcv", "fixed"}:
            raise ValueError("rLOESS span_selection must be gcv or fixed.")
        if not 0 < span <= 1 or iterations < 1:
            raise ValueError("rLOESS needs 0 < span <= 1 and iterations >= 1.")
        if cv_folds not in (0,) and cv_folds < 2:
            raise ValueError("rLOESS cv_folds must be zero or at least two.")
        self.implementation = implementation
        self.span_selection = span_selection
        self.span = float(span)
        self.iterations = int(iterations)
        self.cv_folds = int(cv_folds)
        self.random_state = int(random_state)
        self.validation_strategy = validation_strategy
        self.provenance: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] = {}

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the shared algorithm contract recorded in the audit."""
        return {
            "span_selection": self.span_selection,
            "span": self.span,
            "span_range": [0.05, 0.95],
            "degree": 2,
            "optimizer": "bounded Brent",
            "optimizer_tolerance": float(_OPTIMIZE_TOL),
            "family": "symmetric",
            "iterations": self.iterations,
            "kernel_family": "gaussian; symmetric weights applied externally",
            "surface": "interpolate",
            "statistics": "approximate",
            "trace_hat": "exact",
            "cell": 0.2,
            "criterion": "n * (SSE / (n - 1)) / (n - trace_hat)^2",
            "QConly": True,
            "batchwise": True,
            "keepScale": True,
            "scale": "log2 -> residual + nonblank log-mean -> exp2",
            "minimum_qc": 5,
            "minimum_qc_for_gcv": 6,
            "minimum_distinct_qc_orders": 4,
            "minimum_span_neighbours": 5,
            "weighted_design_min_singular_ratio": 1e-10,
            "symmetric_fit": "explicit source lowesw bisquare iterations",
            "source": "MetaNorm rLOESS / fANCOVA GCV / stats lowesw",
            "source_adaptations": [
                "QC-only input; exclude Blank from log-mean centering",
                "fold-local span, fitting and centering for QC validation",
                "reject underdetermined weighted quadratic designs",
                "isolate native code; no full-fit or R fallback",
            ],
            "singular_fit_policy": "unavailable; no pseudoinverse fallback",
            "outside_qc_support": "missing; no extrapolation",
            "blank_policy": "excluded from fit and center; passthrough",
            "oof_center": "exclude held-out QC values",
            "validation_strategy": self.validation_strategy,
            "cv_folds": self.cv_folds,
            "execution": (
                "isolated single native worker"
                if self.implementation == "python"
                else "serial R main thread"
            ),
        }

    def _predict(
        self,
        logged: pd.DataFrame,
        batches: np.ndarray,
        order: np.ndarray,
        train_qc: np.ndarray,
        blank: np.ndarray,
        fit_label: str,
    ) -> np.ndarray:
        if self.implementation == "r":
            result, provenance = run_r_function(
                logged,
                method="robust QC-RLSC",
                package="stats",
                function="stats::loess / fANCOVA GCV criterion",
                code=_R_PREDICT,
                parameters={
                    **self.parameters,
                    "batch": batches.astype(str).tolist(),
                    "order": order.tolist(),
                    "blank": blank.tolist(),
                    "train_qc": train_qc.tolist(),
                    "fit": fit_label,
                },
                seed=self.random_state,
                allow_missing=True,
                allow_all_missing_input=True,
                defer_output_domain=True,
                transforms=("QC-only log2 LOESS drift predictions",),
            )
            if fit_label == "full":
                self.provenance = provenance
            self.diagnostics["fit_warnings"][fit_label] = provenance["warnings"]
            selected = provenance.get("diagnostics", {}).get(
                "selected_spans", []
            )
            for fit in selected:
                fit["feature"] = str(logged.index[int(fit["feature"]) - 1])
            self.diagnostics["selected_spans"][fit_label] = selected
            return result.to_numpy(dtype=float)
        result = np.full(logged.shape, np.nan)
        failures = []
        spans = []
        values = logged.to_numpy(dtype=float)
        for batch in pd.unique(batches):
            train = (batches == batch) & train_qc
            targets = (batches == batch) & ~blank
            for i, feature in enumerate(logged.index):
                try:
                    model = RLOESSModel(
                        span_selection=self.span_selection,
                        span=self.span,
                        iterations=self.iterations,
                    ).fit(order[train], values[i, train])
                    result[i, targets] = model.predict(order[targets])
                    spans.append(
                        {
                            "feature": str(feature),
                            "batch": str(batch),
                            "span": model.span_,
                            "gcv": model.gcv_,
                        }
                    )
                except ValueError as exc:
                    failures.append(
                        {
                            "feature": str(feature),
                            "batch": str(batch),
                            "error": str(exc),
                        }
                    )
        self.diagnostics["fit_warnings"][fit_label] = failures
        self.diagnostics["selected_spans"][fit_label] = spans
        return result

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        *,
        blank_mask: np.ndarray | None = None,
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """Protect the application process from native-library failures."""
        if self.implementation == "r":
            return self._fit_transform(
                intensity_df,
                batch_array,
                qc_mask,
                order_array,
                blank_mask=blank_mask,
            )
        # Own this executor: joblib Parallel adds memmapping state to its
        # global reusable executor, so sharing that pool would corrupt the
        # later QC-SVR/SERRF execution context.
        executor = ProcessPoolExecutor(max_workers=1)
        future = executor.submit(
            _native_fit_transform,
            self,
            intensity_df,
            batch_array,
            qc_mask,
            order_array,
            blank_mask,
        )
        try:
            output, provenance, diagnostics = future.result()
        except BaseException:
            # Never reuse a process after a native error: a C error may have
            # damaged its heap even if Python received a normal exception.
            executor.shutdown(wait=False, kill_workers=True)
            raise
        executor.shutdown(wait=True)
        self.provenance = provenance
        self.diagnostics = diagnostics
        return output

    def _fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        *,
        blank_mask: np.ndarray | None = None,
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """Correct all samples and independently refit each held-out fold."""
        values = intensity_df.to_numpy(dtype=float, copy=True)
        batches = np.asarray(batch_array)
        qc = np.asarray(qc_mask, dtype=bool)
        order = np.asarray(order_array, dtype=float)
        blank = (
            np.zeros(values.shape[1], dtype=bool)
            if blank_mask is None
            else np.asarray(blank_mask, dtype=bool)
        )
        if any(
            array.shape != (values.shape[1],)
            for array in (batches, qc, order, blank)
        ):
            raise ValueError("rLOESS metadata must align with the samples.")
        if pd.isna(batches).any() or not np.isfinite(order).all():
            raise ValueError("rLOESS needs batch labels and finite orders.")
        if (qc & blank).any():
            raise ValueError("A sample cannot be both QC and Blank.")
        if np.isinf(values[:, ~blank]).any() or (values[:, ~blank] <= 0).any():
            raise ValueError("robust QC-RLSC needs positive raw intensities.")
        logged_values = np.full(values.shape, np.nan)
        logged_values[:, ~blank] = np.log2(values[:, ~blank])
        logged = pd.DataFrame(
            logged_values,
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        self.diagnostics = {
            "algorithm": "MetaNorm rLOESS",
            "parameters": self.parameters,
            "fit_warnings": {},
            "selected_spans": {},
            "validation": "held-out QC; fold-local GCV, fit and centering",
        }
        if self.implementation == "python":
            self.provenance = {
                "implementation": "python",
                "method": "robust QC-RLSC",
                "package": "scikit-misc",
                "package_version": importlib.metadata.version("scikit-misc"),
                "function": "skmisc.loess.loess",
                "parameters": self.parameters,
                "seed": self.random_state,
            }

        def correct(prediction: np.ndarray, excluded: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                center = np.nanmean(
                    logged_values[:, ~blank & ~excluded], axis=1
                )
                result = np.exp2(logged_values - prediction + center[:, None])
            result[:, blank] = values[:, blank]
            result[np.isnan(values)] = np.nan
            return result

        full_prediction = self._predict(
            logged, batches, order, qc, blank, "full"
        )
        full = correct(full_prediction, np.zeros(qc.shape, dtype=bool))
        oof = full.copy()
        oof[:, qc] = np.nan
        folds = assign_qc_folds(
            batches,
            qc,
            self.cv_folds,
            self.random_state,
            strategy=self.validation_strategy,
            order_array=order,
        )
        for fold in np.unique(folds[folds >= 0]):
            held_out = folds == fold
            prediction = self._predict(
                logged, batches, order, qc & ~held_out, blank, f"fold_{fold}"
            )
            corrected = correct(prediction, held_out)
            oof[:, held_out] = corrected[:, held_out]
        self.diagnostics["fold_assignments"] = folds.tolist()
        self.diagnostics["qc_oof_observed_coverage"] = float(
            (np.isfinite(oof[:, qc]) & np.isfinite(values[:, qc])).sum()
            / max(1, np.isfinite(values[:, qc]).sum())
        )
        frames = []
        for result in (full, oof):
            frame = pd.DataFrame(
                result, index=intensity_df.index, columns=intensity_df.columns
            )
            frame.attrs.update(intensity_df.attrs)
            frames.append(frame)
        return {"robust QC-RLSC corrected": (frames[0], frames[1])}


def _native_fit_transform(
    engine: RobustRLSCCorrector,
    frame: pd.DataFrame,
    batches: np.ndarray,
    qc: np.ndarray,
    order: np.ndarray,
    blank: np.ndarray | None,
) -> tuple[dict, dict, dict]:
    """Run the native computation in its supervised, dedicated worker."""
    output = engine._fit_transform(frame, batches, qc, order, blank_mask=blank)
    return output, engine.provenance, engine.diagnostics
