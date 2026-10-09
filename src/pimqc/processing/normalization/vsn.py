"""Native affine VSN with profile likelihood and intensity-stratified LTS.

The statistical model and defaults follow Bioconductor vsn 3.78.1:
``R/vsn2.R`` (vsnLTS/vsnMatrix) and ``src/vsn2.c`` (loglik/grad_loglik).
Each sample has an offset and a positive slope. Optimization is performed
in log-slope coordinates, with an analytical gradient, using SciPy's
L-BFGS-B implementation. Optimizer paths can differ from R's implementation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.stats import rankdata


@njit(cache=True)
def _vsn_nll_gradient(
    parameters: np.ndarray, values: np.ndarray
) -> tuple[float, np.ndarray]:
    """Profile out feature means and residual variance, including Jacobian.

    For z_ij = a_j + exp(b_j) * x_ij and h_ij = asinh(z_ij),
    the negative log likelihood is
    N/2 * (log(2*pi*RSS/N) + 1) + sum(log(hypot(1,z_ij))-b_j).
    Missing observations contribute to neither term nor the gradient.
    """
    rows, columns = values.shape
    slopes = np.exp(parameters[columns:])
    gradient = np.zeros(2 * columns)
    residual = np.empty_like(values)
    reciprocal = np.empty_like(values)
    calibrated = np.empty_like(values)
    count = 0
    squares = 0.0
    jacobian = 0.0

    for row in range(rows):
        total = 0.0
        observed = 0
        for column in range(columns):
            value = values[row, column]
            if not np.isnan(value):
                z = parameters[column] + slopes[column] * value
                h = np.arcsinh(z)
                radius = np.hypot(1.0, z)
                calibrated[row, column] = z
                reciprocal[row, column] = 1.0 / radius
                residual[row, column] = h
                total += h
                observed += 1
                jacobian += np.log(radius) - parameters[columns + column]
        if observed:
            mean = total / observed
            for column in range(columns):
                if not np.isnan(values[row, column]):
                    residual[row, column] -= mean
                    squares += residual[row, column] ** 2
            count += observed

    if count == 0 or squares <= np.finfo(np.float64).tiny:
        return np.inf, gradient
    variance = squares / count
    for column in range(columns):
        for row in range(rows):
            value = values[row, column]
            if not np.isnan(value):
                inverse = reciprocal[row, column]
                derivative = (
                    residual[row, column] / variance
                    + calibrated[row, column] * inverse
                ) * inverse
                gradient[column] += derivative
                gradient[columns + column] += (
                    derivative * slopes[column] * value - 1.0
                )
    nll = count / 2 * (np.log(2 * np.pi * variance) + 1) + jacobian
    return nll, gradient


def _lts_selection(transformed: np.ndarray, quantile: float) -> np.ndarray:
    """Keep the lowest mean quintile and low-RSS rows in each other quintile.

    As in vsnLTS, an incomplete row has an undefined RSS and is excluded
    above the lowest quintile. The likelihood itself handles missing cells.
    """
    means = np.nanmean(transformed, axis=1)
    ranks = rankdata(means, method="average")
    slices = np.asarray(pd.cut(ranks, bins=5, labels=False))
    squares = np.sum((transformed - means[:, None]) ** 2, axis=1)
    selected = slices == 0
    for group in range(1, 5):
        members = (slices == group) & np.isfinite(squares)
        if members.any():
            cutoff = np.quantile(squares[members], quantile)
            selected |= members & (squares <= cutoff)
    return selected


def fit_vsn(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit affine VSN to raw intensities and return generalized log2 values.

    Seven LTS rounds fit all eligible rows first, then iteratively refit
    intensity-stratified low-residual features. No feature subsampling or
    additional data-dependent shift is applied. The output offset is
    log2(2 * geometric_mean(sample_slopes)), matching vsn's log2 convention.
    Unlike R's default 42-row guard, the native API permits smaller inputs
    with at least three varying observations per sample; small fits remain
    statistically underpowered and their convergence metadata is retained.
    """
    values = df.to_numpy(dtype=np.float64, copy=True)
    if values.shape[1] < 2:
        raise ValueError("VSN requires at least two sample columns.")
    if np.isinf(values).any():
        raise ValueError("VSN input must not contain infinite values.")
    observed_rows = np.isfinite(values).any(axis=1)
    fitting = values[observed_rows]
    if fitting.shape[0] < 3 or (np.isfinite(fitting).sum(axis=0) < 3).any():
        raise ValueError("VSN requires at least three observations per sample.")
    if (np.nanstd(fitting, axis=0) == 0).any():
        raise ValueError("VSN cannot fit a sample with constant intensities.")

    columns = values.shape[1]
    parameters = np.r_[np.zeros(columns), np.ones(columns)]
    bounds = [(None, None)] * columns + [(-100.0, 100.0)] * columns
    selected = np.ones(fitting.shape[0], dtype=bool)
    rounds = []
    for iteration in range(7):
        fit_data = np.ascontiguousarray(fitting[selected])
        if (np.isfinite(fit_data).sum(axis=0) < 2).any():
            raise ValueError("VSN trimming left insufficient sample overlap.")
        result = minimize(
            _vsn_nll_gradient,
            parameters,
            args=(fit_data,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxcor": 5,
                "ftol": 5e7 * np.finfo(float).eps,
                "gtol": 2e-4,
                "maxiter": 60000,
                "maxfun": 100000,
            },
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError("VSN likelihood has no finite fitted solution.")
        parameters = result.x
        rounds.append(
            {
                "round": iteration + 1,
                "features": int(selected.sum()),
                "converged": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "negative_log_likelihood": float(result.fun),
            }
        )
        if iteration < 6:
            transformed = np.arcsinh(
                fitting * np.exp(parameters[columns:]) + parameters[:columns]
            )
            selected = _lts_selection(transformed, 0.9)

    converged = all(item["converged"] for item in rounds)
    if not converged:
        warnings.warn(
            "VSN returned a finite fit, but at least one LTS optimization "
            "did not converge; inspect vsn_optimizer metadata.",
            RuntimeWarning,
            stacklevel=2,
        )
    slopes = np.exp(parameters[columns:])
    offset = float(1.0 + np.mean(parameters[columns:]) / np.log(2.0))
    transformed = (
        np.arcsinh(values * slopes + parameters[:columns]) / np.log(2.0)
        - offset
    )
    if not np.isfinite(transformed[np.isfinite(values)]).all():
        raise ValueError("VSN transformation produced nonfinite values.")
    output = pd.DataFrame(transformed, index=df.index, columns=df.columns)
    output.attrs.update(df.attrs)
    metadata = {
        "vsn_model": "sample_affine_profile_likelihood_lts",
        "vsn_scale": float(np.exp(np.mean(parameters[columns:]))),
        "vsn_shift": -offset,
        "vsn_scale_definition": "geometric mean of sample slopes",
        "vsn_shift_definition": "additive generalized-log2 output offset",
        "vsn_sample_ids": [str(column) for column in df.columns],
        "vsn_sample_offsets": parameters[:columns].tolist(),
        "vsn_sample_slopes": slopes.tolist(),
        "vsn_lts_quantile": 0.9,
        "vsn_lts_rounds": 7,
        "vsn_fit_feature_count": int(selected.sum()),
        "vsn_converged": converged,
        "vsn_optimizer": rounds,
    }
    return output, metadata
