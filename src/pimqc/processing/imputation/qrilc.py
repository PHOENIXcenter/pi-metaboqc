"""Sample-wise QRILC fitting and sampling on the caller's analysis scale.

The one-dimensional Gibbs branch used by imputeLCMD/tmvtnorm treats its
``sigma`` argument as a standard deviation, despite the multivariate API's
covariance terminology. This module follows that actual sampling distribution;
it does not attempt to reproduce R's random-number stream.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass(frozen=True)
class QRILCTail:
    """Parameters of the fitted, upper-truncated univariate normal."""

    location: float
    fitted_scale: float
    sampling_sd: float
    upper: float


def fit_qrilc_tail(
    observed: np.ndarray,
    missing_fraction: float,
    tune_sigma: float = 1.0,
) -> QRILCTail:
    """Fit imputeLCMD's normal-score regression without drawing values."""
    if not np.isfinite(tune_sigma) or tune_sigma <= 0:
        raise ValueError("QRILC tune_sigma must be finite and positive.")
    if not 0 < missing_fraction < 0.99:
        raise ValueError("QRILC missing fraction must be between 0 and 0.99.")
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 1 or len(observed) < 2:
        raise ValueError("QRILC fitting requires at least two observations.")
    if not np.isfinite(observed).all():
        raise ValueError("QRILC observations must be finite.")
    probs = np.arange(100, dtype=float) * 0.01 + 0.001
    theoretical = stats.norm.ppf(
        np.linspace(missing_fraction + 0.001, 0.991, len(probs))
    )
    empirical = np.quantile(observed, probs, method="linear")
    slope, intercept = np.polyfit(theoretical, empirical, deg=1)
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("QRILC fitted scale must be finite and positive.")
    # tmvtnorm's univariate rtnorm.gibbs uses sigma as SD, not variance.
    sampling_sd = float(slope * tune_sigma)
    if not np.isfinite(sampling_sd) or sampling_sd <= 0:
        raise ValueError("QRILC sampling SD must be finite and positive.")
    return QRILCTail(
        location=float(intercept),
        fitted_scale=float(slope),
        sampling_sd=sampling_sd,
        upper=float(stats.norm.ppf(
            missing_fraction + 0.001, loc=intercept, scale=slope
        )),
    )


def impute_qrilc(
    frame: pd.DataFrame, *, tune_sigma: float, seed: int
) -> pd.DataFrame:
    """Draw left-tail values, preserving observations and signed log values.

    Fewer than three observations or a constant observed column retain the
    native constant-fill extension, with an explicit warning. This is not an
    original-R distribution fit. Raw-intensity inversion is the stage's job.
    """
    if not np.isfinite(tune_sigma) or tune_sigma <= 0:
        raise ValueError("QRILC tune_sigma must be finite and positive.")
    values = frame.to_numpy(dtype=float, copy=True)
    if np.isinf(values).any():
        raise ValueError("QRILC observations must be finite or missing.")
    rng = np.random.default_rng(seed)
    fallback_columns = []
    for column in range(values.shape[1]):
        missing = np.isnan(values[:, column])
        count = int(missing.sum())
        if not count:
            continue
        observed = values[~missing, column]
        if len(observed) < 3 or np.ptp(observed) == 0:
            fallback_columns.append(frame.columns[column])
            values[missing, column] = observed.min() if len(observed) else 0.0
            continue
        tail = fit_qrilc_tail(
            observed, count / len(values), tune_sigma=tune_sigma
        )
        values[missing, column] = stats.truncnorm.rvs(
            a=-np.inf,
            b=(tail.upper - tail.location) / tail.sampling_sd,
            loc=tail.location,
            scale=tail.sampling_sd,
            size=count,
            random_state=rng,
        )
    if fallback_columns:
        logger.warning(
            "QRILC {} columns have insufficient variation/observations; "
            "using native constant-fill extension, not an original-R "
            "distribution fit. Column IDs: {}",
            len(fallback_columns),
            fallback_columns,
        )
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)
