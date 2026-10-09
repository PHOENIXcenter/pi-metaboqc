"""Temporary calculation matrices shared by native and original R adapters.

This is an input adaptation, not a missing-data estimator or final imputation.
Only observed fitting samples determine the feature medians. Callers restore
the returned missing mask after correction and retain their own Blank policy.
"""

from typing import Any

import numpy as np
import pandas as pd


def prepare_median_input(
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    *,
    scale: str = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fill a detached feature-by-sample matrix on the declared fit scale.

    Fully missing fitting features fail rather than borrowing another feature
    or inventing zeros. Non-fitting columns never influence the medians.
    Their placeholders are only a convenience for native projection; original
    R adapters continue to exclude Blanks and return them unchanged.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Temporary median input requires a nonempty table.")
    fit = np.asarray(fit_mask)
    if fit.shape != (frame.shape[1],) or fit.dtype.kind != "b":
        raise ValueError("fit_mask must be an aligned boolean sample mask.")
    if not fit.any():
        raise ValueError("Temporary median input requires fitting samples.")
    if scale not in {"raw", "log1p"}:
        raise ValueError("Temporary median scale must be raw or log1p.")
    values = frame.to_numpy(dtype=float, copy=True)
    if np.isinf(values[:, fit]).any():
        raise ValueError("Infinite fitting values cannot be median-adapted.")
    if scale == "log1p" and (values[:, fit] < 0).any():
        raise ValueError("Log1p fitting intensities must be nonnegative.")
    if scale == "log1p":
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.log1p(values)
    missing = np.isnan(values)
    observed_counts = (~missing[:, fit]).sum(axis=1)
    unsupported = observed_counts == 0
    if unsupported.any():
        ids = frame.index[unsupported].tolist()
        raise ValueError(
            "No observed non-Blank reference for fully missing features: "
            f"{ids[:5]!r}. Temporary median adaptation cannot fit them."
        )
    medians = np.nanmedian(values[:, fit], axis=1)
    filled = np.where(missing, medians[:, None], values)
    count = int(missing[:, fit].sum())
    diagnostics = {
        "applied": bool(count),
        "method": "feature_median",
        "scale": scale,
        "reference": "non_blank_observed",
        "input_missing_cells": count,
        "temporary_filled_cells": count,
        "affected_feature_count": int(missing[:, fit].any(axis=1).sum()),
        "min_observed_per_feature": int(observed_counts.min()),
        "output_policy": "restore_original_missing_positions",
        "upstream_native_missing_support": False,
    }
    return (
        pd.DataFrame(filled, index=frame.index, columns=frame.columns),
        pd.DataFrame(missing, index=frame.index, columns=frame.columns),
        diagnostics,
    )
