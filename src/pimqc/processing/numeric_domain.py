"""Stage-specific numeric output rules, separate from algorithm kernels.

Signed model coordinates are not intrinsically invalid. Correction operates
on positive intensities; imputation reconstructs log2(raw + 1). Normalization
uses its own declared scale and must not reuse these positivity policies.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Never repair labels or reorder scientific results implicitly."""
    if not left.index.equals(right.index) or not left.columns.equals(
        right.columns
    ):
        raise ValueError("Numeric-domain matrices must have identical axes.")


def inspect_numeric_domain(frame: pd.DataFrame) -> dict[str, int]:
    """Count values without assigning a meaning to their signs."""
    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values)
    return {
        "cell_count": int(values.size),
        "finite_count": int(finite.sum()),
        "negative_count": int((finite & (values < 0)).sum()),
        "zero_count": int((finite & (values == 0)).sum()),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
    }


def _affected_cells(frame: pd.DataFrame, mask: np.ndarray) -> list[dict]:
    """Use stable, displayable identities rather than matrix positions."""
    return [
        {"feature": str(frame.index[row]), "sample": str(frame.columns[col])}
        for row, col in zip(*np.nonzero(mask))
    ]


def apply_correction_domain(
    frame: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert invalid raw-scale predictions to missing, never to raw data."""
    _aligned(frame, reference)
    values = frame.to_numpy(dtype=float, copy=True)
    before = reference.to_numpy(dtype=float)
    observed = np.isfinite(before) & (before > 0)
    invalid = ~np.isfinite(values) | (values <= 0)
    new_missing = observed & invalid
    report = {
        **inspect_numeric_domain(frame),
        "policy": "nonpositive_to_missing",
        "value_scale": "raw_positive",
        "invalid_count": int(invalid.sum()),
        "input_missing_count": int((~observed).sum()),
        "new_missing_count": int(new_missing.sum()),
        "observed_reference_count": int(observed.sum()),
        "valid_observed_count": int((observed & ~invalid).sum()),
        "observed_coverage": (
            float((observed & ~invalid).sum() / observed.sum())
            if observed.any()
            else None
        ),
        "affected_cells": _affected_cells(frame, new_missing),
    }
    values[invalid] = np.nan
    result = pd.DataFrame(values, index=frame.index, columns=frame.columns)
    result.attrs = copy.deepcopy(frame.attrs)
    return result, report


def repair_imputed_log_values(
    output_log: pd.DataFrame,
    training_log: pd.DataFrame,
    *,
    role_labels: np.ndarray,
    target_mask: pd.DataFrame | np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair finite nonpositive imputations from training-only raw minima.

    The default target is every missing training entry. A partial target can
    separate original S-route cells from correction-induced missingness without
    allowing previously imputed values to become lower-bound evidence.
    Nonfinite predictions remain model failures, not floor substitutions.
    """
    _aligned(output_log, training_log)
    values = output_log.to_numpy(dtype=float, copy=True)
    training = training_log.to_numpy(dtype=float)
    roles = np.asarray(role_labels)
    if roles.shape != (values.shape[1],) or pd.isna(roles).any():
        raise ValueError("One nonmissing role label is required per sample.")
    missing = np.isnan(training)
    if np.isinf(training).any() or (training[~missing] < 0).any():
        raise ValueError("Training values must be nonnegative log2p1 data.")
    if not np.array_equal(values[~missing], training[~missing]):
        raise ValueError("Imputation changed observed training values.")
    if target_mask is None:
        target = missing
    else:
        if isinstance(target_mask, pd.DataFrame):
            _aligned(target_mask, training_log)
        target = np.asarray(target_mask)
        if target.dtype != np.bool_ or target.shape != values.shape:
            raise ValueError("Imputation target mask must be aligned boolean.")
        if (target & ~missing).any():
            raise ValueError("Imputation targets must be missing in training.")
    if not np.isfinite(values[target]).all():
        raise ValueError("Imputation produced nonfinite target predictions.")
    with np.errstate(over="ignore", invalid="ignore"):
        reconstructed = np.expm1(values[target] * np.log(2.0))
        raw_training = np.expm1(training * np.log(2.0))
    if not np.isfinite(reconstructed).all():
        raise ValueError("Imputation inverse produced nonfinite intensities.")
    invalid = target & (values <= 0)
    fallback_count = 0
    floors = []
    for role in pd.unique(roles):
        columns = roles == role
        role_training = raw_training[:, columns]
        role_positive = role_training[
            np.isfinite(role_training) & (role_training > 0)
        ]
        for row in np.flatnonzero(invalid[:, columns].any(axis=1)):
            positions = invalid[row] & columns
            observed = raw_training[row, columns]
            positive = observed[np.isfinite(observed) & (observed > 0)]
            if not positive.size:
                positive = role_positive
                fallback_count += int(positions.sum())
            if not positive.size:
                raise ValueError(
                    f"No positive training evidence for sample role {role!r}."
                )
            floor = float(positive.min() / 2.0)
            if not np.isfinite(floor) or floor <= 0:
                raise ValueError(
                    "Positive imputation floor is unrepresentable."
                )
            values[row, positions] = np.log1p(floor) / np.log(2.0)
            floors.append(floor)
    result = pd.DataFrame(
        values, index=output_log.index, columns=output_log.columns
    )
    result.attrs = copy.deepcopy(output_log.attrs)
    return result, {
        "policy": "nonpositive_imputation_to_training_half_min",
        "value_scale": "log2p1",
        "target_count": int(target.sum()),
        "nonpositive_count": int(invalid.sum()),
        "repaired_count": int(invalid.sum()),
        "role_global_fallback_count": fallback_count,
        "minimum_replacement_raw": min(floors) if floors else None,
        "maximum_replacement_raw": max(floors) if floors else None,
        "affected_cells": _affected_cells(output_log, invalid),
        "observed_unchanged": True,
    }
