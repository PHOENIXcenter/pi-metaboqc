"""Assign deterministic held-out QC folds within acquisition batches.

Shared fold construction keeps correction candidates comparable while leaving
the fitting and prediction of excluded QC samples to each method's adapter.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def assign_qc_folds(
    batches: np.ndarray,
    qc_mask: np.ndarray,
    cv_folds: int,
    random_state: int,
    *,
    strategy: str = "random",
    order_array: np.ndarray | None = None,
) -> np.ndarray:
    """Assign random or contiguous chronological within-batch QC folds.

    Random is the unchanged production default. Blocked validation is an
    optional interpolation/extrapolation stress test, not forward chaining.
    Stable sorting makes tied injection orders deterministic.
    """
    if strategy not in {"random", "blocked"}:
        raise ValueError("QC validation strategy must be random or blocked.")
    orders = None
    if strategy == "blocked":
        orders = np.asarray(order_array, dtype=float)
        if orders.shape != np.asarray(qc_mask).shape:
            raise ValueError("Blocked QC validation needs aligned orders.")
        if not np.isfinite(orders[np.asarray(qc_mask, dtype=bool)]).all():
            raise ValueError("Blocked QC validation needs finite QC orders.")
    folds = np.full(len(qc_mask), -1, dtype=int)
    if cv_folds == 0:
        return folds
    for batch in pd.unique(batches):
        indices = np.flatnonzero((batches == batch) & qc_mask)
        if indices.size < 3:
            continue
        if strategy == "random":
            splitter = KFold(
                n_splits=min(cv_folds, indices.size),
                shuffle=True,
                random_state=random_state,
            )
        else:
            indices = indices[np.argsort(orders[indices], kind="stable")]
            splitter = KFold(
                n_splits=min(cv_folds, indices.size), shuffle=False
            )
        for fold, (_, held_out) in enumerate(splitter.split(indices)):
            folds[indices[held_out]] = fold
    return folds
