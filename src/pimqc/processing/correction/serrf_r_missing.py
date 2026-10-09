"""Define NA-safe execution boundaries for the pinned original SERRF function.

Runtime subscript repairs preserve the verified source on disk. Support checks
identify features that cannot be fitted without inventing batch observations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NA_SAFE_SUBSCRIPTS = """
function(definition) {
    replacements <- 0L
    walk <- function(node) {
        if (!is.call(node)) return(node)
        for (i in seq_along(node)) {
            if (!identical(node[[i]], quote(expr=))) {
                node[[i]] <- walk(node[[i]])
            }
        }
        if (identical(node[[1]], as.name("[")) && length(node) >= 3L) {
            for (i in seq.int(3L, length(node))) {
                if (identical(node[[i]], quote(expr=))) next
                index <- node[[i]]
                if (is.call(index) && length(index) == 3L &&
                    as.character(index[[1]]) %in% c("==", "<", ">") &&
                    identical(index[[3]], 0)) {
                    replacements <<- replacements + 1L
                    node[[i]] <- call("which", index)
                }
            }
        }
        node
    }
    repaired <- walk(definition)
    if (replacements != 10L) {
        stop("Unexpected original SERRF zero/negative subscript layout")
    }
    repaired
}
"""

RUNTIME_FIXES = [
    "NA-safe zero/negative subscript comparisons via which",
    "features without observations in any fit batch excluded, not imputed",
]


def supported_serrf_features(
    data: pd.DataFrame,
    batches: np.ndarray,
    fit_mask: np.ndarray,
    n_correlated_features: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Require a real low-value reference in every fitted feature/batch."""
    observed = np.isfinite(data.to_numpy(dtype=float))
    supported = np.ones(len(data), dtype=bool)
    missing_by_batch: dict[str, list[str]] = {}
    for batch in pd.unique(batches[fit_mask]):
        no_reference = ~observed[:, fit_mask & (batches == batch)].any(axis=1)
        if no_reference.any():
            missing_by_batch[str(batch)] = [
                str(item) for item in data.index[no_reference]
            ]
            supported &= ~no_reference
    diagnostics = {
        "unsupported_feature_ids": [
            str(item) for item in data.index[~supported]
        ],
        "unsupported_feature_count": int((~supported).sum()),
        "all_missing_features_by_fit_batch": missing_by_batch,
        "unsupported_feature_policy": (
            "all non-Blank outputs missing; no cross-batch fill"
        ),
        "support_reference": "fitting samples only; held-out QCs excluded",
    }
    if supported.sum() <= n_correlated_features:
        raise ValueError(
            "Original SERRF requires more supported features than "
            "n_correlated_features after excluding all-missing fit batches."
        )
    return supported, diagnostics
