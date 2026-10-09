"""Call the installed MetaNorm worker independently of the production adapter.

The helper retains the original full-fit worker settings used by the reference
tests. Feature-specific R fitting failures remain missing instead of being
replaced by the Python result or the uncorrected input.
"""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

from .helpers import require_r_package


def _run_metanorm_rloess(
    intensity_df: pd.DataFrame,
    order: np.ndarray,
    batch: np.ndarray,
    sample_type: np.ndarray,
    *,
    qc_only: bool,
) -> pd.DataFrame:
    """Run Metanorm rLOESS and preserve per-feature fitting failures as NaN."""
    require_r_package("metanorm")
    run_rloess = ro.r(
        """
        function(mat, order, batch, sample_type, qc_only) {
            suppressPackageStartupMessages(library(metanorm))
            batch <- as.factor(as.character(batch))
            result <- lapply(seq_len(nrow(mat)), function(i) {
                tryCatch(
                    metanorm::metanormWorker(
                        raw = unname(mat[i, ]),
                        order = as.numeric(order),
                        keepScale = TRUE,
                        QConly = qc_only,
                        QCcheck = FALSE,
                        QCcheckp = 0.1,
                        changepoints = FALSE,
                        type = as.character(sample_type),
                        batch = batch,
                        batchwise = TRUE,
                        weights = rep(1, ncol(mat)),
                        model = "rLOESS",
                        k = min(ncol(mat) * 0.9, 10),
                        cv = "GCV",
                        plotdir = NULL,
                        plottype = "pdf",
                        i = i
                    ),
                    error = function(e) rep(NA_real_, ncol(mat))
                )
            })
            do.call(rbind, result)
        }
        """
    )
    values = intensity_df.to_numpy(dtype=float, copy=True)
    r_matrix = ro.r["matrix"](
        ro.FloatVector(values.ravel(order="F")),
        nrow=values.shape[0],
        ncol=values.shape[1],
    )
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_result = run_rloess(
            r_matrix,
            ro.FloatVector(order.tolist()),
            ro.StrVector(batch.tolist()),
            ro.StrVector(sample_type.tolist()),
            bool(qc_only),
        )
        result = np.asarray(r_result, dtype=float)
    return pd.DataFrame(
        result, index=intensity_df.index, columns=intensity_df.columns
    )
