"""Call original VSN independently to check the production R adapter.

This helper does not compare the Python estimator or load research test modules.
Stage-level reference tests use it to verify original-package output fidelity.
"""

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def run_r_vsn(df_input: pd.DataFrame) -> pd.DataFrame:
    """Execute Bioconductor vsn2 via the rpy2 interface."""
    r_script = """
    function(df) {
        suppressWarnings(suppressPackageStartupMessages(library(vsn)))
        mat <- as.matrix(df)
        mat[is.nan(mat)] <- NA
        fit <- vsn2(mat, verbose=FALSE)
        res <- predict(fit, mat)
        return(as.data.frame(res))
    }
    """
    r_vsn_func = ro.r(r_script)

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df_result = r_vsn_func(df_input)

    r_df_result.index = df_input.index
    r_df_result.columns = df_input.columns
    return r_df_result
