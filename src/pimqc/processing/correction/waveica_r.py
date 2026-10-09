"""Run the author's installed WaveICA 2.0 package through optional rpy2.

The adapter orders and transposes input/output around the original R function.
Missing cells receive temporary non-Blank feature medians on the raw scale,
then are restored after correction. The original R function is unchanged.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ...constants import DEFAULT_RANDOM_SEED
from ..r_backend import run_r_function
from .missing_input import prepare_median_input


_WAVEICA_R_CODE = """
function(mat, params) {
    dependencies <- c("waveslim", "parallel", "ica", "mgcv",
                      "JADE", "corpcor")
    for (package in dependencies) {
        if (!requireNamespace(package, quietly=TRUE)) {
            stop(paste("WaveICA2.0 requires the R package", package))
        }
    }
    previous_options <- options(mc.cores=1L)
    on.exit(options(previous_options), add=TRUE)
    input <- t(mat)
    colnames(input) <- paste0("feature_", seq_len(ncol(input)))
    rownames(input) <- paste0("sample_", seq_len(nrow(input)))
    result <- WaveICA2.0::WaveICA_2.0(
        data=input, wf="haar", Injection_Order=params$Injection_Order,
        alpha=params$alpha, Cutoff=params$Cutoff, K=as.integer(params$K)
    )
    if (!is.matrix(result$data_wave)) {
        stop("WaveICA2.0 did not return the expected data_wave matrix")
    }
    t(result$data_wave)
}
"""


class WaveICARCorrector:
    """Call original WaveICA2.0 on chronologically ordered non-Blank samples.

The upstream function has no frozen-model prediction interface. Blank samples
are therefore excluded completely and left unchanged, rather than estimated
with a different algorithm. At least ten uniquely ordered non-Blank samples
are required by the original GAM smoother's default basis dimension.
"""

    def __init__(
        self,
        n_components: int = 10,
        cutoff: float = 0.1,
        alpha: float = 0.0,
        random_state: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        """Validate original-package options without initializing R."""
        if (
            isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, Integral)
            or n_components < 2
        ):
            raise ValueError("R WaveICA n_components must be an integer >= 2.")
        for name, value in (("cutoff", cutoff), ("alpha", alpha)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"R WaveICA {name} must lie in [0, 1].")
        self.n_components = int(n_components)
        self.cutoff = float(cutoff)
        self.alpha = float(alpha)
        self.random_state = random_state
        self.provenance: dict[str, Any] = {}

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        order_array: np.ndarray,
        blank_mask: np.ndarray | None = None,
    ) -> dict[str, tuple[pd.DataFrame, None]]:
        """Return a full-fit stage with original labels and frozen Blanks."""
        if not isinstance(intensity_df, pd.DataFrame) or intensity_df.empty:
            raise ValueError("R WaveICA requires a nonempty DataFrame.")
        if intensity_df.shape[0] < 2:
            raise ValueError("R WaveICA requires at least two features.")
        n_samples = intensity_df.shape[1]
        order = np.asarray(order_array, dtype=float)
        if order.shape != (n_samples,):
            raise ValueError(
                "R WaveICA injection orders must align to samples."
            )
        blanks = (
            np.zeros(n_samples, dtype=bool)
            if blank_mask is None
            else np.asarray(blank_mask)
        )
        if blanks.shape != (n_samples,) or blanks.dtype.kind != "b":
            raise ValueError("R WaveICA blank_mask must be an aligned boolean.")
        retained = np.flatnonzero(~blanks)
        retained_order = order[retained]
        if retained.size < 10:
            raise ValueError(
                "R WaveICA requires at least ten non-Blank samples for the "
                "original GAM smoother."
            )
        if not np.isfinite(retained_order).all():
            raise ValueError("R WaveICA requires finite non-Blank orders.")
        if np.unique(retained_order).size != retained.size:
            raise ValueError(
                "R WaveICA requires globally unique injection orders; "
                "batch-local order resets cannot define sample chronology."
            )
        filled, missing, missing_diagnostics = prepare_median_input(
            intensity_df, ~blanks, scale="raw"
        )
        missing_count = int(missing.iloc[:, ~blanks].to_numpy().sum())
        if missing_count:
            logger.warning(
                "R WaveICA temporarily median-filled {} missing cells; "
                "original missing positions will be restored.",
                missing_count,
            )
        sorted_positions = retained[np.argsort(retained_order, kind="stable")]
        sorted_frame = filled.iloc[:, sorted_positions].copy()
        corrected, provenance = run_r_function(
            sorted_frame,
            method="WaveICA 2.0",
            package="WaveICA2.0",
            function="WaveICA2.0::WaveICA_2.0",
            code=_WAVEICA_R_CODE,
            parameters={
                "Injection_Order": order[sorted_positions].tolist(),
                "wf": "haar",
                "alpha": self.alpha,
                "Cutoff": self.cutoff,
                "K": self.n_components,
                "mc.cores": 1,
            },
            seed=self.random_state,
            defer_output_domain=True,
            transforms=(
                "Blank samples excluded and retained unchanged: original "
                "WaveICA2.0 has no frozen-model prediction interface",
                "stable sort non-Blank samples by global injection order",
                "transpose to samples by features with synthetic R dimnames",
                "raw input scale preserved; temporary feature median "
                "filling from non-Blank samples, then restore original NA; "
                "upstream WaveICA does not natively support NA",
                "transpose result and restore original sample positions",
            ),
        )
        result = intensity_df.astype(float).copy(deep=True)
        corrected_values = corrected.to_numpy(dtype=float).copy()
        corrected_values[missing.iloc[:, sorted_positions].to_numpy()] = np.nan
        result.iloc[:, sorted_positions] = corrected_values
        provenance["missing_input_adapter"] = missing_diagnostics
        provenance["input_policy"] = {
            "fitted_sample_count": int(retained.size),
            "excluded_blank_positions": np.flatnonzero(blanks).tolist(),
            "sorted_sample_positions": sorted_positions.tolist(),
            "blank_output": "unchanged",
            "effective_component_limit": min(
                self.n_components, intensity_df.shape[0], int(retained.size)
            ),
            "wavelet_levels": int(np.floor(np.log2(retained.size))),
            "oof": "unavailable",
        }
        self.provenance = provenance
        return {"WaveICA corrected": (result, None)}
