"""Call the original, complete-matrix RUV-III implementation in ``ruv``.

This adapter is not RUV-III-C. It requires explicit negative controls and a
replicate design. Temporary feature medians complete the original package's
input, with missing positions restored after correction.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ...constants import DEFAULT_RANDOM_SEED
from ..r_backend import run_r_function
from .missing_input import prepare_median_input
from .ruv import _observed_control_support


_RUVIII_CODE = """
function(mat, params) {
    M <- 1 * outer(params$replicate_codes, unique(params$replicate_codes), "==")
    result <- ruv::RUVIII(
        Y=t(mat), M=M, ctl=as.logical(params$controls),
        k=as.integer(params$k), eta=NULL, include.intercept=TRUE,
        average=FALSE, fullalpha=NULL, return.info=FALSE, inputcheck=TRUE
    )
    t(result)
}
"""


class RUVIIIRCorrector:
    """Apply original RUV-III to log2(x + 1) non-Blank intensities.

    Blank measurements remain unchanged: the original function has no public
    fit/predict interface. Defaults assume all pooled QCs are technical
    replicates, while every other sample is its own replicate group. Supply
    ``replicate_groups`` to replace this assumption with a real study design.
    Biological condition labels must not be substituted for replicate IDs.
    """

    def __init__(
        self, k: int = 3, random_state: int = DEFAULT_RANDOM_SEED
    ) -> None:
        """Record options without importing R or changing its random state."""
        if (
            isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer))
            or k < 1
        ):
            raise ValueError("RUV-III k must be a positive integer.")
        self.k = int(k)
        self.random_state = random_state
        self.provenance: dict[str, Any] = {}

    @staticmethod
    def _mask(values: Any, count: int, name: str) -> np.ndarray:
        """Reject malformed masks instead of silently coercing strings."""
        mask = np.asarray(values)
        if mask.shape != (count,) or mask.dtype.kind != "b":
            raise ValueError(f"{name} must be a boolean mask for every sample.")
        return mask

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        qc_mask: np.ndarray,
        control_features: pd.Index,
        blank_mask: np.ndarray | None = None,
        replicate_groups: pd.Series | np.ndarray | list[Any] | None = None,
    ) -> dict[str, tuple[pd.DataFrame, None]]:
        """Return the direct original-package correction and no OOF matrix.

        Replicate IDs, when supplied, must align with all input columns. A
        Series must have exactly the same index as those columns. Missing
        replicate IDs are allowed only for excluded Blank columns.
        """
        self.provenance = {}
        if not isinstance(intensity_df, pd.DataFrame) or intensity_df.empty:
            raise ValueError("RUV-III requires a nonempty intensity dataframe.")
        if (
            not intensity_df.index.is_unique
            or not intensity_df.columns.is_unique
        ):
            raise ValueError("RUV-III requires unique feature and sample IDs.")
        count = intensity_df.shape[1]
        qc = self._mask(qc_mask, count, "qc_mask")
        blank = (
            np.zeros(count, dtype=bool)
            if blank_mask is None
            else self._mask(blank_mask, count, "blank_mask")
        )
        if (qc & blank).any():
            raise ValueError("QC and Blank masks must not overlap.")
        eligible = ~blank
        if eligible.sum() < 2:
            raise ValueError("RUV-III requires at least two non-Blank samples.")
        values = intensity_df.loc[:, eligible].to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(
                "Original RUV-III requires nonnegative intensities without "
                "infinity; NaNs use temporary feature-median completion."
            )
        controls = pd.Index(control_features)
        if controls.empty or not controls.is_unique:
            raise ValueError("RUV-III requires unique negative-control IDs.")
        unknown = controls.difference(intensity_df.index)
        if not unknown.empty:
            raise ValueError(f"Unknown RUV-III controls: {unknown.tolist()}.")
        control_mask = intensity_df.index.isin(controls)
        if replicate_groups is None:
            groups = np.arange(count, dtype=int).astype(object)
            groups[qc] = -1
            group_values = groups[eligible]
            design_source = "pooled_QC_replicates_other_samples_singletons"
        else:
            if isinstance(replicate_groups, pd.Series) and not (
                replicate_groups.index.equals(intensity_df.columns)
            ):
                raise ValueError(
                    "RUV-III replicate Series index must match sample columns."
                )
            groups = np.asarray(replicate_groups, dtype=object)
            if groups.shape != (count,):
                raise ValueError(
                    "RUV-III replicate IDs must cover all samples."
                )
            group_values = groups[eligible]
            if pd.isna(group_values).any():
                raise ValueError("Non-Blank RUV-III replicate IDs are missing.")
            strings = [isinstance(value, str) for value in group_values]
            numbers = [
                isinstance(value, (int, float, np.number))
                and not isinstance(value, (bool, np.bool_))
                for value in group_values
            ]
            if not (all(strings) or all(numbers)):
                raise ValueError(
                    "RUV-III replicate IDs must be uniformly strings or "
                    "numeric scalars, without mixed types."
                )
            if all(strings) and any(
                not value.strip() for value in group_values
            ):
                raise ValueError("RUV-III replicate IDs must not be empty.")
            if (
                all(numbers)
                and not np.isfinite(np.asarray(group_values, dtype=float)).all()
            ):
                raise ValueError(
                    "RUV-III numeric replicate IDs must be finite."
                )
            design_source = "explicit_technical_replicate_groups"
        codes, labels = pd.factorize(group_values, sort=False)
        residual_df = int(eligible.sum()) - len(labels)
        if self.k > min(residual_df, int(control_mask.sum())):
            raise ValueError(
                "RUV-III k exceeds replicate residual degrees of freedom "
                "or the number of negative controls. Add genuine replicate "
                "information/controls or explicitly choose a smaller k."
            )
        supported_controls = control_mask & _observed_control_support(
            values, codes
        )
        if int(supported_controls.sum()) < self.k:
            raise ValueError(
                "RUV-III observed negative-control support is below k; "
                "each control needs varying real replicate observations."
            )
        filled, missing, adapter = prepare_median_input(
            intensity_df, eligible, scale="log1p"
        )
        if adapter["applied"]:
            logger.warning(
                "R RUV-III temporarily median-filled {} missing fit cells "
                "in log space; original missing positions will be restored.",
                adapter["temporary_filled_cells"],
            )
        log_values = (
            filled.loc[:, eligible].to_numpy(dtype=float) / np.log(2.0)
        )
        residual_controls = log_values[control_mask].T.copy()
        for group in np.unique(codes):
            selected = codes == group
            residual_controls[selected] -= residual_controls[selected].mean(
                axis=0
            )
        control_rank = int(np.linalg.matrix_rank(residual_controls))
        if self.k > control_rank:
            raise ValueError(
                "RUV-III negative-control replicate residual rank is below k."
            )
        params = {
            "k": self.k,
            "replicate_codes": codes.tolist(),
            "controls": control_mask.tolist(),
            "eta": None,
            "include.intercept": True,
            "average": False,
            "fullalpha": None,
            "return.info": False,
            "inputcheck": True,
        }
        logged = pd.DataFrame(
            log_values,
            index=intensity_df.index,
            columns=intensity_df.columns[eligible],
        )
        corrected_log, provenance = run_r_function(
            logged,
            method="RUV-III",
            package="ruv",
            function="ruv::RUVIII",
            code=_RUVIII_CODE,
            parameters=params,
            seed=self.random_state,
            defer_output_domain=True,
            transforms=(
                "exclude Blank samples; retain original Blank intensities",
                "log2(raw + 1); temporary non-Blank observed feature medians",
                "transpose to samples by features for RUVIII",
                "exp2(original RUVIII output) - 1; no recentering or clipping",
                "restore input missing positions",
            ),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = np.exp2(corrected_log.to_numpy(dtype=float)) - 1.0
        corrected[missing.loc[:, eligible].to_numpy(dtype=bool)] = np.nan
        output = intensity_df.astype(float).copy(deep=True)
        output.loc[:, eligible] = corrected
        output.attrs = copy.deepcopy(intensity_df.attrs)
        self.provenance = {
            **provenance,
            "replicate_design": design_source,
            "replicate_labels": [str(label) for label in labels],
            "fit_sample_ids": [str(item) for item in logged.columns],
            "control_feature_ids": [str(item) for item in controls],
            "observed_supported_control_ids": [
                str(item) for item in intensity_df.index[supported_controls]
            ],
            "observed_supported_control_count": int(
                supported_controls.sum()
            ),
            "missing_input_adapter": adapter,
            "replicate_residual_df": residual_df,
            "control_residual_rank": control_rank,
            "blank_policy": "excluded_from_fit_and_retained_unchanged",
            "blank_sample_ids": [
                str(item) for item in intensity_df.columns[blank]
            ],
            "validation": "full_fit_only; original package has no predict",
        }
        return {"RUV corrected": (output, None)}
