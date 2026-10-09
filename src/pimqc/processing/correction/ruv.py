"""Implement RUV-III unwanted-variation removal for intensity matrices.

``RUVCorrector`` estimates unwanted factors from QC samples and control
features, applies the SVD-based projection, and returns corrected numerical
data. It does not depend on configuration, export, or visualization.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .missing_input import prepare_median_input


def _observed_control_support(
    values: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Require real varying observations within at least one replicate group."""
    supported = np.zeros(values.shape[0], dtype=bool)
    for group in np.unique(groups):
        block = values[:, groups == group]
        for feature, observed in enumerate(block):
            observed = observed[~np.isnan(observed)]
            if observed.size >= 2 and np.ptp(observed) > 0:
                supported[feature] = True
    return supported


class RUVCorrector:
    """QC/control-residual variant of RUV, not an exact original-R port."""

    def __init__(self, k: int = 3) -> None:
        """Initialize the unwanted-factor model.

        Args:
            k: Number of unwanted latent factors to remove.
        """
        self.k = k
        self.diagnostics: dict = {}

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        qc_mask: np.ndarray,
        control_features: pd.Index,
        blank_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """Fit RUV-III on eligible samples and transform the full matrix.

        Args:
            intensity_df: Feature-by-sample intensity matrix.
            qc_mask: Boolean mask identifying pooled-QC samples.
            control_features: Features used to estimate unwanted variation.
            blank_mask: Optional mask excluding blanks from model fitting.

        Returns:
            Mapping containing the corrected matrix and optional OOF result.

        Raises:
            ValueError: If controls or eligible fitting samples are missing.
        """

        logger.info(f"Executing RUV-III (k={self.k})...")
        self.diagnostics = {}
        if control_features.empty:
            raise ValueError("RUV-III requires at least one control feature.")

        Y_raw = intensity_df.T.values.astype(np.float64)
        n_samples, n_features = Y_raw.shape
        qc_mask = np.asarray(qc_mask, dtype=bool)
        if blank_mask is None:
            blank_mask = np.zeros(n_samples, dtype=bool)
        else:
            blank_mask = np.asarray(blank_mask, dtype=bool)
            if blank_mask.shape != qc_mask.shape:
                raise ValueError("blank_mask must match the sample dimension.")
        fit_mask = ~blank_mask
        if not np.any(fit_mask):
            raise ValueError("RUV-III requires at least one non-Blank sample.")

        # Nonpositive raw values are input-domain missingness.  Do not let
        # them enter the fitted model as artificial zeros; the output stage
        # will retain these positions as missing rather than restoring them.
        if np.isinf(Y_raw).any():
            raise ValueError("RUV-III intensities must not contain infinity.")
        safe_frame = intensity_df.astype(float).where(intensity_df > 0)
        filled, missing, adapter = prepare_median_input(
            safe_frame, fit_mask, scale="log1p"
        )
        if adapter["applied"]:
            logger.warning(
                "RUV temporarily median-filled {} missing fit cells in log "
                "space; original missing positions will be restored.",
                adapter["temporary_filled_cells"],
            )
        Y = filled.T.to_numpy(dtype=float)
        nan_mask = missing.T.to_numpy(dtype=bool)

        # Blank rows are held out of all fitted quantities.  QC rows share one
        # group, while each non-QC biological row remains its own RUV group.
        fit_indices = np.flatnonzero(fit_mask)
        group_ids = []
        uid_counter = 1
        for row_idx in fit_indices:
            is_qc = qc_mask[row_idx]
            if is_qc:
                group_ids.append(0)
            else:
                group_ids.append(uid_counter)
                uid_counter += 1

        n_groups = len(set(group_ids))
        M = np.zeros((len(fit_indices), n_groups), dtype=np.float64)
        for row_idx, g_id in enumerate(group_ids):
            M[row_idx, g_id] = 1.0

        group_sizes = M.T @ M
        Y_fit = Y[fit_mask, :]
        group_means = np.linalg.solve(group_sizes, M.T @ Y_fit)
        Y0 = Y_fit - (M @ group_means)

        # Filled values cannot establish control support in replicate groups.
        support = _observed_control_support(
            safe_frame.loc[:, fit_mask].to_numpy(dtype=float),
            np.asarray(group_ids),
        )
        ctl_mask = intensity_df.index.isin(control_features)
        supported_controls = ctl_mask & support
        if int(supported_controls.sum()) < self.k:
            raise ValueError(
                "RUV-III observed negative-control support is below k; "
                "each control needs varying real replicate observations."
            )
        Y0_ctl = Y0[:, ctl_mask]

        U, S, Vt = np.linalg.svd(Y0_ctl, full_matrices=False)
        if self.k > Y0_ctl.shape[0]:
            raise ValueError("RUV-III k exceeds the fitting sample count.")
        safe_k = self.k
        residual_rank = int(np.linalg.matrix_rank(Y0_ctl))
        self.diagnostics.update({
            "estimator": "QC control-residual SVD with recentering",
            "requested_k": int(self.k), "effective_k": int(safe_k),
            "control_feature_ids": intensity_df.index[ctl_mask].tolist(),
            "observed_supported_control_ids": (
                intensity_df.index[supported_controls].tolist()
            ),
            "observed_supported_control_count": int(
                supported_controls.sum()
            ),
            "control_residual_rank": residual_rank,
            "replicate_residual_df": int((qc_mask & fit_mask).sum()) - 1,
            "input_negative_count": int((Y_raw < 0).sum()),
            "input_zero_count": int((Y_raw == 0).sum()),
            "output_policy": "unmodified kernel; stage maps invalid to missing",
            "missing_input_adapter": adapter,
        })
        if residual_rank < safe_k:
            logger.warning(
                "RUV requested factors exceed the observed control rank; "
                "some fitted directions are unidentifiable."
            )
        alpha_ctl = Vt[:safe_k, :]

        W_fit = Y_fit[:, ctl_mask] @ alpha_ctl.T
        W_means = np.linalg.solve(group_sizes, M.T @ W_fit)
        W0 = W_fit - (M @ W_means)
        alpha_full = np.linalg.lstsq(W0, Y0, rcond=None)[0]

        # Freeze the fitted correction centre.  Blanks are projected with the
        # same alpha matrices and centre, but never update either quantity.
        correction_fit = W_fit @ alpha_full
        correction_center = np.mean(correction_fit, axis=0)
        W_all = Y[:, ctl_mask] @ alpha_ctl.T
        correction = W_all @ alpha_full - correction_center

        Y_corr_log = Y - correction
        with np.errstate(over="ignore", invalid="ignore"):
            Y_corrected = np.expm1(Y_corr_log)
        required = ~nan_mask
        self.diagnostics.update({
            "output_negative_count": int(
                ((Y_corrected < 0) & required).sum()
            ),
            "output_zero_count": int(
                ((Y_corrected == 0) & required).sum()
            ),
            "output_nonfinite_count": int(
                (~np.isfinite(Y_corrected) & required).sum()
            ),
        })
        if self.diagnostics["output_negative_count"] or (
            self.diagnostics["output_nonfinite_count"]
        ):
            logger.warning(
                "Native RUV returned invalid raw intensities; the correction "
                "stage maps these to missing. Counts remain in diagnostics."
            )

        if nan_mask.any():
            Y_corrected[nan_mask] = np.nan

        res_df_full = intensity_df.copy()
        res_df_full.iloc[:, :] = Y_corrected.T

        return {"RUV-III": (res_df_full, None)}
