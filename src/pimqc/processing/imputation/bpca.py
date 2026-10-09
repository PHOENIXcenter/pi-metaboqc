"""Reconstruct missing values with a centered Bayesian PCA model.

The implementation follows the pcaMethods BPCA update structure and restores
observed column means after fitting. Input validation and fallback behavior
are kept explicit rather than silently changing the numerical input domain.
"""

from typing import Any

import numpy as np


class BayesianPCAImputer:
    """Estimate missing entries in an observation-by-variable matrix.

    The model uses the initialization, Bayesian updates, and precision-change
    stopping rule of pcaMethods BPCA. Its ``pca(center=TRUE, scale="none")``
    preprocessing is part of this estimator: observed column means are removed
    *before* zero-filling for the initial covariance. Reconstruction uses the
    final scores and updated loadings, then restores those column means.

    Entirely missing variables have no estimable mean or loading. They receive
    the global observed mean (zero for entirely missing input) and do not enter
    the model. One-dimensional inputs receive observed column means.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        threshold: float = 1e-4,
    ) -> None:
        """Initialize BPCA model settings."""
        self.n_components = max(1, int(n_components))
        self.max_iter = max(1, int(max_iter))
        self.threshold = float(threshold)

    @staticmethod
    def _safe_inverse(mat: np.ndarray) -> np.ndarray:
        """Invert a small matrix with pseudo-inverse fallback."""
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(mat)

    def _initialize_model(self, y: np.ndarray) -> dict[str, Any]:
        """Initialize from centered data with at least one value per column."""
        rows, cols = y.shape
        nans = np.isnan(y)
        yest = np.where(nans, 0.0, y)
        comps = min(self.n_components, rows, cols)
        cov_y = np.atleast_2d(np.cov(yest, rowvar=False))
        u, s, _ = np.linalg.svd(cov_y, full_matrices=False)
        s = np.maximum(s[:comps], 0.0)
        pa = u[:, :comps] * np.sqrt(s)
        residual_var = float(np.trace(cov_y) - np.sum(s))
        tau = 1.0 / residual_var if residual_var > 1e-10 else 1e10
        tau = float(np.clip(tau, 1e-10, 1e10))

        galpha0 = 1e-10
        balpha0 = 1.0
        alpha_denom = tau * np.diag(pa.T @ pa) + 2 * galpha0 / balpha0
        alpha = (2 * galpha0 + cols) / alpha_denom
        return {
            "rows": rows,
            "cols": cols,
            "comps": comps,
            "yest": yest.copy(),
            "row_miss": np.flatnonzero(nans.any(axis=1)),
            "row_nomiss": np.flatnonzero(~nans.any(axis=1)),
            "nans": nans,
            "mean": np.nanmean(y, axis=0),
            "pa": pa,
            "tau": tau,
            "scores": np.zeros((rows, comps), dtype=float),
            "galpha0": galpha0,
            "balpha0": balpha0,
            "alpha": alpha,
            "gmu0": 0.001,
            "btau0": 1.0,
            "gtau0": 1e-10,
            "sigw": np.eye(comps),
        }

    def _do_step(self, model: dict[str, Any], y: np.ndarray) -> dict[str, Any]:
        """Perform one Bayesian update, including missing-value uncertainty."""
        rows = model["rows"]
        cols = model["cols"]
        comps = model["comps"]
        pa = model["pa"]
        tau = model["tau"]
        mean = model["mean"]
        nans = model["nans"]
        scores = np.zeros((rows, comps), dtype=float)
        t_mat = np.zeros((cols, comps), dtype=float)
        tr_s = 0.0
        rx = np.eye(comps) + tau * (pa.T @ pa) + model["sigw"]
        rx_inv = self._safe_inverse(rx)

        idx_nomiss = model["row_nomiss"]
        if len(idx_nomiss) > 0:
            dy = y[idx_nomiss, :] - mean
            x = tau * rx_inv @ pa.T @ dy.T
            t_mat += dy.T @ x.T
            tr_s += float(np.sum(dy * dy))
            scores[idx_nomiss, :] = x.T

        for i in model["row_miss"]:
            missing = nans[i, :]
            observed = ~missing
            dyo = y[i, observed] - mean[observed]
            wm = pa[missing, :]
            wo = pa[observed, :]
            rx_obs_inv = self._safe_inverse(rx - tau * (wm.T @ wm))
            x = rx_obs_inv @ (tau * wo.T @ dyo)
            dy_full = np.zeros(cols, dtype=float)
            dy_full[observed] = dyo
            dy_full[missing] = wm @ x
            model["yest"][i, :] = dy_full + mean
            t_mat += np.outer(dy_full, x)
            t_mat[missing, :] += wm @ rx_obs_inv
            tr_s += float(
                dy_full @ dy_full
                + missing.sum() / tau
                + np.trace(wm @ rx_obs_inv @ wm.T)
            )
            scores[i, :] = x

        t_mat /= rows
        tr_s /= rows
        dw = (
            rx_inv
            + tau * t_mat.T @ pa @ rx_inv
            + np.diag(model["alpha"]) / rows
        )
        dw_inv = self._safe_inverse(dw)
        pa_new = t_mat @ dw_inv
        tau_num = cols + 2 * model["gtau0"] / rows
        tau_den = (
            tr_s
            - np.trace(t_mat.T @ pa_new)
            + (
                float(np.dot(mean, mean)) * model["gmu0"]
                + 2 * model["gtau0"] / model["btau0"]
            ) / rows
        )
        tau_new = float(tau_num / max(float(tau_den), 1e-12))
        tau_new = float(np.clip(tau_new, 1e-10, 1e10))
        sigw_new = dw_inv * (cols / rows)
        alpha_denom = (
            tau_new * np.diag(pa_new.T @ pa_new)
            + np.diag(sigw_new)
            + 2 * model["galpha0"] / model["balpha0"]
        )
        model.update(
            scores=scores,
            pa=pa_new,
            tau=tau_new,
            sigw=sigw_new,
            alpha=(2 * model["galpha0"] + cols)
            / np.maximum(alpha_denom, 1e-12),
        )
        return model

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit centered BPCA and replace only missing entries.

        ``n_iter_`` and ``converged_`` report the precision stopping rule;
        reaching ``max_iter`` still returns that iteration's reconstruction.
        No positivity constraint is part of the Gaussian BPCA model.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("BPCA input must be a 2D matrix.")
        if np.isinf(y).any():
            raise ValueError("BPCA input must contain finite values or NaN.")
        self.n_iter_ = 0
        self.converged_ = False
        missing = np.isnan(y)
        if not missing.any() or not y.size:
            return y.copy()

        counts = (~missing).sum(axis=0)
        total_count = int(counts.sum())
        global_mean = float(np.nansum(y) / total_count) if total_count else 0.0
        center = np.divide(
            np.nansum(y, axis=0), counts,
            out=np.full(y.shape[1], global_mean), where=counts > 0,
        )
        result = np.where(missing, center, y)
        active = counts > 0
        if y.shape[0] < 2 or active.sum() < 2:
            return result

        # pcaMethods::pca centers before BPCA_initmodel zero-fills its input.
        centered = y[:, active] - center[active]
        if not np.any(np.nan_to_num(centered)):
            return result
        model = self._initialize_model(centered)
        tau_old = 1000.0
        for step in range(1, self.max_iter + 1):
            model = self._do_step(model, centered)
            self.n_iter_ = step
            if step % 10 == 0:
                dtau = abs(np.log10(model["tau"]) - np.log10(tau_old))
                if dtau < self.threshold:
                    self.converged_ = True
                    break
                tau_old = model["tau"]

        # completeObs uses fitted scores and the *updated* loadings. yest is
        # an E-step working estimate using the previous iteration's loadings.
        reconstructed = model["scores"] @ model["pa"].T + center[active]
        if not np.isfinite(reconstructed).all():
            raise FloatingPointError("BPCA reconstruction is nonfinite.")
        result[:, active] = np.where(
            missing[:, active], reconstructed, y[:, active]
        )
        return result
