"""Route-aware missing-value imputation and candidate selection.

MissingValueImputer applies special handling (S-route) such as QRILC or
censored estimates and reconstruction (R-route) using Median, MinProb, KNN,
LLS, BPCA, or metric-guided candidate selection. Route labels are operational,
not inferred statistical missingness mechanisms.
It evaluates masked reconstruction, distribution fidelity, and sample-structure
preservation, then records the selected strategy and imputed stage metrics.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import KNNImputer

from ...config import resolve_stage_config
from ...config.schema import normalize_imputation_route_options
from ...constants import DEFAULT_RANDOM_SEED
from ...core import DatasetProcessor, MetaboDataset
from ...core.routes import (
    R_ROUTE,
    S_ROUTE,
    ROUTE_DEFINITIONS,
    routes_from_metadata,
)
from ...plotting.payloads import ImputationPlotPayload, snapshot_dataset
from ...runtime import log_execution_time
from ...statistics import metrics as su
from ...statistics import sample_structure as structure_stats
from ...statistics import selection as selection_utils
from ..audit import ImputationAuditPayload
from ..numeric_domain import repair_imputed_log_values
from ..stage import StageResult
from .bpca import BayesianPCAImputer
from .methods import IMPUTATION_METHODS
from .qrilc import impute_qrilc
from .runner import ImputationStageRunner


class MissingValueImputer(DatasetProcessor):
    """Missing value imputation engine with hybrid stratified evaluation."""

    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "mar_method",
            "mnar_method",
            "r_route_method",
            "s_route_method",
            "mnar_fraction",
            "knn_neighbors",
            "lls_neighbors",
            "bpca_components",
            "bpca_max_iter",
            "bpca_tol",
            "sim_mask_ratio",
            "global_seed",
            "implementation",
        }
    )

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Optional[Dict[str, Any]] = None,
        mar_method: Optional[str] = None,
        mnar_method: Optional[str] = None,
        mnar_fraction: Optional[float] = None,
        knn_neighbors: Optional[int] = None,
        lls_neighbors: Optional[int] = None,
        bpca_components: Optional[int] = None,
        bpca_max_iter: Optional[int] = None,
        bpca_tol: Optional[float] = None,
        sim_mask_ratio: Optional[float] = None,
        global_seed: Optional[int] = None,
        implementation: Optional[str] = None,
        *,
        r_route_method: Optional[str] = None,
        s_route_method: Optional[str] = None,
    ) -> None:
        """Initialize the missing-value imputation engine.

        Args:
            data: Explicit dataset to impute.
            pipeline_params: Global configuration dictionary.
            mar_method: Legacy alias for ``r_route_method``.
            mnar_method: Legacy alias for ``s_route_method``.
            r_route_method: Reconstruction-route method or ``Auto``.
            s_route_method: Special-handling method, normally QRILC.
            mnar_fraction: Legacy name for the S-route LOD multiplier.
            knn_neighbors: Number of neighbors for the KNN algorithm.
            lls_neighbors: Number of neighbors for the LLS algorithm.
            bpca_components: Number of principal components for BPCA.
            bpca_max_iter: Maximum BPCA EM/Bayesian update steps.
            bpca_tol: BPCA precision-change convergence threshold.
            sim_mask_ratio: Ratio for simulated masking during evaluation.
            implementation: ``python`` or optional original-package ``r``.
                The R path supports R-route BPCA/Auto and S-route QRILC.
        """
        super().__init__(data)

        imp_configs = resolve_stage_config(
            pipeline_params,
            "MissingValueImputer",
            {
                "mar_method": "Auto",
                "mnar_method": "QRILC",
                "mnar_fraction": 0.5,
                "knn_neighbors": 5,
                "lls_neighbors": 15,
                "bpca_components": 2,
                "bpca_max_iter": 100,
                "bpca_tol": 1e-4,
                "sim_mask_ratio": 0.05,
                "implementation": "python",
            },
            normalize_imputation_route_options(
                {
                    "mar_method": mar_method,
                    "mnar_method": mnar_method,
                    "r_route_method": r_route_method,
                    "s_route_method": s_route_method,
                    "mnar_fraction": mnar_fraction,
                    "knn_neighbors": knn_neighbors,
                    "lls_neighbors": lls_neighbors,
                    "bpca_components": bpca_components,
                    "bpca_max_iter": bpca_max_iter,
                    "bpca_tol": bpca_tol,
                    "sim_mask_ratio": sim_mask_ratio,
                    "global_seed": global_seed,
                    "implementation": implementation,
                }
            ),
        )

        self.config.update(imp_configs)

    def _validate_implementation(self) -> None:
        """Reject unsupported R choices without substituting Python methods."""
        implementation = str(
            self.config.get("implementation", "python")
        ).lower()
        if implementation not in {"python", "r"}:
            raise ValueError("implementation must be either 'python' or 'r'.")
        self.config["implementation"] = implementation
        if implementation != "r":
            return
        if self.dataset.context.is_logged or self.dataset.context.is_scaled:
            raise ValueError(
                "R imputation requires unlogged, unscaled intensities; "
                "the stage applies log2(x + 1) before the original R call."
            )
        mar_key = IMPUTATION_METHODS.resolve(self.config["mar_method"]).key
        if mar_key not in {"BPCA", "AUTO"}:
            raise ValueError(
                "implementation='r' requires mar_method='BPCA' or 'Auto'; "
                "use r_route_method='BPCA' or 'Auto'. Other explicit "
                "R-route methods have no original R adapter."
            )
        if str(self.config["mnar_method"]).upper() != "QRILC":
            raise ValueError(
                "implementation='r' requires mnar_method='QRILC'; "
                "the preferred alias is s_route_method='QRILC'. "
                "Python LOD heuristics are not silently substituted."
            )

    def _method_implementation(self, method: str) -> str:
        """Resolve AUTO substitutions before execution, never after failure."""
        key = str(method).replace("-", "").upper()
        if self.config.get("implementation") == "r" and key in {
            "BPCA",
            "QRILC",
        }:
            return "r"
        return "python"

    def _impute_by_r(
        self,
        df_log: pd.DataFrame,
        *,
        method: str,
        global_seed: int,
    ) -> pd.DataFrame:
        """Call the original R method on log-scale data, without fallback."""
        if not df_log.isna().any().any():
            return df_log.copy(deep=True)
        from ..r_backend import run_r_method

        if method == "BPCA":
            parameters = {
                "n_components": self.config.get("bpca_components", 2),
                "max_iter": self.config.get("bpca_max_iter", 100),
                "threshold": self.config.get("bpca_tol", 1e-4),
            }
        elif method == "QRILC":
            parameters = {"tune_sigma": 1.0}
        else:
            raise ValueError(f"Unsupported R imputation method: {method}.")
        result, provenance = run_r_method(
            method, df_log, seed=global_seed, **parameters
        )
        roles = []
        if len(self.qc_data.columns.intersection(df_log.columns)):
            roles.append("QC")
        if len(self.actual_data.columns.intersection(df_log.columns)):
            roles.append("Sample")
        self._implementation_provenance.append(
            {
                **provenance,
                "phase": self._imputation_phase,
                "sample_roles": roles,
                "input_transform": "log2(x + 1); zero treated as missing",
                "output_transform": "exp2(x) - 1",
                "feature_count": len(df_log.index),
                "sample_count": len(df_log.columns),
            }
        )
        return result

    # =========================================================================
    # Imputation-related Metrics
    # =========================================================================
    def calc_imp_quality_metrics(
        self, raw_obj: pd.DataFrame, imp_obj: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Calculate post-imputation distribution QA metrics for the passport.

        Computes Jensen-Shannon and Wasserstein distances for final QC and
        sample
        matrices separately. These values are retained as post-imputation QA
        diagnostics; AUTO selection instead uses masked-value metrics.

        Returns:
        Contains the quantified post-imputation QA metrics.

        """
        metrics = {
            "JSD": {"QC": {}, "Sample": {}},
            "Wasserstein": {"QC": {}, "Sample": {}},
            "JSD_Score": {"QC": {}, "Sample": {}},
            "Wasserstein_Score": {"QC": {}, "Sample": {}},
        }
        raw_log = np.log2(raw_obj.astype(float).replace({0: np.nan}) + 1.0)
        imp_log = np.log2(imp_obj.astype(float) + 1.0)

        sample_type = imp_obj.attrs.get("sample_type", "Sample Type")
        roles = imp_obj.attrs.get("sample_dict", {})
        qc_mask = imp_obj.columns.get_level_values(sample_type) == roles.get(
            "QC sample", "QC"
        )
        sample_mask = imp_obj.columns.get_level_values(
            sample_type
        ) == roles.get("Actual sample", "Sample")
        qc_cols = imp_obj.columns[qc_mask].intersection(raw_log.columns)
        sam_cols = imp_obj.columns[sample_mask].intersection(raw_log.columns)

        for grp, cols in [("QC", qc_cols), ("Sample", sam_cols)]:
            if cols.empty:
                continue

            r_slice = raw_log[cols].values.flatten()
            i_slice = imp_log[cols].values.flatten()

            # Data Before Imputation (All)
            obs = r_slice[~np.isnan(r_slice)]

            # Data After Imputation (All)
            imp_all = i_slice[~np.isnan(i_slice)]

            # Imputed Data (Patches only)
            mask_missing = np.isnan(r_slice)
            imp_only = i_slice[mask_missing]
            imp_only = imp_only[~np.isnan(imp_only)]

            if len(obs) > 0 and len(imp_all) > 0:
                dist_1 = su.calc_distribution_distance_metrics(obs, imp_all)
                jsd_val = su.finite_or_nan(dist_1.get("jsd"))
                wd_val = su.finite_or_nan(dist_1.get("wasserstein_normalized"))
                metrics["JSD"][grp]["Before vs After (All)"] = jsd_val
                metrics["Wasserstein"][grp]["Before vs After (All)"] = wd_val
                if np.isfinite(jsd_val):
                    metrics["JSD_Score"][grp]["Before vs After (All)"] = float(
                        np.clip(1.0 - jsd_val, 0.0, 1.0)
                    )
                if np.isfinite(wd_val):
                    metrics["Wasserstein_Score"][grp][
                        "Before vs After (All)"
                    ] = float(1.0 / (1.0 + max(wd_val, 0.0)))

            if len(obs) > 0 and len(imp_only) > 0:
                dist_2 = su.calc_distribution_distance_metrics(obs, imp_only)
                jsd_val = su.finite_or_nan(dist_2.get("jsd"))
                wd_val = su.finite_or_nan(dist_2.get("wasserstein_normalized"))
                metrics["JSD"][grp]["Before vs Imputed Only"] = jsd_val
                metrics["Wasserstein"][grp]["Before vs Imputed Only"] = wd_val
                if np.isfinite(jsd_val):
                    metrics["JSD_Score"][grp]["Before vs Imputed Only"] = float(
                        np.clip(1.0 - jsd_val, 0.0, 1.0)
                    )
                if np.isfinite(wd_val):
                    metrics["Wasserstein_Score"][grp][
                        "Before vs Imputed Only"
                    ] = float(1.0 / (1.0 + max(wd_val, 0.0)))

        return metrics

    # =========================================================================
    # Core Algorithms (Log2 Space)
    # =========================================================================
    @staticmethod
    def impute_by_constant(
        df_log: pd.DataFrame, fraction: float = 1.0, imp_mode: str = "row"
    ) -> pd.DataFrame:
        """Imputes missing values using a constant LOD heuristic.

        Args:
            df: The dataset (typically log-transformed).
            fraction: The heuristic multiplier (e.g., 0.5 for half-minimum).
            imp_mode: "row" (feature-wise), "column" (sample-wise), or "global".
            is_log2: If True, executes fractional math in linear space safely.

        Returns:
            Dataframe with constant imputation applied.
        """
        if imp_mode in ("row", "row-wise", "row min"):
            raw_mins = df_log.min(axis=1)
        elif imp_mode in ("column", "column-wise", "column min"):
            raw_mins = df_log.min(axis=0)
        else:  # elif imp_mode in ("global", "global min"):
            raw_mins = df_log.min().min()

        linear_mins = np.exp2(raw_mins) - 1.0
        target_mins = np.log2((linear_mins * fraction) + 1.0)

        # Broadcast the computed minimums to fill NaNs
        if imp_mode in ("row", "row-wise", "row min"):
            return df_log.apply(lambda x: x.fillna(target_mins[x.name]), axis=1)
        else:
            return df_log.fillna(target_mins)

    @staticmethod
    def impute_by_qrilc(
        df_log: pd.DataFrame,
        tune_sigma: float = 1.0,
        global_seed: int = DEFAULT_RANDOM_SEED,
    ) -> pd.DataFrame:
        """Impute missing values using QRILC logic for left-censored data.

        This follows the imputeLCMD::impute.QRILC orientation: for each sample
        column, fit observed sample quantiles against theoretical normal
        quantiles, then draw missing values from the left tail of the estimated
        censored distribution. The project matrix stores features in rows and
        samples in columns, matching imputeLCMD's input orientation.

        Ref:
            Missing value imputation approach for mass spectrometry-based
            metabolomics data (Scientific reports, 2018)
        The kernel preserves signed draws. The production stage applies its
        documented positive-output policy after reconstruction.

        """
        return impute_qrilc(df_log, tune_sigma=tune_sigma, seed=global_seed)

    @staticmethod
    def impute_by_knn(
        df_log: pd.DataFrame, n_neighbors: int = 5
    ) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbors algorithm."""
        # ``KNNImputer`` drops columns that are entirely missing.  Because the
        # matrix is transposed before fitting, those columns correspond to
        # all-missing features in the input and would otherwise make the
        # returned array one row shorter than ``df_log.index``.  Keep those
        # features in the working matrix and use the neutral log-space value
        # (zero) when no observed value is available for them.
        all_missing = df_log.isna().all(axis=1)
        fit_df = df_log.copy()
        if all_missing.any():
            fit_df.loc[all_missing, :] = 0.0

        # Scale neighbor count for isolated small groups.
        n_samples = fit_df.shape[1]
        safe_k = min(n_neighbors, n_samples - 1)

        if safe_k < 1:
            # Fallback to feature median if insufficient neighbors (e.g., n=1)
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(
                0.0
            )

        imputer = KNNImputer(n_neighbors=safe_k, weights="distance")
        arr_imp = imputer.fit_transform(fit_df.T).T
        arr_imp = np.where(df_log.isna().to_numpy(), arr_imp, df_log.to_numpy())

        return pd.DataFrame(arr_imp, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_lls(
        df_log: pd.DataFrame, n_neighbors: int = 15
    ) -> pd.DataFrame:
        """Impute missing values using Local Least Squares (LLS) regression.

        Finds 'k' complete features that are highly correlated with the target
        feature, and constructs a local linear regression model to predict
        the missing values.
        """
        arr_log = df_log.values
        res_arr = arr_log.copy()

        # Identify complete features to serve as the candidate neighbor pool
        complete_mask = ~np.isnan(arr_log).any(axis=1)
        complete_features = arr_log[complete_mask]
        n_complete = complete_features.shape[0]

        # Fallback: If dataset is too sparse and lacks complete features
        if n_complete < 2:
            logger.debug(
                "Insufficient complete features for LLS. "
                "Falling back to median."
            )
            return df_log.apply(lambda x: x.fillna(x.median()), axis=1).fillna(
                0.0
            )

        safe_k = min(n_neighbors, n_complete)

        for i in range(arr_log.shape[0]):
            row = arr_log[i]
            missing_mask = np.isnan(row)

            # Skip if no missing values
            if not missing_mask.any():
                continue

            obs_mask = ~missing_mask
            w_obs = row[obs_mask]

            # Fallback: Need at least 3 points for stable linear regression
            if obs_mask.sum() < 3:
                res_arr[i, missing_mask] = (
                    np.nanmedian(row) if obs_mask.sum() > 0 else 0.0
                )
                continue

            # Vectorized Pearson correlation to find closest complete
            # features
            A_obs = complete_features[:, obs_mask]

            w_mean = np.mean(w_obs)
            A_mean = np.mean(A_obs, axis=1, keepdims=True)

            w_centered = w_obs - w_mean
            A_centered = A_obs - A_mean

            cov = np.sum(A_centered * w_centered, axis=1)
            var_w = np.sum(w_centered**2)
            var_A = np.sum(A_centered**2, axis=1)

            denom = np.sqrt(var_w * var_A)
            corr = np.zeros_like(cov)
            valid_corr = denom > 1e-9
            # Use absolute correlation since negative correlation is also
            # useful for regression
            corr[valid_corr] = np.abs(cov[valid_corr] / denom[valid_corr])

            # Select top K neighbors
            top_k_idx = np.argsort(corr)[-safe_k:]

            # Construct matrices for Least Squares estimation
            # A_mat: neighbors' observed values (Shape: n_neighbors x
            # n_observed)
            A_mat = A_obs[top_k_idx]
            # B_mat: neighbors' values at target's missing positions
            B_mat = complete_features[top_k_idx][:, missing_mask]

            # Solve linear system: A_mat.T * x = w_obs
            try:
                # Used for numerical stability over matrix inverse
                x, _, _, _ = np.linalg.lstsq(A_mat.T, w_obs, rcond=None)

                # Predict: x.T * B_mat
                w_miss = x.T @ B_mat
                # Retain the model prediction; stage output policy is separate.
                res_arr[i, missing_mask] = w_miss
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular or highly collinear
                res_arr[i, missing_mask] = np.nanmedian(row)

        return pd.DataFrame(res_arr, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_bpca(
        df_log: pd.DataFrame,
        n_components: int = 2,
        max_iter: int = 100,
        threshold: float = 1e-4,
    ) -> pd.DataFrame:
        """Impute missing values using Bayesian PCA in log2 space.

        pcaMethods treats rows as observations and columns as variables. The
        project matrix stores features in rows and samples in columns, so this
        wrapper transposes the matrix before fitting BPCA and restores the
        original orientation afterward. Feature means are removed before the
        model is initialized. Observed entries and signed reconstructions are
        preserved: the Gaussian BPCA model has no positivity constraint.
        """
        if df_log.empty or not df_log.isna().any().any():
            return df_log.copy()

        arr = df_log.to_numpy(dtype=float)
        safe_components = max(
            1, min(int(n_components), arr.shape[0], arr.shape[1])
        )
        imputer = BayesianPCAImputer(
            n_components=safe_components,
            max_iter=max_iter,
            threshold=threshold,
        )

        # BPCA returns an imputed observation-by-variable matrix.
        arr_imp = imputer.fit_transform(arr.T).T

        return pd.DataFrame(arr_imp, index=df_log.index, columns=df_log.columns)

    @staticmethod
    def impute_by_minprob(
        df_log: pd.DataFrame,
        global_seed: int = DEFAULT_RANDOM_SEED,
    ) -> pd.DataFrame:
        """Impute using a normal distribution to simulate values below LOD.

        This method adopts a left-shifted Gaussian distribution (Perseus style)
        without hard clipping, preserving the natural variance of the unobserved
        low-abundance tail.
        """
        rng = np.random.default_rng(global_seed)
        res_df = df_log.copy()
        for col in res_df.columns:
            s = res_df[col]
            if s.isna().sum() == 0:
                continue

            valid = s.dropna()
            m, sd = valid.mean(), valid.std()

            # Shift the distribution leftward to simulate the missing tail
            # standard parameters: shift by -1.8 std, width of 0.3 std
            shift_mean = m - 1.8 * sd
            shift_std = max(0.3 * sd, 0.01)

            # Draw random values simulating noise below the detection limit
            drawn = rng.normal(
                loc=shift_mean, scale=shift_std, size=s.isna().sum()
            )

            res_df.loc[s.isna(), col] = drawn

        return res_df

    def _apply_isolated(
        self,
        df_slice: pd.DataFrame,
        imp_func: Callable[..., pd.DataFrame],
        **kwargs: object,
    ) -> pd.DataFrame:
        """Executes imputation independently on QC and biological samples.

        Prefer role-local evidence. If a role cannot fill a value, use a
        pooled fallback and record the resulting cross-role fills explicitly.
        """
        qc_cols = self.qc_data.columns.intersection(df_slice.columns)
        sam_cols = self.actual_data.columns.intersection(df_slice.columns)

        res_dfs = []
        global_fallback = None  # Lazy evaluation for edge cases

        # Impute QC using strictly QC context
        if not qc_cols.empty:
            res_qc = imp_func(df_slice[qc_cols], **kwargs)
            if res_qc.isna().any().any():
                global_fallback = imp_func(df_slice, **kwargs)
                self._record_isolation_fallback(
                    "QC", res_qc, global_fallback[qc_cols]
                )
                res_qc = res_qc.combine_first(global_fallback[qc_cols])
            res_dfs.append(res_qc)

        # Impute Samples using strictly Sample context
        if not sam_cols.empty:
            res_sam = imp_func(df_slice[sam_cols], **kwargs)
            if res_sam.isna().any().any():
                if global_fallback is None:
                    global_fallback = imp_func(df_slice, **kwargs)
                self._record_isolation_fallback(
                    "Sample", res_sam, global_fallback[sam_cols]
                )
                res_sam = res_sam.combine_first(global_fallback[sam_cols])
            res_dfs.append(res_sam)

        if not res_dfs:
            return df_slice

        # Reconstruct matrix ensuring original column order
        return pd.concat(res_dfs, axis=1)[df_slice.columns]

    def _record_isolation_fallback(
        self, role: str, local: pd.DataFrame, pooled: pd.DataFrame
    ) -> None:
        """Record values filled using cross-role rather than local evidence."""
        filled = int((local.isna() & pooled.notna()).sum().sum())
        if filled:
            records = self.__dict__.setdefault(
                "_isolation_fallback_records", []
            )
            records.append({"role": role, "filled_values": filled})

    def _repair_output(
        self,
        output: pd.DataFrame,
        training: pd.DataFrame,
        *,
        target_mask: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply the same train-only positive-output policy in every phase."""
        labels = training.columns.get_level_values(
            self.dataset.schema.sample_type
        ).to_numpy()
        return repair_imputed_log_values(
            output,
            training,
            role_labels=labels,
            target_mask=target_mask,
        )

    def _run_mar_method(
        self,
        training: pd.DataFrame,
        method: str,
        seed: int,
    ) -> pd.DataFrame:
        """Run a raw algorithm without the production output policy."""
        key = IMPUTATION_METHODS.resolve(method).key
        if self._method_implementation(key) == "r":
            return self._apply_isolated(
                training, self._impute_by_r, method=key, global_seed=seed
            )
        methods = {
            "MINPROB": (self.impute_by_minprob, {"global_seed": seed}),
            "KNN": (
                self.impute_by_knn,
                {"n_neighbors": self.config.get("knn_neighbors", 5)},
            ),
            "LLS": (
                self.impute_by_lls,
                {"n_neighbors": self.config.get("lls_neighbors", 15)},
            ),
            "BPCA": (
                self.impute_by_bpca,
                {
                    "n_components": self.config.get("bpca_components", 2),
                    "max_iter": self.config.get("bpca_max_iter", 100),
                    "threshold": self.config.get("bpca_tol", 1e-4),
                },
            ),
            "QRILC": (self.impute_by_qrilc, {"global_seed": seed}),
            "MEDIAN": (
                lambda frame: frame.apply(
                    lambda row: row.fillna(row.median()), axis=1
                ),
                {},
            ),
        }
        function, kwargs = methods[key]
        return self._apply_isolated(training, function, **kwargs)

    def _correction_missing_mask(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Identify algorithm-induced gaps without relabeling features."""
        mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
        records = self.dataset.feature_metadata.get(
            "correction_missing_samples"
        )
        if records is None:
            return mask
        sample_ids = frame.columns.get_level_values(
            self.dataset.schema.sample_id
        )
        for feature, samples in records.items():
            if feature in mask.index and isinstance(samples, (tuple, list)):
                mask.loc[feature, :] = sample_ids.isin(samples)
        return mask & frame.isna()

    # =========================================================================
    # Evaluation Logic (Hybrid Masking & Stratified NRMSE)
    # =========================================================================

    @staticmethod
    def generate_gmm_noise_mask(
        df_log: pd.DataFrame,
        mask_ratio: float,
        noise_factor: float = 1.5,
        global_seed: int = DEFAULT_RANDOM_SEED,
        batch_array: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate a low-intensity-weighted GMM reconstruction benchmark mask.

        This stress test hides observed values for batch-wise evaluation;
        it does not identify or simulate a proven MAR/MNAR mechanism.
        """
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(global_seed)
        shape = df_log.shape
        mask_arr = np.zeros(shape, dtype=bool)

        # Fallback to Global GMM if no batch metadata is provided
        if batch_array is None or len(np.unique(batch_array)) <= 1:
            valid_mask = ~df_log.isna().values
            valid_data = df_log.values[valid_mask].reshape(-1, 1)
            target_nas = int(valid_mask.sum() * mask_ratio)

            if target_nas == 0 or len(valid_data) < 10:
                return pd.DataFrame(
                    False, index=df_log.index, columns=df_log.columns
                )

            gmm = GaussianMixture(n_components=2, random_state=global_seed)
            gmm.fit(valid_data)
            lower_cluster_idx = np.argmin(gmm.means_)

            base_prob = gmm.predict_proba(valid_data)[:, lower_cluster_idx]
            final_score = base_prob + rng.uniform(
                0, noise_factor, size=base_prob.shape
            )
            cutoff_score = np.sort(final_score)[-target_nas]

            mask_arr[valid_mask] = final_score >= cutoff_score
            return pd.DataFrame(
                mask_arr, index=df_log.index, columns=df_log.columns
            )

        # Advanced Logic: Batch-wise independent GMM masking
        unique_batches = np.unique(batch_array)
        for b in unique_batches:
            b_cols_idx = np.where(batch_array == b)[0]
            b_data = df_log.iloc[:, b_cols_idx].values

            valid_mask_b = ~np.isnan(b_data)
            valid_data_b = b_data[valid_mask_b].reshape(-1, 1)
            target_nas_b = int(valid_mask_b.sum() * mask_ratio)

            if target_nas_b == 0:
                continue

            # Defensive mechanism for extremely small batches
            if len(valid_data_b) < 10:
                if len(valid_data_b) > 0:
                    cutoff_val = np.percentile(
                        valid_data_b, (target_nas_b / len(valid_data_b)) * 100
                    )
                    b_mask = np.zeros_like(b_data, dtype=bool)
                    b_mask[valid_mask_b] = valid_data_b.flatten() <= cutoff_val
                    mask_arr[:, b_cols_idx] = b_mask
                continue

            try:
                gmm = GaussianMixture(n_components=2, random_state=global_seed)
                gmm.fit(valid_data_b)
                lower_cluster_idx = np.argmin(gmm.means_)

                base_prob = gmm.predict_proba(valid_data_b)[
                    :, lower_cluster_idx
                ]
                final_score = base_prob + rng.uniform(
                    0, noise_factor, size=base_prob.shape
                )
                cutoff_score = np.sort(final_score)[-target_nas_b]

                b_mask = np.zeros_like(b_data, dtype=bool)
                b_mask[valid_mask_b] = final_score >= cutoff_score
                mask_arr[:, b_cols_idx] = b_mask
            except Exception as e:
                # Soft fallback to percentile truncation if GMM fails to
                # converge on edge-case batches
                logger.debug(
                    f"GMM failed for batch {b}: {e}. "
                    "Falling back to empirical percentile."
                )
                cutoff_val = np.percentile(
                    valid_data_b, (target_nas_b / len(valid_data_b)) * 100
                )
                b_mask = np.zeros_like(b_data, dtype=bool)
                b_mask[valid_mask_b] = valid_data_b.flatten() <= cutoff_val
                mask_arr[:, b_cols_idx] = b_mask

        return pd.DataFrame(
            mask_arr, index=df_log.index, columns=df_log.columns
        )

    @staticmethod
    def compute_stratified_nrmse(
        df_true: pd.DataFrame,
        df_imp: pd.DataFrame,
        mask_df: pd.DataFrame,
        lod_q: float = 0.25,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Calculate NRMSE by feature-median abundance strata.

        Masked observations inherit the median of their source feature. The
        ``lod_q`` quantile of those feature medians separates the low and high
        strata; it is an evaluation cutoff, not an analytical detection limit.
        """
        feat_meds = df_true.median(axis=1).fillna(0)
        lod_val = feat_meds.quantile(lod_q)

        t_all = df_true.values[mask_df.values]
        p_all = df_imp.values[mask_df.values]
        med_all = np.tile(feat_meds.values[:, None], (1, df_true.shape[1]))[
            mask_df.values
        ]

        low_m, hi_m = (med_all <= lod_val), (med_all > lod_val)

        def _get_nrmse(t: np.ndarray, p: np.ndarray) -> float:
            if len(t) < 2 or (np.max(t) - np.min(t)) < 1e-9:
                return np.nan
            rmse = np.sqrt(np.mean((t - p) ** 2))
            return float(rmse / (np.max(t) - np.min(t)))

        # Compile the metrics dictionary
        metrics = {
            "NRMSE_Total": _get_nrmse(t_all, p_all),
            "NRMSE_Low": _get_nrmse(t_all[low_m], p_all[low_m]),
            "NRMSE_High": _get_nrmse(t_all[hi_m], p_all[hi_m]),
            "Count_Low": int(np.sum(low_m)),
            "Count_High": int(np.sum(hi_m)),
            "Threshold": float(lod_val),
            "Threshold_Quantile": float(lod_q),
        }

        # Return exactly 3 objects to match the unpacking logic
        return metrics, t_all, p_all

    @staticmethod
    def _low_value_reference_impute(masked_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing entries with one-half of the global observed minimum."""
        observed = masked_df.to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            fill_value = 0.0
        else:
            linear_observed = np.exp2(observed) - 1.0
            positive_values = linear_observed[
                np.isfinite(linear_observed) & (linear_observed > 0)
            ]
            if positive_values.size == 0:
                fill_value = float(np.nanmin(observed))
            else:
                fill_value = float(
                    np.log2(np.nanmin(positive_values) * 0.5 + 1.0)
                )
        return masked_df.fillna(fill_value)

    @staticmethod
    def _reference_improvement_score(reference: object, value: object) -> float:
        """Score improvement over a lower-is-better reference metric."""
        score = su.relative_change_lower_better(reference, value)
        return (
            float(np.clip(score, 0.0, 1.0))
            if np.isfinite(score)
            else float("nan")
        )

    def _evaluate_imputation_candidate(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        method: str,
        ratio: float = 0.05,
        global_seed: int = DEFAULT_RANDOM_SEED,
        batch_array: Optional[np.ndarray] = None,
    ) -> tuple:
        """Evaluate one R-route candidate on an artificial mask."""
        mar_data = df_log.loc[idx_mar, target_cols].astype(float)
        mar_data.attrs["is_logged"] = True

        mask = getattr(self, "_evaluation_mask", None)
        if mask is None:
            mask = self.generate_gmm_noise_mask(
                mar_data,
                ratio,
                noise_factor=1.5,
                global_seed=global_seed,
                batch_array=batch_array,
            )

        masked_df = mar_data.copy()
        masked_df[mask] = np.nan

        method_key = str(method).replace(" ", "").replace("-", "").upper()
        provenance_start = len(getattr(self, "_implementation_provenance", []))
        if method_key in (
            "HALFGLOBALMIN",
            "HALFGLOBALMINREFERENCE",
            "LOWVALUEREF",
        ):
            imp_res = self._low_value_reference_impute(masked_df)
        else:
            imp_res = self._run_mar_method(masked_df, method, global_seed)
        route_targets = getattr(self, "_evaluation_route_targets", None)
        target_mask = (
            masked_df.isna() if route_targets is None else mask | route_targets
        )
        imp_res, domain = self._repair_output(
            imp_res, masked_df, target_mask=target_mask
        )
        imp_res = imp_res.mask(masked_df.isna() & ~target_mask)
        imp_res.attrs["is_logged"] = True

        eval_met, t_vals, p_vals = self.compute_stratified_nrmse(
            mar_data, imp_res, mask
        )
        dist_metrics = su.calc_distribution_distance_metrics(t_vals, p_vals)
        structure_metrics = structure_stats.calc_sample_structure_preservation(
            raw_obj=mar_data,
            transformed_obj=imp_res,
            sample_cols=target_cols,
            max_features=5000,
            seed=global_seed,
        )
        eval_met.update(
            {
                "status": "ok",
                "implementation": self._method_implementation(method),
                "implementation_provenance": list(
                    getattr(self, "_implementation_provenance", [])[
                        provenance_start:
                    ]
                ),
                "numerical_domain": domain,
                "masked_count": int(mask.to_numpy().sum()),
                "JSD_Total": dist_metrics["jsd"],
                "Wasserstein_Total": dist_metrics["wasserstein"],
                "Wasserstein_Normalized": dist_metrics[
                    "wasserstein_normalized"
                ],
                "Sample_Structure_Score": structure_metrics[
                    "sample_structure_composite_preservation"
                ],
                "Trustworthiness": structure_metrics[
                    "sample_structure_trustworthiness"
                ],
                "Distance_Rank_Preservation": structure_metrics[
                    "sample_structure_rank_preservation"
                ],
                "Distance_Scale_Preservation": structure_metrics[
                    "sample_structure_scale_preservation"
                ],
            }
        )
        return eval_met, t_vals, p_vals

    @staticmethod
    def _score_imputation_candidates(
        cache: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]],
        reference_metrics: dict[str, float],
    ) -> pd.DataFrame:
        """Score R-route candidates against a half-global-min reference."""
        rows = []
        for method, (metrics, _, _) in cache.items():
            if metrics.get("status", "ok") != "ok":
                continue
            rows.append(
                {
                    "method": method,
                    "nrmse_total": metrics.get("NRMSE_Total"),
                    "nrmse_low": metrics.get("NRMSE_Low"),
                    "jsd_total": metrics.get("JSD_Total"),
                    "wasserstein_normalized": metrics.get(
                        "Wasserstein_Normalized"
                    ),
                    "sample_structure_score": metrics.get(
                        "Sample_Structure_Score"
                    ),
                }
            )
        score_df = pd.DataFrame(rows)
        if score_df.empty:
            return score_df

        ref_nrmse_total = reference_metrics.get("NRMSE_Total")
        ref_nrmse_low = reference_metrics.get("NRMSE_Low")
        ref_jsd = reference_metrics.get("JSD_Total")
        ref_wasserstein = reference_metrics.get("Wasserstein_Normalized")

        score_df["nrmse_total_score"] = score_df["nrmse_total"].apply(
            lambda value: MissingValueImputer._reference_improvement_score(
                ref_nrmse_total, value
            )
        )
        score_df["nrmse_low_score"] = score_df["nrmse_low"].apply(
            lambda value: MissingValueImputer._reference_improvement_score(
                ref_nrmse_low, value
            )
        )
        score_df["jsd_score"] = score_df["jsd_total"].apply(
            lambda value: MissingValueImputer._reference_improvement_score(
                ref_jsd, value
            )
        )
        score_df["wasserstein_score"] = score_df[
            "wasserstein_normalized"
        ].apply(
            lambda value: MissingValueImputer._reference_improvement_score(
                ref_wasserstein, value
            )
        )

        reconstruction_scores = []
        distribution_scores = []
        auto_scores = []

        def planned_mean(parts: list[tuple[object, float, bool]]) -> float:
            """Do not reward candidate-specific missing metric values."""
            denominator = sum(weight for _, weight, active in parts if active)
            if denominator == 0:
                return float("nan")
            numerator = sum(
                (float(value) if np.isfinite(su.finite_or_nan(value)) else 0)
                * weight
                for value, weight, active in parts
                if active
            )
            return numerator / denominator

        active_total = np.isfinite(su.finite_or_nan(ref_nrmse_total))
        active_low = np.isfinite(su.finite_or_nan(ref_nrmse_low))
        active_jsd = np.isfinite(su.finite_or_nan(ref_jsd))
        active_wd = np.isfinite(su.finite_or_nan(ref_wasserstein))
        active_structure = np.isfinite(
            su.finite_or_nan(reference_metrics.get("Sample_Structure_Score"))
        )
        for row in score_df.itertuples():
            reconstruction_score = planned_mean(
                [
                    (row.nrmse_total_score, 0.70, active_total),
                    (row.nrmse_low_score, 0.30, active_low),
                ]
            )
            distribution_score = planned_mean(
                [
                    (row.jsd_score, 0.50, active_jsd),
                    (row.wasserstein_score, 0.50, active_wd),
                ]
            )
            auto_score = planned_mean(
                [
                    (reconstruction_score, 0.65, active_total or active_low),
                    (distribution_score, 0.20, active_jsd or active_wd),
                    (row.sample_structure_score, 0.15, active_structure),
                ]
            )
            reconstruction_scores.append(reconstruction_score)
            distribution_scores.append(distribution_score)
            auto_scores.append(auto_score)

        score_df["reconstruction_score"] = reconstruction_scores
        score_df["distribution_preservation_score"] = distribution_scores
        score_df["sample_structure_score"] = pd.to_numeric(
            score_df["sample_structure_score"], errors="coerce"
        )
        score_df["auto_score"] = auto_scores
        return score_df

    def _select_best_imputation_method(
        self,
        df_log: pd.DataFrame,
        idx_mar: pd.Index,
        target_cols: pd.Index,
        ratio: float = 0.05,
        global_seed: int = DEFAULT_RANDOM_SEED,
        batch_array: Optional[np.ndarray] = None,
    ) -> tuple:
        """Select an R-route imputer from masked-reconstruction benchmarks."""
        candidates = ["KNN", "MinProb", "QRILC", "Median", "LLS", "BPCA"]
        self._implementation_provenance = getattr(
            self, "_implementation_provenance", []
        )
        cache = {}
        self._evaluation_mask = self.generate_gmm_noise_mask(
            df_log.loc[idx_mar, target_cols],
            ratio,
            global_seed=global_seed,
            batch_array=batch_array,
        )
        reference_metrics, _, _ = self._evaluate_imputation_candidate(
            df_log=df_log,
            idx_mar=idx_mar,
            target_cols=target_cols,
            method="HalfGlobalMinReference",
            ratio=ratio,
            global_seed=global_seed,
            batch_array=batch_array,
        )

        for cand in candidates:
            logger.info(f'Simulating "{cand}" on the R-route subset...')
            provenance_start = len(self._implementation_provenance)
            try:
                emet, tv, pv = self._evaluate_imputation_candidate(
                    df_log=df_log,
                    idx_mar=idx_mar,
                    target_cols=target_cols,
                    method=cand,
                    ratio=ratio,
                    global_seed=global_seed,
                    batch_array=batch_array,
                )
            except Exception as error:
                logger.warning(f"Imputation candidate {cand} failed: {error}")
                emet = {
                    "status": "failed",
                    "reason": str(error),
                    "implementation": self._method_implementation(cand),
                    "implementation_provenance": list(
                        self._implementation_provenance[provenance_start:]
                    ),
                }
                tv, pv = np.array([]), np.array([])
            cache[cand] = (emet, tv, pv)
        self._evaluation_mask = None
        self._evaluation_route_targets = None

        score_df = self._score_imputation_candidates(cache, reference_metrics)
        if score_df.empty:
            raise ValueError("All AUTO imputation candidates failed.")
        if not score_df.empty and score_df["auto_score"].notna().any():
            score_df = selection_utils.rank_candidates(
                score_df,
                score_column="auto_score",
                tie_breakers=(("nrmse_total", True), ("method", True)),
            )
            selected_method = str(score_df.iloc[0]["method"])
            self._ranked_imputation_methods = score_df["method"].tolist()
            for row in score_df.itertuples():
                metrics = cache[row.method][0]
                metrics["Reconstruction_Score"] = su.finite_or_nan(
                    row.reconstruction_score
                )
                metrics["Distribution_Preservation_Score"] = su.finite_or_nan(
                    row.distribution_preservation_score
                )
                metrics["JSD_Score"] = su.finite_or_nan(row.jsd_score)
                metrics["Wasserstein_Score"] = su.finite_or_nan(
                    row.wasserstein_score
                )
                metrics["Sample_Structure_Score"] = su.finite_or_nan(
                    row.sample_structure_score
                )
                metrics["Auto_Score"] = su.finite_or_nan(row.auto_score)
                metrics["Low_Value_Reference_NRMSE_Total"] = su.finite_or_nan(
                    reference_metrics.get("NRMSE_Total")
                )
                metrics["Low_Value_Reference_NRMSE_Low"] = su.finite_or_nan(
                    reference_metrics.get("NRMSE_Low")
                )
                metrics["Low_Value_Reference_JSD_Total"] = su.finite_or_nan(
                    reference_metrics.get("JSD_Total")
                )
                metrics["Low_Value_Reference_Wasserstein_Normalized"] = (
                    su.finite_or_nan(
                        reference_metrics.get("Wasserstein_Normalized")
                    )
                )
        else:
            # No input-supported evaluation metric: use a transparent stable
            # order, still requiring successful final numerical-domain checks.
            order = ["Median", "KNN", "LLS", "BPCA", "MinProb", "QRILC"]
            self._ranked_imputation_methods = [
                method
                for method in order
                if cache[method][0].get("status", "ok") == "ok"
            ]
            selected_method = self._ranked_imputation_methods[0]
            self.config["auto_evaluation_status"] = "input_unsupported"

        best_score = cache[selected_method][0].get("Auto_Score", float("nan"))
        logger.info(
            f"R-route algorithm selected: {selected_method} "
            f"(score={best_score:.3f})"
        )
        return selected_method, cache

    @property
    def imputation_metrics(self) -> Dict[str, Any]:
        """Extracts key parameters and performance metrics from imputation.

        Returns:
            dict: A structured dictionary of imputation metadata for reporting.
        """
        requested_method = self.config.get("requested_method", "auto")
        selected_method = self.config.get("selected_method", "Unknown")
        selected_label = self.config.get("selected_label", selected_method)
        mnar_meth = self.config.get("mnar_method", "row")
        mnar_frac = self.config.get("mnar_fraction", 0.5)

        status = self.config.get("imputation_status", "Pending")
        if (
            status == "Skipped"
            or mnar_frac is None
            or str(mnar_meth).upper() in {"QRILC", "NOT REQUIRED"}
        ):
            reported_mnar_frac = None
        else:
            reported_mnar_frac = float(mnar_frac)

        def _safe_round(val: object) -> float:
            if pd.isna(val):
                return float("nan")
            return round(float(val), 4)

        raw_results = self.config.get("candidate_results", {})
        candidate_results = []
        candidate_metric_keys = (
            "nrmse_low",
            "nrmse_high",
            "nrmse_total",
            "jsd_total",
            "wasserstein_total",
            "wasserstein_normalized",
            "reconstruction_score",
            "distribution_preservation_score",
            "jsd_score",
            "wasserstein_score",
            "sample_structure_score",
            "trustworthiness",
            "distance_rank_preservation",
            "distance_scale_preservation",
            "auto_score",
        )
        for method, metrics in raw_results.items():
            metrics = metrics if isinstance(metrics, dict) else {}
            record = {
                "method": method,
                "implementation": metrics.get("implementation", "python"),
                "implementation_provenance": metrics.get(
                    "implementation_provenance", []
                ),
                "selected": method == selected_method,
                "status": metrics.get("status", "ok"),
                "reason": metrics.get("reason"),
                "numerical_domain": metrics.get("numerical_domain", {}),
                "final_numerical_domain": metrics.get(
                    "final_numerical_domain", {}
                ),
            }
            record.update(
                {
                    metric: _safe_round(metrics.get(metric))
                    for metric in candidate_metric_keys
                }
            )
            candidate_results.append(record)

        is_auto = bool(self.config.get("is_auto", False))
        selected_score = next(
            (
                record["auto_score"]
                for record in candidate_results
                if record["selected"]
            ),
            None,
        )
        valid_scores = sorted(
            (
                score
                for record in candidate_results
                if (score := record.get("auto_score")) is not None
                and np.isfinite(score)
                and record["status"] == "ok"
            ),
            reverse=True,
        )
        selection_margin = (
            valid_scores[0] - valid_scores[1]
            if is_auto and len(valid_scores) > 1
            else None
        )

        if status == "Skipped":
            idx_mar = pd.Index([])
            idx_mnar = pd.Index([])
        else:
            feature_metadata = self.dataset.feature_metadata
            missingness = self._missingness_labels()
            idx_mar = feature_metadata.index[
                missingness == R_ROUTE
            ].intersection(self.frame.index)
            idx_mnar = feature_metadata.index[
                missingness == S_ROUTE
            ].intersection(self.frame.index)

        # Retrieve the unified QA metrics (JSD) from the data passport
        qa_metrics = self.config.get("imputation_qa_metrics", {})

        metrics = {
            "imputation_status": status,
            "strategies": {
                "s_route_method": (
                    "Not required" if status == "Skipped" else mnar_meth
                ),
                "s_route_fraction": reported_mnar_frac,
                "mnar_method": (
                    "Not required" if status == "Skipped" else mnar_meth
                ),
                "mnar_fraction": reported_mnar_frac,
            },
            "selection": {
                "route": R_ROUTE,
                "requested_implementation": self.config.get(
                    "implementation", "python"
                ),
                "implementation": (
                    self._method_implementation(selected_method)
                    if selected_method not in {"Not required", "Unknown"}
                    else self.config.get("implementation", "python")
                ),
                "mnar_implementation": self._method_implementation(mnar_meth),
                "s_route_implementation": self._method_implementation(
                    mnar_meth
                ),
                "implementation_provenance": list(
                    getattr(self, "_implementation_provenance", [])
                ),
                "requested_method": requested_method,
                "selected_method": selected_method,
                "selected_label": selected_label,
                "is_auto": is_auto,
                "selected_score": selected_score,
                "selection_margin": selection_margin,
                "candidate_results": candidate_results,
                "evaluation_status": self.config.get(
                    "auto_evaluation_status", "evaluated"
                ),
                "benchmark_available": self.config.get(
                    "imputation_benchmark_available", False
                ),
            },
            "feature_distribution": {
                "r_route_count": len(idx_mar),
                "s_route_count": len(idx_mnar),
                "mar_count": len(idx_mar),
                "mnar_count": len(idx_mnar),
            },
            "qa_metrics": qa_metrics,
            "numerical_domain": list(
                getattr(self, "_numerical_domain_records", [])
            ),
            "missingness_routing": {
                "route_definitions": dict(ROUTE_DEFINITIONS),
                "feature_labels_unchanged": True,
                "correction_reconstruction_count": self.config.get(
                    "correction_reconstruction_count", 0
                ),
                "correction_route": "R-route (reconstruction)",
            },
            "isolation_fallbacks": list(
                getattr(self, "_isolation_fallback_records", [])
            ),
            "skip_reason": self.config.get("imputation_skip_reason"),
        }

        return metrics

    def _missingness_labels(self) -> pd.Series:
        """Read canonical operational routes, accepting legacy data labels."""
        return routes_from_metadata(self.dataset.feature_metadata)

    def transform_imputation(self) -> StageResult[MetaboDataset]:
        """Perform imputation without writing files or rendering figures."""
        self._isolation_fallback_records = []
        self._implementation_provenance = []
        self._numerical_domain_records = []
        self._evaluation_mask = None
        self._evaluation_route_targets = None
        self._ranked_imputation_methods = []
        self._imputation_phase = "final_mnar"
        self._validate_implementation()
        # =====================================================================
        # Parameter Extraction
        # =====================================================================
        # StageRunner resolves defaults, TOML settings, and call-time notebook
        # overrides before this calculation-only method is reached.
        _mnar = self.config.get("mnar_method", "QRILC")
        _frac = self.config.get("mnar_fraction", 0.5)
        requested_mar = self.config.get("mar_method", "Auto")
        _mar = requested_mar
        _knn_k = self.config.get("knn_neighbors", 5)
        _lls_k = self.config.get("lls_neighbors", 15)
        _bpca_k = self.config.get("bpca_components", 2)
        _bpca_max_iter = self.config.get("bpca_max_iter", 100)
        _bpca_tol = self.config.get("bpca_tol", 1e-4)
        _ratio = self.config.get("sim_mask_ratio", 0.05)

        _seed = self.config.get("global_seed", DEFAULT_RANDOM_SEED)
        target_cols = self.frame.columns.difference(self.blank_data.columns)
        target_matrix = self.frame.loc[:, target_cols].replace(0, np.nan)
        missingness = self._missingness_labels()
        routed_metadata = self.dataset.feature_metadata.copy(deep=True)
        routed_metadata["imputation_route"] = missingness
        observed = target_matrix.to_numpy(dtype=float)
        if np.isinf(observed).any() or (observed < 0).any():
            raise ValueError(
                "Imputation requires finite, nonnegative observed "
                "intensities; missing observations may be NaN."
            )
        if self.dataset.context.is_logged or self.dataset.context.is_scaled:
            raise ValueError("Imputation requires unlogged, unscaled data.")
        context_updates = {
            "pipeline_stage": "Imputation",
            "extra_attrs": {
                **self.dataset.context.extra_attrs,
                "value_scale": "raw_positive",
            },
        }

        if not target_matrix.isna().any().any():
            logger.info(
                "No missing values detected in target samples. "
                "Bypassing imputation and propagating the matrix unchanged."
            )
            self.config.update(
                {
                    "pipeline_stage": "Imputation",
                    "imputation_status": "Skipped",
                    "imputation_skip_reason": (
                        "No missing values detected in target samples."
                    ),
                    "selected_method": "Not required",
                    "selected_label": "Not required",
                    "requested_method": requested_mar,
                    "is_auto": False,
                    "candidate_results": {},
                    "imputation_qa_metrics": {},
                }
            )
            imputed_dataset = self._to_dataset(
                self.frame,
                context_updates=context_updates,
                feature_metadata=routed_metadata,
            )

            logger.success(
                "Missing value imputation skipped: no missing values found."
            )
            return StageResult(
                data=imputed_dataset,
                audit=ImputationAuditPayload(
                    metric_values=self.imputation_metrics,
                    candidate_results={},
                    requested_method=requested_mar,
                    selected_method="Not required",
                    selected_label="Not required",
                    is_auto=False,
                    mar_feature_count=0,
                    has_candidate_cache=False,
                    skipped=True,
                    plot_payload=None,
                ),
            )

        batch_col = self.config.get("batch", "Batch")
        batch_array = target_cols.get_level_values(batch_col).values

        mnar_info = (
            f"{_mnar}"
            if (str(_mnar).upper() == "QRILC")
            else (f"{_mnar} (LOD={_frac}x)")
        )

        mar_spec = IMPUTATION_METHODS.resolve(_mar)
        if mar_spec.key == "AUTO":
            mar_info = (
                f"Auto (Evaluating KNN={_knn_k}, LLS (K={_lls_k}), "
                f"BPCA (PCs={_bpca_k}), MinProb, Median)"
            )
        elif mar_spec.key == "KNN":
            mar_info = f"KNN (K={_knn_k})"
        elif mar_spec.key == "LLS":
            mar_info = f"LLS (K={_lls_k})"
        elif mar_spec.key == "BPCA":
            mar_info = f"BPCA (PCs={_bpca_k}, MaxIter={_bpca_max_iter})"
        else:
            mar_info = f"{_mar}"

        logger.info(
            f"Hybrid Imputation Engine Initialized. "
            f"R-route: {mar_info} | S-route: {mnar_info} | Sim_Mask: {_ratio}"
        )

        df_log = np.log1p(
            self.frame.astype(float).replace({0: np.nan})
        ) / np.log(2.0)
        training_log = df_log.copy(deep=True)
        correction_missing = self._correction_missing_mask(training_log)
        mnar_targets = training_log.isna() & ~correction_missing
        mnar_targets.loc[missingness != S_ROUTE, :] = False
        mar_targets = training_log.isna() & ~mnar_targets
        self.config["correction_reconstruction_count"] = int(
            correction_missing.loc[:, target_cols].to_numpy().sum()
        )

        # =====================================================================
        # S-route: Localized LOD Imputation or QRILC
        # =====================================================================
        feature_metadata = self.dataset.feature_metadata
        idx_mnar = feature_metadata.index[missingness == S_ROUTE].intersection(
            df_log.index
        )

        if mnar_targets.loc[idx_mnar, target_cols].to_numpy().any():
            logger.info(
                f"Applying {_mnar} to {len(idx_mnar)} S-route features."
            )

            mnar_training = training_log.loc[idx_mnar, target_cols]
            # QRILC fits each sample's distribution across all features.
            # Keep original missingness; R-route fills are not evidence.
            qrilc_training = training_log.loc[:, target_cols]
            if self.config.get("implementation", "python") == "r":
                mnar_imp = self._impute_by_r(
                    qrilc_training,
                    method="QRILC",
                    global_seed=_seed,
                )
            elif str(_mnar).upper() == "QRILC":
                mnar_imp = MissingValueImputer.impute_by_qrilc(
                    df_log=qrilc_training, global_seed=_seed
                )
            else:
                mnar_imp = MissingValueImputer.impute_by_constant(
                    df_log=mnar_training,
                    fraction=_frac,
                    imp_mode=_mnar,
                )

            mnar_imp = mnar_imp.loc[idx_mnar, target_cols]
            mnar_mask = mnar_targets.loc[idx_mnar, target_cols]
            try:
                mnar_imp, domain = self._repair_output(
                    mnar_imp, mnar_training, target_mask=mnar_mask
                )
            except ValueError as error:
                raise ValueError(
                    f"Fixed S-route failed output validation: {error}"
                ) from error
            self._numerical_domain_records.append(
                {"phase": "final_mnar", "method": _mnar, **domain}
            )
            df_log.loc[idx_mnar, target_cols] = mnar_training.where(
                ~mnar_mask, mnar_imp
            )
        else:
            logger.info("S-route index empty. Bypassing special handling.")

        # =====================================================================
        # R-route: Candidate Evaluation and Imputation
        # =====================================================================
        idx_mar = feature_metadata.index[missingness == R_ROUTE].intersection(
            df_log.index
        )
        idx_mar = idx_mar.union(
            correction_missing.index[
                correction_missing.loc[:, target_cols].any(axis=1)
            ],
            sort=False,
        )
        cache, eval_met, t_vals, p_vals = {}, {}, [], []
        is_auto = mar_spec.key == "AUTO"

        if len(idx_mar) > 0:
            self._imputation_phase = "evaluation_mar"
            self._evaluation_route_targets = mar_targets.loc[
                idx_mar, target_cols
            ]
            if is_auto:
                _mar, cache = self._select_best_imputation_method(
                    training_log,
                    idx_mar,
                    target_cols,
                    _ratio,
                    _seed,
                    batch_array,
                )
                eval_met, t_vals, p_vals = cache[_mar]
                mar_spec = IMPUTATION_METHODS.resolve(_mar)
            else:
                eval_met, t_vals, p_vals = self._evaluate_imputation_candidate(
                    training_log,
                    idx_mar,
                    target_cols,
                    _mar,
                    _ratio,
                    _seed,
                    batch_array,
                )

            self._isolation_fallback_records = []
            self._imputation_phase = "final_mar"
            logger.info(f"Executing isolated '{_mar}' on R-route features.")
            mar_slice = training_log.loc[idx_mar, target_cols]
            mar_mask = mar_targets.loc[idx_mar, target_cols]
            ranked = self._ranked_imputation_methods or [_mar]
            for candidate in ranked if is_auto else [_mar]:
                provenance_start = len(self._implementation_provenance)
                try:
                    mar_imp = self._run_mar_method(mar_slice, candidate, _seed)
                    mar_imp, domain = self._repair_output(
                        mar_imp, mar_slice, target_mask=mar_mask
                    )
                except Exception as error:
                    if not is_auto:
                        raise
                    metrics = cache[candidate][0]
                    metrics["status"] = "final_fit_failed"
                    metrics["reason"] = str(error)
                    metrics.setdefault("implementation_provenance", []).extend(
                        self._implementation_provenance[provenance_start:]
                    )
                    logger.warning(
                        f"Final {candidate} fit failed; trying next eligible "
                        f"AUTO candidate: {error}"
                    )
                    continue
                _mar = candidate
                if cache:
                    eval_met, t_vals, p_vals = cache[candidate]
                self._numerical_domain_records.append(
                    {"phase": "final_mar", "method": candidate, **domain}
                )
                if cache:
                    cache[candidate][0]["final_numerical_domain"] = domain
                eval_met.setdefault("implementation_provenance", []).extend(
                    self._implementation_provenance[provenance_start:]
                )
                df_log.loc[idx_mar, target_cols] = df_log.loc[
                    idx_mar, target_cols
                ].where(~mar_mask, mar_imp)
                break
            else:
                raise ValueError(
                    "All eligible AUTO imputation candidates failed final "
                    "output validation. Fixed S-route fills were not changed."
                )

        # =====================================================================
        # Matrix Reconstruction and Passport Update
        # =====================================================================
        final_log = pd.concat(
            [df_log[target_cols], df_log[self.blank_data.columns]], axis=1
        )[self.frame.columns]

        if final_log.loc[:, target_cols].isna().any().any():
            raise ValueError(
                "Imputation could not fill all target values. Check feature "
                "coverage and sample roles; no completed result was produced."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            res_val = np.expm1(final_log * np.log(2.0))
        # Preserve actual observations exactly, not a log/exp round trip.
        observed_mask = self.frame.notna() & self.frame.gt(0)
        res_val = res_val.where(~observed_mask, self.frame)
        res_val.loc[:, self.blank_data.columns] = self.blank_data
        target_values = res_val.loc[:, target_cols].to_numpy(dtype=float)
        if not np.isfinite(target_values).all() or (target_values <= 0).any():
            implementation = self.config.get("implementation", "python")
            raise ValueError(
                f"{implementation} imputation produced nonfinite or "
                "nonpositive intensities after the output policy."
            )
        imputed_frame = pd.DataFrame(res_val).copy(deep=True)
        imputed_frame.attrs.update(self.config)

        display_mar_method = IMPUTATION_METHODS.display_name(_mar)
        if not len(idx_mar):
            _mar = "Not required"
            display_mar_method = "Not required"
            is_auto = False

        eval_source = (
            cache if cache else {display_mar_method: (eval_met, t_vals, p_vals)}
        )
        if not len(idx_mar):
            eval_source = {}
        cand_mets = {}
        for m_name, (m_eval, _, _) in eval_source.items():
            cand_mets[m_name] = {
                "implementation": m_eval.get(
                    "implementation", self._method_implementation(m_name)
                ),
                "implementation_provenance": m_eval.get(
                    "implementation_provenance", []
                ),
                "status": m_eval.get("status", "ok"),
                "reason": m_eval.get("reason"),
                "numerical_domain": m_eval.get("numerical_domain", {}),
                "final_numerical_domain": m_eval.get(
                    "final_numerical_domain", {}
                ),
                "nrmse_low": m_eval.get("NRMSE_Low", float("nan")),
                "nrmse_high": m_eval.get("NRMSE_High", float("nan")),
                "nrmse_total": m_eval.get("NRMSE_Total", float("nan")),
                "jsd_total": m_eval.get("JSD_Total", float("nan")),
                "wasserstein_total": m_eval.get(
                    "Wasserstein_Total", float("nan")
                ),
                "wasserstein_normalized": m_eval.get(
                    "Wasserstein_Normalized", float("nan")
                ),
                "reconstruction_score": m_eval.get(
                    "Reconstruction_Score", float("nan")
                ),
                "distribution_preservation_score": m_eval.get(
                    "Distribution_Preservation_Score", float("nan")
                ),
                "jsd_score": m_eval.get("JSD_Score", float("nan")),
                "wasserstein_score": m_eval.get(
                    "Wasserstein_Score", float("nan")
                ),
                "sample_structure_score": m_eval.get(
                    "Sample_Structure_Score", float("nan")
                ),
                "trustworthiness": m_eval.get("Trustworthiness", float("nan")),
                "distance_rank_preservation": m_eval.get(
                    "Distance_Rank_Preservation", float("nan")
                ),
                "distance_scale_preservation": m_eval.get(
                    "Distance_Scale_Preservation", float("nan")
                ),
                "auto_score": m_eval.get("Auto_Score", float("nan")),
            }
        self.config.update(
            {
                "pipeline_stage": "Imputation",
                "imputation_status": "Completed",
                "imputation_skip_reason": None,
                "selected_method": _mar,
                "selected_label": display_mar_method,
                "requested_method": requested_mar,
                "is_auto": is_auto,
                "mnar_method": _mnar,
                "mnar_fraction": _frac,
                "candidate_results": cand_mets,
            }
        )

        # =====================================================================
        # Quality Metrics and Result Assembly
        # =====================================================================
        logger.info("Calculating imputation-related metrics...")
        qa_metrics = self.calc_imp_quality_metrics(
            raw_obj=self.frame, imp_obj=imputed_frame
        )
        self.config["imputation_qa_metrics"] = qa_metrics

        imputed_dataset = self._to_dataset(
            imputed_frame,
            context_updates=context_updates,
            feature_metadata=routed_metadata,
        )

        benchmark_results = (
            (cache if cache else {_mar: (eval_met, t_vals, p_vals)})
            if len(idx_mar) > 0
            else {}
        )
        selected_benchmark = benchmark_results.get(_mar)
        self.config["imputation_benchmark_available"] = bool(
            selected_benchmark is not None
            and selected_benchmark[0].get("status", "ok") == "ok"
            and len(selected_benchmark[1]) > 0
            and len(selected_benchmark[2]) > 0
        )
        return StageResult(
            data=imputed_dataset,
            audit=ImputationAuditPayload(
                metric_values=self.imputation_metrics,
                candidate_results=benchmark_results,
                selected_method=_mar,
                selected_label=display_mar_method,
                requested_method=requested_mar,
                is_auto=is_auto,
                mar_feature_count=len(idx_mar),
                has_candidate_cache=bool(cache),
                skipped=False,
                plot_payload=ImputationPlotPayload(
                    raw_data=snapshot_dataset(self.dataset),
                    imputed_data=snapshot_dataset(imputed_dataset),
                    sample_structure=structure_stats.calc_sample_structure_diagnostics(
                        raw_obj=self.frame,
                        transformed_obj=imputed_frame,
                        seed=int(_seed),
                    ),
                    global_seed=int(
                        self.config.get("global_seed", DEFAULT_RANDOM_SEED)
                    ),
                ),
            ),
        )

    @log_execution_time
    def run_imputation(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[MetaboDataset]:
        """Return the structured missing-value imputation stage result.

        Keyword settings use the same names as the imputation configuration,
        such as ``r_route_method``, ``s_route_method``, ``knn_neighbors``,
        ``bpca_components``, ``sim_mask_ratio``, and ``global_seed``. They
        take precedence over TOML settings and built-in defaults for this
        processor instance.

        Historical ``mar_method``/``mnar_method`` remain accepted aliases;
        saved configuration uses these legacy keys for compatibility.
        """
        runner = ImputationStageRunner(
            processor=self,
            output_dir=output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS,
        )
        result = runner.run()
        logger.success("Missing value imputation completed successfully.")
        return result
