"""Signal-correction orchestration, candidate evaluation, and selection.

SignalCorrector prepares QC, batch, and injection-order inputs; applies a
configured correction method or evaluates AUTO candidates; and records metrics
for QC precision and sample-structure preservation. It writes corrected stage
matrices and audit artifacts while delegating numerical kernels and plotting.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ...config import resolve_stage_config
from ...constants import DEFAULT_RANDOM_SEED
from ...core import DatasetProcessor, MetaboDataset
from ...plotting.payloads import (
    CorrectionPlotPayload,
    snapshot_dataset,
    snapshot_plot_value,
)
from ...runtime import log_execution_time
from ...statistics import metrics as su
from ...statistics import sample_structure as structure_stats
from ...statistics import selection as selection_utils
from ..audit import CorrectionAuditPayload
from ..numeric_domain import apply_correction_domain
from ..stage import StageResult
from .algorithms import (
    _format_correction_method_label,
    _normalize_correction_method,
    _parse_correction_candidate,
)
from .regression import RegressionCorrector
from .metanorm import MetanormRLOESSCorrector
from .runner import CorrectionStageRunner
from .ruv import RUVCorrector
from .ruv_r import RUVIIIRCorrector
from .serrf import SERRFCorrector
from .serrf_r import SERRFRCorrector
from .waveica import WaveICA2Corrector
from .waveica_r import WaveICARCorrector

FitPredictCallable = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CorrectionModel = (
    RandomForestRegressor
    | TransformedTargetRegressor
    | Pipeline
    | FitPredictCallable
)


# =============================================================================
# Signal-Correction Processor
# =============================================================================
class SignalCorrector(DatasetProcessor):
    """
    Quality control-based signal drift correction dispatcher.

    Orchestrates the dynamic execution routing between RegressionCorrector,
    SERRFCorrector, WaveICA2Corrector, and RUVCorrector. Extracts
    domain-specific metadata into pure mathematical arrays to preserve engine
    purity. Manages file exports, dual-mode RSD tracking, and downstream
    visualization diagnostics.
    """

    _R_METHODS = frozenset(
        {"Metanorm-rLOESS", "WaveICA 2.0", "RUV-III", "SERRF"}
    )

    def _candidate_implementation(
        self, method: str, candidate_params: dict[str, Any] | None = None
    ) -> str:
        """Resolve AUTO substitutions before fitting, never after failure."""
        robust = (candidate_params or {}).get(
            "robust", self.config.get("rlsc_robust", True)
        )
        return (
            "r"
            if self.config.get("implementation") == "r"
            and (method in self._R_METHODS or (method == "QC-RLSC" and robust))
            else "python"
        )

    _RUNTIME_CONFIG_KEYS = frozenset(
        {
            "base_est",
            "implementation",
            "loess_span",
            "loess_degree",
            "rlsc_span_selection",
            "rlsc_span_grid",
            "rlsc_min_qc",
            "rlsc_robust",
            "rloess_span_selection",
            "rloess_span",
            "rloess_iterations",
            "rf_n_tree",
            "serrf_n_tree",
            "serrf_corr_features",
            "serrf_r_source",
            "serrf_backend",
            "serrf_batch_size",
            "svr_kernel",
            "svr_c",
            "svr_gamma",
            "ruv_k",
            "ruv_control_features",
            "ruv_replicate_column",
            "waveica_components",
            "waveica_cutoff",
            "waveica_alpha",
            "waveica_levels",
            "waveica_spline_knots",
            "waveica_max_iter",
            "regression_backend",
            "regression_batch_size",
            "cv_folds",
            "n_jobs",
            "global_seed",
        }
    )

    def __init__(
        self,
        data: MetaboDataset,
        pipeline_params: Optional[Dict[str, Any]] = None,
        base_est: Optional[str] = None,
        loess_span: Optional[float] = None,
        loess_degree: Optional[int] = None,
        rlsc_span_selection: Optional[str] = None,
        rlsc_span_grid: Optional[list[float]] = None,
        rlsc_min_qc: Optional[int] = None,
        rlsc_robust: Optional[bool] = None,
        rloess_span_selection: Optional[str] = None,
        rloess_span: Optional[float] = None,
        rloess_iterations: Optional[int] = None,
        rf_n_tree: Optional[int] = None,
        serrf_n_tree: Optional[int] = None,
        serrf_corr_features: Optional[int] = None,
        serrf_backend: Optional[str] = None,
        serrf_batch_size: Optional[Union[str, int]] = None,
        svr_kernel: Optional[str] = None,
        svr_c: Optional[Union[float, int]] = None,
        svr_gamma: Optional[Union[str, float]] = None,
        ruv_k: Optional[int] = None,
        waveica_components: Optional[int] = None,
        waveica_cutoff: Optional[float] = None,
        waveica_levels: Optional[int] = None,
        waveica_spline_knots: Optional[int] = None,
        waveica_max_iter: Optional[int] = None,
        regression_backend: Optional[str] = None,
        regression_batch_size: Optional[Union[str, int]] = None,
        cv_folds: Optional[int] = None,
        n_jobs: Optional[int] = None,
        global_seed: Optional[int] = None,
        implementation: Optional[str] = None,
        serrf_r_source: Optional[str] = None,
        ruv_control_features: Optional[list[str]] = None,
        ruv_replicate_column: Optional[str] = None,
        waveica_alpha: Optional[float] = None,
    ) -> None:
        """Initialize the signal drift correction dispatcher."""
        super().__init__(data)

        sc_configs = resolve_stage_config(
            pipeline_params,
            "SignalCorrector",
            {
                "base_est": "Auto",
                "implementation": "python",
                "loess_span": 0.3,
                "loess_degree": 1,
                "rlsc_span_selection": "fixed",
                "rlsc_span_grid": [0.3, 0.5, 0.7],
                "rlsc_min_qc": 7,
                "rlsc_robust": True,
                "rloess_span_selection": "gcv",
                "rloess_span": 0.75,
                "rloess_iterations": 4,
                "rf_n_tree": 500,
                "serrf_n_tree": 100,
                "serrf_corr_features": 10,
                "serrf_r_source": None,
                "serrf_backend": "loky",
                "serrf_batch_size": "auto",
                "svr_kernel": "rbf",
                "svr_c": 500,
                "svr_gamma": 1.0,
                "cv_folds": 5,
                "ruv_k": 5,
                "ruv_control_features": None,
                "ruv_replicate_column": None,
                "waveica_components": 10,
                "waveica_cutoff": 0.1,
                "waveica_alpha": 0.0,
                "waveica_levels": None,
                "waveica_spline_knots": 5,
                "waveica_max_iter": 1000,
                "regression_backend": "loky",
                "regression_batch_size": "auto",
                "n_jobs": self.config.get("n_jobs", -1),
                "global_seed": self.config.get(
                    "global_seed", DEFAULT_RANDOM_SEED
                ),
            },
            {
                "base_est": base_est,
                "implementation": implementation,
                "loess_span": loess_span,
                "loess_degree": loess_degree,
                "rlsc_span_selection": rlsc_span_selection,
                "rlsc_span_grid": rlsc_span_grid,
                "rlsc_min_qc": rlsc_min_qc,
                "rlsc_robust": rlsc_robust,
                "rloess_span_selection": rloess_span_selection,
                "rloess_span": rloess_span,
                "rloess_iterations": rloess_iterations,
                "rf_n_tree": rf_n_tree,
                "serrf_n_tree": serrf_n_tree,
                "serrf_corr_features": serrf_corr_features,
                "serrf_r_source": serrf_r_source,
                "serrf_backend": serrf_backend,
                "serrf_batch_size": serrf_batch_size,
                "svr_kernel": svr_kernel,
                "svr_c": svr_c,
                "svr_gamma": svr_gamma,
                "cv_folds": cv_folds,
                "ruv_k": ruv_k,
                "ruv_control_features": ruv_control_features,
                "ruv_replicate_column": ruv_replicate_column,
                "waveica_components": waveica_components,
                "waveica_cutoff": waveica_cutoff,
                "waveica_alpha": waveica_alpha,
                "waveica_levels": waveica_levels,
                "waveica_spline_knots": waveica_spline_knots,
                "waveica_max_iter": waveica_max_iter,
                "regression_backend": regression_backend,
                "regression_batch_size": regression_batch_size,
                "n_jobs": n_jobs,
                "global_seed": global_seed,
            },
        )

        # Integrate unified properties into internal attributes dictionary
        self.config.update(sc_configs)

    # =========================================================================
    # Domain Preprocessing and Statistical Methods
    # =========================================================================
    @staticmethod
    def extract_qc_rsd_series(df_obj: pd.DataFrame) -> pd.Series:
        """Extracts the RSD series for QC samples across all features."""
        sample_type_col = df_obj.attrs.get("sample_type", "Sample Type")
        qc_label = df_obj.attrs.get("sample_dict", {}).get("QC sample", "QC")
        mask = df_obj.columns.get_level_values(sample_type_col) == qc_label
        qc_data = df_obj.loc[:, mask].astype(float)

        means = qc_data.mean(axis=1)
        rsd = qc_data.std(axis=1, ddof=1) / means.where(means > 0)
        return rsd.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def calculate_median_qc_rsd(df_obj: pd.DataFrame) -> float:
        """Calculates the scalar median RSD of QC samples."""
        rsd_series = SignalCorrector.extract_qc_rsd_series(df_obj)
        if rsd_series.empty:
            return float("nan")
        return float(rsd_series.median())

    @staticmethod
    def calculate_featurewise_d_ratio(
        before_obj: pd.DataFrame,
        after_obj: pd.DataFrame,
        *,
        support_mask: pd.DataFrame | None = None,
        min_qc: int = 3,
        min_sample: int = 3,
    ) -> dict[str, Any]:
        """Calculate canonical feature-wise D-ratio diagnostics.

        D-ratio is the technical QC dispersion relative to the combined
        biological and technical dispersion::

            D_i = SD(QC) / sqrt(SD(sample)**2 + SD(QC)**2)

        The returned values are percentages.  ``support_mask`` is always
        established from the original observations when supplied; therefore
        imputed or newly generated cells cannot enlarge the denominator or
        make a candidate appear better by dropping difficult observations.
        ``after_obj`` may contain missing/invalid corrected cells, in which
        case only the fixed original support is evaluated.
        """
        before = before_obj.astype(float)
        after = after_obj.reindex(index=before.index, columns=before.columns)
        after = after.astype(float)
        if support_mask is None:
            support = np.isfinite(before.to_numpy()) & (
                before.to_numpy() > 0
            )
        else:
            support = (
                support_mask.reindex_like(before)
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            support &= np.isfinite(before.to_numpy()) & (
                before.to_numpy() > 0
            )

        sample_type_col = before.attrs.get("sample_type", "Sample Type")
        labels = before.attrs.get("sample_dict", {})
        qc_label = labels.get("QC sample", "QC")
        actual_label = labels.get("Actual sample", "Sample")
        sample_types = before.columns.get_level_values(sample_type_col)
        qc_mask = np.asarray(sample_types == qc_label, dtype=bool)
        sample_mask = np.asarray(sample_types == actual_label, dtype=bool)
        after_values = after.to_numpy(dtype=float)
        before_values = before.to_numpy(dtype=float)

        baseline = np.full(len(before.index), np.nan, dtype=float)
        current = np.full(len(before.index), np.nan, dtype=float)
        score_support = np.zeros(len(before.index), dtype=bool)
        for row in range(len(before.index)):
            q_support = support[row] & qc_mask
            s_support = support[row] & sample_mask
            if q_support.sum() < min_qc or s_support.sum() < min_sample:
                continue
            q_before = before_values[row, q_support]
            s_before = before_values[row, s_support]
            qsd_before = np.std(q_before, ddof=1)
            ssd_before = np.std(s_before, ddof=1)
            denominator = np.hypot(qsd_before, ssd_before)
            if np.isfinite(denominator) and denominator > 0:
                baseline[row] = 100.0 * qsd_before / denominator
                score_support[row] = (
                    qsd_before > np.finfo(float).eps
                    and ssd_before > np.finfo(float).eps
                )

            q_after = after_values[row, q_support]
            s_after = after_values[row, s_support]
            q_after = q_after[np.isfinite(q_after) & (q_after > 0)]
            s_after = s_after[np.isfinite(s_after) & (s_after > 0)]
            if len(q_after) < min_qc or len(s_after) < min_sample:
                continue
            qsd_after = np.std(q_after, ddof=1)
            ssd_after = np.std(s_after, ddof=1)
            denominator = np.hypot(qsd_after, ssd_after)
            if np.isfinite(denominator) and denominator > 0:
                current[row] = 100.0 * qsd_after / denominator

        baseline_series = pd.Series(baseline, index=before.index, dtype=float)
        current_series = pd.Series(current, index=before.index, dtype=float)
        baseline_support = pd.Series(score_support, index=before.index)
        baseline_values = baseline_series.loc[baseline_support]
        current_values = current_series.loc[baseline_support]
        if baseline_values.empty:
            score = float("nan")
            improvement = pd.Series(dtype=float)
        else:
            base = baseline_values.to_numpy(dtype=float)
            curr = current_values.to_numpy(dtype=float)
            finite_current = np.isfinite(curr)
            improvement_values = np.zeros_like(base, dtype=float)
            improvement_values[finite_current] = (
                (base[finite_current] - curr[finite_current])
                / base[finite_current]
            )
            improvement = pd.Series(
                np.clip(improvement_values, 0.0, 1.0),
                index=baseline_values.index,
                dtype=float,
            )
            score = float(improvement.mean())
        return {
            "baseline": baseline_series,
            "current": current_series,
            "improvement": improvement,
            "baseline_median": (
                float(baseline_series.median())
                if baseline_series.notna().any()
                else float("nan")
            ),
            "current_median": (
                float(current_series.median())
                if current_series.notna().any()
                else float("nan")
            ),
            "score": score,
            "support_count": int(baseline_support.sum()),
            "evaluated_count": int(
                current_series.loc[baseline_support].notna().sum()
            ),
        }

    @staticmethod
    def calculate_featurewise_qc_rsd_improvement(
        before_obj: pd.DataFrame,
        after_obj: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate paired feature-wise QC-RSD improvement diagnostics."""
        before_rsd = SignalCorrector.extract_qc_rsd_series(before_obj)
        after_rsd = SignalCorrector.extract_qc_rsd_series(after_obj)
        support = before_rsd.index[
            np.isfinite(before_rsd) & (before_rsd > np.finfo(float).eps)
        ]
        sample_type = before_obj.attrs.get("sample_type", "Sample Type")
        qc_label = before_obj.attrs.get("sample_dict", {}).get(
            "QC sample", "QC"
        )
        qc = before_obj.columns.get_level_values(sample_type) == qc_label
        reference = before_obj.loc[:, qc].to_numpy(dtype=float)
        candidate = after_obj.loc[:, qc].to_numpy(dtype=float)
        required = np.isfinite(reference) & (reference > 0)
        available = np.isfinite(candidate) & (candidate > 0)
        complete = pd.Series(
            (~required | available).all(axis=1), index=before_obj.index
        )
        after_rsd = after_rsd.loc[complete.reindex(after_rsd.index)]
        common_idx = before_rsd.index.intersection(after_rsd.index, sort=False)
        if common_idx.empty:
            return {
                "score": 0.0 if len(support) else float("nan"),
                "median": float("nan"),
                "values": pd.Series(dtype=float),
                "support_count": len(support),
                "evaluated_count": 0,
            }

        before_vals = pd.to_numeric(before_rsd.loc[common_idx], errors="coerce")
        after_vals = pd.to_numeric(after_rsd.loc[common_idx], errors="coerce")
        valid = (
            np.isfinite(before_vals.to_numpy(dtype=float))
            & np.isfinite(after_vals.to_numpy(dtype=float))
            & (before_vals.to_numpy(dtype=float) > np.finfo(float).eps)
        )
        if not np.any(valid):
            return {
                "score": float("nan"),
                "median": float("nan"),
                "values": pd.Series(dtype=float),
            }

        before_vals = before_vals.iloc[np.flatnonzero(valid)]
        after_vals = after_vals.iloc[np.flatnonzero(valid)]
        signed_improvement = (before_vals - after_vals) / before_vals
        signed_improvement = signed_improvement.replace(
            [np.inf, -np.inf], np.nan
        )
        signed_improvement = signed_improvement.dropna()
        if signed_improvement.empty:
            return {
                "score": float("nan"),
                "median": float("nan"),
                "values": pd.Series(dtype=float),
            }

        clipped_improvement = signed_improvement.clip(lower=0.0, upper=1.0)
        winsor_low, winsor_high = np.nanpercentile(
            clipped_improvement.to_numpy(dtype=float), [5.0, 95.0]
        )
        winsorized = clipped_improvement.clip(
            lower=winsor_low, upper=winsor_high
        )
        return {
            "score": float(winsorized.reindex(support, fill_value=0).mean()),
            "median": float(
                np.nanmedian(signed_improvement.to_numpy(dtype=float))
            ),
            "values": signed_improvement,
            "support_count": len(support),
            "evaluated_count": len(signed_improvement),
        }

    @staticmethod
    def _fixed_metric_score(
        scores: tuple[float, float, float],
        supported: tuple[bool, bool, bool],
    ) -> float:
        """Drop only input-unsupported metrics, never candidate failures."""
        weights = (0.35, 0.35, 0.30)
        total = sum(weight for weight, use in zip(weights, supported) if use)
        if not total:
            return float("nan")
        return sum(
            weight * (float(value) if np.isfinite(value) else 0.0)
            for value, weight, use in zip(scores, weights, supported)
            if use
        ) / total

    @staticmethod
    def _technical_precision_score(
        median_qc_rsd_score: float,
        featurewise_qc_rsd_score: float,
        supported: tuple[bool, bool],
    ) -> float:
        """Combine the two QC-precision components on fixed support.

        The median QC-RSD and feature-wise QC-RSD improvements are two
        views of the same technical-precision objective.  Input support is
        fixed across candidates: an unavailable input metric is omitted, but
        a candidate that fails to produce that metric contributes zero.
        This keeps AUTO from rewarding a partially reported candidate.
        """
        values = (median_qc_rsd_score, featurewise_qc_rsd_score)
        weight_sum = float(sum(0.5 for use in supported if use))
        if weight_sum <= 0:
            return float("nan")
        weighted = 0.0
        for value, use in zip(values, supported):
            if not use:
                continue
            numeric = su.finite_or_nan(value)
            if np.isfinite(numeric):
                weighted += 0.5 * float(np.clip(numeric, 0.0, 1.0))
        return float(weighted / weight_sum)

    @classmethod
    def _correction_auto_score(
        cls,
        technical_precision_score: float,
        d_ratio_preservation_score: float,
        sample_structure_score: float,
        supported: tuple[bool, bool, bool],
    ) -> float:
        """Combine technical, biological and structure preservation scores.

        Correction AUTO is intentionally a three-part score: technical
        precision, biological-variation preservation (canonical D-ratio),
        and sample-structure preservation.  Relative lower-is-better scores
        are already clipped at zero by their metric calculators; the sample
        structure score is an absolute preservation score and is therefore
        only clipped to its natural [0, 1] range here.
        """
        return cls._fixed_metric_score(
            (
                technical_precision_score,
                d_ratio_preservation_score,
                sample_structure_score,
            ),
            supported,
        )

    def _correction_feature_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Track newly missing cells without altering feature route tags."""
        metadata = self.dataset.feature_metadata.copy(deep=True)
        reference = self.frame.to_numpy(dtype=float)
        output = frame.to_numpy(dtype=float)
        newly_missing = (
            np.isfinite(reference)
            & (reference > 0)
            & (~np.isfinite(output) | (output <= 0))
        )
        columns = frame.columns
        sample_ids = (
            columns.get_level_values(self.dataset.schema.sample_id)
            if isinstance(columns, pd.MultiIndex)
            else columns
        )
        previous = metadata.get(
            "correction_missing_samples",
            pd.Series([()] * len(metadata), index=metadata.index),
        )
        records = []
        for row, feature in enumerate(frame.index):
            old = previous.get(feature, ())
            old = old if isinstance(old, (tuple, list)) else ()
            new = sample_ids[newly_missing[row]].tolist()
            records.append(tuple(dict.fromkeys((*old, *new))))
        metadata["correction_missing_samples"] = pd.Series(
            records, index=frame.index, dtype=object
        )
        return metadata

    def _calculate_qc_baseline_means(
        self, batch_col: str, sample_type_col: str, qc_label: str
    ) -> pd.DataFrame:
        """Calculate batch-wise QC mean to reverse-engineer visual baselines."""
        qc_df = self.frame.loc[
            :, self.frame.columns.get_level_values(sample_type_col) == qc_label
        ]
        batch_levels = qc_df.columns.get_level_values(batch_col)
        int_base = qc_df.T.groupby(batch_levels).mean().T

        base_int_bc = pd.DataFrame(
            index=self.frame.index, columns=self.frame.columns, dtype=float
        )
        for batch in self.frame.columns.get_level_values(batch_col).unique():
            mask = self.frame.columns.get_level_values(batch_col) == batch
            bc_block = pd.concat([int_base[batch]] * mask.sum(), axis=1)
            base_int_bc.loc[:, mask] = bc_block.values

        return base_int_bc

    def _prepare_ruv_control_features(
        self, empirical_ratio: float = 0.05
    ) -> pd.Index:
        """Use explicit controls, otherwise record the empirical heuristic."""
        requested = self.config.get("ruv_control_features")
        if requested is not None:
            controls = pd.Index(requested)
            if controls.empty or not controls.is_unique:
                raise ValueError(
                    "RUV control features must be nonempty/unique."
                )
            missing = controls.difference(self.frame.index)
            if len(missing):
                raise ValueError(
                    f"Unknown RUV control features: {list(missing)}"
                )
            return controls
        is_list = self.valid_internal_standards
        orf_list = self.valid_outlier_reference_features
        base_controls = set(is_list + orf_list)

        empirical_controls = []
        if not self.actual_data.empty:
            actual_data = self.actual_data.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                r_series = actual_data.std(axis=1, ddof=1)
                r_series = r_series / actual_data.mean(axis=1)

            valid_rsd = r_series.replace([np.inf, -np.inf], np.nan).dropna()
            n_empirical = max(10, int(len(self.frame) * empirical_ratio))
            empirical_controls = valid_rsd.nsmallest(n_empirical).index.tolist()

        combined_controls = base_controls.union(empirical_controls)
        valid_ctl = pd.Index(list(combined_controls)).intersection(
            self.frame.index
        )

        logger.info(
            f"RUV-III Control Features: {len(valid_ctl)} total "
            f"({len(base_controls)} predefined, "
            f"{len(empirical_controls)} empirical)."
        )
        return valid_ctl

    def _evaluate_correction_candidates(
        self,
        methods_to_run: list,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        blank_mask: np.ndarray,
        order_array: np.ndarray,
        batch_col: str,
        sample_type_col: str,
        qc_label: str,
    ) -> Dict[str, Any]:
        """
        Evaluate configured correction candidates and collect selection metrics.
        """
        results_store = {}
        raw_rsd = SignalCorrector.calculate_median_qc_rsd(self.frame)
        baseline_feature_rsd = self.extract_qc_rsd_series(self.frame)
        baseline_structure = structure_stats.calc_sample_structure_diagnostics(
            raw_obj=self.frame,
            transformed_obj=self.frame,
            max_features=5000,
            seed=int(self.config.get("global_seed", DEFAULT_RANDOM_SEED)),
        )
        baseline_structure_score = su.finite_or_nan(
            baseline_structure["metrics"].get(
                "sample_structure_composite_preservation"
            )
        )
        baseline_d_ratio = self.calculate_featurewise_d_ratio(
            self.frame,
            self.frame,
        )
        biological_count = int((
            self.frame.columns.get_level_values(sample_type_col)
            == self.dataset.schema.roles.actual
        ).sum())
        score_support = (
            bool(np.isfinite(raw_rsd)),
            bool((baseline_feature_rsd > np.finfo(float).eps).any()),
            biological_count >= 3
            and bool(np.isfinite(baseline_structure_score)),
        )
        component_support = (
            score_support[0] or score_support[1],
            bool(baseline_d_ratio.get("support_count", 0) > 0),
            score_support[2],
        )
        for raw_method in methods_to_run:
            method, candidate_label, candidate_params = (
                _parse_correction_candidate(raw_method)
            )
            implementation = self._candidate_implementation(
                method, candidate_params
            )
            if implementation == "r":
                self._validate_r_options(method, implementation)
            logger.info(f"--- Evaluating Method: {candidate_label} ---")

            # Route to specific engine
            if method == "Metanorm-rLOESS":
                engine = MetanormRLOESSCorrector(
                    random_state=self.config.get(
                        "global_seed", DEFAULT_RANDOM_SEED
                    ),
                )
                stages_output = engine.fit_transform(
                    intensity_df=self.frame,
                    batch_array=batch_array,
                    qc_mask=qc_mask,
                    order_array=order_array,
                    sample_type_array=(
                        self.frame.columns.get_level_values(
                            sample_type_col
                        ).to_numpy(dtype=str)
                    ),
                )
            elif implementation == "r" and method != "QC-RLSC":
                seed = self.config.get("global_seed", DEFAULT_RANDOM_SEED)
                if method == "WaveICA 2.0":
                    engine = WaveICARCorrector(
                        n_components=self.config["waveica_components"],
                        cutoff=self.config["waveica_cutoff"],
                        alpha=self.config["waveica_alpha"],
                        random_state=seed,
                    )
                    stages_output = engine.fit_transform(
                        self.frame, order_array, blank_mask=blank_mask
                    )
                elif method == "RUV-III":
                    column = self.config.get("ruv_replicate_column")
                    groups = None
                    if column is not None:
                        metadata = self.dataset.sample_metadata
                        if column not in metadata.columns:
                            raise ValueError(
                                f"RUV replicate column {column!r} is missing."
                            )
                        groups = metadata.reindex(
                            self.dataset.intensity.columns
                        )[column].to_numpy()
                    engine = RUVIIIRCorrector(
                        k=self.config["ruv_k"], random_state=seed
                    )
                    stages_output = engine.fit_transform(
                        self.frame,
                        qc_mask,
                        self.config["ruv_control_features"],
                        blank_mask=blank_mask,
                        replicate_groups=groups,
                    )
                elif method == "SERRF":
                    engine = SERRFRCorrector(
                        source_path=self.config["serrf_r_source"],
                        random_state=seed,
                        cv_folds=self.config.get("cv_folds", 5),
                        n_correlated_features=self.config[
                            "serrf_corr_features"
                        ],
                    )
                    stages_output = engine.fit_transform(
                        self.frame,
                        batch_array,
                        qc_mask,
                        order_array,
                        blank_mask=blank_mask,
                    )
                else:
                    raise ValueError(f"Unsupported R method: {method}")
            elif method == "SERRF":
                engine = SERRFCorrector(
                    n_estimators=self.config.get("serrf_n_tree", 100),
                    cv_folds=self.config.get("cv_folds", 5),
                    n_corr_features=self.config.get("serrf_corr_features", 10),
                    random_state=self.config.get(
                        "global_seed", DEFAULT_RANDOM_SEED
                    ),
                    n_jobs=self.config.get("n_jobs", -1),
                    joblib_backend=self.config.get("serrf_backend", "loky"),
                    joblib_batch_size=self.config.get(
                        "serrf_batch_size", "auto"
                    ),
                )
                stages_output = engine.fit_transform(
                    intensity_df=self.frame,
                    batch_array=batch_array,
                    qc_mask=qc_mask,
                    order_array=order_array,
                    blank_mask=blank_mask,
                )
            elif method == "RUV-III":
                ctrl_features = self._prepare_ruv_control_features()
                engine = RUVCorrector(k=self.config.get("ruv_k", 3))
                stages_output = engine.fit_transform(
                    intensity_df=self.frame,
                    qc_mask=qc_mask,
                    control_features=ctrl_features,
                    blank_mask=blank_mask,
                )
                engine.diagnostics["control_selection"] = (
                    "explicit"
                    if self.config.get("ruv_control_features") is not None
                    else "declared plus empirical low-Sample-RSD heuristic"
                )
            elif method == "WaveICA 2.0":
                engine = WaveICA2Corrector(
                    n_components=self.config.get("waveica_components", 10),
                    cutoff=self.config.get("waveica_cutoff", 0.1),
                    n_levels=self.config.get("waveica_levels"),
                    spline_knots=self.config.get("waveica_spline_knots", 5),
                    max_iter=self.config.get("waveica_max_iter", 1000),
                    random_state=self.config.get(
                        "global_seed", DEFAULT_RANDOM_SEED
                    ),
                )
                stages_output = engine.fit_transform(
                    intensity_df=self.frame,
                    order_array=order_array,
                    batch_array=batch_array,
                    blank_mask=blank_mask,
                )
            else:
                candidate_attrs = dict(self.config)
                candidate_attrs.update(candidate_params)
                candidate_attrs["implementation"] = implementation
                engine = RegressionCorrector(method=method, **candidate_attrs)
                stages_output = engine.fit_transform(
                    intensity_df=self.frame,
                    batch_array=batch_array,
                    qc_mask=qc_mask,
                    order_array=order_array,
                    blank_mask=blank_mask,
                )

            # Extract DataFrames and calculate RSD tracking
            rsd_hist_oof = {"Original": raw_rsd}
            rsd_hist_full = {"Original": raw_rsd}
            stage_dfs = {"Original": self.frame}
            stage_oof_dfs = {}
            stage_domains = {}

            for stage_name, (full_df, oof_df) in stages_output.items():
                clean_name = stage_name.replace("\n", " ")
                final_df, full_domain = apply_correction_domain(
                    full_df, self.frame
                )
                stage_domains[clean_name] = {"full": full_domain, "oof": None}
                final_df.attrs.update(self.config)
                final_df.attrs["implementation"] = implementation
                final_df.attrs["pipeline_stage"] = "Correction"
                final_df.attrs["value_scale"] = "raw_positive"
                final_df.attrs["qc_rsd_baseline"] = raw_rsd

                curr_full_rsd = SignalCorrector.calculate_median_qc_rsd(
                    final_df
                )
                rsd_hist_full[clean_name] = curr_full_rsd
                final_df.attrs["qc_rsd_current_full"] = curr_full_rsd

                if oof_df is not None:
                    oof_wrap, oof_domain = apply_correction_domain(
                        oof_df, self.frame
                    )
                    stage_domains[clean_name]["oof"] = oof_domain
                    oof_wrap.attrs.update(self.config)
                    oof_wrap.attrs["implementation"] = implementation
                    oof_wrap.attrs["pipeline_stage"] = "Correction"
                    oof_wrap.attrs["value_scale"] = "raw_positive"
                    curr_oof_rsd = SignalCorrector.calculate_median_qc_rsd(
                        oof_wrap
                    )
                    if not np.isfinite(su.finite_or_nan(curr_oof_rsd)):
                        curr_oof_rsd = None
                    rsd_hist_oof[clean_name] = curr_oof_rsd
                    final_df.attrs["qc_rsd_current_oof"] = curr_oof_rsd
                    stage_oof_dfs[clean_name] = oof_wrap
                else:
                    rsd_hist_oof[clean_name] = None
                    final_df.attrs["qc_rsd_current_oof"] = None

                stage_dfs[clean_name] = final_df

            for name, df in stage_dfs.items():
                if name != "Original":
                    df.attrs["rsd_history_oof"] = rsd_hist_oof
                    df.attrs["rsd_history_full"] = rsd_hist_full

            # Algebraically Reverse-Engineer Smooth Fit Baseline Matrix
            pred_df = None
            if "Intra-batch corrected" in stage_dfs and method not in (
                "SERRF",
                "RUV-III",
                "WaveICA 2.0",
            ):
                try:
                    base_int_bc = self._calculate_qc_baseline_means(
                        batch_col, sample_type_col, qc_label
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        raw_pred_df = base_int_bc * (
                            self.frame / stage_dfs["Intra-batch corrected"]
                        )
                    pred_df = pd.DataFrame(raw_pred_df).copy(deep=True)
                    pred_df.attrs.update(self.config)
                except Exception as e:
                    logger.debug(f"Baseline back-calc failed: {e}")

            # Extract final RSD to evaluate performance
            final_stage = list(stage_dfs.keys())[-1]
            final_full = rsd_hist_full[final_stage]
            final_oof = rsd_hist_oof.get(final_stage)
            final_oof_data = stage_oof_dfs.get(final_stage)
            validation = {
                "status": "available"
                if final_oof is not None
                else "unavailable",
                "requested_folds": self.config.get("cv_folds", 5),
                "qc_value_coverage": (
                    float(
                        np.isfinite(
                            final_oof_data.loc[:, qc_mask].to_numpy(dtype=float)
                        ).sum()
                        / max(1, self.frame.loc[:, qc_mask].size)
                    )
                    if final_oof_data is not None
                    else 0.0
                ),
                "evaluation_basis": (
                    "oof" if final_oof is not None else "full_model"
                ),
            }
            requested_folds = int(self.config.get("cv_folds", 5))
            qc_counts = [
                int(qc_mask[batch_array == batch].sum())
                for batch in np.unique(batch_array)
            ]
            minimum_folds = 2 if method == "SERRF" else 3
            validation["effective_folds_by_batch"] = [
                min(requested_folds, count)
                if min(requested_folds, count) >= minimum_folds
                and final_oof is not None
                else 0
                for count in qc_counts
            ]
            if method == "SERRF":
                diagnostics = getattr(engine, "diagnostics", {})
                fold_counts = diagnostics.get("effective_folds_by_batch", {})
                validation["effective_folds_by_batch"] = [
                    fold_counts.get(str(batch), 0)
                    for batch in np.unique(batch_array)
                ]
                validation["scope"] = diagnostics.get("validation")
            # Sklearn folds adapt per feature when QC observations are missing.
            # Record the actual fold-count range, not just the batch maximum.
            validation["feature_fold_counts_by_batch"] = {}
            for batch in np.unique(batch_array):
                valid_counts = (
                    self.frame.loc[:, qc_mask & (batch_array == batch)]
                    .notna()
                    .sum(axis=1)
                )
                if method in ("QC-SVR", "QC-RFSC"):
                    counts = {
                        min(requested_folds, int(count))
                        if min(requested_folds, int(count)) >= 3
                        else 0
                        for count in valid_counts
                    }
                    validation["feature_fold_counts_by_batch"][str(batch)] = (
                        sorted(counts) if final_oof is not None else [0]
                    )
            if final_oof is not None and validation["qc_value_coverage"] < 1.0:
                validation["status"] = "partial"
            full_only_methods = (
                "RUV-III",
                "WaveICA 2.0",
                "Metanorm-rLOESS",
            )
            if method in full_only_methods:
                validation["requested_folds"] = None
                validation["reason"] = (
                    "This algorithm uses a jointly fitted transformation, "
                    "not held-out prediction; QC metrics are descriptive."
                )
            missing_required_oof = (
                (method == "SERRF" and implementation == "r"
                 or method == "QC-RLSC" and candidate_params.get(
                     "robust", self.config.get("rlsc_robust", True)
                 ))
                and final_oof is None
            )
            if missing_required_oof:
                validation["evaluation_basis"] = "unavailable"
                validation["reason"] = (
                    f"{candidate_label} has no usable held-out QC result; "
                    "full-fit "
                    "metrics are descriptive and cannot substitute for OOF."
                )
            eval_rsd = final_full if method in full_only_methods else final_oof
            if eval_rsd is None and not missing_required_oof:
                eval_rsd = final_full

            median_qc_rsd_improvement_score = float(
                np.clip(
                    su.relative_change_lower_better(
                        raw_rsd, su.finite_or_nan(eval_rsd)
                    ),
                    0.0,
                    1.0,
                )
            )
            final_corrected_df = stage_dfs[final_stage]
            eval_corrected_df = (
                stage_oof_dfs.get(final_stage)
                if method not in full_only_methods
                else final_corrected_df
            )
            if eval_corrected_df is None:
                eval_corrected_df = (
                    self.frame * np.nan
                    if missing_required_oof
                    else final_corrected_df
                )
            # D-ratio follows the same evaluation basis as QC-RSD: ordinary
            # candidates use held-out predictions, while methods without an
            # independent prediction interface remain descriptive full-fit
            # diagnostics.  Both use the original positive observations as
            # the fixed support.
            d_ratio_full = self.calculate_featurewise_d_ratio(
                self.frame,
                final_corrected_df,
            )
            d_ratio_oof = (
                self.calculate_featurewise_d_ratio(
                    self.frame,
                    final_oof_data,
                )
                if final_oof_data is not None
                else None
            )
            d_ratio_eval = (
                d_ratio_full
                if method in full_only_methods or d_ratio_oof is None
                else d_ratio_oof
            )
            validation["d_ratio_support_count"] = d_ratio_eval.get(
                "support_count", 0
            )
            validation["d_ratio_evaluated_count"] = d_ratio_eval.get(
                "evaluated_count", 0
            )
            validation["d_ratio_observed_coverage"] = (
                float(
                    d_ratio_eval.get("evaluated_count", 0)
                    / max(1, d_ratio_eval.get("support_count", 0))
                )
            )
            validation["d_ratio_coverage_warning"] = bool(
                d_ratio_eval.get("evaluated_count", 0)
                < d_ratio_eval.get("support_count", 0)
            )
            reference_values = self.frame.to_numpy(dtype=float)
            evaluation_values = eval_corrected_df.to_numpy(dtype=float)
            input_support = np.isfinite(reference_values) & (
                reference_values > 0
            )
            evaluation_valid = np.isfinite(evaluation_values) & (
                evaluation_values > 0
            )
            qc_required = input_support[:, qc_mask]
            qc_retained = evaluation_valid[:, qc_mask] & qc_required
            qc_coverage = float(qc_retained.sum() / max(1, qc_required.sum()))
            # Keep the QC-RSD estimate on its observed common support.  A
            # small number of invalid corrected cells is recorded as coverage
            # evidence, but must not erase the metric for legacy regression
            # candidates (QC-SVR/QC-RLSC/robust QC-RLSC).  The former behavior
            # converted every partial result to a zero score and made these
            # unchanged methods appear to lose a score component after the
            # shared raw-positive domain policy was introduced.
            validation["score_qc_observed_coverage"] = qc_coverage
            validation["score_qc_coverage_warning"] = bool(
                qc_coverage < 1.0
            )
            featurewise_improvement = (
                SignalCorrector.calculate_featurewise_qc_rsd_improvement(
                    before_obj=self.frame,
                    after_obj=eval_corrected_df,
                )
            )
            featurewise_qc_rsd_improvement_score = su.finite_or_nan(
                featurewise_improvement.get("score")
            )
            sample_structure = (
                structure_stats.calc_sample_structure_diagnostics(
                    raw_obj=self.frame,
                    transformed_obj=final_corrected_df,
                    max_features=5000,
                    seed=int(
                        self.config.get("global_seed", DEFAULT_RANDOM_SEED)
                    ),
                )
            )
            structure_metrics = sample_structure["metrics"]
            sample_structure_score = su.finite_or_nan(
                structure_metrics.get("sample_structure_composite_preservation")
            )
            sample_structure_score = (
                float(np.clip(sample_structure_score, 0.0, 1.0))
                if np.isfinite(sample_structure_score)
                else float("nan")
            )
            structure_components = (
                "sample_structure_trustworthiness",
                "sample_structure_rank_preservation",
                "sample_structure_scale_shift_preservation",
                "sample_structure_scale_delta_preservation",
            )
            missing_structure_components = [
                key for key in structure_components
                if np.isfinite(su.finite_or_nan(
                    baseline_structure["metrics"].get(key)
                )) and not np.isfinite(su.finite_or_nan(
                    structure_metrics.get(key)
                ))
            ]
            validation["missing_structure_components"] = (
                missing_structure_components
            )
            if missing_structure_components:
                validation["sample_structure_metric_warning"] = True
            else:
                validation["sample_structure_metric_warning"] = False
            sample_mask = self.frame.columns.get_level_values(
                sample_type_col
            ) == self.dataset.schema.roles.actual
            sample_required = input_support[:, sample_mask]
            sample_output = final_corrected_df.loc[:, sample_mask].to_numpy(
                dtype=float
            )
            sample_retained = (
                np.isfinite(sample_output) & (sample_output > 0)
            ) & sample_required
            sample_coverage = float(
                sample_retained.sum() / max(1, sample_required.sum())
            )
            validation["score_sample_observed_coverage"] = sample_coverage
            validation["score_sample_coverage_warning"] = bool(
                sample_coverage < 1.0
            )
            validation["score_support"] = dict(zip(
                ("median_qc_rsd", "featurewise_qc_rsd", "sample_structure"),
                score_support,
            ))
            validation["score_component_support"] = dict(zip(
                (
                    "technical_precision",
                    "d_ratio_preservation",
                    "sample_structure",
                ),
                component_support,
            ))
            validation["score_missing_metric_policy"] = (
                "candidate failures score zero; only input-unsupported "
                "metrics are excluded for all candidates"
            )
            validation["score_feature_support_count"] = (
                featurewise_improvement.get("support_count", 0)
            )
            validation["score_feature_evaluated_count"] = (
                featurewise_improvement.get("evaluated_count", 0)
            )
            validation["eligible_for_auto"] = bool(
                not missing_required_oof
                and stage_domains[final_stage]["full"][
                    "valid_observed_count"
                ] > 0
                and (
                    (score_support[0] and np.isfinite(
                        su.finite_or_nan(eval_rsd)
                    ))
                    or (score_support[1] and featurewise_improvement.get(
                        "evaluated_count", 0
                    ) > 0)
                    or (score_support[2] and np.isfinite(su.finite_or_nan(
                        structure_metrics.get(
                            "sample_structure_composite_preservation"
                        )
                    )))
                    or (
                        component_support[1]
                        and d_ratio_eval.get("evaluated_count", 0) > 0
                    )
                )
            )
            technical_precision_score = self._technical_precision_score(
                median_qc_rsd_improvement_score,
                featurewise_qc_rsd_improvement_score,
                score_support[:2],
            )
            auto_score = self._correction_auto_score(
                technical_precision_score,
                d_ratio_eval.get("score"),
                sample_structure_score,
                component_support,
            )
            if missing_required_oof:
                auto_score = None

            results_store[candidate_label] = {
                "implementation": implementation,
                "validation": validation,
                "native_diagnostics": getattr(engine, "diagnostics", {}),
                "numeric_domain": stage_domains,
                "implementation_provenance": (
                    [engine.provenance]
                    if getattr(engine, "provenance", None)
                    else []
                ),
                "stage_qc_rsd": {
                    label: self.extract_qc_rsd_series(frame)
                    for label, frame in stage_dfs.items()
                },
                "stage_oof_qc_rsd": {
                    label: self.extract_qc_rsd_series(frame)
                    for label, frame in stage_oof_dfs.items()
                },
                "method": method,
                "candidate_label": candidate_label,
                "candidate_params": candidate_params,
                "stage_dfs": stage_dfs,
                "stage_oof_dfs": stage_oof_dfs,
                "pred_df": pred_df,
                "final_rsd_full": final_full,
                "final_rsd_oof": final_oof,
                "eval_rsd": eval_rsd,
                "median_qc_rsd_improvement_score": (
                    median_qc_rsd_improvement_score
                ),
                "featurewise_qc_rsd_improvement_score": (
                    featurewise_qc_rsd_improvement_score
                ),
                "featurewise_qc_rsd_improvement_median": (
                    featurewise_improvement.get("median")
                ),
                "featurewise_qc_rsd_improvement_values": (
                    featurewise_improvement.get("values")
                ),
                "technical_precision_score": technical_precision_score,
                "d_ratio_baseline": baseline_d_ratio.get("baseline"),
                "d_ratio_current_full": d_ratio_full.get("current"),
                "d_ratio_current_oof": (
                    d_ratio_oof.get("current") if d_ratio_oof else None
                ),
                "d_ratio_baseline_median": baseline_d_ratio.get(
                    "baseline_median"
                ),
                "d_ratio_current_full_median": d_ratio_full.get(
                    "current_median"
                ),
                "d_ratio_current_oof_median": (
                    d_ratio_oof.get("current_median")
                    if d_ratio_oof
                    else None
                ),
                "d_ratio_evaluation_basis": (
                    "full_model"
                    if method in full_only_methods or d_ratio_oof is None
                    else "oof"
                ),
                "d_ratio_preservation_score": d_ratio_eval.get("score"),
                "d_ratio_support_count": d_ratio_eval.get("support_count", 0),
                "d_ratio_evaluated_count": d_ratio_eval.get(
                    "evaluated_count", 0
                ),
                "sample_structure_score": sample_structure_score,
                "sample_structure_metrics": structure_metrics,
                "sample_structure": sample_structure,
                "auto_score": auto_score,
            }

            log_rsd = eval_rsd
            if log_rsd is not None:
                logger.info(
                    f"{candidate_label} Eval QC RSD: {log_rsd * 100:.2f}%"
                )

        return results_store

    def _select_best_correction_method(
        self, results_store: Dict[str, Any]
    ) -> str:
        """
        Identify the optimal correction method using Auto score.

        AUTO combines technical precision (median and feature-wise QC-RSD),
        canonical D-ratio improvement, and actual-sample structure
        preservation. Methods with held-out QC predictions use those for
        QC metrics; jointly fitted methods use full-model diagnostics.
        """
        if not results_store:
            return ""

        rank_rows = []
        for method, result in results_store.items():
            if not result.get("validation", {}).get("eligible_for_auto", True):
                continue
            auto_score = su.finite_or_nan(result.get("auto_score"))
            rank_rows.append(
                {
                    "method": method,
                    # Preserve the prior fallback policy for unavailable scores.
                    "auto_score": (
                        auto_score
                        if np.isfinite(auto_score)
                        else np.finfo(float).min
                    ),
                    "eval_rsd": self._get_correction_eval_rsd(method, result),
                }
            )

        if not rank_rows:
            raise ValueError(
                "No correction AUTO candidate has usable output and metrics."
            )
        ranked = selection_utils.rank_candidates(
            pd.DataFrame(rank_rows),
            score_column="auto_score",
            tie_breakers=(("eval_rsd", True), ("method", True)),
        )
        return str(ranked.iloc[0]["method"])

    @staticmethod
    def _get_correction_eval_rsd(method: str, result: dict[str, Any]) -> float:
        """Return the QC-RSD metric used for correction-method selection."""
        cached_eval = su.finite_or_nan(result.get("eval_rsd"))
        if np.isfinite(cached_eval):
            return cached_eval

        canonical_method = _normalize_correction_method(
            result.get("method", method)
        )
        if canonical_method in (
            "RUV-III",
            "WaveICA 2.0",
            "Metanorm-rLOESS",
        ):
            return float(result.get("final_rsd_full", float("inf")))

        eval_rsd = result.get("final_rsd_oof")
        if eval_rsd is None:
            eval_rsd = result.get("final_rsd_full", float("inf"))
        return float(eval_rsd)

    # =========================================================================
    # Core Pipeline Execution Flow
    # =========================================================================
    def transform_correction(self) -> StageResult[dict[str, MetaboDataset]]:
        """Evaluate correction candidates and return the selected stages."""
        requested_config = dict(self.config)
        # Cast the internal matrix to float for safe in-place regression
        # updates.
        self._replace_frame(self.frame.astype(float))

        self.config["pipeline_stage"] = "Original"
        # Extract domain context strictly into mathematical arrays
        sample_type_col = self.config.get("sample_type", "Sample Type")
        batch_col = self.config.get("batch", "Batch")
        inject_order_col = self.config.get("inject_order", "Inject Order")

        sample_dict = self.config.get("sample_dict", {})
        qc_label = sample_dict.get("QC sample", "QC")
        actual_label = sample_dict.get("Actual sample", "Sample")
        blank_label = sample_dict.get("Blank sample", "Blank")

        qc_mask = (
            self.frame.columns.get_level_values(sample_type_col) == qc_label
        )
        blank_mask = (
            self.frame.columns.get_level_values(sample_type_col) == blank_label
        )
        batch_array = self.frame.columns.get_level_values(batch_col).values
        order_array = self.frame.columns.get_level_values(
            inject_order_col
        ).values

        if blank_mask.any():
            logger.info(
                "Blank policy: {} Blank samples follow each candidate's "
                "declared fitting and output policy.",
                int(blank_mask.sum()),
            )

        req_method, req_label, req_params = _parse_correction_candidate(
            self.config.get("base_est", "QC-RLSC")
        )
        if req_method == "QC-RLSC":
            req_params.setdefault("robust", self.config.get("rlsc_robust"))
            req_label = "robust QC-RLSC" if req_params["robust"] else "QC-RLSC"
            self.config["rlsc_robust"] = req_params["robust"]
            if req_params["robust"] and (
                self.config.get("is_logged") or self.config.get("is_scaled")
            ):
                raise ValueError(
                    "robust QC-RLSC expects unlogged, unscaled intensities."
                )
        is_auto = req_method == "AUTO"
        requested_method = req_method
        implementation = self.config.get("implementation", "python")
        if implementation not in {"python", "r"}:
            raise ValueError("implementation must be 'python' or 'r'.")
        if (
            implementation == "r"
            and not is_auto
            and req_method not in self._R_METHODS
            and not (req_method == "QC-RLSC" and req_params.get("robust"))
        ):
            raise ValueError(
                "R correction supports AUTO or explicit "
                "robust QC-RLSC, Metanorm-rLOESS, WaveICA 2.0, RUV-III, "
                "or SERRF; ordinary QC-RLSC and QC-SVR are Python methods."
            )
        if not is_auto or implementation == "python":
            self._validate_r_options(req_method, implementation)
        if req_method == "Metanorm-rLOESS":
            if implementation != "r":
                raise ValueError("Metanorm-rLOESS requires implementation='r'.")
            if self.config.get("is_logged", False):
                raise ValueError(
                    "Metanorm-rLOESS expects unlogged positive intensities."
                )
            if self.config.get("is_scaled", False):
                raise ValueError(
                    "Metanorm-rLOESS expects unscaled positive intensities."
                )

        if req_method == "AUTO":
            methods_to_run = [
                "SERRF",
                "RUV-III",
                "WaveICA 2.0",
                {
                    "method": "QC-RLSC",
                    "label": "QC-RLSC",
                    "params": {"robust": False},
                },
                {
                    "method": "QC-RLSC",
                    "label": "robust QC-RLSC",
                    "params": {"robust": True},
                },
                "QC-SVR",
            ]
            logger.info("AUTO mode enabled. Evaluating multiple methods.")
        else:
            methods_to_run = [{
                "method": req_method, "label": req_label, "params": req_params
            }]

        # ---------------------------------------------------------------------
        # Computation & Evaluation Phase
        # ---------------------------------------------------------------------
        results_store = {}
        failed_candidates = []
        candidate_implementations = {}
        for candidate in methods_to_run:
            method, label, params = _parse_correction_candidate(candidate)
            candidate_implementation = self._candidate_implementation(
                method, params
            )
            candidate_implementations[label] = candidate_implementation
            try:
                results_store.update(
                    self._evaluate_correction_candidates(
                        methods_to_run=[candidate],
                        batch_array=batch_array,
                        qc_mask=qc_mask,
                        blank_mask=blank_mask,
                        order_array=order_array,
                        batch_col=batch_col,
                        sample_type_col=sample_type_col,
                        qc_label=qc_label,
                    )
                )
            except Exception as exc:
                if not is_auto:
                    raise
                failed_candidates.append(
                    {
                        "method": label,
                        "implementation": candidate_implementation,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                logger.warning("Correction candidate {} failed: {}", label, exc)
        if not results_store:
            raise ValueError(
                f"All correction candidates failed: {failed_candidates}"
            )

        # ---------------------------------------------------------------------
        # Selection Phase
        # ---------------------------------------------------------------------
        selected_label = (
            self._select_best_correction_method(results_store)
            if is_auto
            else next(iter(results_store))
        )
        selected_result = results_store[selected_label]
        selected_method = selected_result.get(
            "method",
            _normalize_correction_method(selected_label),
        )

        if is_auto:
            selected_rsd = self._get_correction_eval_rsd(
                method=selected_label, result=selected_result
            )
            selected_score = su.finite_or_nan(selected_result.get("auto_score"))

            logger.success(
                "Auto selection: "
                f"{_format_correction_method_label(selected_label)} is "
                f"optimal (score = {selected_score:.3f}, "
                f"Eval QC RSD = {selected_rsd * 100:.2f}%)."
            )
            # Update metric tracker to reflect dynamically chosen algorithm
            self.config["base_est"] = selected_method
            self.config["correction_method_label"] = selected_label
            selected_params = selected_result.get("candidate_params", {})
            if selected_method == "QC-RLSC":
                self.config["rlsc_robust"] = selected_params.get(
                    "robust", self.config.get("rlsc_robust", True)
                )

            # Ensure the propagated DataFrames carry the resolved name
            for df in selected_result["stage_dfs"].values():
                df.attrs["base_est"] = selected_method
                df.attrs["correction_method_label"] = selected_label

        candidate_results = (
            [
                {
                    "method": str(label),
                    "implementation": result.get(
                        "implementation", candidate_implementations[label]
                    ),
                    "implementation_provenance": result.get(
                        "implementation_provenance", []
                    ),
                    "selected": str(label) == str(selected_label),
                    "status": (
                        "ok"
                        if result.get("validation", {}).get(
                            "eligible_for_auto", True
                        )
                        else "ineligible"
                    ),
                    "validation": result.get("validation", {}),
                    "numeric_domain": result.get("numeric_domain", {}),
                    "eval_rsd": result.get("eval_rsd"),
                    "final_rsd_oof": result.get("final_rsd_oof"),
                    "final_rsd_full": result.get("final_rsd_full"),
                    "median_qc_rsd_improvement_score": result.get(
                        "median_qc_rsd_improvement_score"
                    ),
                    "featurewise_qc_rsd_improvement_score": result.get(
                        "featurewise_qc_rsd_improvement_score"
                    ),
                    "technical_precision_score": result.get(
                        "technical_precision_score"
                    ),
                    "d_ratio_baseline_median": result.get(
                        "d_ratio_baseline_median"
                    ),
                    "d_ratio_current_full_median": result.get(
                        "d_ratio_current_full_median"
                    ),
                    "d_ratio_current_oof_median": result.get(
                        "d_ratio_current_oof_median"
                    ),
                    "d_ratio_evaluation_basis": result.get(
                        "d_ratio_evaluation_basis"
                    ),
                    "d_ratio_preservation_score": result.get(
                        "d_ratio_preservation_score"
                    ),
                    "sample_structure_score": result.get(
                        "sample_structure_score"
                    ),
                    "auto_score": result.get("auto_score"),
                }
                for label, result in results_store.items()
            ]
            if is_auto
            else []
        )
        valid_scores = sorted(
            (
                score
                for result in candidate_results
                if result["status"] == "ok" and np.isfinite(
                    score := su.finite_or_nan(result.get("auto_score"))
                )
            ),
            reverse=True,
        )
        selection = {
            "requested_implementation": implementation,
            "implementation": selected_result.get(
                "implementation", candidate_implementations[selected_label]
            ),
            "candidate_implementations": candidate_implementations,
            "implementation_provenance": selected_result.get(
                "implementation_provenance", []
            ),
            "native_diagnostics": selected_result.get("native_diagnostics", {}),
            "numeric_domain": selected_result.get("numeric_domain", {}),
            "validation": selected_result.get("validation", {}),
            "failed_candidates": failed_candidates,
            "requested_method": requested_method,
            "selected_method": selected_method,
            "selected_label": selected_label,
            "is_auto": is_auto,
            "selected_score": (
                float(selected_result.get("auto_score"))
                if is_auto
                and np.isfinite(
                    su.finite_or_nan(selected_result.get("auto_score"))
                )
                else None
            ),
            "selection_margin": (
                valid_scores[0] - valid_scores[1]
                if is_auto and len(valid_scores) > 1
                else None
            ),
            "candidate_results": candidate_results,
        }
        self.config["selection"] = selection
        self.config["is_auto"] = is_auto

        # Keep candidate stage matrices local; only selected outputs cross the
        # StageResult boundary and no ad-hoc DataFrame attribute is created.
        stage_dfs = selected_result["stage_dfs"]
        for df in stage_dfs.values():
            df.attrs["selection"] = dict(selection)
            df.attrs["is_auto"] = is_auto
        selected_stages = {
            name: stage
            for name, stage in stage_dfs.items()
            if name != "Original"
        }
        final_stage = list(selected_stages.values())[-1]
        prediction = selected_result["pred_df"]
        self.config.update(final_stage.attrs)
        self._correction_output_attrs = dict(self.config)
        selected_datasets = {
            name: self._to_dataset(
                stage,
                context_updates={
                    "pipeline_stage": "Correction",
                    "extra_attrs": {
                        **self.dataset.context.extra_attrs,
                        "value_scale": "raw_positive",
                    },
                },
                feature_metadata=self._correction_feature_metadata(stage),
            )
            for name, stage in selected_stages.items()
        }
        prediction_dataset = (
            self._to_dataset(prediction) if prediction is not None else None
        )
        result = StageResult(
            data=selected_datasets,
            audit=CorrectionAuditPayload(
                metric_values=self.correction_metrics,
                requested_method=requested_method,
                selected_method=selected_method,
                selected_label=selected_label,
                is_auto=is_auto,
                sample_type_column=sample_type_col,
                batch_column=batch_col,
                injection_order_column=inject_order_col,
                qc_label=qc_label,
                actual_label=actual_label,
                plot_payload=CorrectionPlotPayload(
                    source_data=snapshot_dataset(self.dataset),
                    selected_stages={
                        name: snapshot_dataset(stage)
                        for name, stage in selected_datasets.items()
                    },
                    candidate_results=snapshot_plot_value(results_store),
                    selected_prediction=(
                        snapshot_dataset(prediction_dataset)
                        if prediction_dataset is not None
                        else None
                    ),
                    internal_standard_ids=tuple(self.valid_internal_standards),
                    boundary_type=self.config.get("boundary", "IQR"),
                ),
            ),
        )
        for key in (
            "base_est", "implementation", "rlsc_robust",
            "rloess_span_selection",
            "rloess_span",
            "rloess_iterations",
        ):
            self.config[key] = requested_config[key]
        return result

    def _validate_r_options(self, method: str, implementation: str) -> None:
        """Reject misleading implementation-specific inputs before fitting."""
        r_only = (
            "serrf_r_source",
            "ruv_replicate_column",
        )
        if implementation != "r":
            if any(self.config.get(key) is not None for key in r_only) or (
                self.config.get("waveica_alpha", 0.0) != 0.0
            ):
                raise ValueError(
                    "R-specific options require implementation='r'."
                )
            if (
                self.config.get("ruv_control_features") is not None
                and method not in {"RUV-III", "AUTO"}
            ):
                raise ValueError(
                    "Explicit ruv_control_features require method='RUV-III'."
                )
            return
        if method != "Metanorm-rLOESS" and (
            self.config.get("is_logged") or self.config.get("is_scaled")
        ):
            raise ValueError(
                f"R {method} expects unlogged, unscaled raw intensities."
            )
        if method == "RUV-III" and not self.config.get("ruv_control_features"):
            raise ValueError(
                "R RUV-III requires explicit ruv_control_features; "
                "empirical low-RSD features are not assumed negative controls."
            )
        if method == "SERRF" and not self.config.get("serrf_r_source"):
            raise ValueError(
                "R SERRF requires serrf_r_source pointing to the pinned "
                "author source or its verified extracted function file."
            )
        native_defaults = {}
        if method == "WaveICA 2.0":
            native_defaults = {
                "waveica_levels": None,
                "waveica_spline_knots": 5,
                "waveica_max_iter": 1000,
            }
        elif method == "SERRF":
            native_defaults = {
                "serrf_n_tree": 100,
                "serrf_backend": "loky",
                "serrf_batch_size": "auto",
            }
        changed = [
            key
            for key, default in native_defaults.items()
            if self.config.get(key) != default
        ]
        if changed:
            raise ValueError(
                f"Native Python options do not tune R {method}: {changed}."
            )

    @log_execution_time
    def run_signal_correction(
        self,
        output_dir: str | None = None,
        **runtime_overrides: object,
    ) -> StageResult[dict[str, MetaboDataset]]:
        """Return the structured signal-correction stage result.

        Named keyword settings such as ``base_est``, ``loess_span``,
        ``rlsc_robust``, or ``serrf_n_tree`` take precedence over the pipeline
        configuration and module defaults for this processor instance.
        """
        return CorrectionStageRunner(
            self,
            output_dir,
            runtime_overrides=runtime_overrides,
            allowed_override_keys=self._RUNTIME_CONFIG_KEYS,
        ).run()

    @property
    def correction_metrics(self) -> Dict[str, Any]:
        """Extracts comprehensive multi-stage correction metrics."""
        execution = {
            **self.config,
            **getattr(self, "_correction_output_attrs", {}),
        }
        stage = execution.get("pipeline_stage", "Unknown")
        rsd_base = execution.get("qc_rsd_baseline")
        rsd_curr_oof = execution.get("qc_rsd_current_oof")
        rsd_curr_full = execution.get("qc_rsd_current_full")
        hist_oof = execution.get("rsd_history_oof", {})
        hist_full = execution.get("rsd_history_full", {})
        method = _normalize_correction_method(
            execution.get("base_est", "Unknown")
        )

        metrics = {
            "correction_status": stage,
            "selection": execution.get(
                "selection",
                {
                    "requested_method": method,
                    "selected_method": method,
                    "selected_label": execution.get(
                        "correction_method_label", method
                    ),
                    "is_auto": execution.get("is_auto", False),
                    "selected_score": None,
                    "selection_margin": None,
                    "candidate_results": [],
                },
            ),
            "overall_performance": {
                "median_qc_rsd_baseline": rsd_base,
                "median_qc_rsd_current_oof": rsd_curr_oof,
                "median_qc_rsd_current_full": rsd_curr_full,
                "relative_noise_reduction_oof": None,
                "relative_noise_reduction_full": None,
            },
            "stages_executed": [],
        }

        if rsd_base is not None and rsd_base > 0:
            if rsd_curr_oof is not None:
                oof_reduction = (rsd_base - rsd_curr_oof) / rsd_base
                metrics["overall_performance"][
                    "relative_noise_reduction_oof"
                ] = oof_reduction
            if rsd_curr_full is not None:
                full_reduction = (rsd_base - rsd_curr_full) / rsd_base
                metrics["overall_performance"][
                    "relative_noise_reduction_full"
                ] = full_reduction

        for stage_name in hist_oof.keys():
            if stage_name == "Original":
                continue

            alg_identifier = method
            if "Inter-batch" in stage_name:
                alg_identifier = "QC Median Alignment"

            # Dynamically build parameter dict based on the executed algorithm
            stage_params = {}
            if alg_identifier != "QC Median Alignment":
                if alg_identifier in ("QC-RLSC", "LOESS"):
                    stage_params["loess_span"] = execution.get("loess_span")
                    stage_params["loess_degree"] = execution.get("loess_degree")
                    stage_params["rlsc_span_selection"] = execution.get(
                        "rlsc_span_selection"
                    )
                    stage_params["rlsc_span_grid"] = execution.get(
                        "rlsc_span_grid"
                    )
                    stage_params["rlsc_min_qc"] = execution.get("rlsc_min_qc")
                    stage_params["rlsc_robust"] = execution.get("rlsc_robust")
                    if execution.get("rlsc_robust"):
                        provenance = metrics["selection"].get(
                            "implementation_provenance", []
                        )
                        if provenance:
                            stage_params = dict(provenance[0]["parameters"])
                elif alg_identifier in ("QC-RFSC", "RF"):
                    stage_params["n_estimators"] = execution.get("rf_n_tree")
                elif alg_identifier == "QC-SVR":
                    stage_params["svr_kernel"] = execution.get("svr_kernel")
                    stage_params["svr_c"] = execution.get("svr_c")
                    stage_params["svr_gamma"] = execution.get("svr_gamma")
                elif alg_identifier == "SERRF":
                    stage_params["n_estimators"] = execution.get("serrf_n_tree")
                    stage_params["n_corr_features"] = execution.get(
                        "serrf_corr_features"
                    )
                    stage_params["backend"] = execution.get("serrf_backend")
                    stage_params["batch_size"] = execution.get(
                        "serrf_batch_size"
                    )
                elif alg_identifier == "RUV-III":
                    stage_params["ruv_k"] = execution.get("ruv_k")
                elif alg_identifier == "WaveICA 2.0":
                    stage_params["n_components"] = execution.get(
                        "waveica_components"
                    )
                    stage_params["cutoff"] = execution.get("waveica_cutoff")
                    stage_params["n_levels"] = execution.get("waveica_levels")
                    stage_params["spline_knots"] = execution.get(
                        "waveica_spline_knots"
                    )
                    stage_params["max_iter"] = execution.get("waveica_max_iter")
                elif alg_identifier == "Metanorm-rLOESS":
                    provenance = metrics["selection"].get(
                        "implementation_provenance", []
                    )
                    if provenance:
                        stage_params.update(provenance[0]["parameters"])

                if metrics["selection"].get("implementation") == "r":
                    provenance = metrics["selection"].get(
                        "implementation_provenance", []
                    )
                    stage_params = (
                        dict(provenance[0]["parameters"]) if provenance else {}
                    )
                elif alg_identifier not in (
                    "RUV-III",
                    "WaveICA 2.0",
                    "Metanorm-rLOESS",
                ):
                    stage_params["cv_folds"] = execution.get("cv_folds")

            metrics["stages_executed"].append(
                {
                    "stage_name": stage_name,
                    "algorithm": alg_identifier,
                    "parameters": stage_params,
                    "stage_qc_rsd_oof": hist_oof.get(stage_name),
                    "stage_qc_rsd_full": hist_full.get(stage_name),
                }
            )

        return metrics
