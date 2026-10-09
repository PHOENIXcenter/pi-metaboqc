"""Batch-wise native SERRF with independently refitted QC validation.

The workflow follows the author's correlation selection, role-specific
standardization and multiplicative batch alignment. Forests use sklearn,
not ranger. Random imputation and post-hoc outlier replacement are omitted.
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from ...constants import DEFAULT_RANDOM_SEED
from ...runtime import joblib_execution_context, joblib_progress
from .qc_validation import assign_qc_folds


class SERRFCorrector:
    """Native SERRF with within-batch held-out QC validation.

    Biological predictor and batch location/scale estimates use the available
    cohort, as in the original. This is transductive correction, not a model
    for unseen biological cohorts. QC folds refit selection, standardization,
    forests and response scaling without the held-out QC measurements.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        cv_folds: int = 5,
        n_corr_features: int = 10,
        random_state: int = DEFAULT_RANDOM_SEED,
        n_jobs: int = -1,
        joblib_backend: str = "loky",
        joblib_batch_size: Union[str, int] = "auto",
        cv_strategy: str = "random",
    ) -> None:
        """Configure forests and validation.

        ``cv_strategy='random'`` preserves the production splitter. The
        optional ``'blocked'`` strategy sorts QC observations by injection
        order within each batch before assigning contiguous folds.
        """
        if n_estimators < 1 or cv_folds < 2 or n_corr_features < 0:
            raise ValueError(
                "Require n_estimators >= 1, cv_folds >= 2 and "
                "n_corr_features >= 0."
            )
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        if cv_strategy not in {"random", "blocked"}:
            raise ValueError("cv_strategy must be random or blocked.")
        self.n_estimators = n_estimators
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.n_corr_features = n_corr_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.joblib_backend = str(joblib_backend).lower()
        self.joblib_batch_size = joblib_batch_size

    @staticmethod
    def _correlation(values: np.ndarray) -> np.ndarray:
        """Pairwise Spearman correlation on positive finite observations."""
        frame = pd.DataFrame(
            np.where(np.isfinite(values) & (values > 0), values, np.nan)
        )
        return frame.corr(method="spearman", min_periods=3).to_numpy()

    def _select_predictors(
        self,
        feature: int,
        qc_corr: np.ndarray,
        sample_corr: np.ndarray,
    ) -> np.ndarray:
        """Expand ranked lists until their overlap is sufficiently large."""
        qc_scores = qc_corr[:, feature]
        sample_scores = sample_corr[:, feature]
        valid = np.isfinite(qc_scores) & np.isfinite(sample_scores)
        valid[feature] = False
        available = np.flatnonzero(valid)
        count = min(self.n_corr_features, available.size)
        if count == 0:
            return np.empty(0, dtype=int)
        qc_rank = available[
            np.argsort(-np.abs(qc_scores[available]), kind="stable")
        ]
        sample_rank = available[
            np.argsort(-np.abs(sample_scores[available]), kind="stable")
        ]
        # The author's first overlap can contain more than the requested num.
        for cutoff in range(count, available.size + 1):
            selected = qc_rank[:cutoff][
                np.isin(qc_rank[:cutoff], sample_rank[:cutoff])
            ]
            if selected.size >= count:
                return selected
        return available

    @staticmethod
    def _scale_fit(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate predictor scale using only the supplied training role."""
        clean = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        means = pd.DataFrame(clean).mean().to_numpy()
        scales = pd.DataFrame(clean).std(ddof=1).to_numpy()
        good = np.isfinite(means) & np.isfinite(scales) & (scales > 0)
        return means, scales, good

    @staticmethod
    def _transform(
        values: np.ndarray, means: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        """Map missing predictors to their training-role mean."""
        clean = np.where(np.isfinite(values) & (values > 0), values, means)
        return (clean - means) / scales

    @staticmethod
    def _ratio(
        values: np.ndarray, baseline: np.ndarray, anchor: float
    ) -> np.ndarray:
        """Return missing when no positive finite denominator is available."""
        result = np.full_like(values, np.nan, dtype=float)
        valid = (
            np.isfinite(values)
            & (values > 0)
            & np.isfinite(baseline)
            & (baseline > 0)
        )
        result[valid] = values[valid] / baseline[valid] * anchor
        return result

    def _fit_feature_pass(
        self,
        feature: int,
        values: np.ndarray,
        orders: np.ndarray,
        qc_mask: np.ndarray,
        sample_mask: np.ndarray,
        training_qc: np.ndarray,
        contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        diagnostics: dict,
    ) -> np.ndarray:
        """Fit all batches using one full-data or fold-specific QC set."""
        y = values[:, feature]
        result = np.full_like(y, np.nan, dtype=float)
        invalid_baseline = np.zeros(y.size, dtype=bool)
        valid_y = np.isfinite(y) & (y > 0)
        global_train = training_qc & valid_y
        if global_train.sum() < 2:
            diagnostics["unsupported_feature_batch_fits"] += len(contexts)
            return result
        qc_anchor = float(np.median(y[global_train]))
        valid_samples = sample_mask & valid_y
        sample_anchor = (
            float(np.median(y[valid_samples]))
            if valid_samples.any()
            else qc_anchor
        )
        for in_batch, qc_corr, sample_corr in contexts:
            train = in_batch & global_train
            query_qc = in_batch & qc_mask
            samples = in_batch & sample_mask
            if train.sum() < 2:
                diagnostics["unsupported_feature_batch_fits"] += 1
                result[query_qc] = np.nan
                continue
            selected = self._select_predictors(feature, qc_corr, sample_corr)
            if self.n_corr_features == 0:
                x_train = orders[train, None]
                x_qc = orders[query_qc, None]
                x_sample = orders[samples, None]
            elif selected.size:
                predictors = values[:, selected]
                qc_mean, qc_scale, good = self._scale_fit(predictors[train])
                if samples.sum() >= 2:
                    sm_mean, sm_scale, sm_good = self._scale_fit(
                        predictors[samples]
                    )
                    good &= sm_good
                else:
                    sm_mean, sm_scale = qc_mean, qc_scale
                predictors = predictors[:, good]
                x_train = self._transform(
                    predictors[train], qc_mean[good], qc_scale[good]
                )
                x_qc = self._transform(
                    predictors[query_qc], qc_mean[good], qc_scale[good]
                )
                x_sample = self._transform(
                    predictors[samples], sm_mean[good], sm_scale[good]
                )
            else:
                x_train = np.empty((train.sum(), 0))
                x_qc = np.empty((query_qc.sum(), 0))
                x_sample = np.empty((samples.sum(), 0))

            center = float(np.mean(y[train]))
            response = y[train] - center
            good_samples = samples & valid_y
            # Match the author's amplitude guard in QC-heavy batches.
            if good_samples.sum() >= 2 and 2 * train.sum() >= samples.sum():
                sample_sd = float(np.std(y[good_samples], ddof=1))
                qc_sd = float(np.std(y[train], ddof=1))
                if sample_sd > 0 and qc_sd > sample_sd:
                    response = response / (qc_sd / sample_sd)
            if x_train.shape[1]:
                diagnostics["forest_feature_batch_fits"] += 1
                forest = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features="sqrt",
                    min_samples_split=5,
                    min_samples_leaf=1,
                    random_state=self.random_state,
                    n_jobs=1,
                )
                forest.fit(x_train, response)
                pred_train = forest.predict(x_train)
                pred_qc = forest.predict(x_qc)
                pred_sample = (
                    forest.predict(x_sample) if samples.any() else np.empty(0)
                )
            else:
                diagnostics["location_only_feature_batch_fits"] += 1
                # Insufficient covariates permit location alignment only.
                pred_train = np.zeros(train.sum())
                pred_qc = np.zeros(query_qc.sum())
                pred_sample = np.zeros(samples.sum())

            train_ratio = self._ratio(y[train], center + pred_train, qc_anchor)
            valid_train = np.isfinite(train_ratio) & (train_ratio > 0)
            train_median = (
                float(np.median(train_ratio[valid_train]))
                if valid_train.any()
                else float("nan")
            )
            qc_factor = qc_anchor / train_median
            qc_baseline = center + pred_qc
            invalid_baseline[query_qc] = ~np.isfinite(qc_baseline) | (
                qc_baseline <= 0
            )
            result[query_qc] = (
                self._ratio(y[query_qc], center + pred_qc, qc_anchor)
                * qc_factor
            )
            if good_samples.any():
                sample_center = float(np.mean(y[good_samples]))
                baseline = sample_center + pred_sample - np.mean(pred_sample)
                invalid_baseline[samples] = ~np.isfinite(baseline) | (
                    baseline <= 0
                )
                normalized = self._ratio(y[samples], baseline, sample_anchor)
                usable = np.isfinite(normalized) & (normalized > 0)
                if usable.any():
                    normalized *= sample_anchor / np.median(normalized[usable])
                result[samples] = normalized

        # Preserve QC's standardized position relative to the sample cohort.
        # Each fold recomputes this scalar with its own fitted sample output.
        mapping_samples = valid_samples & np.isfinite(result) & (result > 0)
        corrected_samples = result[mapping_samples]
        if corrected_samples.size >= 2:
            raw_sd = float(np.std(y[mapping_samples], ddof=1))
            corrected_sd = float(np.std(corrected_samples, ddof=1))
            if raw_sd > 0:
                desired_qc = (
                    np.median(corrected_samples)
                    + (qc_anchor - np.median(y[mapping_samples]))
                    / raw_sd * corrected_sd
                )
                mapping_qc = global_train & np.isfinite(result) & (result > 0)
                train_median = (
                    float(np.median(result[mapping_qc]))
                    if mapping_qc.any()
                    else float("nan")
                )
                if (
                    np.isfinite(desired_qc)
                    and desired_qc > 0
                    and np.isfinite(train_median)
                    and train_median > 0
                ):
                    result[qc_mask] *= desired_qc / train_median
        diagnostics["invalid_baseline_predictions"] += int(
            invalid_baseline.sum()
        )
        diagnostics["invalid_qc_baseline_predictions"] += int(
            (invalid_baseline & qc_mask).sum()
        )
        diagnostics["invalid_sample_baseline_predictions"] += int(
            (invalid_baseline & sample_mask).sum()
        )
        diagnostics["invalid_sample_indices"].extend(
            np.flatnonzero(invalid_baseline & sample_mask).tolist()
        )
        result[~valid_y | invalid_baseline] = np.nan
        return result

    def _process_single_feature(
        self,
        feature: int,
        values: np.ndarray,
        orders: np.ndarray,
        qc_mask: np.ndarray,
        sample_mask: np.ndarray,
        blank_mask: np.ndarray,
        folds: np.ndarray,
        contexts: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ) -> tuple[int, np.ndarray, np.ndarray, dict]:
        """Refit every estimated operation before predicting held-out QCs."""
        diagnostics = {
            stage: {
                "forest_feature_batch_fits": 0,
                "location_only_feature_batch_fits": 0,
                "unsupported_feature_batch_fits": 0,
                "invalid_baseline_predictions": 0,
                "invalid_qc_baseline_predictions": 0,
                "invalid_sample_baseline_predictions": 0,
                "invalid_sample_indices": [],
            }
            for stage in ("full", "oof")
        }
        full = self._fit_feature_pass(
            feature,
            values,
            orders,
            qc_mask,
            sample_mask,
            qc_mask,
            contexts[0],
            diagnostics["full"],
        )
        # Project fitted QC correction factors onto blanks by injection order.
        y = values[:, feature]
        for in_batch, _, _ in contexts[0]:
            reference = in_batch & qc_mask & np.isfinite(full) & (y > 0)
            targets = in_batch & blank_mask
            if reference.any() and targets.any():
                idx = np.flatnonzero(reference)
                idx = idx[np.argsort(orders[idx], kind="stable")]
                factor = np.interp(
                    orders[targets], orders[idx], full[idx] / y[idx]
                )
                full[targets] = y[targets] * factor
        oof = full.copy()
        oof[qc_mask] = np.nan
        for fold, fold_context in enumerate(contexts[1:]):
            held_out = qc_mask & (folds == fold)
            if not held_out.any():
                continue
            prediction = self._fit_feature_pass(
                feature,
                values,
                orders,
                qc_mask,
                sample_mask,
                qc_mask & ~held_out,
                fold_context,
                diagnostics["oof"],
            )
            valid = held_out & np.isfinite(y) & (y > 0)
            oof[valid] = prediction[valid]
        return feature, full, oof, diagnostics

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        blank_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Return full correction and independent held-out QC correction.

        Correlations are estimated separately within each batch and QC fold.
        Missing/nonpositive responses remain missing and are omitted
        from validation. At least two training QCs are needed per feature.
        """
        logger.info("Fitting batch-wise native SERRF with refitted QC folds...")
        values = intensity_df.T.to_numpy(dtype=float, copy=True)
        n_samples, n_features = values.shape
        batches = np.asarray(batch_array)
        orders = np.asarray(order_array, dtype=float)
        qc_mask = np.asarray(qc_mask)
        if qc_mask.dtype != np.bool_:
            raise ValueError("qc_mask must be boolean.")
        if blank_mask is None:
            blank_mask = np.zeros(n_samples, dtype=bool)
        blank_mask = np.asarray(blank_mask)
        if blank_mask.dtype != np.bool_:
            raise ValueError("blank_mask must be boolean.")
        if any(
            array.shape != (n_samples,)
            for array in (batches, orders, qc_mask, blank_mask)
        ):
            raise ValueError("Batch, order and masks must match sample count.")
        if pd.isna(batches).any() or not np.isfinite(orders).all():
            raise ValueError("Batch labels and injection orders must be valid.")
        if np.any(qc_mask & blank_mask):
            raise ValueError("QC and blank masks cannot overlap.")
        if qc_mask.sum() < 3:
            raise ValueError("SERRF requires at least three QCs.")
        sample_mask = ~(qc_mask | blank_mask)
        folds = assign_qc_folds(
            batches, qc_mask, self.cv_folds, self.random_state,
            strategy=self.cv_strategy, order_array=orders,
        )
        batch_masks = [batches == batch for batch in pd.unique(batches)]
        sample_corrs = [
            self._correlation(values[in_batch & sample_mask])
            if (in_batch & sample_mask).sum() >= 3
            else self._correlation(values[in_batch & qc_mask])
            for in_batch in batch_masks
        ]
        contexts = []
        for fold in range(-1, int(folds.max()) + 1):
            train = qc_mask if fold < 0 else qc_mask & (folds != fold)
            context = []
            for in_batch, sample_corr in zip(batch_masks, sample_corrs):
                qc_corr = self._correlation(values[in_batch & train])
                if (in_batch & sample_mask).sum() < 3:
                    sample_corr = qc_corr
                context.append((in_batch, qc_corr, sample_corr))
            contexts.append(context)

        workers = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        workers = min(workers, max(1, n_features))
        backend = self.joblib_backend
        if backend not in {"threading", "loky"}:
            logger.warning(f"Unknown SERRF backend {backend}; using loky.")
            backend = "loky"
        batch_size = self.joblib_batch_size
        if isinstance(batch_size, str) and batch_size.lower() != "auto":
            batch_size = int(batch_size)
        elif isinstance(batch_size, str):
            batch_size = "auto"
        with joblib_execution_context(backend):
            with joblib_progress(total=n_features, desc="SERRF"):
                results = Parallel(n_jobs=workers, batch_size=batch_size)(
                    delayed(self._process_single_feature)(
                        feature,
                        values,
                        orders,
                        qc_mask,
                        sample_mask,
                        blank_mask,
                        folds,
                        contexts,
                    )
                    for feature in range(n_features)
                )
        full_values = np.empty_like(values)
        oof_values = np.empty_like(values)
        for feature, full, oof, _ in results:
            full_values[:, feature] = full
            oof_values[:, feature] = oof
        full_frame = pd.DataFrame(
            full_values.T,
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        oof_frame = pd.DataFrame(
            oof_values.T,
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        self.diagnostics = {
            "workflow": "batch-wise role-scaled native SERRF",
            "cv_strategy": self.cv_strategy,
            "forest": "sklearn; sqrt mtry; min_samples_split=5",
            "validation": (
                "within-batch held-out QCs; fold-refitted selection, scaling "
                "and forests; biological cohort remains transductive"
            ),
            "fold_assignments": folds.tolist(),
            "effective_folds_by_batch": {
                str(batches[in_batch][0]): int(
                    np.unique(folds[in_batch & (folds >= 0)]).size
                )
                for in_batch in batch_masks
            },
            "qc_oof_value_coverage": float(
                np.isfinite(oof_values[qc_mask]).sum()
                / max(1, oof_values[qc_mask].size)
            ),
            "qc_oof_coverage_denominator": "all QC intensity cells",
            "qc_oof_eligible_value_coverage": float(
                np.isfinite(oof_values[qc_mask]).sum()
                / max(
                    1,
                    (
                        np.isfinite(values[qc_mask]) & (values[qc_mask] > 0)
                    ).sum(),
                )
            ),
            "fit_counts": {
                stage: {
                    key: sum(item[3][stage][key] for item in results)
                    for key in (
                        "forest_feature_batch_fits",
                        "location_only_feature_batch_fits",
                        "unsupported_feature_batch_fits",
                        "invalid_baseline_predictions",
                        "invalid_qc_baseline_predictions",
                        "invalid_sample_baseline_predictions",
                    )
                }
                for stage in ("full", "oof")
            },
            "fit_count_interpretation": (
                "OOF counts sum fold refits; baseline counts include all "
                "QC and sample predictions used for fold scaling"
            ),
            "invalid_sample_baseline_cells": [
                {
                    "feature": str(intensity_df.index[feature]),
                    "sample": str(intensity_df.columns[sample]),
                }
                for feature, _, _, diagnostic in results
                for sample in diagnostic["full"]["invalid_sample_indices"]
            ],
            "differences_from_author": [
                "sklearn forest and configured seed/tree count",
                "no random imputation or post-hoc outlier replacement",
                "location-only fallback for unavailable predictors",
                "blank projection uses interpolated fitted QC factors",
            ],
        }
        full_counts = self.diagnostics["fit_counts"]["full"]
        failed_samples = full_counts["invalid_sample_baseline_predictions"]
        failed_qcs = full_counts["invalid_qc_baseline_predictions"]
        unsupported = full_counts["unsupported_feature_batch_fits"]
        location_only = full_counts["location_only_feature_batch_fits"]
        coverage = self.diagnostics["qc_oof_eligible_value_coverage"]
        degraded = (
            failed_samples
            or failed_qcs
            or unsupported
            or location_only
            or coverage < 1
        )
        self.diagnostics["status"] = "degraded" if degraded else "ok"
        if degraded:
            logger.warning(
                "SERRF correction is degraded: {} sample and {} QC values "
                "have invalid baselines and become missing; "
                "{} unsupported and {} location-only feature/batch fits; "
                "eligible held-out QC coverage is {:.1%}.",
                failed_samples,
                failed_qcs,
                unsupported,
                location_only,
                coverage,
            )
        full_frame.attrs["serrf_diagnostics"] = self.diagnostics
        oof_frame.attrs["serrf_diagnostics"] = self.diagnostics
        return {"SERRF": (full_frame, oof_frame)}
