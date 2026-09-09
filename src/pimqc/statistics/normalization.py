"""Precompute normalization diagnostics before constructing plot payloads.

Scientific RLE summaries and paired significance tests live in this module so
saved normalization audits render without recomputing statistical evidence.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics as su


def paired_wilcoxon_pvalue(
    before_values: pd.Series | None,
    after_values: pd.Series | None,
) -> float:
    """
    Calculate a paired Wilcoxon signed-rank p-value for matched QC samples.
    """
    if before_values is None or after_values is None:
        return float("nan")

    common_index = before_values.index.intersection(after_values.index)
    if len(common_index) < 3:
        return float("nan")

    before_arr = before_values.loc[common_index].to_numpy(dtype=float)
    after_arr = after_values.loc[common_index].to_numpy(dtype=float)
    finite_mask = np.isfinite(before_arr) & np.isfinite(after_arr)
    before_arr = before_arr[finite_mask]
    after_arr = after_arr[finite_mask]
    if before_arr.size < 3:
        return float("nan")
    if np.allclose(before_arr, after_arr):
        return 1.0

    try:
        return float(
            stats.wilcoxon(
                before_arr,
                after_arr,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        return float("nan")


def calculate_rle_diagnostics(
    stages: list[tuple[str, pd.DataFrame]],
) -> dict[str, dict]:
    """Return compact per-stage RLE summaries and paired QC significance."""
    plot_records = []
    sample_metric_values: dict[str, dict[str, pd.Series]] = {}
    for label, obj in stages:
        log_d = su._extract_log2_target(obj)
        if log_d is None or log_d.empty:
            continue

        qc_cols = su._role_columns(obj, "QC sample", "QC").intersection(
            log_d.columns
        )
        if len(qc_cols) < 2:
            continue

        global_feature_median = log_d.median(axis=1)
        qc_rle = log_d[qc_cols].astype(float).sub(global_feature_median, axis=0)
        qc_rle = qc_rle.replace([np.inf, -np.inf], np.nan)

        sample_medians = qc_rle.median(axis=0).replace(
            [np.inf, -np.inf], np.nan
        )
        sample_q25 = qc_rle.quantile(0.25, axis=0)
        sample_q75 = qc_rle.quantile(0.75, axis=0)
        sample_iqrs = (sample_q75 - sample_q25).replace(
            [np.inf, -np.inf], np.nan
        )
        sample_iqrs = sample_iqrs.replace([np.inf, -np.inf], np.nan)

        sample_medians = sample_medians.dropna()
        sample_iqrs = sample_iqrs.dropna()
        if sample_medians.empty or sample_iqrs.empty:
            continue

        sample_metric_values[label] = {
            "RLE center offset": sample_medians.abs(),
            "RLE spread": sample_iqrs,
        }
        center_values = sample_metric_values[label]["RLE center offset"]
        spread_values = sample_metric_values[label]["RLE spread"]

        center_offset = su.finite_or_nan(center_values.median())
        rle_spread = su.finite_or_nan(spread_values.median())
        if not all(np.isfinite(v) for v in [center_offset, rle_spread]):
            continue

        plot_records.extend(
            [
                {
                    "Metric": "RLE center offset",
                    "Stage": label,
                    "Value": center_offset,
                    "Q25": su.finite_or_nan(center_values.quantile(0.25)),
                    "Q75": su.finite_or_nan(center_values.quantile(0.75)),
                },
                {
                    "Metric": "RLE spread",
                    "Stage": label,
                    "Value": rle_spread,
                    "Q25": su.finite_or_nan(spread_values.quantile(0.25)),
                    "Q75": su.finite_or_nan(spread_values.quantile(0.75)),
                },
            ]
        )

    output = {
        label: {
            "rle_records": [
                record for record in plot_records if record["Stage"] == label
            ]
        }
        for label, _ in stages
    }
    output.setdefault("After Norm", {})["rle_pvalues"] = {
        metric: paired_wilcoxon_pvalue(
            sample_metric_values.get("Before Norm", {}).get(metric),
            sample_metric_values.get("After Norm", {}).get(metric),
        )
        for metric in ("RLE center offset", "RLE spread")
    }
    return output
