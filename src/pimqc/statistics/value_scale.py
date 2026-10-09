"""Resolve explicit evaluation scales for processed intensity matrices.

Scale-aware diagnostics distinguish raw and transformed values without
clipping signed outputs or silently treating generalized-log values as raw
measurements. Unsupported scale declarations remain validation errors.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


VALUE_SCALES = frozenset(
    {"raw_positive", "log2", "log2p1", "vsn_glog", "centered_scaled"}
)


def resolve_value_scale(settings: Mapping[str, Any]) -> str:
    """Prefer explicit stamps; identify established legacy transforms only."""
    if settings.get("is_scaled", False):
        return "centered_scaled"
    explicit = settings.get("value_scale")
    if explicit is not None:
        if explicit not in VALUE_SCALES:
            raise ValueError(f"Unknown numeric value scale: {explicit!r}.")
        return str(explicit)
    if settings.get("is_logged", False):
        method = str(settings.get("norm_method", "")).upper()
        return "vsn_glog" if method == "VSN" else "log2"
    return "raw_positive"


def positive_view_rsd(
    frame: pd.DataFrame, settings: Mapping[str, Any]
) -> tuple[pd.Series, dict[str, Any]]:
    """Calculate CV in a named positive view, stably up to row scaling.

    VSN's exp2 view is not its inverse or an original-intensity measurement.
    For log2p1, retaining the minus-one term is essential. Multiplicative
    per-feature scaling leaves CV unchanged and avoids exponential overflow.
    """
    scale = resolve_value_scale(settings)
    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values)
    rsd = pd.Series(np.nan, index=frame.index, dtype=float)
    report: dict[str, Any] = {
        "input_scale": scale,
        "evaluation_scale": {
            "raw_positive": "raw intensity",
            "log2": "inverse log2 intensity",
            "log2p1": "inverse log2p1 intensity",
            "vsn_glog": "exponentiated VSN diagnostic (not raw inverse)",
            "centered_scaled": "unavailable for centered/scaled input",
        }[scale],
        "is_raw_intensity_inverse": scale in {"log2", "log2p1"},
        "nonfinite_input_count": int((~finite).sum()),
        "excluded_nonpositive_count": 0,
        "valid_feature_count": 0,
        "unavailable_feature_count": len(frame),
        "status": "unavailable",
    }
    if scale == "centered_scaled":
        return rsd, report
    valid = finite.copy()
    if scale in {"raw_positive", "log2p1"}:
        valid &= values > 0
        report["excluded_nonpositive_count"] = int(
            (finite & (values <= 0)).sum()
        )
    for row in range(len(frame)):
        observed = values[row, valid[row]]
        if observed.size < 2:
            continue
        if scale == "raw_positive":
            positive = observed / observed.max()
        else:
            maximum = observed.max()
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                positive = np.exp2(observed - maximum)
                if scale == "log2p1":
                    positive *= -np.expm1(-observed * np.log(2.0))
        mean = positive.mean()
        if np.isfinite(positive).all() and np.isfinite(mean) and mean > 0:
            rsd.iloc[row] = positive.std(ddof=1) / mean
    count = int(rsd.notna().sum())
    report.update(
        valid_feature_count=count,
        unavailable_feature_count=int(len(frame) - count),
        status="available" if count else "unavailable",
    )
    return rsd, report
