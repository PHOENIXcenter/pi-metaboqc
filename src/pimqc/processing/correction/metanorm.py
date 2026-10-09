"""Call the original Metanorm rLOESS kernel through the optional R runtime.

This unadapted full-fit reference includes Blank in its centering mean and
does not supply OOF validation. Production robust QC-RLSC shares its rLOESS
definition but provides nonblank centering and held-out QC fit/predict.
The exported worker avoids the high-level entry point's PSOCK cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..r_backend import run_r_function


_RLOESS_CODE = """
function(mat, params) {
    batch <- factor(params$batch)
    result <- lapply(seq_len(nrow(mat)), function(i) {
        tryCatch(
            metanorm::metanormWorker(
                raw = unname(mat[i, ]), order = params$order,
                keepScale = params$keepScale, QConly = params$QConly,
                QCcheck = params$QCcheck, QCcheckp = params$QCcheckp,
                changepoints = params$changepoints, type = params$type,
                batch = batch, batchwise = params$batchwise,
                weights = params$weights, model = params$model,
                k = params$k, cv = params$cv,
                plotdir = NULL, plottype = "pdf", i = i
            ),
            error = function(e) stop(
                sprintf("Metanorm feature %d: %s", i, conditionMessage(e))
            )
        )
    })
    do.call(rbind, result)
}
"""


class MetanormRLOESSCorrector:
    """Fit QC-only, batchwise robust quadratic LOESS with GCV span choice.

    Input intensities must be positive (existing missing values are allowed).
    Correction uses ``log2 -> keepScale=TRUE -> exp2``. No pseudocount,
    clipping, extrapolation, extra batch alignment, or fallback is introduced.
    """

    def __init__(self, *, random_state: int) -> None:
        """Record the reproducible R seed without initializing R."""
        self.random_state = random_state
        self.provenance: dict[str, Any] = {}

    def fit_transform(
        self,
        intensity_df: pd.DataFrame,
        batch_array: np.ndarray,
        qc_mask: np.ndarray,
        order_array: np.ndarray,
        sample_type_array: np.ndarray | None = None,
    ) -> dict[str, tuple[pd.DataFrame, None]]:
        """Return original-package corrections without fabricated OOF data."""
        values = intensity_df.to_numpy(dtype=float, copy=True)
        batches = np.asarray(batch_array)
        qc = np.asarray(qc_mask, dtype=bool)
        order = np.asarray(order_array, dtype=float)
        n_samples = values.shape[1]
        sample_types = (
            np.asarray(sample_type_array, dtype=str)
            if sample_type_array is not None
            else np.where(qc, "QC", "Sample")
        )
        if any(
            array.shape != (n_samples,)
            for array in (batches, qc, order, sample_types)
        ):
            raise ValueError("Metanorm metadata must align with all samples.")
        if pd.isna(batches).any() or not np.isfinite(order).all():
            raise ValueError("Metanorm requires finite order and batch labels.")
        if np.isinf(values).any() or (values <= 0).any():
            raise ValueError(
                "Metanorm-rLOESS requires positive raw intensities; "
                "resolve zero/negative/infinite values explicitly first."
            )
        for batch in pd.unique(batches):
            batch_mask = batches == batch
            fit_mask = batch_mask & qc
            if np.unique(order[fit_mask]).size < 4:
                raise ValueError(
                    f"Metanorm batch {batch!r} requires at least four "
                    "distinct QC injection orders."
                )
            # The original stats::predict.loess does not extrapolate. Check
            # per feature because missing boundary QCs narrow its support.
            for feature, row in zip(intensity_df.index, values):
                observed = np.isfinite(row)
                observed_qc = fit_mask & observed
                support = order[observed_qc]
                if np.unique(support).size < 4:
                    raise ValueError(
                        f"Metanorm feature {feature!r}, batch {batch!r}: "
                        "fewer than four finite distinct QC orders."
                    )
                targets = order[batch_mask & observed]
                if targets.size == 0:
                    raise ValueError(
                        f"Metanorm feature {feature!r}, batch {batch!r}: "
                        "no finite observations are available."
                    )
                if targets.min() < support.min() or (
                    targets.max() > support.max()
                ):
                    outside = batch_mask & observed & (
                        (order < support.min()) | (order > support.max())
                    )
                    sample = intensity_df.columns[np.flatnonzero(outside)[0]]
                    raise ValueError(
                        f"Metanorm feature {feature!r}, batch {batch!r}: "
                        f"sample {sample!r} outside QC order range; original "
                        "rLOESS cannot extrapolate."
                    )
        # Metanorm recognizes the literal QC label. Preserve Blank and other
        # labels, but prevent a non-QC's coincidental label from fitting.
        r_types = np.where(
            qc, "QC", np.where(sample_types == "QC", "Non-QC", sample_types)
        )
        parameters = {
            "order": order.tolist(),
            "batch": batches.astype(str).tolist(),
            "type": r_types.tolist(),
            "input_type": sample_types.tolist(),
            "keepScale": True,
            "QConly": True,
            "QCcheck": False,
            "QCcheckp": 0.1,
            "changepoints": False,
            "batchwise": True,
            "weights": [1.0] * n_samples,
            "model": "rLOESS",
            "k": min(n_samples * 0.9, 10),
            "cv": "GCV",
        }
        logged = pd.DataFrame(
            np.log2(values),
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        corrected_log, self.provenance = run_r_function(
            logged,
            method="Metanorm-rLOESS",
            package="metanorm",
            function="metanorm::metanormWorker",
            code=_RLOESS_CODE,
            parameters=parameters,
            seed=self.random_state,
            transforms=(
                "log2(raw intensities)",
                "QC-only fitting; original full-input log-mean centering "
                "includes all sample types, including Blank",
                "QC labels canonicalized to QC; non-QC labels retained",
                "exp2(corrected values)",
            ),
            allow_missing=True,
            defer_output_domain=True,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = np.exp2(corrected_log.to_numpy(dtype=float))
        frame = pd.DataFrame(
            corrected,
            index=intensity_df.index,
            columns=intensity_df.columns,
        )
        frame.attrs.update(intensity_df.attrs)
        return {"Metanorm-rLOESS corrected": (frame, None)}
