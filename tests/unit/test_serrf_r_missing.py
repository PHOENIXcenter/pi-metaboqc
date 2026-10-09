"""Guard batch-reference support without initializing R."""

import numpy as np
import pytest

from pimqc.processing.correction.serrf_r_missing import (
    supported_serrf_features,
)
from tests.unit.test_serrf_r_backend import _inputs


def test_missing_batch_support_ignores_held_out_values():
    """Exclude held-out responses when determining batchwise feature support."""
    data, batches, qc, _, blank = _inputs()
    held_out = np.zeros(len(qc), dtype=bool)
    held_out[0] = True
    data.loc[0, (batches == "B1") & ~held_out] = np.nan
    before = data.copy(deep=True)
    supported, report = supported_serrf_features(
        data, batches, ~blank & ~held_out, 4,
    )
    assert not supported[0]
    assert supported[1:].all()
    assert report["all_missing_features_by_fit_batch"] == {"B1": ["0"]}
    assert report["unsupported_feature_count"] == 1
    assert data.equals(before)
    full_support, _ = supported_serrf_features(data, batches, ~blank, 4)
    assert full_support.all()


def test_no_supported_predictors_raises_instead_of_borrowing_a_batch():
    """Fail when no feature is supported rather than borrowing another batch."""
    data, batches, _, _, blank = _inputs()
    data.loc[:, batches == "B1"] = np.nan
    with pytest.raises(ValueError, match="supported features"):
        supported_serrf_features(data, batches, ~blank, 4)
