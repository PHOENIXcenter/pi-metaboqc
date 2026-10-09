"""Small real-R AUTO runs exercise provider substitution and normal scoring."""

import numpy as np

from pimqc import DataNormalizer, MissingValueImputer

from .helpers import require_r_package
from .test_imputation_r_backend import _dataset


def test_original_r_auto_imputation():
    """Both original R kernels participate in the existing MAR benchmark."""
    require_r_package("pcaMethods")
    require_r_package("imputeLCMD")
    data = _dataset(False)
    result = MissingValueImputer(
        data, mar_method="Auto", implementation="r", global_seed=18
    ).run_imputation()
    selection = result.audit.metrics["selection"]
    rows = {row["method"]: row for row in selection["candidate_results"]}
    for method, package in (("BPCA", "pcaMethods"), ("QRILC", "imputeLCMD")):
        assert rows[method]["status"] == "ok", rows[method]
        assert rows[method]["implementation"] == "r"
        assert all(
            record["package"] == package and record["package_version"]
            for record in rows[method]["implementation_provenance"]
        )
        assert rows[method]["implementation_provenance"]
    assert selection["requested_implementation"] == "r"
    assert selection["implementation"] == (
        rows[selection["selected_method"]]["implementation"]
    )
    output = result.data.intensity.to_numpy()
    assert (np.isfinite(output) & (output > 0)).all()
    observed = data.intensity.notna().to_numpy()
    np.testing.assert_array_equal(
        output[observed], data.intensity.to_numpy()[observed]
    )


def test_original_r_auto_normalization():
    """Original VSN competes with native strategies using unchanged scores."""
    require_r_package("vsn")
    data = _dataset(False)
    data.intensity = data.intensity.fillna(250.0)
    result = DataNormalizer(
        data, norm_method="Auto", implementation="r", n_jobs=1,
        global_seed=18,
    ).run_normalization()
    selection = result.audit.selection
    rows = {row["method"]: row for row in result.audit.candidate_results}
    vsn = rows["VSN"]
    assert vsn["status"] == "ok", vsn
    assert vsn["implementation"] == "r"
    assert vsn["implementation_provenance"][0]["package"] == "vsn"
    assert vsn["implementation_provenance"][0]["package_version"]
    assert selection["requested_implementation"] == "r"
    assert selection["implementation"] == (
        rows[selection["selected_method"]]["implementation"]
    )
    assert np.isfinite(result.data.intensity.to_numpy()).all()
