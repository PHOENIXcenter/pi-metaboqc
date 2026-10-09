"""Exercise the complete VSN stage against an independent original R call."""

import numpy as np
import pandas as pd

from pimqc import DataNormalizer, MetaboDatasetBuilder
from pimqc.serialization import read_audit_payload, write_audit_payload

from .helpers import require_r_package
from .vsn_reference import run_r_vsn


def test_r_vsn_stage_matches_original_and_serializes(tmp_path):
    """The stage neither shifts the original R output nor logs it again."""
    require_r_package("vsn")
    rng = np.random.default_rng(30)
    columns = [f"S{i}" for i in range(15)]
    raw = pd.DataFrame(
        rng.lognormal(6, 0.25, (100, 15)),
        index=[f"F{i}" for i in range(100)],
        columns=columns,
    )
    metadata = pd.DataFrame(
        {
            "Sample Name": columns,
            "Sample Type": ["QC"] * 5 + ["Sample"] * 9 + ["Blank"],
            "Batch": ["B1"] * 15,
            "Inject Order": range(15),
        }
    )
    data = MetaboDatasetBuilder(metadata, raw).run_build().data
    result = DataNormalizer(
        data, norm_method="VSN", implementation="r", global_seed=30
    ).run_normalization()
    reference = run_r_vsn(data.intensity.iloc[:, :-1])
    np.testing.assert_allclose(
        result.data.intensity.to_numpy(),
        reference.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    provenance = result.audit.selection["implementation_provenance"]
    assert provenance[0]["package"] == "vsn"
    assert provenance[0]["package_version"]
    path = tmp_path / "normalization-audit"
    write_audit_payload(result.audit, path)
    assert read_audit_payload(path).selection == result.audit.selection
