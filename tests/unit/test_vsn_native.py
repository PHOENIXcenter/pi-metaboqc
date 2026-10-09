"""Check affine VSN mathematics, robustness, missingness, and audit data."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import spearmanr

from pimqc.processing.normalization import DataNormalizer
from pimqc.processing.normalization.vsn import (
    _lts_selection,
    _vsn_nll_gradient,
)


def _model_data() -> pd.DataFrame:
    """Generate additive Gaussian errors on a known generalized-log scale."""
    rng = np.random.default_rng(42)
    mean = np.linspace(-1, 5, 300)[:, None]
    slopes = np.exp(np.linspace(-2, 2, 12))
    offsets = np.linspace(-0.8, 0.8, 12)
    values = (np.sinh(mean + rng.normal(0, 0.15, (300, 12))) - offsets) / slopes
    return pd.DataFrame(
        values,
        index=[f"F{i}" for i in range(300)],
        columns=[f"S{i}" for i in range(12)],
    )


def test_profile_likelihood_gradient_with_missing_values():
    """The analytic Jacobian agrees with independent numerical derivatives."""
    values = np.array(
        [
            [-2.0, -1.5, np.nan],
            [0.5, 0.2, 0.8],
            [3.0, 5.0, 4.0],
            [8.0, np.nan, 12.0],
        ]
    )
    parameters = np.array([0.1, -0.3, 0.2, -0.1, 0.2, 0.0])
    objective, gradient = _vsn_nll_gradient(parameters, values)
    numeric = approx_fprime(
        parameters, lambda p: _vsn_nll_gradient(p, values)[0], 1e-7
    )
    assert np.isfinite(objective)
    np.testing.assert_allclose(gradient, numeric, rtol=2e-5, atol=5e-6)


def test_lts_preserves_lowest_slice_and_trims_high_residuals():
    """Trimming keeps low intensities and follows R's missing-row rule."""
    means = np.arange(100, dtype=float)
    deviations = np.linspace(0.1, 0.2, 100)
    deviations[[5, 25, 45, 65, 85]] = 2.0
    transformed = means[:, None] + deviations[:, None] * [-1, 0, 1]
    transformed[2, 1] = np.nan
    transformed[50, 1] = np.nan
    selected = _lts_selection(transformed, 0.9)
    assert selected[:20].all()
    assert not selected[[25, 45, 50, 65, 85]].any()
    assert selected.sum() == 91


def test_vsn_stabilizes_known_model_and_preserves_transform_contract():
    """Sample slopes remove calibration while the glog stabilizes variance."""
    source = _model_data()
    original = source.copy(deep=True)
    output, metadata = DataNormalizer.calc_vsn_normalization(source)
    pd.testing.assert_frame_equal(source, original)
    assert output.index.equals(source.index)
    assert output.columns.equals(source.columns)
    assert metadata["vsn_sample_ids"] == source.columns.tolist()
    raw_rho = spearmanr(source.mean(axis=1), source.std(axis=1)).statistic
    fit_rho = spearmanr(output.mean(axis=1), output.std(axis=1)).statistic
    assert raw_rho > 0.9
    assert abs(fit_rho) < 0.2
    slopes = np.array(metadata["vsn_sample_slopes"])
    offsets = np.array(metadata["vsn_sample_offsets"])
    reconstructed = (
        np.sinh((output.to_numpy() - metadata["vsn_shift"]) * np.log(2))
        - offsets
    ) / slopes
    np.testing.assert_allclose(reconstructed, source, rtol=1e-12, atol=1e-12)
    assert slopes.max() / slopes.min() > 20
    assert metadata["vsn_scale"] == pytest.approx(
        np.exp(np.mean(np.log(slopes)))
    )
    assert metadata["vsn_converged"]
    assert len(metadata["vsn_optimizer"]) == 7
    json.dumps(metadata, allow_nan=False)


def test_vsn_retains_nan_mask_and_all_missing_rows():
    """Excluded missing cells are preserved at their original identities."""
    source = _model_data()
    source.iloc[::11, 2] = np.nan
    source.iloc[0, :] = np.nan
    output, metadata = DataNormalizer.calc_vsn_normalization(source)
    np.testing.assert_array_equal(output.isna(), source.isna())
    assert np.isfinite(output.to_numpy()[source.notna().to_numpy()]).all()
    assert metadata["vsn_fit_feature_count"] < source.shape[0]


def test_vsn_reports_finite_nonconverged_fit(monkeypatch):
    """A solver iteration limit remains visible in the warning and audit."""
    from pimqc.processing.normalization import vsn

    original_minimize = vsn.minimize

    def limited_minimize(*args, **kwargs):
        kwargs["options"]["maxiter"] = 1
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr(vsn, "minimize", limited_minimize)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        output, metadata = DataNormalizer.calc_vsn_normalization(_model_data())
    assert np.isfinite(output.to_numpy()).all()
    assert not metadata["vsn_converged"]
    assert any(item["status"] != 0 for item in metadata["vsn_optimizer"])


def test_vsn_parameters_survive_native_stage_audit(tmp_path):
    """The native stage keeps glog values and aligned sample coefficients."""
    from pimqc import MetaboDatasetBuilder
    from pimqc.serialization import read_audit_payload, write_audit_payload

    raw = _model_data()
    sample_metadata = pd.DataFrame(
        {
            "Sample Name": raw.columns,
            "Sample Type": ["QC"] * 4 + ["Sample"] * 8,
            "Batch": ["B1"] * 12,
            "Inject Order": range(12),
        }
    )
    dataset = MetaboDatasetBuilder(sample_metadata, raw).run_build().data
    engine = DataNormalizer(dataset, norm_method="VSN")
    source = engine._extract_ordered_target_matrix()
    expected, coefficients = DataNormalizer.calc_vsn_normalization(source)
    result = engine.run_normalization()
    np.testing.assert_allclose(result.data.intensity, expected)
    assert result.data.context.is_logged
    assert result.data.context.log_base == "2"
    assert result.audit.metrics["vsn_parameters"] == coefficients
    audit_path = tmp_path / "native-vsn-audit"
    write_audit_payload(result.audit, audit_path)
    restored = read_audit_payload(audit_path)
    assert restored.metrics["vsn_parameters"] == coefficients


@pytest.mark.parametrize(
    "values, message",
    [
        (np.ones((10, 1)), "two sample"),
        (np.ones((10, 3)), "constant"),
        (np.full((10, 3), np.nan), "three observations"),
        (np.full((10, 3), np.inf), "infinite"),
    ],
)
def test_vsn_rejects_unidentifiable_input(values, message):
    """Invalid data fail explicitly instead of emitting arbitrary glog data."""
    with pytest.raises(ValueError, match=message):
        DataNormalizer.calc_vsn_normalization(pd.DataFrame(values))
