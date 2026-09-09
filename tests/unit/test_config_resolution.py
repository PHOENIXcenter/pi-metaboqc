"""Tests for consistent stage configuration precedence and validation."""

import pytest

from pimqc.config import resolve_stage_config, validate_pipeline_params


def test_resolve_stage_config_applies_documented_precedence() -> None:
    """Apply explicit values over TOML values and defaults."""
    resolved = resolve_stage_config(
        pipeline_params={"Stage": {"shared": "toml", "from_toml": 2}},
        section_name="Stage",
        defaults={"shared": "default", "from_default": 1},
        explicit_overrides={"shared": "explicit", "ignored": None},
    )

    assert resolved == {
        "shared": "explicit",
        "from_default": 1,
        "from_toml": 2,
    }


@pytest.mark.parametrize(
    "parameters",
    [
        {"ObsoleteFilter": {"mv_qc_tol": 0.1}},
        {"FeatureFilter": {"mv_qc_tol": 0.1, "misspelled": 0.2}},
    ],
)
def test_pipeline_validation_rejects_legacy_sections_and_unknown_fields(
    parameters: dict[str, object],
) -> None:
    """Never silently replace invalid configuration with stage defaults."""
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        validate_pipeline_params(parameters)


def test_direct_stage_resolution_rejects_unknown_pipeline_section() -> None:
    """Reject unknown sections when a processor bypasses file loading."""
    with pytest.raises(ValueError, match="Invalid pipeline configuration"):
        resolve_stage_config(
            {"ObsoleteFilter": {}},
            "FeatureFilter",
            {"mv_global_tol": 0.7},
        )


def test_dataset_configuration_can_disable_biological_groups() -> None:
    """Allow an explicit null group column for the documented fallback path."""
    resolved = validate_pipeline_params({"Dataset": {"bio_group": None}})

    assert resolved["Dataset"]["bio_group"] is None
