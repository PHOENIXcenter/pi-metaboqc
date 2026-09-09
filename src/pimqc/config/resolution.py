"""Shared resolution of stage defaults, pipeline settings, and overrides.

resolve_stage_config applies a single precedence rule across stages: built-in
defaults are extended by the relevant TOML section and then by explicit,
non-None runtime overrides. The helper prevents configuration behavior from
drifting between filtering, correction, imputation, and normalization.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from .schema import PipelineConfig


_PIPELINE_SECTION_NAMES = frozenset(PipelineConfig.model_fields)


def resolve_stage_config(
    pipeline_params: Mapping[str, object] | None,
    section_name: str,
    defaults: Mapping[str, object],
    explicit_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve one stage's settings without mutating caller-owned mappings.

    Precedence is stable across every stage: class defaults, then the named
    pipeline section, then explicit non-``None`` constructor arguments.
    """

    resolved = dict(defaults)
    if pipeline_params is not None:
        if section_name in _PIPELINE_SECTION_NAMES:
            # Direct processor construction uses the same strict boundary as
            # file-based loading. Unknown sections and fields are rejected
            # before stage-specific defaults are consulted.
            try:
                normalized_params = PipelineConfig.model_validate(
                    pipeline_params
                ).model_dump()
            except ValidationError as exc:
                raise ValueError(
                    "Invalid pipeline configuration. Check section names and "
                    "fields against the current API."
                ) from exc
        else:
            normalized_params = pipeline_params

        section = normalized_params.get(section_name)
        if section is not None:
            if not isinstance(section, Mapping):
                raise TypeError(
                    f"Pipeline section '{section_name}' must be a mapping."
                )
            resolved.update(section)

    if explicit_overrides is not None:
        resolved.update(
            {
                key: value
                for key, value in explicit_overrides.items()
                if value is not None
            }
        )
    return validate_stage_values(section_name, resolved)


def validate_stage_values(
    section_name: str, values: Mapping[str, object]
) -> dict[str, object]:
    """Validate scientific settings while retaining dataset context fields."""
    section = PipelineConfig.model_fields.get(section_name)
    if section is None:
        return dict(values)
    model = section.annotation
    settings = {key: values[key] for key in model.model_fields if key in values}
    try:
        validated = model.model_validate(settings).model_dump()
        if "global_seed" in values:
            seed_model = PipelineConfig.model_fields["Dataset"].annotation
            validated["global_seed"] = seed_model.model_validate(
                {"global_seed": values["global_seed"]}
            ).global_seed
    except ValidationError as exc:
        raise ValueError(f"Invalid {section_name} settings: {exc}") from exc
    return {**values, **validated}


def validate_pipeline_params(
    pipeline_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate and normalize an in-memory pipeline configuration.

    File-based configuration already crosses the same Pydantic boundary in
    :func:`pimqc.io.load_pipeline_config`; this helper applies that boundary to
    ``MetaboDatasetBuilder`` and ``run_pipeline`` calls as well. Consequently,
    legacy section names and unknown fields fail loudly instead of being
    ignored and replaced by defaults.
    """

    if pipeline_params is None:
        return {}
    if not isinstance(pipeline_params, Mapping):
        raise TypeError("pipeline_params must be a mapping or None.")
    try:
        return PipelineConfig.model_validate(pipeline_params).model_dump()
    except ValidationError as exc:
        raise ValueError(
            "Invalid pipeline configuration. Check section names and fields "
            "against the current API."
        ) from exc


__all__ = ["resolve_stage_config", "validate_pipeline_params"]
