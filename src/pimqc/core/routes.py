"""Operational imputation routes, not inferred missingness mechanisms.

R-route is the reconstruction route and S-route is the special-handling
route. Historical MAR/MNAR labels remain accepted solely as routing aliases.
No label establishes a statistical or physical cause of missingness.
"""

from __future__ import annotations

import pandas as pd

R_ROUTE = "R-route"
S_ROUTE = "S-route"
ROUTE_DEFINITIONS = {
    R_ROUTE: "reconstruction route",
    S_ROUTE: "special-handling route",
}


def normalize_route(value: object, *, strict: bool = True) -> object:
    """Resolve a route label, including historical tracking-table aliases.

    With ``strict=False`` unrecognized statuses such as INVALID are preserved,
    so standalone quality filtering need not manufacture a route assignment.
    """
    if isinstance(value, str):
        label = value.strip().casefold()
        for canonical, aliases in (
            (R_ROUTE, ("r-route", "mar")),
            (S_ROUTE, ("s-route", "mnar")),
        ):
            if label in aliases or any(
                label == f"{alias} ({suffix})"
                for alias in aliases
                for suffix in ("group", "qc", "group & qc")
            ):
                return canonical
    if strict:
        raise ValueError(
            "imputation_route/missingness_type must contain R-route or "
            "S-route (legacy MAR/MNAR aliases are accepted); "
            f"received {value!r}."
        )
    return value


def normalize_route_labels(
    labels: pd.Series, *, strict: bool = True
) -> pd.Series:
    """Normalize an indexed route series without changing feature identity."""
    return labels.map(lambda value: normalize_route(value, strict=strict))


def routes_from_metadata(
    metadata: pd.DataFrame,
    *,
    default: str | None = R_ROUTE,
    strict: bool = True,
) -> pd.Series:
    """Read canonical routes or legacy labels, rejecting conflicting columns.

    The default retains the historical reconstruction behavior for imputation
    without feature routing metadata. Quality-only actions pass
    ``default=None``.
    """
    canonical = metadata.get("imputation_route")
    legacy = metadata.get("missingness_type")
    labels = canonical if canonical is not None else legacy
    if labels is None:
        return pd.Series(default, index=metadata.index, dtype=object)
    routes = normalize_route_labels(labels, strict=strict)
    if canonical is not None and legacy is not None:
        old_routes = normalize_route_labels(legacy, strict=strict)
        comparable = routes.isin(ROUTE_DEFINITIONS) & old_routes.isin(
            ROUTE_DEFINITIONS
        )
        conflict = comparable & routes.ne(old_routes)
        if conflict.any():
            raise ValueError(
                "Conflicting imputation_route and missingness_type for "
                f"features: {metadata.index[conflict].tolist()}"
            )
    return routes


__all__ = [
    "R_ROUTE",
    "S_ROUTE",
    "ROUTE_DEFINITIONS",
    "normalize_route",
    "normalize_route_labels",
    "routes_from_metadata",
]
