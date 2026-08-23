"""Canonical, component-safe namespaces for isolating agent memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias


Namespace: TypeAlias = tuple[str, ...]
DEFAULT_NAMESPACE: Namespace = ("default",)


def normalize_namespace(
    namespace: str | Sequence[str] | None = None,
    *,
    user_id: str | None = None,
) -> Namespace:
    """Return a validated namespace, deriving a user scope when requested.

    Components remain separate instead of being joined with punctuation.  This
    avoids ambiguous prefix matching (for example ``team.a`` versus
    ``team``, ``a``) in persistent backends.
    """

    if namespace is None:
        raw = ("users", user_id) if user_id is not None else DEFAULT_NAMESPACE
    elif isinstance(namespace, str):
        raw = (namespace,)
    else:
        raw = tuple(namespace)

    if not raw:
        raise ValueError("namespace must contain at least one component")
    normalized: list[str] = []
    for component in raw:
        if not isinstance(component, str):
            raise TypeError("namespace components must be strings")
        value = component.strip()
        if not value:
            raise ValueError("namespace components must not be empty")
        normalized.append(value)
    return tuple(normalized)


def namespace_matches(
    candidate: Sequence[str] | None,
    namespace: str | Sequence[str] | None,
    *,
    include_descendants: bool = False,
) -> bool:
    """Compare namespace components exactly, or by an explicit prefix."""

    expected = normalize_namespace(namespace)
    actual = normalize_namespace(candidate)
    if include_descendants:
        return actual[: len(expected)] == expected
    return actual == expected


__all__ = [
    "DEFAULT_NAMESPACE",
    "Namespace",
    "namespace_matches",
    "normalize_namespace",
]
