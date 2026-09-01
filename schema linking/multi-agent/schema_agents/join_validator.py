from __future__ import annotations

from .models import DatabaseSchema


def declared_fk(
    schema: DatabaseSchema,
    left: str,
    right: str,
) -> tuple[str, str] | None:
    """Return the Spider-declared orientation for a proposed FK pair."""
    for declared_left, declared_right in schema.foreign_keys:
        if (left, right) == (declared_left, declared_right):
            return declared_left, declared_right
        if (left, right) == (declared_right, declared_left):
            return declared_left, declared_right
    return None
