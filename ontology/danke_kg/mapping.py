from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import DatatypeProperty, KnowledgeClass, KnowledgeSchema, ObjectProperty
from .relational import RelationalSchema

TEXTUAL_TYPES = {"text", "varchar", "char", "string", "others"}


@dataclass
class RDBMapping:
    """The mapping mu from a knowledge schema S^K to a relational schema S^R."""

    class_table: dict[str, str] = field(default_factory=dict)
    property_column: dict[str, tuple[str, str]] = field(default_factory=dict)
    object_property_fk: dict[str, tuple[str, str]] = field(default_factory=dict)

    def table_for(self, class_name: str) -> str:
        return self.class_table[class_name]

    def column_for(self, property_name: str) -> tuple[str, str]:
        return self.property_column[property_name]

    def fk_for(self, object_property_name: str) -> tuple[str, str]:
        return self.object_property_fk[object_property_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_table": dict(self.class_table),
            "property_column": {
                name: list(pair) for name, pair in self.property_column.items()
            },
            "object_property_fk": {
                name: list(pair) for name, pair in self.object_property_fk.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RDBMapping":
        return cls(
            class_table=dict(raw.get("class_table", {})),
            property_column={
                name: (pair[0], pair[1])
                for name, pair in raw.get("property_column", {}).items()
            },
            object_property_fk={
                name: (pair[0], pair[1])
                for name, pair in raw.get("object_property_fk", {}).items()
            },
        )


def is_likely_indexed(column_type: str, column_key: str, schema: RelationalSchema) -> bool:
    """Datatype properties whose values are worth keyword-matching (Section 3.1)."""
    if column_key in schema.primary_keys:
        return False
    if any(column_key in fk_pair for fk_pair in schema.foreign_keys):
        return False
    return column_type.lower() in TEXTUAL_TYPES


def direct_mapping(
    schema: RelationalSchema,
    inferred_fk_pairs: set[tuple[str, str]] | None = None,
) -> tuple[KnowledgeSchema, RDBMapping]:
    """Build the 'quick' direct-mapping baseline described in Section 5.4:

    tables -> classes, columns -> datatype properties, declared foreign
    keys -> object properties. Korean display names already present in the
    source schema (Spider's/AI Hub's table_names/column_names) seed the
    primary labels directly. This skeleton is later enriched by the
    multi-agent pipeline in orchestrator.py with synonyms, rankings, and
    indexed/value-synonym annotations.

    `inferred_fk_pairs` marks a subset of `schema.foreign_keys` as *not*
    declared referential-integrity constraints but heuristically inferred
    join keys (Section 5.4: "if the database does not have referential
    integrity constraints, the object properties should be defined
    directly") -- e.g. from shared code columns across AI Hub tables that
    never declare real foreign keys. Those get `ObjectProperty.inferred=True`
    and a higher default weight, since they are less certain than a
    declared constraint.
    """
    inferred_fk_pairs = inferred_fk_pairs or set()
    knowledge_schema = KnowledgeSchema(db_id=schema.db_id)
    mapping = RDBMapping()

    for table in schema.tables:
        class_name = _safe_name(table)
        knowledge_schema.add_class(
            KnowledgeClass(
                name=class_name,
                source_table=table,
                primary_label=schema.korean_tables.get(table, table),
            )
        )
        mapping.class_table[class_name] = table

    for column in schema.columns:
        class_name = _safe_name(column.table)
        prop_name = f"{class_name}_{_safe_name(column.name)}"
        indexed = is_likely_indexed(column.column_type, column.key, schema)
        knowledge_schema.add_datatype_property(
            DatatypeProperty(
                name=prop_name,
                domain=class_name,
                source_table=column.table,
                source_column=column.name,
                range_types=[column.column_type],
                primary_label=schema.korean_columns.get(column.key, column.name),
                indexed=indexed,
            )
        )
        mapping.property_column[prop_name] = (column.table, column.name)

    seen_pairs: set[tuple[str, str]] = set()
    for index, (left, right) in enumerate(schema.foreign_keys):
        if (left, right) in seen_pairs or (right, left) in seen_pairs:
            continue
        seen_pairs.add((left, right))
        left_table, right_table = left.split(".", 1)[0], right.split(".", 1)[0]
        left_class, right_class = _safe_name(left_table), _safe_name(right_table)
        if left_class not in knowledge_schema.classes or right_class not in knowledge_schema.classes:
            continue
        inferred = (left, right) in inferred_fk_pairs or (right, left) in inferred_fk_pairs
        op_name = f"{left_class}_{right_class}_fk{index}"
        knowledge_schema.add_object_property(
            ObjectProperty(
                name=op_name,
                domain=left_class,
                range=right_class,
                source_fk=(left, right),
                weight=1.5 if inferred else 1.0,
                inferred=inferred,
            )
        )
        mapping.object_property_fk[op_name] = (left, right)

    return knowledge_schema, mapping


def _safe_name(identifier: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in identifier).strip("_") or "x"
