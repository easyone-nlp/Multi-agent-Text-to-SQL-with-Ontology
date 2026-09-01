from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeClass:
    """A class c = (r_c, l_c, L_c): ranking, primary label, synonym labels."""

    name: str
    source_table: str
    ranking: float = 1.0
    primary_label: str = ""
    synonyms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.primary_label:
            self.primary_label = self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_table": self.source_table,
            "ranking": round(self.ranking, 4),
            "primary_label": self.primary_label,
            "synonyms": list(self.synonyms),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KnowledgeClass":
        return cls(
            name=str(raw["name"]),
            source_table=str(raw["source_table"]),
            ranking=float(raw.get("ranking", 1.0)),
            primary_label=str(raw.get("primary_label", "")),
            synonyms=[str(item) for item in raw.get("synonyms", [])],
        )


@dataclass
class DatatypeProperty:
    """A datatype property p = (r_p, l_p, L_p, c_p, T_p).

    c_p is the domain class name; T_p is the range of data types.
    `indexed` marks p as an indexed datatype property whose values
    participate in keyword matching (Section 3.1).
    """

    name: str
    domain: str
    source_table: str
    source_column: str
    range_types: list[str] = field(default_factory=list)
    ranking: float = 1.0
    primary_label: str = ""
    synonyms: list[str] = field(default_factory=list)
    indexed: bool = False
    value_synonyms: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.primary_label:
            self.primary_label = self.source_column

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "source_table": self.source_table,
            "source_column": self.source_column,
            "range_types": list(self.range_types),
            "ranking": round(self.ranking, 4),
            "primary_label": self.primary_label,
            "synonyms": list(self.synonyms),
            "indexed": self.indexed,
            "value_synonyms": {
                value: list(synonyms) for value, synonyms in self.value_synonyms.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatatypeProperty":
        return cls(
            name=str(raw["name"]),
            domain=str(raw["domain"]),
            source_table=str(raw["source_table"]),
            source_column=str(raw["source_column"]),
            range_types=[str(item) for item in raw.get("range_types", [])],
            ranking=float(raw.get("ranking", 1.0)),
            primary_label=str(raw.get("primary_label", "")),
            synonyms=[str(item) for item in raw.get("synonyms", [])],
            indexed=bool(raw.get("indexed", False)),
            value_synonyms={
                str(value): [str(synonym) for synonym in synonyms]
                for value, synonyms in raw.get("value_synonyms", {}).items()
            },
        )


@dataclass
class ObjectProperty:
    """An object property o = (r_o, l_o, L_o, c_o^1, c_o^2).

    `weight` is the traversal cost used by the Steiner-tree matching
    optimization (Section 3.2): lower weight => a path through this
    relationship is considered more relevant.
    """

    name: str
    domain: str
    range: str
    source_fk: tuple[str, str]
    weight: float = 1.0
    primary_label: str = ""
    synonyms: list[str] = field(default_factory=list)
    inferred: bool = False

    def __post_init__(self) -> None:
        if not self.primary_label:
            self.primary_label = f"{self.domain}_{self.range}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "range": self.range,
            "source_fk": list(self.source_fk),
            "weight": round(self.weight, 4),
            "primary_label": self.primary_label,
            "synonyms": list(self.synonyms),
            "inferred": self.inferred,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ObjectProperty":
        left, right = raw["source_fk"]
        return cls(
            name=str(raw["name"]),
            domain=str(raw["domain"]),
            range=str(raw["range"]),
            source_fk=(str(left), str(right)),
            weight=float(raw.get("weight", 1.0)),
            primary_label=str(raw.get("primary_label", "")),
            synonyms=[str(item) for item in raw.get("synonyms", [])],
            inferred=bool(raw.get("inferred", False)),
        )


@dataclass
class KnowledgeSchema:
    """A knowledge schema S^K = (C, P, O) over classes, datatype and object properties."""

    db_id: str
    classes: dict[str, KnowledgeClass] = field(default_factory=dict)
    datatype_properties: dict[str, DatatypeProperty] = field(default_factory=dict)
    object_properties: dict[str, ObjectProperty] = field(default_factory=dict)

    def add_class(self, knowledge_class: KnowledgeClass) -> None:
        self._assert_unique_name(knowledge_class.name)
        self.classes[knowledge_class.name] = knowledge_class

    def add_datatype_property(self, prop: DatatypeProperty) -> None:
        self._assert_unique_name(prop.name)
        self.datatype_properties[prop.name] = prop

    def add_object_property(self, prop: ObjectProperty) -> None:
        self._assert_unique_name(prop.name)
        self.object_properties[prop.name] = prop

    def _assert_unique_name(self, name: str) -> None:
        if (
            name in self.classes
            or name in self.datatype_properties
            or name in self.object_properties
        ):
            raise ValueError(f"knowledge schema element 이름 중복: {name!r}")

    def properties_of(self, class_name: str) -> list[DatatypeProperty]:
        return [
            prop
            for prop in self.datatype_properties.values()
            if prop.domain == class_name
        ]

    def object_properties_of(self, class_name: str) -> list[ObjectProperty]:
        return [
            prop
            for prop in self.object_properties.values()
            if prop.domain == class_name or prop.range == class_name
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_id": self.db_id,
            "classes": [c.to_dict() for c in self.classes.values()],
            "datatype_properties": [p.to_dict() for p in self.datatype_properties.values()],
            "object_properties": [o.to_dict() for o in self.object_properties.values()],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KnowledgeSchema":
        schema = cls(db_id=str(raw["db_id"]))
        for item in raw.get("classes", []):
            schema.add_class(KnowledgeClass.from_dict(item))
        for item in raw.get("datatype_properties", []):
            schema.add_datatype_property(DatatypeProperty.from_dict(item))
        for item in raw.get("object_properties", []):
            schema.add_object_property(ObjectProperty.from_dict(item))
        return schema
