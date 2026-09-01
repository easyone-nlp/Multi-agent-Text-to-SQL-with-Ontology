from __future__ import annotations

import difflib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import KnowledgeSchema

NO_MATCH = "noMatch"


@dataclass(frozen=True)
class DictionaryEntry:
    """One row of DANKE's dictionary (Section 3.1): a (keyword -> match) pair.

    entry_type is "class" | "property" | "value":
      - class/property entries associate a label or synonym with c_k = a class.
      - value entries associate a data value v (or its synonym) with c_p, the
        class of the indexed datatype property p that stores v.
    """

    entry_type: str
    key: str
    target_class: str
    target_property: str | None = None
    value: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_type": self.entry_type,
            "key": self.key,
            "target_class": self.target_class,
            "target_property": self.target_property,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DictionaryEntry":
        return cls(
            entry_type=str(raw["entry_type"]),
            key=str(raw["key"]),
            target_class=str(raw["target_class"]),
            target_property=raw.get("target_property"),
            value=raw.get("value"),
        )


@dataclass
class Dictionary:
    """DANKE's Storage Module dictionary: metadata entries + data entries."""

    entries: list[DictionaryEntry] = field(default_factory=list)
    _index: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def add(self, entry: DictionaryEntry) -> None:
        key = entry.key.casefold()
        if not key:
            return
        existing = self._index.setdefault(key, [])
        if any(self.entries[i] == entry for i in existing):
            return
        existing.append(len(self.entries))
        self.entries.append(entry)

    def lookup_exact(self, keyword: str) -> list[DictionaryEntry]:
        indices = self._index.get(keyword.casefold(), [])
        return [self.entries[i] for i in indices]

    def lookup_fuzzy(self, keyword: str, cutoff: float = 0.72, limit: int = 3) -> list[DictionaryEntry]:
        close = difflib.get_close_matches(
            keyword.casefold(), self._index.keys(), n=limit, cutoff=cutoff
        )
        results: list[DictionaryEntry] = []
        for key in close:
            results.extend(self.entries[i] for i in self._index[key])
        return results

    @property
    def metadata_entry_count(self) -> int:
        return sum(1 for entry in self.entries if entry.entry_type in {"class", "property"})

    @property
    def data_entry_count(self) -> int:
        return sum(1 for entry in self.entries if entry.entry_type == "value")

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Dictionary":
        dictionary = cls()
        for item in raw.get("entries", []):
            dictionary.add(DictionaryEntry.from_dict(item))
        return dictionary


def build_dictionary(
    schema: KnowledgeSchema,
    database_path: Path | None = None,
    max_domain_values: int = 20,
) -> Dictionary:
    """Populate metadata entries from S^K and, when a sqlite file is given,

    sample data entries for indexed datatype properties -- mirroring how
    DANKE's Preparation/Data-Extraction modules build the dictionary from
    the knowledge schema and the underlying database (Section 3.1).
    """
    dictionary = Dictionary()

    for knowledge_class in schema.classes.values():
        dictionary.add(
            DictionaryEntry("class", knowledge_class.primary_label, knowledge_class.name)
        )
        for synonym in knowledge_class.synonyms:
            dictionary.add(DictionaryEntry("class", synonym, knowledge_class.name))

    for prop in schema.datatype_properties.values():
        dictionary.add(
            DictionaryEntry("property", prop.primary_label, prop.domain, prop.name)
        )
        for synonym in prop.synonyms:
            dictionary.add(DictionaryEntry("property", synonym, prop.domain, prop.name))

    for prop in schema.datatype_properties.values():
        if not prop.indexed:
            continue
        for value, synonyms in prop.value_synonyms.items():
            dictionary.add(DictionaryEntry("value", value, prop.domain, prop.name, value))
            for synonym in synonyms:
                dictionary.add(DictionaryEntry("value", synonym, prop.domain, prop.name, value))

    if database_path is not None and database_path.is_file():
        _add_sampled_values(dictionary, schema, database_path, max_domain_values)

    return dictionary


def _add_sampled_values(
    dictionary: Dictionary,
    schema: KnowledgeSchema,
    database_path: Path,
    max_domain_values: int,
) -> None:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        for prop in schema.datatype_properties.values():
            if not prop.indexed or prop.value_synonyms:
                continue
            try:
                cursor.execute(
                    f'SELECT DISTINCT "{prop.source_column}" FROM "{prop.source_table}" '
                    f'WHERE "{prop.source_column}" IS NOT NULL LIMIT ?',
                    (max_domain_values,),
                )
                rows = cursor.fetchall()
            except sqlite3.Error:
                continue
            for (value,) in rows:
                text = str(value).strip()
                if text:
                    dictionary.add(DictionaryEntry("value", text, prop.domain, prop.name, text))
    finally:
        connection.close()


@dataclass
class MatchResult:
    keyword: str
    entries: list[DictionaryEntry]
    fuzzy: bool

    @property
    def matched(self) -> bool:
        return bool(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "matched": self.matched,
            "fuzzy": self.fuzzy,
            "matches": [entry.to_dict() for entry in self.entries] if self.matched else NO_MATCH,
        }


class MatchingDiscoveryService:
    """DANKE's Matching Discovery Service: K -> K_M = {(k, d_k)}.

    Exact match first; a fuzzy fallback approximates DANKE's engine-native
    fuzzy matching when no dictionary entry matches k exactly. Unmatched
    keywords resolve to d_k = "noMatch" (Section 3.2).
    """

    def __init__(self, dictionary: Dictionary, fuzzy_cutoff: float = 0.72) -> None:
        self.dictionary = dictionary
        self.fuzzy_cutoff = fuzzy_cutoff

    def match(self, keywords: list[str]) -> list[MatchResult]:
        results: list[MatchResult] = []
        for keyword in keywords:
            exact = self.dictionary.lookup_exact(keyword)
            if exact:
                results.append(MatchResult(keyword, exact, fuzzy=False))
                continue
            fuzzy = self.dictionary.lookup_fuzzy(keyword, cutoff=self.fuzzy_cutoff)
            results.append(MatchResult(keyword, fuzzy, fuzzy=True))
        return results

    def matched_classes(self, results: list[MatchResult]) -> list[str]:
        classes: list[str] = []
        for result in results:
            for entry in result.entries:
                if entry.target_class not in classes:
                    classes.append(entry.target_class)
        return classes
