from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RelColumn:
    index: int
    table: str
    name: str
    column_type: str

    @property
    def key(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class RelationalSchema:
    """D with relational schema S^R_D (Spider / AI Hub tables.json format).

    `korean_tables`/`korean_columns` capture the dataset's own localized
    display names (Spider's table_names/column_names, which for the AI Hub
    NL2SQL annotations are genuine Korean labels) so they can seed class and
    datatype-property primary labels before any LLM enrichment runs.
    """

    db_id: str
    tables: list[str]
    columns: list[RelColumn]
    primary_keys: set[str]
    foreign_keys: list[tuple[str, str]]
    korean_tables: dict[str, str] = field(default_factory=dict)
    korean_columns: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_spider(cls, raw: dict[str, Any]) -> "RelationalSchema":
        tables = raw["table_names_original"]
        localized_tables = raw.get("table_names", tables)
        types = raw.get("column_types", [])
        localized_names = raw.get("column_names", [])
        columns: list[RelColumn] = []
        by_index: dict[int, str] = {}
        korean_columns: dict[str, str] = {}
        for index, (table_index, name) in enumerate(raw["column_names_original"]):
            if table_index < 0:
                continue
            column = RelColumn(
                index=index,
                table=tables[table_index],
                name=name,
                column_type=types[index] if index < len(types) else "unknown",
            )
            columns.append(column)
            by_index[index] = column.key
            if index < len(localized_names):
                localized = localized_names[index][1]
                if localized and localized != name:
                    korean_columns[column.key] = localized

        korean_tables = {
            table: localized
            for table, localized in zip(tables, localized_tables)
            if localized and localized != table
        }

        primary_keys = {
            by_index[index] for index in raw.get("primary_keys", []) if index in by_index
        }
        foreign_keys = [
            (by_index[left], by_index[right])
            for left, right in raw.get("foreign_keys", [])
            if left in by_index and right in by_index
        ]
        return cls(
            db_id=raw["db_id"],
            tables=tables,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            korean_tables=korean_tables,
            korean_columns=korean_columns,
        )

    def columns_for(self, table: str) -> list[RelColumn]:
        return [column for column in self.columns if column.table == table]

    def column(self, key: str) -> RelColumn | None:
        return next((column for column in self.columns if column.key == key), None)


def load_schemas(path: Path) -> dict[str, RelationalSchema]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {record["db_id"]: RelationalSchema.from_spider(record) for record in records}


def default_tables_path() -> Path:
    workspace = Path(__file__).resolve().parents[2]
    return workspace / "data" / "hugging face" / "Spider 1.0" / "dev" / "dev_table.json"


def default_database_root() -> Path:
    workspace = Path(__file__).resolve().parents[2]
    return workspace / "data" / "hugging face" / "Spider 1.0" / "dev" / "database"


def database_path(database_root: Path, db_id: str) -> Path:
    root = database_root.resolve()
    candidate = (root / db_id / f"{db_id}.sqlite").resolve()
    if root not in candidate.parents:
        raise ValueError(f"unsafe db_id path: {db_id!r}")
    return candidate
