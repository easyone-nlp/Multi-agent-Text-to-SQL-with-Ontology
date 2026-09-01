from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .relational import RelColumn, RelationalSchema

SPLIT_DIRS = {
    "Training": ("TS", "TL"),
    "Validation": ("VS", "VL"),
}

JOIN_KEY_PATTERN = re.compile(r"(_CD|_CODE|_ID|_NO|_KEY)$", re.IGNORECASE)


def default_aihub_root() -> Path:
    workspace = Path(__file__).resolve().parents[2]
    return workspace / "data" / "ai hub"


@dataclass
class AiHubDbSchema:
    """One AI Hub NL2SQL 'data' record: a single (usually one-table) DB."""

    db_id: str
    source: str
    annotation_file: str
    schema: RelationalSchema
    sqlite_path: Path | None


def load_aihub_schemas(data_root: Path, split: str = "Validation") -> list[AiHubDbSchema]:
    """Scan data_root/<split>/01.원천데이터/<VS|TS>/**/*_db_annotation.json

    and pair each schema record with its sqlite file, mirroring how
    schema linking/AutoLink/ai hub run/prepare_aihub.py locates the same
    files (see that script for the authoritative directory layout).
    """
    if split not in SPLIT_DIRS:
        raise ValueError(f"split must be one of {sorted(SPLIT_DIRS)}, got {split!r}")
    source_dir, _label_dir = SPLIT_DIRS[split]
    root = data_root / split / "01.원천데이터" / source_dir
    if not root.is_dir():
        raise FileNotFoundError(f"AI Hub source-data 디렉터리를 찾을 수 없습니다: {root}")

    annotation_paths = sorted(root.glob("**/*_db_annotation.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"{root} 아래에 *_db_annotation.json 파일이 없습니다.")
    sqlite_by_db_id = {path.stem: path for path in root.glob("**/*.sqlite")}

    results: list[AiHubDbSchema] = []
    for annotation_path in annotation_paths:
        with annotation_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("data", []):
            db_id = str(record["db_id"])
            results.append(
                AiHubDbSchema(
                    db_id=db_id,
                    source=str(record.get("source", "")),
                    annotation_file=annotation_path.name,
                    schema=RelationalSchema.from_spider(record),
                    sqlite_path=sqlite_by_db_id.get(db_id),
                )
            )
    return results


def group_by_source(schemas: list[AiHubDbSchema]) -> dict[str, list[AiHubDbSchema]]:
    groups: dict[str, list[AiHubDbSchema]] = defaultdict(list)
    for item in schemas:
        groups[item.source].append(item)
    return dict(groups)


def combine_source(
    source_name: str, items: list[AiHubDbSchema]
) -> tuple[RelationalSchema, dict[tuple[str, str], str]]:
    """Merge every table of every db_id under one `source` into a single

    RelationalSchema, since AI Hub represents each (usually single-table)
    dataset as its own isolated sqlite file even when several are
    thematically related (Section 5.4's "quick way" needs one D). Tables
    are renamed "<db_id>__<table>" to avoid collisions; the returned dict
    maps (db_id, original_table) -> renamed table so callers can also
    physically merge the underlying sqlite files with the same names.
    """
    tables: list[str] = []
    columns: list[RelColumn] = []
    primary_keys: set[str] = set()
    foreign_keys: list[tuple[str, str]] = []
    korean_tables: dict[str, str] = {}
    korean_columns: dict[str, str] = {}
    table_rename: dict[tuple[str, str], str] = {}
    index = 0

    for item in items:
        schema = item.schema
        local_rename = {table: f"{item.db_id}__{table}" for table in schema.tables}
        for original_table, new_table in local_rename.items():
            tables.append(new_table)
            korean_tables[new_table] = schema.korean_tables.get(original_table, original_table)
            table_rename[(item.db_id, original_table)] = new_table

        for column in schema.columns:
            new_table = local_rename[column.table]
            new_column = RelColumn(
                index=index, table=new_table, name=column.name, column_type=column.column_type
            )
            columns.append(new_column)
            index += 1
            korean = schema.korean_columns.get(column.key)
            if korean:
                korean_columns[new_column.key] = korean

        for key in schema.primary_keys:
            original_table, column_name = key.split(".", 1)
            primary_keys.add(f"{local_rename[original_table]}.{column_name}")

        for left, right in schema.foreign_keys:
            left_table, left_column = left.split(".", 1)
            right_table, right_column = right.split(".", 1)
            foreign_keys.append(
                (
                    f"{local_rename[left_table]}.{left_column}",
                    f"{local_rename[right_table]}.{right_column}",
                )
            )

    combined = RelationalSchema(
        db_id=_slug(source_name),
        tables=tables,
        columns=columns,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        korean_tables=korean_tables,
        korean_columns=korean_columns,
    )
    return combined, table_rename


def infer_join_columns(schema: RelationalSchema, min_tables: int = 2) -> set[tuple[str, str]]:
    """Heuristically find candidate join keys when no FK is declared

    (true for nearly all AI Hub records; Section 5.4 notes that without
    referential-integrity constraints, object properties must be defined
    some other way). Columns whose name looks like an entity code (_CD,
    _CODE, _ID, _NO, _KEY) and that recur, identically named, across at
    least `min_tables` different tables are treated as a shared join key;
    every pair of tables sharing it gets a candidate object property.
    """
    buckets: dict[str, dict[str, RelColumn]] = defaultdict(dict)
    for column in schema.columns:
        if not JOIN_KEY_PATTERN.search(column.name):
            continue
        buckets[column.name.upper()].setdefault(column.table, column)

    pairs: set[tuple[str, str]] = set()
    for candidates in buckets.values():
        if len(candidates) < min_tables:
            continue
        ordered = sorted(candidates.values(), key=lambda c: c.table)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                pairs.add((left.key, right.key))
    return pairs


def validate_join_overlap(
    sqlite_path: Path,
    pairs: set[tuple[str, str]],
    min_overlap: float = 0.05,
    sample_limit: int = 5000,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Keep only inferred joins whose columns actually share values.

    Same-named columns are not always the same code system: e.g. Korean
    public datasets reuse "ADMDONG_CD" for administrative codes at
    different granularities (2-digit sigungu vs. 5-digit dong vs. 10-digit
    legal-dong), so a name match alone can silently produce a join with
    zero matching rows. This queries the (already physically merged)
    sqlite file and drops any pair whose distinct-value overlap, relative
    to the smaller side, falls below `min_overlap`. Returns
    (validated_pairs, rejected_pairs) so callers can report what was cut.
    """
    if not pairs or not sqlite_path.is_file():
        return set(), set(pairs)

    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cache: dict[str, set[str]] = {}

    def values_of(key: str) -> set[str]:
        if key not in cache:
            table, column = key.split(".", 1)
            try:
                rows = connection.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" '
                    f'WHERE "{column}" IS NOT NULL LIMIT ?',
                    (sample_limit,),
                ).fetchall()
            except sqlite3.Error:
                cache[key] = set()
            else:
                cache[key] = {str(row[0]) for row in rows}
        return cache[key]

    validated: set[tuple[str, str]] = set()
    rejected: set[tuple[str, str]] = set()
    try:
        for left, right in pairs:
            left_values, right_values = values_of(left), values_of(right)
            smaller = min(len(left_values), len(right_values))
            overlap = len(left_values & right_values) / smaller if smaller else 0.0
            (validated if overlap >= min_overlap else rejected).add((left, right))
    finally:
        connection.close()
    return validated, rejected


def build_combined_sqlite(
    items: list[AiHubDbSchema],
    table_rename: dict[tuple[str, str], str],
    output_path: Path,
) -> list[str]:
    """Physically merge each db_id's own sqlite file into one database

    under the renamed tables, so the ontology's dictionary can sample real
    values and its synthesized views are actually executable -- AI Hub
    ships one sqlite per db_id, not one shared database per source.
    Returns the list of tables that could not be copied (missing/failed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    connection = sqlite3.connect(str(output_path))
    skipped: list[str] = []
    try:
        for index, item in enumerate(items):
            new_tables = [
                new_table
                for (db_id, _orig), new_table in table_rename.items()
                if db_id == item.db_id
            ]
            if item.sqlite_path is None or not item.sqlite_path.is_file():
                skipped.extend(new_tables)
                continue
            alias = f"src{index}"
            connection.execute("ATTACH DATABASE ? AS " + alias, (str(item.sqlite_path),))
            try:
                for original_table in item.schema.tables:
                    new_table = table_rename[(item.db_id, original_table)]
                    try:
                        connection.execute(
                            f'CREATE TABLE "{new_table}" AS '
                            f'SELECT * FROM {alias}."{original_table}"'
                        )
                    except sqlite3.Error:
                        skipped.append(new_table)
            finally:
                connection.execute(f"DETACH DATABASE {alias}")
        connection.commit()
    finally:
        connection.close()
    return skipped


def source_summary(schemas: list[AiHubDbSchema]) -> list[dict[str, Any]]:
    groups = group_by_source(schemas)
    summary = []
    for source, items in groups.items():
        summary.append(
            {
                "source": source,
                "db_count": len(items),
                "table_count": sum(len(item.schema.tables) for item in items),
                "annotation_files": sorted({item.annotation_file for item in items}),
            }
        )
    return sorted(summary, key=lambda entry: (entry["table_count"], entry["source"]))


def _slug(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_") or "source"
