from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from .models import DatabaseSchema, LinkingResult
from .sql_agents import safe_select_error


@dataclass
class RecallScore:
    table_recall: float
    column_recall: float | None
    table_precision: float
    column_precision: float
    strict_table_recall: bool
    strict_column_recall: bool
    gold_tables: set[str]
    gold_columns: set[str]


@dataclass
class SQLQualityScore:
    normalized_exact_match: bool
    execution_match: bool | None
    predicted_execution_success: bool
    gold_execution_success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_exact_match": self.normalized_exact_match,
            "execution_match": self.execution_match,
            "predicted_execution_success": self.predicted_execution_success,
            "gold_execution_success": self.gold_execution_success,
            "error": self.error,
        }


@dataclass
class _QueryRows:
    success: bool
    rows: list[tuple[Any, ...]]
    error: str | None = None
    truncated: bool = False


def evaluate(result: LinkingResult, schema: DatabaseSchema, sql: str) -> RecallScore:
    gold_tables, gold_columns = extract_gold_links(schema, sql)
    predicted_tables = set(result.tables)
    predicted_columns = set(result.columns)
    return RecallScore(
        table_recall=_recall(predicted_tables, gold_tables) or 0.0,
        column_recall=_recall(predicted_columns, gold_columns),
        table_precision=_precision(predicted_tables, gold_tables),
        column_precision=_precision(predicted_columns, gold_columns),
        strict_table_recall=gold_tables <= predicted_tables,
        strict_column_recall=gold_columns <= predicted_columns,
        gold_tables=gold_tables,
        gold_columns=gold_columns,
    )


def evaluate_generated_sql(
    predicted_sql: str,
    gold_sql: str,
    database_path: Path,
    max_rows: int = 10_000,
) -> SQLQualityScore:
    predicted = _execute_all(predicted_sql, database_path, max_rows)
    gold = _execute_all(gold_sql, database_path, max_rows)
    exact = normalize_sql(predicted_sql) == normalize_sql(gold_sql)

    errors: list[str] = []
    if predicted.error:
        errors.append(f"prediction: {predicted.error}")
    if gold.error:
        errors.append(f"gold: {gold.error}")
    if predicted.truncated or gold.truncated:
        errors.append(f"row limit exceeded ({max_rows})")

    execution_match: bool | None
    if not gold.success or predicted.truncated or gold.truncated:
        execution_match = None
    elif not predicted.success:
        execution_match = False
    else:
        ordered = re.search(r"\bORDER\s+BY\b", gold_sql, flags=re.I) is not None
        execution_match = _same_rows(predicted.rows, gold.rows, ordered)

    return SQLQualityScore(
        normalized_exact_match=exact,
        execution_match=execution_match,
        predicted_execution_success=predicted.success,
        gold_execution_success=gold.success,
        error="; ".join(errors) or None,
    )


def normalize_sql(sql: str) -> str:
    normalized = sql.strip().rstrip(";")
    normalized = re.sub(r'"([A-Za-z_][A-Za-z0-9_ ]*)"', r"\1", normalized)
    normalized = re.sub(r"`([A-Za-z_][A-Za-z0-9_ ]*)`", r"\1", normalized)
    normalized = re.sub(r"\[([A-Za-z_][A-Za-z0-9_ ]*)\]", r"\1", normalized)
    normalized = normalized.casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*([(),=<>+*/-])\s*", r"\1", normalized)
    return normalized.strip()


def extract_gold_links(
    schema: DatabaseSchema, sql: str
) -> tuple[set[str], set[str]]:
    normalized = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql.lower())
    gold_tables = {
        table for table in schema.tables if _contains_identifier(normalized, table.lower())
    }
    gold_columns: set[str] = set()

    for column in schema.columns:
        qualified = f"{column.table}.{column.name}".lower()
        if _contains_identifier(normalized, qualified):
            gold_columns.add(column.key)

    by_name: dict[str, list[str]] = {}
    for column in schema.columns:
        by_name.setdefault(column.name.lower(), []).append(column.key)
    for name, keys in by_name.items():
        if not _contains_identifier(normalized, name):
            continue
        scoped = [key for key in keys if key.split(".", 1)[0] in gold_tables]
        if len(scoped) == 1:
            gold_columns.add(scoped[0])
        elif len(keys) == 1:
            gold_columns.add(keys[0])
    return gold_tables, gold_columns


def _execute_all(sql: str, database_path: Path, max_rows: int) -> _QueryRows:
    safety_error = safe_select_error(sql)
    if safety_error:
        return _QueryRows(False, [], safety_error)
    if not database_path.is_file():
        return _QueryRows(False, [], f"database not found: {database_path}")

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{database_path.resolve()}?mode=ro", uri=True, timeout=5.0
        )
        connection.execute("PRAGMA query_only = ON")
        calls = 0

        def guard() -> int:
            nonlocal calls
            calls += 1
            return int(calls > 20_000)

        connection.set_progress_handler(guard, 1_000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return _QueryRows(True, rows[:max_rows], truncated=True)
        return _QueryRows(True, rows)
    except sqlite3.Error as error:
        return _QueryRows(False, [], str(error))
    finally:
        if connection is not None:
            connection.close()


def _same_rows(
    predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]], ordered: bool
) -> bool:
    predicted_rows = [_canonical_row(row) for row in predicted]
    gold_rows = [_canonical_row(row) for row in gold]
    if ordered:
        return predicted_rows == gold_rows
    return Counter(predicted_rows) == Counter(gold_rows)


def _canonical_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(round(value, 8) if isinstance(value, float) else value for value in row)


def _contains_identifier(text: str, identifier: str) -> bool:
    escaped = re.escape(identifier).replace(r"\ ", r"[ _]")
    return re.search(rf"(?<![0-9a-z_]){escaped}(?![0-9a-z_])", text) is not None


def _recall(predicted: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(predicted & gold) / len(gold)


def _precision(predicted: set[str], gold: set[str]) -> float:
    if not predicted:
        return 1.0 if not gold else 0.0
    return len(predicted & gold) / len(predicted)
