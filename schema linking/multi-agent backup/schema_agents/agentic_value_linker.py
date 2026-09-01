from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .agentic_agents import (
    AgentResponse,
    StructuredQwenAgent,
    _dict_list,
    _json,
    _resolve_columns,
    _resolve_one_column,
    _resolve_tables,
    _string_list,
)
from .embedding_retriever import RetrievedSchema
from .models import DatabaseSchema


class ValueLinkerAgent(StructuredQwenAgent):
    name = "value_linker"

    def __init__(
        self,
        model,
        json_retries: int = 1,
        max_candidates: int = 4,
        max_domain_values: int = 20,
        progress_limit: int = 20_000,
    ) -> None:
        super().__init__(model, json_retries=json_retries)
        self.max_candidates = max_candidates
        self.max_domain_values = max_domain_values
        self.progress_limit = progress_limit

    def link(
        self,
        question: str,
        task: str,
        decomposition: dict[str, Any],
        schema_selection: dict[str, Any],
        retrieved: RetrievedSchema,
        schema: DatabaseSchema,
        database_path: Path,
    ) -> AgentResponse:
        proposal = self.ask(
            (
                "You are a value-linking specialist with a read-only SQLite probe tool. "
                "For every manager-decomposed filter, propose several semantically plausible "
                "columns and DB value variants. Do not merely repeat the current schema "
                "selection; include alternatives when entity ownership is ambiguous. "
                "Do not generate SQL."
            ),
            (
                f"Question: {question}\n"
                f"Manager task: {task}\n"
                f"Decomposition:\n{_json(decomposition)}\n"
                f"Current schema selection:\n{_json(schema_selection)}\n"
                f"Retrieved schema:\n{retrieved.render(schema)}\n"
                "Return exactly:\n"
                '{"conditions":[{"condition_id":"f1","span":"...",'
                '"operator":"=","candidate_columns":["table.column"],'
                '"candidate_values":["English_or_encoded_DB_value"],'
                '"probe_mode":"exact","value_origin":"explicit_or_translated_or_encoded",'
                '"reason":"..."}]}\n'
                "probe_mode is exact, contains, domain, range, or none. "
                "Use domain for categorical/code columns, range for numeric comparisons, "
                "none for column-only or subquery-derived predicates. COUNT/MIN/MAX/output "
                "requests are not filter values. Candidate values must be entailed by the "
                "condition span through literal, translation, transliteration, or a clearly "
                "marked encoded category. Give at most 4 candidate columns per condition."
            ),
        )
        probe_plan = validate_probe_plan(
            proposal.payload, schema, retrieved, self.max_candidates
        )
        evidence = self._execute_probes(
            probe_plan, database_path
        )
        resolution = self.ask(
            (
                "You are the value-linking specialist resolving DB probe evidence. "
                "Choose a filter column only when semantics and evidence support it. "
                "DB existence alone is not semantic proof. Preserve unresolved conditions "
                "instead of guessing."
            ),
            (
                f"Question: {question}\n"
                f"Decomposition:\n{_json(decomposition)}\n"
                f"Schema selection:\n{_json(schema_selection)}\n"
                f"Probe plan:\n{_json(probe_plan)}\n"
                f"Read-only DB evidence:\n{_json(evidence)}\n"
                "Return exactly:\n"
                '{"selected_tables":["table"],'
                '"selected_columns":["table.column"],'
                '"filters":[{"condition_id":"f1","span":"...",'
                '"column":"table.column","operator":"=","value":"stored value",'
                '"branch":"main","evidence":"exact DB match"}],'
                '"unresolved":[{"condition_id":"f2","reason":"..."}]}\n'
                "Use the actual matched stored value when available. A range probe validates "
                "the column/domain but does not replace the requested comparison boundary."
            ),
        )
        resolution.payload = validate_value_output(
            resolution.payload, schema
        )
        resolution.payload["probe_plan"] = probe_plan
        resolution.payload["evidence"] = evidence
        resolution.attempts += proposal.attempts
        return resolution

    def _execute_probes(
        self,
        conditions: list[dict[str, Any]],
        database_path: Path,
    ) -> list[dict[str, Any]]:
        if not database_path.is_file():
            return [{
                "status": "error",
                "error": f"database not found: {database_path}",
            }]
        connection: sqlite3.Connection | None = None
        evidence: list[dict[str, Any]] = []
        try:
            connection = sqlite3.connect(
                f"file:{database_path.resolve()}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            connection.execute("PRAGMA query_only = ON")
            calls = 0

            def guard() -> int:
                nonlocal calls
                calls += 1
                return int(calls > self.progress_limit)

            connection.set_progress_handler(guard, 1_000)
            for condition in conditions:
                for column in condition["candidate_columns"]:
                    try:
                        evidence.append(
                            _probe_one(
                                connection,
                                condition,
                                column,
                                self.max_domain_values,
                            )
                        )
                    except sqlite3.Error as error:
                        evidence.append({
                            "condition_id": condition["condition_id"],
                            "span": condition["span"],
                            "column": column,
                            "probe_mode": condition["probe_mode"],
                            "status": "error",
                            "error": str(error),
                        })
        except sqlite3.Error as error:
            evidence.append({"status": "error", "error": str(error)})
        finally:
            if connection is not None:
                connection.close()
        return evidence


def validate_probe_plan(
    payload: dict[str, Any],
    schema: DatabaseSchema,
    retrieved: RetrievedSchema,
    max_candidates: int,
) -> list[dict[str, Any]]:
    allowed_columns = set(retrieved.columns)
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(_dict_list(payload.get("conditions", []))[:16]):
        columns = [
            column
            for column in _resolve_columns(
                raw.get("candidate_columns", []), schema
            )
            if column in allowed_columns
        ][:max_candidates]
        if not columns:
            continue
        mode = str(raw.get("probe_mode", "exact")).strip().casefold()
        if mode not in {"exact", "contains", "domain", "range", "none"}:
            mode = "exact"
        values = raw.get("candidate_values", [])
        if not isinstance(values, list):
            values = []
        values = [
            value[:200] if isinstance(value, str) else value
            for value in values[:8]
            if value is None or isinstance(value, (str, int, float))
        ]
        result.append({
            "condition_id": str(raw.get("condition_id", f"f{index + 1}")),
            "span": str(raw.get("span", ""))[:200],
            "operator": _operator(raw.get("operator")),
            "candidate_columns": columns,
            "candidate_values": values,
            "probe_mode": mode,
            "value_origin": str(raw.get("value_origin", ""))[:80],
            "reason": str(raw.get("reason", ""))[:300],
        })
    return result


def validate_value_output(
    payload: dict[str, Any],
    schema: DatabaseSchema,
) -> dict[str, Any]:
    tables = _resolve_tables(payload.get("selected_tables", []), schema)
    columns = _resolve_columns(payload.get("selected_columns", []), schema)
    filters: list[dict[str, Any]] = []
    for raw in _dict_list(payload.get("filters", [])):
        column = _resolve_one_column(raw.get("column"), schema)
        if column is None:
            continue
        item = {
            "condition_id": str(raw.get("condition_id", "")),
            "span": str(raw.get("span", "")),
            "column": column,
            "operator": _operator(raw.get("operator")),
            "value": raw.get("value"),
            "branch": str(raw.get("branch", "main")),
            "evidence": str(raw.get("evidence", "")),
        }
        filters.append(item)
        if column not in columns:
            columns.append(column)
        table = column.split(".", 1)[0]
        if table not in tables:
            tables.append(table)
    return {
        "selected_tables": tables,
        "selected_columns": columns,
        "filters": filters,
        "unresolved": _dict_list(payload.get("unresolved", [])),
    }


def _probe_one(
    connection: sqlite3.Connection,
    condition: dict[str, Any],
    column_key: str,
    domain_limit: int,
) -> dict[str, Any]:
    table, column = column_key.split(".", 1)
    mode = condition["probe_mode"]
    values = condition["candidate_values"]
    base = {
        "condition_id": condition["condition_id"],
        "span": condition["span"],
        "column": column_key,
        "probe_mode": mode,
        "candidate_values": values,
    }
    if mode == "none":
        return {**base, "status": "not_applicable"}
    if mode == "domain":
        sql = (
            f"SELECT {_quote(column)}, COUNT(*) AS frequency "
            f"FROM {_quote(table)} WHERE {_quote(column)} IS NOT NULL "
            f"GROUP BY {_quote(column)} ORDER BY frequency DESC LIMIT ?"
        )
        rows = connection.execute(sql, (domain_limit + 1,)).fetchall()
        observed = [row[0] for row in rows[:domain_limit]]
        matched = _match_values(values, observed)
        return {
            **base,
            "status": "matched" if matched else "observed",
            "matched_values": matched,
            "observed_values": observed,
            "domain_truncated": len(rows) > domain_limit,
        }
    if mode == "range":
        sql = (
            f"SELECT MIN(CAST({_quote(column)} AS REAL)), "
            f"MAX(CAST({_quote(column)} AS REAL)), COUNT(*) "
            f"FROM {_quote(table)} WHERE {_quote(column)} IS NOT NULL"
        )
        minimum, maximum, count = connection.execute(sql).fetchone()
        return {
            **base,
            "status": "observed",
            "minimum": minimum,
            "maximum": maximum,
            "count": count,
        }
    if not values:
        return {**base, "status": "no_values"}

    placeholders = ", ".join("?" for _ in values)
    exact_sql = (
        f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
        f"WHERE {_quote(column)} IN ({placeholders}) LIMIT 8"
    )
    matched = [
        row[0] for row in connection.execute(exact_sql, values).fetchall()
    ]
    if not matched:
        strings = [
            str(value).casefold() for value in values if value is not None
        ]
        if strings:
            placeholders = ", ".join("?" for _ in strings)
            casefold_sql = (
                f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
                f"WHERE lower(CAST({_quote(column)} AS TEXT)) "
                f"IN ({placeholders}) LIMIT 8"
            )
            matched = [
                row[0]
                for row in connection.execute(
                    casefold_sql, strings
                ).fetchall()
            ]
    if not matched and mode == "contains":
        strings = [
            str(value).casefold().strip("%")
            for value in values
            if value is not None and str(value).strip("%")
        ]
        if strings:
            predicates = " OR ".join(
                f"instr(lower(CAST({_quote(column)} AS TEXT)), ?) > 0"
                for _ in strings
            )
            contains_sql = (
                f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
                f"WHERE {predicates} LIMIT 8"
            )
            matched = [
                row[0]
                for row in connection.execute(
                    contains_sql, strings
                ).fetchall()
            ]
    return {
        **base,
        "status": "matched" if matched else "not_found",
        "matched_values": matched,
    }


def _match_values(candidates: list[Any], observed: list[Any]) -> list[Any]:
    keys = {
        str(value).strip().casefold()
        for value in candidates
        if value is not None
    }
    return [
        value
        for value in observed
        if str(value).strip().casefold() in keys
    ]


def _operator(value: Any) -> str:
    candidate = str(value or "=").strip().upper()
    allowed = {
        "=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "IN", "NOT IN",
        "IS", "IS NOT",
    }
    return candidate if candidate in allowed else "="


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
