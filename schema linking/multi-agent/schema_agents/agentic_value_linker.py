from __future__ import annotations

import json
import re
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
                "columns and English DB value variants. The SQLite databases store values "
                "in English: translate or romanize Korean value mentions before probing, "
                "and never emit a Korean string as a candidate value. Do not merely repeat "
                "the current schema selection; include alternatives when entity ownership "
                "is ambiguous. Do not generate SQL."
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
                "condition span through an English translation, romanization, standard "
                "English name, or a clearly marked encoded category. Preserve language-neutral "
                "numbers and dates. Never copy a Korean string into candidate_values. For "
                "example, 프랑스 -> France and 볼보 -> Volvo, while 공식 여부 may map to T/F. "
                "Give at most 4 candidate columns per condition."
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
                "For exact, contains, and domain probes, output a filter only when the "
                "evidence status is matched, and copy a value from matched_values exactly. "
                "Never describe not_found, observed, no_values, or error as an exact DB match. "
                "A range probe validates the column/domain but does not replace the requested "
                "numeric comparison boundary. Preserve unsupported conditions as unresolved."
            ),
        )
        resolution.payload = validate_value_output(
            resolution.payload,
            schema,
            probe_plan,
            evidence,
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
            if (
                value is None
                or isinstance(value, (int, float))
                or (isinstance(value, str) and not _contains_hangul(value))
            )
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
    probe_plan: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ground model filters in read-only DB evidence before accepting them."""
    plan_by_id = {
        str(item.get("condition_id", "")): item for item in probe_plan
    }
    evidence_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in evidence:
        condition_id = str(row.get("condition_id", ""))
        column = _resolve_one_column(row.get("column"), schema)
        if condition_id and column is not None:
            evidence_by_key.setdefault((condition_id, column), []).append(row)

    filters: list[dict[str, Any]] = []
    columns: list[str] = []
    tables: list[str] = []
    unresolved = _dict_list(payload.get("unresolved", []))
    resolved_ids: set[str] = set()
    for raw in _dict_list(payload.get("filters", [])):
        condition_id = str(raw.get("condition_id", ""))
        column = _resolve_one_column(raw.get("column"), schema)
        plan = plan_by_id.get(condition_id)
        if column is None or plan is None:
            unresolved.append({
                "condition_id": condition_id,
                "reason": "filter rejected: no validated probe plan/column",
            })
            continue
        if column not in plan.get("candidate_columns", []):
            unresolved.append({
                "condition_id": condition_id,
                "reason": f"filter rejected: {column} was not probed",
            })
            continue

        mode = str(plan.get("probe_mode", "exact"))
        rows = evidence_by_key.get((condition_id, column), [])
        operator = _operator(raw.get("operator"))
        value = raw.get("value")
        evidence_label = ""
        if mode in {"exact", "contains", "domain"}:
            matched_values = _unique_values([
                matched
                for row in rows
                if row.get("status") == "matched"
                for matched in row.get("matched_values", [])
            ])
            value = _validated_stored_value(value, matched_values, operator)
            if value is _NO_MATCH:
                statuses = sorted({str(row.get("status")) for row in rows})
                unresolved.append({
                    "condition_id": condition_id,
                    "reason": (
                        f"filter rejected: DB probe did not validate a stored value "
                        f"for {column}; statuses={statuses or ['missing']}"
                    ),
                })
                continue
            evidence_label = "validated DB match"
        elif mode == "range":
            if not any(row.get("status") == "observed" for row in rows):
                unresolved.append({
                    "condition_id": condition_id,
                    "reason": f"range filter rejected: no observed domain for {column}",
                })
                continue
            if not _is_numeric_boundary(value):
                unresolved.append({
                    "condition_id": condition_id,
                    "reason": "range filter rejected: boundary is not numeric",
                })
                continue
            evidence_label = "validated numeric column domain"
        else:
            unresolved.append({
                "condition_id": condition_id,
                "reason": "filter unresolved: predicate requires SQL/subquery reasoning",
            })
            continue

        filters.append({
            "condition_id": condition_id,
            "span": str(raw.get("span", "")),
            "column": column,
            "operator": operator,
            "value": value,
            "branch": str(raw.get("branch", "main")),
            "evidence": evidence_label,
        })
        resolved_ids.add(condition_id)
        if column not in columns:
            columns.append(column)
        table = column.split(".", 1)[0]
        if table not in tables:
            tables.append(table)

    unresolved_ids = {
        str(item.get("condition_id", "")) for item in unresolved
    }
    for condition_id in plan_by_id:
        if condition_id not in resolved_ids and condition_id not in unresolved_ids:
            unresolved.append({
                "condition_id": condition_id,
                "reason": "filter unresolved: no evidence-grounded model decision",
            })
    return {
        "selected_tables": tables,
        "selected_columns": columns,
        "filters": filters,
        "unresolved": unresolved,
    }


_NO_MATCH = object()
_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힣]")


def _contains_hangul(value: str) -> bool:
    return bool(_HANGUL.search(value))


def _unique_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (type(value).__name__, str(value).strip().casefold())
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _validated_stored_value(
    proposed: Any,
    matched_values: list[Any],
    operator: str,
) -> Any:
    if not matched_values:
        return _NO_MATCH
    proposed_values = proposed if isinstance(proposed, list) else [proposed]
    selected: list[Any] = []
    for requested in proposed_values:
        requested_key = str(requested).strip().casefold()
        for stored in matched_values:
            if str(stored).strip().casefold() == requested_key:
                selected.append(stored)
                break
    if operator in {"IN", "NOT IN"}:
        if selected:
            return _unique_values(selected)
        return list(matched_values)
    if selected:
        return selected[0]
    if len(matched_values) == 1:
        return matched_values[0]
    return _NO_MATCH


def _is_numeric_boundary(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
        except ValueError:
            return False
        return True
    return False


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
