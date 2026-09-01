from __future__ import annotations

import re
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_client import ChatModel, ModelClientError
from .models import DatabaseSchema, LinkingResult


@dataclass
class SQLAgentEvent:
    agent: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "status": self.status, **self.details}


@dataclass
class ExecutionResult:
    success: bool
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "columns": self.columns,
            "rows": self.rows,
            "error": self.error,
        }


@dataclass
class SQLGenerationResult:
    sql: str
    success: bool
    attempts: int
    execution: ExecutionResult
    trace: list[SQLAgentEvent]

    def to_dict(self, include_trace: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sql": self.sql,
            "success": self.success,
            "attempts": self.attempts,
            "execution": self.execution.to_dict(),
        }
        if include_trace:
            result["agent_trace"] = [event.to_dict() for event in self.trace]
        return result


class SQLDrafterAgent:
    name = "sql_drafter"

    def __init__(self, model: ChatModel | None) -> None:
        self.model = model

    def draft(
        self, question: str, schema: DatabaseSchema, linking: LinkingResult
    ) -> tuple[str, SQLAgentEvent]:
        if self.model is None:
            raise ModelClientError("SQL generation requires a Qwen chat model")
        response = self.model.complete(
            "You are the final SQL generation specialist. Generate one SQLite "
            "SELECT query from the manager-approved schema package. Output SQL only.",
            generation_prompt(question, schema, linking),
        )
        sql = extract_sql(response)
        return sql, SQLAgentEvent(
            self.name,
            "ok",
            {"mode": "model", "model": self.model.model, "sql": sql},
        )


class SQLSafetyValidator:
    name = "sql_safety_validator"

    def review(
        self, sql: str, schema: DatabaseSchema, linking: LinkingResult
    ) -> SQLAgentEvent:
        issues: list[str] = []
        safety_error = safe_select_error(sql)
        if safety_error:
            issues.append(safety_error)
        lower_sql = sql.lower()
        known_tables = {table.lower() for table in schema.tables}
        mentioned_tables = {
            table
            for table in known_tables
            if re.search(
                rf"(?<![0-9a-z_]){re.escape(table)}(?![0-9a-z_])", lower_sql
            )
        }
        if not mentioned_tables:
            issues.append("SQL에 알려진 테이블이 없음")
        selected = {table.lower() for table in linking.tables}
        outside = mentioned_tables - selected
        if outside:
            issues.append(f"schema linker 후보 밖 테이블 사용: {sorted(outside)}")
        return SQLAgentEvent(
            self.name,
            "ok" if not issues else "warning",
            {"valid": not issues, "issues": issues},
        )


class SQLiteExecutorAgent:
    name = "sqlite_executor"

    def __init__(self, preview_rows: int = 5, progress_limit: int = 20_000) -> None:
        self.preview_rows = preview_rows
        self.progress_limit = progress_limit

    def execute(self, sql: str, database_path: Path) -> ExecutionResult:
        safety_error = safe_select_error(sql)
        if safety_error:
            return ExecutionResult(False, error=safety_error)
        if not database_path.is_file():
            return ExecutionResult(False, error=f"database not found: {database_path}")

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
                return int(calls > self.progress_limit)

            connection.set_progress_handler(guard, 1_000)
            cursor = connection.execute(sql)
            columns = [description[0] for description in cursor.description or []]
            rows = [list(row) for row in cursor.fetchmany(self.preview_rows)]
            return ExecutionResult(True, columns=columns, rows=rows)
        except sqlite3.Error as error:
            return ExecutionResult(False, error=str(error))
        finally:
            if connection is not None:
                connection.close()


class SQLRepairAgent:
    name = "sql_repair"

    def __init__(self, model: ChatModel | None) -> None:
        self.model = model

    def repair(
        self,
        question: str,
        schema: DatabaseSchema,
        linking: LinkingResult,
        sql: str,
        error: str,
    ) -> tuple[str, SQLAgentEvent]:
        if self.model is None:
            raise ModelClientError("SQL repair requires a Qwen chat model")
        prompt = (
            f"{generation_prompt(question, schema, linking)}\n\n"
            f"Failed SQL:\n{sql}\n\nSQLite error:\n{error}\n\n"
            "Return corrected SQL only."
        )
        response = self.model.complete(
            "You are the same SQL generation specialist. Repair the SQLite "
            "SELECT query using the execution error. Output SQL only.",
            prompt,
        )
        repaired = extract_sql(response)
        return repaired, SQLAgentEvent(
            self.name,
            "ok",
            {"mode": "model", "model": self.model.model, "sql": repaired},
        )


def extract_sql(response: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", response, flags=re.I | re.S)
    candidate = fenced.group(1) if fenced else response
    match = re.search(r"\b(?:SELECT|WITH)\b.*", candidate.strip(), flags=re.I | re.S)
    if not match:
        raise ValueError("model response에 SELECT/WITH SQL이 없음")
    sql = match.group(0).strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    error = safe_select_error(sql)
    if error:
        raise ValueError(error)
    return sql


def safe_select_error(sql: str) -> str | None:
    stripped = sql.strip()
    if not stripped:
        return "빈 SQL"
    without_trailing = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in without_trailing:
        return "여러 SQL statement는 허용하지 않음"
    if not re.match(r"^(SELECT|WITH)\b", without_trailing, flags=re.I):
        return "SELECT 또는 WITH query만 허용"
    banned = re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REINDEX|VACUUM)\b",
        without_trailing,
        flags=re.I,
    )
    if banned:
        return f"금지된 SQL keyword: {banned.group(1).upper()}"
    return None


def generation_prompt(
    question: str, schema: DatabaseSchema, linking: LinkingResult
) -> str:
    selected_tables = set(linking.tables)
    selected_columns = set(linking.columns)
    foreign_keys = [
        (left, right)
        for left, right in schema.foreign_keys
        if left.split(".", 1)[0] in selected_tables
        and right.split(".", 1)[0] in selected_tables
        and left in selected_columns
        and right in selected_columns
    ]

    annotations: dict[str, list[str]] = defaultdict(list)
    for key in schema.primary_keys:
        if key in selected_columns:
            annotations[key].append("PK")
    for left, right in foreign_keys:
        annotations[left].append(f"FK -> {right}")
        annotations[right].append(f"REFERENCED BY {left}")

    schema_lines: list[str] = []
    for table in linking.tables:
        rendered_columns: list[str] = []
        for column in schema.columns_for(table):
            if column.key not in selected_columns:
                continue
            roles = annotations.get(column.key, [])
            role_suffix = f" [{'; '.join(roles)}]" if roles else ""
            rendered_columns.append(
                f"{quote_identifier(column.name)} {column.column_type}{role_suffix}"
            )
        rendered = ", ".join(rendered_columns)
        schema_lines.append(
            f"TABLE {quote_identifier(table)} ({rendered or '<no selected columns>'})"
        )

    relationship_lines = [
        f"- {left} (FK) -> {right} (referenced key)"
        for left, right in foreign_keys
    ]
    value_lines: list[str] = []
    for evidence in linking.value_evidence:
        if evidence.matched_values:
            value_lines.append(
                f"- question span {evidence.mention!r} -> {evidence.column} "
                f"{evidence.operator} {evidence.matched_values!r} "
                f"(DB-verified {evidence.probe_mode})"
            )
        elif evidence.probe_mode == "categorical":
            value_lines.append(
                f"- question span {evidence.mention!r} -> {evidence.column}; "
                f"observed domain={evidence.observed_values!r}; "
                "choose a stored value only when its meaning is supported"
            )
    manager_package = json.dumps({
        "query_decomposition": linking.query_decomposition,
        "column_roles": linking.column_roles,
        "grounded_filters": linking.grounded_filters,
        "joins": linking.joins,
        "unresolved": linking.unresolved,
    }, ensure_ascii=False, indent=2)
    return (
        f"Korean question: {question}\n"
        f"Database id: {schema.db_id}\n"
        "Selected schema with Spider metadata:\n"
        + "\n".join(schema_lines)
        + "\n"
        + "Declared PK/FK relationships among selected columns:\n"
        + ("\n".join(relationship_lines) if relationship_lines else "- none")
        + "\n"
        + "DB value-grounding evidence:\n"
        + ("\n".join(value_lines) if value_lines else "- none")
        + "\n"
        + "Manager-approved structured package:\n"
        + manager_package
        + "\n"
        + "Rules: use SQLite syntax; use only the selected schema; "
        + "follow manager-approved column roles, grounded filters, and join ON pairs; "
        + "prefer DB-verified matched values for their linked filter columns; "
        + "do not treat unverified domain samples as the requested value; "
        + "do not assume every selected table must appear; produce one read-only query."
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
