from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_client import ChatModel, ModelClientError
from .models import AgentProposal, DatabaseSchema, ValueEvidence
from .schema_llm_agent import _extract_json_object


@dataclass
class ValueCondition:
    mention: str
    operator: str
    values: list[Any]
    candidate_columns: list[str]
    probe_mode: str


class DBValueGroundingAgent:
    """Verifies model-proposed filter columns against a read-only SQLite DB."""

    name = "db_value_grounder"

    def __init__(
        self,
        model: ChatModel,
        exact_score: float = 6.0,
        categorical_score: float = 5.0,
        max_candidates_per_condition: int = 4,
        max_domain_values: int = 20,
        progress_limit: int = 20_000,
    ) -> None:
        self.model = model
        self.exact_score = exact_score
        self.categorical_score = categorical_score
        self.max_candidates_per_condition = max_candidates_per_condition
        self.max_domain_values = max_domain_values
        self.progress_limit = progress_limit

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
        database_path: Path | None,
    ) -> tuple[AgentProposal, list[ValueEvidence]]:
        if database_path is None:
            return AgentProposal(
                self.name, reasons=["SQLite 경로가 없어 value probe 생략"]
            ), []
        if not database_path.is_file():
            return AgentProposal(
                self.name,
                reasons=[f"SQLite DB가 없어 value probe 생략: {database_path}"],
            ), []

        try:
            response = self.model.complete(
                (
                    "You identify Korean Text-to-SQL filter values and candidate schema "
                    "columns for safe database verification. Do not write SQL. Return one "
                    "JSON object only."
                ),
                value_candidate_prompt(question, schema, previous),
            )
            conditions, ignored = parse_value_conditions(
                response, schema, self.max_candidates_per_condition
            )
        except (ModelClientError, ValueError, json.JSONDecodeError) as error:
            return AgentProposal(
                self.name,
                reasons=[f"value candidate 추출 실패; probe 생략: {error}"],
            ), []

        table_scores: dict[str, float] = {}
        column_scores: dict[str, float] = {}
        evidence: list[ValueEvidence] = []
        errors: list[str] = []
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
            for condition in conditions:
                for column_key in condition.candidate_columns:
                    try:
                        item = self._probe(connection, condition, column_key)
                    except sqlite3.Error as error:
                        errors.append(f"{column_key}: {error}")
                        continue
                    if item is None:
                        continue
                    evidence.append(item)
                    score = (
                        self.categorical_score * item.confidence
                        if condition.probe_mode == "categorical"
                        else self.exact_score * item.confidence
                    )
                    column_scores[column_key] = max(
                        column_scores.get(column_key, 0.0), score
                    )
                    table = column_key.split(".", 1)[0]
                    table_scores[table] = max(table_scores.get(table, 0.0), score)
        except sqlite3.Error as error:
            errors.append(str(error))
        finally:
            if connection is not None:
                connection.close()

        reasons = [
            f"model={self.model.model}; conditions={len(conditions)}; "
            f"verified_columns={len(column_scores)}"
        ]
        if ignored:
            reasons.append(f"유효하지 않아 무시한 후보: {', '.join(ignored[:8])}")
        for item in evidence[:12]:
            detail = item.matched_values or item.observed_values
            reasons.append(
                f"{item.mention!r} -> {item.column}; mode={item.probe_mode}; "
                f"evidence={detail[:8]}"
            )
        if errors:
            reasons.append(f"probe 오류: {'; '.join(errors[:4])}")
        if not evidence and not errors:
            reasons.append("후보 value를 실제 DB에서 확인하지 못함")
        return AgentProposal(
            self.name, table_scores, column_scores, reasons
        ), evidence

    def _probe(
        self,
        connection: sqlite3.Connection,
        condition: ValueCondition,
        column_key: str,
    ) -> ValueEvidence | None:
        table, column = column_key.split(".", 1)
        if condition.probe_mode == "categorical":
            observed = _domain_values(
                connection, table, column, self.max_domain_values + 1
            )
            if not observed or len(observed) > self.max_domain_values:
                return None
            matched = _matching_values(condition.values, observed)
            return ValueEvidence(
                mention=condition.mention,
                column=column_key,
                operator=condition.operator,
                probe_mode="categorical",
                candidate_values=condition.values,
                matched_values=matched,
                observed_values=observed,
                confidence=1.0 if matched else 0.75,
            )

        if not condition.values:
            return None
        exact = _exact_values(connection, table, column, condition.values)
        if exact:
            return ValueEvidence(
                mention=condition.mention,
                column=column_key,
                operator=condition.operator,
                probe_mode="exact",
                candidate_values=condition.values,
                matched_values=exact,
                confidence=1.0,
            )
        casefold = _casefold_values(
            connection, table, column, condition.values
        )
        if casefold:
            return ValueEvidence(
                mention=condition.mention,
                column=column_key,
                operator=condition.operator,
                probe_mode="casefold",
                candidate_values=condition.values,
                matched_values=casefold,
                confidence=0.9,
            )
        if condition.probe_mode == "contains":
            contained = _contains_values(
                connection, table, column, condition.values
            )
            if contained:
                return ValueEvidence(
                    mention=condition.mention,
                    column=column_key,
                    operator=condition.operator,
                    probe_mode="contains",
                    candidate_values=condition.values,
                    matched_values=contained,
                    confidence=0.8,
                )
        return None


def value_candidate_prompt(
    question: str,
    schema: DatabaseSchema,
    previous: list[AgentProposal],
) -> str:
    semantic_agents = {"llm_schema_scout", "llm_schema_critic"}
    current_tables = sorted({
        table
        for proposal in previous
        if proposal.agent in semantic_agents
        for table in proposal.table_scores
    })
    current_columns = sorted({
        column
        for proposal in previous
        if proposal.agent in semantic_agents
        for column in proposal.column_scores
    })
    lines = [
        f"Korean question: {question}",
        f"Database: {schema.db_id}",
        "Schema:",
    ]
    for table in schema.tables:
        rendered = ", ".join(
            f"{column.name}<{column.column_type}>"
            for column in schema.columns_for(table)
        )
        lines.append(f"- {table}({rendered})")
    lines.extend([
        f"Current semantic tables: {current_tables}",
        f"Current semantic columns: {current_columns}",
        "Return exactly this JSON shape:",
        '{"conditions":[{"mention":"question span","operator":"=",'
        '"values":["DB value candidate"],"candidate_columns":'
        '["table.column"],"probe_mode":"exact"}]}',
        "Rules:",
        "- Enumerate every value-bearing filter, including negated, nested, "
        "EXCEPT, INTERSECT, and UNION branches.",
        "- Use only candidate_columns that exist in the schema, at most 4 per condition.",
        "- values must contain likely DB representations, including English "
        "translations, transliterations, or encoded values such as T/F when justified.",
        "- Use probe_mode=exact for names, codes, dates, and numbers.",
        "- Use probe_mode=contains only when the question explicitly requests a substring.",
        "- Use probe_mode=categorical for implied categories or coded booleans; "
        "include the likely stored code in values when it can be inferred.",
        "- Do not include SELECT-only, JOIN-only, GROUP BY-only, or ORDER BY-only columns.",
        "- Do not write SQL and do not invent schema names.",
        "Examples:",
        "- 코드 'AKO'인 공항 -> mention=AKO, values=['AKO'], "
        "candidate_columns=['airports.AirportCode'], probe_mode=exact",
        "- 공식 언어 -> mention=공식 언어, values=['T'], "
        "candidate_columns=['countrylanguage.IsOfficial'], probe_mode=categorical",
        "- 볼보 모델 -> mention=볼보, values=['volvo'], "
        "candidate_columns=['car_names.Model'], probe_mode=exact",
    ])
    return "\n".join(lines)


def parse_value_conditions(
    response: str,
    schema: DatabaseSchema,
    max_candidates_per_condition: int = 4,
) -> tuple[list[ValueCondition], list[str]]:
    payload = _extract_json_object(response)
    raw_conditions = payload.get("conditions", [])
    if not isinstance(raw_conditions, list):
        raise ValueError("conditions는 JSON list여야 함")

    column_map = {column.key.casefold(): column.key for column in schema.columns}
    conditions: list[ValueCondition] = []
    ignored: list[str] = []
    for raw in raw_conditions[:12]:
        if not isinstance(raw, dict):
            continue
        raw_candidates = raw.get("candidate_columns", [])
        raw_values = raw.get("values", [])
        if not isinstance(raw_candidates, list) or not isinstance(raw_values, list):
            continue
        candidates: list[str] = []
        for item in raw_candidates:
            candidate = str(item).strip().strip('`"[]').strip()
            resolved = column_map.get(candidate.casefold())
            if resolved is None:
                ignored.append(candidate)
            elif resolved not in candidates:
                candidates.append(resolved)
            if len(candidates) >= max_candidates_per_condition:
                break
        if not candidates:
            continue
        mode = str(raw.get("probe_mode", "exact")).strip().casefold()
        if mode not in {"exact", "contains", "categorical"}:
            mode = "exact"
        values = [
            value[:200] if isinstance(value, str) else value
            for value in raw_values
            if _supported_value(value)
        ]
        operator = str(raw.get("operator", "=")).strip().upper() or "="
        allowed_operators = {"=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "IN", "NOT IN"}
        conditions.append(ValueCondition(
            mention=str(raw.get("mention", "")).strip()[:200],
            operator=operator if operator in allowed_operators else "=",
            values=values[:8],
            candidate_columns=candidates,
            probe_mode=mode,
        ))
    return conditions, ignored


def _supported_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float))


def _exact_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    values: list[Any],
) -> list[Any]:
    placeholders = ", ".join("?" for _ in values)
    sql = (
        f"SELECT DISTINCT {_quote_identifier(column)} "
        f"FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(column)} IN ({placeholders}) LIMIT 8"
    )
    return [row[0] for row in connection.execute(sql, values).fetchall()]


def _casefold_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    values: list[Any],
) -> list[Any]:
    strings = [str(value).casefold() for value in values if value is not None]
    if not strings:
        return []
    placeholders = ", ".join("?" for _ in strings)
    sql = (
        f"SELECT DISTINCT {_quote_identifier(column)} "
        f"FROM {_quote_identifier(table)} "
        f"WHERE lower(CAST({_quote_identifier(column)} AS TEXT)) "
        f"IN ({placeholders}) LIMIT 8"
    )
    return [row[0] for row in connection.execute(sql, strings).fetchall()]


def _contains_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    values: list[Any],
) -> list[Any]:
    strings = [
        str(value).casefold().strip("%")
        for value in values
        if value is not None
    ]
    strings = [value for value in strings if value]
    if not strings:
        return []
    predicates = " OR ".join(
        f"instr(lower(CAST({_quote_identifier(column)} AS TEXT)), ?) > 0"
        for _ in strings
    )
    sql = (
        f"SELECT DISTINCT {_quote_identifier(column)} "
        f"FROM {_quote_identifier(table)} WHERE {predicates} LIMIT 8"
    )
    return [row[0] for row in connection.execute(sql, strings).fetchall()]


def _domain_values(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    limit: int,
) -> list[Any]:
    sql = (
        f"SELECT {_quote_identifier(column)}, COUNT(*) AS frequency "
        f"FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(column)} IS NOT NULL "
        f"GROUP BY {_quote_identifier(column)} "
        f"ORDER BY frequency DESC LIMIT ?"
    )
    return [row[0] for row in connection.execute(sql, (limit,)).fetchall()]


def _matching_values(candidates: list[Any], observed: list[Any]) -> list[Any]:
    candidate_keys = {
        str(value).strip().casefold()
        for value in candidates
        if value is not None
    }
    return [
        value
        for value in observed
        if str(value).strip().casefold() in candidate_keys
    ]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
