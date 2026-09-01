from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .model_client import ChatModel, ModelClientError
from .models import AgentProposal, DatabaseSchema


@dataclass
class ParsedSchemaSelection:
    tables: list[str]
    columns: list[str]
    ignored: list[str]


class LLMSchemaScoutAgent:
    name = "llm_schema_scout"

    def __init__(self, model: ChatModel, score: float = 4.0) -> None:
        self.model = model
        self.score = score

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
    ) -> AgentProposal:
        del previous
        try:
            response = self.model.complete(
                (
                    "You are a Korean Text-to-SQL semantic schema scout. "
                    "Select only tables and columns whose meaning is directly needed by "
                    "SELECT, WHERE, GROUP BY, HAVING, ORDER BY, or aggregation. "
                    "A deterministic graph agent will add bridge tables and join keys. "
                    "Return one JSON object only."
                ),
                schema_selection_prompt(question, schema),
            )
            parsed = parse_schema_selection(response, schema)
        except (ModelClientError, ValueError, json.JSONDecodeError) as error:
            return AgentProposal(
                self.name,
                reasons=[f"LLM schema linking 실패; 규칙 agent로 fallback: {error}"],
            )

        return _proposal_from_selection(
            self.name, self.model.model, parsed, self.score, "semantic selection"
        )


class LLMSchemaCriticAgent:
    name = "llm_schema_critic"

    def __init__(self, model: ChatModel, score: float = 4.5) -> None:
        self.model = model
        self.score = score

    def propose(
        self,
        question: str,
        schema: DatabaseSchema,
        previous: list[AgentProposal],
    ) -> AgentProposal:
        scout = next(
            (proposal for proposal in previous if proposal.agent == "llm_schema_scout"),
            AgentProposal("llm_schema_scout"),
        )
        current_tables = list(scout.table_scores)
        current_columns = list(scout.column_scores)
        try:
            response = self.model.complete(
                (
                    "You are a conservative schema-linking critic. Find only semantic "
                    "tables or columns missing from the current selection. Focus on every "
                    "branch of negation, subqueries, and set operations. A graph agent will "
                    "add bridge tables and join keys. Return one JSON object only."
                ),
                schema_critic_prompt(
                    question, schema, current_tables, current_columns
                ),
            )
            parsed = parse_schema_selection(response, schema, allow_empty=True)
        except (ModelClientError, ValueError, json.JSONDecodeError) as error:
            return AgentProposal(
                self.name,
                reasons=[f"LLM schema critic 실패; 기존 후보 유지: {error}"],
            )

        parsed.tables = [table for table in parsed.tables if table not in current_tables]
        parsed.columns = [
            column for column in parsed.columns if column not in current_columns
        ]
        return _proposal_from_selection(
            self.name, self.model.model, parsed, self.score, "missing-only review"
        )


def schema_selection_prompt(question: str, schema: DatabaseSchema) -> str:
    lines = _schema_lines(question, schema)
    lines.extend(
        [
            "Return exactly this JSON shape:",
            '{"tables": ["table"], "columns": ["table.column"]}',
            "Semantic-selection rules:",
            "- Use only names present in the schema.",
            "- Select endpoint tables directly needed to answer or filter the question.",
            "- Select only columns needed by SELECT, WHERE, GROUP BY, HAVING, ORDER BY, or aggregation.",
            "- Do NOT select pure bridge tables or PK/FK columns used only for joining.",
            "- Do NOT return every column of a selected table.",
            "- Map every value-bearing condition to a column, including negated conditions.",
            "- Examples: '고양이' needs a pet-type/category column; '1970년' needs a year column.",
            "- For a plain COUNT(*) question, return the counted table and an empty columns list.",
            "- Keep every nested, negative, EXCEPT, INTERSECT, and UNION branch in mind.",
            "Examples:",
            'Question: "가수는 몇 명인가요?" -> {"tables":["singer"],"columns":[]}',
            'Question: "평균 수용 인원이 가장 큰 경기장 이름" -> '
            '{"tables":["stadium"],"columns":["stadium.Name","stadium.Average"]}',
        ]
    )
    return "\n".join(lines)


def schema_critic_prompt(
    question: str,
    schema: DatabaseSchema,
    current_tables: list[str],
    current_columns: list[str],
) -> str:
    lines = _schema_lines(question, schema)
    lines.extend(
        [
            f"Current semantic tables: {current_tables}",
            f"Current semantic columns: {current_columns}",
            "Return exactly this JSON shape:",
            '{"tables": ["missing_table"], "columns": ["missing_table.column"]}',
            "Critic rules:",
            "- Return additions only; do not repeat the current selection.",
            "- If nothing semantic is missing, return {\"tables\":[],\"columns\":[]}.",
            "- Check output, filters, grouping, ordering, HAVING, and aggregation.",
            "- For '없는/아닌/제외', identify the returned entity and the entity whose existence is denied.",
            "- Map every literal or category to its filter column, even inside negation.",
            "- For EXCEPT, INTERSECT, or UNION, inspect every branch.",
            "- Select semantic endpoints, not association tables between endpoints.",
            "- Do NOT add pure bridge tables or PK/FK columns used only for joining.",
            "- Do NOT add speculative columns or every column of a table.",
            "Negation example:",
            "Question: 고양이를 소유하지 않은 학생의 나이",
            "Current: tables=['Student'], columns=['Student.Age']",
            "Return: {\"tables\":[\"Pets\"],\"columns\":[\"Pets.PetType\"]}",
            "Do not return Has_Pet or its IDs; graph completion adds that bridge and join keys.",
        ]
    )
    return "\n".join(lines)


def parse_schema_selection(
    response: str,
    schema: DatabaseSchema,
    allow_empty: bool = False,
) -> ParsedSchemaSelection:
    payload = _extract_json_object(response)
    table_items = payload.get("tables", payload.get("selected_tables", []))
    column_items = payload.get("columns", payload.get("selected_columns", []))
    if not isinstance(table_items, list) or not isinstance(column_items, list):
        raise ValueError("tables와 columns는 JSON list여야 함")

    table_map = {table.casefold(): table for table in schema.tables}
    column_map = {column.key.casefold(): column.key for column in schema.columns}
    columns_by_name: dict[str, list[str]] = {}
    for column in schema.columns:
        columns_by_name.setdefault(column.name.casefold(), []).append(column.key)

    tables: list[str] = []
    columns: list[str] = []
    ignored: list[str] = []
    for item in table_items:
        candidate = _clean_identifier(item)
        resolved = table_map.get(candidate.casefold())
        if resolved is None:
            ignored.append(candidate)
        elif resolved not in tables:
            tables.append(resolved)

    for item in column_items:
        candidate = _column_candidate(item)
        resolved = column_map.get(candidate.casefold())
        if resolved is None and "." not in candidate:
            matches = columns_by_name.get(candidate.casefold(), [])
            if len(matches) == 1:
                resolved = matches[0]
        if resolved is None:
            ignored.append(candidate)
            continue
        if resolved not in columns:
            columns.append(resolved)
        table = resolved.split(".", 1)[0]
        if table not in tables:
            tables.append(table)

    if not tables and not allow_empty:
        raise ValueError("유효한 table 후보가 없음")
    return ParsedSchemaSelection(tables, columns, ignored)


def _schema_lines(question: str, schema: DatabaseSchema) -> list[str]:
    lines = [f"Korean question: {question}", f"Database: {schema.db_id}", "Schema:"]
    for table in schema.tables:
        columns = ", ".join(
            f"{column.name}<{column.column_type}>"
            for column in schema.columns_for(table)
        )
        lines.append(f"- {table}({columns})")
    if schema.primary_keys:
        lines.append("Primary keys: " + ", ".join(sorted(schema.primary_keys)))
    if schema.foreign_keys:
        relationships = [f"{left} = {right}" for left, right in schema.foreign_keys]
        lines.append("Foreign keys: " + "; ".join(relationships))
    return lines


def _proposal_from_selection(
    agent: str,
    model: str,
    parsed: ParsedSchemaSelection,
    score: float,
    mode: str,
) -> AgentProposal:
    table_scores = {table: score for table in parsed.tables}
    column_scores = {column: score for column in parsed.columns}
    for column in parsed.columns:
        table_scores.setdefault(column.split(".", 1)[0], score)
    reasons = [
        f"model={model}; mode={mode}; tables={len(table_scores)}, columns={len(column_scores)}"
    ]
    if parsed.ignored:
        reasons.append(f"존재하지 않아 무시한 후보: {', '.join(parsed.ignored[:8])}")
    return AgentProposal(agent, table_scores, column_scores, reasons)


def _extract_json_object(response: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.I | re.S)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("model 응답에서 JSON object를 찾지 못함")


def _column_candidate(item: Any) -> str:
    if isinstance(item, dict):
        table = item.get("table", "")
        column = item.get("column", item.get("name", ""))
        return f"{_clean_identifier(table)}.{_clean_identifier(column)}"
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"{_clean_identifier(item[0])}.{_clean_identifier(item[1])}"
    return _clean_identifier(item)


def _clean_identifier(item: Any) -> str:
    return str(item).strip().strip('`"[]').strip()
