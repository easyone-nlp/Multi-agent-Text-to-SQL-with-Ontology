from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Column:
    index: int
    table: str
    name: str
    column_type: str

    @property
    def key(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class DatabaseSchema:
    db_id: str
    tables: list[str]
    columns: list[Column]
    primary_keys: set[str]
    foreign_keys: list[tuple[str, str]]

    @classmethod
    def from_spider(cls, raw: dict[str, Any]) -> "DatabaseSchema":
        tables = raw["table_names_original"]
        types = raw.get("column_types", [])
        columns: list[Column] = []
        by_index: dict[int, str] = {}
        for index, (table_index, name) in enumerate(raw["column_names_original"]):
            if table_index < 0:
                continue
            column = Column(
                index=index,
                table=tables[table_index],
                name=name,
                column_type=types[index] if index < len(types) else "unknown",
            )
            columns.append(column)
            by_index[index] = column.key

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
        )

    def columns_for(self, table: str) -> list[Column]:
        return [column for column in self.columns if column.table == table]

    def column(self, key: str) -> Column | None:
        return next((column for column in self.columns if column.key == key), None)


@dataclass
class Example:
    db_id: str
    question: str
    gold_sql: str | None = None


@dataclass
class AgentProposal:
    agent: str
    table_scores: dict[str, float] = field(default_factory=dict)
    column_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "table_scores": _sorted_scores(self.table_scores),
            "column_scores": _sorted_scores(self.column_scores),
            "reasons": self.reasons,
        }


@dataclass
class ValueEvidence:
    mention: str
    column: str
    operator: str
    probe_mode: str
    candidate_values: list[Any] = field(default_factory=list)
    matched_values: list[Any] = field(default_factory=list)
    observed_values: list[Any] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "column": self.column,
            "operator": self.operator,
            "probe_mode": self.probe_mode,
            "candidate_values": self.candidate_values,
            "matched_values": self.matched_values,
            "observed_values": self.observed_values,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ValueEvidence":
        return cls(
            mention=str(raw.get("mention", "")),
            column=str(raw.get("column", "")),
            operator=str(raw.get("operator", "=")),
            probe_mode=str(raw.get("probe_mode", "exact")),
            candidate_values=list(raw.get("candidate_values", [])),
            matched_values=list(raw.get("matched_values", [])),
            observed_values=list(raw.get("observed_values", [])),
            confidence=float(raw.get("confidence", 0.0)),
        )


@dataclass
class LinkingResult:
    db_id: str
    question: str
    tables: list[str]
    columns: list[str]
    table_scores: dict[str, float]
    column_scores: dict[str, float]
    trace: list[AgentProposal]
    value_evidence: list[ValueEvidence] = field(default_factory=list)
    query_decomposition: dict[str, Any] = field(default_factory=dict)
    retrieved_schema: dict[str, Any] = field(default_factory=dict)
    column_roles: dict[str, list[str]] = field(default_factory=dict)
    grounded_filters: list[dict[str, Any]] = field(default_factory=list)
    joins: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    workflow_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_trace: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "db_id": self.db_id,
            "question": self.question,
            "selected_tables": self.tables,
            "selected_columns": self.columns,
            "table_scores": _sorted_scores(self.table_scores),
            "column_scores": _sorted_scores(self.column_scores),
        }
        if self.query_decomposition:
            result["query_decomposition"] = self.query_decomposition
        if self.retrieved_schema:
            result["retrieved_schema"] = self.retrieved_schema
        if self.column_roles:
            result["column_roles"] = self.column_roles
        if self.grounded_filters:
            result["grounded_filters"] = self.grounded_filters
        if self.joins:
            result["joins"] = self.joins
        if self.unresolved:
            result["unresolved"] = self.unresolved
        if include_trace:
            if self.workflow_trace:
                result["agent_trace"] = self.workflow_trace
            else:
                result["agent_trace"] = [
                    proposal.to_dict() for proposal in self.trace
                ]
        if self.value_evidence:
            result["value_grounding"] = [
                evidence.to_dict() for evidence in self.value_evidence
            ]
        return result


def _sorted_scores(scores: dict[str, float]) -> dict[str, float]:
    return {
        key: round(value, 4)
        for key, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if value > 0
    }
