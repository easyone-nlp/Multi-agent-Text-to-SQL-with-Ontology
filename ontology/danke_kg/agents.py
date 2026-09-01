from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

HANGUL_PATTERN = re.compile(r"[가-힣]")

from .json_utils import extract_json_object
from .model_client import ChatModel, ModelClientError
from .relational import RelColumn


class AgentWorkflowError(RuntimeError):
    pass


@dataclass
class AgentResponse:
    payload: dict[str, Any]
    attempts: int


class StructuredLLMAgent:
    """Base specialist: one structured JSON call with a bounded retry."""

    name = "structured_llm_agent"

    def __init__(self, model: ChatModel, json_retries: int = 1) -> None:
        self.model = model
        self.json_retries = json_retries

    def ask(self, system_prompt: str, user_prompt: str) -> AgentResponse:
        prompt = user_prompt
        last_error: Exception | None = None
        last_response = ""
        for attempt in range(self.json_retries + 1):
            try:
                complete_json = getattr(self.model, "complete_json", None)
                response = (
                    complete_json(system_prompt, prompt)
                    if callable(complete_json)
                    else self.model.complete(system_prompt, prompt)
                )
                last_response = response
                return AgentResponse(payload=extract_json_object(response), attempts=attempt + 1)
            except (ModelClientError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                prompt = (
                    f"{user_prompt}\n\nYour previous response was invalid: {error}. "
                    "Return one valid JSON object only, with no prose or markdown."
                )
        excerpt = " ".join(last_response.split())[:300]
        raise AgentWorkflowError(
            f"{self.name} structured output failed: {last_error}; response_excerpt={excerpt!r}"
        )


class TableOntologyAgent(StructuredLLMAgent):
    """Enriches one table (class) and its columns (datatype properties)

    with Korean labels/synonyms/rankings, mirroring how DANKE's dictionary
    stores class/property primary labels and synonym labels (Section 3.1).
    """

    name = "table_ontology_agent"

    def enrich_table(
        self,
        table: str,
        columns: list[RelColumn],
        primary_keys: set[str],
        foreign_keys: list[tuple[str, str]],
        korean_table: str | None = None,
        korean_columns: dict[str, str] | None = None,
        include_synonyms: bool = True,
    ) -> AgentResponse:
        fk_columns = {left for left, _ in foreign_keys} | {right for _, right in foreign_keys}
        korean_columns = korean_columns or {}
        column_lines = []
        for column in columns:
            flags = []
            if column.key in primary_keys:
                flags.append("PK")
            if column.key in fk_columns:
                flags.append("FK")
            flag_text = f" [{','.join(flags)}]" if flags else ""
            hint = korean_columns.get(column.name)
            hint_text = f" -- official Korean name: {hint}" if hint else ""
            column_lines.append(f"- {column.name} ({column.column_type}){flag_text}{hint_text}")

        table_hint = f" (official Korean name: {korean_table})" if korean_table else ""
        if include_synonyms:
            response_shape = (
                '{"class":{"primary_label":"...","synonyms":["...","..."],'
                '"ranking":1.0},'
                '"properties":[{"column":"<exact column name>",'
                '"primary_label":"...","synonyms":["..."],"ranking":1.0,'
                '"indexed":true,'
                '"value_synonyms":[{"value":"<observed or example value>",'
                '"synonyms":["..."]}]}]}'
            )
            synonym_rules = (
                "- primary_label and synonyms should be natural Korean terms end "
                "users search with; include the original English/technical term too.\n"
                "- value_synonyms is only for indexed columns with a small enumerated "
                "domain (status/type/category codes); otherwise return an empty list.\n"
            )
        else:
            response_shape = (
                '{"class":{"primary_label":"...","ranking":1.0},'
                '"properties":[{"column":"<exact column name>",'
                '"primary_label":"...","ranking":1.0,"indexed":true,'
                '"value_synonyms":[{"value":"<observed or example value>",'
                '"synonyms":["..."]}]}]}'
            )
            synonym_rules = (
                "- primary_label MUST be written in Korean -- it is the only label "
                "this schema element will have, so never put the English/technical "
                "term there instead (e.g. for a column named \"Age\" use \"나이\", "
                "never \"age\"). Do NOT generate table/column-name synonyms -- those "
                "will be curated separately later, so omit that field entirely.\n"
                "- value_synonyms is the one exception: still generate it for "
                "indexed columns with a small enumerated domain (status/type/"
                "category codes), since it encodes DB business logic rather than "
                "a naming preference; otherwise return an empty list.\n"
            )
        response = self.ask(
            (
                "You are building a DANKE-style OWL2 knowledge schema for a Korean "
                "Text-to-SQL system: a class per table and a datatype property per "
                "column, each carrying a primary label. You do not invent columns. "
                "When an official Korean name is given for the table or a column, "
                "treat it as authoritative and build the primary_label around it "
                "rather than inventing an unrelated term."
            ),
            (
                f"Table: {table}{table_hint}\n"
                f"Columns:\n" + "\n".join(column_lines) + "\n\n"
                "Return exactly one JSON object:\n"
                f"{response_shape}\n"
                "Rules:\n"
                f"{synonym_rules}"
                "- ranking is a float in [0.1, 5.0] reflecting how central this "
                "class/property is to typical questions (higher = more central).\n"
                "- indexed=true only for columns whose actual values are meaningful "
                "search terms (names, categories, statuses, free text). indexed=false "
                "for surrogate keys, pure numeric measurements, and FK/PK columns.\n"
                "- Cover every listed column exactly once."
            ),
        )
        payload = validate_table_enrichment(response.payload, table, columns)
        if not include_synonyms:
            payload["class"]["synonyms"] = []
            for item in payload["properties"]:
                item["synonyms"] = []
        if not include_synonyms:
            # Informational only (surfaced in the trace for later review) --
            # deliberately not retried; a follow-up "fix these labels" call
            # was tried and turned out less reliable than the first pass.
            payload["_non_korean_labels"] = _non_korean_labels(payload)
        response.payload = payload
        return response


class RelationshipOntologyAgent(StructuredLLMAgent):
    """Enriches declared foreign keys into labeled object properties."""

    name = "relationship_ontology_agent"

    def enrich_relationships(
        self, relationships: list[dict[str, Any]], include_synonyms: bool = True
    ) -> AgentResponse:
        lines = []
        for item in relationships:
            origin = "inferred from a shared code column (no declared FK constraint)" if item.get(
                "inferred"
            ) else "a declared foreign key"
            lines.append(
                f"- {item['left']} -> {item['right']} "
                f"(class {item['left_class']} -> class {item['right_class']}; {origin})"
            )
        if include_synonyms:
            response_shape = (
                '{"object_properties":[{"left":"table.column",'
                '"right":"table.column","primary_label":"...",'
                '"synonyms":["..."],"weight":1.0}]}'
            )
            synonym_rule = ""
        else:
            response_shape = (
                '{"object_properties":[{"left":"table.column",'
                '"right":"table.column","primary_label":"...","weight":1.0}]}'
            )
            synonym_rule = (
                "- Do NOT generate synonyms -- they will be curated separately "
                "later, so omit that field entirely.\n"
            )
        response = self.ask(
            (
                "You are labeling object properties of a DANKE-style knowledge "
                "schema from table relationships. Each object property connects "
                "two classes and needs a Korean relationship phrase and a traversal "
                "weight for Steiner-tree view synthesis."
            ),
            (
                "Relationships:\n" + "\n".join(lines) + "\n\n"
                "Return exactly one JSON object:\n"
                f"{response_shape}\n"
                "Rules:\n"
                "- primary_label is a short Korean phrase for the relationship "
                "(e.g. '소속', '주문한', '작성한').\n"
                f"{synonym_rule}"
                "- weight is a float in [0.1, 5.0]; use a LOWER weight for "
                "relationships that are direct, core, and frequently traversed, "
                "and a HIGHER weight for peripheral/rarely relevant relationships, "
                "since Steiner-tree view synthesis minimizes total path weight. "
                "Relationships inferred from a shared code column (not a declared "
                "FK) are less certain -- give them a weight at least as high as any "
                "declared-FK relationship in this same list, never lower.\n"
                "- Cover every listed relationship exactly once."
            ),
        )
        response.payload = validate_relationship_enrichment(response.payload, relationships)
        if not include_synonyms:
            for item in response.payload["object_properties"]:
                item["synonyms"] = []
        return response


def validate_table_enrichment(
    payload: dict[str, Any],
    table: str,
    columns: list[RelColumn],
) -> dict[str, Any]:
    del table
    class_raw = payload.get("class", {})
    class_out = {
        "primary_label": str(class_raw.get("primary_label", "")).strip(),
        "synonyms": _string_list(class_raw.get("synonyms", [])),
        "ranking": _clamp_ranking(class_raw.get("ranking", 1.0)),
    }

    known_columns = {column.name: column for column in columns}
    seen: set[str] = set()
    properties: list[dict[str, Any]] = []
    for raw in _dict_list(payload.get("properties", [])):
        column_name = str(raw.get("column", "")).strip()
        if column_name not in known_columns or column_name in seen:
            continue
        seen.add(column_name)
        value_synonyms: dict[str, list[str]] = {}
        for entry in _dict_list(raw.get("value_synonyms", [])):
            value = str(entry.get("value", "")).strip()
            if not value:
                continue
            value_synonyms[value] = _string_list(entry.get("synonyms", []))
        properties.append(
            {
                "column": column_name,
                "primary_label": str(raw.get("primary_label", "")).strip(),
                "synonyms": _string_list(raw.get("synonyms", [])),
                "ranking": _clamp_ranking(raw.get("ranking", 1.0)),
                "indexed": bool(raw.get("indexed", False)),
                "value_synonyms": value_synonyms,
            }
        )
    return {"class": class_out, "properties": properties}


def validate_relationship_enrichment(
    payload: dict[str, Any],
    relationships: list[dict[str, str]],
) -> dict[str, Any]:
    known_pairs = {(item["left"], item["right"]) for item in relationships}
    seen: set[tuple[str, str]] = set()
    object_properties: list[dict[str, Any]] = []
    for raw in _dict_list(payload.get("object_properties", [])):
        left = str(raw.get("left", "")).strip()
        right = str(raw.get("right", "")).strip()
        if (left, right) not in known_pairs or (left, right) in seen:
            continue
        seen.add((left, right))
        object_properties.append(
            {
                "left": left,
                "right": right,
                "primary_label": str(raw.get("primary_label", "")).strip(),
                "synonyms": _string_list(raw.get("synonyms", [])),
                "weight": _clamp_ranking(raw.get("weight", 1.0)),
            }
        )
    return {"object_properties": object_properties}


def _non_korean_labels(payload: dict[str, Any]) -> list[str]:
    """Primary labels missing Hangul entirely -- a sign the model fell back

    to English, which is otherwise invisible once synonyms are disabled
    (there is no separate synonyms field left to hold the English term).
    """
    missing: list[str] = []
    class_label = payload.get("class", {}).get("primary_label", "")
    if class_label and not HANGUL_PATTERN.search(class_label):
        missing.append(f"class={class_label!r}")
    for item in payload.get("properties", []):
        label = item.get("primary_label", "")
        if label and not HANGUL_PATTERN.search(label):
            missing.append(f"{item.get('column')}={label!r}")
    return missing


def _clamp_ranking(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(5.0, max(0.1, number))


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
