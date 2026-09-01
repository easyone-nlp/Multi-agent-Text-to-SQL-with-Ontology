from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .embedding_retriever import RetrievedSchema
from .join_validator import declared_fk
from .model_client import ChatModel, ModelClientError
from .models import DatabaseSchema
from .schema_llm_agent import _extract_json_object


class AgentWorkflowError(RuntimeError):
    pass


@dataclass
class AgentResponse:
    payload: dict[str, Any]
    attempts: int


class StructuredQwenAgent:
    name = "structured_qwen_agent"

    def __init__(self, model: ChatModel, json_retries: int = 1) -> None:
        self.model = model
        self.json_retries = json_retries

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AgentResponse:
        prompt = user_prompt
        last_error: Exception | None = None
        last_response = ""
        for attempt in range(self.json_retries + 1):
            try:
                complete_json = getattr(self.model, "complete_json", None)
                if callable(complete_json):
                    response = complete_json(system_prompt, prompt)
                else:
                    response = self.model.complete(system_prompt, prompt)
                last_response = response
                return AgentResponse(
                    payload=_extract_json_object(response),
                    attempts=attempt + 1,
                )
            except (ModelClientError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                prompt = (
                    f"{user_prompt}\n\nYour previous response was invalid: {error}. "
                    "Return one valid JSON object only, with no prose or markdown."
                )
        excerpt = " ".join(last_response.split())[:300]
        raise AgentWorkflowError(
            f"{self.name} structured output failed: {last_error}; "
            f"response_excerpt={excerpt!r}"
        )


class ManagerOrchestratorAgent(StructuredQwenAgent):
    name = "orchestrator"

    def decompose(
        self,
        question: str,
        include_sql_generation: bool,
        schema_linker_mode: str = "embedding_only",
    ) -> AgentResponse:
        return self.ask(
            (
                "You are the manager of a Korean Text-to-SQL multi-agent system. "
                "You do not perform schema linking, value lookup, joins, or SQL yourself. "
                "Decompose the request, assign bounded tasks to specialists, and keep "
                "ownership of the final result."
            ),
            (
                f"Korean question: {question}\n"
                f"SQL generation requested: {include_sql_generation}\n"
                f"Schema linker mode: {schema_linker_mode}\n"
                "Available specialists: schema_linker, value_linker, join_linker, "
                "sql_generator. Embedding retrieval always runs. Schema linker runs "
                "only when mode=qwen; embedding_only forwards strict retrieval candidates.\n"
                "Return exactly one JSON object with this shape:\n"
                "{"
                '"intent_summary":"...",'
                '"retrieval_query":"multilingual semantic retrieval query",'
                '"outputs":[{"span":"...","semantic_role":"..."}],'
                '"filters":[{"id":"f1","span":"...","operator":"...",'
                '"value_mention":"...","entity":"...","negated":false,'
                '"branch":"main"}],'
                '"grouping":[],"ordering":[],"aggregations":[],'
                '"set_operations":[],"relationships":[],'
                '"initial_tasks":[{"agent":"schema_linker","task":"..."},'
                '{"agent":"value_linker","task":"..."}],'
                '"sql_generation_requested":true'
                "}\n"
                "Rules:\n"
                "- Decompose every nested, negative, UNION, INTERSECT, and EXCEPT branch.\n"
                "- A filter must represent an actual predicate, not COUNT, MIN, MAX, "
                "an output, or a grouping request.\n"
                "- Request schema_linker only when mode=qwen. Request value_linker only when literal, "
                "entity, category, date, number, or coded predicate grounding is needed.\n"
                "- relationships describe entity connections needed by the question.\n"
                "- Do not invent table or column names because schema is not visible yet."
            ),
        )

    def route(
        self,
        question: str,
        decomposition: dict[str, Any],
        retrieved: RetrievedSchema,
        schema_selection: dict[str, Any],
    ) -> AgentResponse:
        return self.ask(
            (
                "You are a manager routing specialist agents. Inspect the shared state "
                "and decide which remaining specialists are required. Do not redo their work."
            ),
            (
                f"Question: {question}\n"
                f"Decomposition:\n{_json(decomposition)}\n"
                f"Retrieved schema:\n{_json(retrieved.to_dict())}\n"
                f"Schema candidate source: {schema_selection.get('source', 'unknown')}\n"
                f"Current schema candidates:\n{_json(schema_selection)}\n"
                "Return exactly:\n"
                '{"run_value_linker":true,"value_task":"...",'
                '"run_join_linker":true,"join_task":"...",'
                '"reasons":["..."]}\n'
                "Run value_linker when any filter value/category remains ungrounded. "
                "Run join_linker when multiple semantic tables, a relationship, a bridge, "
                "a subquery relationship, or join keys may be required. The manager decides "
                "whether a relationship task exists and describes that task; it does not "
                "invent concrete table names. In embedding_only mode, join_linker selects "
                "the concrete endpoint tables from retrieved candidates."
            ),
        )

class SemanticSchemaLinkerAgent(StructuredQwenAgent):
    name = "schema_linker"

    def link(
        self,
        question: str,
        decomposition: dict[str, Any],
        retrieved: RetrievedSchema,
        schema: DatabaseSchema,
    ) -> AgentResponse:
        response = self.ask(
            (
                "You are a semantic schema-linking specialist. Map decomposed Korean "
                "query roles to the embedding-retrieved English schema. Do not plan joins "
                "or guess database cell values. Return structured JSON only."
            ),
            (
                f"Question: {question}\n"
                f"Manager decomposition:\n{_json(decomposition)}\n"
                f"Embedding-retrieved schema:\n{retrieved.render(schema)}\n"
                "Return exactly:\n"
                '{"selected_tables":["table"],'
                '"selected_columns":["table.column"],'
                '"roles":[{"column":"table.column","roles":["select","filter"],'
                '"question_spans":["..."],"reason":"..."}],'
                '"unresolved":[{"span":"...","needed_role":"filter",'
                '"retrieval_hint":"..."}]}\n'
                "Roles are select, filter, group, having, order, aggregate, entity. "
                "Cover every decomposition item, but do not select pure bridge tables or "
                "join-only keys; join_linker handles those. Use only retrieved identifiers."
            ),
        )
        response.payload = validate_schema_selection(
            response.payload, schema, retrieved
        )
        return response


class JoinLinkerAgent(StructuredQwenAgent):
    name = "join_linker"

    def link(
        self,
        question: str,
        task: str,
        schema: DatabaseSchema,
        semantic_tables: list[str],
        semantic_columns: list[str],
        decomposition: dict[str, Any],
        schema_linker_mode: str = "embedding_only",
    ) -> AgentResponse:
        response = self.ask(
            (
                "You are a join-linking specialist. From the semantic candidates, "
                "select the concrete endpoint tables needed by the question, then add "
                "only indispensable bridge tables and explicit ON conditions. Every ON "
                "pair must be copied from the declared Spider foreign-key metadata. "
                "Never invent or infer a relationship. Return JSON only."
            ),
            (
                f"Question: {question}\n"
                f"Manager task: {task}\n"
                f"Decomposition:\n{_json(decomposition)}\n"
                f"Schema linker mode: {schema_linker_mode}\n"
                f"Semantic table candidates: {semantic_tables}\n"
                f"Semantic column candidates: {semantic_columns}\n"
                f"Full schema and declared relationships:\n{render_schema(schema)}\n"
                "Return exactly:\n"
                "{\"endpoint_tables\":[\"semantic_table\"],"
                "\"tables\":[\"endpoint_or_required_bridge\"],"
                "\"columns\":[\"join_key\"],"
                "\"joins\":[{\"left\":\"fk_table.column\","
                "\"right\":\"referenced_table.column\","
                "\"join_type\":\"INNER\",\"inferred\":false,"
                "\"reason\":\"...\"}],\"unresolved\":[\"...\"]}\n"
                "endpoint_tables must be a subset of the semantic table candidates and "
                "must exclude bridge-only tables. When mode is embedding_only, candidates "
                "are noisy retrieval results: select only endpoints required by the "
                "question. Copy each left/right pair exactly from the FOREIGN KEYS list. "
                "If no declared FK path exists, leave it unresolved instead of producing "
                "an inferred join. Do not connect separate set-operation/subquery branches "
                "unless the decomposition explicitly relates them."
            ),
        )
        response.payload = validate_join_output(
            response.payload,
            schema,
            allowed_endpoint_tables=semantic_tables,
        )
        return response


def validate_schema_selection(
    payload: dict[str, Any],
    schema: DatabaseSchema,
    retrieved: RetrievedSchema,
) -> dict[str, Any]:
    allowed_tables = set(retrieved.tables)
    allowed_columns = set(retrieved.columns)
    tables = _resolve_tables(payload.get("selected_tables", []), schema)
    tables = [table for table in tables if table in allowed_tables]
    columns = _resolve_columns(payload.get("selected_columns", []), schema)
    columns = [
        column
        for column in columns
        if column in allowed_columns and column.split(".", 1)[0] in allowed_tables
    ]
    roles: list[dict[str, Any]] = []
    allowed_roles = {
        "select", "filter", "group", "having", "order", "aggregate", "entity"
    }
    for raw in _dict_list(payload.get("roles", [])):
        column = _resolve_one_column(raw.get("column"), schema)
        if column is None or column not in allowed_columns:
            continue
        item_roles = [
            str(role).casefold()
            for role in raw.get("roles", [])
            if str(role).casefold() in allowed_roles
        ]
        roles.append({
            "column": column,
            "roles": list(dict.fromkeys(item_roles)),
            "question_spans": _string_list(raw.get("question_spans", [])),
            "reason": str(raw.get("reason", "")),
        })
        if column not in columns:
            columns.append(column)
        table = column.split(".", 1)[0]
        if table not in tables:
            tables.append(table)
    return {
        "selected_tables": tables,
        "selected_columns": columns,
        "roles": roles,
        "unresolved": _dict_list(payload.get("unresolved", [])),
    }


def validate_join_output(
    payload: dict[str, Any],
    schema: DatabaseSchema,
    allowed_endpoint_tables: list[str] | None = None,
) -> dict[str, Any]:
    """Accept only declared Spider FK edges and complete endpoint paths."""
    unresolved = _string_list(payload.get("unresolved", []))
    strict_workflow = allowed_endpoint_tables is not None
    allowed_endpoints = set(
        _resolve_tables(allowed_endpoint_tables or [], schema)
    )
    endpoint_tables = _resolve_tables(
        payload.get("endpoint_tables", []), schema
    )
    if strict_workflow:
        endpoint_tables = [
            table for table in endpoint_tables if table in allowed_endpoints
        ]
        if not endpoint_tables:
            endpoint_tables = [
                table
                for table in _resolve_tables(payload.get("tables", []), schema)
                if table in allowed_endpoints
            ]

    accepted: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for raw in _dict_list(payload.get("joins", [])):
        left = _resolve_one_column(raw.get("left"), schema)
        right = _resolve_one_column(raw.get("right"), schema)
        if left is None or right is None:
            unresolved.append("join rejected: unknown column identifier")
            continue
        if left.split(".", 1)[0] == right.split(".", 1)[0]:
            unresolved.append(
                f"join rejected: same-table pair {left} = {right}"
            )
            continue
        if bool(raw.get("inferred", False)):
            unresolved.append(
                f"join rejected: inferred relationship is not allowed: {left} = {right}"
            )
            continue
        declared = declared_fk(schema, left, right)
        if declared is None:
            unresolved.append(
                f"join rejected: not declared in Spider FK metadata: {left} = {right}"
            )
            continue
        if declared in seen_pairs:
            continue
        seen_pairs.add(declared)
        declared_left, declared_right = declared
        accepted.append({
            "left": declared_left,
            "right": declared_right,
            "join_type": _join_type(raw.get("join_type")),
            "inferred": False,
            "reason": str(raw.get("reason", "declared Spider FK")),
        })

    if strict_workflow:
        accepted = _keep_complete_endpoint_components(
            accepted, endpoint_tables, unresolved
        )
        tables = list(endpoint_tables)
        columns: list[str] = []
    else:
        tables = _resolve_tables(payload.get("tables", []), schema)
        columns = _resolve_columns(payload.get("columns", []), schema)

    for join in accepted:
        for column in (join["left"], join["right"]):
            if column not in columns:
                columns.append(column)
            table = column.split(".", 1)[0]
            if table not in tables:
                tables.append(table)
    return {
        "endpoint_tables": endpoint_tables,
        "tables": tables,
        "columns": columns,
        "joins": accepted,
        "unresolved": list(dict.fromkeys(unresolved)),
    }


def _keep_complete_endpoint_components(
    joins: list[dict[str, Any]],
    endpoint_tables: list[str],
    unresolved: list[str],
) -> list[dict[str, Any]]:
    if len(endpoint_tables) < 2:
        if joins:
            unresolved.append(
                "join rejected: fewer than two validated semantic endpoints"
            )
        return []

    adjacency: dict[str, set[str]] = {}
    for join in joins:
        left_table = join["left"].split(".", 1)[0]
        right_table = join["right"].split(".", 1)[0]
        adjacency.setdefault(left_table, set()).add(right_table)
        adjacency.setdefault(right_table, set()).add(left_table)

    component_by_table: dict[str, int] = {}
    components: list[set[str]] = []
    for table in adjacency:
        if table in component_by_table:
            continue
        component_id = len(components)
        stack = [table]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            component_by_table[current] = component_id
            stack.extend(adjacency.get(current, set()) - component)
        components.append(component)

    endpoint_set = set(endpoint_tables)
    valid_component_ids = {
        index
        for index, component in enumerate(components)
        if len(component & endpoint_set) >= 2
    }
    kept = [
        join
        for join in joins
        if component_by_table.get(join["left"].split(".", 1)[0])
        in valid_component_ids
    ]
    if len(kept) != len(joins):
        unresolved.append(
            "join rejected: incomplete/dangling FK path did not connect two endpoints"
        )
    connected_endpoints = {
        endpoint
        for component_id in valid_component_ids
        for endpoint in components[component_id] & endpoint_set
    }
    missing = [
        endpoint for endpoint in endpoint_tables if endpoint not in connected_endpoints
    ]
    if missing:
        unresolved.append(
            "declared FK path did not connect endpoint(s): " + ", ".join(missing)
        )
    return kept


def validate_final_package(
    payload: dict[str, Any],
    schema: DatabaseSchema,
) -> dict[str, Any]:
    tables = _resolve_tables(payload.get("selected_tables", []), schema)
    columns = _resolve_columns(payload.get("selected_columns", []), schema)
    roles: dict[str, list[str]] = {}
    raw_roles = payload.get("column_roles", {})
    if isinstance(raw_roles, dict):
        for raw_column, raw_values in raw_roles.items():
            column = _resolve_one_column(raw_column, schema)
            if column is not None:
                roles[column] = _string_list(raw_values)
                if column not in columns:
                    columns.append(column)
    filters: list[dict[str, Any]] = []
    for raw in _dict_list(payload.get("grounded_filters", [])):
        column = _resolve_one_column(raw.get("column"), schema)
        if column is None:
            continue
        item = dict(raw)
        item["column"] = column
        filters.append(item)
        if column not in columns:
            columns.append(column)
    join_output = validate_join_output(
        {
            "tables": tables,
            "columns": columns,
            "joins": payload.get("joins", []),
            "unresolved": payload.get("unresolved", []),
        },
        schema,
    )
    for column in join_output["columns"]:
        if column not in columns:
            columns.append(column)
    for table in join_output["tables"]:
        if table not in tables:
            tables.append(table)
    for column in columns:
        table = column.split(".", 1)[0]
        if table not in tables:
            tables.append(table)
    return {
        "selected_tables": tables,
        "selected_columns": columns,
        "column_roles": roles,
        "grounded_filters": filters,
        "joins": join_output["joins"],
        "unresolved": _string_list(payload.get("unresolved", [])),
    }


def render_schema(schema: DatabaseSchema) -> str:
    lines: list[str] = []
    for table in schema.tables:
        columns = ", ".join(
            f"{column.name}<{column.column_type}>"
            for column in schema.columns_for(table)
        )
        lines.append(f"- {table}({columns})")
    if schema.primary_keys:
        lines.append("Primary keys: " + ", ".join(sorted(schema.primary_keys)))
    if schema.foreign_keys:
        lines.append("Declared foreign keys:")
        lines.extend(
            f"- {left} -> {right}" for left, right in schema.foreign_keys
        )
    return "\n".join(lines)


def _resolve_tables(items: Any, schema: DatabaseSchema) -> list[str]:
    if not isinstance(items, list):
        return []
    mapping = {table.casefold(): table for table in schema.tables}
    result: list[str] = []
    for item in items:
        resolved = mapping.get(str(item).strip().strip('`"[]').casefold())
        if resolved is not None and resolved not in result:
            result.append(resolved)
    return result


def _resolve_columns(items: Any, schema: DatabaseSchema) -> list[str]:
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for item in items:
        resolved = _resolve_one_column(item, schema)
        if resolved is not None and resolved not in result:
            result.append(resolved)
    return result


def _resolve_one_column(item: Any, schema: DatabaseSchema) -> str | None:
    candidate = str(item or "").strip().strip('`"[]')
    mapping = {column.key.casefold(): column.key for column in schema.columns}
    return mapping.get(candidate.casefold())


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _join_type(value: Any) -> str:
    candidate = str(value or "INNER").strip().upper()
    return candidate if candidate in {"INNER", "LEFT", "CROSS"} else "INNER"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
