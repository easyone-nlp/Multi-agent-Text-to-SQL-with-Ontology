from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agentic_agents import (
    AgentWorkflowError,
    JoinLinkerAgent,
    ManagerOrchestratorAgent,
    SemanticSchemaLinkerAgent,
    validate_final_package,
)
from .agentic_value_linker import ValueLinkerAgent
from .embedding_retriever import EmbeddingSchemaRetriever
from .model_client import (
    ChatModel,
    EmbeddingModel,
    build_chat_model,
    build_embedding_model,
)
from .models import DatabaseSchema, LinkingResult


DEFAULT_AGENTIC_CONFIG: dict[str, Any] = {
    "provider": "openai_compatible",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": "http://localhost:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.0,
    "timeout_seconds": 60,
    "json_retries": 1,
    "embedding": {
        "provider": "openai_compatible",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "base_url": "http://localhost:8001/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 60,
        "top_k_tables": 6,
        "top_k_columns": 24,
    },
    "value_linker": {
        "max_candidates": 4,
        "max_domain_values": 20,
    },
}


class AgenticMultiAgentSchemaLinker:
    """Manager-style Qwen workflow with embedding retrieval and specialists."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        chat_model: ChatModel | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        merged = _merge_config(DEFAULT_AGENTIC_CONFIG, config or {})
        model = chat_model or build_chat_model(merged)
        if model is None:
            raise ValueError(
                "Agentic schema linking은 Qwen chat model이 필요합니다. "
                "provider=openai_compatible를 사용하세요."
            )
        embedding_config = dict(merged["embedding"])
        embedder = embedding_model or build_embedding_model(embedding_config)
        retries = int(merged["json_retries"])
        self.manager = ManagerOrchestratorAgent(model, retries)
        self.schema_linker = SemanticSchemaLinkerAgent(model, retries)
        self.value_linker = ValueLinkerAgent(
            model,
            json_retries=retries,
            max_candidates=int(
                merged["value_linker"]["max_candidates"]
            ),
            max_domain_values=int(
                merged["value_linker"]["max_domain_values"]
            ),
        )
        self.join_linker = JoinLinkerAgent(model, retries)
        self.retriever = EmbeddingSchemaRetriever(
            embedder,
            top_k_tables=int(embedding_config["top_k_tables"]),
            top_k_columns=int(embedding_config["top_k_columns"]),
        )

    def link(
        self,
        question: str,
        schema: DatabaseSchema,
        database_path: Path | None = None,
        include_sql_generation: bool = False,
    ) -> LinkingResult:
        events: list[dict[str, Any]] = []

        decomposition_response = self.manager.decompose(
            question, include_sql_generation
        )
        decomposition = decomposition_response.payload
        events.append(_event(
            "orchestrator",
            "ok",
            "query decomposition and initial specialist plan",
            decomposition_response.attempts,
            decomposition,
        ))

        retrieval_query = str(
            decomposition.get("retrieval_query") or question
        )
        retrieved = self.retriever.retrieve(retrieval_query, schema)
        events.append({
            "agent": "embedding_retriever",
            "status": "ok",
            "task": "initial schema top-k retrieval",
            "model": self.retriever.model.model,
            "output": retrieved.to_dict(),
        })

        schema_response = self.schema_linker.link(
            question, decomposition, retrieved, schema
        )
        schema_output = schema_response.payload
        events.append(_event(
            "schema_linker",
            "ok",
            "map decomposition roles to retrieved schema",
            schema_response.attempts,
            schema_output,
        ))

        route_response = self.manager.route(
            question, decomposition, retrieved, schema_output
        )
        routing = route_response.payload
        events.append(_event(
            "orchestrator",
            "ok",
            "route remaining value and join specialists",
            route_response.attempts,
            routing,
        ))

        value_output: dict[str, Any] = {
            "selected_tables": [],
            "selected_columns": [],
            "filters": [],
            "unresolved": [],
        }
        if bool(routing.get("run_value_linker", False)):
            if database_path is None:
                value_output["unresolved"] = [{
                    "reason": "manager requested value_linker but DB path is missing"
                }]
                events.append({
                    "agent": "value_linker",
                    "status": "error",
                    "task": str(routing.get("value_task", "")),
                    "output": value_output,
                })
            else:
                try:
                    value_response = self.value_linker.link(
                        question,
                        str(routing.get("value_task", "")),
                        decomposition,
                        schema_output,
                        retrieved,
                        schema,
                        database_path,
                    )
                    value_output = value_response.payload
                    events.append(_event(
                        "value_linker",
                        "ok",
                        str(routing.get("value_task", "")),
                        value_response.attempts,
                        value_output,
                    ))
                except AgentWorkflowError as error:
                    value_output["unresolved"] = [{"reason": str(error)}]
                    events.append({
                        "agent": "value_linker",
                        "status": "error",
                        "task": str(routing.get("value_task", "")),
                        "error": str(error),
                        "output": value_output,
                    })
        else:
            events.append({
                "agent": "value_linker",
                "status": "skipped",
                "task": str(routing.get("value_task", "")),
                "output": value_output,
            })

        semantic_tables = _unique(
            schema_output.get("selected_tables", [])
            + value_output.get("selected_tables", [])
        )
        semantic_columns = _unique(
            schema_output.get("selected_columns", [])
            + value_output.get("selected_columns", [])
        )
        join_output: dict[str, Any] = {
            "tables": [],
            "columns": [],
            "joins": [],
            "unresolved": [],
        }
        if bool(routing.get("run_join_linker", False)):
            try:
                join_response = self.join_linker.link(
                    question,
                    str(routing.get("join_task", "")),
                    schema,
                    semantic_tables,
                    semantic_columns,
                    decomposition,
                )
                join_output = join_response.payload
                events.append(_event(
                    "join_linker",
                    "ok",
                    str(routing.get("join_task", "")),
                    join_response.attempts,
                    join_output,
                ))
            except AgentWorkflowError as error:
                join_output["unresolved"] = [str(error)]
                events.append({
                    "agent": "join_linker",
                    "status": "error",
                    "task": str(routing.get("join_task", "")),
                    "error": str(error),
                    "output": join_output,
                })
        else:
            events.append({
                "agent": "join_linker",
                "status": "skipped",
                "task": str(routing.get("join_task", "")),
                "output": join_output,
            })

        final = aggregate_specialist_outputs(
            schema,
            schema_output,
            value_output,
            join_output,
        )
        events.append(_event(
            "deterministic_aggregator",
            "ok",
            "union specialist schema and assemble final package",
            0,
            final,
        ))

        return LinkingResult(
            db_id=schema.db_id,
            question=question,
            tables=final["selected_tables"],
            columns=final["selected_columns"],
            table_scores={},
            column_scores={},
            trace=[],
            query_decomposition=decomposition,
            retrieved_schema=retrieved.to_dict(),
            column_roles=final["column_roles"],
            grounded_filters=final["grounded_filters"],
            joins=final["joins"],
            unresolved=final["unresolved"],
            workflow_trace=events,
        )


MultiAgentSchemaLinker = AgenticMultiAgentSchemaLinker


def aggregate_specialist_outputs(
    schema: DatabaseSchema,
    schema_output: dict[str, Any],
    value_output: dict[str, Any],
    join_output: dict[str, Any],
) -> dict[str, Any]:
    """Losslessly union validated specialist outputs without another model call."""
    tables = _unique([
        *_list(schema_output.get("selected_tables")),
        *_list(value_output.get("selected_tables")),
        *_list(join_output.get("tables")),
    ])
    columns = _unique([
        *_list(schema_output.get("selected_columns")),
        *_list(value_output.get("selected_columns")),
        *_list(join_output.get("columns")),
    ])

    column_roles: dict[str, list[str]] = {}
    for item in _dict_list(schema_output.get("roles")):
        _add_roles(column_roles, item.get("column"), item.get("roles"))

    grounded_filters = [
        dict(item) for item in _dict_list(value_output.get("filters"))
    ]
    for item in grounded_filters:
        _add_roles(column_roles, item.get("column"), ["filter"])

    joins = [dict(item) for item in _dict_list(join_output.get("joins"))]
    for column in _list(join_output.get("columns")):
        _add_roles(column_roles, column, ["join"])
    for item in joins:
        _add_roles(column_roles, item.get("left"), ["join"])
        _add_roles(column_roles, item.get("right"), ["join"])

    unresolved = _unique([
        *_unresolved("schema_linker", schema_output.get("unresolved")),
        *_unresolved("value_linker", value_output.get("unresolved")),
        *_unresolved("join_linker", join_output.get("unresolved")),
    ])
    return validate_final_package(
        {
            "selected_tables": tables,
            "selected_columns": columns,
            "column_roles": column_roles,
            "grounded_filters": grounded_filters,
            "joins": joins,
            "unresolved": unresolved,
        },
        schema,
    )


def _event(
    agent: str,
    status: str,
    task: str,
    attempts: int,
    output: dict[str, Any],
) -> dict[str, Any]:
    return {
        "agent": agent,
        "status": status,
        "task": task,
        "attempts": attempts,
        "output": output,
    }


def _unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in _list(value) if isinstance(item, dict)]


def _add_roles(
    roles: dict[str, list[str]],
    column: Any,
    values: Any,
) -> None:
    if not isinstance(column, str) or not isinstance(values, list):
        return
    target = roles.setdefault(column, [])
    for value in values:
        role = str(value)
        if role and role not in target:
            target.append(role)


def _unresolved(source: str, value: Any) -> list[str]:
    result: list[str] = []
    for item in _list(value):
        detail = (
            item
            if isinstance(item, str)
            else json.dumps(item, ensure_ascii=False, sort_keys=True)
        )
        if detail:
            result.append(f"{source}: {detail}")
    return result


def _merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
