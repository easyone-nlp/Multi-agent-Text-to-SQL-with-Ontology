from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agentic_agents import (
    AgentWorkflowError,
    JoinLinkerAgent,
    ManagerOrchestratorAgent,
    SemanticSchemaLinkerAgent,
    validate_join_output,
    validate_final_package,
)
from .agentic_value_linker import ValueLinkerAgent
from .danke_multisource_retriever import MultiSourceDankeRetriever
from .embedding_retriever import EmbeddingSchemaRetriever, RetrievedSchema
from .hybrid_danke_embedding_retriever import HybridDankeEmbeddingRetriever
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
    "schema_linker_mode": "embedding_only",
    "danke": {
        "knowledge_schema_dir": "ontology/versions/danke-spider-ko-v1.0.0-20260728",
        "semantic_table_budget": 3,
        "semantic_column_budget": 8,
        "bridge_table_budget": 2,
        "fuzzy_cutoff": 0.72,
        "max_ngram": 3,
        "use_synonyms": True,
        "use_steiner": True,
    },
    "embedding": {
        "provider": "openai_compatible",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "base_url": "http://localhost:8001/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 60,
        "top_k_tables": 6,
        "top_k_columns": 24,
        "strict_top_k": False,
    },
    "value_linker": {
        "max_candidates": 4,
        "max_domain_values": 20,
        "max_rescue_queries": 4,
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
        self.schema_linker_mode = str(
            merged["schema_linker_mode"]
        ).strip().casefold()
        if self.schema_linker_mode not in {
            "qwen",
            "embedding_only",
            "danke_multisource",
            "danke_embedding_hybrid",
        }:
            raise ValueError(
                "schema_linker_mode는 qwen, embedding_only 또는 "
                "danke_multisource 또는 danke_embedding_hybrid여야 합니다."
            )
        retries = int(merged["json_retries"])
        self.manager = ManagerOrchestratorAgent(model, retries)
        self.schema_linker = (
            SemanticSchemaLinkerAgent(model, retries)
            if self.schema_linker_mode == "qwen"
            else None
        )
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
        self.max_value_rescue_queries = max(
            1, int(merged["value_linker"].get("max_rescue_queries", 4))
        )
        self.join_linker = JoinLinkerAgent(model, retries)
        if self.schema_linker_mode == "danke_multisource":
            self.retriever = MultiSourceDankeRetriever(dict(merged["danke"]))
        else:
            embedding_config = dict(merged["embedding"])
            embedder = embedding_model or build_embedding_model(embedding_config)
            embedding_retriever = EmbeddingSchemaRetriever(
                embedder,
                top_k_tables=int(embedding_config["top_k_tables"]),
                top_k_columns=int(embedding_config["top_k_columns"]),
                strict_top_k=bool(embedding_config["strict_top_k"]),
            )
            self.retriever = (
                HybridDankeEmbeddingRetriever(
                    dict(merged["danke"]), embedding_retriever
                )
                if self.schema_linker_mode == "danke_embedding_hybrid"
                else embedding_retriever
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
            question,
            include_sql_generation,
            self.schema_linker_mode,
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
        if self.schema_linker_mode in {
            "danke_multisource",
            "danke_embedding_hybrid",
        }:
            retrieved = self.retriever.retrieve_multisource(
                question, decomposition, schema
            )
        else:
            retrieved = self.retriever.retrieve(retrieval_query, schema)
        events.append({
            "agent": (
                "danke_multisource_retriever"
                if self.schema_linker_mode == "danke_multisource"
                else (
                    "danke_embedding_hybrid_retriever"
                    if self.schema_linker_mode == "danke_embedding_hybrid"
                    else "embedding_retriever"
                )
            ),
            "status": "ok",
            "task": "initial schema top-k retrieval",
            "model": self.retriever.model.model,
            "output": retrieved.to_dict(),
        })

        if self.schema_linker_mode == "qwen":
            assert self.schema_linker is not None
            schema_response = self.schema_linker.link(
                question, decomposition, retrieved, schema
            )
            schema_output = schema_response.payload
            schema_output["source"] = "qwen_schema_linker"
            events.append(_event(
                "schema_linker",
                "ok",
                "map decomposition roles to retrieved schema",
                schema_response.attempts,
                schema_output,
            ))
        else:
            schema_output = retriever_only_schema_output(
                retrieved, self.schema_linker_mode
            )
            events.append(_event(
                "schema_linker",
                "skipped",
                f"{self.schema_linker_mode} uses retrieval candidates directly",
                0,
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
                    rescue_queries = _value_rescue_queries(
                        question,
                        decomposition,
                        str(routing.get("value_task", "")),
                        self.max_value_rescue_queries,
                    )
                    rescue_results = [
                        self.retriever.retrieve(query, schema)
                        for query in rescue_queries
                    ]
                    value_retrieved = _merge_retrieved_schemas(
                        retrieved, rescue_results
                    )
                    events.append({
                        "agent": "value_retrieval_rescue",
                        "status": "ok",
                        "task": "filter-specific retrieval over the full schema",
                        "model": self.retriever.model.model,
                        "output": {
                            "queries": rescue_queries,
                            "retrievals": [
                                item.to_dict() for item in rescue_results
                            ],
                            "merged": value_retrieved.to_dict(),
                        },
                    })
                    value_response = self.value_linker.link(
                        question,
                        str(routing.get("value_task", "")),
                        decomposition,
                        schema_output,
                        value_retrieved,
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
                    self.schema_linker_mode,
                )
                join_output = join_response.payload
                if self.schema_linker_mode == "danke_embedding_hybrid":
                    join_output = _merge_hybrid_structural_evidence(
                        join_output, retrieved, schema
                    )
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


def retriever_only_schema_output(
    retrieved: RetrievedSchema, source: str = "embedding_only"
) -> dict[str, Any]:
    metadata = retrieved.metadata or {}
    if source == "danke_embedding_hybrid":
        return {
            "source": source,
            "selected_tables": list(metadata.get("semantic_tables", [])),
            "selected_columns": list(metadata.get("semantic_columns", [])),
            "structural_evidence": {
                "bridge_tables": list(metadata.get("bridge_tables", [])),
                "join_key_columns": list(
                    metadata.get("join_key_columns", [])
                ),
                "join_edges": list(metadata.get("join_edges", [])),
            },
            "roles": [],
            "unresolved": [],
        }
    return {
        "source": source,
        "selected_tables": list(retrieved.tables),
        "selected_columns": list(retrieved.columns),
        "roles": [],
        "unresolved": [],
    }


def embedding_only_schema_output(
    retrieved: RetrievedSchema,
) -> dict[str, Any]:
    return retriever_only_schema_output(retrieved, "embedding_only")


def _merge_hybrid_structural_evidence(
    join_output: dict[str, Any],
    retrieved: RetrievedSchema,
    schema: DatabaseSchema,
) -> dict[str, Any]:
    """Keep validated Steiner/FK hints structural through final aggregation."""
    metadata = retrieved.metadata or {}
    semantic_tables = list(metadata.get("semantic_tables", []))
    bridge_tables = list(metadata.get("bridge_tables", []))
    validated_edges = [
        edge
        for edge in metadata.get("join_edges", [])
        if isinstance(edge, dict) and edge.get("validated")
    ]
    if not validated_edges:
        return join_output
    hinted_joins = [
        {
            "left": str(edge["proposed_left"]),
            "right": str(edge["proposed_right"]),
            "join_type": "INNER",
            "inferred": False,
            "reason": "validated DANKE Steiner edge",
            "provenance": {
                "source": "danke_steiner_spider_fk_validator",
                "object_property": edge.get("object_property"),
                "weight": edge.get("weight"),
            },
        }
        for edge in validated_edges
    ]
    merged = {
        "endpoint_tables": semantic_tables,
        "tables": _unique(
            [
                *semantic_tables,
                *bridge_tables,
                *join_output.get("tables", []),
            ]
        ),
        "columns": _unique(
            [
                *metadata.get("join_key_columns", []),
                *join_output.get("columns", []),
            ]
        ),
        "joins": [*hinted_joins, *join_output.get("joins", [])],
        "unresolved": [
            item
            for item in join_output.get("unresolved", [])
            if not str(item).startswith("join rejected:")
        ],
        "structural_evidence_source": "hybrid_retriever_validated_fk",
    }
    validated = validate_join_output(
        merged,
        schema,
        allowed_endpoint_tables=semantic_tables,
    )
    validated["structural_evidence_source"] = (
        "hybrid_retriever_validated_fk"
    )
    return validated



def _value_rescue_queries(
    question: str,
    decomposition: dict[str, Any],
    task: str,
    limit: int,
) -> list[str]:
    """Build filter-focused multilingual queries for a second full-schema search."""
    queries: list[str] = []
    filters = _dict_list(decomposition.get("filters"))
    for item in filters[:limit]:
        parts = [
            "SQL filter predicate column retrieval",
            f"question: {question}",
            f"filter span: {item.get('span', '')}",
            f"value mention: {item.get('value_mention', '')}",
            f"entity or attribute: {item.get('entity', '')}",
            f"operator: {item.get('operator', '')}",
            f"manager hint: {task}",
        ]
        query = "; ".join(part for part in parts if not part.endswith(": "))
        if query not in queries:
            queries.append(query)
    if not queries and task:
        queries.append(
            f"SQL filter predicate column retrieval; question: {question}; "
            f"manager hint: {task}"
        )
    return queries[:limit]


def _merge_retrieved_schemas(
    initial: RetrievedSchema,
    rescues: list[RetrievedSchema],
) -> RetrievedSchema:
    tables = _unique([
        *initial.tables,
        *(table for item in rescues for table in item.tables),
    ])
    columns = _unique([
        *initial.columns,
        *(column for item in rescues for column in item.columns),
    ])
    queries = [initial.query, *(item.query for item in rescues)]
    return RetrievedSchema(
        tables=tables,
        columns=columns,
        query=" || ".join(queries),
        mode="value_rescue_union",
        top_k_tables=initial.top_k_tables,
        top_k_columns=initial.top_k_columns,
    )

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
