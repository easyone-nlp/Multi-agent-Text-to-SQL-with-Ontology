from __future__ import annotations

from pathlib import Path
from typing import Any

from .agents import IntentScoutAgent, LexicalScoutAgent, ReviewerAgent, SchemaGraphAgent
from .model_client import build_chat_model
from .models import DatabaseSchema, LinkingResult
from .schema_llm_agent import LLMSchemaCriticAgent, LLMSchemaScoutAgent
from .value_grounding_agent import DBValueGroundingAgent


DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "heuristic",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": "http://localhost:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.0,
    "timeout_seconds": 60,
    "max_tables": 4,
    "max_columns": 16,
    "relative_table_threshold": 0.35,
    "relative_column_threshold": 0.4,
    "include_primary_keys": True,
    "enable_value_grounding": True,
    "value_grounding_exact_score": 6.0,
    "value_grounding_categorical_score": 5.0,
    "value_grounding_max_candidates": 4,
    "value_grounding_max_domain_values": 20,
    "weights": {
        "lexical_scout": 1.0,
        "llm_schema_critic": 1.5,
        "intent_scout": 0.6,
        "llm_schema_scout": 1.5,
        "db_value_grounder": 1.0,
        "schema_graph": 0.45,
    },
}


class LegacyWeightedSchemaLinker:
    """Archived weighted linker; the CLI no longer uses this implementation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = {**DEFAULT_CONFIG, **(config or {})}
        self.scouts = [LexicalScoutAgent(), IntentScoutAgent()]
        model = build_chat_model(merged)
        self.critic_agent = None
        self.value_agent = None
        if model is not None:
            self.scouts.append(LLMSchemaScoutAgent(model))
            self.critic_agent = LLMSchemaCriticAgent(model)
            if bool(merged["enable_value_grounding"]):
                self.value_agent = DBValueGroundingAgent(
                    model,
                    exact_score=float(merged["value_grounding_exact_score"]),
                    categorical_score=float(
                        merged["value_grounding_categorical_score"]
                    ),
                    max_candidates_per_condition=int(
                        merged["value_grounding_max_candidates"]
                    ),
                    max_domain_values=int(
                        merged["value_grounding_max_domain_values"]
                    ),
                )
        self.graph_agent = SchemaGraphAgent()
        self.reviewer = ReviewerAgent(
            weights=merged["weights"],
            max_tables=int(merged["max_tables"]),
            max_columns=int(merged["max_columns"]),
            relative_table_threshold=float(merged["relative_table_threshold"]),
            relative_column_threshold=float(merged["relative_column_threshold"]),
            include_primary_keys=bool(merged["include_primary_keys"]),
        )

    def link(
        self,
        question: str,
        schema: DatabaseSchema,
        database_path: Path | None = None,
    ) -> LinkingResult:
        proposals = [agent.propose(question, schema, []) for agent in self.scouts]
        if self.critic_agent is not None:
            proposals.append(self.critic_agent.propose(question, schema, proposals))
        value_evidence = []
        if self.value_agent is not None:
            value_proposal, value_evidence = self.value_agent.propose(
                question, schema, proposals, database_path
            )
            proposals.append(value_proposal)
        proposals.append(self.graph_agent.propose(question, schema, proposals))
        tables, columns, table_scores, column_scores, review = self.reviewer.decide(
            schema, proposals
        )
        proposals.append(review)
        return LinkingResult(
            db_id=schema.db_id,
            question=question,
            tables=tables,
            columns=columns,
            table_scores=table_scores,
            column_scores=column_scores,
            trace=proposals,
            value_evidence=value_evidence,
        )


# Runtime entry point: manager-style Qwen workflow.
# Legacy weighted classes above remain only for old imports/results.
from .agentic_orchestrator import (
    AgenticMultiAgentSchemaLinker as MultiAgentSchemaLinker,
)
