from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_client import ChatModel, ModelClientError, build_chat_model
from .models import DatabaseSchema, LinkingResult
from .sql_agents import (
    ExecutionResult,
    SQLAgentEvent,
    SQLDrafterAgent,
    SQLGenerationResult,
    SQLRepairAgent,
    SQLSafetyValidator,
    SQLiteExecutorAgent,
)


DEFAULT_SQL_CONFIG: dict[str, Any] = {
    "provider": "openai_compatible",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": "http://localhost:8000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "temperature": 0.0,
    "timeout_seconds": 60,
    "max_repairs": 2,
    "preview_rows": 5,
}


class MultiAgentSQLGenerator:
    """Drafts, critiques, executes, and repairs a SQL query."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        chat_model: ChatModel | None = None,
    ) -> None:
        merged = {**DEFAULT_SQL_CONFIG, **(config or {})}
        model = chat_model or build_chat_model(merged)
        if model is None:
            raise ValueError(
                "Agentic SQL generation은 Qwen chat model이 필요합니다. "
                "provider=openai_compatible를 사용하세요."
            )
        self.drafter = SQLDrafterAgent(model)
        self.critic = SQLSafetyValidator()
        self.executor = SQLiteExecutorAgent(preview_rows=int(merged["preview_rows"]))
        self.repairer = SQLRepairAgent(model)
        self.max_repairs = int(merged["max_repairs"])

    def generate(
        self,
        question: str,
        schema: DatabaseSchema,
        linking: LinkingResult,
        database_path: Path,
    ) -> SQLGenerationResult:
        trace: list[SQLAgentEvent] = []
        try:
            sql, event = self.drafter.draft(question, schema, linking)
        except (ValueError, ModelClientError) as error:
            message = f"SQL draft failed: {error}"
            trace.append(
                SQLAgentEvent(
                    self.drafter.name,
                    "error",
                    {"error": message},
                )
            )
            return SQLGenerationResult(
                sql="",
                success=False,
                attempts=1,
                execution=ExecutionResult(False, error=message),
                trace=trace,
            )
        trace.append(event)
        execution = self.executor.execute(sql, database_path)
        attempts = 1
        seen_sql = {sql}

        for repair_index in range(self.max_repairs + 1):
            trace.append(self.critic.review(sql, schema, linking))
            trace.append(
                SQLAgentEvent(
                    self.executor.name,
                    "ok" if execution.success else "error",
                    execution.to_dict(),
                )
            )
            if execution.success or repair_index >= self.max_repairs:
                break
            try:
                repaired, repair_event = self.repairer.repair(
                    question,
                    schema,
                    linking,
                    sql,
                    execution.error or "unknown execution error",
                )
            except (ValueError, ModelClientError) as error:
                trace.append(
                    SQLAgentEvent(
                        self.repairer.name,
                        "error",
                        {"error": f"SQL repair failed: {error}"},
                    )
                )
                break
            trace.append(repair_event)
            if repaired in seen_sql:
                trace.append(
                    SQLAgentEvent(
                        "coordinator",
                        "stopped",
                        {"reason": "repair가 이미 시도한 SQL을 반복함"},
                    )
                )
                break
            sql = repaired
            seen_sql.add(sql)
            attempts += 1
            execution = self.executor.execute(sql, database_path)

        return SQLGenerationResult(
            sql=sql,
            success=execution.success,
            attempts=attempts,
            execution=execution,
            trace=trace,
        )
