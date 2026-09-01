from __future__ import annotations

import json
import unittest

from schema_agents.agentic_orchestrator import AgenticMultiAgentSchemaLinker
from schema_agents.models import DatabaseSchema


class FakeChatModel:
    model = "fake-qwen"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


class FakeEmbeddingModel:
    model = "fake-qwen3-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]


class AgenticMultiAgentSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "concert_singer",
            "table_names_original": ["singer", "concert"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
                [0, "Age"],
                [1, "concert_ID"],
                [1, "concert_Name"],
            ],
            "column_types": ["text", "number", "text", "number", "number", "text"],
            "primary_keys": [1, 4],
            "foreign_keys": [],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_manager_pipeline_returns_qwen_schema_package(self) -> None:
        responses = [
            {
                "intent_summary": "가수 이름 출력",
                "retrieval_query": "singer name",
                "outputs": [{"span": "가수의 이름", "semantic_role": "name"}],
                "filters": [],
                "grouping": [],
                "ordering": [],
                "aggregations": [],
                "set_operations": [],
                "relationships": [],
                "initial_tasks": [
                    {"agent": "schema_linker", "task": "find singer name"}
                ],
                "sql_generation_requested": False,
            },
            {
                "selected_tables": ["singer"],
                "selected_columns": ["singer.Name"],
                "roles": [{
                    "column": "singer.Name",
                    "roles": ["select"],
                    "question_spans": ["이름"],
                    "reason": "requested output",
                }],
                "unresolved": [],
            },
            {
                "run_value_linker": False,
                "value_task": "",
                "run_join_linker": False,
                "join_task": "",
                "reasons": ["one table and no filter"],
            },
            {
                "selected_tables": ["singer"],
                "selected_columns": ["singer.Name"],
                "column_roles": {"singer.Name": ["select"]},
                "grounded_filters": [],
                "joins": [],
                "unresolved": [],
            },
        ]
        linker = AgenticMultiAgentSchemaLinker(
            config={
                "embedding": {
                    "top_k_tables": 2,
                    "top_k_columns": 6,
                }
            },
            chat_model=FakeChatModel(responses),
            embedding_model=FakeEmbeddingModel(),
        )

        result = linker.link("가수의 이름을 보여줘", self.schema)

        self.assertEqual(result.tables, ["singer"])
        self.assertEqual(result.columns, ["singer.Name"])
        self.assertEqual(result.column_roles["singer.Name"], ["select"])
        self.assertEqual(result.table_scores, {})
        self.assertEqual(
            [event["agent"] for event in result.workflow_trace],
            [
                "orchestrator",
                "embedding_retriever",
                "schema_linker",
                "orchestrator",
                "value_linker",
                "join_linker",
                "orchestrator",
            ],
        )


if __name__ == "__main__":
    unittest.main()
