from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from schema_agents.models import DatabaseSchema, LinkingResult, ValueEvidence
from schema_agents.sql_agents import SQLiteExecutorAgent, generation_prompt
from schema_agents.sql_orchestrator import MultiAgentSQLGenerator


class FakeSQLChatModel:
    model = "fake-qwen"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "SELECT COUNT(*) FROM singer"


class SQLGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "concert_singer",
            "table_names_original": ["singer"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
                [0, "Age"],
            ],
            "column_types": ["text", "number", "text", "number"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_count_query_is_generated_and_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "concert_singer.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE singer (Singer_ID INTEGER, Name TEXT, Age INTEGER);"
                "INSERT INTO singer VALUES (1, 'A', 20), (2, 'B', 30);"
            )
            connection.close()
            question = "가수는 몇 명이 있나요?"
            linking = LinkingResult(
                db_id="concert_singer",
                question=question,
                tables=["singer"],
                columns=["singer.Singer_ID"],
                table_scores={},
                column_scores={},
                trace=[],
                query_decomposition={
                    "aggregations": [{"function": "count", "target": "singer"}]
                },
                column_roles={"singer.Singer_ID": ["aggregate"]},
            )
            result = MultiAgentSQLGenerator(
                chat_model=FakeSQLChatModel()
            ).generate(
                question, self.schema, linking, database
            )
            self.assertTrue(result.success)
            self.assertIn("COUNT(*)", result.sql)
            self.assertEqual(result.execution.rows, [[2]])

    def test_generation_prompt_uses_only_linked_columns(self) -> None:
        linking = LinkingResult(
            db_id="concert_singer",
            question="가수 이름",
            tables=["singer"],
            columns=["singer.Name"],
            table_scores={"singer": 6.0},
            column_scores={"singer.Name": 6.0},
            trace=[],
        )
        prompt = generation_prompt("가수 이름", self.schema, linking)
        self.assertIn("\"Name\" text", prompt)
        self.assertNotIn("Singer_ID", prompt)
        self.assertNotIn("Age", prompt)

    def test_generation_prompt_includes_verified_value_evidence(self) -> None:
        linking = LinkingResult(
            db_id="concert_singer",
            question="이름이 A인 가수",
            tables=["singer"],
            columns=["singer.Name"],
            table_scores={"singer": 6.0},
            column_scores={"singer.Name": 6.0},
            trace=[],
            value_evidence=[
                ValueEvidence(
                    mention="A",
                    column="singer.Name",
                    operator="=",
                    probe_mode="exact",
                    candidate_values=["A"],
                    matched_values=["A"],
                    confidence=1.0,
                )
            ],
        )

        prompt = generation_prompt(linking.question, self.schema, linking)

        self.assertIn("DB value-grounding evidence", prompt)
        self.assertIn("singer.Name = ['A']", prompt)
        self.assertIn("DB-verified exact", prompt)

    def test_generation_prompt_includes_spider_column_metadata(self) -> None:
        raw = {
            "db_id": "join_chain",
            "table_names_original": ["maker", "model", "car"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"],
                [0, "name"],
                [1, "maker_id"],
                [1, "model_id"],
                [2, "model_id"],
                [2, "year"],
            ],
            "column_types": [
                "text", "number", "text", "number", "number", "number", "number"
            ],
            "primary_keys": [1, 4, 5],
            "foreign_keys": [[3, 1], [5, 4]],
        }
        schema = DatabaseSchema.from_spider(raw)
        linking = LinkingResult(
            db_id="join_chain",
            question="1970년에 자동차를 생산한 제조사 이름",
            tables=["maker", "model", "car"],
            columns=[
                "maker.id",
                "maker.name",
                "model.maker_id",
                "model.model_id",
                "car.model_id",
                "car.year",
            ],
            table_scores={table: 6.0 for table in ["maker", "model", "car"]},
            column_scores={},
            trace=[],
        )

        prompt = generation_prompt(linking.question, schema, linking)

        self.assertIn('"name" text', prompt)
        self.assertIn('"year" number', prompt)
        self.assertIn('"id" number [PK; REFERENCED BY model.maker_id]', prompt)
        self.assertIn('"maker_id" number [FK -> maker.id]', prompt)
        self.assertIn(
            "model.maker_id (FK) -> maker.id (referenced key)", prompt
        )
        self.assertNotIn("Shortest valid join paths", prompt)

    def test_executor_rejects_write_statement(self) -> None:
        result = SQLiteExecutorAgent().execute(
            "DELETE FROM singer", Path("does-not-matter.sqlite")
        )
        self.assertFalse(result.success)
        self.assertIn("SELECT", result.error or "")


if __name__ == "__main__":
    unittest.main()
