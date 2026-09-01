from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from schema_agents.evaluation import evaluate, evaluate_generated_sql
from schema_agents.models import DatabaseSchema, LinkingResult


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "concert_singer",
            "table_names_original": ["singer", "concert"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
                [1, "concert_ID"],
                [1, "Name"],
            ],
            "column_types": ["text", "number", "text", "number", "text"],
            "primary_keys": [1, 3],
            "foreign_keys": [],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_schema_precision_and_scoped_column_recall(self) -> None:
        result = LinkingResult(
            db_id="concert_singer",
            question="가수 이름",
            tables=["singer", "concert"],
            columns=["singer.Name", "concert.Name"],
            table_scores={},
            column_scores={},
            trace=[],
        )
        score = evaluate(result, self.schema, "SELECT Name FROM singer")
        self.assertEqual(score.table_recall, 1.0)
        self.assertEqual(score.table_precision, 0.5)
        self.assertEqual(score.column_recall, 1.0)
        self.assertEqual(score.column_precision, 0.5)
        self.assertTrue(score.strict_table_recall)
        self.assertTrue(score.strict_column_recall)

    def test_sql_exact_and_execution_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE singer (Name TEXT);"
                "INSERT INTO singer VALUES ('A'), ('B');"
            )
            connection.close()
            score = evaluate_generated_sql(
                'SELECT "Name" FROM "singer"', "select name from singer", database
            )
            self.assertTrue(score.normalized_exact_match)
            self.assertTrue(score.execution_match)

    def test_execution_match_respects_gold_order_by(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE singer (Name TEXT);"
                "INSERT INTO singer VALUES ('A'), ('B');"
            )
            connection.close()
            score = evaluate_generated_sql(
                "SELECT Name FROM singer ORDER BY Name DESC",
                "SELECT Name FROM singer ORDER BY Name ASC",
                database,
            )
            self.assertFalse(score.execution_match)


if __name__ == "__main__":
    unittest.main()
