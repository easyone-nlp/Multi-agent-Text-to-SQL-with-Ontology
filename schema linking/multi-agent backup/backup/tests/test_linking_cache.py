from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from main import LinkingResultCache
from schema_agents.models import DatabaseSchema, Example


class LinkingResultCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "concert_singer",
            "table_names_original": ["singer"],
            "column_names_original": [
                [-1, "*"],
                [0, "Singer_ID"],
                [0, "Name"],
            ],
            "column_types": ["text", "number", "text"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
        self.schema = DatabaseSchema.from_spider(raw)
        self.example = Example("concert_singer", "가수 이름", "SELECT Name FROM singer")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _cache(self, record: dict[str, object]) -> LinkingResultCache:
        path = Path(self.temp_dir.name) / "linking.json"
        path.write_text(
            json.dumps({"summary": {}, "results": [record]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return LinkingResultCache(path)

    def _record(self) -> dict[str, object]:
        return {
            "index": 7,
            "db_id": "concert_singer",
            "question": "가수 이름",
            "selected_tables": ["singer"],
            "selected_columns": ["singer.Name"],
            "table_scores": {"singer": 6.0},
            "column_scores": {"singer.Name": 6.0},
            "schema_linking_model": {
                "scout_success": True,
                "critic_success": True,
            },
        }

    def test_resolves_saved_linking_result(self) -> None:
        result, item = self._cache(self._record()).resolve(
            7, self.example, self.schema
        )
        self.assertEqual(result.tables, ["singer"])
        self.assertEqual(result.columns, ["singer.Name"])
        self.assertEqual(result.table_scores, {"singer": 6.0})
        self.assertEqual(item["index"], 7)

    def test_restores_value_grounding_evidence(self) -> None:
        record = self._record()
        record["value_grounding"] = [
            {
                "mention": "A",
                "column": "singer.Name",
                "operator": "=",
                "probe_mode": "exact",
                "candidate_values": ["A"],
                "matched_values": ["A"],
                "observed_values": [],
                "confidence": 1.0,
            }
        ]

        result, _ = self._cache(record).resolve(
            7, self.example, self.schema
        )

        self.assertEqual(result.value_evidence[0].column, "singer.Name")
        self.assertEqual(result.value_evidence[0].matched_values, ["A"])

    def test_falls_back_to_db_and_question_when_index_differs(self) -> None:
        result, item = self._cache(self._record()).resolve(
            99, self.example, self.schema
        )
        self.assertEqual(result.db_id, "concert_singer")
        self.assertEqual(item["index"], 7)

    def test_rejects_schema_mismatch(self) -> None:
        record = self._record()
        record["selected_columns"] = ["singer.Unknown"]
        with self.assertRaisesRegex(SystemExit, "현재 Spider schema가 일치하지 않습니다"):
            self._cache(record).resolve(7, self.example, self.schema)


if __name__ == "__main__":
    unittest.main()
