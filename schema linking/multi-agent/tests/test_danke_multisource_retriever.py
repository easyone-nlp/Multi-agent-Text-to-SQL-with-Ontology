from __future__ import annotations

import json
import unittest
from pathlib import Path

from schema_agents.danke_multisource_retriever import MultiSourceDankeRetriever
from schema_agents.data import default_tables_path, load_schemas


WORKSPACE = Path(__file__).resolve().parents[3]
QUERY_CACHE = (
    WORKSPACE
    / "ontology"
    / "output"
    / "retriever_only_v1"
    / "shared_queries_0_99.json"
)


class MultiSourceDankeRetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cache = json.loads(QUERY_CACHE.read_text(encoding="utf-8"))
        cls.records = {item["index"]: item for item in cache["records"]}
        cls.schemas = load_schemas(default_tables_path("validation"))
        cls.retriever = MultiSourceDankeRetriever(
            {
                "knowledge_schema_dir": (
                    "ontology/versions/danke-spider-ko-v1.0.0-20260728"
                ),
                "semantic_table_budget": 3,
                "semantic_column_budget": 8,
                "bridge_table_budget": 2,
                "generic_property_table_gate": True,
            }
        )

    def test_original_question_rescues_english_retrieval_query(self) -> None:
        record = self.records[28]
        result = self.retriever.retrieve_multisource(
            record["question"],
            record["decomposition"],
            self.schemas[record["db_id"]],
        )
        self.assertTrue({"concert", "stadium"}.issubset(result.tables))
        self.assertEqual(result.mode, "danke_multisource")

    def test_generic_name_does_not_create_singer_table(self) -> None:
        record = self.records[28]
        result = self.retriever.retrieve_multisource(
            record["question"],
            record["decomposition"],
            self.schemas[record["db_id"]],
        )
        self.assertNotIn("singer", result.tables)
        raw = result.metadata["raw"]
        generic_targets = {
            item["target_class"]
            for item in raw["generic_property_only_matches"]
        }
        self.assertIn("singer", generic_targets)

    def test_bridge_budget_is_separate(self) -> None:
        record = self.records[99]
        result = self.retriever.retrieve_multisource(
            record["question"],
            record["decomposition"],
            self.schemas[record["db_id"]],
        )
        metadata = result.metadata
        self.assertEqual(len(metadata["semantic_tables"]), 3)
        self.assertEqual(metadata["bridge_tables"], ["model_list"])
        self.assertEqual(
            set(result.tables),
            {"car_makers", "car_names", "cars_data", "model_list"},
        )


if __name__ == "__main__":
    unittest.main()
