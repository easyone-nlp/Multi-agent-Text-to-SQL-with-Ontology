from __future__ import annotations

import json
import unittest
from pathlib import Path

from schema_agents.hybrid_danke_embedding_retriever import (
    _is_bridge_like_table,
    _strong_embedding_table_reasons,
)
from schema_agents.agentic_orchestrator import (
    _merge_hybrid_structural_evidence,
    retriever_only_schema_output,
)
from schema_agents.embedding_retriever import RetrievedSchema
from schema_agents.models import Column, DatabaseSchema


PROJECT = Path(__file__).resolve().parents[1]


class HybridRetrieverPolicyTest(unittest.TestCase):
    def test_production_default_is_frozen_hybrid_steiner_on(self) -> None:
        config = json.loads(
            (PROJECT / "config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["schema_linking"]["schema_linker_mode"],
            "danke_embedding_hybrid",
        )
        self.assertTrue(
            config["schema_linking"]["danke"]["use_steiner"]
        )

    def test_only_association_table_is_bridge_like(self) -> None:
        schema = DatabaseSchema(
            db_id="test",
            tables=["student", "pet", "has_pet"],
            columns=[
                Column(0, "student", "id", "number"),
                Column(1, "student", "name", "text"),
                Column(2, "pet", "id", "number"),
                Column(3, "pet", "type", "text"),
                Column(4, "has_pet", "student_id", "number"),
                Column(5, "has_pet", "pet_id", "number"),
            ],
            primary_keys={"student.id", "pet.id"},
            foreign_keys=[
                ("has_pet.student_id", "student.id"),
                ("has_pet.pet_id", "pet.id"),
            ],
        )
        self.assertTrue(_is_bridge_like_table(schema, "has_pet"))
        self.assertFalse(_is_bridge_like_table(schema, "student"))
        self.assertFalse(_is_bridge_like_table(schema, "pet"))

    def test_generic_atomic_source_cannot_create_embedding_evidence(self) -> None:
        rankings = [
            {
                "source_id": "question",
                "source_type": "original_question",
                "source_text": "학생의 이름",
                "ranking": [
                    {"key": "student", "score": 0.8},
                    {"key": "pet", "score": 0.7},
                ],
            },
            {
                "source_id": "retrieval_query",
                "source_type": "retrieval_query",
                "source_text": "학생 이름 검색",
                "ranking": [
                    {"key": "student", "score": 0.8},
                    {"key": "pet", "score": 0.7},
                ],
            },
            {
                "source_id": "decomposition.outputs[0].span",
                "source_type": "plan_output_span",
                "source_text": "name",
                "ranking": [
                    {"key": "unrelated", "score": 0.9},
                    {"key": "student", "score": 0.8},
                ],
            },
        ]
        reasons = _strong_embedding_table_reasons(rankings)
        self.assertIn("original_and_retrieval_top2", reasons["student"])
        self.assertNotIn("unrelated", reasons)

    def test_bridge_remains_outside_downstream_semantic_candidates(self) -> None:
        schema = DatabaseSchema(
            db_id="test",
            tables=["student", "pet", "has_pet"],
            columns=[
                Column(0, "student", "id", "number"),
                Column(1, "pet", "id", "number"),
                Column(2, "has_pet", "student_id", "number"),
                Column(3, "has_pet", "pet_id", "number"),
            ],
            primary_keys={"student.id", "pet.id"},
            foreign_keys=[
                ("has_pet.student_id", "student.id"),
                ("has_pet.pet_id", "pet.id"),
            ],
        )
        retrieved = RetrievedSchema(
            tables=["student", "pet", "has_pet"],
            columns=[
                "student.id",
                "pet.id",
                "has_pet.student_id",
                "has_pet.pet_id",
            ],
            query="student pets",
            metadata={
                "semantic_tables": ["student", "pet"],
                "semantic_columns": ["student.id", "pet.id"],
                "bridge_tables": ["has_pet"],
                "join_key_columns": [
                    "has_pet.student_id",
                    "student.id",
                    "has_pet.pet_id",
                    "pet.id",
                ],
                "join_edges": [
                    {
                        "proposed_left": "has_pet.student_id",
                        "proposed_right": "student.id",
                        "object_property": "has_pet_student_fk",
                        "validated": True,
                        "weight": 1.0,
                    },
                    {
                        "proposed_left": "has_pet.pet_id",
                        "proposed_right": "pet.id",
                        "object_property": "has_pet_pet_fk",
                        "validated": True,
                        "weight": 1.0,
                    },
                ],
            },
        )
        semantic = retriever_only_schema_output(
            retrieved, "danke_embedding_hybrid"
        )
        self.assertEqual(semantic["selected_tables"], ["student", "pet"])
        self.assertNotIn("has_pet", semantic["selected_tables"])
        structural = _merge_hybrid_structural_evidence(
            {"tables": [], "columns": [], "joins": [], "unresolved": []},
            retrieved,
            schema,
        )
        self.assertEqual(structural["endpoint_tables"], ["student", "pet"])
        self.assertIn("has_pet", structural["tables"])
        self.assertEqual(len(structural["joins"]), 2)


if __name__ == "__main__":
    unittest.main()
