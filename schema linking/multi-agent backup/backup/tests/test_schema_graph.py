from __future__ import annotations

import unittest

from schema_agents.agents import ReviewerAgent, SchemaGraphAgent
from schema_agents.models import AgentProposal, DatabaseSchema


class SchemaGraphPathTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "chain",
            "table_names_original": ["A", "B", "C"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"],
                [0, "name"],
                [1, "a_id"],
                [1, "c_id"],
                [2, "id"],
                [2, "value"],
            ],
            "column_types": [
                "text",
                "number",
                "text",
                "number",
                "number",
                "number",
                "text",
            ],
            "primary_keys": [1, 5],
            "foreign_keys": [[3, 1], [4, 5]],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_shortest_path_adds_bridge_and_join_endpoints(self) -> None:
        scout = AgentProposal(
            "llm_schema_scout",
            table_scores={"A": 4.0, "C": 4.0},
            column_scores={"A.name": 4.0, "C.value": 4.0},
        )
        graph = SchemaGraphAgent().propose("question", self.schema, [scout])

        self.assertEqual(graph.table_scores, {"B": 10.0})
        self.assertEqual(
            set(graph.column_scores),
            {"A.id", "B.a_id", "B.c_id", "C.id"},
        )
        self.assertIn("A→B→C", graph.reasons[1])

        reviewer = ReviewerAgent(
            weights={"llm_schema_scout": 1.5, "schema_graph": 0.45},
            max_tables=4,
            max_columns=16,
            relative_table_threshold=0.35,
            relative_column_threshold=0.4,
            include_primary_keys=True,
        )
        tables, columns, *_ = reviewer.decide(self.schema, [scout, graph])
        self.assertEqual(set(tables), {"A", "B", "C"})
        self.assertEqual(
            set(columns),
            {"A.name", "C.value", "A.id", "B.a_id", "B.c_id", "C.id"},
        )

    def test_single_semantic_anchor_does_not_expand_neighbors(self) -> None:
        scout = AgentProposal(
            "llm_schema_scout", table_scores={"A": 4.0}
        )
        graph = SchemaGraphAgent().propose("question", self.schema, [scout])
        self.assertEqual(graph.table_scores, {})
        self.assertEqual(graph.column_scores, {})

    def test_reviewer_keeps_verified_value_column_as_trusted_candidate(self) -> None:
        scout = AgentProposal(
            "llm_schema_scout",
            table_scores={"A": 4.0},
            column_scores={"A.name": 4.0},
        )
        value = AgentProposal(
            "db_value_grounder",
            table_scores={"C": 6.0},
            column_scores={"C.value": 6.0},
        )
        graph = SchemaGraphAgent().propose(
            "value filter", self.schema, [scout, value]
        )
        reviewer = ReviewerAgent(
            weights={
                "llm_schema_scout": 1.5,
                "db_value_grounder": 1.0,
                "schema_graph": 0.45,
            },
            max_tables=4,
            max_columns=16,
            relative_table_threshold=0.35,
            relative_column_threshold=0.4,
            include_primary_keys=True,
        )
        tables, columns, *_ = reviewer.decide(
            self.schema, [scout, value, graph]
        )
        self.assertEqual(set(tables), {"A", "B", "C"})
        self.assertIn("C.value", columns)


if __name__ == "__main__":
    unittest.main()
