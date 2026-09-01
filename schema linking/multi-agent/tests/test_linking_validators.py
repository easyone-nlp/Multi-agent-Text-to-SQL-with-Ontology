from __future__ import annotations

import unittest

from schema_agents.agentic_agents import validate_join_output
from schema_agents.agentic_orchestrator import (
    DEFAULT_AGENTIC_CONFIG,
    _merge_retrieved_schemas,
)
from schema_agents.agentic_value_linker import (
    validate_probe_plan,
    validate_value_output,
)
from schema_agents.embedding_retriever import RetrievedSchema
from schema_agents.models import Column, DatabaseSchema


def sample_schema() -> DatabaseSchema:
    columns = [
        Column(0, "singer", "Country", "text"),
        Column(1, "singer", "Name", "text"),
        Column(2, "car_makers", "Id", "number"),
        Column(3, "car_makers", "Maker", "text"),
        Column(4, "model_list", "Maker", "number"),
        Column(5, "model_list", "Model", "text"),
        Column(6, "car_names", "Model", "text"),
        Column(7, "car_names", "MakeId", "number"),
        Column(8, "cars_data", "Id", "number"),
        Column(9, "cars_data", "Year", "number"),
    ]
    return DatabaseSchema(
        db_id="test",
        tables=["singer", "car_makers", "model_list", "car_names", "cars_data"],
        columns=columns,
        primary_keys={"car_makers.Id", "model_list.Model", "car_names.MakeId"},
        foreign_keys=[
            ("model_list.Maker", "car_makers.Id"),
            ("car_names.Model", "model_list.Model"),
            ("cars_data.Id", "car_names.MakeId"),
        ],
    )


class ValueValidatorTests(unittest.TestCase):
    def test_probe_plan_removes_korean_string_values(self) -> None:
        schema = sample_schema()
        retrieved = RetrievedSchema(
            tables=["singer"],
            columns=["singer.Country"],
            query="country filter",
        )
        plan = validate_probe_plan(
            {
                "conditions": [{
                    "condition_id": "f1",
                    "span": "프랑스",
                    "candidate_columns": ["singer.Country"],
                    "candidate_values": ["프랑스", "France", 1970],
                    "probe_mode": "exact",
                }]
            },
            schema,
            retrieved,
            4,
        )
        self.assertEqual(plan[0]["candidate_values"], ["France", 1970])

    def test_not_found_cannot_become_exact_match(self) -> None:
        schema = sample_schema()
        plan = [{
            "condition_id": "f1",
            "candidate_columns": ["singer.Country"],
            "probe_mode": "exact",
        }]
        result = validate_value_output(
            {
                "filters": [{
                    "condition_id": "f1",
                    "column": "singer.Country",
                    "operator": "=",
                    "value": "France",
                    "evidence": "exact DB match",
                }]
            },
            schema,
            plan,
            [{
                "condition_id": "f1",
                "column": "singer.Country",
                "probe_mode": "exact",
                "status": "not_found",
                "matched_values": [],
            }],
        )
        self.assertEqual(result["filters"], [])
        self.assertTrue(result["unresolved"])

    def test_matched_value_is_canonicalized_to_stored_value(self) -> None:
        schema = sample_schema()
        plan = [{
            "condition_id": "f1",
            "candidate_columns": ["singer.Country"],
            "probe_mode": "exact",
        }]
        result = validate_value_output(
            {
                "filters": [{
                    "condition_id": "f1",
                    "column": "singer.Country",
                    "operator": "=",
                    "value": "프랑스",
                }]
            },
            schema,
            plan,
            [{
                "condition_id": "f1",
                "column": "singer.Country",
                "probe_mode": "exact",
                "status": "matched",
                "matched_values": ["France"],
            }],
        )
        self.assertEqual(result["filters"][0]["value"], "France")
        self.assertEqual(result["filters"][0]["evidence"], "validated DB match")


class JoinValidatorTests(unittest.TestCase):
    def test_only_declared_complete_fk_path_is_accepted(self) -> None:
        schema = sample_schema()
        result = validate_join_output(
            {
                "endpoint_tables": ["car_makers", "cars_data"],
                "tables": ["car_makers", "model_list", "car_names", "cars_data"],
                "joins": [
                    {"left": "model_list.Maker", "right": "car_makers.Id"},
                    {"left": "car_names.Model", "right": "model_list.Model"},
                    {"left": "cars_data.Id", "right": "car_names.MakeId"},
                    {"left": "car_makers.Maker", "right": "cars_data.Id"},
                ],
            },
            schema,
            allowed_endpoint_tables=["car_makers", "cars_data"],
        )
        self.assertEqual(len(result["joins"]), 3)
        self.assertIn("model_list", result["tables"])
        self.assertTrue(any("not declared" in item for item in result["unresolved"]))

    def test_incomplete_path_is_removed(self) -> None:
        schema = sample_schema()
        result = validate_join_output(
            {
                "endpoint_tables": ["car_makers", "cars_data"],
                "joins": [
                    {"left": "model_list.Maker", "right": "car_makers.Id"},
                    {"left": "car_names.Model", "right": "model_list.Model"},
                ],
            },
            schema,
            allowed_endpoint_tables=["car_makers", "cars_data"],
        )
        self.assertEqual(result["joins"], [])
        self.assertTrue(any("incomplete" in item for item in result["unresolved"]))

    def test_inferred_join_is_rejected(self) -> None:
        schema = sample_schema()
        result = validate_join_output(
            {
                "endpoint_tables": ["car_makers", "cars_data"],
                "joins": [{
                    "left": "car_makers.Maker",
                    "right": "cars_data.Id",
                    "inferred": True,
                }],
            },
            schema,
            allowed_endpoint_tables=["car_makers", "cars_data"],
        )
        self.assertEqual(result["joins"], [])
        self.assertTrue(any("inferred" in item for item in result["unresolved"]))


class RescueRetrievalTests(unittest.TestCase):
    def test_rescue_union_expands_value_candidates(self) -> None:
        initial = RetrievedSchema(
            tables=["singer"], columns=["singer.Name"], query="initial"
        )
        rescue = RetrievedSchema(
            tables=["singer"], columns=["singer.Country"], query="filter"
        )
        merged = _merge_retrieved_schemas(initial, [rescue])
        self.assertEqual(merged.columns, ["singer.Name", "singer.Country"])
        self.assertEqual(merged.mode, "value_rescue_union")

    def test_default_mode_skips_schema_linker(self) -> None:
        self.assertEqual(
            DEFAULT_AGENTIC_CONFIG["schema_linker_mode"], "embedding_only"
        )


if __name__ == "__main__":
    unittest.main()
