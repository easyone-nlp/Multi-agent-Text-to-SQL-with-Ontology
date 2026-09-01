from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from schema_agents.models import AgentProposal, DatabaseSchema
from schema_agents.value_grounding_agent import (
    DBValueGroundingAgent,
    parse_value_conditions,
)


class FakeChatModel:
    model = "fake-qwen"

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


class DBValueGroundingAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "flight",
            "table_names_original": ["airports"],
            "column_names_original": [
                [-1, "*"],
                [0, "AirportCode"],
                [0, "AirportName"],
                [0, "City"],
            ],
            "column_types": ["text", "text", "text", "text"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_exact_probe_keeps_only_column_containing_value(self) -> None:
        response = (
            '{"conditions":[{"mention":"AKO","operator":"=",'
            '"values":["AKO"],"candidate_columns":'
            '["airports.AirportCode","airports.AirportName"],'
            '"probe_mode":"exact"}]}'
        )
        model = FakeChatModel(response)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "flight.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE airports (AirportCode TEXT, AirportName TEXT, City TEXT);"
                "INSERT INTO airports VALUES ('AKO', 'Colorado Plains', 'Akron');"
            )
            connection.close()

            proposal, evidence = DBValueGroundingAgent(model).propose(
                "코드 'AKO'인 공항", self.schema, [AgentProposal("llm_schema_scout")],
                database,
            )

        self.assertIn("airports.AirportCode", proposal.column_scores)
        self.assertNotIn("airports.AirportName", proposal.column_scores)
        self.assertEqual(evidence[0].matched_values, ["AKO"])
        self.assertIn("Do not write SQL", model.system_prompt)

    def test_categorical_probe_returns_observed_domain_and_mapping(self) -> None:
        raw = {
            "db_id": "world",
            "table_names_original": ["countrylanguage"],
            "column_names_original": [[-1, "*"], [0, "IsOfficial"]],
            "column_types": ["text", "text"],
            "primary_keys": [],
            "foreign_keys": [],
        }
        schema = DatabaseSchema.from_spider(raw)
        model = FakeChatModel(
            '{"conditions":[{"mention":"공식 언어","operator":"=",'
            '"values":["T"],"candidate_columns":'
            '["countrylanguage.IsOfficial"],"probe_mode":"categorical"}]}'
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "world.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE countrylanguage (IsOfficial TEXT);"
                "INSERT INTO countrylanguage VALUES ('T'), ('F'), ('F');"
            )
            connection.close()

            proposal, evidence = DBValueGroundingAgent(model).propose(
                "공식 언어 수", schema, [], database
            )

        self.assertIn("countrylanguage.IsOfficial", proposal.column_scores)
        self.assertEqual(evidence[0].matched_values, ["T"])
        self.assertEqual(set(evidence[0].observed_values), {"T", "F"})

    def test_parser_rejects_unknown_schema_candidates(self) -> None:
        response = (
            '{"conditions":[{"mention":"AKO","operator":"=",'
            '"values":["AKO"],"candidate_columns":'
            '["airports.AirportCode","fake.Code"],"probe_mode":"exact"}]}'
        )
        conditions, ignored = parse_value_conditions(response, self.schema)
        self.assertEqual(
            conditions[0].candidate_columns, ["airports.AirportCode"]
        )
        self.assertEqual(ignored, ["fake.Code"])


if __name__ == "__main__":
    unittest.main()
