from __future__ import annotations

import unittest

from schema_agents.models import AgentProposal, DatabaseSchema
from schema_agents.schema_llm_agent import (
    LLMSchemaCriticAgent,
    LLMSchemaScoutAgent,
    parse_schema_selection,
)


class FakeChatModel:
    model = "fake-qwen"

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


class LLMSchemaScoutTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "world_1",
            "table_names_original": ["country", "countrylanguage"],
            "column_names_original": [
                [-1, "*"],
                [0, "Code"],
                [0, "Name"],
                [1, "CountryCode"],
                [1, "Language"],
            ],
            "column_types": ["text", "text", "text", "text", "text"],
            "primary_keys": [1],
            "foreign_keys": [[3, 1]],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_parser_accepts_fenced_json_and_normalizes_case(self) -> None:
        response = """<think>reason</think>```json
        {"tables": ["COUNTRYLANGUAGE"],
         "columns": [["countrylanguage", "language"],
                     {"table": "country", "column": "Code"}]}
        ```"""
        parsed = parse_schema_selection(response, self.schema)
        self.assertEqual(parsed.tables, ["countrylanguage", "country"])
        self.assertEqual(
            parsed.columns, ["countrylanguage.Language", "country.Code"]
        )

    def test_parser_allows_empty_critic_selection(self) -> None:
        parsed = parse_schema_selection(
            "{\"tables\": [], \"columns\": []}", self.schema, allow_empty=True
        )
        self.assertEqual((parsed.tables, parsed.columns), ([], []))

    def test_agent_returns_validated_scores(self) -> None:
        model = FakeChatModel(
            '{"tables": ["country"], "columns": ["country.Name", "fake.bad"]}'
        )
        proposal = LLMSchemaScoutAgent(model).propose(
            "국가 이름", self.schema, []
        )
        self.assertEqual(proposal.table_scores["country"], 4.0)
        self.assertEqual(proposal.column_scores["country.Name"], 4.0)
        self.assertNotIn("fake.bad", proposal.column_scores)
        self.assertIn("Korean question: 국가 이름", model.user_prompt)

    def test_invalid_response_falls_back_without_crashing(self) -> None:
        proposal = LLMSchemaScoutAgent(FakeChatModel("not json")).propose(
            "국가 이름", self.schema, []
        )
        self.assertEqual(proposal.table_scores, {})
        self.assertIn("fallback", proposal.reasons[0])


    def test_critic_returns_only_missing_additions(self) -> None:
        model = FakeChatModel(
            "{\"tables\": [\"country\", \"countrylanguage\"], "
            "\"columns\": [\"country.Name\", \"countrylanguage.Language\"]}"
        )
        scout = AgentProposal(
            "llm_schema_scout",
            table_scores={"country": 4.0},
            column_scores={"country.Name": 4.0},
        )
        proposal = LLMSchemaCriticAgent(model).propose(
            "언어와 국가 이름", self.schema, [scout]
        )
        self.assertEqual(proposal.table_scores, {"countrylanguage": 4.5})
        self.assertEqual(
            proposal.column_scores, {"countrylanguage.Language": 4.5}
        )
        self.assertIn("Current semantic tables", model.user_prompt)


if __name__ == "__main__":
    unittest.main()
