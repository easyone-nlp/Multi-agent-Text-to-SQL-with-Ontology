from __future__ import annotations

import unittest

from schema_agents.gold_schema import build_gold_schema_result
from schema_agents.models import DatabaseSchema, Example


class GoldSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = {
            "db_id": "car_1",
            "table_names_original": ["car_makers", "model_list"],
            "column_names_original": [
                [-1, "*"],
                [0, "Id"],
                [0, "Maker"],
                [1, "Model"],
                [1, "Maker"],
            ],
            "column_types": ["text", "number", "text", "text", "number"],
            "primary_keys": [1, 3],
            "foreign_keys": [[4, 1]],
        }
        self.schema = DatabaseSchema.from_spider(raw)

    def test_extracts_alias_qualified_gold_identifiers(self) -> None:
        example = Example(
            db_id="car_1",
            question="모델 X를 만드는 제조사는?",
            gold_sql=(
                "SELECT T1.Maker FROM car_makers AS T1 "
                "JOIN model_list AS T2 ON T2.Maker = T1.Id "
                "WHERE T2.Model = 'X'"
            ),
        )

        result = build_gold_schema_result(example, self.schema)

        self.assertEqual(result.tables, ["car_makers", "model_list"])
        self.assertEqual(
            result.columns,
            [
                "car_makers.Id",
                "car_makers.Maker",
                "model_list.Model",
                "model_list.Maker",
            ],
        )
        self.assertEqual(result.query_decomposition, {})
        self.assertEqual(result.grounded_filters, [])
        self.assertEqual(result.joins, [])

    def test_extracts_column_inside_having_aggregate(self) -> None:
        raw = {
            "db_id": "world_1",
            "table_names_original": ["country"],
            "column_names_original": [
                [-1, "*"],
                [0, "Population"],
                [0, "GovernmentForm"],
                [0, "LifeExpectancy"],
            ],
            "column_types": ["text", "number", "text", "number"],
            "primary_keys": [],
            "foreign_keys": [],
        }
        schema = DatabaseSchema.from_spider(raw)
        result = build_gold_schema_result(
            Example(
                db_id="world_1",
                question="평균 기대수명이 72보다 큰 정부 형태별 인구 합계",
                gold_sql=(
                    "SELECT SUM(Population), GovernmentForm FROM country "
                    "GROUP BY GovernmentForm HAVING AVG(LifeExpectancy) > 72"
                ),
            ),
            schema,
        )
        self.assertEqual(
            result.columns,
            [
                "country.Population",
                "country.GovernmentForm",
                "country.LifeExpectancy",
            ],
        )

    def test_requires_gold_sql(self) -> None:
        with self.assertRaises(ValueError):
            build_gold_schema_result(
                Example(db_id="car_1", question="질문"),
                self.schema,
            )


if __name__ == "__main__":
    unittest.main()
