from __future__ import annotations

import unittest

from schema_agents.models import DatabaseSchema
from scripts.analyze_gold_schema_failures import (
    ExecutionSnapshot,
    compare_structures,
    extract_structure,
)


class GoldSchemaFailureAnalysisTest(unittest.TestCase):
    def compare(self, schema, gold_sql, predicted_sql):
        gold = extract_structure(gold_sql, schema)
        predicted = extract_structure(predicted_sql, schema)
        execution = ExecutionSnapshot(True, [], [])
        _, errors, _ = compare_structures(
            gold,
            predicted,
            None,
            None,
            execution,
            execution,
            False,
        )
        return errors

    def test_double_quoted_spider_value_is_not_a_join_column(self):
        schema = self.schema(
            ["AIRLINES"],
            [(0, "Airline"), (0, "Country")],
        )
        errors = self.compare(
            schema,
            'SELECT Country FROM AIRLINES WHERE Airline = "JetBlue Airways"',
            "SELECT Country FROM AIRLINES WHERE Airline = 'Jetblue Airways'",
        )
        self.assertIn("VALUE_GROUNDING", errors)
        self.assertNotIn("JOIN_CONDITION", errors)
        self.assertNotIn("EXTRA_FILTER", errors)

    def test_reversed_equality_join_has_the_same_edge(self):
        schema = self.schema(
            ["Student", "Has_Pet"],
            [(0, "StuID"), (0, "Fname"), (1, "StuID")],
        )
        errors = self.compare(
            schema,
            "SELECT Student.Fname FROM Student JOIN Has_Pet "
            "ON Student.StuID = Has_Pet.StuID",
            "SELECT Student.Fname FROM Student JOIN Has_Pet "
            "ON Has_Pet.StuID = Student.StuID",
        )
        self.assertNotIn("JOIN_CONDITION", errors)

    def test_same_scalar_subquery_with_inner_operator_error(self):
        schema = self.schema(
            ["museum"],
            [(0, "Name"), (0, "Num_of_Staff"), (0, "Open_Year")],
        )
        errors = self.compare(
            schema,
            "SELECT Name FROM museum WHERE Num_of_Staff > "
            "(SELECT MIN(Num_of_Staff) FROM museum WHERE Open_Year > 2010)",
            "SELECT Name FROM museum WHERE Num_of_Staff > "
            "(SELECT MIN(Num_of_Staff) FROM museum WHERE Open_Year >= 2010)",
        )
        self.assertIn("WRONG_FILTER_OPERATOR", errors)
        self.assertNotIn("SUBQUERY", errors)

    def test_between_matches_equivalent_inclusive_bounds(self):
        schema = self.schema(
            ["cars_data"],
            [(0, "Year"), (0, "Weight")],
        )
        errors = self.compare(
            schema,
            "SELECT Year FROM cars_data WHERE Weight BETWEEN 3000 AND 4000",
            "SELECT Year FROM cars_data WHERE Weight >= 3000 AND Weight <= 4000",
        )
        self.assertNotIn("MISSING_FILTER", errors)
        self.assertNotIn("EXTRA_FILTER", errors)
        self.assertNotIn("WRONG_FILTER_OPERATOR", errors)

    @staticmethod
    def schema(tables, columns):
        raw_columns = [[-1, "*"]]
        raw_columns.extend([[table_index, name] for table_index, name in columns])
        return DatabaseSchema.from_spider({
            "db_id": "test",
            "table_names_original": tables,
            "column_names_original": raw_columns,
            "column_types": ["text"] * len(raw_columns),
            "primary_keys": [],
            "foreign_keys": [],
        })


if __name__ == "__main__":
    unittest.main()
