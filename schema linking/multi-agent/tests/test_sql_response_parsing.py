from __future__ import annotations

import unittest

from schema_agents.sql_agents import extract_sql


class SQLResponseParsingTest(unittest.TestCase):
    def test_extracts_first_statement_from_multiple_candidates(self) -> None:
        response = "SELECT Name FROM singer; SELECT * FROM concert;"
        self.assertEqual(extract_sql(response), "SELECT Name FROM singer")

    def test_keeps_semicolon_inside_string_literal(self) -> None:
        response = "SELECT Name FROM singer WHERE Bio = 'a;b'; explanation"
        self.assertEqual(
            extract_sql(response),
            "SELECT Name FROM singer WHERE Bio = 'a;b'",
        )

    def test_extracts_sql_from_fence_with_trailing_prose(self) -> None:
        response = "```sql\nSELECT COUNT(*) FROM singer;\n```\n설명"
        self.assertEqual(extract_sql(response), "SELECT COUNT(*) FROM singer")

    def test_rejects_response_without_query(self) -> None:
        with self.assertRaises(ValueError):
            extract_sql("SQL을 생성할 수 없습니다.")


if __name__ == "__main__":
    unittest.main()
