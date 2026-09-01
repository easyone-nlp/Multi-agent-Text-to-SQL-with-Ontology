from __future__ import annotations

from .evaluation import extract_gold_links
from .models import DatabaseSchema, Example, LinkingResult


def build_gold_schema_result(
    example: Example,
    schema: DatabaseSchema,
) -> LinkingResult:
    """Build an oracle linking package from gold SQL identifiers only.

    The gold SQL text, literals, operators, clauses, and query structure are not
    copied into the package consumed by the SQL generator.
    """

    if not example.gold_sql:
        raise ValueError("gold schema 모드는 gold SQL이 있는 dataset 예제가 필요합니다.")

    gold_tables, gold_columns = extract_gold_links(
        schema,
        example.gold_sql,
        require_sqlglot=True,
    )
    if not gold_tables:
        raise ValueError(
            f"gold SQL에서 유효한 table을 추출하지 못했습니다: db_id={example.db_id}"
        )

    ordered_tables = [table for table in schema.tables if table in gold_tables]
    ordered_columns = [
        column.key for column in schema.columns if column.key in gold_columns
    ]
    return LinkingResult(
        db_id=example.db_id,
        question=example.question,
        tables=ordered_tables,
        columns=ordered_columns,
        table_scores={},
        column_scores={},
        trace=[],
    )
