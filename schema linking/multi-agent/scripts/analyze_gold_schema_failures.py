#!/usr/bin/env python3
"""Deterministic failure analysis for gold-schema Text-to-SQL results.

This script does not call a language model. It parses gold and predicted SQL with
sqlglot, re-executes both queries against read-only SQLite databases, compares
clause-level structures, and assigns a deterministic error taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import sqlglot
from sqlglot import exp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schema_agents.data import (  # noqa: E402
    database_path,
    default_database_root,
    default_tables_path,
    load_schemas,
)
from schema_agents.evaluation import extract_gold_links  # noqa: E402
from schema_agents.models import DatabaseSchema  # noqa: E402
from schema_agents.sql_agents import safe_select_error  # noqa: E402


TAXONOMY = [
    "VALUE_GROUNDING",
    "WRONG_FILTER_COLUMN",
    "WRONG_FILTER_OPERATOR",
    "MISSING_FILTER",
    "EXTRA_FILTER",
    "AGGREGATION",
    "DISTINCT",
    "GROUP_BY",
    "HAVING",
    "ORDER_BY",
    "LIMIT_OR_SUPERLATIVE",
    "JOIN_PATH",
    "JOIN_CONDITION",
    "MISSING_TABLE_OR_COLUMN",
    "SUBQUERY",
    "SET_OPERATION",
    "NEGATION",
    "SELECT_TARGET",
    "OTHER_SEMANTIC_ERROR",
    "EXECUTION_ERROR",
    "UNKNOWN",
]

# Primary error is a deterministic heuristic, not a causal model judgment.
# Higher-level query-shape errors precede local clause differences.
ERROR_FAMILIES = {
    "VALUE_AND_FILTER": [
        "VALUE_GROUNDING",
        "WRONG_FILTER_COLUMN",
        "WRONG_FILTER_OPERATOR",
        "MISSING_FILTER",
        "EXTRA_FILTER",
    ],
    "AGGREGATION_AND_GROUPING": [
        "AGGREGATION", "DISTINCT", "GROUP_BY", "HAVING"
    ],
    "JOIN": ["JOIN_PATH", "JOIN_CONDITION"],
    "QUERY_COMPOSITION": ["SUBQUERY", "SET_OPERATION", "NEGATION"],
    "PROJECTION_ORDER_LIMIT": [
        "SELECT_TARGET", "ORDER_BY", "LIMIT_OR_SUPERLATIVE"
    ],
    "SCHEMA_IDENTIFIER": ["MISSING_TABLE_OR_COLUMN"],
    "EXECUTION": ["EXECUTION_ERROR"],
    "UNRESOLVED": ["OTHER_SEMANTIC_ERROR", "UNKNOWN"],
}

PRIMARY_PRIORITY = [
    "EXECUTION_ERROR",
    "SET_OPERATION",
    "SUBQUERY",
    "JOIN_PATH",
    "JOIN_CONDITION",
    "NEGATION",
    "VALUE_GROUNDING",
    "WRONG_FILTER_COLUMN",
    "WRONG_FILTER_OPERATOR",
    "MISSING_FILTER",
    "EXTRA_FILTER",
    "HAVING",
    "GROUP_BY",
    "LIMIT_OR_SUPERLATIVE",
    "AGGREGATION",
    "DISTINCT",
    "ORDER_BY",
    "SELECT_TARGET",
    "MISSING_TABLE_OR_COLUMN",
    "OTHER_SEMANTIC_ERROR",
    "UNKNOWN",
]

COMPARISON_TYPES = tuple(
    cls
    for cls in (
        getattr(exp, "EQ", None),
        getattr(exp, "NEQ", None),
        getattr(exp, "GT", None),
        getattr(exp, "GTE", None),
        getattr(exp, "LT", None),
        getattr(exp, "LTE", None),
        getattr(exp, "Like", None),
        getattr(exp, "ILike", None),
        getattr(exp, "In", None),
        getattr(exp, "Between", None),
        getattr(exp, "Is", None),
        getattr(exp, "Exists", None),
    )
    if cls is not None
)

SET_OPERATION_TYPES = tuple(
    cls
    for cls in (
        getattr(exp, "Union", None),
        getattr(exp, "Intersect", None),
        getattr(exp, "Except", None),
    )
    if cls is not None
)


@dataclass
class ExecutionSnapshot:
    success: bool
    columns: list[str]
    rows: list[tuple[Any, ...]]
    error: str | None = None
    truncated: bool = False
    max_rows: int = 10_000

    def public(self, preview_rows: int, ordered: bool) -> dict[str, Any]:
        canonical = [_canonical_row(row) for row in self.rows]
        digest_rows = canonical if ordered else sorted(
            canonical, key=lambda row: json.dumps(_json_value(row), ensure_ascii=False)
        )
        digest_payload = json.dumps(
            _json_value(digest_rows),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result: dict[str, Any] = {
            "success": self.success,
            "columns": self.columns,
            "row_count": None if self.truncated else len(self.rows),
            "row_count_lower_bound": self.max_rows + 1 if self.truncated else None,
            "rows_preview": _json_value(self.rows[:preview_rows]),
            "preview_limit": preview_rows,
            "truncated": self.truncated,
            "result_sha256": hashlib.sha256(digest_payload).hexdigest(),
            "error": self.error,
        }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze gold-schema Text-to-SQL EX failures without an LLM"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs/qwen_gold_schema_text_to_sql_all.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs/gold_schema_failure_analysis.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/gold_schema_failure_summary.csv",
    )
    parser.add_argument("--split", choices=("validation", "dev", "train"), default="validation")
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--max-execution-rows", type=int, default=10_000)
    parser.add_argument("--preview-rows", type=int, default=20)
    parser.add_argument("--representatives", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_execution_rows < 1 or args.preview_rows < 0 or args.representatives < 1:
        raise SystemExit("row limits and representatives must be positive")
    payload = _load_result_file(args.input)
    tables_path = args.tables or default_tables_path(args.split)
    database_root = args.database_root or default_database_root(args.split)
    schemas = load_schemas(tables_path)

    analyses: list[dict[str, Any]] = []
    skipped_successes = 0
    for record in payload["results"]:
        sql_evaluation = record.get("sql_evaluation", {})
        if sql_evaluation.get("execution_match") is True:
            skipped_successes += 1
            continue
        db_id = str(record.get("db_id", ""))
        schema = schemas.get(db_id)
        if schema is None:
            analyses.append(_unavailable_schema_analysis(record, db_id))
            continue
        db_file = database_path(database_root, db_id)
        analyses.append(
            analyze_record(
                record,
                schema,
                db_file,
                max_rows=args.max_execution_rows,
                preview_rows=args.preview_rows,
            )
        )

    summary = build_summary(
        analyses,
        total_examples=len(payload["results"]),
        skipped_successes=skipped_successes,
        source_summary=payload.get("summary", {}),
        representative_limit=args.representatives,
    )
    output = {
        "source": {
            "input": str(args.input),
            "tables": str(tables_path),
            "database_root": str(database_root),
            "classifier": "deterministic_sqlglot_ast_v1",
            "uses_language_model": False,
        },
        "methodology": {
            "failure_selection": "stored sql_evaluation.execution_match is not true",
            "execution_result_storage": (
                "row count, deterministic SHA-256, and bounded row preview; full result rows "
                "are compared in memory up to max_execution_rows"
            ),
            "primary_error": (
                "first detected error in the documented PRIMARY_PRIORITY order; "
                "secondary_errors retain every other detected taxonomy label"
            ),
            "unknown_policy": (
                "UNKNOWN is used for parse failures, unavailable schema/gold execution, "
                "or cases whose deterministic evidence is insufficient"
            ),
            "primary_priority": PRIMARY_PRIORITY,
            "taxonomy": TAXONOMY,
            "max_execution_rows": args.max_execution_rows,
            "preview_rows": args.preview_rows,
        },
        "summary": summary,
        "failures": analyses,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_summary_csv(args.output_csv, summary)
    print_frequency_summary(summary)
    print(f"JSON: {args.output_json}")
    print(f"CSV:  {args.output_csv}")


def _load_result_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"input JSON not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SystemExit("input JSON must contain a results list")
    for position, record in enumerate(payload["results"]):
        if not isinstance(record, dict):
            raise SystemExit(f"results[{position}] is not an object")
        if not isinstance(record.get("gold_sql"), str):
            raise SystemExit(f"results[{position}] has no gold_sql")
        sql_generation = record.get("sql_generation")
        if not isinstance(sql_generation, dict) or not isinstance(sql_generation.get("sql"), str):
            raise SystemExit(f"results[{position}] has no predicted sql")
    return payload


def analyze_record(
    record: dict[str, Any],
    schema: DatabaseSchema,
    db_file: Path,
    *,
    max_rows: int,
    preview_rows: int,
) -> dict[str, Any]:
    gold_sql = record["gold_sql"]
    predicted_sql = record["sql_generation"]["sql"]
    ordered = bool(re.search(r"\bORDER\s+BY\b", gold_sql, flags=re.I))
    gold_execution = execute_read_only(gold_sql, db_file, max_rows)
    predicted_execution = execute_read_only(predicted_sql, db_file, max_rows)
    recomputed_match = execution_match(gold_execution, predicted_execution, ordered)

    gold_structure, gold_parse_error = safe_extract_structure(gold_sql, schema)
    predicted_structure, predicted_parse_error = safe_extract_structure(predicted_sql, schema)
    differences, detected, notes = compare_structures(
        gold_structure,
        predicted_structure,
        gold_parse_error,
        predicted_parse_error,
        gold_execution,
        predicted_execution,
        recomputed_match,
    )
    ordered_errors = [error for error in PRIMARY_PRIORITY if error in detected]
    if not ordered_errors:
        ordered_errors = ["UNKNOWN"]
    primary = ordered_errors[0]
    secondary = ordered_errors[1:]

    return {
        "example_id": record.get("index"),
        "db_id": record.get("db_id"),
        "question": record.get("question"),
        "gold_sql": gold_sql,
        "predicted_sql": predicted_sql,
        "stored_sql_evaluation": record.get("sql_evaluation"),
        "recomputed_execution_match": recomputed_match,
        "execution_evaluation_status": (
            "failed" if recomputed_match is False else "unevaluated"
        ),
        "gold_execution_result": gold_execution.public(preview_rows, ordered),
        "predicted_execution_result": predicted_execution.public(preview_rows, ordered),
        "gold_structure": gold_structure,
        "predicted_structure": predicted_structure,
        "parse_errors": {
            "gold": gold_parse_error,
            "prediction": predicted_parse_error,
        },
        "differences": differences,
        "primary_error": primary,
        "secondary_errors": secondary,
        "all_detected_errors": ordered_errors,
        "classification_notes": notes,
        "classification_method": "deterministic_sqlglot_ast_v1",
    }


def _unavailable_schema_analysis(record: dict[str, Any], db_id: str) -> dict[str, Any]:
    return {
        "example_id": record.get("index"),
        "db_id": db_id,
        "question": record.get("question"),
        "gold_sql": record.get("gold_sql"),
        "predicted_sql": record.get("sql_generation", {}).get("sql"),
        "stored_sql_evaluation": record.get("sql_evaluation"),
        "recomputed_execution_match": None,
        "execution_evaluation_status": "unevaluated",
        "gold_execution_result": None,
        "predicted_execution_result": None,
        "gold_structure": None,
        "predicted_structure": None,
        "parse_errors": {"gold": "schema unavailable", "prediction": "schema unavailable"},
        "differences": {},
        "primary_error": "UNKNOWN",
        "secondary_errors": [],
        "all_detected_errors": ["UNKNOWN"],
        "classification_notes": [f"schema not found for db_id={db_id}"],
        "classification_method": "deterministic_sqlglot_ast_v1",
    }


def safe_extract_structure(
    sql: str, schema: DatabaseSchema
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return extract_structure(sql, schema), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def extract_structure(sql: str, schema: DatabaseSchema) -> dict[str, Any]:
    parse_sql = normalize_spider_quoted_literals(sql, schema)
    tree = sqlglot.parse_one(parse_sql, read="sqlite")
    context = SQLContext(tree, schema)
    valid_tables, valid_columns = extract_gold_links(
        schema, parse_sql, require_sqlglot=True
    )

    select_targets: list[str] = []
    distinct_scopes: list[bool] = []
    group_by: list[str] = []
    having: list[str] = []
    order_by: list[dict[str, Any]] = []
    limits: list[str] = []
    filters: list[dict[str, Any]] = []
    join_conditions: list[str] = []
    join_edges: list[dict[str, Any]] = []
    join_types: list[str] = []

    selects = list(tree.find_all(exp.Select))
    if isinstance(tree, exp.Select) and tree not in selects:
        selects.insert(0, tree)
    for select in selects:
        distinct_scopes.append(select.args.get("distinct") is not None)
        for projection in select.expressions:
            select_targets.append(context.signature(_unwrap_alias(projection)))
        group = select.args.get("group")
        if group is not None:
            group_by.extend(context.signature(item) for item in group.expressions)
        having_expression = select.args.get("having")
        if having_expression is not None:
            having.append(context.signature(having_expression.this))
        order = select.args.get("order")
        if order is not None:
            for item in order.expressions:
                ordered_expression = item.this if isinstance(item, exp.Ordered) else item
                order_by.append({
                    "expression": context.signature(ordered_expression),
                    "direction": "DESC" if bool(item.args.get("desc")) else "ASC",
                })
        limit = select.args.get("limit")
        if limit is not None:
            limit_expression = limit.expression or limit.args.get("expression")
            limits.append(context.signature(limit_expression) if limit_expression else "LIMIT")

    explicit_join_nodes = list(tree.find_all(exp.Join))
    for join in explicit_join_nodes:
        join_types.append(_join_type(join))
        on_expression = join.args.get("on")
        if on_expression is not None:
            for predicate in flatten_and(on_expression):
                join_conditions.append(context.signature(predicate))
                join_edges.extend(extract_join_edges(predicate, context))
        using = join.args.get("using")
        if using:
            names = [str(item.name if hasattr(item, "name") else item) for item in using]
            join_conditions.append("USING(" + ",".join(names) + ")")

    for where in tree.find_all(exp.Where):
        for predicate in normalize_filter_predicates(where.this):
            edges = extract_join_edges(predicate, context)
            if edges and is_pure_column_join(predicate):
                join_conditions.append(context.signature(predicate))
                join_edges.extend(edges)
            else:
                filters.append(extract_filter(predicate, context))

    aggregations = sorted(
        context.signature(node)
        for node in tree.find_all(exp.AggFunc)
    )
    set_operations = [
        {
            "operation": type(node).__name__.upper(),
            "distinct": bool(node.args.get("distinct", True)),
        }
        for node in tree.walk()
        if isinstance(node, SET_OPERATION_TYPES)
    ]
    subquery_nodes = list(tree.find_all(exp.Subquery))
    subqueries = [
        context.signature(node, mask_literals=True)
        for node in subquery_nodes
    ]
    subquery_kinds = [_subquery_kind(node) for node in subquery_nodes]
    negations = [
        context.signature(node, mask_literals=True)
        for node in tree.walk()
        if isinstance(node, (exp.Not, exp.NEQ))
    ]

    raw_tables = sorted({node.name for node in tree.find_all(exp.Table)})
    unknown_tables = sorted(
        name for name in raw_tables
        if name.casefold() not in context.table_lookup
    )
    unknown_columns = sorted(context.unresolved_columns)
    return {
        "tables": sorted(valid_tables),
        "columns": sorted(valid_columns),
        "raw_tables": raw_tables,
        "unknown_tables": unknown_tables,
        "unknown_columns": unknown_columns,
        "select_targets": select_targets,
        "joins": {
            "join_types": join_types,
            "conditions": sorted(join_conditions),
            "edges": sorted(join_edges, key=lambda item: json.dumps(item, sort_keys=True)),
        },
        "filters": filters,
        "aggregation": aggregations,
        "distinct": {
            "by_select_scope": distinct_scopes,
            "any": any(distinct_scopes),
        },
        "group_by": group_by,
        "having": having,
        "order_by": order_by,
        "limit": limits,
        "subquery": {
            "count": len(subqueries),
            "kinds": subquery_kinds,
            "signatures": subqueries,
        },
        "set_operation": set_operations,
        "negation": {
            "count": len(negations),
            "signatures": negations,
        },
        "normalized_ast_sql": context.signature(tree),
    }


class SQLContext:
    def __init__(self, tree: exp.Expression, schema: DatabaseSchema) -> None:
        self.tree = tree
        self.schema = schema
        self.table_lookup = {table.casefold(): table for table in schema.tables}
        self.columns_by_table: dict[str, dict[str, str]] = {}
        for column in schema.columns:
            self.columns_by_table.setdefault(column.table, {})[
                column.name.casefold()
            ] = column.key
        self.aliases: dict[str, str] = {}
        self.derived_aliases: set[str] = set()
        self.tables: set[str] = set()
        self.unresolved_columns: set[str] = set()
        for table_expression in tree.find_all(exp.Table):
            table = self.table_lookup.get(table_expression.name.casefold())
            if table is None:
                continue
            self.tables.add(table)
            self.aliases[table_expression.name.casefold()] = table
            self.aliases[table_expression.alias_or_name.casefold()] = table
        for subquery in tree.find_all(exp.Subquery):
            alias = subquery.alias_or_name
            if alias:
                self.derived_aliases.add(alias.casefold())

    def column_key(self, column: exp.Column) -> str:
        name = column.name
        normalized_name = name.casefold()
        qualifier = column.table
        if qualifier:
            normalized_qualifier = qualifier.casefold()
            table = self.aliases.get(normalized_qualifier)
            if table is not None:
                key = self.columns_by_table.get(table, {}).get(normalized_name)
                if key is not None:
                    return key
            if normalized_qualifier in self.derived_aliases:
                return f"<derived>.{name}"
            unresolved = f"{qualifier}.{name}"
            self.unresolved_columns.add(unresolved)
            return unresolved
        candidates = {
            self.columns_by_table[table][normalized_name]
            for table in self.tables
            if normalized_name in self.columns_by_table.get(table, {})
        }
        if len(candidates) == 1:
            return next(iter(candidates))
        if not candidates:
            self.unresolved_columns.add(name)
        return name

    def signature(
        self,
        expression: exp.Expression | None,
        mask_literals: bool = False,
    ) -> str:
        if expression is None:
            return ""
        expression = _unwrap_alias(expression).copy()

        def transform(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column):
                key = self.column_key(node)
                if "." in key:
                    table, column = key.split(".", 1)
                    return exp.column(column, table=table)
                return exp.column(key)
            if isinstance(node, exp.Table):
                table = self.table_lookup.get(node.name.casefold(), node.name)
                return exp.to_table(table)
            if mask_literals and isinstance(node, (exp.Literal, exp.Boolean, exp.Null)):
                return exp.Var(this="__VALUE__")
            return node

        normalized = expression.transform(transform).sql(
            dialect="sqlite", pretty=False, normalize=True
        )
        return re.sub(r"\s+", " ", normalized).strip()


def normalize_spider_quoted_literals(
    sql: str, schema: DatabaseSchema
) -> str:
    """Treat Spider double-quoted non-identifiers as SQLite string literals."""
    known_identifiers = {
        *(table.casefold() for table in schema.tables),
        *(column.name.casefold() for column in schema.columns),
    }

    def replace(match: re.Match[str]) -> str:
        value = match.group(1)
        if value.casefold() in known_identifiers:
            return match.group(0)
        before = sql[:match.start()].rstrip()
        after = sql[match.end():].lstrip()
        if before.endswith(".") or after.startswith("."):
            return match.group(0)
        return "'" + value.replace("'", "''") + "'"

    return re.sub(r'"([^"]*)"', replace, sql)


def _subquery_kind(node: exp.Subquery) -> str:
    parent = node.parent
    if isinstance(parent, exp.In):
        return "IN"
    if isinstance(parent, exp.Exists):
        return "EXISTS"
    if isinstance(parent, (exp.From, exp.Join)):
        return "DERIVED_TABLE"
    if isinstance(parent, COMPARISON_TYPES):
        return "SCALAR_COMPARISON"
    return "SCALAR"


def _unwrap_alias(expression: exp.Expression) -> exp.Expression:
    while isinstance(expression, exp.Alias):
        expression = expression.this
    return expression


def _join_type(join: exp.Join) -> str:
    parts = [join.args.get("side"), join.args.get("kind")]
    rendered = " ".join(str(part).upper() for part in parts if part)
    return rendered or "INNER"


def flatten_and(expression: exp.Expression) -> list[exp.Expression]:
    if isinstance(expression, exp.And):
        return [*flatten_and(expression.left), *flatten_and(expression.right)]
    return [expression]


def normalize_filter_predicates(expression: exp.Expression) -> list[exp.Expression]:
    predicates: list[exp.Expression] = []
    for predicate in flatten_and(expression):
        if isinstance(predicate, exp.Between) and not any(
            isinstance(node, exp.Not) for node in predicate.walk()
        ):
            target = predicate.args.get("this")
            low = predicate.args.get("low")
            high = predicate.args.get("high")
            if target is not None and low is not None and high is not None:
                predicates.extend([
                    exp.GTE(this=target.copy(), expression=low.copy()),
                    exp.LTE(this=target.copy(), expression=high.copy()),
                ])
                continue
        predicates.append(predicate)
    return predicates


def is_pure_column_join(expression: exp.Expression) -> bool:
    if not isinstance(expression, COMPARISON_TYPES):
        return False
    left = expression.args.get("this")
    right = expression.args.get("expression")
    if left is None or right is None:
        return False
    return bool(list(left.find_all(exp.Column))) and bool(list(right.find_all(exp.Column))) \
        and not list(expression.find_all(exp.Literal))


def extract_join_edges(
    expression: exp.Expression, context: SQLContext
) -> list[dict[str, Any]]:
    candidates: list[exp.Expression] = []
    if isinstance(expression, COMPARISON_TYPES):
        candidates.append(expression)
    candidates.extend(
        node for node in expression.find_all(COMPARISON_TYPES)
        if node is not expression
    )
    edges: list[dict[str, Any]] = []
    for comparison in candidates:
        left = comparison.args.get("this")
        right = comparison.args.get("expression")
        if left is None or right is None:
            continue
        left_columns = list(left.find_all(exp.Column))
        right_columns = list(right.find_all(exp.Column))
        if isinstance(left, exp.Column):
            left_columns.insert(0, left)
        if isinstance(right, exp.Column):
            right_columns.insert(0, right)
        if not left_columns or not right_columns:
            continue
        left_key = context.column_key(left_columns[0])
        right_key = context.column_key(right_columns[0])
        left_table = left_key.split(".", 1)[0] if "." in left_key else None
        right_table = right_key.split(".", 1)[0] if "." in right_key else None
        if left_table == right_table:
            continue
        operator = type(comparison).__name__.upper()
        if operator == "EQ" and right_key < left_key:
            left_key, right_key = right_key, left_key
        edges.append({"left": left_key, "right": right_key, "operator": operator})
    return edges


def extract_filter(expression: exp.Expression, context: SQLContext) -> dict[str, Any]:
    columns = sorted({context.column_key(node) for node in expression.find_all(exp.Column)})
    if isinstance(expression, exp.Column):
        columns = [context.column_key(expression)]
    operators = sorted({
        type(node).__name__.upper()
        for node in expression.walk()
        if isinstance(node, COMPARISON_TYPES)
    })
    if isinstance(expression, COMPARISON_TYPES):
        operators = sorted(set([*operators, type(expression).__name__.upper()]))
    values = [_literal_value(node) for node in expression.walk() if _is_value_node(node)]
    return {
        "expression": context.signature(expression),
        "shape": context.signature(expression, mask_literals=True),
        "columns": columns,
        "operators": operators,
        "values": values,
        "normalized_values": [_normalized_literal(item) for item in values],
        "negated": any(isinstance(node, (exp.Not, exp.NEQ)) for node in expression.walk()),
    }


def _is_value_node(node: exp.Expression) -> bool:
    return isinstance(node, (exp.Literal, exp.Boolean, exp.Null))


def _literal_value(node: exp.Expression) -> dict[str, Any]:
    if isinstance(node, exp.Literal):
        return {
            "type": "string" if node.is_string else "number",
            "value": node.this,
        }
    if isinstance(node, exp.Boolean):
        return {"type": "boolean", "value": bool(node.this)}
    return {"type": "null", "value": None}


def _normalized_literal(item: dict[str, Any]) -> str:
    value = item.get("value")
    if value is None:
        return "NULL"
    rendered = str(value).strip()
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", rendered):
        try:
            return f"NUMBER:{float(rendered):.12g}"
        except ValueError:
            pass
    return f"{item.get('type', 'unknown').upper()}:{rendered}"


def compare_structures(
    gold: dict[str, Any] | None,
    predicted: dict[str, Any] | None,
    gold_parse_error: str | None,
    predicted_parse_error: str | None,
    gold_execution: ExecutionSnapshot,
    predicted_execution: ExecutionSnapshot,
    recomputed_match: bool | None,
) -> tuple[dict[str, Any], set[str], list[str]]:
    errors: set[str] = set()
    notes: list[str] = []
    differences: dict[str, Any] = {
        "stored_vs_recomputed_execution_match": None,
    }
    if not predicted_execution.success:
        errors.add("EXECUTION_ERROR")
        notes.append(f"predicted SQL execution failed: {predicted_execution.error}")
    if not gold_execution.success or gold_execution.truncated or predicted_execution.truncated:
        errors.add("UNKNOWN")
        notes.append("execution match is unavailable because gold failed or a result was truncated")
    if gold_parse_error or predicted_parse_error or gold is None or predicted is None:
        errors.add("UNKNOWN")
        notes.append(
            f"AST parse unavailable: gold={gold_parse_error!r}, prediction={predicted_parse_error!r}"
        )
        return differences, errors, notes

    differences["tables"] = set_difference(gold["tables"], predicted["tables"])
    differences["columns"] = set_difference(gold["columns"], predicted["columns"])
    differences["unknown_identifiers"] = {
        "gold_tables": gold["unknown_tables"],
        "predicted_tables": predicted["unknown_tables"],
        "gold_columns": gold["unknown_columns"],
        "predicted_columns": predicted["unknown_columns"],
    }
    missing_tables = differences["tables"]["gold_only"]
    missing_columns = differences["columns"]["gold_only"]
    if missing_tables or missing_columns:
        errors.add("MISSING_TABLE_OR_COLUMN")

    differences["set_operation"] = list_difference(
        gold["set_operation"], predicted["set_operation"], order_sensitive=True
    )
    if difference_exists(differences["set_operation"]):
        errors.add("SET_OPERATION")

    differences["subquery"] = {
        "gold_count": gold["subquery"]["count"],
        "predicted_count": predicted["subquery"]["count"],
        "kinds": list_difference(
            gold["subquery"]["kinds"],
            predicted["subquery"]["kinds"],
            order_sensitive=False,
        ),
        "signatures": list_difference(
            gold["subquery"]["signatures"],
            predicted["subquery"]["signatures"],
            order_sensitive=False,
        ),
    }
    if (
        gold["subquery"]["count"] != predicted["subquery"]["count"]
        or difference_exists(differences["subquery"]["kinds"])
    ):
        errors.add("SUBQUERY")

    gold_multi_table = len(gold["tables"]) > 1
    table_sets_differ = set(gold["tables"]) != set(predicted["tables"])
    if gold_multi_table and table_sets_differ:
        errors.add("JOIN_PATH")
    differences["joins"] = {
        "join_types": list_difference(
            gold["joins"]["join_types"], predicted["joins"]["join_types"], False
        ),
        "conditions": list_difference(
            gold["joins"]["conditions"], predicted["joins"]["conditions"], False
        ),
        "edges": list_difference(
            gold["joins"]["edges"], predicted["joins"]["edges"], False
        ),
    }
    join_type_diff = difference_exists(differences["joins"]["join_types"])
    join_edge_diff = difference_exists(differences["joins"]["edges"])
    raw_condition_diff = difference_exists(differences["joins"]["conditions"])
    no_comparable_edges = not gold["joins"]["edges"] and not predicted["joins"]["edges"]
    if join_type_diff or join_edge_diff or (no_comparable_edges and raw_condition_diff):
        if gold_multi_table or len(predicted["tables"]) > 1:
            errors.add("JOIN_CONDITION")

    filter_diff, filter_errors = compare_filters(gold["filters"], predicted["filters"])
    differences["filters"] = filter_diff
    errors.update(filter_errors)

    differences["negation"] = {
        "gold_count": gold["negation"]["count"],
        "predicted_count": predicted["negation"]["count"],
        "signatures": list_difference(
            gold["negation"]["signatures"],
            predicted["negation"]["signatures"],
            False,
        ),
    }
    if (
        gold["negation"]["count"] != predicted["negation"]["count"]
        or difference_exists(differences["negation"]["signatures"])
    ):
        errors.add("NEGATION")

    clause_specs = [
        ("aggregation", "AGGREGATION", False),
        ("group_by", "GROUP_BY", False),
        ("having", "HAVING", False),
        ("order_by", "ORDER_BY", True),
        ("limit", "LIMIT_OR_SUPERLATIVE", True),
        ("select_targets", "SELECT_TARGET", True),
    ]
    for field, label, order_sensitive in clause_specs:
        differences[field] = list_difference(
            gold[field], predicted[field], order_sensitive
        )
        if difference_exists(differences[field]):
            errors.add(label)

    differences["distinct"] = {
        "gold": gold["distinct"],
        "predicted": predicted["distinct"],
        "different": gold["distinct"] != predicted["distinct"],
    }
    if differences["distinct"]["different"]:
        errors.add("DISTINCT")

    gold_minmax = any(_contains_minmax(item) for item in gold["aggregation"])
    pred_minmax = any(_contains_minmax(item) for item in predicted["aggregation"])
    gold_order_limit = bool(gold["order_by"] and gold["limit"])
    pred_order_limit = bool(predicted["order_by"] and predicted["limit"])
    if (gold_minmax and pred_order_limit) or (pred_minmax and gold_order_limit):
        errors.add("LIMIT_OR_SUPERLATIVE")
        notes.append("MIN/MAX and ORDER BY ... LIMIT are used differently")

    if recomputed_match is False and not errors:
        errors.add("OTHER_SEMANTIC_ERROR")
        notes.append("EX differs but no supported AST feature difference was isolated")
    if recomputed_match is None and not errors:
        errors.add("UNKNOWN")
    return differences, errors, notes


def compare_filters(
    gold_filters: list[dict[str, Any]],
    predicted_filters: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    gold_remaining = list(range(len(gold_filters)))
    pred_remaining = list(range(len(predicted_filters)))
    matched: list[dict[str, Any]] = []
    errors: set[str] = set()

    def pair_where(
        predicate: Callable[[dict[str, Any], dict[str, Any]], bool],
        classification: str | None,
    ) -> None:
        nonlocal gold_remaining, pred_remaining
        for gold_index in list(gold_remaining):
            match_index = next(
                (
                    pred_index
                    for pred_index in pred_remaining
                    if predicate(gold_filters[gold_index], predicted_filters[pred_index])
                ),
                None,
            )
            if match_index is None:
                continue
            gold_remaining.remove(gold_index)
            pred_remaining.remove(match_index)
            matched.append({
                "gold": gold_filters[gold_index],
                "predicted": predicted_filters[match_index],
                "classification": classification or "MATCH",
            })
            if classification:
                errors.add(classification)

    pair_where(lambda left, right: left["expression"] == right["expression"], None)
    pair_where(lambda left, right: left["shape"] == right["shape"], "VALUE_GROUNDING")
    pair_where(
        lambda left, right: left["columns"] == right["columns"]
        and left["normalized_values"] == right["normalized_values"],
        "WRONG_FILTER_OPERATOR",
    )
    pair_where(
        lambda left, right: left["operators"] == right["operators"]
        and left["normalized_values"] == right["normalized_values"],
        "WRONG_FILTER_COLUMN",
    )
    pair_where(
        lambda left, right: left["columns"] == right["columns"]
        and left["operators"] == right["operators"],
        "VALUE_GROUNDING",
    )

    missing = [gold_filters[index] for index in gold_remaining]
    extra = [predicted_filters[index] for index in pred_remaining]
    if missing:
        errors.add("MISSING_FILTER")
    if extra:
        errors.add("EXTRA_FILTER")
    return {
        "matched_or_paired": matched,
        "missing_gold_filters": missing,
        "extra_predicted_filters": extra,
    }, errors


def set_difference(gold: Iterable[Any], predicted: Iterable[Any]) -> dict[str, Any]:
    gold_set = set(gold)
    predicted_set = set(predicted)
    return {
        "gold_only": sorted(gold_set - predicted_set),
        "predicted_only": sorted(predicted_set - gold_set),
        "different": gold_set != predicted_set,
    }


def list_difference(
    gold: list[Any], predicted: list[Any], order_sensitive: bool
) -> dict[str, Any]:
    gold_keys = [_stable_key(item) for item in gold]
    predicted_keys = [_stable_key(item) for item in predicted]
    gold_counter = Counter(gold_keys)
    predicted_counter = Counter(predicted_keys)
    missing_keys = list((gold_counter - predicted_counter).elements())
    extra_keys = list((predicted_counter - gold_counter).elements())
    lookup = {_stable_key(item): item for item in [*gold, *predicted]}
    order_changed = order_sensitive and not missing_keys and not extra_keys \
        and gold_keys != predicted_keys
    return {
        "gold_only": [lookup[key] for key in missing_keys],
        "predicted_only": [lookup[key] for key in extra_keys],
        "order_changed": order_changed,
        "different": bool(missing_keys or extra_keys or order_changed),
    }


def difference_exists(difference: dict[str, Any]) -> bool:
    return bool(difference.get("different"))


def _stable_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contains_minmax(signature: str) -> bool:
    upper = signature.upper()
    return "MIN(" in upper or "MAX(" in upper


def execute_read_only(
    sql: str, db_file: Path, max_rows: int
) -> ExecutionSnapshot:
    safety_error = safe_select_error(sql)
    if safety_error:
        return ExecutionSnapshot(False, [], [], error=safety_error, max_rows=max_rows)
    if not db_file.is_file():
        return ExecutionSnapshot(
            False, [], [], error=f"database not found: {db_file}", max_rows=max_rows
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{db_file.resolve()}?mode=ro", uri=True, timeout=5.0
        )
        connection.execute("PRAGMA query_only = ON")
        calls = 0

        def guard() -> int:
            nonlocal calls
            calls += 1
            return int(calls > 20_000)

        connection.set_progress_handler(guard, 1_000)
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        return ExecutionSnapshot(
            True,
            columns,
            rows[:max_rows] if truncated else rows,
            truncated=truncated,
            max_rows=max_rows,
        )
    except sqlite3.Error as error:
        return ExecutionSnapshot(False, [], [], error=str(error), max_rows=max_rows)
    finally:
        if connection is not None:
            connection.close()


def execution_match(
    gold: ExecutionSnapshot,
    predicted: ExecutionSnapshot,
    ordered: bool,
) -> bool | None:
    if not gold.success or gold.truncated or predicted.truncated:
        return None
    if not predicted.success:
        return False
    gold_rows = [_canonical_row(row) for row in gold.rows]
    predicted_rows = [_canonical_row(row) for row in predicted.rows]
    if ordered:
        return predicted_rows == gold_rows
    return Counter(predicted_rows) == Counter(gold_rows)


def _canonical_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(round(value, 8) if isinstance(value, float) else value for value in row)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def build_summary(
    analyses: list[dict[str, Any]],
    *,
    total_examples: int,
    skipped_successes: int,
    source_summary: dict[str, Any],
    representative_limit: int,
) -> dict[str, Any]:
    primary_counts = Counter(item["primary_error"] for item in analyses)
    secondary_counts = Counter(
        error for item in analyses for error in item["secondary_errors"]
    )
    any_counts = Counter(
        error for item in analyses for error in set(item["all_detected_errors"])
    )
    denominator = len(analyses)
    representatives: dict[str, list[dict[str, Any]]] = {}
    for error_type in TAXONOMY:
        candidates = [
            item for item in analyses if error_type in item["all_detected_errors"]
        ]
        candidates.sort(
            key=lambda item: (
                item["primary_error"] != error_type,
                len(item["all_detected_errors"]),
                item.get("example_id") if isinstance(item.get("example_id"), int) else 10**9,
            )
        )
        representatives[error_type] = [
            {
                "example_id": item.get("example_id"),
                "db_id": item.get("db_id"),
                "question": item.get("question"),
                "primary_error": item.get("primary_error"),
                "secondary_errors": item.get("secondary_errors"),
                "gold_sql": item.get("gold_sql"),
                "predicted_sql": item.get("predicted_sql"),
            }
            for item in candidates[:representative_limit]
        ]

    frequencies = []
    for error_type in TAXONOMY:
        frequencies.append({
            "error_type": error_type,
            "primary_count": primary_counts[error_type],
            "primary_percentage_of_analyzed": _percentage(
                primary_counts[error_type], denominator
            ),
            "secondary_count": secondary_counts[error_type],
            "any_case_count": any_counts[error_type],
            "any_case_percentage_of_analyzed": _percentage(
                any_counts[error_type], denominator
            ),
            "representative_example_ids": [
                item["example_id"] for item in representatives[error_type]
            ],
        })
    frequencies.sort(
        key=lambda item: (-item["primary_count"], -item["any_case_count"], item["error_type"])
    )
    family_frequencies: list[dict[str, Any]] = []
    for family, labels in ERROR_FAMILIES.items():
        label_set = set(labels)
        primary_count = sum(
            item["primary_error"] in label_set for item in analyses
        )
        any_count = sum(
            bool(label_set & set(item["all_detected_errors"])) for item in analyses
        )
        family_frequencies.append({
            "error_family": family,
            "included_error_types": labels,
            "primary_count": primary_count,
            "primary_percentage_of_analyzed": _percentage(primary_count, denominator),
            "any_case_count": any_count,
            "any_case_percentage_of_analyzed": _percentage(any_count, denominator),
        })
    family_frequencies.sort(
        key=lambda item: (-item["primary_count"], item["error_family"])
    )

    return {
        "source_run_summary": source_summary,
        "total_examples": total_examples,
        "stored_ex_success_count": skipped_successes,
        "analyzed_non_success_count": denominator,
        "stored_execution_match_false_count": sum(
            item.get("stored_sql_evaluation", {}).get("execution_match") is False
            for item in analyses
        ),
        "stored_execution_match_none_count": sum(
            item.get("stored_sql_evaluation", {}).get("execution_match") is None
            for item in analyses
        ),
        "recomputed_match_disagreement_count": sum(
            item.get("stored_sql_evaluation", {}).get("execution_match")
            != item.get("recomputed_execution_match")
            for item in analyses
        ),
        "primary_error_counts": dict(primary_counts.most_common()),
        "secondary_error_counts": dict(secondary_counts.most_common()),
        "any_occurrence_counts": dict(any_counts.most_common()),
        "error_frequencies": frequencies,
        "error_family_frequencies": family_frequencies,
        "representative_cases": representatives,
    }


def _percentage(count: int, denominator: int) -> float:
    return round(count / denominator * 100, 2) if denominator else 0.0


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    representatives = summary["representative_cases"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "error_type",
                "primary_count",
                "primary_percentage_of_analyzed",
                "secondary_count",
                "any_case_count",
                "any_case_percentage_of_analyzed",
                "representative_example_ids",
                "representative_cases",
            ],
        )
        writer.writeheader()
        for row in summary["error_frequencies"]:
            error_type = row["error_type"]
            writer.writerow({
                **row,
                "representative_example_ids": json.dumps(
                    row["representative_example_ids"], ensure_ascii=False
                ),
                "representative_cases": json.dumps(
                    representatives[error_type], ensure_ascii=False
                ),
            })


def print_frequency_summary(summary: dict[str, Any]) -> None:
    print(
        "Analyzed",
        summary["analyzed_non_success_count"],
        "non-success cases out of",
        summary["total_examples"],
    )
    print("Primary error family frequency:")
    for row in summary["error_family_frequencies"]:
        if row["primary_count"]:
            print(
                f"- {row['error_family']}: primary={row['primary_count']}, "
                f"any={row['any_case_count']}"
            )
    print("Primary error frequency:")
    for row in summary["error_frequencies"]:
        if row["primary_count"] or row["any_case_count"]:
            print(
                f"- {row['error_type']}: primary={row['primary_count']}, "
                f"any={row['any_case_count']}"
            )


if __name__ == "__main__":
    main()
