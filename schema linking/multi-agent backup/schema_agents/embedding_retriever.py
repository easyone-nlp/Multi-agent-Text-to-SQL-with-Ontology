from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .model_client import EmbeddingModel, ModelClientError
from .models import Column, DatabaseSchema


@dataclass
class RetrievedSchema:
    tables: list[str]
    columns: list[str]
    query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tables": self.tables,
            "columns": self.columns,
        }

    def render(self, schema: DatabaseSchema) -> str:
        selected_columns = set(self.columns)
        lines: list[str] = []
        for table in self.tables:
            columns = [
                column
                for column in schema.columns_for(table)
                if column.key in selected_columns
            ]
            rendered = ", ".join(
                _render_column(column, schema) for column in columns
            )
            lines.append(f"- {table}({rendered})")
        return "\n".join(lines)


class EmbeddingSchemaRetriever:
    """Initial multilingual schema retrieval; scores are not used downstream."""

    def __init__(
        self,
        model: EmbeddingModel,
        top_k_tables: int = 6,
        top_k_columns: int = 24,
    ) -> None:
        self.model = model
        self.top_k_tables = top_k_tables
        self.top_k_columns = top_k_columns
        self._cache: dict[
            str,
            tuple[list[str], list[list[float]], list[str], list[list[float]]],
        ] = {}

    def retrieve(
        self,
        query: str,
        schema: DatabaseSchema,
    ) -> RetrievedSchema:
        (
            table_names,
            table_vectors,
            column_keys,
            column_vectors,
        ) = self._schema_vectors(schema)
        query_vectors = self.model.embed([query])
        if len(query_vectors) != 1:
            raise ModelClientError("embedding query vector가 반환되지 않았습니다.")
        query_vector = query_vectors[0]

        top_tables = _top_keys(
            query_vector, table_names, table_vectors, self.top_k_tables
        )
        top_columns = _top_keys(
            query_vector, column_keys, column_vectors, self.top_k_columns
        )
        tables = list(top_tables)
        for key in top_columns:
            table = key.split(".", 1)[0]
            if table not in tables:
                tables.append(table)

        columns = list(top_columns)
        for table in top_tables:
            for column in schema.columns_for(table):
                if column.key not in columns:
                    columns.append(column.key)
        return RetrievedSchema(tables=tables, columns=columns, query=query)

    def _schema_vectors(
        self,
        schema: DatabaseSchema,
    ) -> tuple[list[str], list[list[float]], list[str], list[list[float]]]:
        cached = self._cache.get(schema.db_id)
        if cached is not None:
            return cached

        table_names = list(schema.tables)
        table_documents = [
            _table_document(table, schema) for table in table_names
        ]
        column_keys = [column.key for column in schema.columns]
        column_documents = [
            _column_document(column, schema) for column in schema.columns
        ]
        vectors = self.model.embed(table_documents + column_documents)
        table_vectors = vectors[: len(table_documents)]
        column_vectors = vectors[len(table_documents) :]
        cached = (table_names, table_vectors, column_keys, column_vectors)
        self._cache[schema.db_id] = cached
        return cached


def _table_document(table: str, schema: DatabaseSchema) -> str:
    columns = ", ".join(
        f"{column.name} ({column.column_type})"
        for column in schema.columns_for(table)
    )
    return f"database table {table}; columns: {columns}"


def _column_document(column: Column, schema: DatabaseSchema) -> str:
    metadata: list[str] = []
    if column.key in schema.primary_keys:
        metadata.append("primary key")
    for left, right in schema.foreign_keys:
        if column.key == left:
            metadata.append(f"foreign key to {right}")
        if column.key == right:
            metadata.append(f"referenced by {left}")
    suffix = "; ".join(metadata)
    return (
        f"database column {column.key}; type {column.column_type}"
        + (f"; {suffix}" if suffix else "")
    )


def _render_column(column: Column, schema: DatabaseSchema) -> str:
    roles: list[str] = [column.column_type]
    if column.key in schema.primary_keys:
        roles.append("PK")
    for left, right in schema.foreign_keys:
        if column.key == left:
            roles.append(f"FK->{right}")
        if column.key == right:
            roles.append(f"REFERENCED_BY->{left}")
    return f"{column.name}<{';'.join(roles)}>"


def _top_keys(
    query: list[float],
    keys: list[str],
    vectors: list[list[float]],
    limit: int,
) -> list[str]:
    ranked = sorted(
        zip(keys, vectors),
        key=lambda item: (-_cosine(query, item[1]), item[0]),
    )
    return [key for key, _ in ranked[: max(1, limit)]]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ModelClientError(
            f"embedding dimension mismatch: {len(left)} != {len(right)}"
        )
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator
