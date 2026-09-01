from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

from .danke_multisource_retriever import MultiSourceDankeRetriever
from .embedding_retriever import (
    EmbeddingSchemaRetriever,
    RetrievedSchema,
    _cosine,
)
from .models import DatabaseSchema


class HybridDankeEmbeddingRetriever:
    """Evidence-tier fusion without adding DANKE and cosine scores.

    Tables are selected first. Only strong DANKE endpoints are protected;
    agreement and embedding recall fill the remaining semantic budget. Columns
    are then reranked inside the final semantic tables. Steiner bridges and
    validated FK keys stay outside both semantic budgets.
    """

    def __init__(
        self,
        danke_config: dict[str, Any],
        embedding_retriever: EmbeddingSchemaRetriever,
    ) -> None:
        self.danke = MultiSourceDankeRetriever(danke_config)
        self.embedding = embedding_retriever
        self.semantic_table_budget = self.danke.semantic_table_budget
        self.semantic_column_budget = self.danke.semantic_column_budget
        self.bridge_table_budget = self.danke.bridge_table_budget
        self.strict_top_k = True
        self.top_k_tables = self.semantic_table_budget
        self.top_k_columns = self.semantic_column_budget
        self.model = SimpleNamespace(
            model=(
                "danke_embedding_hybrid_v1:"
                f"{self.embedding.model.model}"
            )
        )

    def retrieve_multisource(
        self,
        question: str,
        decomposition: dict[str, Any],
        schema: DatabaseSchema,
    ) -> RetrievedSchema:
        retrieval_query = str(
            decomposition.get("retrieval_query") or question
        )
        danke_result = self.danke.retrieve_multisource(
            question, decomposition, schema
        )
        table_ranking, query_vector, source_table_rankings = (
            self._embedding_table_rankings_by_source(
                question, decomposition, schema
            )
        )
        embedding_top_tables = [
            item["key"]
            for item in table_ranking[: self.semantic_table_budget]
        ]
        danke_metadata = danke_result.metadata or {}
        raw = danke_metadata.get("raw", {})
        eligible_danke_tables = _unique(
            [
                *danke_metadata.get("semantic_tables", []),
                *[
                    item["table"]
                    for item in raw.get("class_ranking", [])
                    if not item.get("fallback", False)
                ],
            ]
        )
        strong_reasons = self._strong_danke_table_reasons(
            raw, self.danke._artifacts(schema.db_id)[1]
        )
        strong_danke_tables = [
            table
            for table in eligible_danke_tables
            if table in strong_reasons
        ]
        danke_bridge_candidates = set(
            danke_metadata.get("bridge_candidates_before_budget", [])
        )
        strong_embedding_reasons = _strong_embedding_table_reasons(
            source_table_rankings
        )
        selected_tables: list[str] = []
        table_selection: list[dict[str, Any]] = []

        def select_table(table: str, tier: str) -> None:
            if table in selected_tables or len(selected_tables) >= self.semantic_table_budget:
                return
            selected_tables.append(table)
            embedding_item = next(
                (item for item in table_ranking if item["key"] == table),
                None,
            )
            table_selection.append(
                {
                    "table": table,
                    "tier": tier,
                    "danke_strong_reasons": strong_reasons.get(table, []),
                    "danke_candidate": table in eligible_danke_tables,
                    "embedding_strong_reasons": strong_embedding_reasons.get(
                        table, []
                    ),
                    "embedding_rank": (
                        table_ranking.index(embedding_item) + 1
                        if embedding_item is not None
                        else None
                    ),
                    "embedding_cosine": (
                        embedding_item["score"] if embedding_item else None
                    ),
                }
            )

        for table in strong_danke_tables:
            select_table(table, "danke_strong")
        for table in embedding_top_tables:
            if (
                table in eligible_danke_tables
                and table in strong_embedding_reasons
                and not _is_bridge_like_table(schema, table)
            ):
                select_table(table, "danke_embedding_agreement")
        excluded_embedding_bridges: list[str] = []
        for table in embedding_top_tables:
            if (
                table not in strong_embedding_reasons
                or _is_bridge_like_table(schema, table)
                or (
                    table in danke_bridge_candidates
                    and table not in strong_danke_tables
                )
            ):
                if (
                    _is_bridge_like_table(schema, table)
                    or table in danke_bridge_candidates
                ):
                    excluded_embedding_bridges.append(table)
                continue
            select_table(table, "embedding_fill")
        if not selected_tables and embedding_top_tables:
            select_table(
                embedding_top_tables[0], "embedding_no_danke_fallback"
            )

        column_ranking = self._embedding_column_ranking(
            query_vector, selected_tables, schema
        )
        embedding_top_columns = [
            item["key"]
            for item in column_ranking[: self.semantic_column_budget]
        ]
        danke_column_ranking, danke_column_evidence = (
            self._danke_columns_within_tables(
                selected_tables, raw, schema.db_id
            )
        )
        strong_danke_columns = [
            item["column"]
            for item in danke_column_ranking
            if _strong_column_evidence(
                danke_column_evidence.get(item["column"], [])
            )
        ]
        danke_column_candidates = [
            item["column"]
            for item in danke_column_ranking[: self.semantic_column_budget]
        ]
        selected_columns: list[str] = []
        column_selection: list[dict[str, Any]] = []

        def select_column(column: str, tier: str) -> None:
            if column in selected_columns or len(selected_columns) >= self.semantic_column_budget:
                return
            selected_columns.append(column)
            embedding_item = next(
                (item for item in column_ranking if item["key"] == column),
                None,
            )
            column_selection.append(
                {
                    "column": column,
                    "tier": tier,
                    "danke_evidence": danke_column_evidence.get(column, []),
                    "embedding_rank_within_final_tables": (
                        column_ranking.index(embedding_item) + 1
                        if embedding_item is not None
                        else None
                    ),
                    "embedding_cosine": (
                        embedding_item["score"] if embedding_item else None
                    ),
                }
            )

        for column in strong_danke_columns:
            select_column(column, "danke_strong")
        for column in danke_column_candidates[:3]:
            select_column(column, "danke_top3_anchor")
        for column in embedding_top_columns:
            if column in danke_column_candidates:
                select_column(column, "danke_embedding_agreement")
        for column in embedding_top_columns:
            select_column(column, "embedding_fill")
        for column in danke_column_candidates:
            select_column(column, "danke_weak_fill")

        bridge_tables, join_key_columns, join_edges, steiner_raw = (
            self._structural_evidence(selected_tables, schema)
        )
        effective_tables = _unique([*selected_tables, *bridge_tables])
        all_columns = _unique([*selected_columns, *join_key_columns])
        return RetrievedSchema(
            tables=effective_tables,
            columns=all_columns,
            query=retrieval_query,
            mode="danke_embedding_hybrid",
            top_k_tables=self.semantic_table_budget,
            top_k_columns=self.semantic_column_budget,
            metadata={
                "semantic_tables": selected_tables,
                "semantic_columns": selected_columns,
                "bridge_tables": bridge_tables,
                "join_key_columns": join_key_columns,
                "join_edges": join_edges,
                "table_selection": table_selection,
                "column_selection": column_selection,
                "excluded_embedding_bridge_candidates": excluded_embedding_bridges,
                "embedding": {
                    "query": retrieval_query,
                    "table_top_k": embedding_top_tables,
                    "table_ranking": table_ranking,
                    "table_rankings_by_source": source_table_rankings,
                    "column_top_k_within_final_tables": embedding_top_columns,
                    "column_ranking_within_final_tables": column_ranking,
                },
                "danke": danke_metadata,
                "hybrid_contract": {
                    "score_addition": False,
                    "table_tiers": [
                        "danke_strong",
                        "danke_embedding_agreement",
                        "embedding_fill",
                        "embedding_no_danke_fallback",
                    ],
                    "column_tiers": [
                        "danke_strong",
                        "danke_top3_anchor",
                        "danke_embedding_agreement",
                        "embedding_fill",
                        "danke_weak_fill",
                    ],
                    "danke_column_anchor_limit": 3,
                    "columns_selected_after_tables": True,
                    "bridges_outside_semantic_budget": True,
                    "join_keys_outside_semantic_budget": True,
                },
                "steiner_forest_for_final_endpoints": steiner_raw,
            },
        )

    def retrieve(self, query: str, schema: DatabaseSchema) -> RetrievedSchema:
        return self.retrieve_multisource(
            query,
            {"retrieval_query": query},
            schema,
        )

    def _embedding_table_rankings_by_source(
        self,
        question: str,
        decomposition: dict[str, Any],
        schema: DatabaseSchema,
    ) -> tuple[
        list[dict[str, Any]],
        list[float],
        list[dict[str, Any]],
    ]:
        from ontology.compare_retrievers_fair import _danke_query_sources

        table_names, table_vectors, _keys, _vectors = (
            self.embedding._schema_vectors(schema)
        )
        retrieval_query = str(
            decomposition.get("retrieval_query") or question
        )
        sources = _danke_query_sources(
            {
                "question": question,
                "retrieval_query": retrieval_query,
                "decomposition": decomposition,
            },
            mode="multi_source",
        )
        vectors = self.embedding.model.embed(
            [str(source["text"]) for source in sources]
        )
        if len(vectors) != len(sources):
            raise RuntimeError(
                "embedding source vector 수가 입력 source 수와 다릅니다."
            )
        source_rankings: list[dict[str, Any]] = []
        retrieval_ranking: list[dict[str, Any]] | None = None
        retrieval_vector: list[float] | None = None
        for source, vector in zip(sources, vectors):
            ranking = _rank_vectors(vector, table_names, table_vectors)
            source_rankings.append(
                {
                    "source_id": source["source_id"],
                    "source_type": source["source_type"],
                    "source_text": source["text"],
                    "ranking": ranking,
                }
            )
            if source["source_id"] == "retrieval_query":
                retrieval_ranking = ranking
                retrieval_vector = vector
        if retrieval_ranking is None or retrieval_vector is None:
            raise RuntimeError("retrieval_query embedding source가 없습니다.")
        return retrieval_ranking, retrieval_vector, source_rankings

    def _embedding_column_ranking(
        self,
        query_vector: list[float],
        tables: list[str],
        schema: DatabaseSchema,
    ) -> list[dict[str, Any]]:
        _table_names, _table_vectors, keys, vectors = (
            self.embedding._schema_vectors(schema)
        )
        allowed = set(tables)
        filtered = [
            (key, vector)
            for key, vector in zip(keys, vectors)
            if key.split(".", 1)[0] in allowed
        ]
        return _rank_vectors(
            query_vector,
            [key for key, _vector in filtered],
            [vector for _key, vector in filtered],
        )

    def _strong_danke_table_reasons(
        self, raw: dict[str, Any], mapping: Any
    ) -> dict[str, list[str]]:
        reasons: dict[str, list[str]] = defaultdict(list)
        for evidence in raw.get("matches", []):
            if evidence.get("generic_property"):
                continue
            target_class = str(evidence.get("target_class") or "")
            if target_class not in mapping.class_table:
                continue
            table = mapping.table_for(target_class)
            entry_type = str(evidence.get("entry_type") or "")
            exact = bool(evidence.get("exact"))
            similarity = float(evidence.get("similarity", 0.0))
            source_types = set(evidence.get("source_types", []))
            if entry_type == "class" and (exact or similarity >= 0.8):
                reasons[table].append("direct_class")
            elif entry_type == "value" and (exact or similarity >= 0.82):
                reasons[table].append("direct_value")
            elif (
                entry_type == "property"
                and exact
                and any(source.startswith("plan_filter_") for source in source_types)
            ):
                reasons[table].append("exact_filter_property")
        return {table: _unique(items) for table, items in reasons.items()}

    def _danke_columns_within_tables(
        self,
        tables: list[str],
        raw: dict[str, Any],
        db_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        from ontology.compare_retrievers_fair import _rank_properties

        knowledge_schema, mapping, _dictionary = self.danke._artifacts(db_id)
        table_classes = {
            table: class_name
            for class_name, table in mapping.class_table.items()
        }
        classes = [
            table_classes[table]
            for table in tables
            if table in table_classes
        ]
        property_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        column_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for evidence in raw.get("matches", []):
            property_name = evidence.get("target_property")
            if not property_name or property_name not in mapping.property_column:
                continue
            table, column = mapping.column_for(property_name)
            if table not in tables:
                continue
            property_hits[property_name].append(evidence)
            column_evidence[f"{table}.{column}"].append(evidence)
        ranking = _rank_properties(
            classes, property_hits, knowledge_schema, mapping
        )
        return ranking, dict(column_evidence)

    def _structural_evidence(
        self, semantic_tables: list[str], schema: DatabaseSchema
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        from ontology.compare_retrievers_fair import _ordered_bridge_classes
        from ontology.danke_kg.graph import build_knowledge_graph, steiner_forest
        from schema_agents.join_validator import declared_fk

        knowledge_schema, mapping, _dictionary = self.danke._artifacts(schema.db_id)
        table_classes = {
            table: class_name
            for class_name, table in mapping.class_table.items()
        }
        semantic_classes = [
            table_classes[table]
            for table in semantic_tables
            if table in table_classes
        ]
        trees = (
            steiner_forest(
                build_knowledge_graph(knowledge_schema), semantic_classes
            )
            if self.danke.use_steiner and len(semantic_classes) >= 2
            else []
        )
        bridge_classes = _ordered_bridge_classes(
            trees, semantic_classes
        )[: self.bridge_table_budget]
        bridge_tables = [
            mapping.table_for(name)
            for name in bridge_classes
            if name in mapping.class_table
        ]
        kept_classes = set(semantic_classes) | set(bridge_classes)
        join_edges: list[dict[str, Any]] = []
        join_key_columns: list[str] = []
        for tree in trees:
            for left_class, right_class, op_name, weight in tree.edges:
                if left_class not in kept_classes or right_class not in kept_classes:
                    continue
                if op_name not in mapping.object_property_fk:
                    continue
                proposed_left, proposed_right = mapping.fk_for(op_name)
                validated = declared_fk(
                    schema, proposed_left, proposed_right
                )
                edge = {
                    "left_class": left_class,
                    "right_class": right_class,
                    "object_property": op_name,
                    "proposed_left": proposed_left,
                    "proposed_right": proposed_right,
                    "validated_fk": list(validated) if validated else None,
                    "validated": validated is not None,
                    "weight": weight,
                }
                join_edges.append(edge)
                if validated:
                    for column in validated:
                        if column not in join_key_columns:
                            join_key_columns.append(column)
        return (
            bridge_tables,
            join_key_columns,
            join_edges,
            [tree.to_dict() for tree in trees],
        )


def _strong_column_evidence(evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        if item.get("entry_type") == "value":
            return True
        if item.get("exact"):
            return True
        if len(set(item.get("source_types", []))) >= 2:
            return True
    return False


def _strong_embedding_table_reasons(
    source_rankings: list[dict[str, Any]],
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = defaultdict(list)
    ranks_by_source: dict[str, dict[str, int]] = {}
    for source in source_rankings:
        ranks_by_source[str(source["source_id"])] = {
            item["key"]: index
            for index, item in enumerate(source["ranking"], start=1)
        }
        source_type = str(source["source_type"])
        text = str(source["source_text"]).strip().casefold()
        if (
            source_type.startswith("plan_")
            and source_type
            not in {"plan_filter_value_mention"}
            and not _generic_or_numeric_source(text)
            and source["ranking"]
        ):
            reasons[source["ranking"][0]["key"]].append(
                f"atomic_{source_type}_top1"
            )
    original = ranks_by_source.get("question", {})
    retrieval = ranks_by_source.get("retrieval_query", {})
    for table in set(original) & set(retrieval):
        if original[table] <= 2 and retrieval[table] <= 2:
            reasons[table].append("original_and_retrieval_top2")
    return {table: _unique(items) for table, items in reasons.items()}


def _generic_or_numeric_source(text: str) -> bool:
    normalized = "".join(character for character in text if character.isalnum())
    return (
        not normalized
        or normalized.isdigit()
        or normalized
        in {
            "id",
            "identifier",
            "name",
            "count",
            "아이디",
            "식별자",
            "이름",
            "수",
        }
    )


def _is_bridge_like_table(schema: DatabaseSchema, table: str) -> bool:
    table_columns = {column.key for column in schema.columns_for(table)}
    fk_columns: set[str] = set()
    outgoing_foreign_keys = 0
    for left, _right in schema.foreign_keys:
        if left in table_columns:
            outgoing_foreign_keys += 1
            fk_columns.add(left)
    non_key_columns = table_columns - fk_columns - schema.primary_keys
    return outgoing_foreign_keys >= 2 and len(non_key_columns) <= 1


def _rank_vectors(
    query: list[float],
    keys: list[str],
    vectors: list[list[float]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {"key": key, "score": float(_cosine(query, vector))}
            for key, vector in zip(keys, vectors)
        ),
        key=lambda item: (-item["score"], item["key"]),
    )
    return [
        {"key": item["key"], "score": round(item["score"], 8)}
        for item in ranked
    ]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
