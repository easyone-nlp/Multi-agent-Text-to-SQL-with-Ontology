from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = Path(__file__).resolve().parent
MULTI_AGENT_DIR = WORKSPACE / "schema linking" / "multi-agent"
if str(ONTOLOGY_DIR) not in sys.path:
    sys.path.insert(0, str(ONTOLOGY_DIR))
if str(MULTI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(MULTI_AGENT_DIR))

from danke_kg.dictionary import (  # noqa: E402
    Dictionary,
    DictionaryEntry,
    MatchingDiscoveryService,
)
from danke_kg.graph import build_knowledge_graph, steiner_forest  # noqa: E402
from danke_kg.mapping import RDBMapping  # noqa: E402
from danke_kg.models import KnowledgeSchema  # noqa: E402
from danke_kg.multiagent_bridge import WORD_PATTERN  # noqa: E402
from schema_agents.data import default_tables_path, load_schemas  # noqa: E402
from schema_agents.embedding_retriever import EmbeddingSchemaRetriever  # noqa: E402
from schema_agents.evaluation import extract_gold_links  # noqa: E402
from schema_agents.join_validator import declared_fk  # noqa: E402
from schema_agents.model_client import build_embedding_model  # noqa: E402
from schema_agents.models import DatabaseSchema  # noqa: E402

VARIANTS = {
    "full": {"use_synonyms": True, "use_steiner": True},
    "no_steiner": {"use_synonyms": True, "use_steiner": False},
    "primary_only": {"use_synonyms": False, "use_steiner": True},
    "primary_only_no_steiner": {
        "use_synonyms": False,
        "use_steiner": False,
    },
}

GENERIC_PROPERTY_KEYS = {
    "id",
    "identifier",
    "name",
    "아이디",
    "식별자",
    "이름",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fair retriever-only comparison using one cached retrieval_query per "
            "question. Embedding and DANKE receive identical query strings and "
            "separate semantic/bridge/join-key budgets."
        )
    )
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--knowledge-schema-dir",
        type=Path,
        default=(
            ONTOLOGY_DIR
            / "versions"
            / "danke-spider-ko-v1.0.0-20260728"
        ),
    )
    parser.add_argument("--tables", type=Path, default=default_tables_path("validation"))
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--semantic-table-budget", type=int, default=3)
    parser.add_argument("--semantic-column-budget", type=int, default=8)
    parser.add_argument("--bridge-table-budget", type=int, default=2)
    parser.add_argument("--fuzzy-cutoff", type=float, default=0.72)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument(
        "--danke-input-mode",
        choices=("retrieval_query", "multi_source"),
        default="retrieval_query",
        help=(
            "retrieval_query: 기존 단일 입력. multi_source: original question, "
            "retrieval query, plan의 output/filter/group/order/relationship span을 "
            "각각 검색한 뒤 ontology item 기준으로 중복 제거합니다."
        ),
    )
    parser.add_argument("--limit", type=int, help="Optional smoke-test prefix of the cache.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = json.loads(args.query_cache.read_text(encoding="utf-8"))
    source_records = list(cache.get("records", []))
    if args.limit is not None:
        source_records = source_records[: max(0, args.limit)]
    if not source_records:
        raise SystemExit("query cache에 비교할 record가 없습니다.")

    schemas = load_schemas(args.tables)
    embedding_model = build_embedding_model(
        {
            "provider": "openai_compatible",
            "model": args.embedding_model,
            "base_url": args.embedding_base_url,
            "timeout_seconds": args.embedding_timeout_seconds,
        }
    )
    embedding = EmbeddingSchemaRetriever(
        embedding_model,
        top_k_tables=args.semantic_table_budget,
        top_k_columns=args.semantic_column_budget,
        strict_top_k=True,
    )

    metadata = _metadata(args, cache, source_records)
    completed: dict[int, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            previous.get("metadata", {})
            .get("query_cache", {})
            .get("sha256")
            != metadata["query_cache"]["sha256"]
        ):
            raise SystemExit(
                "기존 output의 query-cache checksum이 현재 입력과 다릅니다. "
                "--no-resume 또는 다른 output을 사용하세요."
            )
        completed = {
            int(record["index"]): record
            for record in previous.get("records", [])
        }

    kg_cache: dict[str, tuple[KnowledgeSchema, RDBMapping]] = {}
    dictionary_cache: dict[tuple[str, bool], Dictionary] = {}

    for position, source in enumerate(source_records, start=1):
        index = int(source["index"])
        if index in completed:
            print(f"[{position}/{len(source_records)}] index={index} cached")
            continue
        db_id = str(source["db_id"])
        schema = schemas.get(db_id)
        if schema is None:
            raise SystemExit(f"Spider tables metadata에 db_id={db_id!r}가 없습니다.")
        knowledge_schema, mapping = _load_kg(
            db_id, args.knowledge_schema_dir, kg_cache
        )
        query = str(source["retrieval_query"])
        danke_query_sources = _danke_query_sources(
            source, mode=args.danke_input_mode
        )
        gold_sql = str(source.get("gold_sql") or "")
        gold_tables, gold_columns = extract_gold_links(
            schema, gold_sql, require_sqlglot=True
        )
        gold_join_keys = _gold_join_key_proxy(schema, gold_columns)
        gold_semantic_columns = gold_columns - gold_join_keys

        print(
            f"[{position}/{len(source_records)}] index={index} "
            f"db_id={db_id} embedding+danke..."
        )
        embedding_output = _embedding_retrieve(
            embedding, query, schema, args
        )
        danke_outputs: dict[str, dict[str, Any]] = {}
        for name, switches in VARIANTS.items():
            dictionary_key = (db_id, bool(switches["use_synonyms"]))
            dictionary = dictionary_cache.get(dictionary_key)
            if dictionary is None:
                dictionary = _build_dictionary(
                    knowledge_schema,
                    use_synonyms=bool(switches["use_synonyms"]),
                )
                dictionary_cache[dictionary_key] = dictionary
            danke_outputs[name] = _danke_retrieve(
                query_sources=danke_query_sources,
                schema=schema,
                knowledge_schema=knowledge_schema,
                mapping=mapping,
                dictionary=dictionary,
                use_steiner=bool(switches["use_steiner"]),
                args=args,
            )

        record = {
            "index": index,
            "db_id": db_id,
            "question": source["question"],
            "retrieval_query": query,
            "danke_query_sources": danke_query_sources,
            "orchestrator_attempts": source.get("orchestrator_attempts"),
            "gold_sql": gold_sql,
            "gold": {
                "tables": sorted(gold_tables),
                "columns": sorted(gold_columns),
                "semantic_columns": sorted(gold_semantic_columns),
                "join_key_columns_proxy": sorted(gold_join_keys),
            },
            "embedding": embedding_output,
            "danke": danke_outputs,
        }
        record["evaluation"] = _evaluate_record(
            record,
            gold_tables,
            gold_columns,
            gold_semantic_columns,
            gold_join_keys,
        )
        completed[index] = record
        _write_output(
            args.output,
            metadata,
            {"status": "partial", "completed": len(completed)},
            [
                completed[int(item["index"])]
                for item in source_records
                if int(item["index"]) in completed
            ],
        )

    ordered = [completed[int(item["index"])] for item in source_records]
    summary = _summarize(ordered)
    _write_output(args.output, metadata, summary, ordered)
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def _metadata(
    args: argparse.Namespace,
    query_cache: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": (
            "retriever_only_danke_multisource_v2"
            if args.danke_input_mode == "multi_source"
            else "retriever_only_fair_budget_v1"
        ),
        "query_cache": {
            "path": str(args.query_cache.resolve()),
            "sha256": _sha256(args.query_cache),
            "orchestrator": query_cache.get("metadata", {}).get("orchestrator"),
        },
        "selection": {
            "indices": [int(record["index"]) for record in records],
            "count": len(records),
            "db_counts": dict(Counter(str(record["db_id"]) for record in records)),
        },
        "embedding": {
            "model": args.embedding_model,
            "base_url": args.embedding_base_url,
            "strict_top_k": True,
        },
        "danke": {
            "knowledge_schema_dir": str(args.knowledge_schema_dir.resolve()),
            "fuzzy_cutoff": args.fuzzy_cutoff,
            "max_ngram": args.max_ngram,
            "input_mode": args.danke_input_mode,
            "generic_property_gate": {
                "keys": sorted(GENERIC_PROPERTY_KEYS),
                "table_candidate_allowed": False,
                "column_evidence_within_selected_class_allowed": True,
            },
            "variants": VARIANTS,
        },
        "budgets": {
            "semantic_tables": args.semantic_table_budget,
            "semantic_columns": args.semantic_column_budget,
            "bridge_tables_separate": args.bridge_table_budget,
            "join_key_columns_separate": True,
        },
        "ranking_policy": {
            "danke_tables": (
                "exact match first, then best string similarity, entry type "
                "(value > property > class), number of hits, ontology class ranking, name"
            ),
            "danke_columns": (
                "direct matched property/value first, then exactness, similarity, "
                "ontology property ranking, identifier"
            ),
            "no_match_fallback": (
                "highest ontology class ranking, then class identifier, still capped "
                "by the semantic table budget"
            ),
            "multi_source_table_fusion": (
                "Deduplicate by ontology item, retain every source-level match as "
                "provenance, protect up to two direct class matches from the original "
                "question and structured sources, backfill that anchor from the "
                "original-question rank, then fill the remaining semantic-table "
                "budget from the merged rank."
            ),
        },
        "evaluation_contract": {
            "same_query": (
                "Embedding consumes the cached retrieval_query. DANKE consumes it "
                "alone in retrieval_query mode, or searches cached original question, "
                "retrieval_query, and selected plan fields independently in "
                "multi_source mode."
            ),
            "table_semantic": "semantic tables only; maximum 3 by default",
            "table_effective": (
                "semantic tables plus separately budgeted Steiner bridge tables"
            ),
            "column_semantic": (
                "semantic columns evaluated against gold columns after removing the "
                "declared-FK join-key proxy"
            ),
            "column_all": (
                "semantic columns plus separately emitted validated join-key columns "
                "evaluated against all gold columns"
            ),
            "gold_join_key_proxy": (
                "A declared Spider FK endpoint pair is considered a gold join-key pair "
                "when both endpoint columns occur in extracted gold columns. This is a "
                "proxy because the evaluator does not retain SQL clause roles."
            ),
            "sql_ex": (
                "Not measured: this is intentionally a retriever-only experiment."
            ),
        },
    }


def _embedding_retrieve(
    retriever: EmbeddingSchemaRetriever,
    query: str,
    schema: DatabaseSchema,
    args: argparse.Namespace,
) -> dict[str, Any]:
    table_names, table_vectors, column_keys, column_vectors = (
        retriever._schema_vectors(schema)
    )
    query_vectors = retriever.model.embed([query])
    if len(query_vectors) != 1:
        raise RuntimeError("embedding query vector가 정확히 하나 반환되지 않았습니다.")
    query_vector = query_vectors[0]
    table_ranking = _rank_vectors(query_vector, table_names, table_vectors)
    semantic_tables = [
        item["key"] for item in table_ranking[: args.semantic_table_budget]
    ]
    allowed_tables = set(semantic_tables)
    allowed_columns = [
        (key, vector)
        for key, vector in zip(column_keys, column_vectors)
        if key.split(".", 1)[0] in allowed_tables
    ]
    column_ranking = _rank_vectors(
        query_vector,
        [key for key, _ in allowed_columns],
        [vector for _, vector in allowed_columns],
    )
    semantic_columns = [
        item["key"] for item in column_ranking[: args.semantic_column_budget]
    ]
    return {
        "raw": {
            "table_ranking": table_ranking,
            "column_ranking_within_selected_tables": column_ranking,
        },
        "semantic_tables": semantic_tables,
        "semantic_columns": semantic_columns,
        "bridge_tables": [],
        "join_key_columns": [],
        "join_edges": [],
        "effective_tables": semantic_tables,
        "all_columns": semantic_columns,
    }


def _rank_vectors(
    query: list[float],
    keys: list[str],
    vectors: list[list[float]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {"key": key, "score": _cosine(query, vector)}
            for key, vector in zip(keys, vectors)
        ),
        key=lambda item: (-item["score"], item["key"]),
    )
    return [
        {"key": item["key"], "score": round(float(item["score"]), 8)}
        for item in ranked
    ]


def _danke_retrieve(
    *,
    query_sources: list[dict[str, Any]],
    schema: DatabaseSchema,
    knowledge_schema: KnowledgeSchema,
    mapping: RDBMapping,
    dictionary: Dictionary,
    use_steiner: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    matching = MatchingDiscoveryService(
        dictionary, fuzzy_cutoff=args.fuzzy_cutoff
    )
    source_searches: list[dict[str, Any]] = []
    evidence_by_item: dict[tuple[Any, ...], dict[str, Any]] = {}
    for source in query_sources:
        phrases = _candidate_phrases(str(source["text"]), args.max_ngram)
        matched_results = [
            result for result in matching.match(phrases) if result.matched
        ]
        source_searches.append(
            {
                **source,
                "candidate_phrases": phrases,
                "matched_phrase_count": len(matched_results),
            }
        )
        for result in matched_results:
            for entry in result.entries:
                similarity = difflib.SequenceMatcher(
                    None, result.keyword.casefold(), entry.key.casefold()
                ).ratio()
                provenance = {
                    "source_id": source["source_id"],
                    "source_type": source["source_type"],
                    "source_text": source["text"],
                    "locators": source.get("locators", []),
                    "keyword": result.keyword,
                    "dictionary_key": entry.key,
                    "exact": not result.fuzzy,
                    "similarity": round(similarity, 6),
                }
                item_key = (
                    entry.entry_type,
                    entry.target_class,
                    entry.target_property,
                    entry.value,
                )
                evidence = evidence_by_item.get(item_key)
                if evidence is None:
                    evidence = {
                        "entry_type": entry.entry_type,
                        "target_class": entry.target_class,
                        "target_property": entry.target_property,
                        "value": entry.value,
                        "provenance": [],
                    }
                    evidence_by_item[item_key] = evidence
                provenance_key = (
                    provenance["source_id"],
                    provenance["keyword"].casefold(),
                    provenance["dictionary_key"].casefold(),
                )
                if not any(
                    (
                        item["source_id"],
                        item["keyword"].casefold(),
                        item["dictionary_key"].casefold(),
                    )
                    == provenance_key
                    for item in evidence["provenance"]
                ):
                    evidence["provenance"].append(provenance)

    match_evidence: list[dict[str, Any]] = []
    generic_property_table_gate = bool(
        getattr(args, "generic_property_table_gate", True)
    )
    generic_property_keys = {
        _normalize_generic_key(str(item))
        for item in getattr(args, "generic_property_keys", GENERIC_PROPERTY_KEYS)
    }
    class_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    property_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evidence in evidence_by_item.values():
        best = max(
            evidence["provenance"],
            key=lambda item: (
                bool(item["exact"]),
                float(item["similarity"]),
                str(item["source_id"]),
            ),
        )
        evidence["keyword"] = best["keyword"]
        evidence["dictionary_key"] = best["dictionary_key"]
        evidence["exact"] = bool(best["exact"])
        evidence["similarity"] = float(best["similarity"])
        evidence["source_ids"] = list(
            dict.fromkeys(item["source_id"] for item in evidence["provenance"])
        )
        evidence["source_types"] = list(
            dict.fromkeys(item["source_type"] for item in evidence["provenance"])
        )
        match_evidence.append(evidence)
        evidence["generic_property"] = _is_generic_property_evidence(
            evidence, knowledge_schema, generic_property_keys
        )
        if not (
            generic_property_table_gate and evidence["generic_property"]
        ):
            class_hits[evidence["target_class"]].append(evidence)
        if evidence["target_property"]:
            property_hits[evidence["target_property"]].append(evidence)

    class_ranking = sorted(
        (
            _class_rank_record(name, hits, knowledge_schema)
            for name, hits in class_hits.items()
            if name in knowledge_schema.classes
        ),
        key=_class_sort_key,
    )
    no_match_fallback = not class_ranking
    if no_match_fallback:
        class_ranking = [
            {
                "class": name,
                "table": mapping.table_for(name),
                "exact": False,
                "best_similarity": 0.0,
                "best_entry_type": "none",
                "hit_count": 0,
                "ontology_ranking": knowledge_schema.classes[name].ranking,
                "fallback": True,
            }
            for name in sorted(
                knowledge_schema.classes,
                key=lambda item: (
                    -knowledge_schema.classes[item].ranking,
                    item,
                ),
            )
            if name in mapping.class_table
        ]

    source_class_rankings = _source_class_rankings(
        [
            item
            for item in match_evidence
            if not (
                generic_property_table_gate and item["generic_property"]
            )
        ],
        query_sources,
        knowledge_schema,
    )
    original_ranking = sorted(
        source_class_rankings.get("question", []),
        key=_original_anchor_sort_key,
    )
    protected_count = min(2, args.semantic_table_budget)
    protected_classes = [
        item["class"]
        for item in original_ranking
        if item.get("has_class_match")
    ][:protected_count]
    for source in query_sources:
        if len(protected_classes) >= protected_count:
            break
        source_id = str(source["source_id"])
        if source_id == "question":
            continue
        for item in sorted(
            source_class_rankings.get(source_id, []),
            key=_original_anchor_sort_key,
        ):
            name = item["class"]
            if item.get("has_class_match") and name not in protected_classes:
                protected_classes.append(name)
                break
    for item in original_ranking:
        if len(protected_classes) >= protected_count:
            break
        if item["class"] not in protected_classes:
            protected_classes.append(item["class"])
    semantic_classes = _unique(
        [
            *protected_classes,
            *[item["class"] for item in class_ranking],
        ]
    )[: args.semantic_table_budget]
    semantic_tables = [
        mapping.table_for(name)
        for name in semantic_classes
        if name in mapping.class_table
    ]
    property_ranking = _rank_properties(
        semantic_classes,
        property_hits,
        knowledge_schema,
        mapping,
    )
    semantic_columns = [
        item["column"]
        for item in property_ranking[: args.semantic_column_budget]
    ]

    trees = (
        steiner_forest(
            build_knowledge_graph(knowledge_schema),
            semantic_classes,
        )
        if use_steiner and len(semantic_classes) >= 2
        else []
    )
    raw_bridge_classes = _ordered_bridge_classes(trees, semantic_classes)
    bridge_classes = raw_bridge_classes[: args.bridge_table_budget]
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
            left, right = mapping.fk_for(op_name)
            validated = declared_fk(schema, left, right)
            edge = {
                "left_class": left_class,
                "right_class": right_class,
                "object_property": op_name,
                "proposed_left": left,
                "proposed_right": right,
                "validated_fk": list(validated) if validated else None,
                "validated": validated is not None,
                "inferred": knowledge_schema.object_properties[op_name].inferred,
                "weight": weight,
            }
            join_edges.append(edge)
            if validated:
                for key in validated:
                    if key not in join_key_columns:
                        join_key_columns.append(key)

    raw_semantic_classes = [item["class"] for item in class_ranking]
    raw_trees = (
        steiner_forest(
            build_knowledge_graph(knowledge_schema),
            raw_semantic_classes,
        )
        if use_steiner and len(raw_semantic_classes) >= 2
        else []
    )
    raw_bridge = _ordered_bridge_classes(raw_trees, raw_semantic_classes)
    raw_tables = _unique(
        [
            *[
                mapping.table_for(name)
                for name in raw_semantic_classes
                if name in mapping.class_table
            ],
            *[
                mapping.table_for(name)
                for name in raw_bridge
                if name in mapping.class_table
            ],
        ]
    )
    raw_columns = _unique(
        [
            f"{table}.{column}"
            for name in raw_semantic_classes
            for prop in knowledge_schema.properties_of(name)
            for table, column in [mapping.column_for(prop.name)]
        ]
    )
    effective_tables = _unique([*semantic_tables, *bridge_tables])
    all_columns = _unique([*semantic_columns, *join_key_columns])
    return {
        "raw": {
            "query_sources": query_sources,
            "source_searches": source_searches,
            "deduplication_key": (
                "entry_type,target_class,target_property,value; all source-level "
                "matches are retained under provenance"
            ),
            "generic_property_only_matches": [
                item for item in match_evidence if item["generic_property"]
            ],
            "matches": match_evidence,
            "class_ranking": class_ranking,
            "class_rankings_by_source": source_class_rankings,
            "protected_original_classes": protected_classes,
            "property_ranking_within_selected_tables": property_ranking,
            "unbudgeted_tables": raw_tables,
            "unbudgeted_columns": raw_columns,
            "steiner_forest": [tree.to_dict() for tree in raw_trees],
        },
        "no_match_fallback": no_match_fallback,
        "semantic_tables": semantic_tables,
        "semantic_columns": semantic_columns,
        "bridge_tables": bridge_tables,
        "bridge_candidates_before_budget": [
            mapping.table_for(name)
            for name in raw_bridge_classes
            if name in mapping.class_table
        ],
        "bridge_budget_truncated": len(raw_bridge_classes) > len(bridge_classes),
        "join_key_columns": join_key_columns,
        "join_edges": join_edges,
        "effective_tables": effective_tables,
        "all_columns": all_columns,
    }


def _class_rank_record(
    name: str,
    hits: list[dict[str, Any]],
    knowledge_schema: KnowledgeSchema,
) -> dict[str, Any]:
    type_rank = {"value": 3, "property": 2, "class": 1}
    best = max(
        hits,
        key=lambda item: (
            bool(item["exact"]),
            float(item["similarity"]),
            type_rank.get(str(item["entry_type"]), 0),
        ),
    )
    class_matches = [item for item in hits if item["entry_type"] == "class"]
    best_class = max(
        class_matches,
        key=lambda item: (
            bool(item["exact"]),
            float(item["similarity"]),
        ),
        default=None,
    )
    return {
        "class": name,
        "table": knowledge_schema.classes[name].source_table,
        "exact": bool(best["exact"]),
        "best_similarity": float(best["similarity"]),
        "best_entry_type": str(best["entry_type"]),
        "has_class_match": best_class is not None,
        "best_class_exact": bool(best_class and best_class["exact"]),
        "best_class_similarity": (
            float(best_class["similarity"]) if best_class else 0.0
        ),
        "hit_count": len(hits),
        "ontology_ranking": knowledge_schema.classes[name].ranking,
        "fallback": False,
    }


def _is_generic_property_evidence(
    evidence: dict[str, Any],
    knowledge_schema: KnowledgeSchema,
    generic_property_keys: set[str],
) -> bool:
    del knowledge_schema
    if evidence.get("entry_type") != "property":
        return False
    matched_dictionary_keys = {
        _normalize_generic_key(str(item.get("dictionary_key") or ""))
        for item in evidence["provenance"]
    }
    return bool(matched_dictionary_keys) and all(
        key in generic_property_keys for key in matched_dictionary_keys
    )


def _normalize_generic_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _class_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    type_rank = {"value": 3, "property": 2, "class": 1, "none": 0}
    return (
        -int(bool(item["exact"])),
        -float(item["best_similarity"]),
        -type_rank.get(str(item["best_entry_type"]), 0),
        -int(item["hit_count"]),
        -float(item["ontology_ranking"]),
        str(item["class"]),
    )


def _original_anchor_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(item.get("has_class_match"))),
        -int(bool(item.get("best_class_exact"))),
        -float(item.get("best_class_similarity", 0.0)),
        *_class_sort_key(item),
    )


def _source_class_rankings(
    evidence: list[dict[str, Any]],
    query_sources: list[dict[str, Any]],
    knowledge_schema: KnowledgeSchema,
) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for source in query_sources:
        source_id = str(source["source_id"])
        class_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in evidence:
            candidates = [
                match
                for match in item["provenance"]
                if str(match["source_id"]) == source_id
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda match: (
                    bool(match["exact"]),
                    float(match["similarity"]),
                    str(match["dictionary_key"]),
                ),
            )
            class_hits[str(item["target_class"])].append(
                {
                    **item,
                    "keyword": best["keyword"],
                    "dictionary_key": best["dictionary_key"],
                    "exact": bool(best["exact"]),
                    "similarity": float(best["similarity"]),
                }
            )
        rankings[source_id] = sorted(
            (
                _class_rank_record(name, hits, knowledge_schema)
                for name, hits in class_hits.items()
                if name in knowledge_schema.classes
            ),
            key=_class_sort_key,
        )
    return rankings


def _rank_properties(
    semantic_classes: list[str],
    property_hits: dict[str, list[dict[str, Any]]],
    knowledge_schema: KnowledgeSchema,
    mapping: RDBMapping,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for class_name in semantic_classes:
        for prop in knowledge_schema.properties_of(class_name):
            hits = property_hits.get(prop.name, [])
            best = max(
                hits,
                key=lambda item: (
                    bool(item["exact"]),
                    float(item["similarity"]),
                ),
                default=None,
            )
            table, column = mapping.column_for(prop.name)
            records.append(
                {
                    "property": prop.name,
                    "column": f"{table}.{column}",
                    "direct_match": best is not None,
                    "exact": bool(best and best["exact"]),
                    "best_similarity": (
                        float(best["similarity"]) if best else 0.0
                    ),
                    "ontology_ranking": prop.ranking,
                }
            )
    return sorted(
        records,
        key=lambda item: (
            -int(item["direct_match"]),
            -int(item["exact"]),
            -float(item["best_similarity"]),
            -float(item["ontology_ranking"]),
            str(item["column"]),
        ),
    )


def _ordered_bridge_classes(
    trees: list[Any],
    semantic_classes: list[str],
) -> list[str]:
    semantic = set(semantic_classes)
    bridges: list[str] = []
    for tree in trees:
        for left, right, _op, _weight in tree.edges:
            for name in (left, right):
                if name not in semantic and name not in bridges:
                    bridges.append(name)
    return bridges


def _build_dictionary(
    schema: KnowledgeSchema,
    *,
    use_synonyms: bool,
) -> Dictionary:
    dictionary = Dictionary()
    for knowledge_class in schema.classes.values():
        dictionary.add(
            DictionaryEntry(
                "class",
                knowledge_class.primary_label,
                knowledge_class.name,
            )
        )
        if use_synonyms:
            for synonym in knowledge_class.synonyms:
                dictionary.add(
                    DictionaryEntry(
                        "class", synonym, knowledge_class.name
                    )
                )
    for prop in schema.datatype_properties.values():
        dictionary.add(
            DictionaryEntry(
                "property",
                prop.primary_label,
                prop.domain,
                prop.name,
            )
        )
        if use_synonyms:
            for synonym in prop.synonyms:
                dictionary.add(
                    DictionaryEntry(
                        "property",
                        synonym,
                        prop.domain,
                        prop.name,
                    )
                )
        if not prop.indexed:
            continue
        for value, synonyms in prop.value_synonyms.items():
            dictionary.add(
                DictionaryEntry(
                    "value", value, prop.domain, prop.name, value
                )
            )
            if use_synonyms:
                for synonym in synonyms:
                    dictionary.add(
                        DictionaryEntry(
                            "value",
                            synonym,
                            prop.domain,
                            prop.name,
                            value,
                        )
                    )
    return dictionary


def _evaluate_record(
    record: dict[str, Any],
    gold_tables: set[str],
    gold_columns: set[str],
    gold_semantic_columns: set[str],
    gold_join_keys: set[str],
) -> dict[str, Any]:
    evaluation = {
        "embedding": _evaluate_output(
            record["embedding"],
            gold_tables,
            gold_columns,
            gold_semantic_columns,
            gold_join_keys,
        ),
        "danke": {},
    }
    for name, output in record["danke"].items():
        evaluation["danke"][name] = _evaluate_output(
            output,
            gold_tables,
            gold_columns,
            gold_semantic_columns,
            gold_join_keys,
        )
        evaluation["danke"][name]["raw_tables"] = _metric(
            set(output["raw"]["unbudgeted_tables"]), gold_tables
        )
        evaluation["danke"][name]["raw_columns"] = _metric(
            set(output["raw"]["unbudgeted_columns"]), gold_columns
        )
    return evaluation


def _evaluate_output(
    output: dict[str, Any],
    gold_tables: set[str],
    gold_columns: set[str],
    gold_semantic_columns: set[str],
    gold_join_keys: set[str],
) -> dict[str, Any]:
    return {
        "semantic_tables": _metric(
            set(output["semantic_tables"]), gold_tables
        ),
        "effective_tables": _metric(
            set(output["effective_tables"]), gold_tables
        ),
        "semantic_columns": _metric(
            set(output["semantic_columns"]), gold_semantic_columns
        ),
        "join_key_columns": _metric(
            set(output["join_key_columns"]), gold_join_keys
        ),
        "all_columns": _metric(
            set(output["all_columns"]), gold_columns
        ),
    }


def _metric(predicted: set[str], gold: set[str]) -> dict[str, Any]:
    recall = len(predicted & gold) / len(gold) if gold else None
    precision = (
        len(predicted & gold) / len(predicted)
        if predicted
        else (1.0 if not gold else 0.0)
    )
    return {
        "recall": round(recall, 6) if recall is not None else None,
        "precision": round(precision, 6),
        "strict_recall": gold <= predicted,
        "missing": sorted(gold - predicted),
        "extra": sorted(predicted - gold),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    method_metrics: dict[str, Any] = {
        "embedding": _summarize_method(
            [record["evaluation"]["embedding"] for record in records],
            [record["embedding"] for record in records],
        ),
        "danke": {},
    }
    for variant in VARIANTS:
        method_metrics["danke"][variant] = _summarize_method(
            [
                record["evaluation"]["danke"][variant]
                for record in records
            ],
            [record["danke"][variant] for record in records],
        )

    outcomes: dict[str, Any] = {}
    axes = {
        "table_semantic": "semantic_tables",
        "table_effective": "effective_tables",
        "column_semantic": "semantic_columns",
        "column_all": "all_columns",
    }
    for label, metric_name in axes.items():
        groups = {
            "both_complete": [],
            "danke_only": [],
            "embedding_only": [],
            "both_fail": [],
        }
        for record in records:
            embed_success = bool(
                record["evaluation"]["embedding"][metric_name][
                    "strict_recall"
                ]
            )
            danke_success = bool(
                record["evaluation"]["danke"]["full"][metric_name][
                    "strict_recall"
                ]
            )
            group = _outcome_group(danke_success, embed_success)
            groups[group].append(int(record["index"]))
        outcomes[label] = {
            name: {"count": len(indices), "indices": indices}
            for name, indices in groups.items()
        }

    failures = [
        record
        for record in records
        if not record["evaluation"]["danke"]["full"]["effective_tables"][
            "strict_recall"
        ]
    ]
    rescued = [
        int(record["index"])
        for record in failures
        if record["evaluation"]["embedding"]["effective_tables"][
            "strict_recall"
        ]
    ]
    both_failed = [
        int(record["index"])
        for record in failures
        if not record["evaluation"]["embedding"]["effective_tables"][
            "strict_recall"
        ]
    ]

    return {
        "status": "complete",
        "examples": len(records),
        "db_counts": dict(Counter(record["db_id"] for record in records)),
        "metrics": method_metrics,
        "four_way_outcomes": outcomes,
        "danke_effective_strict_table_failures": {
            "count": len(failures),
            "indices": [int(record["index"]) for record in failures],
            "rescued_by_embedding_count": len(rescued),
            "rescued_by_embedding_indices": rescued,
            "failed_by_both_count": len(both_failed),
            "failed_by_both_indices": both_failed,
        },
        "interpretation_guard": (
            "SQL EX is intentionally absent. Hybrid necessity should be judged "
            "from four-way retrieval outcomes, budget-matched precision/recall, "
            "and the full/no_steiner/primary_only factorial ablation."
        ),
    }


def _summarize_method(
    evaluations: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = [
        "semantic_tables",
        "effective_tables",
        "semantic_columns",
        "join_key_columns",
        "all_columns",
    ]
    if evaluations and "raw_tables" in evaluations[0]:
        metric_names.extend(["raw_tables", "raw_columns"])
    summary: dict[str, Any] = {}
    for name in metric_names:
        recalls = [
            item[name]["recall"]
            for item in evaluations
            if item[name]["recall"] is not None
        ]
        precisions = [item[name]["precision"] for item in evaluations]
        strict = [bool(item[name]["strict_recall"]) for item in evaluations]
        summary[name] = {
            "mean_recall": _mean_or_none(recalls),
            "mean_precision": _mean_or_none(precisions),
            "strict_recall_rate": _mean_or_none(
                [float(value) for value in strict]
            ),
            "evaluated_recall_count": len(recalls),
        }
    summary["average_candidate_counts"] = {
        "semantic_tables": _mean_or_none(
            [len(item["semantic_tables"]) for item in outputs]
        ),
        "semantic_columns": _mean_or_none(
            [len(item["semantic_columns"]) for item in outputs]
        ),
        "bridge_tables": _mean_or_none(
            [len(item["bridge_tables"]) for item in outputs]
        ),
        "join_key_columns": _mean_or_none(
            [len(item["join_key_columns"]) for item in outputs]
        ),
        "effective_tables": _mean_or_none(
            [len(item["effective_tables"]) for item in outputs]
        ),
        "all_columns": _mean_or_none(
            [len(item["all_columns"]) for item in outputs]
        ),
    }
    return summary


def _outcome_group(danke_success: bool, embedding_success: bool) -> str:
    if danke_success and embedding_success:
        return "both_complete"
    if danke_success:
        return "danke_only"
    if embedding_success:
        return "embedding_only"
    return "both_fail"


def _gold_join_key_proxy(
    schema: DatabaseSchema,
    gold_columns: set[str],
) -> set[str]:
    result: set[str] = set()
    for left, right in schema.foreign_keys:
        if left in gold_columns and right in gold_columns:
            result.update((left, right))
    return result


def _danke_query_sources(
    source: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(source_type: str, locator: str, value: Any) -> None:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return
        text = str(value).strip()
        if not text:
            return
        records.append(
            {
                "source_id": locator,
                "source_type": source_type,
                "text": text,
                "locators": [locator],
            }
        )

    add("retrieval_query", "retrieval_query", source.get("retrieval_query"))
    if mode == "retrieval_query":
        return records

    records = []
    add("original_question", "question", source.get("question"))
    add("retrieval_query", "retrieval_query", source.get("retrieval_query"))
    plan = source.get("decomposition") or {}
    for index, output in enumerate(plan.get("outputs") or []):
        if isinstance(output, dict):
            add("plan_output_span", f"decomposition.outputs[{index}].span", output.get("span"))
        else:
            add("plan_output_span", f"decomposition.outputs[{index}]", output)
    for index, item in enumerate(plan.get("filters") or []):
        if not isinstance(item, dict):
            add("plan_filter_span", f"decomposition.filters[{index}]", item)
            continue
        for field in ("span", "entity", "value_mention"):
            add(
                f"plan_filter_{field}",
                f"decomposition.filters[{index}].{field}",
                item.get(field),
            )
    for field in ("grouping", "ordering", "relationships"):
        for index, item in enumerate(plan.get(field) or []):
            if isinstance(item, dict):
                for subfield in (
                    "span",
                    "entity",
                    "subject",
                    "predicate",
                    "object",
                    "from",
                    "to",
                    "type",
                ):
                    add(
                        f"plan_{field}_{subfield}",
                        f"decomposition.{field}[{index}].{subfield}",
                        item.get(subfield),
                    )
            else:
                add(
                    f"plan_{field}",
                    f"decomposition.{field}[{index}]",
                    item,
                )
    return records


def _candidate_phrases(query: str, max_ngram: int) -> list[str]:
    words = WORD_PATTERN.findall(query)
    phrases: list[str] = []
    for width in range(1, max(1, max_ngram) + 1):
        for index in range(len(words) - width + 1):
            phrase = " ".join(words[index : index + width])
            if len(phrase) >= 2:
                phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def _load_kg(
    db_id: str,
    root: Path,
    cache: dict[str, tuple[KnowledgeSchema, RDBMapping]],
) -> tuple[KnowledgeSchema, RDBMapping]:
    if db_id in cache:
        return cache[db_id]
    schema_path = root / f"{db_id}_knowledge_schema.json"
    mapping_path = root / f"{db_id}_mapping.json"
    if not schema_path.is_file() or not mapping_path.is_file():
        raise SystemExit(
            f"DANKE artifact 누락: db_id={db_id!r}, "
            f"schema={schema_path}, mapping={mapping_path}"
        )
    knowledge_schema = KnowledgeSchema.from_dict(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    mapping = RDBMapping.from_dict(
        json.loads(mapping_path.read_text(encoding="utf-8"))
    )
    cache[db_id] = (knowledge_schema, mapping)
    return cache[db_id]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(
            f"embedding dimension mismatch: {len(left)} != {len(right)}"
        )
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _write_output(
    path: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary": summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
