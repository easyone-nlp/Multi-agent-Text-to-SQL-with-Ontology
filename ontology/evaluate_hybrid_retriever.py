from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
MULTI_AGENT_DIR = WORKSPACE / "schema linking" / "multi-agent"
for path in (WORKSPACE, MULTI_AGENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from schema_agents.danke_multisource_retriever import (  # noqa: E402
    MultiSourceDankeRetriever,
)
from schema_agents.data import default_tables_path, load_schemas  # noqa: E402
from schema_agents.embedding_retriever import EmbeddingSchemaRetriever  # noqa: E402
from schema_agents.evaluation import extract_gold_links  # noqa: E402
from schema_agents.hybrid_danke_embedding_retriever import (  # noqa: E402
    HybridDankeEmbeddingRetriever,
)
from schema_agents.model_client import build_embedding_model  # noqa: E402
from schema_agents.models import DatabaseSchema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate evidence-tier DANKE+Embedding hybrid on cached queries."
    )
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--danke-baseline",
        type=Path,
        default=(
            WORKSPACE
            / "ontology/output/retriever_only_v2/"
            "multisource_generic_gate_top3_8_bridge2.json"
        ),
    )
    parser.add_argument(
        "--knowledge-schema-dir",
        type=Path,
        default=(
            WORKSPACE
            / "ontology/versions/danke-spider-ko-v1.0.0-20260728"
        ),
    )
    parser.add_argument("--tables", type=Path, default=default_tables_path("validation"))
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = json.loads(args.query_cache.read_text(encoding="utf-8"))
    source_records = list(cache.get("records", []))
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
        top_k_tables=3,
        top_k_columns=8,
        strict_top_k=True,
    )
    danke_config = {
        "knowledge_schema_dir": str(args.knowledge_schema_dir),
        "semantic_table_budget": 3,
        "semantic_column_budget": 8,
        "bridge_table_budget": 2,
        "fuzzy_cutoff": 0.72,
        "max_ngram": 3,
        "use_synonyms": True,
        "use_steiner": True,
        "generic_property_table_gate": True,
        "generic_property_keys": [
            "id",
            "identifier",
            "name",
            "아이디",
            "식별자",
            "이름",
        ],
    }
    hybrid = HybridDankeEmbeddingRetriever(danke_config, embedding)
    baseline = json.loads(args.danke_baseline.read_text(encoding="utf-8"))
    baseline_by_index = {
        int(record["index"]): record for record in baseline["records"]
    }
    records: list[dict[str, Any]] = []
    for position, source in enumerate(source_records, start=1):
        index = int(source["index"])
        schema = schemas[str(source["db_id"])]
        print(f"[{position}/{len(source_records)}] index={index} db_id={schema.db_id}")
        retrieved = hybrid.retrieve_multisource(
            str(source["question"]),
            dict(source.get("decomposition") or {}),
            schema,
        )
        metadata = retrieved.metadata or {}
        _validate_contract(retrieved.to_dict(), schema)
        gold_tables, gold_columns = extract_gold_links(
            schema,
            str(source.get("gold_sql") or ""),
            require_sqlglot=True,
        )
        gold_join_keys = _gold_join_key_proxy(schema, gold_columns)
        gold_semantic_columns = gold_columns - gold_join_keys
        baseline_record = baseline_by_index[index]
        baseline_output = baseline_record["danke"]["full"]
        record = {
            "index": index,
            "db_id": schema.db_id,
            "question": source["question"],
            "retrieval_query": source["retrieval_query"],
            "gold_sql": source.get("gold_sql"),
            "gold": {
                "tables": sorted(gold_tables),
                "columns": sorted(gold_columns),
                "semantic_columns": sorted(gold_semantic_columns),
                "join_key_columns_proxy": sorted(gold_join_keys),
            },
            "hybrid": retrieved.to_dict(),
            "danke_official": {
                "semantic_tables": baseline_output["semantic_tables"],
                "semantic_columns": baseline_output["semantic_columns"],
                "bridge_tables": baseline_output["bridge_tables"],
                "join_key_columns": baseline_output["join_key_columns"],
                "effective_tables": baseline_output["effective_tables"],
                "all_columns": baseline_output["all_columns"],
            },
        }
        record["evaluation"] = {
            "hybrid": _evaluate(
                metadata,
                gold_tables,
                gold_columns,
                gold_semantic_columns,
                gold_join_keys,
            ),
            "danke_official": _evaluate(
                baseline_output,
                gold_tables,
                gold_columns,
                gold_semantic_columns,
                gold_join_keys,
            ),
        }
        records.append(record)
    summary = _summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": {
                    "experiment": "danke_embedding_evidence_tier_hybrid_v1",
                    "query_cache": str(args.query_cache.resolve()),
                    "danke_baseline": str(args.danke_baseline.resolve()),
                    "embedding_model": args.embedding_model,
                    "embedding_base_url": args.embedding_base_url,
                    "budgets": {
                        "semantic_tables": 3,
                        "semantic_columns": 8,
                        "bridge_tables_separate": 2,
                        "join_keys_separate": True,
                    },
                    "score_addition": False,
                },
                "summary": summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def _validate_contract(output: dict[str, Any], schema: DatabaseSchema) -> None:
    metadata = output["metadata"]
    semantic_tables = list(metadata["semantic_tables"])
    semantic_columns = list(metadata["semantic_columns"])
    bridge_tables = list(metadata["bridge_tables"])
    if len(semantic_tables) > 3 or len(semantic_columns) > 8 or len(bridge_tables) > 2:
        raise RuntimeError("hybrid candidate budget violation")
    allowed_tables = set(semantic_tables)
    if any(column.split(".", 1)[0] not in allowed_tables for column in semantic_columns):
        raise RuntimeError("final semantic table 밖의 semantic column이 있습니다.")
    if set(semantic_tables) & set(bridge_tables):
        raise RuntimeError("bridge table이 semantic endpoint budget에 포함됐습니다.")
    if not set(output["tables"]).issubset(set(schema.tables)):
        raise RuntimeError("존재하지 않는 table이 선택됐습니다.")


def _gold_join_key_proxy(
    schema: DatabaseSchema, gold_columns: set[str]
) -> set[str]:
    result: set[str] = set()
    for left, right in schema.foreign_keys:
        if left in gold_columns and right in gold_columns:
            result.update((left, right))
    return result


def _evaluate(
    output: dict[str, Any],
    gold_tables: set[str],
    gold_columns: set[str],
    gold_semantic_columns: set[str],
    gold_join_keys: set[str],
) -> dict[str, Any]:
    semantic_tables = set(output["semantic_tables"])
    effective_tables = semantic_tables | set(output["bridge_tables"])
    semantic_columns = set(output["semantic_columns"])
    all_columns = semantic_columns | set(output["join_key_columns"])
    return {
        "semantic_tables": _metric(semantic_tables, gold_tables),
        "effective_tables": _metric(effective_tables, gold_tables),
        "semantic_columns": _metric(semantic_columns, gold_semantic_columns),
        "all_columns": _metric(all_columns, gold_columns),
        "join_key_columns": _metric(set(output["join_key_columns"]), gold_join_keys),
    }


def _metric(predicted: set[str], gold: set[str]) -> dict[str, Any]:
    intersection = predicted & gold
    return {
        "recall": len(intersection) / len(gold) if gold else None,
        "precision": len(intersection) / len(predicted) if predicted else (1.0 if not gold else 0.0),
        "strict": gold.issubset(predicted),
        "missing": sorted(gold - predicted),
        "extra": sorted(predicted - gold),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for method in ("danke_official", "hybrid"):
        metrics[method] = {}
        for name in (
            "semantic_tables",
            "effective_tables",
            "semantic_columns",
            "all_columns",
            "join_key_columns",
        ):
            items = [record["evaluation"][method][name] for record in records]
            recalls = [item["recall"] for item in items if item["recall"] is not None]
            metrics[method][name] = {
                "mean_recall": _mean(recalls),
                "mean_precision": _mean([item["precision"] for item in items]),
                "strict_recall_rate": _mean([float(item["strict"]) for item in items]),
            }
    comparisons: dict[str, Any] = {}
    for name in ("effective_tables", "semantic_columns", "all_columns"):
        rescued: list[int] = []
        regressed: list[int] = []
        both: list[int] = []
        neither: list[int] = []
        for record in records:
            danke = bool(record["evaluation"]["danke_official"][name]["strict"])
            hybrid = bool(record["evaluation"]["hybrid"][name]["strict"])
            if hybrid and not danke:
                rescued.append(record["index"])
            elif danke and not hybrid:
                regressed.append(record["index"])
            elif danke and hybrid:
                both.append(record["index"])
            else:
                neither.append(record["index"])
        comparisons[name] = {
            "rescued_by_hybrid": rescued,
            "regressed_from_danke": regressed,
            "both_strict": both,
            "neither_strict": neither,
        }
    return {
        "status": "complete",
        "examples": len(records),
        "db_counts": dict(Counter(record["db_id"] for record in records)),
        "metrics": metrics,
        "comparisons": comparisons,
    }


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


if __name__ == "__main__":
    main()
