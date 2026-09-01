from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = Path(__file__).resolve().parent
MULTI_AGENT_DIR = WORKSPACE / "schema linking" / "multi-agent"
sys.path.insert(0, str(ONTOLOGY_DIR))
sys.path.insert(0, str(MULTI_AGENT_DIR))

from danke_kg.mapping import RDBMapping  # noqa: E402
from danke_kg.models import KnowledgeSchema  # noqa: E402
from danke_kg.multiagent_bridge import DankeAugmentedSchemaLinker  # noqa: E402

from schema_agents.agentic_orchestrator import AgenticMultiAgentSchemaLinker  # noqa: E402
from schema_agents.data import (  # noqa: E402
    database_path,
    default_database_root,
    default_tables_path,
    load_schemas,
)
from schema_agents.evaluation import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the DANKE-KG-backed retriever (DankeAugmentedSchemaLinker) "
            "against the original embedding retriever (AgenticMultiAgentSchemaLinker) "
            "over the first N rows of Spider-Ko validation.csv, across whichever "
            "db_ids those rows reference, scoring both against gold SQL via "
            "schema_agents.evaluation.evaluate()."
        )
    )
    parser.add_argument("--limit", type=int, default=100, help="First N rows of validation.csv")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--knowledge-schema-dir",
        type=Path,
        default=ONTOLOGY_DIR / "output" / "spider_ko",
        help="Directory containing <db_id>_knowledge_schema.json / _mapping.json",
    )
    parser.add_argument("--config", type=Path, default=MULTI_AGENT_DIR / "config.json")
    parser.add_argument(
        "--compare-embedding",
        action="store_true",
        default=True,
        help="Also run the original AgenticMultiAgentSchemaLinker (embedding retriever) for comparison.",
    )
    parser.add_argument(
        "--danke-only",
        action="store_true",
        help="Skip the embedding-retriever comparison run (DANKE retriever only).",
    )
    parser.add_argument(
        "--output", type=Path, default=ONTOLOGY_DIR / "output" / "multiagent_experiment_100.json"
    )
    return parser.parse_args()


def load_questions(limit: int, offset: int) -> list[dict[str, str]]:
    csv_path = WORKSPACE / "data" / "hugging face" / "Spider 1.0 ko" / "validation.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[offset : offset + limit]


def _mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def main() -> None:
    args = parse_args()
    compare_embedding = args.compare_embedding and not args.danke_only

    with args.config.open(encoding="utf-8") as handle:
        full_config = json.load(handle)
    linking_config = dict(full_config.get("schema_linking", full_config))
    # Override in-memory only (never touch files under `schema linking/`):
    # the LLM server has been slow under load, so give every call more room.
    linking_config["timeout_seconds"] = 120
    if isinstance(linking_config.get("embedding"), dict):
        linking_config["embedding"] = {**linking_config["embedding"], "timeout_seconds": 120}
    # The shared port-8000 vLLM instance now serves a different model
    # (Qwen/Qwen3-8B) than schema_linking/config.json requests
    # (Qwen/Qwen3-4B-Instruct-2507) -- externally restarted, out of our
    # control. Point at our own dedicated instance serving the exact model
    # the config already asks for, in-memory only.
    linking_config["model"] = "Qwen/Qwen3-4B-Instruct-2507"
    linking_config["base_url"] = "http://localhost:8000/v1"

    questions = load_questions(args.limit, args.offset)
    if not questions:
        raise SystemExit("선택된 validation.csv 구간에 질문이 없습니다.")
    db_ids = sorted({row["db_id"] for row in questions})
    print(f"questions={len(questions)} unique_db_ids={db_ids} compare_embedding={compare_embedding}")

    rel_schemas = load_schemas(default_tables_path("validation"))
    danke_linkers: dict[str, DankeAugmentedSchemaLinker] = {}
    rel_schema_by_db: dict[str, Any] = {}
    db_file_by_db: dict[str, Path] = {}
    missing = []
    for db_id in db_ids:
        schema_path = args.knowledge_schema_dir / f"{db_id}_knowledge_schema.json"
        mapping_path = args.knowledge_schema_dir / f"{db_id}_mapping.json"
        if not schema_path.is_file() or not mapping_path.is_file():
            missing.append(db_id)
            continue
        if db_id not in rel_schemas:
            raise SystemExit(f"tables JSON에 db_id={db_id!r}가 없습니다.")
        danke_schema = KnowledgeSchema.from_dict(json.loads(schema_path.read_text(encoding="utf-8")))
        danke_mapping = RDBMapping.from_dict(json.loads(mapping_path.read_text(encoding="utf-8")))
        danke_linkers[db_id] = DankeAugmentedSchemaLinker(linking_config, danke_schema, danke_mapping)
        rel_schema_by_db[db_id] = rel_schemas[db_id]
        db_file_by_db[db_id] = database_path(default_database_root("validation"), db_id)
    if missing:
        raise SystemExit(
            f"다음 db_id의 DANKE knowledge_schema/mapping이 없습니다: {missing}. "
            f"먼저 build_ontology.py --db-id <id> --no-synonyms 로 만들어주세요."
        )

    embedding_linker = AgenticMultiAgentSchemaLinker(linking_config) if compare_embedding else None

    results: list[dict[str, Any]] = []
    danke_table_recalls: list[float] = []
    danke_table_precisions: list[float] = []
    danke_strict_table: list[bool] = []
    danke_column_recalls: list[float] = []
    embed_table_recalls: list[float] = []
    embed_table_precisions: list[float] = []
    embed_strict_table: list[bool] = []
    embed_column_recalls: list[float] = []
    agreements = 0
    per_db_counts: dict[str, int] = defaultdict(int)
    failures = 0

    for index, row in enumerate(questions, start=args.offset):
        db_id = row["db_id"]
        question = row.get("question_ko") or row.get("question", "")
        gold_sql = row.get("query", "")
        per_db_counts[db_id] += 1
        rel_schema = rel_schema_by_db[db_id]
        db_file = db_file_by_db[db_id]
        record: dict[str, Any] = {"index": index, "db_id": db_id, "question": question, "gold_sql": gold_sql}
        print(f"\n[{index}] ({db_id}) {question}")

        try:
            start = time.monotonic()
            danke_result = danke_linkers[db_id].link(question, rel_schema, db_file)
            danke_elapsed = time.monotonic() - start
            danke_score = evaluate(danke_result, rel_schema, gold_sql)
            record["danke"] = {
                "elapsed_sec": round(danke_elapsed, 2),
                "selected_tables": danke_result.tables,
                "selected_columns": danke_result.columns,
                "table_recall": danke_score.table_recall,
                "table_precision": danke_score.table_precision,
                "strict_table_recall": danke_score.strict_table_recall,
                "column_recall": danke_score.column_recall,
                "gold_tables": sorted(danke_score.gold_tables),
            }
            danke_table_recalls.append(danke_score.table_recall)
            danke_table_precisions.append(danke_score.table_precision)
            danke_strict_table.append(danke_score.strict_table_recall)
            if danke_score.column_recall is not None:
                danke_column_recalls.append(danke_score.column_recall)
            print(
                f"  [DANKE]     tables={danke_result.tables} recall={danke_score.table_recall:.2f} "
                f"precision={danke_score.table_precision:.2f} "
                f"strict={danke_score.strict_table_recall} ({danke_elapsed:.1f}s)"
            )
        except Exception as error:  # noqa: BLE001 - keep the 100-question batch going
            failures += 1
            record["danke_error"] = str(error)
            print(f"  [DANKE]     ERROR: {error}")

        if embedding_linker is not None:
            try:
                start = time.monotonic()
                embed_result = embedding_linker.link(question, rel_schema, db_file)
                embed_elapsed = time.monotonic() - start
                embed_score = evaluate(embed_result, rel_schema, gold_sql)
                record["embedding"] = {
                    "elapsed_sec": round(embed_elapsed, 2),
                    "selected_tables": embed_result.tables,
                    "selected_columns": embed_result.columns,
                    "table_recall": embed_score.table_recall,
                    "table_precision": embed_score.table_precision,
                    "strict_table_recall": embed_score.strict_table_recall,
                    "column_recall": embed_score.column_recall,
                }
                embed_table_recalls.append(embed_score.table_recall)
                embed_table_precisions.append(embed_score.table_precision)
                embed_strict_table.append(embed_score.strict_table_recall)
                if embed_score.column_recall is not None:
                    embed_column_recalls.append(embed_score.column_recall)
                if "danke" in record:
                    agree = set(record["danke"]["selected_tables"]) == set(embed_result.tables)
                    record["tables_agree"] = agree
                    agreements += int(agree)
                print(
                    f"  [EMBEDDING] tables={embed_result.tables} recall={embed_score.table_recall:.2f} "
                    f"precision={embed_score.table_precision:.2f} "
                    f"strict={embed_score.strict_table_recall} ({embed_elapsed:.1f}s)"
                )
            except Exception as error:  # noqa: BLE001
                record["embedding_error"] = str(error)
                print(f"  [EMBEDDING] ERROR: {error}")

        results.append(record)

    summary = {
        "questions": len(questions),
        "unique_db_ids": db_ids,
        "per_db_question_counts": dict(per_db_counts),
        "danke_failures": failures,
        "danke_mean_table_recall": _mean_or_none(danke_table_recalls),
        "danke_mean_table_precision": _mean_or_none(danke_table_precisions),
        "danke_strict_table_recall_rate": _mean_or_none([float(v) for v in danke_strict_table]),
        "danke_mean_column_recall": _mean_or_none(danke_column_recalls),
    }
    if embedding_linker is not None:
        summary.update(
            {
                "embedding_mean_table_recall": _mean_or_none(embed_table_recalls),
                "embedding_mean_table_precision": _mean_or_none(embed_table_precisions),
                "embedding_strict_table_recall_rate": _mean_or_none([float(v) for v in embed_strict_table]),
                "embedding_mean_column_recall": _mean_or_none(embed_column_recalls),
                "selected_table_agreement_rate": round(agreements / len(questions), 4),
            }
        )

    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, ensure_ascii=False, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
