from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from danke_kg.aihub import (
    build_combined_sqlite,
    combine_source,
    default_aihub_root,
    group_by_source,
    infer_join_columns,
    load_aihub_schemas,
    source_summary,
    validate_join_overlap,
)
from danke_kg.dictionary import MatchingDiscoveryService, build_dictionary
from danke_kg.graph import build_knowledge_graph
from danke_kg.orchestrator import OntologyBuilder
from danke_kg.view_synthesis import ViewSynthesisService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DANKE-like ontology/knowledge graph from AI Hub NL2SQL data "
            "(data/ai hub). Since almost every AI Hub db_id is an isolated "
            "single-table database with no declared foreign keys, this groups all "
            "tables published under the same `source` into one combined relational "
            "schema and heuristically infers join keys from shared code columns "
            "before running the same multi-agent enrichment pipeline as "
            "build_ontology.py."
        )
    )
    parser.add_argument("--data-root", type=Path, default=default_aihub_root())
    parser.add_argument("--split", choices=("Training", "Validation"), default="Validation")
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print available `source` groups (table/db counts) and exit.",
    )
    parser.add_argument("--source", help="Exact `source` value to build, e.g. 서울인구관")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--provider", choices=("heuristic", "openai_compatible"))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument(
        "--no-synonyms",
        action="store_true",
        help="Skip class/property/relationship synonym generation; value_synonyms are still generated.",
    )
    parser.add_argument(
        "--min-shared-tables",
        type=int,
        default=2,
        help="Minimum tables that must share a *_CD/_CODE/_ID/_NO/_KEY column name to infer a join.",
    )
    parser.add_argument(
        "--min-join-overlap",
        type=float,
        default=0.05,
        help="Minimum distinct-value overlap ratio (vs. the smaller side) for an inferred join to be kept.",
    )
    parser.add_argument("--max-domain-values", type=int)
    parser.add_argument("--fuzzy-cutoff", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--keywords", nargs="+", help="Demo keyword query, e.g. --keywords 인구 구별")
    parser.add_argument("--only-indexed-columns", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[0/5] scanning AI Hub {args.split} annotations under {args.data_root}")
    schemas = load_aihub_schemas(args.data_root, args.split)
    groups = group_by_source(schemas)

    if args.list_sources:
        for entry in source_summary(schemas):
            print(
                f"  source={entry['source']!r:40s} dbs={entry['db_count']:4d} "
                f"tables={entry['table_count']:4d} files={entry['annotation_files']}"
            )
        return

    if not args.source:
        raise SystemExit("--source가 필요합니다 (사용 가능한 값은 --list-sources로 확인하세요).")
    items = groups.get(args.source)
    if not items:
        raise SystemExit(
            f"source={args.source!r}를 찾지 못했습니다. --list-sources로 정확한 이름을 확인하세요."
        )
    print(f"  matched source={args.source!r}: {len(items)} db_id, "
          f"{sum(len(item.schema.tables) for item in items)} tables")

    config = _load_config(args.config)
    if args.heuristic_only:
        config["provider"] = "heuristic"
    if args.no_synonyms:
        config["include_synonyms"] = False
    for key in ("provider", "model", "base_url"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    max_domain_values = args.max_domain_values or int(config.get("max_domain_values", 20))
    fuzzy_cutoff = args.fuzzy_cutoff or float(config.get("fuzzy_cutoff", 0.72))

    print("[1/6] combining source tables into one relational schema")
    combined_schema, table_rename = combine_source(args.source, items)
    candidate_pairs = infer_join_columns(combined_schema, min_tables=args.min_shared_tables)
    print(
        f"  tables={len(combined_schema.tables)} columns={len(combined_schema.columns)} "
        f"declared_joins={len(combined_schema.foreign_keys)} "
        f"candidate_inferred_joins={len(candidate_pairs)} (name-matched, not yet value-checked)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = combined_schema.db_id
    sqlite_path = args.output_dir / f"{slug}_combined.sqlite"
    print(f"[2/6] merging {len(items)} individual sqlite files into {sqlite_path}")
    skipped = build_combined_sqlite(items, table_rename, sqlite_path)
    if skipped:
        print(f"  warning: {len(skipped)} table(s) could not be copied (missing/broken sqlite): {skipped[:10]}")

    print("[3/6] validating candidate joins against real data (same column name != same code system)")
    inferred_pairs, rejected_pairs = validate_join_overlap(
        sqlite_path, candidate_pairs, min_overlap=args.min_join_overlap
    )
    combined_schema.foreign_keys.extend(sorted(inferred_pairs))
    print(f"  kept={len(inferred_pairs)} rejected_no_value_overlap={len(rejected_pairs)}")
    if rejected_pairs:
        print(f"  rejected pairs (no/low value overlap): {sorted(rejected_pairs)[:10]}")

    print(f"[4/6] building knowledge schema S^K (provider={config.get('provider')})")
    builder = OntologyBuilder(config)
    result = builder.build(combined_schema, inferred_fk_pairs=inferred_pairs)
    print(
        f"  classes={len(result.schema.classes)} "
        f"datatype_properties={len(result.schema.datatype_properties)} "
        f"object_properties={len(result.schema.object_properties)}"
    )
    if args.show_trace:
        for event in result.trace:
            print("  agent_trace:", json.dumps(event, ensure_ascii=False)[:400])

    dictionary = build_dictionary(result.schema, sqlite_path, max_domain_values)
    print(
        f"  metadata_entries={dictionary.metadata_entry_count} "
        f"data_entries={dictionary.data_entry_count}"
    )

    schema_path = args.output_dir / f"{slug}_knowledge_schema.json"
    mapping_path = args.output_dir / f"{slug}_mapping.json"
    dictionary_path = args.output_dir / f"{slug}_dictionary.json"
    _write_json(schema_path, result.schema.to_dict())
    _write_json(mapping_path, result.mapping.to_dict())
    _write_json(dictionary_path, dictionary.to_dict())
    print(f"[5/6] wrote {schema_path}, {mapping_path}, {dictionary_path}")

    if not args.keywords:
        print("[6/6] no --keywords given; skipping matching discovery / view synthesis demo")
        return

    print(f"[6/6] matching discovery for keywords={args.keywords}")
    matching_service = MatchingDiscoveryService(dictionary, fuzzy_cutoff=fuzzy_cutoff)
    matches = matching_service.match(args.keywords)
    for match in matches:
        print(" ", json.dumps(match.to_dict(), ensure_ascii=False))

    matched_classes = matching_service.matched_classes(matches)
    if not matched_classes:
        print("  no dictionary entries matched; nothing to synthesize a view from")
        return

    graph = build_knowledge_graph(result.schema)
    view = ViewSynthesisService(result.schema, result.mapping, graph).synthesize(
        matched_classes, only_indexed_columns=args.only_indexed_columns
    )
    print(f"  matched classes S'={matched_classes}")
    if view.disconnected_classes:
        print(f"  disconnected (outside the largest Steiner tree): {view.disconnected_classes}")
    print("  synthesized view:")
    print(view.sql)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
