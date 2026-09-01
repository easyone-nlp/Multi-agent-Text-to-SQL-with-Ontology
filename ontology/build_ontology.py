from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from danke_kg.dictionary import MatchingDiscoveryService, build_dictionary
from danke_kg.graph import build_knowledge_graph
from danke_kg.orchestrator import OntologyBuilder
from danke_kg.relational import database_path, default_database_root, default_tables_path, load_schemas
from danke_kg.view_synthesis import ViewSynthesisService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DANKE-like ontology/knowledge graph from a relational schema "
            "using the multi-agent enrichment pipeline, then optionally run a "
            "keyword query through Matching Discovery + Steiner-tree View Synthesis."
        )
    )
    parser.add_argument("--tables", type=Path, default=default_tables_path())
    parser.add_argument("--database-root", type=Path, default=default_database_root())
    parser.add_argument("--db-id", required=True, help="db_id in the tables JSON")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--provider", choices=("heuristic", "openai_compatible"))
    parser.add_argument("--model", help="Qwen model id served by the OpenAI-compatible endpoint")
    parser.add_argument("--base-url", help="Example: http://localhost:8000/v1")
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Force the direct-mapping baseline; skip all LLM enrichment calls.",
    )
    parser.add_argument(
        "--no-synonyms",
        action="store_true",
        help=(
            "Skip class/property/relationship synonym generation (to be curated "
            "manually later); value_synonyms are still generated since they encode "
            "DB business logic, not naming preference."
        ),
    )
    parser.add_argument("--max-domain-values", type=int)
    parser.add_argument("--fuzzy-cutoff", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Demo keyword query, e.g. --keywords 학생 이름 나이",
    )
    parser.add_argument("--only-indexed-columns", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    schemas = load_schemas(args.tables)
    if args.db_id not in schemas:
        raise SystemExit(f"tables JSON에 db_id={args.db_id!r}가 없습니다.")
    schema = schemas[args.db_id]
    db_file = database_path(args.database_root, args.db_id)
    if not db_file.is_file():
        db_file = None

    print(f"[1/4] building knowledge schema S^K for db_id={args.db_id!r} (provider={config.get('provider')})")
    builder = OntologyBuilder(config)
    result = builder.build(schema)
    print(
        f"  classes={len(result.schema.classes)} "
        f"datatype_properties={len(result.schema.datatype_properties)} "
        f"object_properties={len(result.schema.object_properties)}"
    )
    if args.show_trace:
        for event in result.trace:
            print("  agent_trace:", json.dumps(event, ensure_ascii=False)[:400])

    print(f"[2/4] building dictionary (metadata + data entries), db_file={db_file}")
    dictionary = build_dictionary(result.schema, db_file, max_domain_values)
    print(
        f"  metadata_entries={dictionary.metadata_entry_count} "
        f"data_entries={dictionary.data_entry_count}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = args.output_dir / f"{args.db_id}_knowledge_schema.json"
    mapping_path = args.output_dir / f"{args.db_id}_mapping.json"
    dictionary_path = args.output_dir / f"{args.db_id}_dictionary.json"
    _write_json(schema_path, result.schema.to_dict())
    _write_json(mapping_path, result.mapping.to_dict())
    _write_json(dictionary_path, dictionary.to_dict())
    print(f"[3/4] wrote {schema_path}, {mapping_path}, {dictionary_path}")

    if not args.keywords:
        print("[4/4] no --keywords given; skipping matching discovery / view synthesis demo")
        return

    print(f"[4/4] matching discovery for keywords={args.keywords}")
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
