from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from danke_kg.dictionary import build_dictionary
from danke_kg.models import KnowledgeSchema

ONTOLOGY_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a human-curated synonym file into already-built "
            "<db_id>_knowledge_schema.json files. Two entry shapes are "
            "supported and auto-detected per entry: column-level "
            "{db_id, table, column, readable_name, synonyms} matched to a "
            "datatype_property by (source_table, source_column), and "
            "table-level {db_id, table, readable_name, synonyms} (no "
            "'column' key) matched to a class by source_table. Only the "
            "synonyms list is updated (existing synonyms kept, new ones "
            "appended, deduped) -- primary_label is left untouched. The "
            "dictionary.json for each db_id is then rebuilt from the "
            "updated schema so it reflects the new synonyms too."
        )
    )
    parser.add_argument("--synonyms-file", type=Path, default=ONTOLOGY_DIR / "synonym_v2.json")
    parser.add_argument("--kg-dir", type=Path, default=ONTOLOGY_DIR / "output" / "spider_ko")
    parser.add_argument("--db-ids", nargs="+", help="Only process these db_ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = json.loads(args.synonyms_file.read_text(encoding="utf-8"))

    by_db: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_db[entry["db_id"]].append(entry)

    db_ids = sorted(by_db)
    if args.db_ids:
        wanted = set(args.db_ids)
        db_ids = [d for d in db_ids if d in wanted]

    total_targets_updated = 0
    total_synonyms_added = 0
    missing_schema: list[str] = []
    unmatched: list[tuple[str, str, str]] = []

    for db_id in db_ids:
        schema_path = args.kg_dir / f"{db_id}_knowledge_schema.json"
        if not schema_path.is_file():
            missing_schema.append(db_id)
            continue

        raw = json.loads(schema_path.read_text(encoding="utf-8"))
        prop_index = {
            (prop["source_table"], prop["source_column"]): prop
            for prop in raw["datatype_properties"]
        }
        class_index = {cls["source_table"]: cls for cls in raw["classes"]}

        updated = 0
        added = 0
        for entry in by_db[db_id]:
            is_column_level = bool(entry.get("column"))
            if is_column_level:
                target = prop_index.get((entry["table"], entry["column"]))
                label = (db_id, entry["table"], entry["column"])
            else:
                target = class_index.get(entry["table"])
                label = (db_id, entry["table"], None)
            if target is None:
                unmatched.append(label)
                continue
            existing = list(target.get("synonyms", []))
            seen = set(existing)
            new_terms = [term for term in entry.get("synonyms", []) if term and term not in seen]
            if not new_terms:
                continue
            target["synonyms"] = existing + new_terms
            updated += 1
            added += len(new_terms)

        if updated:
            schema_path.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            dictionary_path = args.kg_dir / f"{db_id}_dictionary.json"
            dictionary = build_dictionary(KnowledgeSchema.from_dict(raw))
            dictionary_path.write_text(
                json.dumps(dictionary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        total_targets_updated += updated
        total_synonyms_added += added
        print(f"{db_id}: targets_updated={updated} synonyms_added={added}")

    print(
        f"\nTOTAL db_ids={len(db_ids)} targets_updated={total_targets_updated} "
        f"synonyms_added={total_synonyms_added} missing_schema={missing_schema} "
        f"unmatched_entries={len(unmatched)}"
    )
    if unmatched:
        for item in unmatched[:20]:
            print("  unmatched:", item)


if __name__ == "__main__":
    main()
