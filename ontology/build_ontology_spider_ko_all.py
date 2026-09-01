from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from danke_kg.dictionary import build_dictionary
from danke_kg.orchestrator import OntologyBuilder
from danke_kg.relational import database_path, load_schemas

WORKSPACE = Path(__file__).resolve().parents[1]
ONTOLOGY_DIR = Path(__file__).resolve().parent
SPIDER_DIR = WORKSPACE / "data" / "hugging face" / "Spider 1.0"
SPIDER_KO_DIR = WORKSPACE / "data" / "hugging face" / "Spider 1.0 ko"

TRAIN_TABLES = SPIDER_DIR / "train" / "train_tables.json"
TRAIN_DB_ROOT = SPIDER_DIR / "train" / "database"
DEV_TABLES = SPIDER_DIR / "dev" / "dev_table.json"
DEV_DB_ROOT = SPIDER_DIR / "dev" / "database"


def _spider_ko_db_ids() -> set[str]:
    """The exact 160 db_ids Spider-Ko's train.csv+validation.csv reference.

    Spider 1.0's own tables.json files additionally define 6 db_ids
    (academic, geo, imdb, restaurants, scholar, yelp) that no Spider-Ko
    question ever uses -- excluded so this script builds exactly the scope
    confirmed with the user (160 db_ids), not Spider 1.0's full 166.
    """
    db_ids: set[str] = set()
    for name in ("train.csv", "validation.csv"):
        with (SPIDER_KO_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
            db_ids.update(row["db_id"] for row in csv.DictReader(handle))
    return db_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DANKE KG for every db_id referenced by Spider-Ko "
            "(train.csv + validation.csv share no db_id, together 160 "
            "unique db_ids), writing each into a dedicated spider_ko/ "
            "output folder. Resumable: db_ids that already have all three "
            "output files are skipped unless --force is given."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=ONTOLOGY_DIR / "output" / "spider_ko")
    parser.add_argument("--config", type=Path, default=ONTOLOGY_DIR / "config.json")
    parser.add_argument(
        "--include-synonyms",
        action="store_true",
        help=(
            "Also generate table/column/relationship synonyms. Default is "
            "excluded (value_synonyms are still generated), matching the "
            "concert_singer/pets_1/car_1 KGs already built this way."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild db_ids that already have output files.")
    parser.add_argument("--limit", type=int, help="Only process the first N db_ids (smoke-testing).")
    parser.add_argument("--db-ids", nargs="+", help="Only process these specific db_ids.")
    parser.add_argument("--max-domain-values", type=int)
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    config["include_synonyms"] = bool(args.include_synonyms)
    max_domain_values = args.max_domain_values or int(config.get("max_domain_values", 20))

    train_schemas = load_schemas(TRAIN_TABLES)
    dev_schemas = load_schemas(DEV_TABLES)
    overlap = set(train_schemas) & set(dev_schemas)
    if overlap:
        raise SystemExit(f"train/dev db_id overlap (unexpected, please check the data): {sorted(overlap)}")

    all_schemas = {**train_schemas, **dev_schemas}
    db_roots = {db_id: TRAIN_DB_ROOT for db_id in train_schemas}
    db_roots.update({db_id: DEV_DB_ROOT for db_id in dev_schemas})

    spider_ko_ids = _spider_ko_db_ids()
    missing_schema = spider_ko_ids - set(all_schemas)
    if missing_schema:
        raise SystemExit(f"Spider-Ko references db_id(s) with no tables.json entry: {sorted(missing_schema)}")

    db_ids = sorted(spider_ko_ids)
    if args.db_ids:
        wanted = set(args.db_ids)
        missing = wanted - set(db_ids)
        if missing:
            raise SystemExit(f"unknown db_id(s): {sorted(missing)}")
        db_ids = [d for d in db_ids if d in wanted]
    if args.limit:
        db_ids = db_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = OntologyBuilder(config)

    print(
        f"targets={len(db_ids)} include_synonyms={config['include_synonyms']} "
        f"output_dir={args.output_dir} provider={config.get('provider')}"
    )

    succeeded: list[str] = []
    failed: list[dict] = []
    skipped: list[str] = []
    started_at = time.monotonic()

    for index, db_id in enumerate(db_ids, start=1):
        schema_path = args.output_dir / f"{db_id}_knowledge_schema.json"
        mapping_path = args.output_dir / f"{db_id}_mapping.json"
        dictionary_path = args.output_dir / f"{db_id}_dictionary.json"
        if not args.force and schema_path.is_file() and mapping_path.is_file() and dictionary_path.is_file():
            skipped.append(db_id)
            print(f"[{index}/{len(db_ids)}] {db_id}: already built, skipping (--force to rebuild)")
            continue

        schema = all_schemas[db_id]
        db_file = database_path(db_roots[db_id], db_id)
        if not db_file.is_file():
            db_file = None

        start = time.monotonic()
        try:
            result = builder.build(schema)
            dictionary = build_dictionary(result.schema, db_file, max_domain_values)
            _write_json(schema_path, result.schema.to_dict())
            _write_json(mapping_path, result.mapping.to_dict())
            _write_json(dictionary_path, dictionary.to_dict())
            elapsed = time.monotonic() - start
            succeeded.append(db_id)
            errors_in_trace = sum(1 for event in result.trace if event.get("status") == "error")
            print(
                f"[{index}/{len(db_ids)}] {db_id}: classes={len(result.schema.classes)} "
                f"properties={len(result.schema.datatype_properties)} "
                f"object_properties={len(result.schema.object_properties)} "
                f"trace_errors={errors_in_trace} ({elapsed:.1f}s)"
            )
        except Exception as error:  # noqa: BLE001 - keep the 160-db_id batch going
            elapsed = time.monotonic() - start
            failed.append({"db_id": db_id, "error": str(error)})
            print(f"[{index}/{len(db_ids)}] {db_id}: ERROR after {elapsed:.1f}s: {error}")

    total_elapsed = time.monotonic() - started_at
    summary = {
        "total": len(db_ids),
        "succeeded": len(succeeded),
        "skipped": len(skipped),
        "failed": failed,
        "total_elapsed_sec": round(total_elapsed, 1),
    }
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False, indent=2))
    _write_json(args.output_dir / "_build_summary.json", summary)


if __name__ == "__main__":
    main()
