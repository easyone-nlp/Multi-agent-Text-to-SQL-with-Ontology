from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import DatabaseSchema, Example


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parents[1]
DEFAULT_KO_DIR = WORKSPACE_DIR / "data" / "hugging face" / "Spider 1.0 ko"
DEFAULT_SPIDER_DIR = WORKSPACE_DIR / "data" / "hugging face" / "Spider 1.0"


def default_examples_path(split: str) -> Path:
    normalized = "validation" if split in {"dev", "validation"} else "train"
    return DEFAULT_KO_DIR / f"{normalized}.csv"


def default_tables_path(split: str) -> Path:
    if split in {"dev", "validation"}:
        return DEFAULT_SPIDER_DIR / "dev" / "dev_table.json"
    return DEFAULT_SPIDER_DIR / "train" / "train_tables.json"


def default_database_root(split: str) -> Path:
    directory = "dev" if split in {"dev", "validation"} else "train"
    return DEFAULT_SPIDER_DIR / directory / "database"


def database_path(database_root: Path, db_id: str) -> Path:
    root = database_root.resolve()
    candidate = (root / db_id / f"{db_id}.sqlite").resolve()
    if root not in candidate.parents:
        raise ValueError(f"unsafe db_id path: {db_id!r}")
    return candidate


def load_examples(path: Path) -> list[Example]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            Example(
                db_id=row["db_id"],
                question=row.get("question_ko") or row.get("question") or "",
                gold_sql=row.get("query") or None,
            )
            for row in csv.DictReader(handle)
        ]


def load_schemas(path: Path) -> dict[str, DatabaseSchema]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {
        record["db_id"]: DatabaseSchema.from_spider(record) for record in records
    }
