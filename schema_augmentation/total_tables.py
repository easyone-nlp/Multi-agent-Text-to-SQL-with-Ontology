import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPIDER_DATA_DIR = DATA_ROOT / "huggingface"
if not SPIDER_DATA_DIR.exists():
    SPIDER_DATA_DIR = DATA_ROOT / "hugging face"

TRAIN_TABLES = SPIDER_DATA_DIR / "Spider 1.0" / "train" / "train_tables.json"
DEV_TABLES = SPIDER_DATA_DIR / "Spider 1.0" / "dev" / "dev_table.json"
train = read_json(TRAIN_TABLES)
dev = read_json(DEV_TABLES)

merged = {}

for db in train + dev:
    merged[db["db_id"]] = db

all_tables = list(merged.values())

print("train DB 수:", len(train))
print("dev DB 수:", len(dev))
print("합친 DB 수:", len(all_tables))