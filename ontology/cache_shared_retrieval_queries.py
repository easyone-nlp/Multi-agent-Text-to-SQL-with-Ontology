from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
MULTI_AGENT_DIR = WORKSPACE / "schema linking" / "multi-agent"
if str(MULTI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(MULTI_AGENT_DIR))

from schema_agents.agentic_agents import ManagerOrchestratorAgent  # noqa: E402
from schema_agents.data import default_examples_path, load_examples  # noqa: E402
from schema_agents.model_client import build_chat_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call the Qwen manager exactly once per selected question and cache the "
            "decomposition/retrieval_query for retriever-only comparisons."
        )
    )
    parser.add_argument("--split", choices=("validation", "dev", "train"), default="validation")
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--config", type=Path, default=MULTI_AGENT_DIR / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--indices-file",
        type=Path,
        help="Optional JSON array or newline-separated exact dataset indices.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json-retries", type=int, default=1)
    parser.add_argument(
        "--schema-linker-mode",
        choices=("embedding_only", "qwen"),
        default="embedding_only",
    )
    parser.add_argument(
        "--include-sql-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse already cached indices and append only missing records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples_path = args.examples or default_examples_path(args.split)
    examples = load_examples(examples_path)
    indices = _selected_indices(args, len(examples))
    if not indices:
        raise SystemExit("선택된 index가 없습니다.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    chat_config = dict(config.get("schema_linking", config))
    chat_config.update(
        {
            "provider": "openai_compatible",
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "timeout_seconds": args.timeout_seconds,
        }
    )
    manager = ManagerOrchestratorAgent(
        build_chat_model(chat_config),
        json_retries=args.json_retries,
    )

    cached_records: dict[int, dict[str, Any]] = {}
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        _validate_resume(previous, examples_path)
        cached_records = {
            int(record["index"]): record for record in previous.get("records", [])
        }

    metadata = {
        "schema_version": 1,
        "purpose": "single shared orchestrator input for fair retriever-only comparison",
        "dataset": {
            "path": str(examples_path.resolve()),
            "sha256": _sha256(examples_path),
            "split": args.split,
        },
        "selection": {
            "indices": indices,
            "count": len(indices),
            "db_counts": dict(
                Counter(examples[index].db_id for index in indices)
            ),
        },
        "orchestrator": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "timeout_seconds": args.timeout_seconds,
            "json_retries": args.json_retries,
            "schema_linker_mode": args.schema_linker_mode,
            "include_sql_generation": args.include_sql_generation,
            "prompt_source": str(
                (
                    MULTI_AGENT_DIR
                    / "schema_agents"
                    / "agentic_agents.py"
                ).resolve()
            ),
            "prompt_source_sha256": _sha256(
                MULTI_AGENT_DIR / "schema_agents" / "agentic_agents.py"
            ),
        },
    }

    for position, index in enumerate(indices, start=1):
        if index in cached_records and "decomposition" in cached_records[index]:
            print(f"[{position}/{len(indices)}] index={index} cached")
            continue
        example = examples[index]
        print(
            f"[{position}/{len(indices)}] index={index} "
            f"db_id={example.db_id} manager..."
        )
        response = manager.decompose(
            example.question,
            args.include_sql_generation,
            args.schema_linker_mode,
        )
        decomposition = response.payload
        cached_records[index] = {
            "index": index,
            "db_id": example.db_id,
            "question": example.question,
            "gold_sql": example.gold_sql,
            "retrieval_query": str(
                decomposition.get("retrieval_query") or example.question
            ),
            "decomposition": decomposition,
            "orchestrator_attempts": response.attempts,
        }
        _write_cache(
            args.output,
            metadata,
            [cached_records[item] for item in indices if item in cached_records],
        )

    missing = [index for index in indices if index not in cached_records]
    if missing:
        raise SystemExit(f"캐시에 저장되지 않은 index가 있습니다: {missing}")
    _write_cache(
        args.output,
        metadata,
        [cached_records[index] for index in indices],
    )
    print(f"wrote {args.output} ({len(indices)} records)")


def _selected_indices(args: argparse.Namespace, total: int) -> list[int]:
    if args.indices_file:
        text = args.indices_file.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if text.startswith("["):
            raw = json.loads(text)
        else:
            raw = [line.strip() for line in text.splitlines() if line.strip()]
        indices = [int(item) for item in raw]
    else:
        start = max(0, args.offset)
        stop = min(total, start + max(0, args.limit))
        indices = list(range(start, stop))
    if len(indices) != len(set(indices)):
        raise SystemExit("indices에 중복이 있습니다.")
    invalid = [index for index in indices if index < 0 or index >= total]
    if invalid:
        raise SystemExit(f"dataset 범위를 벗어난 index: {invalid}")
    return indices


def _validate_resume(previous: dict[str, Any], examples_path: Path) -> None:
    recorded = (
        previous.get("metadata", {})
        .get("dataset", {})
        .get("sha256")
    )
    current = _sha256(examples_path)
    if recorded and recorded != current:
        raise SystemExit(
            "기존 cache의 dataset checksum이 현재 dataset과 다릅니다. "
            "--no-resume 또는 다른 output을 사용하세요."
        )


def _write_cache(
    output: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"metadata": metadata, "records": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
