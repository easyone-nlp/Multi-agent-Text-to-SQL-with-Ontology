from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from schema_agents.agentic_orchestrator import (
    AgenticMultiAgentSchemaLinker,
)
from schema_agents.agentic_agents import AgentWorkflowError
from schema_agents.data import (
    database_path,
    default_database_root,
    default_examples_path,
    default_tables_path,
    load_examples,
    load_schemas,
)
from schema_agents.evaluation import evaluate, evaluate_generated_sql
from schema_agents.models import DatabaseSchema, Example, LinkingResult, ValueEvidence
from schema_agents.model_client import ModelClientError
from schema_agents.sql_orchestrator import MultiAgentSQLGenerator

class LinkingResultCache:
    """Loads and validates schema-linking results saved by this CLI."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise SystemExit(f"schema linking 입력 파일을 찾을 수 없습니다: {path}")
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as error:
            raise SystemExit(f"schema linking 입력 JSON이 올바르지 않습니다: {error}") from error

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise SystemExit("schema linking 입력은 results JSON list를 포함해야 합니다.")

        self.path = path
        self.by_index: dict[int, dict[str, Any]] = {}
        self.by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for position, item in enumerate(results):
            if not isinstance(item, dict):
                raise SystemExit(f"schema linking results[{position}]가 JSON object가 아닙니다.")
            index = item.get("index")
            db_id = item.get("db_id")
            question = item.get("question")
            if not isinstance(index, int):
                raise SystemExit(f"schema linking results[{position}]에 정수 index가 없습니다.")
            if not isinstance(db_id, str) or not isinstance(question, str):
                raise SystemExit(
                    f"schema linking results[{position}]에 db_id/question이 없습니다."
                )
            if index in self.by_index:
                raise SystemExit(f"schema linking 입력에 중복 index={index}가 있습니다.")
            self.by_index[index] = item
            self.by_key.setdefault((db_id, question), []).append(item)

    def resolve(
        self,
        index: int,
        example: Example,
        schema: DatabaseSchema,
    ) -> tuple[LinkingResult, dict[str, Any]]:
        item = self.by_index.get(index)
        if item is None or not self._matches(item, example):
            matches = self.by_key.get((example.db_id, example.question), [])
            if len(matches) != 1:
                detail = "없음" if not matches else f"{len(matches)}개로 모호함"
                raise SystemExit(
                    "schema linking 입력에서 현재 예제를 정확히 찾지 못했습니다: "
                    f"index={index}, db_id={example.db_id}, match={detail}"
                )
            item = matches[0]

        tables = self._string_list(item, "selected_tables")
        columns = self._string_list(item, "selected_columns")
        valid_tables = set(schema.tables)
        valid_columns = {column.key for column in schema.columns}
        unknown_tables = sorted(set(tables) - valid_tables)
        unknown_columns = sorted(set(columns) - valid_columns)
        if unknown_tables or unknown_columns:
            raise SystemExit(
                "schema linking 입력과 현재 Spider schema가 일치하지 않습니다: "
                f"unknown_tables={unknown_tables}, unknown_columns={unknown_columns}"
            )

        result = LinkingResult(
            db_id=example.db_id,
            question=example.question,
            tables=tables,
            columns=columns,
            table_scores=self._score_map(item, "table_scores"),
            column_scores=self._score_map(item, "column_scores"),
            trace=[],
            value_evidence=self._value_evidence(item, valid_columns),
            query_decomposition=(
                item.get("query_decomposition", {})
                if isinstance(item.get("query_decomposition", {}), dict) else {}
            ),
            retrieved_schema=(
                item.get("retrieved_schema", {})
                if isinstance(item.get("retrieved_schema", {}), dict) else {}
            ),
            column_roles=self._column_roles(item, valid_columns),
            grounded_filters=self._dict_list(item, "grounded_filters"),
            joins=self._dict_list(item, "joins"),
            unresolved=self._string_values(item, "unresolved"),
            workflow_trace=self._dict_list(item, "agent_trace"),
        )
        return result, item

    @staticmethod
    def _matches(item: dict[str, Any], example: Example) -> bool:
        return (
            item.get("db_id") == example.db_id
            and item.get("question") == example.question
        )

    @staticmethod
    def _string_list(item: dict[str, Any], key: str) -> list[str]:
        value = item.get(key)
        if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
            raise SystemExit(f"schema linking 결과의 {key}는 문자열 list여야 합니다.")
        return list(dict.fromkeys(value))

    @staticmethod
    def _score_map(item: dict[str, Any], key: str) -> dict[str, float]:
        value = item.get(key, {})
        if not isinstance(value, dict):
            raise SystemExit(f"schema linking 결과의 {key}는 JSON object여야 합니다.")
        try:
            return {str(name): float(score) for name, score in value.items()}
        except (TypeError, ValueError) as error:
            raise SystemExit(f"schema linking 결과의 {key} score가 숫자가 아닙니다.") from error

    @staticmethod
    def _value_evidence(
        item: dict[str, Any], valid_columns: set[str]
    ) -> list[ValueEvidence]:
        raw_items = item.get("value_grounding", [])
        if not isinstance(raw_items, list):
            raise SystemExit("schema linking 결과의 value_grounding은 list여야 합니다.")
        evidence: list[ValueEvidence] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise SystemExit("value_grounding 항목은 JSON object여야 합니다.")
            parsed = ValueEvidence.from_dict(raw)
            if parsed.column not in valid_columns:
                raise SystemExit(
                    "schema linking value evidence에 존재하지 않는 column이 있습니다: "
                    f"{parsed.column}"
                )
            evidence.append(parsed)
        return evidence


    @staticmethod
    def _dict_list(item: dict[str, Any], key: str) -> list[dict[str, Any]]:
        value = item.get(key, [])
        if not isinstance(value, list):
            return []
        return [entry for entry in value if isinstance(entry, dict)]

    @staticmethod
    def _string_values(item: dict[str, Any], key: str) -> list[str]:
        value = item.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(entry) for entry in value]

    @staticmethod
    def _column_roles(
        item: dict[str, Any], valid_columns: set[str]
    ) -> dict[str, list[str]]:
        value = item.get("column_roles", {})
        if not isinstance(value, dict):
            return {}
        roles: dict[str, list[str]] = {}
        for column, raw_roles in value.items():
            if column not in valid_columns or not isinstance(raw_roles, list):
                continue
            roles[column] = [str(role) for role in raw_roles]
        return roles

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spider-Ko multi-agent Text-to-SQL draft")
    parser.add_argument("--split", choices=("train", "validation", "dev"), default="validation")
    parser.add_argument("--examples", type=Path, help="Spider-Ko CSV path")
    parser.add_argument("--tables", type=Path, help="Spider tables JSON path")
    parser.add_argument("--database-root", type=Path, help="Spider database/<db_id> root")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--db-id", help="Filter examples or identify a custom question schema")
    parser.add_argument("--question", help="Run one custom Korean question (requires --db-id)")
    parser.add_argument("--output", type=Path, help="Optional pretty JSON result path")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument(
        "--linking-input",
        type=Path,
        help="Reuse selected schema from a previous JSON result",
    )
    parser.add_argument(
        "--linking-provider", choices=("openai_compatible",)
    )
    parser.add_argument("--linking-model", help="Qwen model id for all linking agents")
    parser.add_argument("--linking-base-url", help="Linking agents OpenAI-compatible URL")
    parser.add_argument("--embedding-model", help="Embedding model id")
    parser.add_argument(
        "--embedding-base-url", help="Embedding OpenAI-compatible URL"
    )
    parser.add_argument("--top-k-tables", type=int)
    parser.add_argument("--top-k-columns", type=int)
    parser.add_argument("--generate-sql", action="store_true", help="Run draft/critic/execute/repair")
    parser.add_argument("--provider", choices=("openai_compatible",))
    parser.add_argument("--model", help="Model id served by the OpenAI-compatible endpoint")
    parser.add_argument("--base-url", help="Example: http://localhost:8000/v1")
    parser.add_argument("--max-repairs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.question and not args.db_id:
        raise SystemExit("--question을 사용할 때는 --db-id도 필요합니다.")
    if (
        args.linking_input
        and args.output
        and args.linking_input.resolve() == args.output.resolve()
    ):
        raise SystemExit("--output은 --linking-input과 다른 파일이어야 합니다.")

    examples_path = args.examples or default_examples_path(args.split)
    tables_path = args.tables or default_tables_path(args.split)
    database_root = args.database_root or default_database_root(args.split)
    config = _load_config(args.config)
    schemas = load_schemas(tables_path)
    linking_cache = LinkingResultCache(args.linking_input) if args.linking_input else None
    linker = (
        None
        if linking_cache is not None
        else AgenticMultiAgentSchemaLinker(_linking_config(config, args))
    )
    sql_generator = None
    if args.generate_sql:
        sql_generator = MultiAgentSQLGenerator(_sql_config(config, args))

    if args.question:
        examples = [Example(args.db_id, args.question)]
    else:
        examples = load_examples(examples_path)
        if args.db_id:
            examples = [example for example in examples if example.db_id == args.db_id]
        examples = examples[args.offset : args.offset + args.limit]

    if not examples:
        raise SystemExit("조건에 맞는 예제가 없습니다.")

    records: list[dict[str, Any]] = []
    table_recalls: list[float] = []
    column_recalls: list[float] = []
    table_precisions: list[float] = []
    column_precisions: list[float] = []
    strict_table_recalls: list[bool] = []
    strict_column_recalls: list[bool] = []
    predicted_table_counts: list[int] = []
    predicted_column_counts: list[int] = []
    sql_successes: list[bool] = []
    sql_exact_matches: list[bool] = []
    sql_execution_matches: list[bool] = []
    linking_model_successes: list[bool] = []
    linking_scout_successes: list[bool] = []
    linking_critic_successes: list[bool] = []
    linking_workflow_successes: list[bool] = []
    retrieval_table_recalls: list[float] = []
    retrieval_column_recalls: list[float] = []
    strict_retrieval_table_recalls: list[bool] = []
    strict_retrieval_column_recalls: list[bool] = []
    pipeline_failures = 0

    for index, example in enumerate(examples, start=args.offset):
        if example.db_id not in schemas:
            raise SystemExit(f"tables JSON에 db_id={example.db_id!r}가 없습니다.")
        schema = schemas[example.db_id]
        db_file = database_path(database_root, example.db_id)
        cached_item: dict[str, Any] | None = None
        if linking_cache is not None:
            result, cached_item = linking_cache.resolve(index, example, schema)
        else:
            assert linker is not None
            try:
                result = linker.link(
                    example.question,
                    schema,
                    db_file,
                    include_sql_generation=args.generate_sql,
                )
            except (AgentWorkflowError, ModelClientError) as error:
                pipeline_failures += 1
                linking_workflow_successes.append(False)
                record = {
                    "index": index,
                    "db_id": example.db_id,
                    "question": example.question,
                    "pipeline_error": str(error),
                    "schema_linking_workflow": {
                        "success": False,
                        "agents": [],
                    },
                }
                if example.gold_sql:
                    record["gold_sql"] = example.gold_sql
                records.append(record)
                print(json.dumps(record, ensure_ascii=False, indent=2))
                continue

        record = {"index": index, **result.to_dict(args.show_trace and cached_item is None)}
        scout_success: bool | None = None
        critic_success: bool | None = None
        if cached_item is not None:
            record["schema_linking_source"] = {
                "mode": "cached",
                "path": str(linking_cache.path),
                "cached_index": cached_item["index"],
            }
            if args.show_trace and isinstance(cached_item.get("agent_trace"), list):
                record["agent_trace"] = cached_item["agent_trace"]
            model_status = cached_item.get("schema_linking_model")
            if isinstance(model_status, dict):
                cached_scout = model_status.get("scout_success")
                cached_critic = model_status.get("critic_success")
                if isinstance(cached_scout, bool) and isinstance(cached_critic, bool):
                    scout_success = cached_scout
                    critic_success = cached_critic
        else:
            llm_proposal = next(
                (
                    proposal
                    for proposal in result.trace
                    if proposal.agent == "llm_schema_scout"
                ),
                None,
            )
            critic_proposal = next(
                (
                    proposal
                    for proposal in result.trace
                    if proposal.agent == "llm_schema_critic"
                ),
                None,
            )
            if llm_proposal is not None:
                scout_success = bool(llm_proposal.table_scores)
                critic_success = critic_proposal is not None and not any(
                    "실패" in reason for reason in critic_proposal.reasons
                )

        if scout_success is not None and critic_success is not None:
            linking_scout_successes.append(scout_success)
            linking_critic_successes.append(critic_success)
            linking_model_successes.append(scout_success and critic_success)
            record["schema_linking_model"] = {
                "scout_success": scout_success,
                "critic_success": critic_success,
            }
        if result.workflow_trace:
            workflow_success = not any(
                event.get("status") == "error"
                for event in result.workflow_trace
            )
            linking_workflow_successes.append(workflow_success)
            record["schema_linking_workflow"] = {
                "success": workflow_success,
                "agents": [event.get("agent") for event in result.workflow_trace],
            }
        sql_result = None

        if sql_generator is not None:
            sql_result = sql_generator.generate(
                example.question, schema, result, db_file
            )
            record["sql_generation"] = sql_result.to_dict(args.show_trace)
            sql_successes.append(sql_result.success)

        if example.gold_sql:
            score = evaluate(result, schema, example.gold_sql)
            record["gold_sql"] = example.gold_sql
            record["schema_linking_evaluation"] = {
                "table_recall": score.table_recall,
                "table_precision": score.table_precision,
                "strict_table_recall": score.strict_table_recall,
                "column_recall": score.column_recall,
                "column_precision": score.column_precision,
                "strict_column_recall": score.strict_column_recall,
                "gold_tables": sorted(score.gold_tables),
                "gold_columns": sorted(score.gold_columns),
                "missing_tables": sorted(score.gold_tables - set(result.tables)),
                "missing_columns": sorted(score.gold_columns - set(result.columns)),
            }
            table_recalls.append(score.table_recall)
            table_precisions.append(score.table_precision)
            column_precisions.append(score.column_precision)
            strict_table_recalls.append(score.strict_table_recall)
            strict_column_recalls.append(score.strict_column_recall)
            predicted_table_counts.append(len(result.tables))
            predicted_column_counts.append(len(result.columns))
            if score.column_recall is not None:
                column_recalls.append(score.column_recall)

            raw_retrieved_tables = result.retrieved_schema.get("tables")
            raw_retrieved_columns = result.retrieved_schema.get("columns")
            if (
                isinstance(raw_retrieved_tables, list)
                and isinstance(raw_retrieved_columns, list)
            ):
                retrieved_tables = {
                    str(table) for table in raw_retrieved_tables
                }
                retrieved_columns = {
                    str(column) for column in raw_retrieved_columns
                }
                retrieval_table_recall = (
                    len(score.gold_tables & retrieved_tables)
                    / len(score.gold_tables)
                    if score.gold_tables else 1.0
                )
                retrieval_column_recall = (
                    len(score.gold_columns & retrieved_columns)
                    / len(score.gold_columns)
                    if score.gold_columns else None
                )
                strict_retrieval_table = (
                    score.gold_tables <= retrieved_tables
                )
                strict_retrieval_column = (
                    score.gold_columns <= retrieved_columns
                )
                record["schema_retrieval_evaluation"] = {
                    "table_recall": retrieval_table_recall,
                    "strict_table_recall": strict_retrieval_table,
                    "column_recall": retrieval_column_recall,
                    "strict_column_recall": strict_retrieval_column,
                    "missing_tables": sorted(
                        score.gold_tables - retrieved_tables
                    ),
                    "missing_columns": sorted(
                        score.gold_columns - retrieved_columns
                    ),
                }
                retrieval_table_recalls.append(retrieval_table_recall)
                strict_retrieval_table_recalls.append(
                    strict_retrieval_table
                )
                strict_retrieval_column_recalls.append(
                    strict_retrieval_column
                )
                if retrieval_column_recall is not None:
                    retrieval_column_recalls.append(
                        retrieval_column_recall
                    )

            if sql_result is not None:
                sql_score = evaluate_generated_sql(
                    sql_result.sql, example.gold_sql, db_file
                )
                record["sql_evaluation"] = sql_score.to_dict()
                sql_exact_matches.append(sql_score.normalized_exact_match)
                if sql_score.execution_match is not None:
                    sql_execution_matches.append(sql_score.execution_match)

        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

    summary: dict[str, Any] = {
        "examples": len(records),
        "schema_linking_completed": len(records) - pipeline_failures,
        "schema_linking_pipeline_failures": pipeline_failures,
        "schema_linking_evaluated": len(table_recalls),
        "mean_table_recall": _mean_or_none(table_recalls),
        "mean_table_precision": _mean_or_none(table_precisions),
        "strict_table_recall": _mean_or_none(strict_table_recalls),
        "mean_column_recall": _mean_or_none(column_recalls),
        "mean_column_precision": _mean_or_none(column_precisions),
        "strict_column_recall": _mean_or_none(strict_column_recalls),
        "average_predicted_tables": _mean_or_none(predicted_table_counts),
        "average_predicted_columns": _mean_or_none(predicted_column_counts),
        "embedding_mean_table_recall": _mean_or_none(
            retrieval_table_recalls
        ),
        "embedding_strict_table_recall": _mean_or_none(
            strict_retrieval_table_recalls
        ),
        "embedding_mean_column_recall": _mean_or_none(
            retrieval_column_recalls
        ),
        "embedding_strict_column_recall": _mean_or_none(
            strict_retrieval_column_recalls
        ),
    }
    if linking_cache is not None:
        summary["schema_linking_input"] = str(linking_cache.path)
    if linking_model_successes:
        summary["schema_linking_model_success_rate"] = _mean_or_none(linking_model_successes)
        summary["schema_linking_model_failures"] = linking_model_successes.count(False)
        summary["schema_linking_scout_success_rate"] = _mean_or_none(linking_scout_successes)
        summary["schema_linking_critic_success_rate"] = _mean_or_none(linking_critic_successes)
    if linking_workflow_successes:
        summary["schema_linking_workflow_success_rate"] = _mean_or_none(
            linking_workflow_successes
        )
    if sql_successes:
        summary["sql_execution_success_rate"] = _mean_or_none(sql_successes)
        summary["sql_normalized_exact_match"] = _mean_or_none(sql_exact_matches)
        summary["sql_execution_match"] = _mean_or_none(sql_execution_matches)
        summary["sql_execution_match_evaluated"] = len(sql_execution_matches)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(
                {"summary": summary, "results": records},
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")


def _mean_or_none(values: list[Any]) -> float | None:
    return round(mean(values), 4) if values else None


def _linking_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    linking_config = dict(config.get("schema_linking", {}))
    chat_overrides = {
        "provider": args.linking_provider,
        "model": args.linking_model,
        "base_url": args.linking_base_url,
    }
    for key, value in chat_overrides.items():
        if value is not None:
            linking_config[key] = value

    embedding_config = dict(linking_config.get("embedding", {}))
    embedding_overrides = {
        "model": args.embedding_model,
        "base_url": args.embedding_base_url,
        "top_k_tables": args.top_k_tables,
        "top_k_columns": args.top_k_columns,
    }
    for key, value in embedding_overrides.items():
        if value is not None:
            embedding_config[key] = value
    linking_config["embedding"] = embedding_config
    return linking_config


def _sql_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sql_config = dict(config.get("sql_generation", {}))
    for key in ("provider", "model", "base_url", "max_repairs"):
        value = getattr(args, key)
        if value is not None:
            sql_config[key] = value
    return sql_config


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
