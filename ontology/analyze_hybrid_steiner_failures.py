from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = (
    "d591996dbba7bc1089c56f7725f43540d87f4c445a04096173ef19db0b16f74d"
)
EXPECTED_FAILURE_INDICES = [
    11, 16, 23, 31, 32, 41, 42, 47, 53, 54, 57, 58, 61, 62,
    64, 65, 66, 72, 73, 74, 75, 76, 79, 80, 81, 82, 83, 84,
]

# Primary cause means the earliest or most causally decisive error visible in
# the frozen trace. Secondary issues are described in the reason where useful.
CLASSIFICATIONS: dict[int, tuple[str, str]] = {
    11: (
        "A",
        "Orchestrator output order is count then country, so the generator follows "
        "that order while the requested/gold result is country then count.",
    ),
    16: (
        "A",
        "Orchestrator encodes average capacity as AVG aggregation although the "
        "schema contains stadium.Average and gold selects that stored column.",
    ),
    23: (
        "A",
        "The phrase '각 경기장' is not represented as grouping and the stadium "
        "name is not preserved as a grouped output, producing one global count.",
    ),
    31: (
        "D",
        "The plan and validated year retain exclusion intent, but the generator "
        "uses LEFT JOIN with year != 2014 instead of EXCEPT/NOT EXISTS, which is "
        "incorrect for stadiums that have concerts in multiple years.",
    ),
    32: (
        "D",
        "The generator implements set exclusion as row-level year inequality. "
        "It should use EXCEPT/NOT EXISTS at the stadium entity level.",
    ),
    41: (
        "A",
        "The requirement 'both 2014 and 2015' is reduced to one IN filter and "
        "set_operations is empty, losing intersection/all-years semantics.",
    ),
    42: (
        "A",
        "The requirement 'both 2014 and 2015' is reduced to one IN filter and "
        "set_operations is empty, losing intersection/all-years semantics.",
    ),
    47: (
        "A",
        "The plan incorrectly specifies MIN(weight); the task needs weight from "
        "the row with minimum pet_age (ORDER BY pet_age LIMIT 1). It also fails "
        "to represent the dog-type predicate explicitly.",
    ),
    53: (
        "A",
        "The plan omits the dog PetType condition and keeps only female/ownership, "
        "so the generated count includes all pets owned by female students.",
    ),
    54: (
        "A",
        "The plan omits the dog PetType condition and keeps only female/ownership, "
        "so the generated count includes all pets owned by female students.",
    ),
    57: (
        "D",
        "The plan, filter, and joins are sufficient, but the generator emits "
        "Fname and LName and omits DISTINCT; gold requires distinct Fname only.",
    ),
    58: (
        "D",
        "The plan, filter, and joins are sufficient, but the generator emits "
        "Fname and LName and omits DISTINCT; gold requires distinct Fname only.",
    ),
    61: (
        "D",
        "The generator applies row-level PetType != cat after LEFT JOIN. Correct "
        "entity-level exclusion requires NOT IN/NOT EXISTS, otherwise students "
        "with both cat and non-cat pets leak into the result.",
    ),
    62: (
        "D",
        "The generator applies row-level PetType != cat after LEFT JOIN. Correct "
        "entity-level exclusion requires NOT IN/NOT EXISTS.",
    ),
    64: (
        "A",
        "The plan represents 'does not own a cat' as a non-negated != predicate "
        "instead of entity-level anti-existence, leading to an incorrect query.",
    ),
    65: (
        "A",
        "The plan drops the positive dog-exists requirement and retains only the "
        "cat exclusion. The generated SQL also adds LName, but the earlier missing "
        "dog condition is the primary failure.",
    ),
    66: (
        "E",
        "The Korean question asks for student names, while gold SQL returns Fname "
        "and Age. This question/gold inconsistency prevents a clean component blame.",
    ),
    72: (
        "A",
        "The plan orders outputs as PetType, AVG(age), MAX(age), while gold/result "
        "contract expects AVG(age), MAX(age), PetType; EX is order-sensitive.",
    ),
    73: (
        "D",
        "PetType is present in plan outputs/grouping and retrieved columns, but "
        "the generator omits it from SELECT.",
    ),
    74: (
        "D",
        "PetType is present in plan outputs/grouping and retrieved columns, but "
        "the generator omits it from SELECT.",
    ),
    75: (
        "D",
        "The generator adds LName and omits DISTINCT although the requested/gold "
        "projection is distinct Fname and Age.",
    ),
    76: (
        "D",
        "The generator adds LName and omits DISTINCT although the requested/gold "
        "projection is distinct Fname and Age.",
    ),
    79: (
        "A",
        "The plan fixes output order as student ID then pet count, whereas the "
        "gold/result contract is count then student ID.",
    ),
    80: (
        "A",
        "The plan fixes output order as student ID then pet count and does not "
        "record COUNT in aggregations; gold expects count then student ID.",
    ),
    81: (
        "A",
        "Korean '두 마리 이상' means count >= 2 (equivalently > 1), but the plan "
        "encodes > 2. The generator therefore returns no rows.",
    ),
    82: (
        "A",
        "Korean '두 마리 이상' means count >= 2 (equivalently > 1), but the plan "
        "encodes > 2. The generator therefore returns no rows.",
    ),
    83: (
        "B",
        "Value Linker validates age 3 but fails to map Korean 고양이 to stored "
        "Pets.PetType='cat'; generator uses the unvalidated Korean literal.",
    ),
    84: (
        "B",
        "Value Linker combines age into one condition and does not validate the "
        "cat type value, so generator uses Pets.PetType='고양이' instead of 'cat'.",
    ),
}

CATEGORY_NAMES = {
    "A": "Orchestrator 계획 오류",
    "B": "Value Linker 오류",
    "C": "Join Linker 오류",
    "D": "Generator 조립 오류",
    "E": "기타/불명확",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "schema linking/multi-agent/outputs/"
            "qwen_danke_embedding_hybrid_text_to_sql_100.json"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "ontology/output/failure_analysis/"
            "hybrid_steiner_on_ex_failures_0_99.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(
            "ontology/output/failure_analysis/"
            "hybrid_steiner_on_ex_failures_0_99.md"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_bytes = args.input.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "Frozen source checksum mismatch: "
            f"{source_sha256} != {EXPECTED_SOURCE_SHA256}"
        )
    payload = json.loads(source_bytes)
    failures = [
        record
        for record in payload["results"]
        if record["index"] <= 99
        and not bool(
            record.get("sql_evaluation", {}).get("execution_match")
        )
    ]
    indices = [record["index"] for record in failures]
    if indices != EXPECTED_FAILURE_INDICES:
        raise SystemExit(f"Unexpected failure indices: {indices}")
    if set(indices) != set(CLASSIFICATIONS):
        raise SystemExit("Classification map does not cover exact failure set")

    records: list[dict[str, Any]] = []
    for record in failures:
        index = record["index"]
        category, reason = CLASSIFICATIONS[index]
        retrieval_metadata = (
            record.get("retrieved_schema", {}).get("metadata", {}) or {}
        )
        sql_generation = record.get("sql_generation", {}) or {}
        records.append(
            {
                "index": index,
                "question": record["question"],
                "orchestrator_plan": record.get(
                    "query_decomposition", {}
                ),
                "semantic_tables": retrieval_metadata.get(
                    "semantic_tables", []
                ),
                "semantic_columns": retrieval_metadata.get(
                    "semantic_columns", []
                ),
                "bridge_tables": retrieval_metadata.get(
                    "bridge_tables", []
                ),
                "validated_filters": record.get(
                    "grounded_filters", []
                ) or [],
                "validated_joins": record.get("joins", []) or [],
                "generated_sql": sql_generation.get("sql"),
                "execution_result": sql_generation.get("execution"),
                "ex_success": bool(
                    record.get("sql_evaluation", {}).get(
                        "execution_match"
                    )
                ),
                "primary_cause": {
                    "code": category,
                    "name": CATEGORY_NAMES[category],
                    "reason": reason,
                },
                "pipeline_unresolved": record.get("unresolved", []) or [],
            }
        )

    counts = Counter(
        record["primary_cause"]["code"] for record in records
    )
    report = {
        "metadata": {
            "development_indices": [0, 99],
            "heldout_data_accessed": False,
            "source_path": str(args.input),
            "source_sha256": source_sha256,
            "retriever_mode": "danke_embedding_hybrid",
            "steiner": True,
            "join_linker": True,
            "examples": 100,
            "execution_match": payload["summary"][
                "sql_execution_match"
            ],
            "execution_failures": len(records),
            "classification_policy": (
                "Primary cause is the earliest or most causally decisive "
                "error visible in the frozen trace."
            ),
        },
        "category_counts": {
            code: {
                "name": CATEGORY_NAMES[code],
                "count": counts.get(code, 0),
            }
            for code in CATEGORY_NAMES
        },
        "records": records,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["category_counts"], ensure_ascii=False))
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hybrid + Steiner ON 개발 세트 EX 실패 분석",
        "",
        "- 대상: index 0–99 개발 세트",
        "- EX: 0.72 (실패 28건)",
        "- held-out 접근: 하지 않음",
        "- 분류 기준: 최초 또는 가장 인과적으로 결정적인 오류",
        "",
        "## 분류 집계",
        "",
        "| 코드 | 분류 | 건수 |",
        "|---|---|---:|",
    ]
    for code, item in report["category_counts"].items():
        lines.append(f"| {code} | {item['name']} | {item['count']} |")
    lines.extend(
        [
            "",
            "## 문항별 원인",
            "",
            "| Index | 질문 | 원인 | 판단 |",
            "|---:|---|---|---|",
        ]
    )
    for record in report["records"]:
        cause = record["primary_cause"]
        question = str(record["question"]).replace("|", "\\|")
        reason = str(cause["reason"]).replace("|", "\\|")
        lines.append(
            f"| {record['index']} | {question} | "
            f"{cause['code']}. {cause['name']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "전체 Orchestrator plan, semantic table/column, validated filter/join, "
            "생성 SQL, 실행 결과는 같은 이름의 JSON 보고서에 보존되어 있습니다.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
