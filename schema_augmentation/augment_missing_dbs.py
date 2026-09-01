# %%
"""
KG에는 있지만 기존 synonym.json에는 누락된 DB들의 컬럼 synonym을 생성하는 스크립트.

입력:
- output/spider_ko/{db_id}_knowledge_schema.json
- 기존 synonym.json

처리 대상:
- knowledge_schema.json의 datatype_properties
- 즉, 각 DB의 실제 테이블/컬럼

출력:
1. missing_db_synonyms.json
   - 새로 생성한 누락 DB 결과만 저장

2. synonym_complete.json
   - 기존 synonym.json과 새 결과를 병합한 최종 파일

3. missing_db_synonyms_checkpoint.jsonl
   - 중간 저장용 체크포인트

4. missing_db_synonyms_errors_날짜시간.jsonl
   - 최종 실패 컬럼 기록
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, ValidationError
from vllm import LLM, SamplingParams


# ============================================================
# 1. 사용자 설정
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B"

KG_DIR = Path(
    "/home/dilab/Desktop/hackathon/ontology/output/spider_ko"
)

EXISTING_SYNONYM_FILE = Path(
    "/home/dilab/Desktop/hackathon/ontology/synonym.json"
)

MISSING_RESULT_FILE = Path(
    "/home/dilab/Desktop/hackathon/schema_augmentation/missing_db_synonyms.json"
)

MERGED_RESULT_FILE = Path(
    "/home/dilab/Desktop/hackathon/schema_augmentation/synonym_complete.json"
)

CHECKPOINT_FILE = Path(
    "/home/dilab/Desktop/hackathon/schema_augmentation/"
    "missing_db_synonyms_checkpoint.jsonl"
)

ERROR_FILE = Path(
    "/home/dilab/Desktop/hackathon/schema_augmentation/"
    f"missing_db_synonyms_errors_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
)


# 한 프롬프트 안에 넣을 컬럼 수
TARGET_BATCH_SIZE = 3

# vLLM에 동시에 전달할 프롬프트 수
VLLM_REQUEST_BATCH_SIZE = 8

# True이면 기존 체크포인트를 삭제하고 처음부터 시작
# 처음 시험할 때만 True, 이후 재실행할 때는 False 권장
RESET_CHECKPOINT = False

# 시험 실행 시 1~2로 설정 가능
# 전체 20개 DB를 처리하려면 None
MAX_DATABASES: int | None = None


MISSING_DB_IDS = [
    "employee_hire_evaluation",
    "dog_kennels",
    "singer",
    "car_1",
    "course_teach",
    "flight_2",
    "wta_1",
    "cre_Doc_Template_Mgt",
    "poker_player",
    "network_1",
    "museum_visit",
    "orchestra",
    "concert_singer",
    "real_estate_properties",
    "student_transcripts_tracking",
    "tvshow",
    "battle_death",
    "world_1",
    "voter_1",
    "pets_1",
]


# ============================================================
# 2. 로깅
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. Pydantic 응답 스키마
# ============================================================

class GeneratedColumn(BaseModel):
    column: str
    description: str
    synonyms: list[str] = Field(
        min_length=3,
        max_length=8,
    )


class GeneratedResponse(BaseModel):
    db_id: str
    table: str
    columns: list[GeneratedColumn]


# ============================================================
# 4. 공통 유틸리티
# ============================================================

def chunked(
    items: list[Any],
    size: int,
) -> Iterable[list[Any]]:
    """리스트를 지정한 크기로 분할한다."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        file.flush()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "체크포인트 %d번째 줄 JSON 오류: %s",
                    line_number,
                    exc,
                )

    return records


def normalize_text_key(value: str) -> str:
    """유의어 중복 판정용 정규화."""
    return re.sub(
        r"\s+",
        " ",
        value.strip().lower(),
    )


def contains_korean(value: str) -> bool:
    return bool(re.search(r"[가-힣]", value))


def readable_column_name(column_name: str) -> str:
    """primary_label이 없을 때 컬럼명을 읽기 쉽게 변환."""
    value = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        " ",
        column_name,
    )

    value = value.replace("_", " ")

    return re.sub(r"\s+", " ", value).strip()


# ============================================================
# 5. knowledge_schema.json에서 컬럼 추출
# ============================================================

def load_missing_column_tasks(
    kg_dir: Path,
    missing_db_ids: list[str],
    max_databases: int | None = None,
) -> list[dict[str, Any]]:
    """
    누락된 DB의 knowledge_schema.json에서 datatype_properties를 읽어
    컬럼 synonym 생성 작업 목록으로 변환한다.
    """
    selected_db_ids = missing_db_ids

    if max_databases is not None:
        selected_db_ids = selected_db_ids[:max_databases]

    tasks: list[dict[str, Any]] = []

    for requested_db_id in selected_db_ids:
        schema_path = (
            kg_dir
            / f"{requested_db_id}_knowledge_schema.json"
        )

        if not schema_path.exists():
            logger.error(
                "knowledge schema 파일 없음: %s",
                schema_path,
            )
            continue

        try:
            schema = read_json(schema_path)
        except Exception:
            logger.exception(
                "knowledge schema 읽기 실패: %s",
                schema_path,
            )
            continue

        actual_db_id = str(
            schema.get("db_id") or requested_db_id
        )

        properties = schema.get(
            "datatype_properties",
            [],
        )

        if not isinstance(properties, list):
            logger.error(
                "%s의 datatype_properties가 리스트가 아닙니다.",
                requested_db_id,
            )
            continue

        loaded_count = 0

        for prop in properties:
            if not isinstance(prop, dict):
                continue

            source_table = str(
                prop.get("source_table") or ""
            ).strip()

            source_column = str(
                prop.get("source_column") or ""
            ).strip()

            if not source_table or not source_column:
                logger.warning(
                    "source_table/source_column 누락: "
                    "db=%s, property=%s",
                    actual_db_id,
                    prop.get("name"),
                )
                continue

            primary_label = str(
                prop.get("primary_label") or ""
            ).strip()

            task = {
                "db_id": actual_db_id,
                "table": source_table,
                "column": source_column,
                "readable_name": (
                    primary_label
                    if primary_label
                    else readable_column_name(source_column)
                ),
                "range_types": prop.get(
                    "range_types",
                    [],
                ),
                "property_name": prop.get("name"),
            }

            tasks.append(task)
            loaded_count += 1

        logger.info(
            "[로드] %s: %d개 컬럼",
            actual_db_id,
            loaded_count,
        )

    return tasks


# ============================================================
# 6. 프롬프트 생성
# ============================================================

SYSTEM_PROMPT = """
당신은 관계형 데이터베이스와 지식 그래프 온톨로지에 정통한
데이터 아키텍트입니다.

주어진 데이터베이스 컬럼 각각에 대해 다음을 생성하세요.

1. description
- 컬럼의 비즈니스 의미를 설명하는 한국어 한 문장
- 테이블과 컬럼의 문맥을 반영할 것
- 불필요하게 장황하게 쓰지 말 것

2. synonyms
- 일반 사용자가 자연어 질의에서 사용할 만한 유의어
- 한국어 중심으로 작성
- 실무 용어, 줄임말, 띄어쓰기 변형 등을 포함할 수 있음
- 정확히 5개 이상 8개 이하
- 중복되거나 사실상 같은 표현을 반복하지 말 것

중요 규칙:
- db_id, table, column은 입력값을 철자, 대소문자, 밑줄까지
  완전히 동일하게 유지하세요.
- table이나 column의 밑줄을 공백으로 바꾸지 마세요.
- table이나 column의 대소문자를 바꾸지 마세요.
- description은 반드시 한국어를 포함해야 합니다.
- synonyms는 5개 이상 8개 이하로 생성하세요.
- JSON 외의 설명이나 마크다운 코드 블록은 출력하지 마세요.

출력 형식:
{
  "db_id": "입력 db_id 그대로",
  "table": "입력 table 그대로",
  "columns": [
    {
      "column": "입력 column 그대로",
      "description": "한국어 한 문장",
      "synonyms": [
        "유의어1",
        "유의어2",
        "유의어3",
        "유의어4",
        "유의어5"
      ]
    }
  ]
}
""".strip()


def build_prompt(
    task_group: list[dict[str, Any]],
) -> str:
    if not task_group:
        raise ValueError("빈 task_group입니다.")

    db_ids = {
        task["db_id"]
        for task in task_group
    }

    tables = {
        task["table"]
        for task in task_group
    }

    if len(db_ids) != 1:
        raise ValueError(
            f"한 프롬프트에 여러 db_id가 있습니다: {db_ids}"
        )

    if len(tables) != 1:
        raise ValueError(
            f"한 프롬프트에 여러 table이 있습니다: {tables}"
        )

    db_id = task_group[0]["db_id"]
    table = task_group[0]["table"]

    column_lines: list[str] = []

    for task in task_group:
        range_types = task.get("range_types", [])

        if isinstance(range_types, list):
            data_type = ", ".join(
                str(value)
                for value in range_types
            )
        else:
            data_type = str(range_types)

        if not data_type:
            data_type = "unknown"

        column_lines.append(
            "\n".join(
                [
                    f"- column: {task['column']}",
                    (
                        "- current_label: "
                        f"{task['readable_name']}"
                    ),
                    f"- data_type: {data_type}",
                ]
            )
        )

    user_content = (
        f"db_id: {db_id}\n"
        f"table: {table}\n\n"
        "columns:\n"
        + "\n\n".join(column_lines)
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        "아래 컬럼들을 처리하세요.\n\n"
        f"{user_content}\n\n"
        "JSON만 출력하세요. 설명, 추론 과정, 마크다운 코드 블록은 출력하지 마세요.\n"
        "/no_think"
    )


# ============================================================
# 7. 테이블별 프롬프트 작업 생성
# ============================================================

def make_prompt_jobs(
    tasks: list[dict[str, Any]],
    target_batch_size: int,
) -> list[dict[str, Any]]:
    """
    같은 DB와 같은 테이블의 컬럼끼리만 한 프롬프트로 묶는다.
    """
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = {}

    for task in tasks:
        key = (
            task["db_id"],
            task["table"],
        )

        grouped.setdefault(key, []).append(task)

    jobs: list[dict[str, Any]] = []

    for (
        db_id,
        table,
    ), table_tasks in grouped.items():
        for task_group in chunked(
            table_tasks,
            target_batch_size,
        ):
            jobs.append(
                {
                    "db_id": db_id,
                    "table": table,
                    "tasks": task_group,
                    "prompt": build_prompt(task_group),
                }
            )

    return jobs


# ============================================================
# 8. 모델 응답 JSON 추출
# ============================================================

def extract_json_object(
    text: str,
) -> dict[str, Any]:
    """
    모델 응답에서 첫 번째 완전한 JSON 객체를 추출한다.
    """
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError(
                "모델 응답 최상위 JSON이 객체가 아닙니다."
            )

        return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")

    if start == -1:
        raise ValueError(
            "모델 응답에서 JSON 시작 문자를 찾지 못했습니다."
        )

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(cleaned)):
        char = cleaned[index]

        if escape:
            escape = False
            continue

        if char == "\\":
            if in_string:
                escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = cleaned[start:index + 1]

                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"추출한 JSON 파싱 실패: {exc}"
                    ) from exc

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "추출된 JSON 최상위가 객체가 아닙니다."
                    )

                return parsed

    raise ValueError(
        "완전한 JSON 객체를 찾지 못했습니다."
    )


# ============================================================
# 9. synonym 정리 및 검증
# ============================================================

def clean_synonyms(
    synonyms: Any,
    min_count: int = 3,
    max_count: int = 8,
) -> list[str]:
    """
    중복 제거 후 최대 8개까지만 유지한다.

    단순한 컬럼은 자연스러운 유의어가 많지 않을 수 있으므로
    최종 최소 개수는 3개로 검증한다.
    """
    if not isinstance(synonyms, list):
        raise ValueError(
            "synonyms가 리스트 형식이 아닙니다."
        )

    cleaned: list[str] = []
    seen: set[str] = set()

    for raw_value in synonyms:
        value = str(raw_value).strip()

        if not value:
            continue

        duplicate_key = normalize_text_key(value)

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)
        cleaned.append(value)

        if len(cleaned) >= max_count:
            break

    if len(cleaned) < min_count:
        raise ValueError(
            f"중복 제거 후 동의어가 {len(cleaned)}개입니다. "
            f"values={cleaned}"
        )

    return cleaned


def validate_and_convert_response(
    raw_response: dict[str, Any],
    expected_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    모델이 db_id/table 철자를 바꿔도 원본 메타데이터를 사용한다.

    컬럼은 이름으로 우선 매칭하고,
    찾지 못하면 단일 컬럼 요청인 경우 첫 결과를 사용한다.
    """
    try:
        parsed = GeneratedResponse.model_validate(
            raw_response
        )
    except ValidationError as exc:
        raise ValueError(
            "Qwen3 응답이 Pydantic 스키마와 "
            f"일치하지 않습니다. errors={exc.errors()}"
        ) from exc

    generated_by_column = {
        item.column: item
        for item in parsed.columns
    }

    results: list[dict[str, Any]] = []

    for expected in expected_tasks:
        expected_column = expected["column"]

        generated = generated_by_column.get(
            expected_column
        )

        # 모델이 컬럼 대소문자를 바꾼 경우 보조 매칭
        if generated is None:
            lower_matches = [
                item
                for item in parsed.columns
                if item.column.lower()
                == expected_column.lower()
            ]

            if len(lower_matches) == 1:
                generated = lower_matches[0]

        # 단일 컬럼 재시도에서 이름만 변형된 경우
        if (
            generated is None
            and len(expected_tasks) == 1
            and len(parsed.columns) == 1
        ):
            generated = parsed.columns[0]

        if generated is None:
            raise ValueError(
                f"응답에서 컬럼을 찾지 못했습니다: "
                f"expected={expected_column}, "
                f"actual={list(generated_by_column)}"
            )

        description = generated.description.strip()

        if not description:
            raise ValueError(
                f"{expected_column}: description이 비어 있습니다."
            )

        if not contains_korean(description):
            raise ValueError(
                f"{expected_column}: description에 "
                "한국어가 없습니다."
            )

        synonyms = clean_synonyms(
            generated.synonyms,
            min_count=3,
            max_count=8,
        )

        result = {
            # 모델 출력이 아니라 원본 스키마를 강제 사용
            "db_id": expected["db_id"],
            "table": expected["table"],
            "column": expected["column"],
            "readable_name": expected[
                "readable_name"
            ],
            "description": description,
            "synonyms": synonyms,
        }

        results.append(result)

    return results


# ============================================================
# 10. 체크포인트
# ============================================================

def record_key(
    record: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        str(record["db_id"]),
        str(record["table"]),
        str(record["column"]),
    )


def load_checkpoint_records() -> dict[
    tuple[str, str, str],
    dict[str, Any],
]:
    records = load_jsonl(CHECKPOINT_FILE)

    return {
        record_key(record): record
        for record in records
        if all(
            key in record
            for key in (
                "db_id",
                "table",
                "column",
            )
        )
    }


# ============================================================
# 11. 단일 vLLM 호출 처리
# ============================================================

def run_prompt_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    jobs: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    prompts = [
        job["prompt"]
        for job in jobs
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    results: list[tuple[dict[str, Any], str]] = []

    for job, output in zip(jobs, outputs):
        text = output.outputs[0].text
        results.append((job, text))

    return results


# ============================================================
# 12. 재귀 fallback
# ============================================================

def process_job_with_fallback(
    llm: LLM,
    sampling_params: SamplingParams,
    job: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    여러 컬럼 요청이 실패하면 절반으로 나눠 재시도한다.
    한 컬럼까지 실패하면 오류로 기록한다.
    """
    tasks = job["tasks"]

    try:
        outputs = llm.generate(
            [job["prompt"]],
            sampling_params,
        )

        response_text = outputs[0].outputs[0].text
        finish_reason = outputs[0].outputs[0].finish_reason

        print("\n" + "=" * 80)
        print(f"finish_reason: {finish_reason}")
        print(f"response length: {len(response_text)}")
        print(repr(response_text))
        print("=" * 80 + "\n")

        raw_json = extract_json_object(
            response_text
        )

        records = validate_and_convert_response(
            raw_response=raw_json,
            expected_tasks=tasks,
        )

        return records, []

    except Exception as exc:
        if len(tasks) > 1:
            failed_columns = [
                task["column"]
                for task in tasks
            ]

            logger.info(
                "[%s.%s] 재시도 중: %s",
                job["db_id"],
                job["table"],
                failed_columns,
            )

            middle = len(tasks) // 2

            left_tasks = tasks[:middle]
            right_tasks = tasks[middle:]

            all_records: list[dict[str, Any]] = []
            all_errors: list[dict[str, Any]] = []

            for split_tasks in (
                left_tasks,
                right_tasks,
            ):
                if not split_tasks:
                    continue

                split_job = {
                    "db_id": split_tasks[0]["db_id"],
                    "table": split_tasks[0]["table"],
                    "tasks": split_tasks,
                    "prompt": build_prompt(split_tasks),
                }

                records, errors = (
                    process_job_with_fallback(
                        llm=llm,
                        sampling_params=sampling_params,
                        job=split_job,
                    )
                )

                all_records.extend(records)
                all_errors.extend(errors)

            return all_records, all_errors

        task = tasks[0]

        error_record = {
            "db_id": task["db_id"],
            "table": task["table"],
            "column": task["column"],
            "readable_name": task[
                "readable_name"
            ],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        return [], [error_record]


# ============================================================
# 13. 여러 프롬프트 배치 처리
# ============================================================

def process_initial_job_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    jobs: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    먼저 여러 프롬프트를 vLLM에 동시에 넣는다.

    실패한 프롬프트만 개별 fallback 함수로 넘긴다.
    """
    success_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []

    prompt_results = run_prompt_batch(
        llm=llm,
        sampling_params=sampling_params,
        jobs=jobs,
    )

    for job, response_text in prompt_results:
        try:
            raw_json = extract_json_object(
                response_text
            )

            records = validate_and_convert_response(
                raw_response=raw_json,
                expected_tasks=job["tasks"],
            )

            success_records.extend(records)

        except Exception:
            records, errors = (
                process_job_with_fallback(
                    llm=llm,
                    sampling_params=sampling_params,
                    job=job,
                )
            )

            success_records.extend(records)
            error_records.extend(errors)

    return success_records, error_records


# ============================================================
# 14. 결과 병합
# ============================================================

def load_existing_synonym_records(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"기존 synonym 파일이 없습니다: {path}"
        )

    data = read_json(path)

    if not isinstance(data, list):
        raise ValueError(
            "기존 synonym.json의 최상위 구조가 "
            "리스트가 아닙니다."
        )

    return data


def merge_synonym_records(
    existing_records: list[dict[str, Any]],
    additional_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    db_id, table, column 조합을 기준으로 병합한다.
    동일한 키가 있으면 새 결과로 덮어쓴다.
    """
    merged: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for record in existing_records:
        if not all(
            key in record
            for key in (
                "db_id",
                "table",
                "column",
            )
        ):
            logger.warning(
                "기존 파일에서 키가 부족한 레코드 제외: %s",
                record,
            )
            continue

        merged[record_key(record)] = record

    for record in additional_records:
        merged[record_key(record)] = record

    merged_records = list(merged.values())

    merged_records.sort(
        key=lambda item: (
            str(item.get("db_id", "")),
            str(item.get("table", "")),
            str(item.get("column", "")),
        )
    )

    return merged_records


# ============================================================
# 15. 최종 검증
# ============================================================

def verify_missing_databases(
    records: list[dict[str, Any]],
    expected_db_ids: list[str],
) -> list[str]:
    generated_db_ids = {
        str(record.get("db_id"))
        for record in records
        if record.get("db_id") is not None
    }

    return sorted(
        set(expected_db_ids)
        - generated_db_ids
    )


def print_task_summary(
    tasks: list[dict[str, Any]],
) -> None:
    counts = Counter(
        task["db_id"]
        for task in tasks
    )

    logger.info(
        "생성 대상 DB 수: %d",
        len(counts),
    )

    logger.info(
        "생성 대상 컬럼 수: %d",
        len(tasks),
    )

    for db_id in MISSING_DB_IDS:
        if db_id in counts:
            logger.info(
                "  - %s: %d개 컬럼",
                db_id,
                counts[db_id],
            )


# ============================================================
# 16. 메인 실행
# ============================================================

def main() -> None:
    if RESET_CHECKPOINT and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

        logger.info(
            "기존 체크포인트 삭제: %s",
            CHECKPOINT_FILE,
        )

    tasks = load_missing_column_tasks(
        kg_dir=KG_DIR,
        missing_db_ids=MISSING_DB_IDS,
        max_databases=MAX_DATABASES,
    )

    if not tasks:
        raise RuntimeError(
            "처리할 컬럼이 없습니다. "
            "KG_DIR과 파일명을 확인하세요."
        )

    print_task_summary(tasks)

    checkpoint_records = (
        load_checkpoint_records()
    )

    logger.info(
        "체크포인트에서 %d개 컬럼 복원",
        len(checkpoint_records),
    )

    pending_tasks = [
        task
        for task in tasks
        if record_key(task)
        not in checkpoint_records
    ]

    logger.info(
        "남은 처리 대상: %d개 컬럼",
        len(pending_tasks),
    )

    prompt_jobs = make_prompt_jobs(
        tasks=pending_tasks,
        target_batch_size=TARGET_BATCH_SIZE,
    )

    logger.info(
        "생성할 프롬프트 수: %d",
        len(prompt_jobs),
    )

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="auto",

        # GPU 메모리 상황에 따라 0.75~0.90 조절
        gpu_memory_utilization=0.85,

        # 현재 서버가 GPU 1개라면 1
        tensor_parallel_size=1,

        # 프롬프트와 출력 길이 고려
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1000,

        # Qwen3의 thinking 출력을 끄는 옵션이
        # 현재 vLLM/모델 환경에서 지원된다면 사용 가능.
        # extra_args={"chat_template_kwargs": {
        #     "enable_thinking": False
        # }},
    )

    all_errors: list[dict[str, Any]] = []

    total_jobs = len(prompt_jobs)

    for start in range(
        0,
        total_jobs,
        VLLM_REQUEST_BATCH_SIZE,
    ):
        end = min(
            start + VLLM_REQUEST_BATCH_SIZE,
            total_jobs,
        )

        logger.info(
            "vLLM 요청 처리: %d~%d/%d",
            start + 1,
            end,
            total_jobs,
        )

        job_batch = prompt_jobs[start:end]

        try:
            records, errors = (
                process_initial_job_batch(
                    llm=llm,
                    sampling_params=sampling_params,
                    jobs=job_batch,
                )
            )

        except KeyboardInterrupt:
            logger.warning(
                "사용자에 의해 중단되었습니다. "
                "이미 저장된 체크포인트부터 "
                "다음 실행에서 이어집니다."
            )
            raise

        except Exception:
            logger.exception(
                "vLLM 배치 처리 중 예상하지 못한 오류"
            )

            records = []
            errors = []

            # 배치 전체 오류 시 각 작업을 개별 fallback
            for job in job_batch:
                job_records, job_errors = (
                    process_job_with_fallback(
                        llm=llm,
                        sampling_params=sampling_params,
                        job=job,
                    )
                )

                records.extend(job_records)
                errors.extend(job_errors)

        for record in records:
            key = record_key(record)

            if key in checkpoint_records:
                continue

            append_jsonl(
                CHECKPOINT_FILE,
                record,
            )

            checkpoint_records[key] = record

        all_errors.extend(errors)

    # 전체 대상 순서를 기준으로 성공 결과 구성
    missing_results: list[dict[str, Any]] = []

    for task in tasks:
        key = record_key(task)

        if key in checkpoint_records:
            missing_results.append(
                checkpoint_records[key]
            )

    missing_results.sort(
        key=lambda item: (
            item["db_id"],
            item["table"],
            item["column"],
        )
    )

    write_json(
        MISSING_RESULT_FILE,
        missing_results,
    )

    # 오류 저장
    for error in all_errors:
        append_jsonl(
            ERROR_FILE,
            error,
        )

    existing_records = (
        load_existing_synonym_records(
            EXISTING_SYNONYM_FILE
        )
    )

    merged_records = merge_synonym_records(
        existing_records=existing_records,
        additional_records=missing_results,
    )

    write_json(
        MERGED_RESULT_FILE,
        merged_records,
    )

    selected_db_ids = (
        MISSING_DB_IDS
        if MAX_DATABASES is None
        else MISSING_DB_IDS[:MAX_DATABASES]
    )

    missing_after_generation = (
        verify_missing_databases(
            records=missing_results,
            expected_db_ids=selected_db_ids,
        )
    )

    logger.info(
        "누락 DB synonym 생성 완료: %d/%d개 컬럼",
        len(missing_results),
        len(tasks),
    )

    logger.info(
        "누락 DB 결과: %s",
        MISSING_RESULT_FILE,
    )

    logger.info(
        "기존 결과와 병합된 최종 파일: %s",
        MERGED_RESULT_FILE,
    )

    logger.info(
        "체크포인트: %s",
        CHECKPOINT_FILE,
    )

    if all_errors:
        logger.warning(
            "처리하지 못한 컬럼이 있습니다. "
            "오류 로그: %s",
            ERROR_FILE,
        )

    if missing_after_generation:
        logger.warning(
            "결과가 한 건도 생성되지 않은 DB: %s",
            missing_after_generation,
        )
    else:
        logger.info(
            "요청된 모든 DB가 결과에 포함되어 있습니다."
        )


if __name__ == "__main__":
    main()