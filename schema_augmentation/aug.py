from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# =========================================================
# 0. 실행 설정
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SPIDER_DATA_DIR = DATA_ROOT / "huggingface"
if not SPIDER_DATA_DIR.exists():
    SPIDER_DATA_DIR = DATA_ROOT / "hugging face"

INPUT_FILE = SPIDER_DATA_DIR / "Spider 1.0" / "train" / "train_tables.json"
OUTPUT_JSON = PROJECT_ROOT / "schema_augmentation" / "spider_train_schema_ontology.json"
OUTPUT_EXCEL = PROJECT_ROOT / "schema_augmentation" / "spider_train_schema_ontology.xlsx"
CHECKPOINT_JSONL = PROJECT_ROOT / "schema_augmentation" / "spider_train_schema_checkpoint.jsonl"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
ERROR_JSONL = Path(f"spider_train_schema_errors_{RUN_ID}.jsonl")

TARGET_BATCH_SIZE = 3
RESET_CHECKPOINT = False
DRY_RUN = False

# None이면 모든 DB 처리
# 예: {"academic", "college_2"}
TARGET_DB_IDS: set[str] | None = None

# 테스트할 때 일부 DB만 처리하고 싶으면 숫자 지정
# 예: 1이면 첫 번째 DB만 처리
MAX_DATABASES: int | None = None

# Hugging Face에서 직접 로드할 Qwen3 모델
MODEL_NAME = "Qwen/Qwen3-4B"

# 생성 설정
MAX_NEW_TOKENS = 6000
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
DO_SAMPLE = True
ENABLE_THINKING = False

# =========================================================
# 1. Structured Output 모델
# =========================================================
NonEmptyText = Annotated[str, Field(min_length=1)]


class ColumnAugmentation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: NonEmptyText
    description: NonEmptyText
    synonyms: list[NonEmptyText] = Field(min_length=5, max_length=8)


class TableAugmentation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_id: NonEmptyText
    table: NonEmptyText
    columns: list[ColumnAugmentation]


# =========================================================
# 2. 시스템 프롬프트
# =========================================================
SYSTEM_PROMPT = """
You are a senior data architect and Text-to-SQL schema expert.

[Task]
Use the complete SQL CREATE TABLE schema as context.
Generate Korean business descriptions and Korean search aliases only for the
explicitly listed target columns.

Analyze privately before answering:
1. identify the database and table purpose,
2. inspect neighboring columns, types, primary keys, and foreign keys,
3. infer each target column's meaning conservatively,
4. generate Korean terms that users could naturally use in Text-to-SQL queries.

Do not reveal your reasoning process.

[Meaning of synonyms]
"Synonyms" means Korean search aliases that may refer to the same database
column in a natural-language query. They may include Korean translations,
commonly used expressions, abbreviations, domain terms, and natural variants.

[Spider schema conventions]
- column_names_original and table_names_original are physical SQL identifiers.
- Readable names are supporting descriptions, not guaranteed formal definitions.
- Columns ending in id or _id commonly represent identifiers.
- Foreign-key relationships are important evidence for interpreting identifiers.
- Generic names such as name, id, code, type, year, and title must be interpreted
  using their table and neighboring columns.
- Do not infer details that are unsupported by the supplied schema.

[Constraints]
1. Description:
   - Write exactly one concise Korean sentence.
   - Explain the column's business meaning in its table and database context.
   - Do not merely transliterate the English column name.
   - Do not invent unsupported operational details.

2. Synonyms:
   - Return 5 to 8 unique Korean strings.
   - Include the supplied readable column name when it is useful and natural.
   - Prefer terms that could realistically appear in Korean Text-to-SQL questions.
   - Do not include SQL identifiers, sentences, questions, or explanations.
   - Do not include duplicates differing only by spacing, punctuation, or particles.
   - Do not create unnatural abbreviations merely to reach the required count.

3. Identity and coverage:
   - Copy db_id, table name, and target column names exactly.
   - Return exactly one result for every target column.
   - Do not return non-target columns.

4. Output:
   - Return only one valid JSON object.
   - Do not output Markdown code fences.
   - Do not output explanations before or after the JSON.
   - Use this exact top-level structure:
     {
       "db_id": "...",
       "table": "...",
       "columns": [
         {
           "column": "...",
           "description": "...",
           "synonyms": ["...", "...", "...", "...", "..."]
         }
       ]
     }
""".strip()

# =========================================================
# 3. 공통 유틸리티
# =========================================================
def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def quote_identifier(identifier: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$#]*", identifier):
        return identifier
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def sanitize_comment(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("--", "—")).strip()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_output_files() -> None:
    if not RESET_CHECKPOINT:
        return

    for path in [CHECKPOINT_JSONL, OUTPUT_JSON, OUTPUT_EXCEL]:
        if path.exists():
            path.unlink()
            logging.info("기존 파일 삭제: %s", path)


def load_checkpoint(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    results: dict[tuple[str, str, str], dict[str, Any]] = {}

    if not path.exists():
        return results

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                key = (
                    record["db_id"],
                    record["table"],
                    record["column"],
                )
                results[key] = record

            except (json.JSONDecodeError, KeyError, TypeError) as error:
                logging.warning(
                    "체크포인트 %s번째 줄을 건너뜁니다: %s",
                    line_number,
                    error,
                )

    return results


# =========================================================
# 4. Spider 데이터 타입 변환
# =========================================================
def convert_spider_type(spider_type: str) -> str:
    type_mapping = {
        "text": "TEXT",
        "number": "NUMERIC",
        "time": "DATETIME",
        "boolean": "BOOLEAN",
        "others": "TEXT",
    }

    normalized = clean_text(spider_type).lower()
    return type_mapping.get(normalized, "TEXT")


# =========================================================
# 5. Spider train_tables.json 파싱
# =========================================================
def prepare_schema(
    raw_schema: list[dict[str, Any]],
) -> pd.DataFrame:
    schema_records: list[dict[str, Any]] = []
    selected_databases = raw_schema

    if TARGET_DB_IDS is not None:
        selected_databases = [
            database
            for database in selected_databases
            if clean_text(database.get("db_id")) in TARGET_DB_IDS
        ]

    if MAX_DATABASES is not None:
        selected_databases = selected_databases[:MAX_DATABASES]

    for database in selected_databases:
        db_id = clean_text(database.get("db_id"))
        table_names = database.get("table_names_original", [])
        readable_table_names = database.get("table_names", [])
        column_names = database.get("column_names_original", [])
        readable_column_names = database.get("column_names", [])
        column_types = database.get("column_types", [])
        primary_keys = set(database.get("primary_keys", []))
        foreign_key_pairs = database.get("foreign_keys", [])

        if not db_id:
            logging.warning("db_id가 없는 데이터베이스를 건너뜁니다.")
            continue

        if len(column_names) != len(column_types):
            raise ValueError(
                f"{db_id}: column_names_original과 column_types의 길이가 "
                f"다릅니다. columns={len(column_names)}, "
                f"types={len(column_types)}"
            )

        if len(readable_column_names) != len(column_names):
            raise ValueError(
                f"{db_id}: column_names와 column_names_original의 길이가 "
                f"다릅니다."
            )

        foreign_key_map: dict[int, int] = {}

        for pair in foreign_key_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(
                    f"{db_id}: 잘못된 foreign_keys 항목입니다: {pair}"
                )

            source_index, target_index = pair
            foreign_key_map[source_index] = target_index

        for column_index, column_info in enumerate(column_names):
            if not isinstance(column_info, list) or len(column_info) != 2:
                raise ValueError(
                    f"{db_id}: 잘못된 컬럼 항목입니다: {column_info}"
                )

            table_index, column_name = column_info
            column_name = clean_text(column_name)

            # Spider의 특수 전체 컬럼 항목 [-1, "*"] 제외
            if table_index == -1 or column_name == "*":
                continue

            if not 0 <= table_index < len(table_names):
                raise ValueError(
                    f"{db_id}: 잘못된 테이블 인덱스입니다. "
                    f"column_index={column_index}, table_index={table_index}"
                )

            table_name = clean_text(table_names[table_index])

            if table_index < len(readable_table_names):
                table_description = clean_text(
                    readable_table_names[table_index]
                )
            else:
                table_description = table_name

            readable_column_info = readable_column_names[column_index]

            if (
                isinstance(readable_column_info, list)
                and len(readable_column_info) == 2
            ):
                column_description = clean_text(
                    readable_column_info[1]
                )
            else:
                column_description = column_name

            referenced_column_index = foreign_key_map.get(column_index)
            referenced_table = ""
            referenced_column = ""

            if referenced_column_index is not None:
                if not 0 <= referenced_column_index < len(column_names):
                    raise ValueError(
                        f"{db_id}: 잘못된 FK 대상 컬럼 인덱스입니다. "
                        f"source={column_index}, "
                        f"target={referenced_column_index}"
                    )

                referenced_info = column_names[referenced_column_index]
                referenced_table_index = referenced_info[0]
                referenced_column = clean_text(referenced_info[1])

                if not 0 <= referenced_table_index < len(table_names):
                    raise ValueError(
                        f"{db_id}: 잘못된 FK 대상 테이블 인덱스입니다. "
                        f"target_column={referenced_column_index}, "
                        f"table_index={referenced_table_index}"
                    )

                referenced_table = clean_text(
                    table_names[referenced_table_index]
                )

            schema_records.append(
                {
                    "db_id": db_id,
                    "table_index": table_index,
                    "table_name": table_name,
                    "table_description": table_description,
                    "column_index": column_index,
                    "column_name": column_name,
                    "column_description": column_description,
                    "spider_data_type": clean_text(
                        column_types[column_index]
                    ),
                    "data_type": convert_spider_type(
                        column_types[column_index]
                    ),
                    "primary_key": (
                        "Y" if column_index in primary_keys else ""
                    ),
                    "foreign_key": (
                        "Y"
                        if referenced_column_index is not None
                        else ""
                    ),
                    "referenced_table": referenced_table,
                    "referenced_column": referenced_column,
                }
            )

    working_df = pd.DataFrame(schema_records)

    if working_df.empty:
        raise ValueError(
            "train_tables.json에서 처리할 스키마 정보를 찾지 못했습니다."
        )

    duplicate_mask = working_df.duplicated(
        subset=["db_id", "table_name", "column_name"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = working_df.loc[
            duplicate_mask,
            [
                "db_id",
                "table_name",
                "column_name",
                "column_index",
            ],
        ].to_dict("records")

        raise ValueError(
            "동일 DB와 테이블 안에 중복된 컬럼이 있습니다: "
            f"{duplicates}"
        )

    return working_df


# =========================================================
# 6. 테이블별 CREATE TABLE DDL 생성
# =========================================================
def build_table_ddl(table_df: pd.DataFrame) -> str:
    first_row = table_df.iloc[0]

    db_id = clean_text(first_row["db_id"])
    table_name = clean_text(first_row["table_name"])
    table_description = clean_text(first_row["table_description"])

    ddl_items: list[str] = []
    primary_key_columns: list[str] = []
    foreign_key_items: list[str] = []

    for row in table_df.to_dict("records"):
        column_name = quote_identifier(
            clean_text(row["column_name"])
        )
        data_type = clean_text(row["data_type"]) or "TEXT"
        description = sanitize_comment(
            clean_text(row["column_description"])
        )

        comment = f" -- {description}" if description else ""

        ddl_items.append(
            f"    {column_name} {data_type}{comment}"
        )

        if clean_text(row["primary_key"]).upper() == "Y":
            primary_key_columns.append(column_name)

        if clean_text(row["foreign_key"]).upper() == "Y":
            referenced_table = quote_identifier(
                clean_text(row["referenced_table"])
            )
            referenced_column = quote_identifier(
                clean_text(row["referenced_column"])
            )

            foreign_key_items.append(
                f"    FOREIGN KEY ({column_name}) "
                f"REFERENCES {referenced_table} ({referenced_column})"
            )

    if primary_key_columns:
        ddl_items.append(
            "    PRIMARY KEY ("
            + ", ".join(primary_key_columns)
            + ")"
        )

    ddl_items.extend(foreign_key_items)

    comments = [
        f"-- Database: {sanitize_comment(db_id)}",
        (
            "-- Table readable name: "
            f"{sanitize_comment(table_description)}"
        ),
    ]

    return (
        "\n".join(comments)
        + "\n"
        + f"CREATE TABLE {quote_identifier(table_name)} (\n"
        + ",\n".join(ddl_items)
        + "\n);"
    )


# =========================================================
# 7. Qwen3 모델 로드 및 호출
# =========================================================
def load_qwen_model() -> tuple[Any, Any]:
    """Hugging Face Transformers로 Qwen3 토크나이저와 모델을 로드합니다."""
    logging.info("Qwen3 모델 로드 시작: %s", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logging.info("Qwen3 모델 로드 완료")
    return tokenizer, model


def remove_thinking_block(text: str) -> str:
    """혹시 남아 있는 <think>...</think> 블록을 제거합니다."""
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def strip_markdown_code_fence(text: str) -> str:
    """```json ... ``` 또는 ``` ... ``` 형식의 코드 펜스를 제거합니다."""
    text = clean_text(text)

    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        return fenced_match.group(1).strip()

    return text


def extract_json_object(text: str) -> dict[str, Any]:
    """모델 응답에서 첫 번째 완전한 JSON 객체를 추출합니다."""
    cleaned = strip_markdown_code_fence(remove_thinking_block(text))
    start = cleaned.find("{")

    if start < 0:
        raise ValueError(
            "Qwen3 응답에서 JSON 객체 시작 문자를 찾지 못했습니다. "
            f"response={cleaned[:1000]}"
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(cleaned)):
        character = cleaned[index]

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Qwen3 응답의 JSON 객체를 파싱하지 못했습니다. "
                        f"response={candidate[:1000]}"
                    ) from error
                if not isinstance(parsed, dict):
                    raise ValueError("Qwen3 응답의 최상위 JSON 값이 객체가 아닙니다.")
                return parsed

    raise ValueError(
        "Qwen3 응답에서 완전한 JSON 객체를 찾지 못했습니다. "
        f"response={cleaned[:1000]}"
    )


def call_llm(
    tokenizer: Any,
    model: Any,
    *,
    db_id: str,
    table_name: str,
    ddl: str,
    targets: list[dict[str, str]],
) -> TableAugmentation:
    target_json = json.dumps(targets, ensure_ascii=False, indent=2)

    user_prompt = f"""
[Database ID]
{db_id}

[Table name]
{table_name}

[Complete table schema]
{ddl}

[Target columns]
Generate results only for the following columns:
{target_json}

Return only valid JSON.
Do not output Markdown.
Do not output a ```json code fence.
Do not output any explanation before or after the JSON.
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )

    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        generation_kwargs.update(
            {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
            }
        )

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    input_length = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_length:]
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    if not response_text:
        raise ValueError("Qwen3 응답 내용이 비어 있습니다.")

    parsed_json = extract_json_object(response_text)

    try:
        return TableAugmentation.model_validate(parsed_json)
    except ValidationError as error:
        raise ValueError(
            "Qwen3 응답이 Pydantic 스키마와 일치하지 않습니다. "
            f"errors={error.errors()}, response={response_text[:1000]}"
        ) from error


# =========================================================
# 8. 동의어 정규화 및 결과 검증
# =========================================================
def synonym_comparison_key(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[\s·ㆍ_\-–—/]+", "", value)
    value = re.sub(r"[은는이가을를의에에서로으로와과도만]+$", "", value)
    return value


def normalize_synonyms(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for value in values:
        synonym = clean_text(value)

        if not synonym:
            continue

        comparison_key = synonym_comparison_key(synonym)

        if not comparison_key or comparison_key in seen:
            continue

        normalized.append(synonym)
        seen.add(comparison_key)

    return normalized


def is_useful_readable_name(value: str) -> bool:
    value = clean_text(value)

    if not value or value == "*":
        return False

    # 영어 readable name도 한국어 동의어 목록에는 강제로 넣지 않음
    return bool(re.search(r"[가-힣]", value))


def ensure_origin_description(
    synonyms: list[str],
    origin_desc: str,
) -> list[str]:
    origin_desc = clean_text(origin_desc)

    if not is_useful_readable_name(origin_desc):
        return synonyms

    origin_key = synonym_comparison_key(origin_desc)
    existing_keys = {
        synonym_comparison_key(synonym)
        for synonym in synonyms
    }

    if origin_key in existing_keys:
        return synonyms

    if len(synonyms) >= 8:
        synonyms = synonyms[:7]

    return [origin_desc, *synonyms]


def validate_result(
    result: TableAugmentation,
    *,
    expected_db_id: str,
    expected_table: str,
    targets: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if result.db_id != expected_db_id:
        raise ValueError(
            "db_id 불일치: "
            f"expected={expected_db_id}, actual={result.db_id}"
        )

    if result.table != expected_table:
        raise ValueError(
            "테이블명 불일치: "
            f"expected={expected_table}, actual={result.table}"
        )

    expected_names = [
        target["column"]
        for target in targets
    ]

    returned_by_name: dict[str, ColumnAugmentation] = {}

    for item in result.columns:
        if item.column in returned_by_name:
            raise ValueError(
                f"중복 결과 컬럼: {item.column}"
            )

        returned_by_name[item.column] = item

    missing = [
        name
        for name in expected_names
        if name not in returned_by_name
    ]

    extra = [
        name
        for name in returned_by_name
        if name not in expected_names
    ]

    if missing or extra:
        raise ValueError(
            f"컬럼 불일치: missing={missing}, extra={extra}"
        )

    validated: list[dict[str, Any]] = []

    for target in targets:
        column_name = target["column"]
        item = returned_by_name[column_name]

        description = clean_text(item.description)
        synonyms = normalize_synonyms(item.synonyms)
        synonyms = ensure_origin_description(
            synonyms,
            target["readable_name"],
        )

        if not description:
            raise ValueError(
                f"{column_name}: 설명이 비어 있습니다."
            )

        if "\n" in description or "\r" in description:
            raise ValueError(
                f"{column_name}: 설명이 여러 줄입니다."
            )

        if description[-1] not in ".!?。":
            description += "."

        if not 5 <= len(synonyms) <= 8:
            raise ValueError(
                f"{column_name}: 중복 제거 후 동의어가 "
                f"{len(synonyms)}개입니다. values={synonyms}"
            )

        validated.append(
            {
                "db_id": expected_db_id,
                "table": expected_table,
                "column": column_name,
                "readable_name": target["readable_name"],
                "data_type": target["data_type"],
                "description": description,
                "synonyms": synonyms,
            }
        )

    return validated


# =========================================================
# 9. 실패 시 배치 분할 재시도
# =========================================================
def process_with_fallback(
    tokenizer: Any,
    model: Any,
    *,
    db_id: str,
    table_name: str,
    ddl: str,
    targets: list[dict[str, str]],
    result_map: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    if not targets:
        return

    target_names = [target["column"] for target in targets]

    logging.info("[%s.%s] 처리 중: %s", db_id, table_name, target_names)

    try:
        parsed_result = call_llm(
            tokenizer,
            model,
            db_id=db_id,
            table_name=table_name,
            ddl=ddl,
            targets=targets,
        )

        validated_records = validate_result(
            parsed_result,
            expected_db_id=db_id,
            expected_table=table_name,
            targets=targets,
        )

        for record in validated_records:
            key = (record["db_id"], record["table"], record["column"])
            result_map[key] = record
            append_jsonl(CHECKPOINT_JSONL, record)

    except Exception as error:
        logging.exception(
            "[%s.%s] 처리 실패: %s",
            db_id,
            table_name,
            target_names,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(targets) > 1:
            middle = len(targets) // 2
            process_with_fallback(
                tokenizer,
                model,
                db_id=db_id,
                table_name=table_name,
                ddl=ddl,
                targets=targets[:middle],
                result_map=result_map,
            )
            process_with_fallback(
                tokenizer,
                model,
                db_id=db_id,
                table_name=table_name,
                ddl=ddl,
                targets=targets[middle:],
                result_map=result_map,
            )
            return

        append_jsonl(
            ERROR_JSONL,
            {
                "db_id": db_id,
                "table": table_name,
                "column": targets[0]["column"],
                "readable_name": targets[0]["readable_name"],
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


# =========================================================
# 10. 최종 JSON 및 Excel 저장
# =========================================================
def save_outputs(
    working_df: pd.DataFrame,
    result_map: dict[
        tuple[str, str, str],
        dict[str, Any],
    ],
) -> None:
    ordered_results: list[dict[str, Any]] = []
    excel_records: list[dict[str, Any]] = []

    for row in working_df.to_dict("records"):
        key = (
            row["db_id"],
            row["table_name"],
            row["column_name"],
        )

        result = result_map.get(key)

        base_record = {
            "DB ID": row["db_id"],
            "테이블명": row["table_name"],
            "테이블 읽기용 이름": row["table_description"],
            "컬럼명": row["column_name"],
            "컬럼 읽기용 이름": row["column_description"],
            "Spider 데이터 타입": row["spider_data_type"],
            "DDL 데이터 타입": row["data_type"],
            "PK": row["primary_key"],
            "FK": row["foreign_key"],
            "참조 테이블": row["referenced_table"],
            "참조 컬럼": row["referenced_column"],
        }

        if result is None:
            excel_records.append(
                {
                    **base_record,
                    "LLM 비즈니스 설명": "",
                    "LLM 동의어": "",
                    "LLM 동의어 JSON": "",
                    "처리 상태": "미처리 또는 오류",
                }
            )
            continue

        output_record = dict(result)
        ordered_results.append(output_record)

        excel_records.append(
            {
                **base_record,
                "LLM 비즈니스 설명": result["description"],
                "LLM 동의어": ", ".join(result["synonyms"]),
                "LLM 동의어 JSON": json.dumps(
                    result["synonyms"],
                    ensure_ascii=False,
                ),
                "처리 상태": "완료",
            }
        )

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            ordered_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    result_df = pd.DataFrame(excel_records)

    summary_df = (
        result_df.groupby("DB ID", sort=False)
        .agg(
            테이블_수=("테이블명", "nunique"),
            전체_컬럼_수=("컬럼명", "size"),
            완료_컬럼_수=(
                "처리 상태",
                lambda values: (values == "완료").sum(),
            ),
        )
        .reset_index()
    )

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl",
    ) as writer:
        result_df.to_excel(
            writer,
            sheet_name="스키마_증강결과",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="DB별_처리현황",
            index=False,
        )


# =========================================================
# 11. Dry-run
# =========================================================
def run_dry_run(working_df: pd.DataFrame) -> None:
    database_count = working_df["db_id"].nunique()
    table_count = working_df[
        ["db_id", "table_name"]
    ].drop_duplicates().shape[0]
    column_count = len(working_df)

    logging.info(
        "Dry-run 결과: DB %s개, 테이블 %s개, 컬럼 %s개",
        database_count,
        table_count,
        column_count,
    )

    first_row = working_df.iloc[0]
    first_db_id = first_row["db_id"]
    first_table_name = first_row["table_name"]

    first_table_df = working_df[
        (working_df["db_id"] == first_db_id)
        & (working_df["table_name"] == first_table_name)
    ]

    print("\n===== 첫 번째 테이블 DDL =====")
    print(build_table_ddl(first_table_df))
    print("================================\n")


# =========================================================
# 12. 메인 실행
# =========================================================
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    reset_output_files()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"입력 JSON 파일을 찾을 수 없습니다: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        raw_schema = json.load(file)

    if not isinstance(raw_schema, list):
        raise TypeError(
            "train_tables.json의 최상위 구조가 리스트가 아닙니다."
        )

    working_df = prepare_schema(raw_schema)

    logging.info(
        "Spider 파싱 완료: DB %s개, 테이블 %s개, 컬럼 %s개",
        working_df["db_id"].nunique(),
        working_df[["db_id", "table_name"]].drop_duplicates().shape[0],
        len(working_df),
    )

    if DRY_RUN:
        run_dry_run(working_df)
        return

    tokenizer, model = load_qwen_model()
    result_map = load_checkpoint(CHECKPOINT_JSONL)

    logging.info("기존 체크포인트 결과: %s개", len(result_map))

    for (db_id, table_name), table_df in working_df.groupby(
        ["db_id", "table_name"],
        sort=False,
    ):
        ddl = build_table_ddl(table_df)
        pending_targets: list[dict[str, str]] = []

        for row in table_df.to_dict("records"):
            key = (row["db_id"], row["table_name"], row["column_name"])
            if key in result_map:
                continue

            pending_targets.append(
                {
                    "column": row["column_name"],
                    "readable_name": row["column_description"],
                    "data_type": row["data_type"],
                }
            )

        for start in range(0, len(pending_targets), TARGET_BATCH_SIZE):
            batch = pending_targets[start:start + TARGET_BATCH_SIZE]
            process_with_fallback(
                tokenizer,
                model,
                db_id=db_id,
                table_name=table_name,
                ddl=ddl,
                targets=batch,
                result_map=result_map,
            )

    save_outputs(working_df, result_map)

    completed = sum(
        (row["db_id"], row["table_name"], row["column_name"]) in result_map
        for row in working_df.to_dict("records")
    )

    logging.info(
        "스키마 증강 완료: %s/%s개 컬럼, JSON=%s, Excel=%s",
        completed,
        len(working_df),
        OUTPUT_JSON,
        OUTPUT_EXCEL,
    )

    if completed < len(working_df):
        logging.warning(
            "처리하지 못한 컬럼이 있습니다. 오류 로그: %s",
            ERROR_JSONL,
        )


if __name__ == "__main__":
    main()