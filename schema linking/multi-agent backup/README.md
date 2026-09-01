# Spider-Ko Agentic Multi-Agent Text-to-SQL

Spider-Ko의 한국어 질문을 대상으로 schema linking과 SQL 생성을 실험하는 manager-style multi-agent draft입니다.

현재 CLI 런타임은 규칙 기반 lexical/intent agent, weighted voting, deterministic score pruning을 사용하지 않습니다. 초기 후보 검색에만 multilingual embedding similarity를 사용하고, 이후 선택은 Qwen agent의 구조화된 판단으로 수행합니다. 이전 weighted 구현 설명은 [README_LEGACY.md](./backup/README_LEGACY.md)에 보관했습니다.

## 왜 orchestrator가 필요한가

이 프로젝트의 orchestrator는 모든 작업을 직접 해결하는 상위 전문가가 아니라 다음 책임을 가진 manager입니다.

1. 질문을 SELECT, filter, grouping, ordering, aggregation, set operation, 관계 요구로 분해합니다.
2. 현재 shared state를 보고 value linker와 join linker가 필요한지 결정합니다.
3. specialist에게 범위가 제한된 task를 전달합니다.
4. SQL generator가 사용할 구조화된 상태의 소유권을 유지합니다.

즉 자유롭게 대화하는 peer agent 집합이 아니라 manager가 control flow를 유지하고 specialist를 호출하는 구조입니다.

## 현재 파이프라인

```text
한국어 질문 + Spider schema + SQLite DB
                  │
                  ▼
        Orchestrator / Qwen
  query decomposition + initial plan
                  │
                  ▼
 Qwen3-Embedding initial top-k retrieval
      table top-k + column top-k
                  │
                  ▼
         Schema Linker / Qwen
 semantic endpoint와 column role 연결
                  │
                  ▼
        Orchestrator / Qwen
 value/join specialist 필요 여부와 task 결정
          │                       │
          ▼                       ▼
 Value Linker / Qwen         Join Linker / Qwen
 filter 후보 생성           PK/FK/bridge/join ON 결정
 read-only DB probe          필요 시 inferred join 표시
 evidence 기반 재판단
          └───────────┬───────────┘
                      ▼
          Deterministic Aggregator
 schema/value/join 결과의 중복 없는 합집합
                      │
                      ▼
             SQL Generator / Qwen
       SQLite SELECT 생성 → 안전검사 → 실행
                      │
              실패 시 Qwen repair
```

Gold SQL은 agent 입력에 들어가지 않으며 실행 후 평가에만 사용됩니다.

## Agent와 tool의 역할

| 구성 요소 | 모델 | 책임 | 하지 않는 일 |
|---|---|---|---|
| `orchestrator` | Qwen3 chat | query decomposition, routing, specialist task 작성 | table/column 의미 연결, DB 값 조회, join 추론, SQL 작성 |
| `embedding_retriever` | Qwen3-Embedding | 전체 schema에서 초기 table/column top-k 검색 | 최종 schema 결정, downstream 점수 투표 |
| `schema_linker` | Qwen3 chat | decomposition의 output/filter/group/order/aggregate/entity role을 검색된 영어 schema에 연결 | DB value 추측 확정, bridge table과 join key 선택 |
| `value_linker` | Qwen3 chat + read-only SQLite tool | filter마다 후보 column/value/조회 방법을 제안하고 DB evidence를 보고 최종 grounding | 모델이 만든 임의 SQL 실행, DB 존재만으로 의미 확정 |
| `join_linker` | Qwen3 chat | full Spider PK/FK metadata로 endpoint를 연결하고 bridge table, key, 명시적 ON pair를 반환 | 단순 인접 table 추가, semantic output/filter 재선택 |
| `sql_generator` | Qwen3 chat | manager package와 선택 schema metadata로 최종 SQLite SELECT 생성 | schema linking 재수행 |
| `sql_repair` | 같은 Qwen3 chat | 실행 오류가 있을 때 동일 package 범위에서 SQL 수정 | 새로운 table/column 임의 도입 |

다음은 agent가 아니라 deterministic guard/tool입니다.

- schema identifier validator: 모델 응답에서 실제로 존재하지 않는 table/column을 제거합니다.
- deterministic aggregator: schema/value/join 결과의 table과 column을 중복 없이 합치고 filter 및 join endpoint를 보존합니다.
- read-only DB probe: schema whitelist와 bound parameter를 사용하며 SQLite를 `mode=ro`, `query_only`로 엽니다.
- SQL safety validator: 다중 statement와 write/DDL을 거부합니다.
- SQLite executor: SELECT를 읽기 전용으로 실행해 오류와 일부 행을 반환합니다.
- evaluator: gold SQL과 prediction을 동일한 평가 경로로 비교합니다.

이 보호장치는 후보 score를 계산하거나 schema를 선택하지 않습니다.

## Agent 호출 수

예제 하나의 schema linking에서 기본 호출은 다음과 같습니다.

- orchestrator decomposition: 1회
- schema linker: 1회
- orchestrator routing: 1회
- value linker: 필요 시 2회(조회 계획, evidence 해석)
- join linker: 필요 시 1회

따라서 filter와 join이 모두 필요하면 chat model 호출은 보통 6회입니다. JSON 재시도가 발생하면 늘어납니다. 최종 package 합성에는 모델을 호출하지 않습니다. Embedding endpoint는 별도로 schema 문서와 질의 vector를 계산하며 DB별 schema vector는 프로세스 안에서 cache합니다.

SQL 생성을 켜면 초안 1회가 추가되고 실행 실패 시 `max_repairs` 범위에서 repair 호출이 추가됩니다.

## 출력 schema package

각 결과는 단일 pretty JSON의 `results` 배열에 저장됩니다. 주요 필드는 다음과 같습니다.

```json
{
  "query_decomposition": {},
  "retrieved_schema": {
    "tables": [],
    "columns": []
  },
  "selected_tables": [],
  "selected_columns": [],
  "column_roles": {
    "table.column": ["select", "filter"]
  },
  "grounded_filters": [
    {
      "condition_id": "f1",
      "column": "table.column",
      "operator": "=",
      "value": "stored value",
      "evidence": "..."
    }
  ],
  "joins": [
    {
      "left": "child.fk",
      "right": "parent.pk",
      "join_type": "INNER",
      "inferred": false,
      "reason": "..."
    }
  ],
  "unresolved": [],
  "agent_trace": []
}
```

`table_scores`와 `column_scores`는 이전 JSON 형식과의 호환을 위해 빈 object로 남습니다. 최종 선택에는 사용되지 않습니다.

## 모델 서버

Chat agent용 Qwen 서버 예시:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

Embedding 검색 서버 예시:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Embedding-0.6B \
  --served-model-name Qwen/Qwen3-Embedding-0.6B \
  --task embed \
  --port 8001 \
  --gpu-memory-utilization 0.80
```

vLLM 버전에 따라 embedding 옵션 이름이 다를 수 있으므로 서버의 `vllm serve --help`에서 pooling/embed task 옵션을 확인해야 합니다. 두 모델을 같은 GPU에 올릴 경우 메모리 사용률과 context 길이를 낮춰야 합니다.

## 실행

작업 폴더:

```bash
cd "/home/dilab/Desktop/hackathon/schema linking/multi-agent"
```

100개 schema linking 실험:

```bash
python main.py \
  --split validation \
  --limit 100 \
  --linking-provider openai_compatible \
  --linking-model Qwen/Qwen3-4B-Instruct-2507 \
  --linking-base-url http://127.0.0.1:8000/v1 \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --top-k-tables 6 \
  --top-k-columns 24 \
  --show-trace \
  --output outputs/qwen_agentic_schema_linking_100.json
```

Schema linking부터 SQL 생성까지 100개 실험:

```bash
python main.py \
  --split validation \
  --limit 100 \
  --generate-sql \
  --linking-provider openai_compatible \
  --linking-model Qwen/Qwen3-4B-Instruct-2507 \
  --linking-base-url http://127.0.0.1:8000/v1 \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --max-repairs 2 \
  --show-trace \
  --output outputs/qwen_agentic_text_to_sql_100.json
```

`config.json`에 같은 기본값이 있으므로 서버 주소와 모델명이 동일하면 짧게 실행할 수도 있습니다.

```bash
python main.py --split validation --limit 100 --generate-sql \
  --show-trace --output outputs/qwen_agentic_text_to_sql_100.json
```

저장된 새 schema linking 결과로 SQL만 다시 생성:

```bash
python main.py \
  --split validation \
  --limit 100 \
  --linking-input outputs/qwen_agentic_schema_linking_100.json \
  --generate-sql \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --max-repairs 2 \
  --output outputs/qwen_agentic_text_to_sql_from_cache_100.json
```

## 설정

`config.json`의 핵심 항목:

```json
{
  "schema_linking": {
    "provider": "openai_compatible",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": "http://localhost:8000/v1",
    "json_retries": 1,
    "embedding": {
      "model": "Qwen/Qwen3-Embedding-0.6B",
      "base_url": "http://localhost:8001/v1",
      "top_k_tables": 6,
      "top_k_columns": 24
    },
    "value_linker": {
      "max_candidates": 4,
      "max_domain_values": 20
    }
  }
}
```

`top_k`가 너무 작으면 Qwen agent가 정답 column을 볼 수 없고, 너무 크면 prompt noise와 비용이 증가합니다. 첫 실험에서는 table 6, column 24를 시작점으로 사용하고 retrieval recall과 최종 linking recall을 따로 비교하는 것이 좋습니다.

## 평가 지표

Schema linking:

- `mean_table_recall`, `strict_table_recall`
- `mean_column_recall`, `strict_column_recall`
- table/column precision과 평균 선택 개수
- `embedding_mean_*_recall`, `embedding_strict_*_recall`: 초기
  embedding 후보에 gold schema가 포함됐는지 측정
- `schema_linking_workflow_success_rate`

SQL:

- `sql_execution_success_rate`: 실행 가능 비율이며 정답률이 아닙니다.
- `sql_normalized_exact_match`
- `sql_execution_match`

실패 분석에서는 다음 세 단계를 분리해서 보는 것이 중요합니다.

1. embedding retrieval에 gold schema가 있었는가
2. specialist가 검색된 gold schema를 놓쳤는가
3. linking package는 맞지만 SQL generator가 role/filter/join을 잘못 사용했는가

## 파일 구조

```text
schema_agents/
├── agentic_orchestrator.py    # manager-style control flow
├── agentic_agents.py          # orchestrator, schema linker, join linker
├── agentic_value_linker.py    # Qwen value planning + DB evidence resolution
├── embedding_retriever.py     # Qwen3-Embedding top-k retrieval
├── model_client.py            # chat / embedding OpenAI-compatible clients
├── models.py                  # shared state and result package
├── sql_agents.py              # Qwen SQL generator/repair + safety tools
├── sql_orchestrator.py        # bounded generation/execution/repair loop
└── evaluation.py              # schema linking and SQL evaluation
```

실제 모델이나 전체 데이터셋을 호출하지 않는 단위 테스트는 fake chat/embedding client를 주입하는 방식으로 구성합니다.
