# Spider-Ko Agentic Multi-Agent Text-to-SQL

Spider-Ko의 한국어 질문을 대상으로 schema linking과 SQL 생성을 실험하는 manager-style multi-agent draft입니다.

현재 공식 기본값은 `danke_embedding_hybrid`이며 Steiner와 Join Linker를 사용합니다. Orchestrator가 한 번 만든 원질문, retrieval query, output/filter/group/order/relationship span을 각각 검색하고 provenance를 보존한 채 DANKE의 강한 후보를 우선 보호하고 embedding의 강한 후보로 recall을 보완합니다. semantic table 3개, semantic column 8개와 별도 Steiner bridge 2개/FK key evidence를 Value/Join specialist에 전달합니다. `name`, `이름`, `id` 같은 단독 generic property는 새 table을 만들 수 없고 이미 선택된 class 내부 column 보강에만 사용됩니다. `danke_multisource`와 `embedding_only`는 동결된 비교 기준선으로 계속 사용할 수 있습니다.

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
 DANKE multi-source ontology retrieval
 original + retrieval query + plan spans
 semantic top-3/8 + separate bridge/FK
                  │
                  ▼
 DANKE schema candidates (공식 기본)
 [Schema Linker / Qwen은 qwen mode에서만 선택]
                  │
                  ▼
        Orchestrator / Qwen
 value/join specialist 필요 여부와 task 결정
          │                       │
          ▼                       ▼
 filter-specific embedding   Join Linker / Qwen
 rescue over full schema     endpoint/bridge/FK ON 생성
          │                       │
          ▼                       ▼
 Value Linker / Qwen         strict FK validator
 영문 value 후보 + DB probe  Spider FK가 아닌 edge 차단
 evidence 기반 재판단        불완전 path 차단
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
| `orchestrator` | Qwen3 chat | query decomposition, 관계 작업의 필요성 판단, specialist task 작성 | 구체적인 table/FK edge 생성, DB 값 조회, SQL 작성 |
| `danke_multisource_retriever` | deterministic DANKE KG | source별 dictionary 검색, provenance/dedup, semantic budget, Steiner bridge/FK evidence | DB value 확정, generic property 단독 table 생성 |
| `embedding_retriever` | Qwen3-Embedding | `embedding_only` 비교 모드에서 전체 schema top-k 검색 | 공식 DANKE 후보 확정, DB value 판정 |
| `schema_linker` | Qwen3 chat, 선택 사항 | `qwen` mode에서 decomposition role을 검색된 schema에 연결 | 기본 `embedding_only` mode에서는 호출되지 않음 |
| `value_linker` | Qwen3 chat + read-only SQLite tool | rescue 후보에서 filter column과 영문/코드 value를 제안하고 DB evidence로 최종 grounding | 한국어 문자열을 DB 값으로 확정, `not_found`를 exact match로 확정 |
| `join_linker` | Qwen3 chat | embedding 후보 중 concrete endpoint를 고르고 full Spider metadata에서 bridge/key/ON pair 생성 | validator를 우회한 inferred 또는 비-FK join 확정 |
| `sql_generator` | Qwen3 chat | manager package와 선택 schema metadata로 최종 SQLite SELECT 생성 | schema linking 재수행 |
| `sql_repair` | 같은 Qwen3 chat | 실행 오류가 있을 때 동일 package 범위에서 SQL 수정 | 새로운 table/column 임의 도입 |

다음은 agent가 아니라 deterministic guard/tool입니다.

- schema identifier validator: 모델 응답에서 실제로 존재하지 않는 table/column을 제거합니다.
- value evidence validator: exact/contains/domain filter는 DB probe가 `matched`이고 최종 value가 `matched_values`에 있을 때만 통과시킵니다. 저장된 실제 대소문자 값으로 정규화합니다.
- FK validator: Join Linker가 만든 모든 ON pair를 Spider `foreign_keys`와 대조하고 inferred/non-FK edge 및 두 endpoint를 잇지 못하는 dangling path를 제거합니다.
- deterministic aggregator: schema/value/join 결과의 table과 column을 중복 없이 합치고 filter 및 join endpoint를 보존합니다.
- read-only DB probe: schema whitelist와 bound parameter를 사용하며 SQLite를 `mode=ro`, `query_only`로 엽니다.
- SQL safety validator: 다중 statement와 write/DDL을 거부합니다.
- SQLite executor: SELECT를 읽기 전용으로 실행해 오류와 일부 행을 반환합니다.
- evaluator: gold SQL과 prediction을 동일한 평가 경로로 비교합니다.

이 보호장치는 후보 score를 계산하거나 schema를 선택하지 않습니다.

## Agent 호출 수

예제 하나의 schema linking에서 기본 호출은 다음과 같습니다.

- orchestrator decomposition: 1회
- schema linker: 기본 `danke_multisource`에서는 0회 (`qwen` mode에서는 1회)
- orchestrator routing: 1회
- value linker: 필요 시 2회(조회 계획, evidence 해석)
- join linker: 필요 시 1회

따라서 기본 mode에서 filter와 join이 모두 필요하면 chat model 호출은 보통 5회입니다. JSON 재시도가 발생하면 늘어납니다. 최종 package 합성에는 모델을 호출하지 않습니다. DANKE ontology와 dictionary는 db_id별로 프로세스 안에서 cache되며 filter rescue도 같은 retriever contract를 사용합니다.

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

Embedding 비교 실험 서버 예시(`embedding_only`에서만 필요):

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

공식 DANKE multi-source 100개 schema linking 실험:

```bash
python main.py \
  --split validation \
  --limit 100 \
  --schema-linker-mode danke_multisource \
  --linking-provider openai_compatible \
  --linking-model Qwen/Qwen3-4B-Instruct-2507 \
  --linking-base-url http://127.0.0.1:8000/v1 \
  --show-trace \
  --output outputs/qwen_danke_multisource_schema_linking_100.json
```

공식 DANKE+embedding hybrid는 DANKE의 강한 근거를 먼저 보호하고, source별 embedding 근거가 강한 semantic endpoint만 남은 table 예산에 추가합니다. 두 점수를 직접 합산하지 않으며, 최종 semantic table을 고른 뒤 그 내부에서만 column 8개를 선택합니다. Steiner bridge와 검증된 FK key는 semantic 예산 밖의 별도 evidence로 유지됩니다.

동일한 shared query로 retriever-only 100개를 재현:

```bash
cd "/home/dilab/Desktop/hackathon"

/home/dilab/anaconda3/envs/t2s/bin/python ontology/evaluate_hybrid_retriever.py \
  --query-cache ontology/output/retriever_only_v1/shared_queries_0_99.json \
  --danke-baseline ontology/output/retriever_only_v2/multisource_generic_gate_top3_8_bridge2.json \
  --output ontology/output/retriever_only_hybrid_v1/evidence_tier_top3_8_bridge2.json \
  --embedding-base-url http://127.0.0.1:8001/v1
```

Hybrid schema linking부터 SQL 생성·실행·repair까지 100개 실행:

```bash
cd "/home/dilab/Desktop/hackathon/schema linking/multi-agent"

/home/dilab/anaconda3/envs/t2s/bin/python main.py \
  --split validation \
  --limit 100 \
  --schema-linker-mode danke_embedding_hybrid \
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
  --output outputs/qwen_danke_embedding_hybrid_text_to_sql_100.json
```

Schema Linker를 제외하고 strict embedding top-k만 사용하는 100개 ablation:

```bash
python main.py \
  --split validation \
  --limit 100 \
  --schema-linker-mode embedding_only \
  --strict-retrieval-top-k \
  --top-k-tables 3 \
  --top-k-columns 8 \
  --linking-provider openai_compatible \
  --linking-model Qwen/Qwen3-4B-Instruct-2507 \
  --linking-base-url http://127.0.0.1:8000/v1 \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --show-trace \
  --output outputs/qwen_embedding_only_strict_top3_8_100.json
```

이 모드에서는 Schema Linker Qwen 호출을 생략합니다. Strict retrieval은 상위 table 안에서만 column을 검색하고 table의 모든 column을 자동 추가하지 않으므로 실제 후보 수가 각각 3개와 8개를 넘지 않습니다.

동일한 설정으로 schema linking부터 SQL generation까지 실행:

```bash
python main.py \
  --split validation \
  --limit 100 \
  --schema-linker-mode embedding_only \
  --strict-retrieval-top-k \
  --top-k-tables 3 \
  --top-k-columns 8 \
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
  --output outputs/qwen_embedding_only_strict_top3_8_text_to_sql_100.json
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
    "schema_linker_mode": "embedding_only",
    "embedding": {
      "model": "Qwen/Qwen3-Embedding-0.6B",
      "base_url": "http://localhost:8001/v1",
      "top_k_tables": 6,
      "top_k_columns": 24
    },
    "value_linker": {
      "max_candidates": 4,
      "max_domain_values": 20,
      "max_rescue_queries": 4
    }
  }
}
```

`top_k`가 너무 작으면 output/group/order column의 recall이 낮아질 수 있고, 너무 크면 prompt noise와 비용이 증가합니다. Filter column은 초기 top-k 밖에 있어도 filter 전용 rescue retrieval이 다시 전체 schema를 검색합니다. 다만 rescue는 filter 문제만 복구하므로 initial retrieval recall과 최종 linking recall을 계속 분리해서 봐야 합니다.

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
├── embedding_retriever.py     # initial + filter rescue top-k retrieval
├── join_validator.py          # Spider declared-FK pair lookup helper
├── model_client.py            # chat / embedding OpenAI-compatible clients
├── models.py                  # shared state and result package
├── sql_agents.py              # Qwen SQL generator/repair + safety tools
├── sql_orchestrator.py        # bounded generation/execution/repair loop
├── evaluation.py              # schema linking and SQL evaluation
└── orchestrator.py            # legacy compatibility export
```

실제 모델이나 전체 데이터셋을 호출하지 않는 단위 테스트는 fake chat/embedding client를 주입하는 방식으로 구성합니다.


### Gold schema로 SQL 생성 upper-bound 측정

Schema linking 오류를 제거했을 때 현재 SQL generator가 도달할 수 있는 EX를
확인하려면 `--gold-schema`를 사용합니다.

```bash
python main.py \
  --split validation \
  --limit 100 \
  --gold-schema \
  --generate-sql \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --max-repairs 2 \
  --show-trace \
  --output outputs/qwen_gold_schema_text_to_sql_100.json
```

이 모드는 embedding server와 schema-linking agent를 호출하지 않습니다. Gold
SQL은 `sqlglot`으로 파싱해 정답 table/column identifier만 추출합니다. SQL
generator에는 gold SQL 문자열, literal value, 연산자, clause 구조를 전달하지
않고 다음 입력만 제공합니다.

- 한국어 질문
- gold table/column 목록
- 선택된 컬럼의 Spider type, PK, FK metadata
- 빈 decomposition/value/join package

그 후의 `Qwen SQL draft → safety check → read-only SQLite 실행 → 실행 실패 시
Qwen repair → EX 평가`는 기존 방식과 같습니다. 따라서 이 결과는 공정한
end-to-end 점수가 아니라 schema linking이 완벽하다고 가정한 SQL generation
upper bound입니다. 현재 환경에는 gold schema 추출을 위한 `sqlglot`이
설치되어 있어야 합니다.
