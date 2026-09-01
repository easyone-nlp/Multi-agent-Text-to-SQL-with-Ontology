# Spider-Ko Multi-Agent Text-to-SQL (Draft)

Spider-Ko 질의를 받아 관련 스키마를 고르고 SQL을 생성·검토·실행·수정하는 최소 실험 코드입니다. 외부 패키지 없이 실행되며, 실제 모델은 OpenAI-compatible API(vLLM 포함)로 선택 연결할 수 있습니다.

## 전체 흐름

```text
Korean question + DatabaseSchema
        │
        ├─ lexical_scout ───────────┐
        ├─ intent_scout ────────────┤ 점수 보조
        └─ llm_schema_scout ────────┤ Qwen call 1: semantic endpoint 선택
                                    ↓
                           llm_schema_critic
                           Qwen call 2: 누락만 추가
                                    ↓
                          db_value_grounder
             Qwen call 3: filter/value 후보 → read-only SQLite 검증
                   exact/casefold/contains/categorical domain
                                    ↓
                              schema_graph
                      FK 최단 경로·bridge·join key 복원
                                    ↓
                                reviewer
                       가중 합산·candidate pruning
                                    ↓
                               LinkingResult
                      selected_tables / selected_columns
                                    ↓
                 선택 사항: SQL 생성·검토·실행·수정
```

정답 SQL은 위 pipeline에 입력되지 않습니다. 실행이 끝난 뒤 schema linking과 생성 SQL을 평가할 때만 사용합니다.

## Schema linking 구성

이 프로젝트에서 agent는 table/column 점수를 담은 `AgentProposal`을 생성하거나 최종 후보를 결정하는 실행 단위입니다. Component는 agent가 공유하는 schema 표현, 모델 연결, 응답 검증, orchestration, 평가 기능을 의미합니다.

### Agent 역할

| Agent | 방식과 모델 사용 | 입력 | 핵심 역할 | 출력 |
|---|---|---|---|---|
| `lexical_scout` | 규칙 기반, 모델 호출 없음 | 한국어 질문, 전체 schema | 작은 한→영 별칭 사전과 이름 token overlap으로 table/column 후보 점수를 제안 | lexical `AgentProposal` |
| `intent_scout` | 규칙 기반, 모델 호출 없음 | 한국어 질문, 전체 schema | 이름·나이·날짜·위치·평균 같은 의도 단서를 schema column 개념으로 변환 | intent `AgentProposal` |
| `llm_schema_scout` | Qwen 호출 1회 | 질문, table·column·type·PK·FK | SELECT, filter, group, having, order, aggregation에 직접 필요한 semantic endpoint를 고름. 연결 전용 bridge와 join key는 의도적으로 제외 | 검증된 table/column `AgentProposal` |
| `llm_schema_critic` | 같은 Qwen 호출 1회 | 질문, 전체 schema, scout 선택 결과 | scout가 놓친 semantic schema만 추가. 부정, subquery, UNION·EXCEPT·INTERSECT의 모든 branch를 재검토 | missing-only `AgentProposal` |
| `db_value_grounder` | Qwen 호출 1회 + read-only SQLite probe | 질문, 전체 schema, scout·critic 결과, 원본 DB | filter span, 번역·음역·코드값과 후보 column을 제안한 뒤 실제 DB에서 exact/casefold/substring match 또는 categorical domain을 확인 | 신뢰 column proposal + `value_grounding` evidence |
| `schema_graph` | 결정적 graph 탐색, 모델 호출 없음 | 전체 PK/FK graph, scout·critic·value proposal | semantic endpoint 모든 쌍의 FK 최단 경로를 구해 중간 bridge table과 경로의 양쪽 join key를 복원 | graph `AgentProposal` |
| `reviewer` | 규칙 기반 집계, 모델 호출 없음 | 앞 agent의 모든 proposal, 가중치와 pruning 설정 | 점수를 가중 합산하고 신뢰 후보를 제한한 뒤 최종 table/column을 선택 | 최종 후보와 reviewer trace |

`lexical_scout`와 `intent_scout`는 LLM 모드에서도 제거되지 않습니다. 이들은 LLM이 제안한 후보의 점수를 보조하지만, 규칙 agent만 제안한 항목을 LLM 모드의 최종 candidate에 새로 넣지는 않습니다.

### 지원 component

| Component | 파일 | 역할 |
|---|---|---|
| Dataset/schema loader | `schema_agents/data.py` | Spider-Ko CSV, Spider table JSON, SQLite DB 경로를 읽음 |
| `DatabaseSchema` | `schema_agents/models.py` | table, `table.column`, type, PK, FK를 공통 형식으로 보관 |
| `AgentProposal` | `schema_agents/models.py` | agent별 table score, column score, 판단 이유를 전달 |
| Model client | `schema_agents/model_client.py` | vLLM 등 OpenAI-compatible `/chat/completions` endpoint 호출 |
| LLM response validator | `schema_agents/schema_llm_agent.py` | JSON object를 추출하고 실제 schema 이름으로 대소문자를 정규화하며 존재하지 않는 후보를 제거 |
| `MultiAgentSchemaLinker` | `schema_agents/orchestrator.py` | agent 생성과 `scout → critic → value probe → graph → reviewer` 실행 순서를 관리 |
| `LinkingResult` | `schema_agents/models.py` | 최종 table/column, 전체 score, 선택적인 `agent_trace`를 반환 |
| Schema evaluator | `schema_agents/evaluation.py` | gold SQL에서 schema link를 추출해 recall, precision, strict recall을 계산 |

### Candidate 선택 규칙

1. `lexical_scout`, `intent_scout`, 선택적인 `llm_schema_scout`가 각각 proposal을 만듭니다.
2. 모델이 연결되어 있으면 critic이 scout 결과를 보고 누락 후보만 추가합니다.
3. value grounder가 filter 조건과 DB value 후보를 모델로 추출하고, schema whitelist로 검증된 identifier와 parameter binding만 사용해 SQLite를 read-only로 조회합니다. 모델이 만든 SQL은 실행하지 않습니다.
4. exact/casefold/substring match와 low-cardinality categorical domain evidence를 `value_grounding`에 보존합니다. DB에서 확인된 column은 trusted candidate와 graph semantic anchor가 됩니다.
5. graph agent는 scout, critic, value grounder의 table을 semantic anchor로 사용합니다. 두 anchor 사이의 최단 FK 경로에 있는 중간 table과 join column을 추가합니다.
6. reviewer는 `config.json`의 agent별 가중치를 곱해 동일 후보의 점수를 합산합니다.
7. LLM 후보가 존재하면 최종 candidate 집합은 `llm_schema_scout`, `llm_schema_critic`, `db_value_grounder`, `schema_graph`가 제안한 항목으로 제한됩니다. 규칙 agent 점수는 이 candidate들의 순위만 보조합니다.
8. Table은 최고 점수 대비 `relative_table_threshold`를 적용한 뒤 `max_tables`까지 선택합니다. Column은 선택 table에 속한 신뢰 후보를 점수순으로 `max_columns`까지 선택합니다.
9. 최종 결과는 `selected_tables`, `selected_columns`, 전체 score를 포함하며 `--show-trace`를 사용하면 agent별 proposal과 이유도 `agent_trace`에 저장됩니다.

### 모델 호출과 fallback

| 실행 모드 | Schema linking 모델 호출 | 실행 경로 |
|---|---:|---|
| `heuristic` | 예제당 0회 | lexical + intent → heuristic graph 확장 → reviewer |
| `openai_compatible` | 예제당 최대 3회 | lexical + intent + LLM scout → LLM critic → DB value grounder → FK path closure → reviewer |

- Scout 호출이나 JSON parsing이 실패하면 빈 scout proposal과 오류 이유를 남깁니다. 이후 critic 또는 규칙 후보로 계속 진행합니다.
- Critic이 실패하면 scout 후보를 유지하고 graph/reviewer를 계속 실행합니다.
- Value candidate 추출 또는 SQLite probe가 실패하면 빈 value proposal과 이유를 남기고 기존 pipeline을 계속 실행합니다.
- DB probe는 원본 SQLite를 `mode=ro`, `PRAGMA query_only=ON`으로 열고, schema에서 검증한 identifier와 bound parameter만 사용합니다.
- `enable_value_grounding=false`로 세 번째 모델 호출과 DB probe를 비활성화할 수 있습니다.
- LLM semantic anchor가 하나도 없으면 graph agent는 규칙 점수가 가장 높은 table을 seed로 선택해 한 단계 FK 이웃을 확장합니다.
- Graph agent는 Spider schema에 선언된 FK만 사용하므로 누락된 FK metadata는 자동 복원할 수 없습니다.
- Value grounding을 켠 validation 1,034개 전체 실행은 schema linking 모델 요청을 최대 3,102회 발생시킵니다.

## SQL 생성 구성

SQL 생성은 `--generate-sql`을 지정했을 때만 schema linking 뒤에 실행됩니다.

1. `sql_drafter`: 선택된 스키마만 보고 첫 SQLite SQL을 생성합니다.
2. `sql_critic`: read-only 여부와 후보 밖 테이블 사용을 검토합니다.
3. `sqlite_executor`: 원본 Spider DB를 read-only로 열어 결과 일부를 확인합니다.
4. `sql_repair`: 실행 오류를 받아 최대 `max_repairs`회 수정합니다.

`sql_drafter` 입력은 schema linking JSON의 `selected_tables`와 `selected_columns`를 그대로 문자열로 전달하는 데서 끝나지 않습니다. Spider `tables.json`의 metadata를 결합해 다음처럼 렌더링합니다.

- 선택된 column의 `column_type`
- 선택된 key column의 `PK` 여부
- 선택된 FK column의 참조 대상 `FK -> table.column`
- 선택된 column 사이에 Spider가 선언한 FK 관계 목록
- schema linking에서 확인된 question span, filter column, 실제 matched value 또는 categorical domain

이 단계에서는 최단 join path를 계산하거나 bridge table 사용을 강제하지 않습니다. 어떤 table과 관계가 질문 해결에 실제로 필요한지는 SQL 모델이 판단하며, metadata 보강 효과만 비교할 수 있도록 입력 표현만 변경합니다.

## 1. 모델 없이 바로 실행

기본 `heuristic` provider는 GPU와 API key가 필요 없습니다.

```bash
cd "/home/dilab/Desktop/hackathon/schema linking/multi-agent"
python main.py --split validation --limit 3 --generate-sql --show-trace
```

직접 질문하기:

```bash
python main.py --split validation --db-id concert_singer \
  --question "가수의 이름과 나이를 보여줘" \
  --generate-sql --show-trace
```

기존 schema linking만 실행하려면 `--generate-sql`을 생략합니다.

## 2. vLLM/Qwen 모델 연결

예를 들어 OpenAI-compatible 서버가 `localhost:8000`에서 실행 중이면 schema linking과 SQL 생성에 같은 Qwen endpoint를 각각 연결할 수 있습니다.

```bash
python main.py --split validation --limit 3 --generate-sql \
  --linking-provider openai_compatible \
  --linking-model Qwen/Qwen3-4B-Instruct-2507 \
  --linking-base-url http://localhost:8000/v1 \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://localhost:8000/v1 \
  --show-trace
```

Schema linking만 전체 validation 1,034개에 실행하려면 `--generate-sql`을 생략합니다. `--limit 1034`는 전체 실행이며, `--limit 3`은 앞의 3개만 실행합니다.

```bash
python main.py --split validation --limit 1034 \
  --linking-provider openai_compatible \
  --linking-model Qwen/Qwen3-4B-Instruct-2507 \
  --linking-base-url http://localhost:8000/v1 \
  --output outputs/qwen_schema_linking.json
```

결과는 지정한 단일 pretty JSON 파일에 `{"summary": ..., "results": [...]}` 형태로 저장됩니다. API key가 필요한 endpoint는 기본적으로 `OPENAI_API_KEY` 환경 변수를 읽습니다. scout/critic 연결이나 JSON parsing이 실패하면 trace에 오류를 기록하고 이용 가능한 후보로 계속 진행합니다.

## 3. 저장된 schema linking으로 SQL만 생성

이전에 저장한 schema linking JSON은 `--linking-input`으로 재사용할 수 있습니다. 이 모드에서는 `MultiAgentSchemaLinker`와 schema scout/critic/value 호출을 건너뛰고, 저장된 `selected_tables`, `selected_columns`, 선택적인 `value_grounding` evidence를 `LinkingResult`로 복원해 SQL pipeline에 바로 전달합니다.

```bash
python main.py \
  --split validation \
  --limit 1034 \
  --linking-input outputs/qwen_schema_linking_all1.json \
  --generate-sql \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --max-repairs 2 \
  --show-trace \
  --output outputs/qwen_text_to_sql_all1.json
```

대표적인 bridge-table 실패 사례(index 99)만 먼저 확인하려면 다음과 같이 실행합니다.

```bash
python main.py \
  --split validation \
  --offset 99 \
  --limit 1 \
  --linking-input outputs/qwen_schema_linking_all1.json \
  --generate-sql \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --max-repairs 2 \
  --show-trace \
  --output outputs/qwen_text_to_sql_spider_metadata_index99.json
```

전체 재평가 결과는 기존 `qwen_text_to_sql_all1.json`을 덮어쓰지 않도록 새 파일명으로 저장하는 것이 좋습니다.

```bash
python main.py \
  --split validation \
  --limit 1034 \
  --linking-input outputs/qwen_schema_linking_all1.json \
  --generate-sql \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --max-repairs 2 \
  --output outputs/qwen_text_to_sql_all2_spider_metadata.json
```

이 명령에서는 `--linking-provider`, `--linking-model`, `--linking-base-url`이 필요하지 않습니다. Schema linking의 예제당 두 번 호출은 다시 발생하지 않고, SQL 초안 생성 1회와 실행 실패 시 최대 `max_repairs`회의 SQL 수정 호출만 발생합니다.

일부 구간만 생성하거나 중단된 위치에서 재개하려면 같은 input을 유지하고 범위를 지정합니다.

```bash
python main.py \
  --split validation \
  --offset 300 \
  --limit 100 \
  --linking-input outputs/qwen_schema_linking_all1.json \
  --generate-sql \
  --provider openai_compatible \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --output outputs/qwen_text_to_sql_300_399.json
```

Cache loader는 `index`, `db_id`, `question`을 대조하고 현재 Spider schema에 존재하지 않는 table/column이 있으면 실행을 중단합니다. `--output`은 원본 `--linking-input`과 다른 경로여야 합니다. 결과에는 `schema_linking_source`, `sql_generation`, `sql_evaluation`과 SQL summary 지표가 기록됩니다.

## 결과 저장과 설정

```bash
python main.py --split validation --db-id concert_singer --limit 5 \
  --generate-sql --output outputs/concert_singer.json
```

[config.json](./config.json)에서 schema agent 가중치와 SQL provider/model/repair 횟수를 조정할 수 있습니다.

```json
{
  "sql_generation": {
    "provider": "heuristic",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": "http://localhost:8000/v1",
    "max_repairs": 2
  }
}
```

custom schema/DB를 사용할 때는 `--tables`, `--examples`, `--database-root`를 함께 지정합니다. DB는 `<database-root>/<db_id>/<db_id>.sqlite` 구조여야 합니다.

## 평가 지표

저장 JSON은 `summary`와 예제별 `results`로 구성됩니다.

```json
{
  "summary": {"sql_execution_match": 0.5},
  "results": [{"schema_linking_evaluation": {}, "sql_evaluation": {}}]
}
```

Schema linking 지표:

- `mean_*_recall`: 필요한 gold schema를 후보가 얼마나 포함했는지
- `mean_*_precision`: 선택 후보 중 gold schema 비율
- `strict_*_recall`: gold schema를 하나도 빠뜨리지 않은 예제 비율
- `average_predicted_*`: 평균 후보 크기
- `schema_linking_model_success_rate`: scout와 critic이 모두 정상 응답한 예제 비율
- `schema_linking_scout_success_rate`, `schema_linking_critic_success_rate`: 각 모델 단계의 정상 응답 비율

SQL 지표:

- `sql_execution_success_rate`: SQL이 오류 없이 실행된 비율이며 정답률은 아님
- `sql_normalized_exact_match`: 대소문자·공백·식별자 quote를 정규화한 문자열 일치율
- `sql_execution_match`: prediction과 gold의 SQLite 결과 일치율. gold에 `ORDER BY`가 없으면 행 순서를 무시

이 SQL 지표는 stdlib 기반 draft evaluator이며 공식 Spider test-suite 수치로 표기하면 안 됩니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

## 파일 구조

```text
multi-agent/
├── main.py
├── config.json
├── schema_agents/
│   ├── data.py                # CSV/tables/SQLite 경로
│   ├── models.py              # 공통 schema linking 타입
│   ├── schema_llm_agent.py    # Qwen semantic scout·critic 및 JSON validator
│   ├── agents.py              # schema linking agents
│   ├── orchestrator.py        # schema linking 순서
│   ├── model_client.py        # OpenAI-compatible stdlib client
│   ├── sql_agents.py          # drafter/critic/executor/repair
│   ├── sql_orchestrator.py    # bounded repair loop
│   └── evaluation.py          # schema linking 및 SQL 평가
└── tests/
```

## 현재 draft의 한계

- heuristic SQL은 COUNT, 단일 테이블 projection, 단순 aggregate/order 정도만 다룹니다.
- 복잡한 JOIN, nested SQL, GROUP BY는 모델 provider 사용을 전제로 합니다.
- SQL 실행 성공은 정답을 의미하지 않습니다. 현재 summary의 `sql_execution_success_rate`는 문법/실행 가능성 지표입니다.
- evaluator는 SQL 문자열 기반 schema recall이라 alias와 중복 컬럼에 한계가 있습니다.
- LLM schema linking은 예제당 두 번 호출하므로 전체 validation은 2,068회의 모델 요청이 발생합니다.
