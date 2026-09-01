# Spider-Ko 대상 synonym 제외 빌드 + multi-agent 성능 평가

## 요청 배경과 범위 결정

"Spider 1.0 ko 데이터셋 전체에 대해 synonym 없이 KG를 구축해달라"는 요청으로 시작했으나, 이어서 "최종 결과는 validation.csv 앞 100건만 확인하면 된다(multi-agent에 KG 붙였을 때의 성능)"는 요청으로 범위가 좁혀졌습니다. 그에 따라 실제로 필요한 db_id만 확인해 구축 범위를 정했습니다.

```
validation.csv 앞 100행의 고유 db_id: concert_singer, pets_1, car_1  (3개)
```

**왜 이 3개를 하나로 합치지 않았는가**: Spider(-ko)는 [AI Hub의 "서울인구관" 케이스](./report.md)와 정반대 구조입니다. `concert_singer`/`pets_1`/`car_1`은 서로 무관한 도메인의 독립된 DB이고, 질문마다 정답 `db_id`가 이미 주어져 있습니다. multi-agent 파이프라인도 원래 "질문 1개 + 그 질문이 속한 DB 스키마 1개"를 입력으로 받는 구조(`AgenticMultiAgentSchemaLinker.link(question, schema, database_path)`)이므로, KG도 db_id별로 따로 만들고 질문의 db_id에 맞는 KG를 골라 끼우는 것이 맞습니다. 합치면 서로 다른 도메인의 동명 컬럼(`Name` 등)이 사전에서 충돌하고, 존재하지 않는 조인이 Steiner tree에 나타날 수 있습니다.

## `--no-synonyms`: synonym 제외 빌드

`danke_kg/agents.py`, `orchestrator.py`에 `include_synonyms` 토글을 추가했습니다 (`build_ontology.py --no-synonyms` / `build_ontology_aihub.py --no-synonyms`).

사용자와 확인한 최종 범위: **테이블(class)·컬럼(property)·관계(object property) synonym만 제외하고, `value_synonyms`(enum 값 동의어, 예: 성별 "남자"→["남성"])는 유지**합니다. value_synonyms는 이름 취향이 아니라 DB의 실제 비즈니스 로직(코드값 매핑)을 담고 있어서 나중에 사람이 채울 "이름 동의어"와는 성격이 다르다고 판단했습니다.

```bash
cd ontology
python3 build_ontology.py --db-id concert_singer --no-synonyms
python3 build_ontology.py --db-id pets_1 --no-synonyms
python3 build_ontology.py --db-id car_1 --no-synonyms
```

## 실제로 발견한 문제: synonym을 빼니 일부 테이블이 영어 라벨로 남음

synonym 필드가 있을 때는 "primary_label은 한국어로, 원래 영어/기술 용어는 synonym에 넣어라"는 지시가 자연스럽게 언어를 분리해줬습니다. synonym 필드를 없애자, 일부 테이블에서 LLM이 `primary_label` 자체를 영어로 채우는 현상이 나타났습니다. 예:

```
concert_singer: singer, singer_in_concert 테이블 전체가 영어 라벨 ("name","age","country"...)
                반면 stadium, concert 테이블은 정상적으로 한국어 라벨
car_1:          car_makers, car_names, cars_data 테이블 전체가 영어 라벨
pets_1:         Has_Pet 테이블만 영어 라벨 ("has pet")
```

**시도한 대응과 결과**:
1. 프롬프트에 "primary_label은 반드시 한국어여야 한다"는 명시적 규칙과 예시를 추가 → 재현되는 테이블에는 효과 없음(같은 테이블이 재빌드 후에도 여전히 영어).
2. 한글이 전혀 없는 라벨을 감지해 LLM에 재교정을 요청하는 재시도 로직을 추가 → 오히려 일부 테이블에서 응답이 완전히 빈 JSON(`primary_label:""`, `properties:[]`)으로 돌아오는 새로운 실패를 유발해서 **되돌림**. 안정성을 우선해 단순한 첫 호출 결과를 그대로 사용하기로 함.
3. 대신 `TableOntologyAgent`가 각 테이블 보강 결과에 `_non_korean_labels`(한글이 없는 라벨 목록)를 정보성으로 남기도록 해서, 결과 JSON의 `agent_trace`를 보면 어느 테이블이 이 문제를 겪었는지 바로 확인할 수 있게 했습니다.

**결론**: 이건 synonym을 빼면서 생긴 실제 트레이드오프입니다. synonym 필드가 있으면 이 정도로 두드러지지 않았을 가능성이 높습니다(`dog_kennels`를 synonym 포함으로 빌드했을 때는 모든 테이블이 일관되게 한국어 라벨을 받았음, [report.md](./report.md) 참고). 사용자가 나중에 synonym을 직접 채울 계획이므로, 이 영어로 남은 `primary_label`들도 함께 검수해서 한국어로 바꿔주시는 게 필요합니다 — 각 `<db_id>_knowledge_schema.json`에서 한글이 없는 `primary_label`을 찾으면 됩니다.
l
## multi-agent 성능 평가 (validation.csv 앞 100건)

`run_multiagent_experiment.py`를 확장해 db_id가 섞인 100개 질문을 한 번에 처리하고, `schema_agents.evaluation.evaluate()`로 각 질문의 gold SQL 대비 table/column recall을 실제로 채점하도록 했습니다 (이전엔 "두 리트리버가 서로 일치하는가"만 봤다면, 이번엔 "정답과 얼마나 맞는가"까지 봄).

```bash
cd ontology
python3 run_multiagent_experiment.py --limit 100 --output output/multiagent_experiment_100.json
```

실행 로그: `output/multiagent_experiment_100_log.txt` (또는 background task `b0ua5h5oz`), 원본 결과: [`output/multiagent_experiment_100.json`](../output/multiagent_experiment_100.json).

### 전체 요약 (100문항, 3개 db_id)

| db_id | 문항 수 |
|---|---|
| concert_singer | 45 |
| pets_1 | 42 |
| car_1 | 13 |

| 지표 | DANKE KG (no-synonym) | 임베딩 리트리버 |
|---|---:|---:|
| table recall (평균) | **0.760** | 1.000 |
| table precision (평균) | **0.650** | 0.476 |
| strict table recall율 (필요한 테이블을 정확히 포함) | 0.690 | 1.000 |
| column recall (평균) | 0.772 | 1.000 |
| 두 방식 최종 테이블 선택 일치율 | 0.30 | — |
| 실행 실패(예외) | 0 / 100 | 0 / 100 |

### db_id별 breakdown

| db_id | DANKE recall | DANKE precision | DANKE strict | DANKE col_recall | 임베딩 recall | 임베딩 precision | 일치율 |
|---|---:|---:|---:|---:|---:|---:|---:|
| concert_singer | 0.681 | 0.572 | 0.622 | 0.651 | 1.000 | 0.361 | 0.156 |
| pets_1 | 0.853 | 0.782 | 0.786 | 0.912 | 1.000 | 0.651 | 0.476 |
| car_1 | 0.731 | 0.492 | 0.615 | 0.722 | 1.000 | 0.308 | 0.231 |

### 해석

- **recall/strict/column recall은 임베딩이 항상 확실히 우세합니다(전부 1.000).** 임베딩 리트리버는 top-k가 충분히 넉넉하고, 매칭 실패 시에도 점수 기반이라 완전히 빗나가는 경우가 거의 없습니다. 반대로 DANKE는 사전 매칭이 실패하면 (a) 전체 스키마로 폴백하거나 (b) 엉뚱한 소수 테이블에만 매칭되어 정답 테이블을 통째로 놓치는 이분법적 실패 양상을 보입니다.
- **precision은 DANKE가 더 높습니다(0.650 vs 0.476).** 매칭이 성공하는 질문에서는 불필요한 테이블을 덜 가져옵니다 — 예: car_1 "등록된 국가는 몇 개인가요?"에서 DANKE는 `countries` 한 테이블만 정확히 선택(precision 1.00)한 반면 임베딩은 관련 없는 6개 테이블 전부를 반환(precision 0.17).
- **두 방식의 최종 선택 일치율은 30%로 낮습니다.** DB별로도 편차가 큽니다: pets_1은 47.6%로 비교적 자주 일치하지만, concert_singer는 15.6%까지 떨어집니다 — 아래 라벨 문제와 직접 연결됩니다.
- **가장 나쁜 db_id는 concert_singer(recall 0.681)이며, 원인은 앞서 발견한 영어 라벨 문제입니다.** `singer`/`singer_in_concert` 테이블 전체가 영어 라벨("name","age","country"...)로 남아 있어서, "이름", "국가", "나이" 같은 한국어 키워드가 사전에서 전혀 매칭되지 않습니다. 그 결과 아래처럼 완전히 엉뚱한 테이블만 선택되는 recall=0.0 케이스가 6건 반복됩니다:
  - "모든 가수의 이름, 국가, 나이를 나이가 많은 순서부터 보여주세요" → DANKE: `['stadium']` / 정답: `['singer']`
  - "가장 어린 가수의 노래 이름과 발매 연도를 보여주세요" → DANKE: `['stadium']` / 정답: `['singer']`
  - "이름에 'Hey'가 포함된 노래를 가진 가수의 이름과 국가는?" → DANKE: `['stadium']` / 정답: `['singer']`

  우연히 `stadium` 테이블은 한국어 라벨("이름" 등)을 갖고 있어서, "가수"라는 키워드가 매칭 안 되는 대신 "이름"만 `stadium`의 컬럼에 매칭되어 버리는 것으로 보입니다.
- **car_1은 `continents` 테이블 라벨 문제로 누락이 반복됩니다.** "대륙은 몇 개인가요?" 류 질문 2건에서 DANKE는 `['countries']`만 선택하고 정답 `['continents']`를 완전히 놓쳤습니다(recall 0.0). "대륙별 국가 수" 같은 join 질문에서도 `continents`가 계속 빠지고 대신 관련 없는 자동차 테이블들이 Steiner tree로 딸려 들어와 precision까지 낮아졌습니다(0.2).
- **pets_1도 부분 매칭 실패가 있지만 상대적으로 덜 심각합니다.** 주로 다리 테이블(`Has_Pet`)이 질문에 명시적으로 언급되지 않아 Steiner 확장이 안 되는 케이스(예: "반려동물을 가진 학생들의 이름과 나이" → `Student`만 선택, `Has_Pet` 누락, recall 0.5)로, 라벨 언어 문제라기보다는 사전에 명시적 연결어(동의어)가 없어서 생기는 전형적인 dictionary-coverage 한계입니다.

### 결론

이번 100문항 검증에서 DANKE KG 리트리버는 (synonym을 제외하고 빌드했음에도) **매칭에 성공하는 질문에서는 임베딩보다 더 좁고 정확한 후보를 냅니다(precision 우세)**. 하지만 **synonym 제외 + 일부 테이블의 영어 라벨 잔존이라는 두 가지 요인이 겹쳐 recall에서 확실한 손해**를 보고 있으며, 특히 `schema_linker_mode="embedding_only"`가 기본값인 현재 파이프라인 구조상(위 "API 변경 대응" 절 참고) 이 recall 손실이 LLM 재검증 없이 그대로 최종 결과에 반영됩니다. 사용자가 계획한 대로 나중에 synonym을 채우고, `_non_korean_labels`로 표시된 영어 라벨들을 한국어로 교정하면 이 recall 격차는 크게 좁혀질 것으로 예상됩니다 — 이번 실험은 정확히 "synonym 없이 만들면 어떤 대가를 치르는가"를 수치로 보여준 것으로 해석하는 것이 맞습니다. 실전에서는 두 리트리버를 합집합으로 병행하는 하이브리드 방식(위 "다음에 시도해볼 만한 것" 참고)이 recall 손실 없이 DANKE의 precision 이점을 살릴 수 있는 현실적 대안입니다.
