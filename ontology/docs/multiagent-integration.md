# DANKE KG를 기존 multi-agent 파이프라인에 연결하기

## 무엇을 만들었나

`schema linking/multi-agent`의 어떤 파일도 수정하지 않고, 별도 파일 두 개로 DANKE KG를 그 파이프라인에 붙였습니다.

- [`ontology/danke_kg/multiagent_bridge.py`](../danke_kg/multiagent_bridge.py)
  - `DankeSchemaRetriever` — `EmbeddingSchemaRetriever`와 같은 `retrieve(query, schema) -> RetrievedSchema` 인터페이스를 구현. 내부적으로는 임베딩 대신 DANKE KG의 `MatchingDiscoveryService`(사전 매칭) + `steiner_forest`(매칭된 클래스들을 잇는 다리 테이블 확장)로 후보 테이블/컬럼을 만든다. **반환값은 duck-typed 클래스가 아니라 `schema_agents.embedding_retriever.RetrievedSchema`를 그대로 재사용**한다(아래 "API 변경 대응" 참고).
  - `DankeAugmentedSchemaLinker(AgenticMultiAgentSchemaLinker)` — 부모 생성자를 그대로 호출한 뒤 `self.retriever`만 `DankeSchemaRetriever`로 교체하는 서브클래스. 매니저/`SemanticSchemaLinkerAgent`/`ValueLinkerAgent`/`JoinLinkerAgent`는 손대지 않고 그대로 재사용됨.
- [`ontology/run_multiagent_experiment.py`](../run_multiagent_experiment.py) — 위 브리지를 실제로 돌려보는 실험 스크립트. `sys.path`에 `schema linking/multi-agent`를 추가해서 import할 뿐, 그 디렉터리 안의 파일은 하나도 건드리지 않는다.

## 왜 이 지점(retriever)에 붙였나

기존 파이프라인은 `self.retriever.retrieve(query, schema)` 한 번을 호출해 스키마 후보(top-k tables/columns)를 좁힌 뒤, 그 결과를 `SemanticSchemaLinkerAgent`·`JoinLinkerAgent`가 이어받는 구조입니다([agentic_orchestrator.py:111](../../schema%20linking/multi-agent/schema_agents/agentic_orchestrator.py#L111)). DANKE KG는 정확히 이 "후보를 좁히는" 역할(논문의 Matching Discovery + Matching Optimization)을 대체할 수 있어서, 인터페이스만 맞추면 나머지 에이전트는 전혀 수정할 필요가 없었습니다.

## 파이프라인 단계별 Input/Output

`DankeAugmentedSchemaLinker`는 `AgenticMultiAgentSchemaLinker`의 `link()` 흐름을 그대로 따르되, 2단계(retriever)만 DANKE KG로 교체됩니다. 아래는 [`schema_agents/agentic_orchestrator.py`](../../schema%20linking/multi-agent/schema_agents/agentic_orchestrator.py)의 `link()` 실제 코드를 따라간 단계별 데이터 흐름입니다. ★ 표시가 KG가 관여하는 지점입니다.

### 0. 진입점

```
DankeAugmentedSchemaLinker.link(question, schema, database_path, include_sql_generation=False)
```

| | 타입/형태 |
|---|---|
| **Input** `question` | 한국어 질문 문자열 (예: `"각 대륙은 몇 개의 국가를 가지고 있나요?"`) |
| **Input** `schema` | `DatabaseSchema(db_id, tables: list[str], columns: list[Column], primary_keys: set[str], foreign_keys: list[tuple[str,str]])` — Spider `tables.json`에서 로드된 원본 스키마. KG가 아니라 원본 스키마임에 유의(§"KG와 원본 스키마의 관계" 참고). |
| **Input** `database_path` | 실제 sqlite 파일 경로 (value_linker가 DB에 값을 조회할 때 필요) |
| **Output** | `LinkingResult(tables, columns, column_roles, grounded_filters, joins, unresolved, workflow_trace, ...)` — 최종 결과. 아래 8단계에서 조립됨. |

### 1. Manager: query decomposition (KG 관여 없음)

```
manager.decompose(question, include_sql_generation, schema_linker_mode) -> decomposition
```
- **Input**: 질문 문자열, `schema_linker_mode`(`"embedding_only"`)
- **Output** `decomposition`: `{"retrieval_query": str, "entities": [...], "filters": [...], ...}` — LLM이 질문을 분해한 JSON. `retrieval_query`가 다음 단계 리트리버에 그대로 전달됨.

### 2. ★ Retriever: `DankeSchemaRetriever.retrieve()` (KG가 대체하는 지점)

```
retrieved = self.retriever.retrieve(decomposition["retrieval_query"] or question, schema)
```

| | 타입/형태 |
|---|---|
| **Input** `query` | decomposition의 `retrieval_query` (없으면 원본 질문) |
| **Input** `schema` | `DatabaseSchema` — DANKE 리트리버는 이걸 직접 쓰지 않고 버림(`del schema`). 테이블/컬럼 이름은 전부 생성 시 주입된 `self.mapping`(`RDBMapping`, KG의 class/property ↔ 실제 table/column 매핑)에서 옴. |
| 내부 단계 1 | `_candidate_phrases(query)` → 질문을 1~3-gram 어절 조합으로 쪼갠 문자열 리스트 (예: `["대륙", "국가", "대륙 국가", ...]`) |
| 내부 단계 2 | `matching_service.match(keywords)` → 사전(`Dictionary`)에서 exact→fuzzy 매칭, `MatchResult` 리스트 |
| 내부 단계 3 | `matched_classes` → 매칭된 KG 클래스(테이블) 이름 집합 |
| 내부 단계 4 | 매칭된 클래스가 2개 이상이면 `steiner_forest(graph, matched_classes)` → 매칭 안 된 "다리 테이블"까지 포함한 확장된 클래스 집합 |
| 내부 단계 5 | 매칭이 하나도 없으면 KG 전체 클래스로 폴백(`classes = set(knowledge_schema.classes)`) |
| 내부 단계 6 | `mapping.table_for(class_name)` / `mapping.column_for(property_name)` → KG 클래스/속성 이름을 실제 테이블/컬럼 이름으로 역변환 |
| **Output** | `RetrievedSchema(tables: list[str], columns: list["table.column"], query: str, mode="danke_kg")` — `schema_agents.embedding_retriever.RetrievedSchema`와 **동일한 dataclass**를 반환하므로 이후 단계는 이게 KG에서 왔는지 임베딩에서 왔는지 전혀 모름. |

#### 2-1. `DankeSchemaRetriever` 상세

소스: [`ontology/danke_kg/multiagent_bridge.py`](../danke_kg/multiagent_bridge.py) (`DankeSchemaRetriever` 클래스), 알고리즘 본체는 [`dictionary.py`](../danke_kg/dictionary.py)와 [`graph.py`](../danke_kg/graph.py)에 있습니다.

**생성자 파라미터**

```python
DankeSchemaRetriever(
    knowledge_schema: KnowledgeSchema,   # <db_id>_knowledge_schema.json 로드 결과
    mapping: RDBMapping,                 # <db_id>_mapping.json 로드 결과
    dictionary: Dictionary | None = None,# 없으면 build_dictionary(knowledge_schema)로 즉석 생성
    fuzzy_cutoff: float = 0.72,          # difflib 유사도 임계값
    use_fuzzy: bool = True,              # 이번 세션에 False→True로 변경 (아래 이유 참고)
    max_ngram: int = 3,                  # 키워드 후보 n-gram 최대 길이
    expand_with_steiner: bool = True,    # Steiner tree로 다리 테이블 확장 여부
)
```
`run_multiagent_experiment.py`/CLI에는 이 파라미터들이 노출돼 있지 않고 전부 기본값으로 생성됩니다(`DankeAugmentedSchemaLinker.__init__`의 `retriever_kwargs`로만 오버라이드 가능).

**1단계 — 사전(Dictionary) 준비**: `dictionary or build_dictionary(knowledge_schema)`. `build_dictionary()`([dictionary.py:99](../danke_kg/dictionary.py#L99))는 knowledge schema에서 다음 항목들을 `DictionaryEntry(entry_type, key, target_class, target_property, value)` 행으로 펼칩니다.
- `class` 엔트리: 각 클래스의 `primary_label` + `synonyms`마다 하나씩, `target_class`=클래스 이름
- `property` 엔트리: 각 datatype property의 `primary_label` + `synonyms`마다 하나씩, `target_class`=그 속성이 속한 클래스
- `value` 엔트리: `indexed=True`인 속성의 `value_synonyms`(값→동의어 매핑)마다 값 자체와 그 동의어들
- (선택) sqlite 파일이 주어지면 `value_synonyms`가 없는 indexed 컬럼에 대해 실제 distinct 값을 최대 `max_domain_values`개 샘플링해 `value` 엔트리로 추가(`_add_sampled_values`) — 이 브리지에서는 dictionary가 `<db_id>_knowledge_schema.json` 하나로만 만들어지므로 이 값-샘플링 경로는 타지 않고, KG 빌드 시점(`build_ontology.py`)에 이미 반영된 사전을 그대로 씁니다.
- **`--no-synonyms`로 빌드한 KG는 `synonyms` 리스트가 빈 배열**이라, class/property 엔트리 수가 synonym 포함 빌드보다 훨씬 적습니다 — 사전 크기가 작을수록 다음 매칭 단계에서 놓치는 키워드가 늘어나는 것이 이번 100문항 실험에서 recall이 낮게 나온 핵심 원인입니다.
- 내부적으로 `key.casefold()`를 해시 인덱스로 써서(`Dictionary._index: dict[str, list[int]]`) exact lookup은 O(1), 동일 키에 동일 엔트리가 중복 추가되는 것은 자동으로 걸러집니다(`Dictionary.add`).

**2단계 — 키워드 후보 생성**: `_candidate_phrases(query)`([multiagent_bridge.py:89](../danke_kg/multiagent_bridge.py#L89)).
- `WORD_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")`로 질문에서 단어를 추출(조사/구두점 제거).
- 길이 1~`max_ngram`(기본 3)짜리 슬라이딩 윈도우 어절 조합을 전부 만들고 `" ".join(...)`으로 이어붙임. 2글자 미만 조각은 버림.
- 예: `"각 대륙은 몇 개의 국가를 가지고 있나요"` → 단어 `["각","대륙은","몇","개의","국가를","가지고","있나요"]` → 1-gram(`"대륙은"`, `"국가를"`, ...), 2-gram(`"각 대륙은"`, `"대륙은 몇"`, ...), 3-gram까지 생성. `dict.fromkeys(...)`로 중복 제거.
- DANKE 논문(Section 5.2)은 이 키워드 추출을 LLM 호출로 하지만, 여기서는 **사전 조회가 애초에 빠른 exact/fuzzy 매칭이라 n-gram 슬라이딩 윈도우로 대체**했습니다(LLM 왕복 비용을 아끼는 의도적 단순화).

**3단계 — Matching Discovery**: `matching_service.match(keywords)` → 각 키워드마다 `MatchingDiscoveryService.match()`([dictionary.py:200](../danke_kg/dictionary.py#L200))가 실행됩니다.
1. `dictionary.lookup_exact(keyword)` — casefold 후 해시 인덱스 조회. 하나라도 있으면 `MatchResult(fuzzy=False)`로 확정, 아래 fuzzy 단계는 건너뜀.
2. exact가 없으면 `dictionary.lookup_fuzzy(keyword, cutoff=fuzzy_cutoff)` — `difflib.get_close_matches`로 사전의 모든 키와 유사도 비교(`SequenceMatcher` 비율), `cutoff`(기본 0.72) 이상인 것 중 상위 3개 키를 채택.
- `use_fuzzy=False`이면 `DankeSchemaRetriever.retrieve()`에서 이 fuzzy 결과를 아예 버립니다(`matched = [... if self.use_fuzzy or not item.fuzzy]`).
- **이번 세션에 `use_fuzzy` 기본값을 `False`→`True`로 바꾼 이유**: 한국어 질문은 "나이를", "가수는"처럼 명사에 조사가 붙어 사전 키("나이", "가수")와 정확히 일치하지 않는 경우가 많은데, `SequenceMatcher` 유사도가 대략 0.8 안팎으로 나와 fuzzy 매칭이 이런 조사 변형을 잡아줍니다. exact-only였다면 recall이 이번 실험 수치보다 더 낮았을 것입니다.

**4단계 — 매칭 클래스 추출**: `matching_service.matched_classes(matched)` — 매칭된 `MatchResult`들의 `entries[].target_class`를 중복 제거해 나열. (매칭된 게 `class`/`property`/`value` 엔트리 어느 쪽이든, 결국 그 값이 속한 **클래스(테이블)** 하나로 귀결됩니다.)

**5단계 — Steiner tree 확장 (Matching Optimization, 논문 Section 3.2)**: 매칭된 클래스가 2개 이상이면 `steiner_forest(graph, matched_classes)`([graph.py:140](../danke_kg/graph.py#L140))를 호출합니다.
- `graph`는 `build_knowledge_graph(knowledge_schema)`로 미리 만들어둔 `KnowledgeGraph` — 노드는 클래스(테이블), 간선은 object property(선언된 FK 또는 AI Hub의 검증된 inferred FK)이고 간선 가중치는 `ObjectProperty.weight`(선언 FK=1.0, inferred FK=1.5, "다음에 시도해볼 만한 것"과 무관하게 이미 반영됨).
- 알고리즘은 **Kou-Markowsky-Berman 2-근사**: (a) 매칭된 클래스(terminal) 쌍마다 Dijkstra 최단경로 계산 → (b) 그 거리들로 만든 완전그래프(metric closure) 위에서 Kruskal MST → (c) MST에 뽑힌 terminal 쌍들의 실제 최단경로를 원래 그래프 간선으로 펼침 → (d) 펼친 간선들 위에서 다시 한번 MST를 돌려 중복 경로를 제거.
- terminal들이 그래프 상에서 서로 다른 연결요소(component)에 있으면(=조인 경로가 아예 없으면) 컴포넌트별로 트리를 하나씩 만들어 **숲(forest)**을 반환 — 그래서 함수 이름이 `steiner_forest`입니다.
- 결과 트리들의 `nodes`(경유한 클래스 전체 — 질문에 직접 언급되지 않았어도 조인에 필요한 "다리 테이블" 포함)를 전부 합쳐 최종 후보 클래스 집합에 더합니다.
- 매칭이 1개뿐이면 트리 확장 없이 그 클래스 하나만 사용, 0개면 다음 6단계 폴백으로 넘어갑니다.

**6단계 — 폴백**: 매칭된 클래스가 하나도 없으면(`classes`가 빈 집합) KG의 전체 클래스로 대체합니다(`classes = set(knowledge_schema.classes)`) — 임베딩 리트리버가 점수와 무관하게 항상 top-k를 반환하는 것과 대칭되는 안전장치로, "후보가 아예 없어서 실패"하는 상황은 막지만 그 대가로 precision이 크게 떨어집니다.

**7단계 — 역매핑(mapping)**: 최종 클래스 집합의 각 이름을 `mapping.table_for(class_name)`으로 실제 테이블 이름으로, 그 클래스에 속한 모든 datatype property를 `schema.properties_of(class_name)` → `mapping.column_for(property_name)`으로 `"table.column"` 문자열로 변환합니다. 이 시점부터는 KG 내부 이름이 아니라 원본 스키마의 실제 식별자만 남습니다.

**출력**: `RetrievedSchema(tables=[...], columns=["table.column", ...], query=원본 쿼리, mode="danke_kg")`.

**성능 특성**: `retrieve()` 자체는 네트워크 호출이 전혀 없는 순수 사전 조회 + 그래프 탐색이라 밀리초 단위로 끝납니다. 100문항 실험 로그에 보이는 7~130초의 소요 시간은 리트리버가 아니라 같은 질문 안에서 함께 실행되는 manager(decompose/route)·value_linker·join_linker의 LLM 호출 지연이 대부분입니다 — `DankeSchemaRetriever.retrieve()`는 value_linker 단계에서 rescue query당 1회씩(최대 4회) 추가로 불려도 총 소요 시간에 미치는 영향은 무시할 만한 수준입니다.

**알려진 한계** (이미 실험으로 확인된 것들):
- **사전 커버리지 의존**: exact/fuzzy 매칭 모두 결국 사전에 등록된 키(주로 `primary_label`)에 의존하므로, `--no-synonyms` 빌드에서 특정 테이블의 `primary_label`이 영어로 남아 있으면(§"실제로 발견한 문제" 참고) 그 테이블을 가리키는 어떤 한국어 키워드도 절대 매칭되지 않습니다 — fuzzy 매칭도 언어가 다르면 유사도 자체가 낮아 구제되지 않습니다(concert_singer `singer` 테이블 사례).
- **Steiner 확장은 그래프 연결성에 의존**: 매칭된 클래스들 사이에 선언되거나(FK) 추론된 object property 경로가 없으면 다리 테이블을 못 찾습니다. pets_1에서 `Has_Pet`처럼 질문에 명시적으로 언급되지 않고 매칭도 안 되는 다리 테이블이 최종 후보에서 누락되는 경우가 여기 해당합니다.
- **폴백의 이분법**: "부분 매칭"과 "매칭 0건 전체 폴백" 사이의 중간이 없어서, 매칭이 하나라도 있으면 그 좁은 집합 + Steiner 확장만 쓰고, 하나도 없으면 갑자기 전체 스키마로 튑니다. 임베딩처럼 "점수가 낮아도 top-k만큼은 항상 채운다"는 완충 장치가 없습니다.

### 3. Schema linker (기본 `schema_linker_mode="embedding_only"`이면 LLM 스킵)

```
schema_output = embedding_only_schema_output(retrieved)
         = {"source": "embedding_only", "selected_tables": retrieved.tables,
            "selected_columns": retrieved.columns, "roles": [], "unresolved": []}
```
- **Input**: 2단계의 `retrieved` (DANKE 결과) 그대로.
- **Output**: `retrieved.tables`/`retrieved.columns`를 **그대로 복사**한 dict. LLM 재검증이 없으므로, **2단계 DANKE 매칭 결과가 곧 최종 후보**가 됩니다 — 이게 앞 절 "API 변경 대응"에서 설명한, retriever precision이 직접 최종 품질을 좌우하는 이유입니다.
- (`schema_linker_mode="qwen"`이면 대신 `SemanticSchemaLinkerAgent.link(question, decomposition, retrieved, schema)`가 LLM으로 retrieved 후보를 재선별하지만, 이 실험에서는 기본값을 그대로 사용했으므로 이 경로는 타지 않았습니다.)

### 4. Manager: routing (KG 관여 없음)

```
routing = manager.route(question, decomposition, retrieved, schema_output)
        = {"run_value_linker": bool, "value_task": str, "run_join_linker": bool, "join_task": str}
```

### 5. Value linker (조건부, ★ 리트리버가 여러 번 재호출됨)

`routing["run_value_linker"]`가 참일 때만 실행:
- `_value_rescue_queries(...)` → decomposition의 filter별로 필터 특화 쿼리 문자열을 최대 4개 생성
- 그 각각에 대해 **`self.retriever.retrieve(query, schema)`를 다시 호출** — 즉 `DankeSchemaRetriever`가 질문 하나당 최소 1번(2단계) + rescue query 개수만큼 추가로 호출될 수 있음
- `_merge_retrieved_schemas(initial, rescues)` → 모든 호출 결과의 tables/columns 합집합인 `RetrievedSchema(mode="value_rescue_union")`
- **Input** to `value_linker.link()`: 질문, `value_task`, `decomposition`, `schema_output`, 병합된 `RetrievedSchema`, `schema`, `database_path`
- **Output** `value_output`: `{"selected_tables": [...], "selected_columns": [...], "filters": [{"column":..., "operator":..., "value":...}], "unresolved": [...]}` — 실제 sqlite에 쿼리를 날려 필터 값을 검증(grounding)한 결과. **KG의 `value_synonyms`는 이 단계에 직접 연결돼 있지 않음** — value_linker는 자체적으로 DB 값을 조회함(§아래 "다음에 시도해볼 만한 것" 참고, 아직 미구현).

### 6. Join linker (조건부, KG 관여 없음 — 미구현 지점)

`routing["run_join_linker"]`가 참일 때: `join_linker.link(question, join_task, schema, semantic_tables, semantic_columns, decomposition, schema_linker_mode)` → `join_output = {"tables":[...], "columns":[...], "joins":[{"left":..., "right":...}], "unresolved":[...]}`. **KG가 이미 갖고 있는 조인 정보(Steiner tree 엣지, 즉 FK 기반 object property)는 이 단계에 전달되지 않고**, LLM이 원본 스키마의 FK만 보고 매번 새로 판단합니다 — 문서 하단 "다음에 시도해볼 만한 것"에 있는 개선 후보입니다.

### 7. 결정적 집계 (KG/LLM 관여 없음, 순수 코드)

```
final = aggregate_specialist_outputs(schema, schema_output, value_output, join_output)
```
- **Input**: 3, 5, 6단계 출력 3개
- **Output**: `{"selected_tables": union(...), "selected_columns": union(...), "column_roles": {...}, "grounded_filters": [...], "joins": [...], "unresolved": [...]}` — `validate_final_package`가 원본 스키마에 실제 존재하는 테이블/컬럼만 남기고 나머지는 버림(환각 방지).

### 8. 최종 반환값

```python
LinkingResult(
    db_id=schema.db_id, question=question,
    tables=final["selected_tables"], columns=final["selected_columns"],
    column_roles=final["column_roles"], grounded_filters=final["grounded_filters"],
    joins=final["joins"], unresolved=final["unresolved"],
    retrieved_schema=retrieved.to_dict(),   # 2단계 DANKE 원본 출력이 그대로 보존됨
    workflow_trace=events,                  # 0~7단계 각 이벤트 로그
)
```
`run_multiagent_experiment.py`는 이 `LinkingResult.tables`/`.columns`를 `schema_agents.evaluation.evaluate(result, rel_schema, gold_sql)`에 넘겨 gold SQL에서 뽑은 정답 테이블/컬럼 집합과 비교해 recall/precision을 계산합니다(3~7단계는 KG/임베딩 어느 쪽이든 동일 코드 경로이므로, 최종 recall/precision 차이는 사실상 전부 **2단계 retriever 출력의 차이**에서 비롯됩니다).

## 실험: `dog_kennels`, Spider-Ko validation 질문 5개

```bash
cd ontology
python3 run_multiagent_experiment.py --db-id dog_kennels --limit 5 --compare-embedding
```

각 질문에 대해 `DankeAugmentedSchemaLinker`(DANKE 리트리버)와 원본 `AgenticMultiAgentSchemaLinker`(임베딩 리트리버)를 동일 질문으로 각각 실행해 최종 선택 테이블을 비교했습니다.

| 질문 | DANKE 선택 테이블 | 임베딩 선택 테이블 | 일치 |
|---|---|---|---|
| 소유자와 전문가가 모두 살고 있는 주는? | Owners, Professionals | Professionals, Owners | ✅ |
| (같은 질문의 변형) | Owners, Professionals | Professionals, Owners | ✅ |
| 치료를 받은 강아지들의 평균 나이는? | Dogs, Treatments | Dogs, Treatments | ✅ |
| (같은 질문의 변형) | Dogs, Treatments | Dogs, Treatments | ✅ |
| 인디애나 주 거주 또는 치료 3건 이상 전문가는? | Professionals, Treatments | Professionals, Treatments | ✅ |

**결과: 5/5 테이블 선택 일치**, 모두 gold SQL이 실제로 참조하는 테이블과 일치. 소요 시간도 비슷한 수준(질문당 18~31초, 두 방식 모두 매니저/schema_linker/route 등 공유 LLM 호출이 대부분을 차지하고 retriever 자체 비용은 상대적으로 작음). 원본 출력은 `ontology/output/multiagent_experiment.json`.

## DANKE 리트리버의 장단점 (이 실험 기준)

- **장점**
  - 임베딩 서버(`localhost:8001`)가 없어도 동작 — 사전(dictionary) 매칭 + 그래프 탐색만으로 후보를 만듦.
  - 매칭 근거가 사람이 읽을 수 있는 사전 항목(라벨/동의어)이라 왜 그 테이블이 뽑혔는지 설명 가능(임베딩 코사인 유사도보다 해석 가능성이 높음).
  - Steiner tree 확장 덕분에, 키워드가 직접 매칭되지 않은 "다리 테이블"(조인에 필요하지만 질문에 언급되지 않는 테이블)도 자동으로 후보에 포함됨.
- **단점 / 주의할 점**
  - 사전에 없는 표현(동의어로 등록되지 않은 용어)은 놓칠 수 있음 — 임베딩은 의미적으로 가까운 표현을 어느 정도 커버하지만 DANKE 매칭은 사전 커버리지에 의존적. 실전에서는 두 방식을 **합집합으로 병행**하는 것을 권장(이번 실험은 순수 비교를 위해 단독으로만 사용).
  - 사전 품질(LLM이 만든 동의어 목록)이 곧 recall의 상한선. `ontology/docs/report.md`에 정리한 것처럼, LLM이 만든 동의어라도 실제 값과 검증하는 단계가 없으면 틀릴 수 있음.

## API 변경 대응 (실제로 겪은 문제)

브리지를 처음 만든 시점 이후 `schema linking/multi-agent/schema_agents`가 (제가 손대지 않은 사이에) 바뀌어 있었습니다:

- `embedding_retriever.RetrievedSchema`에 `mode`/`top_k_tables`/`top_k_columns` 필드가 추가됨.
- `AgenticMultiAgentSchemaLinker`에 `schema_linker_mode` 설정이 추가되고, **기본값이 `"embedding_only"`** — 즉 기본 설정에서는 `SemanticSchemaLinkerAgent`(LLM 기반 재선별)를 아예 건너뛰고, retriever가 반환한 tables/columns를 그대로 최종 선택으로 씀 (`embedding_only_schema_output()`).
- `value_linker`가 실행될 때 "rescue query"로 `self.retriever.retrieve()`를 여러 번 더 호출하고, 그 결과들을 `initial.top_k_tables`를 참조해 병합함(`_merge_retrieved_schemas`).

처음 만든 `DankeRetrievedSchema`(duck-typed 클래스)에는 `top_k_tables`가 없어서 **모든 질문에서 `'DankeRetrievedSchema' object has no attribute 'top_k_tables'` 에러로 100% 실패**하는 걸 100문항 실험을 돌리다 발견했습니다. 고친 방법: duck-typing을 버리고 `DankeSchemaRetriever.retrieve()`가 **진짜 `schema_agents.embedding_retriever.RetrievedSchema`를 그대로 생성해서 반환**하도록 바꿨습니다 — 이렇게 하면 그 클래스에 필드가 더 추가되더라도 이 브리지가 다시 깨지지 않습니다.

**중요한 함의**: `schema_linker_mode` 기본값이 `embedding_only`이기 때문에, 지금은 retriever가 반환하는 후보가 LLM 재검증 없이 거의 그대로 최종 선택이 됩니다. 즉 **retriever의 precision이 예전보다 훨씬 더 직접적으로 최종 결과 품질에 반영**됩니다 — DANKE 리트리버가 사전에 아무것도 매칭 못 해 "전체 클래스 fallback"을 타면 recall은 지키지만 precision이 크게 떨어지고, 반대로 일부만 매칭되면(예: 아래 사례) 필요한 테이블을 놓칠 수 있습니다.

**실제 관측한 실패 사례** (`concert_singer`, `--no-synonyms`로 빌드한 KG):
> "모든 가수의 이름, 국가, 나이를 나이가 많은 순서부터 어린 순서까지 보여주세요."
> DANKE 선택: `['stadium']` (recall=0.0, 오답) — 임베딩 선택: `['singer','singer_in_concert','concert','stadium']` (recall=1.0)

원인은 [spider-ko-no-synonym.md](./spider-ko-no-synonym.md)에 정리한 라벨 언어 불일치 문제입니다: `singer` 테이블의 컬럼 라벨이 영어("name","country","age")로 남아 있어 한국어 키워드 "이름/국가/나이"가 `singer`에는 전혀 매칭되지 않고, 우연히 한국어 라벨("이름")을 가진 `stadium`에만 매칭되어 버렸습니다. synonym을 포함해 빌드했던 `dog_kennels`(모든 테이블이 일관되게 한국어 라벨)에서는 이런 실패가 없었다는 점에서, synonym 제외 결정과 이 실패가 직접 연결되어 있다고 판단합니다.

## 다음에 시도해볼 만한 것 (미구현)

- **하이브리드 리트리버**: `DankeSchemaRetriever`와 `EmbeddingSchemaRetriever`를 합집합으로 합치는 세 번째 클래스 (recall 보강).
- **`ValueLinkerAgent` 보강**: DANKE dictionary의 `value` 엔트리(예: "즉시"→"IMED")를 `ValueLinkerAgent` 호출 전에 먼저 조회해 이미 그라운딩된 필터를 제공 — LLM 호출/DB 왕복을 줄일 수 있음.
- **`JoinLinkerAgent` 보강**: Steiner tree 엣지(이미 검증된 FK 또는 AI Hub의 value-overlap 검증을 통과한 inferred join)를 후보 조인으로 먼저 제시하고 LLM은 확인/라벨링만 하도록.
