# 설계 매핑: DANKE 논문 ↔ `danke_kg` 코드

논문: *A Text-to-SQL strategy based on large language models and knowledge graphs for real-world databases* (Nascimento et al., Data & Knowledge Engineering 2026), 특히 3장(DANKE)과 5.4절(전략 논의)을 기준으로 삼았습니다.

| 논문 개념 (절) | 정의 | 코드 대응 |
|---|---|---|
| 지식 스키마 S^K=(C,P,O) (3.1) | 클래스 c=(r_c,l_c,L_c), datatype property p=(r_p,l_p,L_p,c_p,T_p), object property o=(r_o,l_o,L_o,c_o¹,c_o²) | [`danke_kg/models.py`](../danke_kg/models.py) — `KnowledgeClass`, `DatatypeProperty`(+`indexed`,`value_synonyms`), `ObjectProperty`(+`weight`,`inferred`) |
| 매핑 μ: S^K→S^R (3.1) | 클래스→테이블, 속성→컬럼 매핑 | [`danke_kg/mapping.py`](../danke_kg/mapping.py) — `RDBMapping`, `direct_mapping()` |
| 유도된 지식그래프 G^K=(V,E) (3.1) | V=C∪P, object property는 두 클래스를 직접 잇는 간선 | [`danke_kg/graph.py`](../danke_kg/graph.py) — `KnowledgeGraph`, `build_knowledge_graph()` |
| Storage Module / dictionary (3.1) | 클래스·속성 라벨/동의어(metadata entries) + indexed property 값(data entries) | [`danke_kg/dictionary.py`](../danke_kg/dictionary.py) — `Dictionary`, `build_dictionary()` |
| Matching Discovery Module / Service (3.2) | 키워드→사전 항목 exact→fuzzy 매칭, 실패 시 `noMatch` | `danke_kg/dictionary.py` — `MatchingDiscoveryService` |
| Matching Optimization Module (3.2) | 매칭된 클래스들을 잇는 최소 Steiner tree(또는 forest) | `danke_kg/graph.py` — `steiner_forest()` (Kou–Markowsky–Berman 근사) |
| View Synthesis Service (3.2, 4.1) | Steiner tree를 관계형 스키마 위의 뷰 V로 컴파일, 컬럼명은 `c_p` 컨벤션 | [`danke_kg/view_synthesis.py`](../danke_kg/view_synthesis.py) — `ViewSynthesisService` |
| "quick way": 직접 매핑 + 사전 보강 (5.4) | FK 없으면 object property를 직접 정의해야 함 | `danke_kg/aihub.py` — `infer_join_columns()` + `validate_join_overlap()` (AI Hub처럼 FK가 없는 데이터용) |
| Example Generation / 데이터 시맨틱 노출 (4.2) | 클래스·속성 설명/동의어를 LLM으로 보강 | [`danke_kg/agents.py`](../danke_kg/agents.py) + [`danke_kg/orchestrator.py`](../danke_kg/orchestrator.py) — `TableOntologyAgent`, `RelationshipOntologyAgent`, `OntologyBuilder` |

## 기존 multi-agent 파이프라인에서 재사용한 패턴

`schema linking/multi-agent/schema_agents/agentic_agents.py`의 구조를 그대로 따랐습니다:

- `StructuredQwenAgent` → `StructuredLLMAgent`(`agents.py`): system/user 프롬프트로 LLM을 호출하고, JSON 파싱 실패 시 재시도하는 베이스 클래스.
- 매니저-스페셜리스트 오케스트레이션 → `OntologyBuilder.build()`: 먼저 결정론적 직접매핑으로 뼈대를 만들고(스키마 링킹의 `deterministic_aggregator`에 해당), 이후 테이블당 1회(`TableOntologyAgent`)·FK 전체 1회(`RelationshipOntologyAgent`) LLM을 호출해 보강.
- `validate_schema_selection`/`validate_join_output` 같은 "LLM 출력은 반드시 실제 스키마 요소로 검증 후 반영" 패턴 → `agents.py`의 `validate_table_enrichment`/`validate_relationship_enrichment` (LLM이 존재하지 않는 컬럼을 지어내도 무시됨).
- `OpenAICompatibleChatModel`(vLLM/Qwen 엔드포인트) → `danke_kg/model_client.py`에 그대로 재구현(경로에 공백이 있는 `schema linking/` 디렉터리를 import하지 않도록 `ontology/`에 독립 사본을 둠).

## 두 개의 CLI

- `build_ontology.py`: Spider `tables.json` 형식(진짜 FK가 선언된 다중 테이블 DB, 예: `dog_kennels`) 대상.
- `build_ontology_aihub.py`: AI Hub NL2SQL 데이터(대부분 FK가 없는 단일 테이블 DB) 대상. `--source`로 같은 발행기관 산하 테이블들을 하나의 스키마로 묶고, 컬럼명 기반으로 조인을 추론한 뒤 실제 데이터로 검증합니다. 자세한 내용과 실제로 발견된 이슈는 [report.md](./report.md) 참고.
