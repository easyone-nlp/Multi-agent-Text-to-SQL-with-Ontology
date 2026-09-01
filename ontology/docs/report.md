# 작업 보고: DANKE 유사 온톨로지/지식그래프 구축

## 1. 무엇을 만들었나

`./ontology/danke_kg`에 DANKE(3장) 정의를 따르는 지식 스키마 모델과, 기존 `schema linking/multi-agent` 파이프라인의 에이전트 패턴을 재사용한 구축 파이프라인을 작성했습니다. 두 가지 입력 소스를 지원합니다.

1. **Spider 형식** (`build_ontology.py`) — `data/hugging face/Spider 1.0`처럼 FK가 선언된 다중 테이블 DB.
2. **AI Hub NL2SQL 데이터** (`build_ontology_aihub.py`) — `data/ai hub`. 이 부분은 사용자가 명시적으로 요청해 추가로 확장한 범위입니다.

세부 매핑은 [architecture.md](./architecture.md) 참고.

## 2. 사용자와 함께 결정한 사항

- **AI Hub 구축 범위**: 검증셋만 640개 이상의 개별 스키마가 있고 대부분 FK 없는 단일 테이블이라, 전체를 하나로 묶거나 개별 db_id 단위로 가는 대신 **`source`(발행기관) 단위로 그룹화**하기로 사용자가 직접 선택함 (AskUserQuestion 응답: "소스(발행기관) 단위 그룹화"). 이 선택에 따라 `build_ontology_aihub.py --source <이름>`이 같은 발행기관의 테이블들을 하나의 관계형 스키마로 합칩니다.
- `--list-sources`로 사용 가능한 source와 테이블 수를 먼저 확인하도록 CLI를 설계함 (예: `서울인구관`=5개 테이블, `서울특별시`=70~112개로 도메인이 섞여 있어 비추천).

## 3. 실제 데이터로 검증하며 발견한 문제와 대응

### 3.1 AI Hub는 FK가 사실상 전혀 없음

검증셋 17개 `_db_annotation.json` 파일(640개 db_id)을 스캔한 결과 FK가 선언된 db_id는 극소수(예: `publicdata_transportation`에 2개, `publicdata_climate`에 3개)였고 나머지는 전부 `foreign_keys: []`였습니다. 논문 5.4절이 언급하는 "FK가 없으면 object property를 직접 정의해야 한다"는 상황 그 자체입니다.

**대응**: `danke_kg/aihub.py`의 `infer_join_columns()`가 `_CD`/`_CODE`/`_ID`/`_NO`/`_KEY`로 끝나는 컬럼명이 2개 이상 테이블에서 동일하게 나타나면 후보 조인으로 제안합니다.

### 3.2 (중요) 컬럼명이 같아도 실제로는 다른 코드 체계인 경우가 있음 — 실측으로 발견

`서울인구관` source(5개 테이블)를 이름 매칭만으로 조인 추론했을 때 후보 16개가 나왔는데, 실제로 병합 sqlite에 대해 뷰를 실행해보니 **행이 0건**이었습니다. 원인을 추적한 결과:

```
POP_010.ADMDONG_CD 샘플: 1113065, 1113066, ...   (7자리)
POP_030.ADMDONG_CD 샘플: 11, 11110, 11200, ...   (2~5자리)
POP_003.ADMDONG_CD 샘플: 1168075000, 1171051000, ... (10자리, 법정동 코드)
POP_021.ADMDONG_CD 샘플: 11120, 11130, ...       (5자리)
```

같은 컬럼명("행정동 코드")이지만 테이블마다 행정동/법정동 코드의 **자릿수(집계 단위)가 다릅니다**. 이름만으로 조인을 추론하면 문법적으로는 유효하지만 의미적으로 틀린(또는 텅 빈) JOIN이 만들어질 수 있다는 걸 실측으로 확인했습니다.

**대응**: `validate_join_overlap()`을 추가해, 이름 매칭으로 나온 후보 조인을 물리적으로 병합한 sqlite에서 **실제 값 겹침 비율**로 재검증하도록 파이프라인을 수정했습니다 (컬럼 A/B의 distinct 값 집합 교집합 ÷ 작은 쪽 크기, 기본 임계값 5%, `--min-join-overlap`으로 조정 가능). `서울인구관` 재실행 결과:

- 후보 16개 → 실제 값 겹침 검증 통과 9개, 기각 7개 (모두 `ADMDONG_CD` 자릿수 불일치 쌍)
- 검증 통과한 조인으로 만든 뷰를 실제로 실행해 정상적으로 행이 반환됨을 확인함.

**남은 리스크**: 이 검증은 "값이 겹치는가"만 보므로, 우연히 값 범위가 겹치는 무관한 코드 컬럼(예: 둘 다 1~5 사이 값을 갖는 서로 다른 의미의 코드)을 완전히 걸러내지는 못합니다. 실제 서비스에 쓰려면 이 위에 `RelationshipOntologyAgent`가 이미 만들어주는 한국어 관계 라벨을 사람이 한 번 검수하거나, 값 겹침 비율을 더 높게(`--min-join-overlap`) 잡는 걸 권장합니다.

### 3.3 AI Hub는 db_id당 별도 sqlite 파일 — 물리적으로 병합해야 실행 가능

DANKE는 하나의 관계형 DB D를 전제하지만, AI Hub는 같은 source 산하 테이블이라도 db_id마다 **별도의 sqlite 파일**입니다. `combine_source()`로 만든 통합 스키마는 개념적으로만 하나이므로, 실제로 조인 가능한 SQL을 만들려면 물리적 병합이 필요했습니다.

**대응**: `build_combined_sqlite()`가 각 db_id의 sqlite를 `ATTACH DATABASE`로 붙여 `<db_id>__<table>`이라는 새 이름으로 복사해 하나의 `<source>_combined.sqlite`를 만듭니다. 사전(dictionary)의 실제 값 샘플링과 뷰 실행 검증 모두 이 병합 DB를 사용합니다.

## 4. 검증 방법 (실측)

- `dog_kennels`(Spider): 2-클래스, 3-클래스(다중 홉) Steiner tree로 생성한 `CREATE VIEW` SQL을 실제 sqlite에 실행해 정상적으로 행이 반환됨을 확인.
- `서울인구관`(AI Hub): 값-겹침 검증 전/후로 생성된 뷰를 각각 실행 — 검증 전(이름 매칭만)에는 0행, 검증 후에는 정상 행이 반환됨을 실측으로 대조 확인.
- 로컬 vLLM(Qwen3-4B-Instruct-2507, `localhost:8000`)이 실제로 떠 있어, heuristic 폴백뿐 아니라 실제 LLM 보강 경로도 두 데이터셋 모두에서 end-to-end로 실행해 확인함.

## 5. 알려진 한계 / 다음에 결정이 필요한 것

- **Training split 미실행**: `build_ontology_aihub.py --split Training`도 지원하지만 이번에는 Validation만 확인했습니다.
- **조인 추론은 컬럼명 접미사 휴리스틱 + 값 겹침 검증**이며, 완전한 시맨틱 검증(예: LLM이 두 코드 체계가 정말 같은 개념인지 판단)은 하지 않습니다. `RelationshipOntologyAgent`가 관계 라벨/weight는 달아주지만, "이 조인을 채택할지"는 현재 순전히 값 겹침 비율로만 결정됩니다.
- **`min_overlap` 기본값(5%)은 임의값**입니다. `서울인구관` 사례에서는 잘 걸러졌지만 다른 source에서는 재튜닝이 필요할 수 있습니다.
- **LLM 서버 의존성**: `config.json`의 `provider: openai_compatible`이 기본값이라, LLM 서버가 꺼져 있으면 테이블당 호출이 실패하며 시간이 걸릴 수 있습니다(`--heuristic-only`로 우회 가능).
- 이번에 만든 산출물(`ontology/output/*.json`, `*_combined.sqlite`)은 스모크 테스트 결과물이라 필요 없으면 삭제해도 됩니다.
