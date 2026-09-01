# synonym_v2.json 병합 + 재실험 결과

[spider-ko-no-synonym.md](./spider-ko-no-synonym.md)에서 synonym을 뺀 채 빌드하고 100문항으로 평가했을 때, recall 손실의 대부분이 "synonym이 없어서"가 아니라 "일부 테이블 라벨이 아예 영어로 남아서"였다는 걸 확인했습니다. 이후 사람이 직접 채운 컬럼 단위 synonym 파일(`ontology/synonym_v2.json`)을 받아 기존 KG에 병합하고, 같은 100문항으로 다시 평가했습니다.

## `synonym_v2.json` 구조

```json
{
  "db_id": "academic",
  "table": "author",
  "column": "aid",
  "readable_name": "aid",
  "data_type": "NUMERIC",
  "description": "저자의 고유 식별 번호.",
  "synonyms": ["aid", "아이디", "고유번호", "아이디값", "아이디 번호", "아이디코드"]
}
```

- 총 4,256개 엔트리, **컬럼(datatype property) 단위**로만 존재 — 테이블(class) 단위 엔트리는 없음.
- db_id 166개 전부가 정확히 `output/spider_ko/`에 있는 166개 db_id와 일치.
- `(table, column)` 키로 매칭했을 때 **4256/4256 전부 매칭 성공**(0건 미매칭) — Spider(-Ko) 원본 컬럼명과 완전히 일치하는 값들이었습니다.

## 병합 방법 (`ontology/apply_synonyms.py`)

```bash
cd ontology
python3 apply_synonyms.py   # 기본: synonym_v2.json 전체를 output/spider_ko/*.json에 병합
```

- 각 엔트리를 `datatype_properties[].source_table`/`source_column`으로 찾아 `synonyms` 리스트에 **추가**(기존 값 유지, 중복 제거) — `primary_label`은 건드리지 않음.
- 병합 후 `<db_id>_dictionary.json`도 새 synonym을 반영해 다시 빌드(`build_dictionary()` 재호출).
- **class(테이블) 단위 `synonyms`는 그대로 빈 배열로 남음** — 파일에 테이블 단위 엔트리가 없기 때문. 예: `academic_knowledge_schema.json`을 열어봐도 `classes[].synonyms`는 전부 `[]`.

결과: 166개 db_id 전체에서 property 4,199개에 synonym 26,638개 추가 (일부 87개 property는 원본 엔트리의 synonyms 리스트 자체가 비어 있어서 추가된 게 없음).

## 100문항 재평가: 병합 전후 비교

| 지표 | 병합 전 (no-synonym) | 병합 후 (synonym_v2 적용) | 변화 |
|---|---:|---:|---:|
| table recall (평균) | 0.760 | **0.943** | +0.183 |
| table precision (평균) | 0.650 | **0.645** | -0.006 (거의 동일) |
| strict table recall율 | 0.690 | **0.920** | +0.230 |
| column recall (평균) | 0.772 | **0.946** | +0.174 |
| 임베딩과 테이블 선택 일치율 | 0.30 | **0.59** | +0.29 |

임베딩 리트리버는 이번에도 변함없이 recall/strict/column recall 전부 1.000(baseline)이고, precision은 0.479로 이전과 동일 — DANKE만 KG가 바뀌었으니 예상대로입니다. **precision은 거의 그대로 유지하면서 recall만 크게 끌어올렸다**는 게 이번 병합의 핵심 성과입니다 — 컬럼 synonym을 넉넉히 추가해도 사전이 과도하게 넓어져서 엉뚱한 테이블을 끌어오는 부작용은 거의 없었다는 뜻입니다.

### db_id별 breakdown (병합 후)

| db_id | recall | precision | strict | col_recall | 임베딩과 일치율 |
|---|---:|---:|---:|---:|---:|
| concert_singer | 0.967 | 0.517 | 0.956 | 0.953 | 0.733 |
| pets_1 | 0.972 | 0.798 | 0.952 | 0.988 | 0.619 |
| car_1 | 0.769 | 0.592 | 0.692 | 0.722 | 0.000 |

concert_singer는 이전 최악(0.681)에서 0.967로 가장 크게 개선됐습니다 — `singer`/`singer_in_concert`의 컬럼 synonym이 채워지면서 "이름/국가/나이" 매칭 실패가 대부분 해소됐습니다. car_1은 여전히 가장 낮고, 임베딩과의 테이블 선택 일치율이 0%(13문항 전부 불일치)까지 떨어졌습니다 — 아래 원인 참고.

## 남은 실패 8건 분석

**1) car_1의 `continents` 테이블 — synonym이 아니라 번역 자체의 문제**

```
"대륙은 몇 개인가요?" → DANKE=['countries'] gold=['continents']  recall=0.0
```
`continents` 테이블의 `primary_label`이 "대륙"이 아니라 **"지역"**으로 LLM이 잘못 번역했고, `synonym_v2.json`이 채운 synonym도 `["지역명","지방","지역 이름",...]`뿐이라 "대륙"이라는 단어 자체가 사전 어디에도 없습니다. 이건 synonym 커버리지 문제가 아니라 **애초에 "continent"를 "지역"으로 잘못 번역한 것**이 원인이라, synonym을 아무리 추가해도 "대륙"이라는 정확한 한국어 단어가 실제로 존재하지 않으면 못 잡습니다. car_1의 나머지 실패 3건(`[88]`,`[89]`,`[90]`)도 전부 같은 원인입니다.

**2) concert_singer/pets_1 — 다리 테이블(bridge table)이 질문에 명시적으로 언급되지 않는 경우**

```
"콘서트가 한 번도 열리지 않은 경기장 이름을 보여주세요" → DANKE=['singer_in_concert'] gold=['concert','stadium']  recall=0.0
"반려동물이 있는 학생들은 각각 몇 마리..." → DANKE=['Has_Pet','Pets'] gold=['Has_Pet','Student']  recall=0.5
```
"경기장"/"학생"이라는 단어가 질문에 있지만 Steiner tree 확장이 다른 클래스를 앵커로 잡아 엉뚱한 방향으로 다리를 놓은 경우 — synonym 커버리지보다는 매칭된 클래스가 여러 개일 때 Steiner tree가 어느 쌍을 최단경로로 잇는지의 문제라, 이번 병합으로는 해결되지 않는 범주입니다.

**3) class(테이블) 단위 synonym 부재는 이번 8건 실패에는 직접 영향 없음**

confusingly `synonym_v2.json`에 테이블 단위 synonym이 없다는 점 자체는, 이번 100문항에서 남은 8건의 직접 원인은 아니었습니다(전부 컬럼/번역/Steiner tree 문제). 다만 "가수" 같은 순수 테이블-지시 표현만 있고 관련 컬럼 키워드가 전혀 없는 질문이 나오면 여전히 리스크가 남아있습니다.

## 결론

컬럼 단위 synonym 병합만으로 DANKE recall이 0.760→0.943로 크게 개선됐고 precision(0.650→0.645)은 거의 그대로 유지됐습니다. 남은 격차(0.943 vs 임베딩의 1.000)는 대부분 (a) `continents`처럼 LLM이 애초에 잘못 번역한 테이블, (b) Steiner tree가 다리 테이블을 잘못 고르는 소수 케이스로 좁혀졌습니다 — 둘 다 synonym 추가로는 못 고치는 범주이므로, 다음 개선은 번역 재검수(`primary_label` 자체 교정)나 Steiner tree 앵커 선택 로직 쪽이 더 유효할 것으로 보입니다.

원본 결과: [`output/multiagent_experiment_100_with_synonyms.json`](../output/multiagent_experiment_100_with_synonyms.json) (병합 전: [`output/multiagent_experiment_100.json`](../output/multiagent_experiment_100.json))
