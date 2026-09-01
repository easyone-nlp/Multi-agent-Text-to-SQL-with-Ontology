# ontology/docs 안내

`./ontology`에 만든 DANKE 유사 온톨로지/지식그래프 구축 파이프라인(`danke_kg` 패키지)에 대한 문서입니다.

- [architecture.md](./architecture.md) — DANKE 논문의 각 절이 `danke_kg`의 어느 모듈에 대응하는지 정리한 설계 매핑표.
- [report.md](./report.md) — 이번 작업에서 결정한 사항, 실제로 검증하며 발견한 이슈(특히 AI Hub 데이터의 조인 오탐 문제), 알려진 한계와 남은 선택지를 정리한 보고서.
- [multiagent-integration.md](./multiagent-integration.md) — DANKE KG를 `schema linking/multi-agent` 파이프라인에 (파일 수정 없이) 연결한 브리지와, 임베딩 리트리버 대비 비교 실험 결과.
- [spider-ko-no-synonym.md](./spider-ko-no-synonym.md) — Spider-Ko(`concert_singer`/`pets_1`/`car_1`) synonym 제외 빌드, 그 과정에서 발견한 라벨 언어 불일치 이슈, validation.csv 앞 100건 성능 평가.
- [kg-construction-process.md](./kg-construction-process.md) — 지금까지 빌드한 KG(`dog_kennels`, `서울인구관`, Spider-Ko 전체 160개 db_id)의 실제 실행 커맨드, 단계별 실제 산출물(JSON) 예시, 파이프라인 시각화(mermaid), 라벨 품질 전수 조사 결과.
- [synonym-merge-results.md](./synonym-merge-results.md) — 사람이 채운 컬럼 단위 synonym 파일(`synonym_v2.json`)을 기존 KG에 병합(`apply_synonyms.py`)한 뒤 100문항으로 재평가한 결과 (recall 0.760→0.943), 남은 실패 원인 분석.

## 빠른 실행 방법

Spider 형식 스키마(`data/hugging face/Spider 1.0/...`) 기준:

```bash
cd ontology
python3 build_ontology.py --db-id dog_kennels --keywords 이름 나이 --show-trace
```

AI Hub NL2SQL 데이터(`data/ai hub/...`) 기준 — 같은 발행기관(source) 산하 테이블을 하나의 스키마로 묶어서 구축:

```bash
cd ontology
python3 build_ontology_aihub.py --list-sources                 # source 목록 확인
python3 build_ontology_aihub.py --source 서울인구관 --keywords "인구 수" "출생아 수"
```

두 스크립트 모두 `--heuristic-only`를 주면 LLM 호출 없이 직접매핑(direct mapping)만으로 빠르게 동작을 확인할 수 있습니다. LLM 보강을 쓰려면 `config.json`이 가리키는 OpenAI-호환 엔드포인트(기본 `http://localhost:8000/v1`, Qwen3-4B-Instruct)가 떠 있어야 합니다.

결과물은 `ontology/output/`에 `<slug>_knowledge_schema.json`, `<slug>_mapping.json`, `<slug>_dictionary.json`으로 저장되고, AI Hub 쪽은 여러 db_id의 sqlite를 하나로 합친 `<slug>_combined.sqlite`도 함께 생성됩니다.
