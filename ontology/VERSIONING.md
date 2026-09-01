# Ontology and synonym versioning

2026-07-28부터 ontology와 동의어 사전은 실험 결과와 독립적으로 명시적 버전을 가진다.

## 현재 동결 버전

- Ontology: `danke-spider-ko-v1.0.0-20260728`
- Synonyms: `v1.0.0`
- 연결된 기준선: `experiment_freezes/2026-07-28_baselines_v1`

## 규칙

1. `ontology/versions/<version>/`과 `ontology/synonym_versions/<version>/` 내부 파일은 공개 후 수정하지 않는다.
2. 오탈자·잘못된 동의어 수정도 기존 파일을 덮어쓰지 않고 새 버전으로 배포한다.
3. 동의어만 호환 가능한 수준으로 수정하면 patch를 올린다. 예: `v1.0.0 → v1.0.1`.
4. entry 추가/삭제 또는 의미가 달라지는 curated revision은 minor를 올린다. 예: `v1.0.0 → v1.1.0`.
5. JSON 구조나 매칭 의미가 호환되지 않게 바뀌면 major를 올린다.
6. ontology를 다시 생성하거나 동의어를 ontology artifact에 재적용하면 새 ontology version을 만든다.
7. 모든 실험 manifest에는 ontology version, synonym version, 두 checksum manifest의 SHA-256을 기록한다.
8. 작업용 `synonym_v2.json`과 `synonym_table.json`은 수정할 수 있지만, 실험 전에 반드시 새 version directory로 publish해야 한다.

## 새 동의어 버전 배포 체크리스트

1. 새 `ontology/synonym_versions/vX.Y.Z/` 디렉터리를 만든다.
2. column/table synonym 파일을 복사한다.
3. 파일별 SHA-256과 변경 이유를 `manifest.json`에 기록한다.
4. `ontology/synonym_versions/registry.json`의 `latest`와 versions 목록을 갱신한다.
5. ontology에 적용했다면 새 ontology snapshot도 만든다.
6. 새 실험 manifest에서 정확한 두 버전을 참조한다.
