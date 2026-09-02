# Phase 0 보고서

감사 명령은 이 폴더에 machine-local JSON 보고서를 생성합니다. 로컬 경로와 실행
시각이 들어 있는 `.local.json` 파일은 Git에서 제외합니다.

재현에 필요한 안정적인 결과는 다음 파일에 보존합니다.

- `checkpoint_audit.summary.json`: 체크포인트 및 frozen-feature 계약 요약
- `dataset_audit.summary.json`: 공식 파일 구조 및 마스크 품질 요약
- `../DECISION.md`: 감사 결과에 따른 최종 연구 gate 결정

Flowers-102 공식 파일의 SHA-256은 `../../manifests/flowers102.json`에도 고정되어
있습니다.
