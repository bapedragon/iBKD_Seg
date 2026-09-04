# Phase 1 보고서

재현 가능한 작은 요약만 Git에 포함합니다. 전체 feature cache, probe checkpoint,
원시 prediction과 run output은 각각 `phase1/results/raw/`와
`phase1/results/runs/`에 저장하며 Git에서 제외합니다.

로컬 실행 보고서는 `*.local.json` 이름으로 저장합니다.

- `phase1a_summary.json`: 전체 Phase 1A 실행의 추적 가능한 소형 정량 요약
- `phase1a_full.local.json`: epoch history와 seed별 원값을 포함한 전체 로컬 보고서
- `PHASE1A_QUALITATIVE.md`: 실제 Flowers 입력–pseudo-mask–예측 panel 설명과 미리보기
- `figures/phase1a/`: 고정 test ID 7개 x 3개 방법의 Git 보존용 PNG 21개
- `figures/phase1a/manifest.json`: panel의 checkpoint, probe 선택값, SHA-256

PNG는 결과를 다시 계산하기 위한 원시 prediction이 아니라 사람이 확인할 수 있도록
선별한 소형 증거입니다. 전체 run 폴더와 probe checkpoint는 계속 Git에서 제외합니다.
