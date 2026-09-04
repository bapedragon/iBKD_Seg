# Phase 0.5 보고서

재현 가능한 작은 요약과 선별한 정성 panel만 Git에 포함합니다. 전체 feature cache,
probe checkpoint와 원시 prediction은 `phase0.5/results/`에 저장하며 Git에서
제외합니다. 전체 로컬 보고서는 `*.local.json` 이름으로 저장합니다.

- `summary.json`: 전체 실행의 추적 가능한 소형 정량 요약과 seed별 선택값
- `full.local.json`: epoch history와 seed별 원값을 포함한 전체 로컬 보고서
- `smoke.local.json`: smoke-test 전체 로컬 보고서
- `QUALITATIVE.md`: 실제 Flowers 입력–pseudo-mask–예측 설명과 미리보기
- `figures/flowers102/`: 고정 test ID 7개 x 3개 방법의 PNG 21개
- `figures/flowers102/manifest.json`: checkpoint, probe 선택값과 PNG SHA-256

PNG는 재계산용 원시 prediction이 아니라 사람이 확인할 수 있도록 선별한 소형
증거입니다. 전체 run 폴더와 probe checkpoint는 계속 Git에서 제외합니다.
