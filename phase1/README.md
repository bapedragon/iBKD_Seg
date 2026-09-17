# Phase 1 — 데이터셋별 공간정보 실험

Phase 1 실험은 데이터셋별로 분리해 관리합니다.

| 폴더 | 데이터셋 | 상태 |
|---|---|---|
| [phase1_pet](phase1_pet/README.md) | Oxford-IIIT Pet | 분류·frozen probe·ALG 진단 완료, No-Go |
| [phase1_cub](phase1_cub/README.md) | CUB-200-2011 | 분류·frozen probe·직접 공간진단 완료; 결과 보존 단계 |
| [phase1_cub_Seg](phase1_cub_Seg/README.md) | CUB-200-2011 mask | 처음부터 segmentation으로 학습하는 별도 경로 |

완료된 분류 기반 실험은 본학습 결과·감사 manifest·재현용 본학습 진입점만 남기며, 선행
실행 점검 산출물은 보존하지 않습니다. 각 폴더의 config, 실행 기록과 결과를 서로
섞지 않습니다. 여러 데이터셋에서
공유할 수 있는 Python 구현은 [`src/ibkd_seg/phase1/`](../src/ibkd_seg/phase1/)에
유지합니다.

2026-09-17부터 CUB 후속은 분류 encoder를 먼저 만든 뒤 decoder를 교체하는 방식이
아니라, pixel mask를 처음부터 정답으로 사용하는 직접 segmentation 학습으로
분리합니다. 기존 `phase1_cub/` 결과는 이 방향 전환 이전의 진단 근거로 보존합니다.
