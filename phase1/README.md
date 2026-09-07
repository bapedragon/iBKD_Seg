# Phase 1 — 데이터셋별 frozen spatial probe

Phase 1 실험은 데이터셋별로 분리해 관리합니다.

| 폴더 | 데이터셋 | 상태 |
|---|---|---|
| [phase1_pet](phase1_pet/README.md) | Oxford-IIIT Pet | 분류·frozen probe·ALG 진단 완료, No-Go |
| [phase1_cub](phase1_cub/README.md) | CUB-200-2011 | 통합 smoke 통과, full v2 LOCK·guided 첫 실행 준비 |

두 폴더의 config, 실행 기록과 결과를 서로 섞지 않습니다. 여러 데이터셋에서
공유할 수 있는 Python 구현은 [`src/ibkd_seg/phase1/`](../src/ibkd_seg/phase1/)에
유지합니다.
