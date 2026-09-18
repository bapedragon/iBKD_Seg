# CUB 실험 계보 색인

`L0`라는 짧은 이름이 서로 다른 실행을 가리키지 않도록 이 문서의 ID를 사용합니다.
앞으로 문서·로그·이슈에서는 `L0`만 단독으로 쓰지 않습니다.

| 고정 ID | 의미 | 상태 | checkpoint 용도 | canonical 위치 |
|---|---|---|---|---|
| `main_l0_v3` | 최초 ResNet-50/224 주 실험의 강한 loader | guided 4방법×3 encoder seed 및 직접진단 완료·감사 | **주 결과 및 원인 분석 입력으로 사용 가능** | `reports/frozen_probe/`, `reports/direct_spatial/` |
| `loader_pilot_l0_seed1` | 이미지-loader Stage B에서 L1/L2와 맞춰 다시 돌린 L0 seed 1 | 완료 로그만 반영, archive/checkpoint 감사 대기 | 원인 분석 입력으로 사용 금지 | `image_loader_experiment/reports/full_v1_log_snapshot/` |
| `loader_pilot_l1_seed1` | 강한 광학 증강만 제거한 loader pilot | 완료 로그만 반영, archive/checkpoint 감사 대기 | loader 선택 근거 외 사용 금지 | `image_loader_experiment/reports/full_v1_log_snapshot/` |
| `loader_pilot_l2_seed1` | crop을 보수적으로 바꾼 loader pilot | Stage B에서 선택 | loader 선택 근거 외 사용 금지 | `image_loader_experiment/reports/full_v1_log_snapshot/` |
| `main_l0_mechanism_v3` | `main_l0_v3`에서 iBKD 레이어 연결 원인을 분리하는 사후 실험 | seed-1 classification-only 완료: learned-all 26.337%로 고정 연결 3조건보다 높음; 후속 확장 종료 | seed-1 사후 진단으로만 보존 | `mechanism_analysis/reports/main_l0_connection_classification_seed1_v3/` |
| `main_l0_repro_aa_v2` | main-L0 iBKD-0.25 seed 1의 제어 A/A 재현성 진단 | 300-epoch A/A 완료; 입력·RNG 통제 확인, test 24.761/25.510% | 성능 비교나 주 결과로 사용 금지 | `reproducibility/` |

## 이름 규칙

- 기존 3-seed 주 결과를 말할 때: `main-L0` 또는 `main_l0_v3`
- loader pilot에서 다시 학습한 L0를 말할 때: `pilot-L0` 또는
  `loader_pilot_l0_seed1`
- 원인 분석에서 기존 checkpoint를 입력으로 쓸 때는 source issue `727/730`과
  `main_l0_v3`를 함께 기록함

## 현재 질문과 입력

“iBKD의 분류 이득이 레이어 연결 학습에서 왔는가?”라는 원인 분석은
`main_l0_v3`의 현상에서 출발합니다. 따라서 기존 checkpoint 관찰 분석에는 issue
727의 batch-128 seed 1과 issue 730의 batch-128 seed 2·3만 사용합니다.
`loader_pilot_l0_seed1`은 이름은 L0지만 별도의 재학습이고 checkpoint archive도
아직 감사되지 않았으므로 섞지 않습니다. 제어 A/A와 단일 replay에서 issue 760의
`10.41%` 저성능이 다시 나타나지 않았고 matched-duration smoke도 통과했습니다.
공통 123-epoch guidance의 seed-1 연결 ablation에서는 learned-all이 세 고정 연결보다
높았습니다. 단일 seed classification-only 결과로만 보존하고 더 확장하지 않으며,
다음 CUB 실험은 pixel mask를 처음부터 사용하는 직접 segmentation 경로로 분리합니다.
