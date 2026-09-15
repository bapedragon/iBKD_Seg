# CUB 실험 계보 색인

`L0`라는 짧은 이름이 서로 다른 실행을 가리키지 않도록 이 문서의 ID를 사용합니다.
앞으로 문서·로그·이슈에서는 `L0`만 단독으로 쓰지 않습니다.

| 고정 ID | 의미 | 상태 | checkpoint 용도 | canonical 위치 |
|---|---|---|---|---|
| `main_l0_v3` | 최초 ResNet-50/224 주 실험의 강한 loader | guided 4방법×3 encoder seed 및 직접진단 완료·감사 | **주 결과 및 원인 분석 입력으로 사용 가능** | `reports/frozen_probe/`, `reports/direct_spatial/` |
| `loader_pilot_l0_seed1` | 이미지-loader Stage B에서 L1/L2와 맞춰 다시 돌린 L0 seed 1 | 완료 로그만 반영, archive/checkpoint 감사 대기 | 원인 분석 입력으로 사용 금지 | `image_loader_experiment/reports/full_v1_log_snapshot/` |
| `loader_pilot_l1_seed1` | 강한 광학 증강만 제거한 loader pilot | 완료 로그만 반영, archive/checkpoint 감사 대기 | loader 선택 근거 외 사용 금지 | `image_loader_experiment/reports/full_v1_log_snapshot/` |
| `loader_pilot_l2_seed1` | crop을 보수적으로 바꾼 loader pilot | Stage B에서 선택 | loader 선택 근거 외 사용 금지 | `image_loader_experiment/reports/full_v1_log_snapshot/` |
| `l2_guided_preliminary_seed1` | 선택 L2에서 guided 4방법을 다시 학습한 예비실험 | 완료 로그 반영, archive/checkpoint 감사 대기 | 최종 확증 결과나 `main_l0_v3` 대체로 사용 금지 | `image_loader_experiment/reports/l2_guided_preliminary_full_seed1_log_snapshot_v1/` |
| `main_l0_mechanism_v1` | `main_l0_v3`에서 iBKD 레이어 연결 원인을 분리하는 사후 실험 | seed-1 실행에서 canonical 재현 실패·controller 기간 혼입으로 해석 보류 | 새 결과는 별도 사후 원인 분석으로만 사용 | `mechanism_analysis/` |
| `main_l0_repro_aa_v2` | main-L0 iBKD-0.25 seed 1의 제어 A/A 재현성 진단 | fail-closed v1이 deformable-conv 제약을 검출; v2 2-epoch smoke 준비 | 성능 비교나 주 결과로 사용 금지 | `reproducibility/` |

## 이름 규칙

- 기존 3-seed 주 결과를 말할 때: `main-L0` 또는 `main_l0_v3`
- loader pilot에서 다시 학습한 L0를 말할 때: `pilot-L0` 또는
  `loader_pilot_l0_seed1`
- L2 후속을 말할 때: `L2 preliminary`; `main-L0`와 합치지 않음
- 원인 분석에서 기존 checkpoint를 입력으로 쓸 때는 source issue `727/730`과
  `main_l0_v3`를 함께 기록함

## 현재 질문과 입력

“iBKD의 분류 이득이 레이어 연결 학습에서 왔는가?”라는 원인 분석은
`main_l0_v3`의 현상에서 출발합니다. 따라서 기존 checkpoint 관찰 분석에는 issue
727의 batch-128 seed 1과 issue 730의 batch-128 seed 2·3만 사용합니다.
`loader_pilot_l0_seed1`은 이름은 L0지만 별도의 재학습이고 checkpoint archive도
아직 감사되지 않았으므로 섞지 않습니다. Seed-1 연결 ablation에서 canonical
`learned_all`이 원래 성능을 재현하지 못했으므로, 현재는
`main_l0_repro_aa_v2` 실행 gate를 먼저 통과하고 300-epoch A/A 변동을 측정해야
합니다.
