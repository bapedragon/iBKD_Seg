# Cityscapes · Segmenter-Ti/16

2026-09-28 CE 비율 표기 통일: 현재 문서의 초기 비율은 L/16과 같은
**첫 step의 `β × guidance / CE`**를 사용합니다.
[계산 근거와 전체 환산표](reports/initial_ce_ratio/README.md)를 확인합니다.
로그 정밀도에 따른 근삿값이며 β·학습 결과·선정 후보는 그대로입니다.
고정 config와 과거 종료 JSON의 `initial_target_ratio`/`initial_ratio`는 당시
25batch 중앙값 기준 목표값으로 보존하며 첫 step 실측값과 구분합니다.

Small 후속 실험 준비는 [별도 S/16 폴더](../Cityscapes_Segmenter-S16/README.md)에 있습니다.
Tiny 10k까지 먼저 진행한 뒤 시작하며, Small의 β는 새로 측정합니다.
기존 v1에서 1,995step에 중단됐던 iBKD λ0.25의 8번 후보는 **#844에서 0→2,000step
새 실행과 전체 val 평가를 완료**했습니다. 이전 중단 checkpoint의 재개는 더 이상 필요하지 않습니다.

2026-09-26 사용자 결정: Tiny를 먼저 검사하며 **기존 OpenMMLab teacher를 유지**합니다.
이 폴더는 L/16 결과를 변경하지 않는 별도 실험입니다. 현재 구현 범위는 초기 손실 측정과
25-step 연결 smoke, 500-step β 후보 검사, 전체 val 평가 시간 측정,
iBKD λ0.25와 LG·ALG의 2,000-step β 후보 비교, iBKD λ0.5의 4개 후보 2k 비교,
FSKD*·C2VKD*의 고정 계수 2k 비교, Vanilla 2k·10k와 ALG 1위·iBKD λ0.25 상위 두 후보의 10k 비교입니다.
**이 실행들은 모두 해당 step·전체 val500 평가를 완료**했습니다.
초기 #834의 LG/ALG 궤적 문제는 후속 v4 반복 검사에서 확인했고,
사용자 제공 v6 종료 로그에서 **24개 모두 500step 완료·공통 조건·저장 상태 비교 통과**를 확인했습니다.
사용자 제공 2k v1의 완료된 1~7번과 #844의 8번을 합쳐 **λ0.25의 8개 모두
2,000step·전체 val500 완료**를 확인했습니다.
[통합 결과 표](reports/beta_screen/ti16_crop512_ibkd_lambda0p25_grid2000/RESULTS.md)와
[수치/검사 요약](reports/beta_screen/ti16_crop512_ibkd_lambda0p25_grid2000/beta_screen_summary.json)에 반영했습니다.
LG·ALG도 **각 8개, 총 16개 모두 2,000step·전체 val500 완료**했습니다.
[LG·ALG 결과 표](reports/beta_screen/ti16_crop512_lg_alg_grid2000/RESULTS.md)에 반영했으며,
두 방법 모두 accuracy/mIoU 1위는 β=0.11195046919685056, 2위는 β=0.009329205766404213입니다.
ALG는 8개 모두 2k까지 가이던스를 유지했고 LG와의 step 비교는 허용오차 내에서 통과했습니다.
2026-09-27 사용자 결정으로 λ0.5는 기존 후보 **1·3·6·8번 4개**를 2k 비교했고,
**네 개 모두 2,000step·전체 val500 완료**했습니다.
[λ0.5 결과 표](reports/beta_screen/ti16_crop512_ibkd_lambda0p5_grid2000_4betas/RESULTS.md)에 반영했습니다.
이로써 계획한 **LG 8개 + ALG 8개 + λ0.25 8개 + λ0.5 4개 = 28개**의 2k 비교가 완료됐습니다.
LG·ALG·iBKD의 후속 후보는 아래 **mIoU 기준 상위 2개씩, 총 8개**입니다.
2026-09-28 반영: FSKD*·C2VKD*의 각 2k·전체 val500 결과는 각각 **mIoU 24.3009%, 14.3731%**입니다.
[고정 비교군 결과](reports/baseline_screen/ti16_crop512_fskd_c2vkd_grid2000/RESULTS.md)에 정리했습니다.
기존 β 후보 28개와 고정 비교군 2개를 합쳐 **총 30개의 2k 실행이 완료**됐습니다.
실행·수치 검사는 통과했지만 특히 C2VKD*는 15개 클래스 IoU가 0으로 초기 성능이 낮습니다.
후속 **Vanilla 2k → ALG 1위 10k → iBKD λ0.25 1위 10k**도 세 실행 모두 완료했습니다.
준비 포함 **7시간 59분 57초**, mIoU는 각각 **36.6126%, 48.3442%, 43.8013%**입니다.
[첫 10k 묶음 결과](reports/followup10k/ti16_alg_ibkd025_top1_10k_vanilla2k/RESULTS.md)에 반영했습니다.
Vanilla를 포함해 2k 완료 결과는 31개이며, 상위 후보 8개 중 현재 10k 완료는 3개입니다.
**Vanilla 단독 10k도 완료**했습니다. mIoU **46.1775%**, accuracy **87.2847%**, 준비 포함
**1시간 56분 49초**입니다. [Vanilla 10k 및 동일 학습량 비교](reports/followup10k/ti16_vanilla_10k/RESULTS.md)에
반영했습니다. **iBKD λ0.25 후보 7의 단독 10k도 완료**, mIoU **42.1373%**, 준비 포함
**4시간 33분 28초**입니다. [후보 7 결과와 최신 비교](reports/followup10k/ti16_ibkd_lambda0p25_b7_10k/RESULTS.md)를 확인합니다.
현재 10k mIoU는 **ALG 48.3442 > Vanilla 46.1775 > iBKD λ0.25 후보 1 43.8013 > 후보 7 42.1373**입니다.
λ0.5의 생략 후보 2·4·5·7번은 실패한 것이 아닙니다.
나머지 다섯 후보(LG 7·1, ALG 1, iBKD λ0.5 1·6)의 10k는 후속 계획이며,
β=2.5의 후속 실험은 사용자 결정에 따라 보류합니다.
500step 점수는 val2 진단이므로 후보 순위 선정에 쓰지 않습니다.
이슈는 사용자가 제출하며 이슈 입력용 MD 파일이나 GitHub 이슈는 생성하지 않습니다.

## 완료: iBKD λ0.25 · 후보 7 단독 10,000step

2026-09-28 사용자 제공 로그에서 **10,000step·전체 val500 완료**를 확인했습니다.
[고정 설정](configs/ibkd025_b7_10k_v1.json)과 [실행 스크립트](scripts/run_ibkd025_b7_10k.sh)의
실행 기록이며 같은 실험을 다시 돌릴 필요는 없습니다. **mIoU 42.1373%, accuracy 86.7490%**,
준비 포함 **4시간 33분 28초**입니다. [상세 결과](reports/followup10k/ti16_ibkd_lambda0p25_b7_10k/RESULTS.md)에 반영했습니다.
대상은 2k 전체 val500 mIoU 2위 **후보 7, β=0.29542009465901614, λ=0.25**입니다.
초기 비율은 L/16과 같은 첫 step 계산으로 **약 17.598%**입니다. β 자체는 변경하지 않으며,
config의 `initial_target_ratio=0.18`은 과거 25batch 중앙값 기준 목표값으로 보존합니다.

```bash
env -u CITYSCAPES_TI16_RESUME CITYSCAPES_TI16_START_RUN=1 bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_ibkd025_b7_10k.sh
```

- **seed1·0→10,000step 새 학습** 하나만 실행합니다. 이전 2k checkpoint는 필요 없습니다.
  완료한 후보 1(β=0.02461834122158468, mIoU 43.8013%)과 같은 학습량으로 비교합니다.
  후보 선정 기록의 SHA-256과 2위 β를 검증하며 자동으로 후보나 λ를 추가하지 않습니다.
- Tiny/16·decoder1·OpenMMLab DeepLabV3-R101 teacher·fine train2975·crop512·batch8·FP32·
  SGD LR0.01·**80k poly LR schedule**, iBKD **warm-up 20epoch**를 그대로 유지합니다.
  가이던스는 조건 충족 시 가장 빨리 7,441step부터 꺼질 수 있으며, 실제 종료 step을 기록합니다.
- **고정 10,000step에서 전체 val500**을 한 번 평가합니다. mIoU를 우선 비교하고,
  pixel accuracy와 19 class IoU도 기록합니다. Test는 사용하지 않습니다.
- 사전 예상은 **약 5~6시간**이었고, 실제로는 준비 포함 **4시간 33분 28초**에 완료했습니다.
  가이던스는 **7,441step부터 종료**했습니다.
  총 10시간 예산, **9시간 58분 중단 요청·120초 저장 여유**를 유지합니다.
- 기존 2k/10k 공통 학습 경로를 사용하므로 별도 H200 smoke 없이 진행합니다.
  250step·epoch 경계·최종·시간 중단 시 학습 상태를 저장하며, 동일 commit·설정·환경의
  이 실행에서 중단된 전체 폴더가 있을 때만 `CITYSCAPES_TI16_RESUME`으로 재개합니다.
- 중간 로그를 유지하고 마지막 **`[CITYSCAPES_TI16_FOLLOWUP_FINAL]`**에
  `pack=ibkd_l025_b7_10k`, 완료/선택 step·epoch, loss·CE·guidance, mIoU·accuracy·19 IoU,
  가이던스 종료 시점, checkpoint 검사·경고·오류·시간·저장 경로를 출력합니다.
  **`initial_ratio_first_step_percent`**는 이번 실행 첫 batch에서 실제 관측한 비율입니다.
  과거 목표값 `initial_ratio`와 구분하며, 첫 step을 실행하지 못했으면 실측값은 `null`입니다.

출력: `/app/output/cityscapes_ti16_ibkd_l025_b7_10k_v1/run_<UTC>_<PID>/`.
통합 요약은 `artifacts/grid_summary.json`, 개별 결과·재개 상태는 `artifacts/ibkd_l025_b7/`에 있습니다.
서버 출력 폴더의 지속 보존을 가정하지 않으므로 완료 후 전체 결과 폴더를 보관합니다.

## 완료: Vanilla 단독 10,000step

2026-09-28 사용자 제공 로그에서 **10,000step·전체 val500 완료**를 확인했습니다.
준비 포함 **1시간 56분 49초**, 실패·시간 중단 없음입니다.
[고정 설정](configs/vanilla10k_v1.json)과 [실행 스크립트](scripts/run_vanilla10k.sh)로
seed1·**0→10,000step 새 학습**을 수행했습니다. 아래 명령은 실행 기록이며 다시 돌릴 필요는 없습니다.

| 방법 | Step | mIoU (%) | Pixel accuracy (%) | Vanilla 대비 mIoU (%p) |
|---|---:|---:|---:|---:|
| ALG · 후보 7 | 10,000 | 48.3442 | 90.2803 | +2.1667 |
| Vanilla | 10,000 | 46.1775 | 87.2847 | 기준 |
| iBKD λ0.25 · 후보 1 | 10,000 | 43.8013 | 86.2812 | −2.3762 |
| iBKD λ0.25 · 후보 7 | 10,000 | 42.1373 | 86.7490 | −4.0402 |

이 단일 seed의 10k에서는 iBKD λ0.25 두 후보가 Vanilla보다 낮습니다. λ0.5 후보와
최종 80k 결과는 아직 없으며, 가이던스 종료가 차이의 원인이라고 단정할 수는 없습니다.
[상세 결과와 검사 범위](reports/followup10k/ti16_vanilla_10k/RESULTS.md)를 확인합니다.

```bash
env -u CITYSCAPES_TI16_RESUME CITYSCAPES_TI16_START_RUN=1 bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_vanilla10k.sh
```

- Tiny/16·decoder1·fine train2975·crop512·batch8·FP32·SGD LR0.01·80k poly LR schedule 등은
  완료한 ALG·iBKD 10k와 같습니다. Teacher·guidance 모듈을 로드하지 않고 **CE만 학습**합니다.
  β 탐색과 guidance 종료 controller는 없으며 LR schedule을 10k로 줄이지 않습니다.
- **고정 10,000step checkpoint에서 전체 val500**을 한 번 평가합니다. mIoU가 1차 지표이고
  pixel accuracy·19 class IoU도 함께 기록합니다. Test는 사용하지 않습니다.
  완료한 ALG 48.3442%·iBKD λ0.25 43.8013%와 같은 학습량의 비교입니다.
- 기존 Vanilla 2k 학습 1,384.2초를 5배 하고 전체 val 약 99초를 더하면 약 1시간 57분입니다.
  준비·실행 편차 포함 사전 예상은 **약 2시간~2시간 15분**, 실제는 **1시간 56분 49초**였습니다.
  기존처럼 10시간 예산 중 **9시간 58분에 중단 요청, 마지막 120초 저장 여유**를 유지합니다.
- 기존 Vanilla 2k 실행이 완료된 공통 학습 경로를 사용하므로 추가 H200 smoke는 없습니다.
  단독 실행 설정·보고, CE 전용 학습·중단/재개, 기존 후보 실행 경로를 포함한 **로컬 검사 59개를 통과**했습니다.
- 250step·epoch 경계·최종·시간 중단 시 optimizer/RNG를 포함해 저장합니다. 이 10k 설정으로
  중단된 경우에는 동일 commit·설정·환경에서 전체 실행 폴더를 복원하고 `CITYSCAPES_TI16_RESUME`을
  지정해 재개할 수 있습니다. 기존 2k 설정의 checkpoint 전환은 포함하지 않습니다.
- 중간 로그를 유지하고 마지막 **`[CITYSCAPES_TI16_FOLLOWUP_FINAL]`**에 `pack=vanilla_10k`,
  목표/완료/선택 step·epoch, loss·CE, mIoU·accuracy·19 IoU, teacher/guidance 미사용 상태,
  평가 중 student 불변·checkpoint 검사, 시간·경고·오류·저장 경로를 출력합니다.
  다른 방법이나 80k 학습으로 자동 진행하지 않습니다.

출력: `/app/output/cityscapes_ti16_vanilla_10k_v1/run_<UTC>_<PID>/`.
통합 요약은 `artifacts/grid_summary.json`, 개별 결과와 재개 상태는 `artifacts/vanilla_10k/`에 있습니다.
서버 출력 폴더의 지속 보존을 가정하지 않으므로 로그·요약·checkpoint를 포함한 전체 폴더를 보관합니다.

## 완료: Vanilla 2k + ALG·iBKD λ0.25 각 1위 10k

2026-09-28 사용자 제공 로그에서 **세 실행 모두 완료**, 실패·시간 중단 없음으로 확인했습니다.
총 **7시간 59분 57초**가 걸렸습니다. 아래 명령은 실행 기록이며 같은 묶음을 다시 돌릴 필요는 없습니다.
[고정 설정](configs/followup10k_alg_ibkd025_top1_vanilla2k_v1.json)과
[실행 스크립트](scripts/run_followup10k_alg_ibkd025_top1_vanilla2k.sh)를 사용한 결과입니다.

```bash
env -u CITYSCAPES_TI16_RESUME CITYSCAPES_TI16_START_RUN=1 bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_followup10k_alg_ibkd025_top1_vanilla2k.sh
```

| 실행 순서 | 방법 | 기존 후보 번호 | β | 학습 범위 | 실제 학습+전체 val |
|---|---|---:|---:|---|---|
| 1 | Vanilla | — | 0 | 0→2,000step | 24분 43초 |
| 2 | ALG | 7 | 0.11195046919685056 | 0→10,000step | 2시간 39분 44초 |
| 3 | iBKD λ0.25 | 1 | 0.02461834122158468 | 0→10,000step | 4시간 47분 27초 |

| 방법 | 평가 step | mIoU (%) | Pixel accuracy (%) | 가이던스 종료 |
|---|---:|---:|---:|---|
| Vanilla | 2,000 | 36.6126 | 86.6675 | 해당 없음 |
| ALG · 후보 7 | 10,000 | 48.3442 | 90.2803 | 10k까지 유지 |
| iBKD λ0.25 · 후보 1 | 10,000 | 43.8013 | 86.2812 | 20epoch 완료, 7,441step부터 꺼짐 |

기존 동일 후보 2k 대비 mIoU는 ALG **+9.2072%p**, iBKD **+6.4576%p**입니다.
이번 두 10k 결과에서는 ALG가 4.5429%p 높습니다. Vanilla 2k를 KD 10k와 동등한 학습량으로
비교하지 않습니다. [상세 해석·검사 범위](reports/followup10k/ti16_alg_ibkd025_top1_10k_vanilla2k/RESULTS.md)를 확인합니다.

ALG와 iBKD는 **기존 2k 전체 val500 mIoU 1위**를 선택했습니다. 선정 기록의 SHA-256과 β를
config에 고정하며 초기 비율을 다시 측정하지 않습니다. 짧은 Vanilla를 먼저 끝낸 뒤 긴 실행을
진행합니다. 세 실행 모두 seed1·동일 초기 student에서 새로 시작하며 **이전 2k checkpoint가 필요 없습니다.**
기존 checkpoint에 8,000step을 추가하는 실행이 아닙니다.

- Tiny/16·decoder1·OpenMMLab DeepLabV3-R101 teacher·fine train2975·crop512·batch8·FP32·
  SGD LR0.01·**80,000step LR schedule**을 유지합니다. 10k 종료에 맞춰 LR schedule을 줄이지 않습니다.
  Vanilla는 teacher와 guidance를 로드하지 않고 CE로만 학습합니다.
- **ALG warm-up=0, iBKD warm-up=20epoch**를 유지합니다. epoch당 372step으로,
  조건을 만족할 경우 가장 이른 가이던스 종료 적용 step은 각각 745·7,441입니다.
  이 step에 반드시 끄는 것이 아니며 실제 종료 여부·epoch·step을 기록합니다.
- 각 실행의 **고정 마지막 step**에서 원본 해상도 전체 val500을 한 번 평가합니다.
  1차 지표는 **mIoU**, 보조 지표는 pixel accuracy이며 19개 class IoU도 기록합니다.
  Vanilla 2k는 기존 2k 결과와 비교하며, 이번 KD 10k와 같은 학습량의 최종 비교로 해석하지 않습니다.
- 기존 연결 smoke·2k 검사를 통과한 경로이므로 추가 H200 smoke 없이 실행합니다.
  새 묶음, CE 단독 학습, controller 종료 전후의 중단/재개, 기존 2k·500step 경로를 포함한
  **로컬 검사 57개를 통과**했고 이후 이번 H200 endpoint 실행도 완료했습니다.
  이는 80k 전체 학습이나 GPU bitwise 재현성을 보장하는 결과는 아닙니다.
- 기존 2k 시간에서 가이던스를 끝까지 유지한다고 계산한 학습+val 합계는 **약 8시간 37분**입니다.
  Vanilla는 짧은 smoke의 update 속도와 기존 입력 처리 비용을 이용한 추정입니다.
  준비·실행 편차 포함 사전 예상은 **약 9시간**, 실제는 **7시간 59분 57초**였습니다.
  스크립트 시작 기준 **9시간 58분에 중단을 요청하고 마지막 120초를 저장 여유**로 둡니다.
  플랫폼의 시작/마감 시각은 `CITYSCAPES_TI16_JOB_STARTED` / `CITYSCAPES_TI16_JOB_DEADLINE`으로
  전달할 수 있습니다. 별도 짧은 시간 상한은 없습니다.
- 250step·epoch 경계·최종·시간 중단 시 optimizer·RNG·controller·진행 중 epoch 상태까지 저장합니다.
  시간 중단이면 다음 방법을 시작하지 않습니다. 수치 오류는 해당 방법의 실패로 기록하고
  남은 방법을 수행합니다. 미완료 평가 수치를 완료 결과로 채우지 않습니다.
- 중간 로그를 유지하며 마지막 **`[CITYSCAPES_TI16_FOLLOWUP_FINAL]`**에 세 실행의 상태,
  β·λ·목표/완료/선택 step·epoch, loss·CE·guidance·mIoU·accuracy·19 IoU, controller 상태와
  종료 step, 공통 입력/초기값 검사, checkpoint 경로, 시간·경고·실패 이유를 모읍니다.
  종료 JSON은 50,000 ASCII byte 이내이며 긴 오류를 포함한 로컬 검사에서는 약 7,300 bytes였습니다.

출력: `/app/output/cityscapes_ti16_alg_ibkd025_top1_10k_vanilla2k_v1/run_<UTC>_<PID>/`.
`artifacts/grid_summary.json`이 통합 결과이며 `artifacts/vanilla_2k/`, `artifacts/alg_b7/`,
`artifacts/ibkd_l025_b1/`에 요약·step 이력·`resume.json`·checkpoint가 남습니다.
**출력 폴더 전체를 보관**합니다. 서버 경로가 다음 이슈에도 남아 있다고 가정하지 않습니다.

시간 중단 후에는 동일 commit·설정·데이터·실행 환경에서 해당 실행 폴더를 복원하고,
`CITYSCAPES_TI16_START_RUN=1/2/3`으로 위 순서를 선택한 뒤 `CITYSCAPES_TI16_RESUME`에
복원된 `resume.json`을 지정합니다. 재개 pointer는 첫 실행에만 적용되고 뒤의 실행은 새로 시작합니다.
**다른 설정의 2k checkpoint를 이 묶음의 10k로 전환하거나 80k로 자동 진행하지 않습니다.**
이 묶음에서는 ALG 1위·λ0.25 1위 두 개를 완료했습니다. 이후 λ0.25 2위도 완료해
현재 상위 후보 8개 중 세 개 완료, 나머지 다섯 개는 후속 계획입니다.

## 완료: FSKD*·C2VKD* 각 2,000step

두 방법 모두 2,000step·전체 val500 평가를 완료했습니다. 준비 포함 **1시간 23분 36초**입니다.
FSKD*는 mIoU 24.3009% / accuracy 80.5187%, C2VKD*는 mIoU 14.3731% / accuracy 77.1672%입니다.
[결과 및 해석](reports/baseline_screen/ti16_crop512_fskd_c2vkd_grid2000/RESULTS.md)과
[수치·검사 요약](reports/baseline_screen/ti16_crop512_fskd_c2vkd_grid2000/baseline_screen_summary.json)을
확인합니다. **현재 2k를 다시 실행할 필요는 없으며 아래 명령은 실행 기록**입니다.

두 방법은 기존 25-step smoke에서 각각 통과했습니다. 당시 공통 검사의 LG/ALG 불일치는
이후 별도 반복 검사로 확인했으며, FSKD*/C2VKD*의 25-step·val2 결과를 2k 성능으로 취급하지 않습니다.
[고정 설정](configs/baseline_grid2000_fskd_c2vkd_v1.json)과
[실행 스크립트](scripts/run_grid2000_fskd_c2vkd.sh)를 사용한 결과입니다.

```bash
env -u CITYSCAPES_TI16_RESUME CITYSCAPES_TI16_START_RUN=1 bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_fskd_c2vkd.sh
```

| 실행 순서 | 방법 | 손실 계수 | 비교군 구분 |
|---|---|---|---|
| 1 | FSKD* | CE 1, logit KD 1, global 100, patch 1, attention 1,000,000 | 공통 사전학습; 공개 DeiT-Ti 분류 recipe의 segmentation 이식 |
| 2 | C2VKD* (CLIP-pool) | 별도 CE 0, PDD 1, global 0.1, patch 0.1, linguistic 0.5 | 추가 CLIP 사전학습을 사용한 보조 비교군 |

- **β 후보 탐색·초기 CE 비율 재조정·가이던스 종료 controller가 없습니다.** 계수는 기존 v3
  smoke와 같고, C2VKD의 PDD에 정답 감독이 들어 있으므로 진단 CE를 총 loss에 다시 더하지 않습니다.
  두 방법 모두 저자의 원본 Cityscapes 설정을 완전 재현한 결과로 표기하지 않습니다.
- Tiny/16, OpenMMLab DeepLabV3-R101 teacher, seed1, crop512, batch8, FP32,
  SGD LR0.01와 **80k LR schedule**은 기존 2k 후보군과 같습니다. 각 방법을 0→2,000step
  새로 학습하고, 마지막 checkpoint에서 원본 해상도 전체 val500을 평가합니다.
- **1차 지표 mIoU**, 보조 pixel accuracy, 19개 class IoU를 함께 기록합니다. test는 사용하지 않습니다.
- 데이터는 기존 추출본을 검사해 재사용하거나 `/app/data/chaoyang`의 검증된 ZIP에서 준비합니다.
  torchsort0.1.10 CUDA 확장과 CLIP RN101 attention pool을 설치/검증하며 teacher를 교체하지 않습니다.
- 하나의 이슈에서 순차 실행합니다. 스크립트 시작 기준 9시간 58분에 중단을 요청하고 마지막
  120초는 저장 여유로 둡니다. 외부 시작 시각/종료 시각이 있으면 `CITYSCAPES_TI16_JOB_STARTED` /
  `CITYSCAPES_TI16_JOB_DEADLINE`으로 전달할 수 있습니다.
- 250step·epoch 경계·최종·시간 중단 시 optimizer/RNG/방법별 모듈까지 저장합니다. 같은 설정으로
  재개하려면 해당 실행 폴더와 checkpoint를 보관하고 `CITYSCAPES_TI16_RESUME`에 `resume.json`을
  지정합니다. `CITYSCAPES_TI16_START_RUN=2`는 C2VKD부터 실행합니다. 서버 출력 경로가 다음 이슈에도
  남아 있다고 가정하지 않습니다. **2k→10k 전환은 후속 설정과 검증이 필요한 별도 작업**입니다.
- 중간 로그를 유지하고 마지막 `[CITYSCAPES_TI16_GRID2000_FINAL]`에 두 방법의 loss·CE·손실 구성항,
  고정 계수, 완료/선택 step·epoch, mIoU·accuracy·19 IoU, teacher/pool 동결, 저장·복원 검사,
  시간·메모리·실패 상태를 모두 출력합니다. 마지막 JSON은 50,000 ASCII byte 이내로 제한합니다.
  수치 오류는 해당 방법의 실패로 기록하고 다음 방법을 수행하며, 시간 중단 시 다음 방법은 시작하지 않습니다.

출력: `/app/output/cityscapes_ti16_fskd_c2vkd_grid2000_v1/run_<UTC>_<PID>/`.
`artifacts/grid_summary.json`이 두 결과 통합 파일이고 각 방법 폴더에 `summary.json`, `steps.jsonl`,
`resume.json`과 checkpoint가 남습니다. JSON/로그/체크포인트를 보관한 뒤 10k 실험을 결정합니다.
사전 예상은 두 방법 합계 약 1시간 30분~2시간이었으며 실제 준비 포함 1시간 23분 36초였습니다.
학습+val 시간은 FSKD* 36분 58초, C2VKD* 37분 28초입니다.
로컬 관련 검사 56개를 통과했습니다. 작은 CPU 모듈로 실제 공통 학습 루프의 고정 손실·
pool 동결·중단/재개 상태 일치를 검사하고, 기존 FSKD/C2VKD 손실 primitive, 설정 잠금,
전체 평가 연결, 실패 시 다음 방법 실행, 종료 로그 길이를 확인했습니다.
이 로컬 코드 검사와 이후 H200 2k 결과는 별개의 검증 기록입니다.

## 후속 계획: mIoU 기준 10k 후보 2개씩

**2026-09-27 사용자 정정: 후속 β 후보 선정의 1차 기준은 전체 val500의 mIoU이며,
pixel accuracy는 보조 지표입니다.** 아래는 고정 2,000step 결과에서 mIoU가 높은 두 후보입니다.
λ0.5는 수행한 네 후보 안에서의 순위입니다. **ALG 1위·λ0.25 1·2위의 10k는 완료**했으며,
나머지 다섯 후보는 아직 후속 계획입니다. 아래 표는 선정 근거인 2k 결과를 유지합니다.

| 실험군 | 2k 비교 수 | mIoU 1위 β | mIoU (%) | mIoU 2위 β | mIoU (%) |
|---|---:|---:|---:|---:|---:|
| LG | 8 | 0.11195046919685056 | 39.1370 | 0.009329205766404213 | 39.0550 |
| ALG | 8 | 0.11195046919685056 | 39.1370 | 0.009329205766404213 | 39.0550 |
| iBKD λ0.25 | 8 | 0.02461834122158468 | 37.3437 | 0.29542009465901614 | 37.1969 |
| iBKD λ0.5 | 4 | 0.03668347229720436 | 37.4001 | 0.2934677783776349 | 36.2344 |

λ0.25는 앞서 제안한 accuracy 기준 8·2번 대신 **1·7번**으로 정정합니다.
LG·ALG는 각각 **7·1번**, λ0.5는 **1·6번**입니다. LG·ALG는 2k 점수가 같아도
10k 중 ALG 종료로 갈라질 수 있으므로 각각 두 개를 학습하는 계획입니다.
λ0.25의 mIoU 2위(7번, 37.1969%)와 3위(2번, 37.1612%) 차이는 0.0357%p입니다.
두 후보로 다음 단계 비용을 제한하는 것은 가능하지만, 2k 순위가 10k/80k의 최적 β를 확정하지는 않습니다.

[후속 후보 목록](reports/beta_screen/ti16_grid2000_top2_miou_v1.json)에 원래 정밀도의 β와
원본 요약 SHA-256, 순위, 선정 기준을 기록했습니다. 이 파일은 후보 계획이며 실행 config가 아닙니다.
기존 2k는 전부 고정 endpoint에서 accuracy와 mIoU를 함께 계산했으므로 순위 변경 때문에
재학습할 필요는 없습니다. 과거 config·종료 JSON의 `primary_metric=pixel_accuracy`는 실행 당시 기록으로
보존합니다. 후속 실행 config와 보고에서는 **mIoU를 1차 지표**로 명시해야 합니다.

## 완료: iBKD λ0.5 · 후보 1/3/6/8 · 2,000step

기존 500step에서 통과한 **4개 후보** 각각의 0→2,000step 학습과 전체 val500 평가를 완료했습니다.
준비 포함 **5시간 19분 12초**가 걸렸고 실패·중단·미실행 후보는 없습니다.
Accuracy 1위는 6번(87.5054%), mIoU 1위는 1번(37.4001%)이며 두 지표 모두 1·6번이 상위 두 개입니다.
**현재 이 네 후보를 다시 돌릴 필요는 없습니다.** 아래 명령과 중단/재개 설명은 실행 기록입니다.
[고정 설정](configs/beta_grid2000_ibkd_l050_4betas_v1.json)을 사용했습니다.

```bash
env -u CITYSCAPES_TI16_RESUME CITYSCAPES_TI16_START_CANDIDATE=1 bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_ibkd050_4betas.sh
```

| 기존 후보 번호 | β | 초기 β×guidance/CE 기준 |
|---|---:|---:|
| 1 | 0.03668347229720436 | 1.5% |
| 3 | 0.11005041689161307 | 4.5% |
| 6 | 0.2934677783776349 | 12% |
| 8 | 0.5869355567552698 | 24% |

최솟값·최댓값과 중간 범위를 남긴 탐색입니다. 500step 성능 순위로 고른 것이 아니며, 나머지
2·4·5·7번은 실패 후보가 아니라 이번 묶음에서 생략한 후보입니다. 1~4로 다시 번호를 매기지 않습니다.
기존 λ0.25의 8개 및 LG·ALG 각 8개 결과는 그대로 보존하고 λ0.5의 탐색 수만 4개로 기록합니다.

- Tiny·OpenMMLab DeepLabV3-R101 teacher·fine train2975·crop512·batch8·seed1·decoder1·FP32·
  SGD LR0.01·80k LR schedule·iBKD CPU 결정성 경로를 유지합니다. β를 재계산하지 않습니다.
  λ=0.5로 `guidance = 0.5 × alignment + 0.5 × fusion`, `loss = CE + β × guidance`를 사용합니다.
- **iBKD warm-up=20epoch**도 유지합니다. 2k는 약 5.38epoch이므로 이번에는 가이던스가 켜진
  구간에서 β를 비교합니다. ALG의 warm-up=0과 혼동하지 않습니다.
- 고정 2,000step에서 전체 fine val500을 한 번 평가해 pixel accuracy·mIoU·19 class IoU를 기록합니다.
  원본 해상도·window/stride512·평가 window batch1·CPU thread4이며 test는 사용하지 않습니다.
  학습 순서는 1→3→6→8이고, 각 후보는 같은 seed와 초기값·입력 순서로 새로 시작합니다.
- 예상 시간은 준비·전체 val 포함 **약 5시간**입니다. 완료까지 더 걸려도 별도 5시간 상한을 두지 않고
  기존 **10시간 예산 중 마지막 2분만 저장에 남겨 9시간58분에 중단 요청**을 보냅니다.
  기본 기준은 스크립트 진입 시각입니다. 플랫폼 실제 시작/마감은 기존
  `CITYSCAPES_TI16_JOB_STARTED`/`CITYSCAPES_TI16_JOB_DEADLINE`으로 전달할 수 있습니다.
- 250step·epoch 경계·최종·시간 중단 시 checkpoint를 저장합니다. 수치 오류는 해당 후보를 기록하고
  남은 후보를 계속 실행합니다. 시간 중단이면 이후 후보는 `not_run`으로 남깁니다.
- 중간 로그를 유지하고 마지막 `[CITYSCAPES_TI16_GRID2000_FINAL]`에 `pack=ibkd_l050_4betas`,
  `configured_pack_runs=4`, `configured_candidates=[1,3,6,8]`과 네 결과를 모읍니다.
  β·λ·선택 step/epoch·loss/CE/guidance·alignment/fusion·정확도/mIoU/class IoU·가이던스 상태·
  저장 검사·실패/중단 이유를 포함합니다. 긴 오류를 넣은 로컬 검사에서도 약 11,448 bytes로
  마지막 65,000자 안에 들어갔습니다. 10k나 80k 학습으로 자동 진입하지 않습니다.

출력: `/app/output/cityscapes_ti16_ibkd_l050_grid2000_4betas_v1/run_<UTC>_<PID>/`.
`artifacts/grid_summary.json`과 후보별 `summary.json`, `resume.json`, `checkpoints/`, `steps.jsonl`을 보존합니다.
시간 중단 후 재개하려면 해당 후보 폴더를 다음 컨테이너로 복원하고 **동일 commit**에서
`CITYSCAPES_TI16_START_CANDIDATE`에 기존 번호 **1/3/6/8 중 하나**를 지정합니다.
`CITYSCAPES_TI16_RESUME`은 그 후보의 복원된 `resume.json`이며 첫 후보에만 적용됩니다.
예를 들어 시작 번호 3이면 3번을 재개하고 6·8번을 새로 학습합니다. checkpoint가 없으면
RESUME을 비우고 해당 후보부터 새로 학습합니다.

로컬 검사 **29개 통과**: 기존 2k/LG·ALG 검사와 새 4후보 선택·CLI·실패 후 진행·중단/재개·
종료 로그 검사를 포함합니다. 작은 CPU 모델의 실제 공유 학습 루프에서 λ0.5 loss 구성과
중단/재개 후 모델·optimizer·scheduler·controller·입력·평가값 일치도 확인했습니다.
이는 실제 H200의 2k 성능이나 안정성을 미리 측정한 결과는 아닙니다.

## 완료: LG·ALG 묶음 2,000-step 후보 비교

기존 500step 검사와 LG·ALG 반복 검사를 통과한 설정으로 **LG 8개 + ALG 8개, 총 16개**를
실행했습니다. **16개 모두 완료됐으므로 지금 다시 실행할 필요는 없습니다.**
준비 포함 실제 실행 시간은 **9시간 2분 34초**, 실패·중단·미실행 후보는 0개였습니다.
아래 명령과 중단/재개 안내는 실행 기록용으로 남깁니다.
[고정 설정](configs/beta_grid2000_lg_alg_v1.json)은 기존 λ0.25 2k v2의 공통 학습·평가 조건을
유지하고, 방법과 후보 목록만 LG·ALG로 바꿉니다.

```bash
env -u CITYSCAPES_TI16_RESUME CITYSCAPES_TI16_START_RUN=1 bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_lg_alg.sh
```

| 후보 | LG·ALG 공통 β | 첫 step β×guidance/CE(약) |
|---|---:|---:|
| 1 | 0.009329205766404213 | 1.461% |
| 2 | 0.018658411532808426 | 2.923% |
| 3 | 0.02798761729921264 | 4.384% |
| 4 | 0.03731682306561685 | 5.845% |
| 5 | 0.05597523459842528 | 8.768% |
| 6 | 0.0746336461312337 | 11.690% |
| 7 | 0.11195046919685056 | 17.535% |
| 8 | 0.1492672922624674 | 23.380% |

2026-09-27 정정: 이전 이 표의 초기 비율 1/2/3/4/6/8/12/16%는 문서 오기였습니다.
당시 고정 config·종료 로그의 25batch 목표는 1.5/3/4.5/6/9/12/18/24%였습니다.
2026-09-28에는 위 표를 첫 step 기준 근삿값으로 환산했습니다. 실행 β와 학습 조건은 동일합니다.

- 실행 순서는 **LG b1 → ALG b1 → LG b2 → ALG b2 → … → LG b8 → ALG b8**입니다.
  각 실행을 seed1의 동일 초기 상태와 입력 순서로 0→2,000step 학습하므로 이전 checkpoint는
  필요하지 않습니다. 첫 step 비율은 초기 loss 강도 설명값이며 학습 중 유지되는 비율이 아닙니다.
  후보 생성에는 당시 25batch 목표를 사용했고, 후속 후보 선택에는 val mIoU를 사용합니다.
- Tiny·OpenMMLab DeepLabV3-R101 teacher·crop512·batch8·FP32·SGD LR0.01·decoder1·
  80,000step LR 스케줄을 유지합니다. 데이터 경로·준비 방법도 기존 2k 묶음과 같습니다.
- **ALG warm-up=0**을 유지합니다. 372step/epoch이며 controller가 가장 일찍 끌 수 있는 시점은
  2epoch 관측 직후 **745step**입니다. 이번 결과에서는 8개 ALG 모두 2k까지 꺼지지 않았습니다. LG는 계속 켜지므로
  2k에서는 두 방법을 별도로 학습합니다. 양쪽 가이던스가 켜진 공통 구간의 수치·입력 동일성도
  검사하며, ALG 종료 이후의 궤적 차이는 정상 동작입니다.
- 각 후보의 **고정 2,000step에서 전체 fine val 500장**을 평가합니다. Accuracy·mIoU·19 class IoU와
  평가 시간을 기록하며 test는 사용하지 않습니다. 10k 진입이나 후보 선정은 자동으로 하지 않습니다.
- 기존 LG 500step 실측 평균 약 462초와 val500 약 108초를 적용하면 준비 포함 **약 8시간35분~9시간5분**
  예상입니다. 실행 속도에 따라 10시간을 넘길 수 있으므로 완료를 보장하는 값은 아닙니다.
  ALG 종료로 빨라지는 효과는 예상 시간에 반영하지 않았습니다.
- 기존과 같이 **10시간 중 마지막 2분을 저장/종료에 남깁니다.** 스크립트 진입 기준 9시간58분에
  새 update·평가 중단을 요청하고 진행 중 작업 후 checkpoint를 저장합니다. 플랫폼의 실제 시작/마감은
  `CITYSCAPES_TI16_JOB_STARTED`/`CITYSCAPES_TI16_JOB_DEADLINE`으로 전달할 수 있습니다.
  전달되지 않으면 스크립트 실행 전 clone 시간까지 플랫폼 마감과 일치한다고 보장하지 않습니다.
- 마지막 `[CITYSCAPES_TI16_GRID2000_FINAL]`은 `pack=lg_alg`로 표시하며 16개 전부의 방법·β·상태·
  step·loss/CE/guidance·accuracy/mIoU/class IoU·가이던스 종료 epoch/step·저장 검사·실패 이유를
  담습니다. 중간 로그도 유지합니다. 종료 요약은 50,000 ASCII bytes 이내이며 긴 오류를 넣은
  16개 결과 검사에서도 약 41,400 bytes로 마지막 65,000자 내에 들어갔습니다.
  한 후보의 수치 오류는 기록한 뒤 다음 후보를 실행하며, 시간 중단이면 나머지는 `not_run`입니다.

출력: `/app/output/cityscapes_ti16_lg_alg_grid2000_v1/run_<UTC>_<PID>/`.
원본 정밀도 결과는 `artifacts/grid_summary.json`, 마지막 요약은 `artifacts/terminal_summary.log`에
남습니다. 준비 실패 시에는 출력 루트의 `terminal_summary.log`에 계획된 16개와 실패 상태를 남깁니다.

중단 후 이어서 실행할 때는 해당 후보 폴더를 다음 컨테이너에서 읽을 수 있게 복원하고 **동일 commit**에서
`CITYSCAPES_TI16_START_RUN`을 실행 순서 번호 1~16으로 지정합니다(예: 6=ALG b3, 16=ALG b8).
`CITYSCAPES_TI16_RESUME`에는 그 실행의 `resume.json` 전체 경로를 지정합니다. 이 pointer는 첫 실행에만
적용하고 나머지는 새로 학습합니다. 앞에서 완료한 실행은 반복하지 않습니다. checkpoint가 없으면
RESUME을 비우고 해당 실행부터 새로 학습합니다. 재개 이슈의 결과는 앞선 이슈 결과와 합쳐 해석합니다.

관련 로컬 검사 **21개 통과**(LG·ALG 묶음 9개 + 기존 2k 회귀 12개). 16개 실행 순서,
ALG controller 종료, 후보 실패 후 계속 실행, 정확한 실행 위치 재개, CLI 부모/자식 연결,
설치 실패 종료 요약, 마지막 로그 크기와 기존 iBKD 중단/재개 경로를 검사했습니다.

## 완료 후 보류: β=2.5의 LG·iBKD λ0.25 500step 진단

#843에서 **LG β=2.5와 iBKD λ=0.25·β=2.5 두 개 모두 0→500step** 완료했습니다.
마지막 CE는 각각 1.83563/2.03634, val2 mIoU는 6.48731%/7.03973%였습니다.
500step 안정성은 통과했지만 기존 낮은 β 후보보다 초기 CE와 간이 평가가 나빠 사용자 결정으로
추가 학습을 보류했습니다. 장기 학습의 열세를 확정한 결과로 해석하지 않습니다.
아래 명령과 설정은 실행 기록용으로 남깁니다.
[고정 설정](configs/high_beta500_v1.json)은 아래 #842에서 100step을 통과한 두 조건을 사용합니다.
이전 checkpoint는 필요하지 않으며, iBKD λ0.5와 비율 환산 β들은 이번 실행에 포함하지 않습니다.
이번 두 조건만 실행하는 범위는 사용자가 명시적으로 정한 것이며, 기존 24후보 설정의 두 λ는 유지합니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_high_beta500.sh
```

- Tiny·OpenMMLab DeepLabV3-R101 teacher·crop512·batch8·seed1·FP32·SGD LR0.01·80k scheduler와
  기존 입력 순서/초기값 설정을 유지합니다. 학습률 warm-up과 gradient clipping은 추가하지 않습니다.
- #842와 같은 매 update 수치 검사를 유지합니다. NaN/Inf나 실행 오류가 발생하면 해당 후보를
  중단하고 남은 후보를 실행합니다. 유한한 loss 급등이나 단일 CE 값으로 자동 탈락시키지 않습니다.
- 학습 로그는 25step마다 출력합니다. 각 후보의 모든 step loss/CE/guidance/gradient와 입력 해시는
  `steps.jsonl`에 저장하며, 최종 JSON에도 **100·250·500step 수치**를 `milestones`로 남깁니다.
  실패로 도달하지 못한 milestone은 `recorded=false`와 `null` 값으로 표시합니다.
- 0/250/epoch 경계/500step 및 시간 중단 시 학습 상태를 저장합니다. 매 step 전체 gradient 해시는
  계산하지 않습니다. 끝에서 teacher 동결 상태와 checkpoint 저장/복원 동일성을 검사합니다.
- **고정 500step에서 기존 val 2장**의 pixel accuracy·mIoU·19 class IoU를 진단용으로 기록합니다.
  전체 val500 성능 비교나 β 최종 순위로 해석하지 않으며, 2k/10k 자동 진입은 없습니다.
- 마지막 `[CITYSCAPES_TI16_HIGH_BETA500_FINAL]`에 두 조건의 완료/실패 상태, 선택 step/epoch,
  loss·CE·guidance·β×guidance/CE·gradient·milestones·평가값·checkpoint 검사·실패 이유를 모읍니다.
  기존 50,000-byte 상한을 유지하여 마지막 65,000자 안에서 결과를 확인할 수 있습니다.
- 출력은 `/app/output/cityscapes_ti16_high_beta500_v1/run_<UTC>_<PID>/`입니다.
  이전 500step 실측은 LG 약 8분, iBKD 약 17분으로, 준비/간이 평가를 포함해 약 30~40분 예상입니다.
  10시간 예산 중 마지막 2분을 저장/종료에 남기며, 기본 기준은 스크립트 진입 시각입니다.
  전달된 `CITYSCAPES_TI16_JOB_STARTED`가 있으면 그 시각을 보존합니다.
  실제 #843의 학습·저장 구간은 LG 462.005초, iBKD 1,017.03초였습니다(준비/평가 제외).

로컬 검사 23개(새 500step 검사 5개, 기존 high-beta100 7개, grid/report 11개)를 통과했습니다.
두 후보 실행·첫 후보 실패 후 계속 진행·milestone 보존·준비 실패 시 종료 JSON·기존 24후보
보고 호환성을 검사했으며, 실제 H200 학습 안정성을 미리 보장하는 검사는 아닙니다.

## 추가 진단: β=2.5와 분류 비율 환산값의 100-step 검사

사용자 요청에 따라 아래 **6개를 각각 새로 초기화해 최대 100 update** 검사합니다.
기존 후보 결과를 대체하지 않는 별도 진단입니다. 사용자 제공 **#842 로그**에서
LG β=2.5와 iBKD λ0.25·β=2.5의 100step 완료를 확인했습니다.
LG β≈13.59는 5step에서 NaN, iBKD λ0.5·β=2.5는 47step에서 Inf로 중단됐습니다.
iBKD λ0.25·β≈35.72 및 λ0.5·β≈53.43은 4step 기록 뒤 `SIGSEGV(-11)`로 종료됐습니다.
두 경우 loss가 이미 10¹⁸ 규모였으나, 직접적인 native 오류 원인은 로그만으로 확정하지 않습니다.

| 방법 | 숫자 β 그대로 | CIFAR-100 초기 손실 비율 환산 β |
|---|---:|---:|
| LG | 2.5 | 13.588629717453085 |
| iBKD λ0.25 | 2.5 | 35.715049953558584 |
| iBKD λ0.5 | 2.5 | 53.428435805136004 |

[고정 설정](configs/high_beta100_v1.json)에 #841의 원본 로그 SHA-256, 분류 코드 commit,
초기 25배치 무업데이트 비율, 당시 Tiny의 25batch 목표 3% 기준 β를 기록했습니다.
당시 환산식은 `분류 초기비율 / 0.03 × Tiny의 25batch 목표 3% 기준 β`입니다.
이 비율은 손실값의 비율이며 gradient 영향력이나 최적 β를 뜻하지 않습니다.
LG·ALG가 동일하게 가이던스를 사용하는 초기 구간이므로 ALG를 중복 실행하지 않습니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_high_beta100.sh
```

모델·OpenMMLab teacher·crop512·batch8·seed1·초기값·입력 순서·학습률 0.01·SGD·80k scheduler는
기존 Tiny와 동일합니다. 학습률 워밍업이나 gradient clipping을 추가하지 않습니다.
업데이트마다 parameter/optimizer 상태도 검사하며, NaN/Inf는 해당 후보를 중단하고 다음
후보로 넘어갑니다. 값이 유한한 일시적 급등만으로 자동 중단하지 않습니다.
실패 step의 loss/CE/guidance/gradient와 실패 단계는 `failure_observation`에 보존합니다.
실패 직전까지 완료한 정상 step 수와 실패를 시도한 step을 구분합니다.

완료 후보는 기존처럼 val 2장의 pixel accuracy·mIoU·class IoU를 연결 진단용으로만 출력합니다.
val500 평가·후보 순위 선정·500/2000step 자동 진입은 없습니다. 모든 후보를 포함한 마지막
`[CITYSCAPES_TI16_HIGH_BETA100_FINAL]`은 50,000 bytes 이내로 제한합니다.
한 후보가 실패하면 종합 상태 `needs_review`와 파이프라인 오류 표시가 나올 수 있으므로
여섯 개별 후보의 `status`, `completed_steps`, `attempted_step`, `failure_observation`을 확인합니다.
출력은 `/app/output/cityscapes_ti16_high_beta100_v1/run_<UTC>_<PID>`에 분리하고,
설치·준비를 포함해 실행 시작 9시간58분에 중단 요청을 보냅니다.

## 완료한 λ0.25 2,000-step 후보 비교와 실행 기록

**iBKD λ=0.25, β 8개 × 2,000step + 후보별 전체 val500 평가는 모두 완료**됐습니다.
1~7번은 v1, 8번은 v2의 새 실행(#844) 결과입니다. 이후 재실행의 기본 설정은 v2입니다.
[새 실행 설정](configs/beta_grid2000_ibkd_l025_v2.json)은 v1에서 시간 예산만 변경합니다.
2026-09-26 사용자 결정: **10시간 기준 마지막 2분만 저장/종료에 남겨 9시간 58분에 자동 중단**합니다.
기존 9시간45분 별도 상한과 9시간42분 중단은 새 실행에 적용하지 않습니다.
[v1 설정](configs/beta_grid2000_ibkd_l025_v1.json)은 기존 결과 식별용으로 그대로 보존합니다.
아래 명령은 8개를 처음부터 다시 학습하는 실행 기록입니다. **현재 이 묶음을 다시 돌릴 필요는 없습니다.**
LG·ALG 각 8개와 λ0.5의 1·3·6·8번 4개 묶음도 완료됐으며 위 결과 표에 기록했습니다.
LG·ALG는 2k 구간에서도 controller 동작으로 갈라질 수 있어 각각 실행했고, 이번에는 종료가 일어나지 않았습니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_ibkd025.sh
```

- β는 기존 후보 순서대로 `0.02461834122158468`, `0.04923668244316936`,
  `0.07385502366475404`, `0.09847336488633872`, `0.14771004732950807`,
  `0.19694672977267744`, `0.29542009465901614`, `0.3938934595453549`입니다.
  β를 재측정하거나 재계산하지 않습니다.
- 각 후보를 seed1의 동일한 공개 Tiny encoder/초기 decoder, 동일 adapter 초기값,
  같은 OpenMMLab DeepLabV3-R101 teacher와 입력 순서로 **0→2,000step 새로 학습**합니다.
  이전 컨테이너의 500step checkpoint는 필요하지 않습니다. Crop512·batch8·decoder1·FP32,
  iBKD CPU 결정성 경로·학습 CPU thread1·SGD LR0.01·80,000step LR 스케줄을 유지합니다.
- **고정 2,000step endpoint에서 전체 fine val 500장을 한 번 평가**합니다.
  평가 경로는 시간 측정을 통과한 원본 해상도·window/stride512·window batch1·CPU thread4입니다.
  당시 실행에서는 accuracy를 1차 비교값으로 선언하고 같은 endpoint의 mIoU와 19 class IoU를 함께 기록했습니다.
  현재 후속 후보 선정 기준은 위에서 정정한 mIoU입니다.
  중간 val 최고점을 선택하지 않으며, 순위나 후속 후보를 자동 확정하지 않습니다.
- 2,000step은 5epoch 완료 + 6번째 epoch의 140batch(약 5.38epoch)입니다.
  iBKD warm-up20epoch를 유지하므로 이번 단계는 **가이던스가 켜진 구간의 β 비교**입니다.
  10k·80k 결과나 가이던스 종료 후 성능으로 해석하지 않습니다.
- 기존 500step 학습 실측 ×4 + 후보마다 val107.7초 + 준비10분으로 **약 9시간 25분**을
  예상합니다. 이번 코드는 학습 연산을 유지하면서 큰 `progress.json`의 재기록을 매 step에서
  25step/epoch 경계/마지막 step으로 줄였습니다. step별 수치·입력 해시는 계속 보존합니다.
  실제 완료 시간은 실행 결과로 확인합니다.
- **10시간(36,000초)에서 저장 여유 2분(120초)만 차감**해, 9시간58분(35,880초)에
  새 update·평가를 멈춥니다. 진행 중인 update/이미지를 마친 뒤 저장합니다.
  250step마다, epoch 경계, 최종/중단 시
  모델·adapter·optimizer·scheduler·controller·RNG·부분 epoch 누적값을 원자적으로 저장합니다.
  최신/이전 두 세대를 유지하고 byte/SHA 및 저장 상태 동일성을 검사합니다.
  단일 작업/저장이 비정상적으로 오래 멈추거나 외부에서 강제 종료되는 상황까지 시간 상한을
  보장하는 것은 아닙니다. 설치·압축 해제 등 준비 시간도 위 시간 예산에 포함됩니다.
- 기본 시작점은 이 실행 스크립트 진입 시각입니다. 플랫폼의 실제 작업 시작 Unix timestamp를
  `CITYSCAPES_TI16_JOB_STARTED`로 전달하면 그 값을 보존합니다. 실제 강제 종료 Unix timestamp를
  알고 있으면 `CITYSCAPES_TI16_JOB_DEADLINE`으로 전달하며, 그 시각의 120초 전을 중단 기준으로 씁니다.
  후보마다 타이머를 다시 시작하지 않습니다. 플랫폼 시각이 없을 때는 스크립트 진입 전
  clone/컨테이너 준비 시간을 알 수 없으므로 이를 포함한 플랫폼 마감과 정확히 같다고 보장하지 않습니다.
- 제한에 걸리면 `paused`, 아직 시작하지 않은 후보는 `not_run`으로 표시합니다.
  2,000step 전 중단 또는 전체 val 도중 중단은 최종 점수를 `null`로 남깁니다.
  평가 도중 중단한 경우 2,000step checkpoint를 유지하고, 재개 시 전체 val을 처음부터 다시 평가합니다.
  NaN/Inf와 실행 오류는 별도로 표시하고 남은 후보는 시간 내에서 계속 실행합니다.

중간 학습/평가 로그를 그대로 출력하고 마지막 **`[CITYSCAPES_TI16_GRID2000_FINAL]`** JSON에
8개 후보 전체의 β·λ·완료/선택 step·선택 epoch·loss/CE/guidance·alignment/fusion·
accuracy·mIoU·19 class IoU·평가 시간·가이던스 상태·checkpoint 경로/저장 검사·실패/중단 이유를
50,000 ASCII bytes 이내로 출력합니다. 설치 실패도 마지막에 8개 계획값과 미실행 상태를 남깁니다.
`class_iou_order`와 `trajectory_order`가 해당 배열의 순서를 설명합니다.

v2 출력: `/app/output/cityscapes_ti16_ibkd_l025_grid2000_v2/run_<UTC>_<PID>/`.
종료 JSON에 `job_budget_seconds=36000`, `save_reserve_seconds=120`,
`training_stop_after_seconds=35880`과 실제 deadline을 남깁니다. 외부 마감이 더 빠르면
`training_stop_after_seconds`도 그에 맞춰 줄어듭니다. 이전 v1 결과는 기존 경로에 남습니다.
`artifacts/grid_summary.json`은 원래 정밀도의 종합 결과이고, 후보별 폴더에는 `summary.json`,
`identity.json`, `resume.json`, `checkpoints/`, `steps.jsonl`, `warnings.json`이 있습니다.
동일 조건 체크는 이번 묶음 내 초기 student/teacher/adapter와 관측한 전체 입력 prefix를 비교합니다.

중단 후 재개할 때는 **이번 2k 실행의 해당 후보 폴더를 운영진을 통해 다음 컨테이너에서
읽을 수 있는 위치로 복원**해야 합니다. `/app/output`이 다음 작업에 자동으로 보인다고 가정하지 않습니다.
처음 이슈와 **동일 commit**에서 다음을 지정합니다.

- `CITYSCAPES_TI16_START_CANDIDATE`: 이어서 시작할 후보 번호(1~8).
- `CITYSCAPES_TI16_RESUME`: 해당 후보의 복원된 `resume.json` 전체 경로.
  이 pointer는 첫 후보에만 적용하고, 뒤 후보는 새로 학습합니다. 앞서 완료한 후보는 다시 실행하지 않습니다.
- 시작 전 중단되어 checkpoint가 없으면 `CITYSCAPES_TI16_RESUME`을 지정하지 않습니다.
  실행한 나머지 후보만 새 종합 결과에 기록하므로 앞선 이슈 결과와 함께 해석합니다.
- config·코드·데이터·runtime·방법/β가 달라지면 재개를 거부합니다. 500step checkpoint를
  이 경로로 임의 승격하지 않습니다. 강제 종료 직전 step까지 항상 저장됐다고 가정하지 않습니다.

관련 로컬 검사 **86개 통과**(신규 2k 검사 10개 + 기존 76개).
작은 CPU 모델에서 실제 공유 학습 루프와 checkpoint 코드를 실행하여, 부분 epoch 중단/재개와
연속 실행의 입력·student/adapter·SGD momentum·scheduler·controller·평가값 일치를 검사했습니다.
2k endpoint에 저장 후 평가만 재실행하는 경로, 후보 실패 후 계속 진행, 시간 중단 후 미실행 표시,
SIGTERM 전달, 설치/child 초기화 실패 종료 로그도 검사했습니다. 이는 v1 구현 당시의 검사 기록입니다.
v2 시간 변경은 이전 중단 시각을 지나 계속 실행하고 9시간58분에 중단하는 경계,
부모/자식의 공통 deadline, 외부 작업 시작 시각 유지 및 v1/v2 종료 보고를 별도로 검사합니다.

**아래는 완료한 전체 val500 시간 측정 v2 기록**입니다.
사용자 제공 #838 로그에서 `passed`, 500장, **107.666초**, 스크립트 전체 **448.051초**,
데이터 준비 **309.948초**, 학습 update0·가중치 무변화·test 미사용을 확인했습니다.
이 측정은 초기 decoder의 시간 확인용이며, mIoU0.6363%와 accuracy1.2507%는 학습 성능이 아닙니다.
[고정 설정](configs/val500_timing_initial_v2.json)을 사용합니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_val500_timing_initial.sh
```

- 기존 Tiny와 같은 **공개 AugReg ImageNet-21k→1k 사전학습 encoder + 학습 전 decoder**를
  seed1로 구성합니다. 이전 Cityscapes `resume.json`이나 학습 가중치는 필요하지 않습니다.
  새 학습·optimizer update는 0회이며 teacher와 guidance module을 생성하지 않습니다.
  teacher 사전학습 파일도 다운로드하지 않습니다. 본 증류 실험의 OpenMMLab teacher 선택은 그대로입니다.
- 공개 Tiny encoder NPZ는 23,226,422 bytes,
  SHA-256 `4b99893dc1a5a2a7d9ad119671c20559850f865e1fa17ed23401a3fefa7fedc9`로 검사합니다.
  고정 upstream 코드와 공식 가중치는 캐시에 없으면 내려받습니다.
- 데이터는 `CITYSCAPES_CROP512_DATA_DIR`(기본
  `/app/scratch/cityscapes_l16_crop512_v3/cityscapes`)의 manifest와 val 파일을 재사용합니다.
  manifest가 없으면 `CITYSCAPES_ZIP_DIR`(기본 `/app/data/chaoyang`)의
  `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`을 기존 업로드 검사와 같은 byte/SHA로
  확인한 뒤 train/val을 압축 해제하고 감사합니다. 기존 파일 내용이 다르면 덮어쓰지 않고 실패합니다.
  원본 데이터 다운로드나 test 압축 해제는 하지 않습니다. val 실제 이미지/정답의 byte/SHA는 평가 전에 검사합니다.
- 기존 Tiny·decoder1·FP32·원본 해상도·window/stride512·window batch1·CPU thread4를 유지합니다.
  공식 fine val 500장을 정렬된 순서대로 딱 한 번 평가하고 test는 사용하지 않습니다.
  가짜 데이터나 해상도 축소로 시간을 추정하지 않습니다.
- `validation_seconds`는 첫 이미지 초기 비용·이미지 읽기·변환·전송·추론·지표 집계를 포함합니다.
  SHA 검사에서 파일을 먼저 읽으므로 OS 파일 캐시가 어느 정도 채워진 조건의 측정입니다.
  반복 학습 사이에 수행하는 같은 평가 코드의 시간 예산에 사용하며 cold-cache 최악 시간은 아닙니다.
- `total_job_seconds`에는 실행 스크립트 시작부터 설치·asset 준비·데이터 감사·모델 로딩·평가까지
  포함합니다. 큐 대기·외부 컨테이너 생성·git clone은 포함하지 않습니다.
- 25장마다 진행 상황을 출력하고 마지막 **`[CITYSCAPES_TI16_VAL500_TIMING_FINAL]`**에
  평가/전체 시간, 처리속도, 메모리, accuracy·mIoU·19 class IoU·유효 픽셀 수,
  `selected_step=0`, `selected_epoch=0`, `optimizer_updates=0`,
  `trained_checkpoint_loaded=false`, 가중치 무변화 및 실행 상태를 출력합니다.
  학습 loss/CE/guidance·β·λ·checkpoint는 `null`입니다.
  **accuracy·mIoU·IoU는 학습 전 decoder의 진단값입니다. 학습한 Vanilla 성능이나 후보 순위에 쓰지 않습니다.**
  동일 구조·입력·평가 경로의 소요 시간을 측정하여 후속 2k 작업의 시간 예산에 사용합니다.
  실제 학습 도중의 GPU 부하·파일 캐시 상태에 따라 시간이 달라질 수 있으므로 여유를 둡니다.

출력은 `/app/output/cityscapes_ti16_val500_timing_initial_v2/run_<UTC>_<PID>/`의 `summary.json`,
`terminal_summary.log`, `per_image_timings.json`, `validation_ids.json`, `warnings.json`,
`asset_provenance.json`, `data_setup.json`입니다. `CITYSCAPES_TI16_VAL_OUTPUT`으로 출력 위치를 지정할 수 있습니다.
설치 실패도 마지막 JSON으로 보고하며, 강제 종료로 출력 기회가 없었던 경우까지 보장하지는 않습니다.
실제 H200 val500 시간은 위 #838 결과에서 확인했습니다.
새 경로 검사 6개와 기존 관련 검사 70개, 총 **로컬 검사 76개 통과**했습니다.
checkpoint·teacher·optimizer 없이 평가하는 분기, ZIP 손상 시 추출 전 중단,
기존 데이터 재사용, 평가 조건 일치, 초기 가중치 무변화 및 설치 실패 종료 JSON을 검사했습니다.
이 검사는 CPU의 작은 모델과 모의 GPU 경로를 포함하며 실제 H200 평가 시간을 대신하지 않습니다.

이전 checkpoint 평가 [v1 설정](configs/val500_timing_v1.json)과 `run_val500_timing.sh`는 기록용으로
보존합니다. 사용자 제공 #837 로그는 이전 컨테이너의 `lg_b1/resume.json`이 없어 preflight에서
실패했고, 평가 0장·학습 update 0회였습니다. 500step 학습 결과 자체가 실패한 것은 아닙니다.
v1은 여전히 검증된 checkpoint를 요구하며, v2로 자동 전환하지 않습니다.

**아래는 완료한 500-step 후보 검사 v6(24개)의 실행 기록**입니다.
2026-09-26 사용자 결정에 따라 **iBKD는 λ=0.25와 λ=0.5를 항상 함께 구성**합니다.
LG 8개 + iBKD λ0.25 8개 + iBKD λ0.5 8개이며,
[고정 config](configs/beta_grid500_shared_lg_8betas_v6.json)의 24개를 순서대로 실행합니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_beta_grid500.sh
```

LG는 **LG·ALG 공통 가이던스 구간의 사전 검사**입니다. 같은 β·초기값·입력에서 두 방법의
학습 연산은 같고, ALG는 가장 빨라도 744step(2epoch 완료)에 종료를 결정해 745step부터
가이던스를 끕니다. 500step에서는 ALG를 중복 실행하지 않으며 별도의 ALG 결과를 만들지 않습니다.
2k 단계에서는 LG와 ALG를 각각 구성해야 합니다. LG 체크포인트의 controller 종류를 이름만
바꿔 ALG로 재개하지 않습니다.

| 후보 | 당시 25batch 목표(기록) | LG β | iBKD λ0.25 β | iBKD λ0.5 β |
|---|---:|---:|---:|---:|
| 1 | 1.5% | 0.00932921 | 0.0246183 | 0.0366835 |
| 2 | 3% | 0.0186584 | 0.0492367 | 0.0733669 |
| 3 | 4.5% | 0.0279876 | 0.073855 | 0.11005 |
| 4 | 6% | 0.0373168 | 0.0984734 | 0.146734 |
| 5 | 9% | 0.0559752 | 0.14771 | 0.220101 |
| 6 | 12% | 0.0746336 | 0.196947 | 0.293468 |
| 7 | 18% | 0.11195 | 0.29542 | 0.440202 |
| 8 | 24% | 0.149267 | 0.393893 | 0.586936 |

위 목표값은 과거 후보 생성 기록입니다. **현재 사용하는 첫 step CE 비율은 아래와 같습니다.**

| 후보 | LG·ALG 첫 step 비율 | iBKD λ0.25 첫 step 비율 | iBKD λ0.5 첫 step 비율 |
|---:|---:|---:|---:|
| 1 | 1.461% | 1.467% | 1.466% |
| 2 | 2.923% | 2.933% | 2.932% |
| 3 | 4.384% | 4.400% | 4.399% |
| 4 | 5.845% | 5.866% | 5.865% |
| 5 | 8.768% | 8.799% | 8.797% |
| 6 | 11.690% | 11.732% | 11.730% |
| 7 | 17.535% | 17.598% | 17.594% |
| 8 | 23.380% | 23.465% | 23.459% |

비율은 남아 있는 첫 step 로그에서 환산한 근삿값입니다. β는 JSON의 전체 정밀도를 유지합니다.
#834의 초기 25batch CE/guidance 중앙값으로 계산한 β₀에 `[0.5,1,1.5,2,3,4,6,8]`을 곱합니다.
기존 3/6/12/24% 후보에 1.5/4.5/9/18%를 추가했습니다. 초기 손실 크기에 대한 탐색 기준이며
학습 내내 유지되는 비율이나 논문의 표준값은 아닙니다. β를 재계산하지 않습니다.
추가 25-step 학습·calibration 없이 바로 500 update하며 Vanilla/FSKD*/C2VKD*는 실행하지 않습니다.
모델·데이터·학습 조건은 v4와 동일하고, 매 step 입력 해시는 기록하되 전체 gradient 해시는 계산하지 않습니다.

- `train2975`, 실제 batch8, 마지막 batch7을 포함해 **372 step=1epoch**입니다.
  500step은 1epoch 완료 + 2epoch의 128 batch, 총 3,999 sample 관측입니다.
  같은 데이터를 다른 epoch에서 반복 관측한 수이며 고유 이미지 수는 아닙니다.
- 완료 epoch의 실제 sample 수로 가중 평균한 guidance만 controller에 전달합니다.
  ALG warm-up0, iBKD20epoch 조건은 유지하며 부분 epoch를 완료 처리하지 않습니다.
- SGD LR0.01 및 **80,000-step schedule**을 유지하고 500에서 멈춥니다.
  다음 2k/10k/80k를 자동 실행하지 않습니다.
- NaN/Inf loss·gradient·parameter/optimizer state는 해당 후보를 중단합니다.
  OOM·입력 오류 등은 별도 runtime failure입니다. 낮은 진단 mIoU나 유한한 loss 급등만으로
  자동 중단하지 않으며, 실패 후보를 영구 제외하거나 우수 후보를 자동 선정하지 않습니다.
- 후보마다 별도 프로세스에서 새로 학습합니다. 실패해도 나머지 후보를 계속 실행합니다.
- 끝에서 고정 val2장의 accuracy·mIoU·19 class IoU를 계산합니다. **전체 val500 평가가 아니며
  성능 순위 선정용이 아닙니다.** 2k 단계의 전체 val 비교는 별도 구성해야 합니다.

출력: `/app/output/cityscapes_ti16_beta_grid500_shared_lg_8betas_v6/run_<UTC>_<PID>/`.
마지막 **`[CITYSCAPES_TI16_GRID500_FINAL]`** JSON은 24개 전체 결과를 담고,
접두사·줄바꿈을 포함해 **50,000 ASCII bytes 이하**로 제한합니다. 따라서 문자 수도 같습니다.
서버가 마지막 65,000자를 제공하면 후속 출력에 15,000자 여유가 있습니다.

- 각 후보의 β·λ·초기 목표 비율·완료/시도 step·선택 step/epoch·최종 loss/CE/guidance·
  alignment/fusion·가중 guidance/CE 비율·gradient norm·LR을 출력합니다.
- 진단 accuracy·mIoU·19 class IoU·평가 클래스/픽셀 수·가이던스 종료·teacher 고정·
  checkpoint 저장 step/검사·경고 횟수·실패 이유 요약·시간/메모리도 포함합니다.
- 초기/마지막25 중앙값·최대·최소는 `trajectory_order`에 표시된 순서입니다.
  클래스별 IoU 배열은 `class_iou_order`의 19개 클래스 순서입니다.
- 출력 수치는 6자리 유효숫자이며 β는 전체 정밀도를 유지합니다. 긴 경고 원문·상태 이력·
  파일별 SHA는 로그에서 반복하지 않고 결과 파일에 보존합니다. 드문 크기 초과 시에는
  `run_columns`와 행 배열로 키를 공유해 모든 후보와 수치 필드를 유지합니다.
- 평가하지 못한 수치는 null, 아직 시작하지 못한 후보는 `not_run`입니다.
  설치 실패 시에도 24개 계획값과 미실행 상태를 출력하며 결과를 만들어내지 않습니다.
- `artifacts/grid_summary.json`에는 원래 정밀도의 종합 결과와 경고를,
  후보별 `summary.json`/`steps.jsonl`/`warnings.json`에는 상세 이력을 저장합니다.
  `artifacts/terminal_summary.log`는 최종 출력 복사본입니다. Shell 실패 처리 시 출력 루트에도 남깁니다.

24개 모두 finite 500step 완료 및 공통 초기값/입력 비교를 통과하면 `passed`,
실패 후보나 비교 불일치가 있으면 `needs_review`와 전체 결과를 남깁니다.
강제 종료로 프로세스가 로그를 쓸 기회가 없었던 경우까지 종료 JSON을 보장하는 뜻은 아닙니다.

각 후보의 `resume.json`과 `checkpoints/`는 최신·이전 두 세대를 보관합니다.
0/250/epoch 경계/500step에 모델·guidance·optimizer·scheduler·controller·RNG·입력 진행 위치·
부분 epoch 누적값을 저장하고 bytes/SHA를 기록합니다. 완료 시 저장 상태를 엄격 비교합니다.
중단된 동일 config/코드 실험은 `tiny_grid --run-id <id> --resume <resume.json>`으로
새 출력 폴더에서 500까지 이어갈 수 있습니다. Dataset/asset 검증을 동일하게 수행한 환경에서 사용합니다.
후속 2k 전환은 **새 단계의 프로토콜과 재개 호환성을 먼저 고정**해야 하며,
현재 CLI의 config identity 검사를 임의로 우회하지 않습니다.

v6 관련 로컬 검사 **63개 통과**. 24개 전체 지표·긴 경고/오류를 넣은 성공/부분 실패/전체 실패
종료 로그는 각각 **36,876 / 38,945 / 41,010 bytes**였습니다.
마지막 65,000자만 남겨도 24개 결과를 모두 읽을 수 있는지 확인했습니다.
에폭 경계·부분 epoch 재개·후보 실패 후 나머지 실행·설치 실패 출력도 검사했습니다.
이는 실제 H200 500step 결과를 미리 보장하는 의미가 아닙니다.

이전 16개 구성은 [v5 config](configs/beta_grid500_both_lambdas_v5.json)에 보존합니다.
기존 이슈의 고정 commit `3e3d4b93188a9b474d4ad2fc76c8cf6140423c18`은 v5 그대로 실행되므로
24개 실행은 새 commit의 이슈 명령을 사용해야 합니다.

**이전 재현성 진단 v4는 H200에서 통과했습니다.** LG-A/LG-B/ALG의 25step 입력·RNG·logits·
gradient·update 후 state 해시가 모두 동일했고, CE scalar의 약 7.15e-7 이하 차이는 허용 오차 내였습니다.
아래는 해당 v4 실행 기록입니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_repeat25_lg_alg.sh
```

[repeat25_lg_alg_v4.json](configs/repeat25_lg_alg_v4.json)에 고정한 세 경로는
`lg_a`, `lg_b`, `alg`이며 각각 별도 Python 프로세스에서 실행합니다.
β는 #834 LG의 초기 제안값 `0.018658411532808426`으로 **모두 완전히 동일하게** 사용합니다.
초기 25 batch 손실 측정은 유지하되 β를 다시 계산하거나 선택하지 않고, 상태를 복원한 뒤
각각 25 update합니다. Tiny/teacher·crop512·batch8·seed1·SGD·증강·FP32 및
80k schedule은 기존과 같습니다. iBKD/FSKD/C2VKD는 이번 진단에서 실행하지 않습니다.

각 step에 입력, Python/NumPy/Torch CPU·CUDA RNG, logits, 전체 student/adapter gradient,
update 후 state의 SHA-256을 기록합니다. 이러한 진단 때문에 이번 step 시간은 일반 학습
속도 추정에 쓰지 않습니다. `warn_only=True`를 유지해 경고가 나도 관찰을 계속하고,
경고 원문·횟수·소스 위치를 성공/실패 모두 마지막 JSON에 남깁니다.

최종 `repeat_comparisons.pairs`에는 LG-A↔LG-B, LG-A↔ALG, LG-B↔ALG **모두** 기록됩니다.
최초 scalar 차이 step·양쪽 값·허용 오차·최대 차이 및 항목별 최초 hash 차이 step을 표시합니다.
입력/초기값/β/RNG가 동일해야 하며, scalar 비교는 기존 `rtol=2e-5, atol=2e-6`를 유지합니다.
출력/gradient/state의 hash 일치는 별도 엄격 진단입니다. Hash가 다르지만 scalar 오차가
허용 범위 내인 경우에도 bitwise 재현이라고 해석하지 않습니다.
세 실행과 비교가 모두 통과해야 전체 `status=passed`입니다. 차이가 나면 전체 failed와
세 쌍의 진단을 남기고, 이후 500/2k/10k 학습을 자동 실행하지 않습니다.

각 실행의 loss·선택 step/epoch·진단 val2 accuracy/mIoU/IoU·재개 검사도 기존처럼 출력합니다.
25step은 1epoch를 채우지 않아 자연스러운 controller 종료는 검사하지 않습니다.
출력 위치: `/app/output/cityscapes_ti16_crop512_smoke25_repeat_v4/run_<UTC>_<PID>/`.
경로별 `summary.json`/`training_progress.json`에 모든 step trace,
`diagnostic_initial.json`에 초기값, `warnings.json`에 경고를 남깁니다.
`artifacts/smoke_summary.json`과 마지막 `[CITYSCAPES_TI16_SMOKE_FINAL]`에 전체 판단이 있습니다.

**아래는 이전 C2VKD* 추가 v3(7경로)의 기록**입니다.
Vanilla, LG, ALG, iBKD λ0.25, iBKD λ0.5, FSKD*, C2VKD*를 순서대로 검사합니다.
고정 손실과 출처는 [FSKD_PROTOCOL.md](FSKD_PROTOCOL.md),
[C2VKD_PROTOCOL.md](C2VKD_PROTOCOL.md), 실행값은
[smoke25_fskd_c2vkd_v3.json](configs/smoke25_fskd_c2vkd_v3.json)에 있습니다.

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_smoke25_fskd_c2vkd.sh
```

각 경로에서 무업데이트 calibration25 batch 후 상태를 복원하고 25 update를 합니다.
LG/ALG/iBKD의 β 4개는 **제안만 하고 최솟값 하나로** smoke합니다.
FSKD*와 C2VKD*는 고정 계수를 쓰며 β 탐색에 포함하지 않습니다.
FSKD*는 공개 DeiT-Ti 분류 예제를 segmentation에 이식한 구성입니다.
C2VKD*는 원본의 미공개 pooling 가중치를 CLIP RN101으로 대체하여 추가 사전학습
정보를 사용하므로 `supplementary_extra_pretraining` 결과로 별도 표시합니다.
두 방법 모두 저자 Cityscapes 설정의 완전 재현이라는 뜻은 아닙니다.

v3는 torchsort0.1.10 CUDA 확장을 설치하고 CLIP RN101 파일(291,791,292 bytes)을
다운로드·해시 검증해 **attention pool만** 사용합니다. OpenMMLab teacher는 유지합니다.
출력은 `/app/output/cityscapes_ti16_crop512_smoke25_fskd_c2vkd_v3/run_<UTC>_<PID>/`입니다.
`artifacts/smoke_summary.json`과 마지막 `[CITYSCAPES_TI16_SMOKE_FINAL]`에
7개 경로의 loss, 고정 계수/β 후보, 완료 step, 진단 pixel accuracy·mIoU·19 IoU,
재개 검사, teacher/pool 고정, 시간·메모리, 비교군 구분을 기록합니다.
**val 2장 점수는 연결 진단용이며 성능 순위를 정할 수 없습니다.**
통과 조건은 7경로 모두 passed 및 공통 초기화·입력 교차 검사 통과입니다.

이전 5경로 v1과 FSKD 추가 6경로 v2는 설정 파일과 실행 스크립트를 보존합니다.
아래의 5경로 상세 설명·기본 `run_smoke25.sh` 명령은 **v1 기록**입니다.
v2 전용 명령은 `run_smoke25_fskd.sh`이며, v2의 C2VKD 보류 표시는 과거 결정입니다.

## 고정 조건

- Teacher: 기존 MMSeg DeepLabV3 ResNetV1c-101-D8 Cityscapes 80k 체크포인트.
  348,988,299 bytes, SHA-256 `9e428899b279f29964cec79ab21bb19193328b8c4d42c0db49ff9070e9ab3b2d`.
  가중치와 BN 통계를 고정하며 CIRKD teacher로 교체하지 않습니다.
- Student: 공식 Segmenter `vit_tiny_patch16_384`, 12블록·192채널, patch16,
  mask-transformer decoder 1블록(폭 192). 깊이는 기존 L/16 비교 조건에서 가져왔습니다.
- Tiny 초기화: timm 0.4.12가 지정한 Google ViT AugReg ImageNet-21k→1k 사전학습 NPZ.
  23,226,422 bytes, SHA-256 `4b99893dc1a5a2a7d9ad119671c20559850f865e1fa17ed23401a3fefa7fedc9`.
  CNN으로 사전 증류한 DeiT checkpoint를 사용하지 않습니다. Tiny와 Large의 AugReg
  사전학습 세부 증강·정규화가 완전히 같다는 주장은 하지 않습니다.
- Cityscapes fine train 2,975 / val 500 / 19 class, test·coarse 미사용.
  원본 MMSeg augmentation·전처리와 L/16의 seed별 입력 순서를 재사용합니다.
- crop/window/stride 512, 실제 batch8, seed1, FP32, TF32/AMP/gradient clipping 없음.
- SGD Nesterov LR0.01, momentum0.9, WD0, poly power0.9, minLR1e-5,
  **80,000-step schedule의 처음 25 update**. LR warm-up 없음.
- LG/ALG: 0-based block `[0,6,11]`, iBKD: 12블록 전체 aggregation.
- ALG guidance warm-up0, iBKD warm-up20epoch, window50, threshold−0.02 유지.
  iBKD는 L/16에서 검증한 `flatmax_cpu_deform_v1` 연산을 재사용합니다.
- Smoke는 DataLoader worker0을 사용합니다. 원래의 sample별 독립 증강 seed와 순서는
  유지하지만 step 시간은 이 smoke 실행 조건의 측정값입니다.

이 설정은 **우리 L/16 crop512 조건을 Tiny로 이식한 실험**입니다.
저자가 공개한 Tiny 전용 Cityscapes 성능의 재현이나 Tiny의 최적 설정을 주장하지 않습니다.
공통 설정 원본은 [L/16 v19](../phase4_cityscapes/configs/paper_l16_crop512_final80000_v19.json),
이번 실행 값은 [smoke25_v1.json](configs/smoke25_v1.json)입니다.

## 이번 smoke의 범위

실행 경로는 Vanilla, LG, ALG, iBKD λ=0.25, iBKD λ=0.5의 5개입니다.

1. 각 경로가 같은 초기 student와 같은 train batch25개(200장)를 사용합니다.
2. Optimizer update 없이 train mode에서 픽셀별 CE와 raw guidance를 측정합니다.
   측정 후 모델·adapter buffer와 RNG를 복원합니다. Teacher는 계속 eval/frozen입니다.
3. `β₀ = 0.03 × median(CE) / median(guidance)`로 시작값을 계산하고
   `[β₀, 2β₀, 4β₀, 8β₀]`를 **제안 후보**로 기록합니다. 실제 batch별 가중 guidance/CE
   비율도 기록합니다. 이는 최적성이나 gradient 영향력 동등성을 보장하는 공식이 아닙니다.
4. 각 KD 경로는 β₀ 하나로 25 update, Vanilla는 CE만으로 25 update합니다.
   이 실행에서는 4개 β를 각각 학습하지 않습니다. λ별 β는 독립적으로 계산합니다.
5. 24번째 update 후 전체 상태를 저장한 뒤, 25번째 update를 재개해 모델·adapter·
   optimizer·schedule·입력을 비교합니다. 이는 1-update 재개 검사이며 장기 재현성 검사는 아닙니다.
6. 고정 val 2장을 원본 해상도에서 sliding window로 평가하고 pixel accuracy·mIoU·
   19 class IoU를 기록합니다. **진단 점수이며 β나 방법의 순위를 고르는 데 쓰지 않습니다.**
7. 방법 간 student 초기값·입력·teacher 일치, LG/ALG의 25-step 궤적 일치를 검사합니다.
   iBKD λ 두 조건의 adapter 초기값도 동일해야 합니다.

25 step은 한 epoch(372 step)를 채우지 않으므로 controller에 가짜 완료 epoch를
전달하지 않습니다. 자연스러운 guidance 종료는 이 smoke에서 검사하지 않습니다.
별도의 합성 loss 검사로 controller 경계·상태 복원을 확인하고 구분해 기록합니다.

## 실행과 결과

저장소 루트에서:

```bash
bash phase4/Cityscapes_Segmenter-Ti16/scripts/run_smoke25.sh
```

입력 ZIP은 `/app/data/chaoyang/`, 압축 해제 데이터는
`/app/scratch/cityscapes_l16_crop512_v3/cityscapes/`를 검증해 재사용합니다.
공식 소스·teacher 캐시는 `/app/scratch/cityscapes_official_l16_v2/upstream/`를 재사용하고
Tiny 초기 가중치만 추가합니다. 기존 L/16 asset 검증의 기본 동작은 유지합니다.

출력은 `/app/output/cityscapes_ti16_crop512_smoke25_v1/run_<UTC>_<PID>/`입니다.
재실행은 새로운 출력 폴더를 만들며 기존 결과를 덮어쓰지 않습니다.

- `artifacts/smoke_summary.json`: 다섯 경로와 교차 검사 결과, 가중치 해시.
- `artifacts/<run>/calibration.json`: 25개 초기 loss, 입력 해시, 후보 β와 실측 비율.
- `artifacts/<run>/summary.json`: update별 loss, 재개 검사, 진단 지표, 시간·메모리.
- 실패하면 `traceback.txt`, `training_progress.json`, `warnings.json` 등 남아 있는 진단 기록.

마지막 `[CITYSCAPES_TI16_SMOKE_FINAL]` JSON에 방법별 마지막 loss·SegLoss·guidance,
λ·실제 β·후보 β, 완료/선택 step·epoch, 진단 pixel accuracy·mIoU·클래스별 IoU(%),
teacher 고정·재개·입력 동일성, 시간·메모리를 함께 출력합니다. 평가하지 못한 값은 null입니다.
성공은 `status=passed`, 다섯 run 모두 passed 및 교차 검사 통과로 판단합니다.

## 실행 코드의 사전 검사

v4 로컬 검사: 관련 단위 검사 52개 통과. 실제 공식 Tiny NPZ와 32×32 합성 입력·합성
teacher feature로 LG-A/LG-B/ALG를 각각 25 update해 loss, RNG, logits, gradient,
update 후 state의 모든 비교가 CPU에서 일치함을 확인했습니다. 실패 시 경고 원문과
종료 JSON 보존, 설치 실패 시 즉시 중단도 검사했습니다. H200에서의 반복 재현성은
이 v4 이슈의 결과로 확인해야 합니다.


v3 추가 검사: 관련 단위 검사 총 45개, Python/shell 문법 및 설치 실패 시 종료 JSON 검사 통과.
공식 Tiny NPZ와 byte/SHA 검증한 실제 CLIP RN101 pool을 CPU에서 연결해, hook 전후
logit 일치·encoder/decoder/두 adapter gradient·pool 무변화 및 momentum/RNG를 포함한
저장/재개 후 동일 update 재현을 확인했습니다. 이 검사는 32×32 합성 입력과 합성 teacher
feature를 썼으며 H200/crop512/실제 teacher의 전체 실행 통과를 뜻하지 않습니다.


2026-09-26 로컬 검사: 관련 단위 검사 37개, Python 문법·shell 문법 검사 통과.
검증한 공식 Tiny NPZ를 로딩해 32×32 합성 입력에서 12개 feature와 segmentation
출력, CPU 학습 1 update 및 RNG·optimizer 복원 후 동일 update 재현을 확인했습니다.
실제 Tiny와 합성 teacher feature를 연결한 다섯 경로의 backward도 통과했습니다.
설치 단계 실패를 주입했을 때 후속 데이터 작업 없이 중단하고 마지막 JSON에 실패를
표시하는 것도 확인했습니다. **이는 H200/실제 Cityscapes smoke 통과를 뜻하지 않습니다.**

## 다음 단계

통과 후 β 제안값과 안정성 기록을 검토해 다음 500/2,000-step 후보 config를 고정합니다.
L/16에서 선정한 β를 Tiny의 검증값으로 사용하지 않습니다. 2k→10k→80k의 장기 재개
runner는 이 smoke에 포함되지 않으며, 준비되기 전 자동으로 실행하지 않습니다.
