# ALG·iBKD λ=0.25 각 1위 β · H200 10k 결과

2026-09-25 제출 로그 점검: **두 후보 모두 10,000 step과 전체 val500 평가를 완료**했습니다.
Best mIoU는 ALG 60.1973%, iBKD 60.0381%입니다. 최종 10k의 차이는 0.0104pp로 작습니다.
이번 실행만으로 방법의 우열이나 최종 β를 확정하지 않습니다.

## 결과

mIoU 단위는 %, 차이는 percentage point(pp)입니다.

| 방법 | β | guidance 보호 | Best mIoU | Best step | Last mIoU (10k) | 마지막 CE |
|---|---:|---:|---:|---:|---:|---:|
| ALG | 0.197479 | 0 epoch | **60.1973** | 8,800 | **60.0486** | 0.254960 |
| iBKD λ=0.25 | 0.387021 | 20 epoch | 60.0381 | 10,000 | 60.0381 | 0.262161 |

Best 기준 ALG가 **0.1592pp**, 같은 10k last 기준 **0.0104pp** 높습니다.
사전 선별 기준은 best val mIoU이며 last는 함께 기록하는 경과 지표입니다.
두 방법은 각각 β 하나만 실행했고 단일 seed의 중간 예산 결과입니다.
Vanilla의 B0 10k 결과는 아직 없어 증류 효과를 Vanilla 대비로 판단하지 않습니다.

전체 `status=completed`, `completed_runs=2`이고 두 후보의 `last_eval_step=10000`입니다.
`selection.status=pending`은 **방법별 두 번째 β가 아직 10k를 완료하지 않았다는 뜻**이며,
이번 두 실행의 실패나 시간 중단을 의미하지 않습니다. `selected_epoch=null`은 iteration 기반
학습이라 `selected_step`을 쓰기 때문입니다. 최상위 loss/metric의 null 대신 `runs`의 방법별 값을 읽습니다.

## Guidance 종료 확인

| 방법 | 종료 관측 구간 | guidance 사용 마지막 step | CE-only 시작 | 종료 시 평활 변화량 |
|---|---:|---:|---:|---:|
| ALG | 13 | 2,418 | 2,419 | −0.0193220 |
| iBKD λ=0.25 | 20 | 3,720 | 3,721 | −0.00494853 |

ALG는 `s >= −0.02`, iBKD는 `s > −0.02`를 만족해 종료됐습니다. iBKD는 보호 기간을
지킨 뒤 **종료 판단이 처음 허용되는 20번째 구간에서 실제 조건을 만족**했습니다.
20 epoch에서 무조건 끄도록 구현된 것은 아닙니다. `warmup_protection_passed=true`입니다.

종료 후에도 CE 학습과 val 평가는 10k까지 계속됐습니다. 마지막 loss가 CE만 포함하고
`beta=0`, `weighted_guidance=0`, `raw_guidance=null`, guide gradient=0인 것은 정상입니다.
마지막 encoder·decoder gradient는 두 후보 모두 유한하고 양수이며 LR도 80k schedule과 일치합니다.

두 controller에 저장된 관측 손실을 다시 입력해 stop epoch, β 이력, 미분·평활값을 확인했습니다.
최대 수치 오차는 `8.67e-18`입니다. 관측 53구간과 미완료 142 step은
`53×186+142=10000`으로 일치합니다. 종료 후에는 raw guidance 관측이 추가되지 않아
loss history가 ALG 13개, iBKD 20개인 것도 정상입니다.

## 시작 방식과 시간

**이번 실행은 `fresh`입니다.** `start_mode=fresh`, 두 후보 모두 `starts_from=initialization`,
`resume_from=null`이므로 2k checkpoint를 이어 쓴 것이 아니라 처음부터 각 10k를 수행했습니다.
실행 commit은 auto 기능이 포함된 `d6569369f68e5a30dc89657bc3693d6866200ca4`이지만,
이번에는 fresh 모드를 지정했습니다. **체크포인트를 탐색했으나 없었다고 해석할 근거는 없습니다.**
서버의 기존 파일 존재 여부와 자동 발견 기능의 H200 동작은 이 로그로 확인하지 못했습니다.

새 iBKD 실행의 2k mIoU는 **45.3352%**이며 이전 warm-up 20·같은 β의 confusion matrix와
완전히 같습니다. 두 방법의 첫 10구간 guidance 평균도 각각 이전 2k 기록과 정확히 일치합니다.
현재 두 후보의 마지막 학습 batch hash와 초기 입력 preflight도 기존 조건과 일치합니다.

| 구간 | 실제 소요 |
|---|---:|
| ALG | 2시간 54분 10초 |
| iBKD λ=0.25 | 4시간 50분 59초 |
| 공통 준비 등 | 8분 27초 |
| **전체** | **7시간 53분 36초** |

기존 코드의 9시간 40분 중단 예산보다 일찍 두 실행을 완료했습니다.
이는 실제 측정값이며 제한을 기준으로 시간을 정한 것이 아닙니다. 두 번째 β의 종료 시점은
다를 수 있으므로 이번 소요시간을 다음 실행의 확정 시간으로 쓰지 않습니다.

## 검증 근거와 한계

- 원본 첨부: 65,251 bytes, SHA-256 `2c1848433d57fd721a77b5c8f2962d28f8d5d005eacf3f4a0550c00a7efa187e`.
- 앞부분은 iBKD worker summary 중간부터 시작하지만, 마지막 group JSON은 완전합니다.
  표의 두 후보 best/last/loss/상태는 모두 이 JSON에서 직접 확인했습니다.
- Best·last·iBKD 2k 및 남아 있는 6,800~10,000 평가 기록의 총 14개 metric,
  서로 다른 confusion matrix 12개를 재계산했습니다. mIoU·pixel accuracy·class IoU의
  최대 절대 오차는 `1.11e-16`입니다. 모든 평가가 val500·동일한 GT 클래스별 픽셀 수이며
  유효 픽셀은 917,018,489개입니다.
- 두 후보 모두 step2→3 재실행, step500 안정성, calibration 입력, teacher 고정 검사가 passed입니다.
  iBKD의 완전한 signature는 잘린 첫 줄에 남아 있어 현재 학습 코드·고정 설정·grid와 대조했습니다.
  Python 3.10.12, PyTorch 2.11.0+cu130, H200 NVL입니다.
- ALG의 개별 signature와 양쪽 25회 전체 평가 이력은 이 첨부에 없습니다. 따라서 ALG의
  best step8800은 저장된 선택 metadata를 확인한 것이며 전체 곡선에서 다시 선별한 것은 아닙니다.
  iBKD의 남아 있는 마지막 9회 평가에서는 10k가 최고입니다. Checkpoint 실물은 직접 로딩하지 않았습니다.
- [파생 점검 JSON](log_audit.json)에 후보별 metric·controller·시간·검증 결과와 확인 범위를 보존했습니다.

## 다음 단계

기존 상위 두 후보 계획대로 남은 **ALG β=0.460784**, **iBKD λ=0.25·β=1.9351**을
같은 10k 예산에서 비교한 뒤 방법별 최종 β를 고릅니다. iBKD 보호 20·ALG 보호 0을 유지합니다.
이번 두 10k를 다시 제출할 필요는 없습니다. 새 이슈나 GPU 작업은 이 점검에서 시작하지 않았습니다.

`/app/output/cityscapes_b0_top1_10k_v1/`의 `group_summary.json`, `run.log`, `runs/`와
후보별 `resume.json`, `checkpoints/` 등 **전체 출력**을 보존합니다. 후속 80k 재개에는 best
student만이 아니라 마지막 10k의 optimizer·RNG·입력 위치·controller를 포함한 전체 상태가 필요합니다.
