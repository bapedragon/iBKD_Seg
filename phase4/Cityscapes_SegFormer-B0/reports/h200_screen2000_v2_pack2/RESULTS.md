# H200 2k v2 pack2 후보별 결과 점검

2026-09-24 사용자 첨부를 점검했습니다. **ALG 네 후보와 iBKD λ=0.25 네 후보의
기록에서 step2000 전체 val500 완료 및 재개·안정성 검사 통과를 확인**했습니다.
완전한 후보 JSON은 7개이고 ALG β=0.197479는 앞부분이 잘린 나머지 필드로 확인했습니다.
그 후보에도 stage=completed, target/last_eval/selected step=2000,
controller 누적 10×186+140=2000, error=null이 남아 있습니다.

후보별 점수에 사전 선별 규칙을 적용하면 다음 10k 후보가 남습니다.

- **ALG: β=0.197479·0.460784**.
- **iBKD λ=0.25: β=0.903048·3.87021**.

## 첨부 범위와 검증

원본은 65,004 bytes이며 SHA-256은
`5ff7cab0ed2d01ef3f332b082a0851b89b56b4679cb006ca246a153391d64b81`입니다.
[점검 JSON](log_audit.json)과 [재계산한 선별 JSON](selection.json)을 저장합니다.

첨부는 group JSON 중간부터 시작합니다. 전체 status·selection·git commit·입력 preflight 상세·
전체 elapsed_seconds가 없으므로 해당 값을 확인했다고 표시하지 않습니다.
Pack2와 v2 식별은 남아 있는 각 후보의 resume 경로에서 확인했습니다.
ALG 첫 후보의 run_id도 해당 경로, 초기 β는 controller.beta_on에서 식별했습니다.
누락된 첫 후보의 completed_steps·마지막 loss·best metrics 원본 필드는 null로 보존했습니다.
다만 selected_step=last_eval_step=2000이므로 해당 후보의 best 점수는 남아 있는 last metrics와
같다고 추론해 순위 계산에 사용했습니다. 저장한 selection은 원본 group selection의 복사본이 아닙니다.

확인된 best/last confusion matrix 15개에서 19개 class IoU·mIoU·pixel accuracy를
재계산했고 최대 절대 오차 `1.11e-16`으로 일치했습니다.
GT 클래스별 픽셀 수가 모두 같고 전체 유효 픽셀은 917,018,489입니다.
각 run의 val500 전체 평가, replay passed, stability500 passed, calibration input passed,
teacher frozen 확인, error=null, test_used=false를 확인했습니다.
마지막 입력 해시도 8개 후보와 기존 pack1에서 동일합니다.
단, 이 첨부만으로 누락된 전체 사전 검사·환경 식별자·checkpoint 실물을 확인한 것은 아닙니다.

## 점수와 guidance 종료

단위는 mIoU %. 모든 후보의 선택 step은 2000입니다.

| 방법 | 초기 β | step2000 mIoU | guidance 종료 판정 step | 다음 단계 |
|---|---:|---:|---:|---|
| ALG | 0.197479 | 44.7611 | 2k까지 유지 | 10k 유지 1순위 |
| ALG | 0.460784 | 44.3550 | 1860 | 10k 유지 2순위 |
| ALG | 0.987394 | 44.0449 | 1488 | 2k 선별 종료 |
| ALG | 1.97479 | 43.2485 | 1116 | 2k 선별 종료 |
| iBKD λ=0.25 | 0.387021 | 45.3854 | 744 | 2k 선별 종료 |
| iBKD λ=0.25 | 0.903048 | **45.4024** | 744 | 10k 유지 1순위 |
| iBKD λ=0.25 | 1.9351 | 45.1269 | 744 | 2k 선별 종료 |
| iBKD λ=0.25 | 3.87021 | **45.4011** | 558 | 10k 유지 2순위 |

완전한 7개 후보의 best metrics는 last metrics와 같고 ALG 첫 후보는 위 범위대로 추론했습니다.
iBKD 상위 두 후보의 차이는 **0.0012998 percentage point**입니다.
2순위와 3순위의 차이도 약 0.0158 point로 작습니다. 반올림 전 best mIoU 순서를 적용했고,
작은 차이를 안정적인 우열로 주장하지 않습니다. 추가 β를 사후에 넣지 않습니다.

이전 pack1의 Vanilla 45.2942%와 비교하면 iBKD 최고값은 **+0.1083 point**입니다.
Seed1의 초기 선별 결과이므로 유의한 개선이나 80k 최종 우위를 확인한 것으로 해석하지 않습니다.
ALG β=0.197479는 아직 guidance가 켜져 있고, 동일 β의 LG와 last metrics 전체가 일치합니다.
현재 코드에서 종료 전의 LG·ALG는 같은 학습을 하므로 이 일치는 예상되는 동작입니다.

## Controller 점검

Warm-up 0, window50, threshold−0.02, 186-step 관측 조건을 8개 모두 확인했습니다.
저장된 구간 손실로 controller를 재계산한 결과 종료 epoch, active, beta history가 같고,
미분·평활값의 최대 절대 오차는 `6.94e-18`이었습니다(허용 절대 오차 `1e-14`).
ALG는 평활 변화량 `>= -0.02`, iBKD는 `> -0.02`에서 종료되는 기존 규칙과 일치합니다.
Window50은 50 epoch의 warm-up을 뜻하지 않으며 초기에는 관측된 구간으로 계산합니다.

표의 step까지는 guidance를 사용하고 **다음 step부터 CE만 사용**합니다.
예를 들어 iBKD β=3.87021은 558까지 guidance, 559부터 CE-only이고,
β=0.903048은 744까지 guidance, 745부터 CE-only입니다. 학습 자체는 2k까지 계속됐습니다.
종료된 7개 후보의 마지막 loss는 CE 하나이고 raw_guidance=null,
weighted_guidance=0, beta=0, guidance gradient norm=0입니다.
이는 controller 종료 결과이며 β 설정이 누락되거나 학습이 중단된 증거가 아닙니다.

## 시간과 후속 작업

8개 run의 elapsed_seconds 합은 **약 5시간 50분 36초**입니다.
설치·공통 준비가 포함된 전체 작업 시간은 잘린 group header에 있어 확인할 수 없습니다.
ALG는 후보당 약 34~41분, iBKD는 약 44~52분입니다.
Guidance가 일찍 꺼진 iBKD β=3.87021이 약 44분으로 가장 짧았습니다.

선정한 후보는 step2000의 전체 optimizer·RNG·controller 상태에서 10k까지 이어갑니다.
이미 꺼진 guidance를 재개하면서 다시 켜지 않습니다.
`/app/output/cityscapes_b0_screen2000_v2/pack2/` 전체 출력 묶음을 보존해야 하며,
10k 준비 시 원본 group_summary의 식별자·selection과 checkpoint 파일을 확인합니다.
Pack3(iBKD λ=0.5) 결과는 아직 제출되지 않았습니다. 이번 점검에서 학습 코드나 프로토콜을
변경하거나 새 GPU 작업을 시작하지 않았습니다.
