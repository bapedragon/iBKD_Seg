# iBKD λ=0.25 · warm-up 20 · β4개 2k 결과

2026-09-25 제출 로그를 점검했습니다. **네 후보 모두 2,000 step·전체 val500을 완료**했고,
guidance가 마지막까지 유지됐습니다. 파일 복원·재실행, step500 안정성, 입력 대조,
teacher 고정 검사도 통과했습니다. 시간 제한 중단이나 실패 후보는 없습니다.

현재 최고 mIoU는 **β=0.387021의 45.3352%**입니다. 기존 warm-up 0보다 세 후보는 소폭 낮고
한 후보는 소폭 높습니다. **이번 2k만으로 warm-up 20의 성능 개선을 확인했다고 볼 수는 없습니다.**
최종 β를 고르거나 후보를 탈락시키지 않았습니다.

## 로그와 확인 범위

- 원본 첨부: 65,004 bytes, SHA-256 `a1512037766e68d5bf38e5c0b48cd93097e56db7271cb2a4e362a9b810c9208f`.
- 실행 commit: `3b157b20eaab53ee8192716a083b4d20800dc6d9`, H200 NVL, PyTorch 2.11.0+cu130.
- 실행 범위: `cityscapes_b0_ibkd_warmup20_v1`, λ=0.25, β4개, target2000, 전체 val500.
- 활성 시간 예산: `job_runtime.maximum_job_seconds=34800`으로 **9시간 40분 적용**을 확인했습니다.
- [점검 JSON](log_audit.json)에 후보별 손실·metric·controller·시간·이전 결과와의 차이를 저장했습니다.

첨부 앞부분은 마지막 후보 worker summary의 중간부터 시작하지만 **마지막 group JSON은 완전**합니다.
네 후보 best/last/2k metric과 마지막 loss는 모두 group JSON에서 직접 읽은 값입니다.
마지막 후보의 signature는 잘린 첫 줄 안에 완전한 중첩 JSON으로 남아 있어 별도 추출했고,
현재 학습 코드·프로토콜·grid 해시와 일치했습니다. 후보별 checkpoint 실물을 직접 로딩한 것은 아닙니다.
매 400-step 평가의 전체 이력은 첨부에 없으므로 학습 곡선 전체를 재구성하지 않았습니다.

네 후보 모두 `selected_step=last_eval_step=2000`이며 best·last·2k 경과 metric이 같습니다.
이 세 metric 기록 12개(서로 다른 confusion matrix는 4개)의 class IoU·mIoU·pixel accuracy를
다시 계산했고 최대 절대 오차는 `1.11e-16`입니다. 유효 픽셀은 모두 917,018,489개이며
GT 클래스별 픽셀 수도 같습니다. 입력 preflight는 기존 smoke·pack1과 같고 마지막 batch hash도
기존 2k 실행과 같습니다. 마지막 손실 혼합식과 80k LR schedule도 일치했습니다.

## Warm-up 0과의 같은 β 비교

mIoU 단위는 %, 차이는 percentage point(pp)입니다. 모든 후보의 best와 last는 step2000입니다.

| β | 기존 warm-up 0 | 이번 warm-up 20 | 차이(pp) | 이번 guidance |
|---|---:|---:|---:|---|
| **0.387021** | 45.3854 | **45.3352** | −0.0501 | 2k까지 on |
| 0.903048 | 45.4024 | 45.1693 | −0.2332 | 2k까지 on |
| 1.9351 | 45.1269 | 45.1991 | +0.0722 | 2k까지 on |
| 3.87021 | 45.4011 | 45.1337 | −0.2675 | 2k까지 on |

비교 원본은 [이전 warm-up 0 pack2](../h200_screen2000_v2_pack2/RESULTS.md)입니다.
그 첨부의 group header는 누락됐지만 비교에 사용한 iBKD 네 후보의 metric 기록은 완전합니다.
Vanilla의 기존 2k mIoU는 **45.2942%**로 이번 최고값과의 차이는 **+0.0411pp**입니다.
단일 seed의 작은 차이이므로 안정적인 성능 우위로 해석하지 않습니다.

이번 관측 순위는 β=0.387021 → 1.9351 → 0.903048 → 3.87021입니다.
이는 경과 설명이며 자동 선별은 `not_performed`입니다. 기존 warm-up 20 계획의 β4개 유지·10k
최종 선별 규칙을 이번 순위에 맞춰 변경하지 않습니다.

## Guidance 보호 동작

네 후보 모두 `warmup_epochs=20`, `active=true`, `stop_epoch=null`, 초기 β 유지입니다.
완료 관측은 **10구간**, 진행 중인 구간은 **140 step**으로 `10×186+140=2000`과 일치합니다.
실제 관측 손실을 기존 controller에 다시 넣었을 때 derivative·활성 상태·β history가 일치했고,
derivative의 최대 절대 오차는 `8.67e-18`입니다.

Warm-up 20에서는 20번째 관측 전까지 종료 판단을 보류하므로
`smoothed_derivative_history`의 null과 `warmup_protection_passed=null`은 정상입니다.
손실 관측 자체가 빠진 것이 아니며, 10개의 raw guidance 평균과 derivative가 저장돼 있습니다.
이전 warm-up 0에서 종료되기 전까지의 관측 손실은 이번 같은 β의 초기 구간과 정확히 같습니다.
이후 기존은 CE-only, 이번은 CE+guidance를 계속하는 차이가 생겼습니다.

| β | 마지막 CE | 마지막 raw guidance | 마지막 β × guidance | 마지막 전체 loss |
|---|---:|---:|---:|---:|
| 0.387021 | 0.558807 | 0.186130 | 0.072036 | 0.630843 |
| 0.903048 | 0.564103 | 0.185546 | 0.167557 | 0.731660 |
| 1.9351 | 0.558565 | 0.182988 | 0.354100 | 0.912665 |
| 3.87021 | 0.520837 | 0.179865 | 0.696115 | 1.216952 |

마지막 student·adapter·core gradient norm은 모두 유한하고 0보다 큽니다.
β=3.87021의 마지막 CE가 가장 낮아도 val mIoU가 가장 높지는 않습니다.
마지막 train batch의 CE와 validation 성능을 구분합니다.

## 시간과 다음 확인

전체 소요는 **7시간 5분 58초**입니다. 후보별로 약 1시간 41분~1시간 51분,
순수 학습 평균은 2.79~3.06초/step입니다. Peak allocated GPU memory는 약 16.96 GiB입니다.
9시간 40분 한도 전에 네 후보 모두 끝났으므로 이번 묶음의 미완료 후보 재개는 필요하지 않습니다.

2k는 **약 10.75 epoch 분량**으로 최소 20-epoch 보호 종료인 step3720보다 이릅니다.
확인한 것은 2k까지의 guidance 유지와 초기 성능입니다. 20-epoch 이후 자동 종료와 장기 성능은
아직 검증하지 않았습니다. 기존 후속 계획은 네 후보를 같은 전체 상태에서 10k로 이어 비교하는
것이며, 이번 점검에서 새 GPU 작업을 시작하지 않았습니다. λ=0.5 warm-up 20도 아직 미실행입니다.

`/app/output/cityscapes_b0_ibkd_warmup20_v1_check2000/lambda025/` 출력 전체를 보존합니다.
후속 재개에는 best student만이 아니라 step2000의 optimizer·RNG·입력 위치·controller를
포함한 전체 상태가 필요합니다. 이번에 학습 코드·고정 설정을 변경하지 않았습니다.
