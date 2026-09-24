# iBKD λ=0.25 · warm-up 20 · β4 smoke 결과

2026-09-24 제출 로그를 점검했습니다. **네 β 모두 32-step 학습·파일 복원/재실행·val2 평가를
완료**했습니다. λ=0.25·warm-up 20의 β4개·2k 재실험으로 진행할 수 있습니다.
이번 점검은 smoke 완료이며 2k나 20-epoch 종료 경계를 실행한 결과는 아닙니다.

## 확인 범위

- 원본 첨부: 65,004 bytes, SHA-256 `4110807b8c1b9605234e3e0101fb02361386157dc772a5c4a3186627a213c623`.
- 실행 commit: `e74ba37049cb03bd9757bf91fe34f7285accf0e8`, H200 NVL, PyTorch 2.11.0+cu130.
- 프로토콜: `cityscapes_b0_ibkd_warmup20_v1`, smoke 범위 `cityscapes_b0_ibkd_warmup20_smoke32_v1`.
- [점검 JSON](log_audit.json): 후보별 loss·gradient norm·metric·controller·시간과 확인 제한을 기록했습니다.

첨부 첫 줄은 앞선 후보 요약의 중간부터 시작하지만 **마지막 전체 group JSON은 완전**합니다.
네 후보의 결과는 이 요약에서 직접 확인했으며 누락값을 추정하거나 다른 실행에서 가져오지 않았습니다.
설치·단위 검사 출력과 앞선 세 후보의 세부 학습 로그는 첨부에 없습니다.
마지막 β=3.87021만 완전한 worker summary와 source signature가 있어 현재 소스·프로토콜·grid
해시를 직접 대조했습니다. 실제 checkpoint 파일은 로컬에서 열어 본 것이 아니라 로그의
파일 복원·재실행 성공 기록을 확인했습니다.

전체 `status=completed`, `completed_runs=4`, 후보별 `completed_steps=32`, `error=null`입니다.
모두 step2 파일 복원(bitwise)·step3 재실행(`rtol=2e-5, atol=2e-6`), teacher frozen 검사와
calibration 입력 대조를 통과했습니다. 네 후보의 마지막 입력 hash도 같습니다.
32-batch 사전 검사에서 26번째 batch의 ignore-only sample이 기록됐지만 배치에는 유효 픽셀이
있고 네 후보 모두 해당 batch를 포함해 정상 학습했습니다.

## 마지막 손실

모든 값은 step32의 마지막 batch 값이며 평균 학습 성능이나 β 선정 지표가 아닙니다.

| β | CE | raw guidance | β × guidance | 전체 loss | guidance |
|---|---:|---:|---:|---:|---|
| 0.387021 | 1.258495 | 0.281350 | 0.108888 | 1.367383 | on |
| 0.903048 | 1.247749 | 0.279494 | 0.252396 | 1.500145 | on |
| 1.9351 | 1.267182 | 0.277289 | 0.536582 | 1.803764 | on |
| 3.87021 | 1.224491 | 0.272735 | 1.055541 | 2.280032 | on |

기록된 마지막 손실에서 `raw = 0.75 × alignment + 0.25 × fusion`, `weighted = β × raw`,
`total = CE + weighted`가 FP32 반올림 범위 내에서 일치했습니다. β가 커지며 전체 loss가
높아지는 것은 guidance 계수가 커진 결과이며, 이 표로 큰 β의 실패를 판단하지 않습니다.
마지막 encoder·decoder·adapter·core gradient norm은 모두 유한하고 0보다 큽니다.
마지막 LR도 80k poly schedule의 step32 값과 일치합니다.

## Warm-up과 평가 해석

네 후보 모두 `warmup_epochs=20`, `active=true`, `stop_epoch=null`, 초기 β 유지입니다.
`intervals=0`, `steps=32`, `samples=512`도 정상입니다. 첫 관측이 step186이므로 아직
관측 완료 구간이 없으며 controller의 loss history가 비어 있는 것이 맞습니다.

`warmup_protection_passed=null`, `minimum_period_reached=false`는 실패가 아닙니다.
**실제 GPU에서 3720-step 보호 기간을 아직 끝내지 않았다는 뜻**입니다.
2k 재실험에서도 보호 기간 안이므로 네 후보 모두 마지막까지 guidance를 사용해야 합니다.

Val2 진단 mIoU는 β 순서대로 13.7501%, 14.2804%, 15.0525%, 15.1707%입니다.
네 confusion matrix에서 mIoU·pixel accuracy·class IoU를 다시 계산했고 최대 오차는
`2.78e-17`, 유효 픽셀은 모두 3,719,892개이며 GT 클래스별 픽셀 수도 같습니다.
**2장 진단값으로 후보 순위를 정하지 않습니다.** `selection=not_performed`와
`selected_epoch=null`, `selected_step=null`을 확인했습니다.

## 시간과 다음 실행

설치·자료 준비를 포함한 전체 작업은 **14분 20.5초**였습니다.
후보별 32-step 순수 학습은 93.8~110.4초, 평균 **2.93~3.45초/step**입니다.
Peak allocated GPU memory는 약 **16.96 GiB(18.21 GB)**입니다.
짧은 실행 평균에는 초기화 후 첫 update와 내장 재실행 비용이 포함되며 장기 속도를 보장하지 않습니다.

다음은 [λ=0.25 β4개·2k 이슈](../../H200_WARMUP20_CHECK2000_ISSUE.md)입니다.
각각 공통 초기화부터 2,000 step, 매 400 step 전체 val500을 평가합니다.
Smoke val2 checkpoint나 기존 warm-up 0 checkpoint는 사용하지 않습니다.
이번 점검에서 추가 GPU 작업을 시작하지 않았고 학습 코드·고정 설정을 변경하지 않았습니다.
