# CUB L2 guided 4방법 seed 1 예비 본실험 — 로그 스냅샷

상태: **본실험 완료 로그 확인, 결과 archive·checkpoint 독립 감사 대기**

선택된 `L2 conservative spatial` loader로 LG, ALG-w20, iBKD λ=0.25/0.5를
각각 encoder seed 1에서 300 epoch 학습하고, 선택한 encoder에 동일한 frozen
segmentation probe를 적용한 예비 결과입니다. 이 문서는 전달받은 H200 로그의
완료 marker와 결과값만 보존합니다. 결과 archive가 도착하기 전이므로 checkpoint
hash, 원본 summary 및 metric 재계산을 완료한 보고서로 간주하지 않습니다.

## 실행 완료 상태

- 분류 학습·선택·test: `4/4`
- Probe LR 후보: `60/60`
- Validation probe 선택: `20/20`
- Official-test probe 평가: `20/20`
- 로그상 새 checkpoint: `24개`
- 전체 실행시간: `3시간 16분 10초`
- 모든 20개 probe 선택 전 official-test mask 접근: `false`

## 분류 결과

주 지표는 CUB 200종 official-test macro Top-1입니다. Checkpoint는 validation
macro Top-1으로 선택했습니다.

| 방법 | 선택 epoch | Validation macro Top-1 | Test macro Top-1 |
|---|---:|---:|---:|
| **LG** | 213 | **32.5000%** | **30.0931%** |
| ALG-w20 | 109 | 27.8333% | 25.4625% |
| iBKD λ=0.25 | 123 | 25.0000% | 21.9337% |
| iBKD λ=0.5 | 242 | 26.0000% | 22.6542% |

분류 순위는 `LG > ALG-w20 > iBKD-0.5 > iBKD-0.25`입니다.

## Frozen segmentation probe 결과

각 값은 동일 encoder에서 실행한 probe seed 5개의 official-test input-224 mIoU
평균 ± 표본 표준편차입니다.

| 방법 | Test input-224 mIoU |
|---|---:|
| LG | 74.1203 ± 0.0724% |
| **ALG-w20** | **74.2158 ± 0.1870%** |
| iBKD λ=0.25 | 72.9239 ± 0.1137% |
| iBKD λ=0.5 | 73.6347 ± 0.1547% |

- ALG-w20 − LG: `+0.0955%p`
- ALG-w20 − iBKD-0.5: `+0.5811%p`
- LG − iBKD-0.5: `+0.4856%p`
- iBKD-0.5 − iBKD-0.25: `+0.7108%p`

## 해석 범위

- 이 L2 seed-1 결과에서는 ALG-w20이 probe mIoU 1위이고 LG가 매우 근소하게
  뒤따릅니다. 두 iBKD 설정은 모두 LG와 ALG-w20보다 낮습니다.
- 가장 높은 iBKD 설정은 λ=0.5지만, ALG-w20보다 `0.5811%p` 낮습니다. 따라서
  이 결과는 iBKD의 공간정보 보존 우위를 지지하지 않습니다.
- `±`는 독립 encoder seed가 아니라 한 encoder에서 반복한 probe seed 변동입니다.
  `independent_encoder_n=1`이므로 방법 간 통계적 유의성이나 동등성을 판단할 수
  없습니다.
- 본 결과는 선택된 loader의 단일 encoder seed 예비 후속 실험입니다. 완료된 기존
  CUB v3 주 결과나 향후 6방법×3seed 확증 결과를 대체하지 않습니다.

분석용 집계값은 [classification_results.csv](classification_results.csv),
[probe_aggregate_results.csv](probe_aggregate_results.csv)에 있으며, probe seed별
로그값은 [probe_test_results.csv](probe_test_results.csv)에 보존합니다.
