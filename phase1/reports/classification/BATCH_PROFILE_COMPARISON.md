# Phase 1 Pet batch 64/128 분류 profile 비교

두 batch profile은 결과를 보기 전에 모두 실행·보고하기로 고정했습니다. 데이터,
validation split, teacher model-state, seed별 student 초기 state와 test-once 계약이
동일함을 다시 확인했습니다.

| 설정 | Batch 64 Test macro Top-1 | Batch 128 Test macro Top-1 | 128 − 64 |
|---|---:|---:|---:|
| Vanilla | 20.486 ± 0.222 | 21.118 ± 0.658 | +0.632 |
| KD | 30.592 ± 1.010 | 30.078 ± 1.086 | -0.514 |
| LG | **38.206 ± 0.742** | **32.993 ± 1.686** | -5.212 |
| ALG | 32.277 ± 1.418 | 22.880 ± 0.258 | -9.397 |
| iBKD λ=0.25 | 29.474 ± 0.333 | 26.716 ± 0.402 | -2.757 |
| iBKD λ=0.5 | 29.528 ± 2.000 | 24.896 ± 1.217 | -4.632 |

값은 같은 encoder seed 3개의 `평균 ± 표본 표준편차`이며 차이는 seed별
`batch128 − batch64`를 먼저 계산한 paired 평균입니다.

## 핵심 해석

- LG는 두 profile 모두 분류 1위지만 batch 128에서 절대값이 낮아졌습니다.
- ALG의 batch 128 급락은 세 seed 모두 adaptive guidance가 epoch 2에 종료된 것과
  함께 나타났습니다. batch 64 종료 epoch는 `118/144/138`입니다.
- iBKD의 controller는 batch 128에서도 λ=0.25가 `104/120/140`, λ=0.5가
  `125/120/137` epoch까지 유지됐습니다. 따라서 batch 128의 iBKD–ALG 차이를
  방법론의 순수 효과로만 읽으면 안 됩니다.
- batch 효과가 방법별로 크게 다르므로 post-hoc으로 더 좋은 batch만 선택하지
  않습니다. 두 profile을 각각 보고하고 batch 128 frozen probe까지 같은 계약으로
  완료해야 합니다.
- 이 표는 분류 결과이므로 공간정보 보존 증거가 아닙니다.

기계가 읽을 수 있는 재계산 결과는 [JSON](batch_profile_comparison.json)과
[CSV](batch_profile_comparison.csv)에 있으며, 생성 명령은 다음과 같습니다.

```bash
python phase1/scripts/compare_classification_profiles.py \
  --batch64-summary phase1/reports/classification/batch64/summary.json \
  --batch128-summary phase1/reports/classification/batch128/summary.json \
  --output-json phase1/reports/classification/batch_profile_comparison.json \
  --output-csv phase1/reports/classification/batch_profile_comparison.csv
```
