# Phase 1 Pet batch 128 frozen segmentation probe 잠정 결과

상태: **H200 실행 로그상 90/90 완료·pass — 전체 산출물 반입 및 독립 감사 대기**

이 문서는 사용자가 먼저 전달한 콘솔 로그에서 확인되는 결과만 기록합니다. 90개
`PROBE_FULL_RAW` 행은 `(variant, encoder seed, probe seed)` 조합이 중복 없이 모두
존재했고, 마지막 줄은 선택 `90/90`, test-once `90/90`, `status=pass`를 보고했습니다.
다만 전달된 로그가 실행 중간부터 시작하므로 270개 LR 후보 전체의 history,
checkpoint hash·strict-load, confusion 재계산 및 정성 panel은 결과 bundle을 받은 뒤
독립적으로 감사해야 합니다.

현재 로그의 기계 판독 사본은 [log_summary.json](log_summary.json)에 있습니다.

## 로그에서 확인한 실행 정보

- 분류 checkpoint profile: batch 128
- 행렬: 6설정 × encoder seed 3 × probe seed 5 = 선택 probe 90개
- 공식 test 로그: 90개 probe 각각 1회, 총 90/90
- 최종 상태: `status=pass`
- 실행 시간: 3,396.27초 = 56분 36.27초
- H200 출력 위치: `/app/output/phase1_pet_probe_b128_full_v1`
- 전달된 로그: 65,004 byte
- 전달 로그 SHA-256:
  `6dda72b1c7ae3d5f0a81d24f74c4ffff2cfaeef414220a9ae9f1e25b4f5b1d86`

## 주 metric 잠정값

주 metric은 224×224로 복원한 동물/배경 2-class global mIoU입니다. 각 encoder
seed에서 probe seed 5개를 먼저 평균하고, 독립 단위인 encoder seed 3개의
`평균 ± 표본 표준편차`를 백분율로 표시했습니다.

| 설정 | Input mIoU | encoder seed 1 / 2 / 3 |
|---|---:|---:|
| Vanilla | 60.454 ± 0.426 | 59.999 / 60.519 / 60.843 |
| KD | 73.765 ± 0.954 | 73.447 / 73.009 / 74.837 |
| **LG** | **82.856 ± 0.291** | **83.171 / 82.801 / 82.597** |
| ALG | 63.378 ± 0.633 | 63.820 / 63.660 / 62.653 |
| iBKD λ=0.25 | 78.256 ± 0.770 | 78.575 / 77.378 / 78.816 |
| iBKD λ=0.5 | 77.195 ± 0.550 | 77.775 / 76.683 / 77.127 |

잠정 순위는 `LG > iBKD-0.25 > iBKD-0.5 > KD > ALG > Vanilla`입니다.

## Paired encoder-seed 비교

| 비교 | 평균 차이 | seed 1 / 2 / 3 차이 | 방향 |
|---|---:|---:|---|
| iBKD-0.25 − ALG | +14.879 | +14.755 / +13.718 / +16.163 | 모두 양수 |
| iBKD-0.5 − ALG | +13.817 | +13.955 / +13.022 / +14.474 | 모두 양수 |
| iBKD-0.25 − LG | -4.600 | -4.596 / -5.422 / -3.781 | 모두 음수 |
| iBKD-0.5 − LG | -5.661 | -5.396 / -6.118 / -5.470 | 모두 음수 |
| iBKD-0.25 − KD | +4.492 | +5.128 / +4.369 / +3.978 | 모두 양수 |
| iBKD-0.5 − KD | +3.430 | +4.328 / +3.673 / +2.290 | 모두 양수 |

단위는 percentage point입니다. probe 15개를 독립 표본처럼 취급하지 않았으며,
encoder seed가 3개뿐이므로 formal p-value나 통계적 유의성을 주장하지 않습니다.

## Batch 64와 함께 본 잠정 해석

| 설정 | Batch 64 | Batch 128 | 128 − 64 |
|---|---:|---:|---:|
| Vanilla | 60.188 | 60.454 | +0.266 |
| KD | 72.529 | 73.765 | +1.235 |
| LG | 83.851 | 82.856 | -0.994 |
| ALG | 82.091 | 63.378 | **-18.713** |
| iBKD λ=0.25 | 80.295 | 78.256 | -2.039 |
| iBKD λ=0.5 | 79.733 | 77.195 | -2.538 |

1. **LG가 두 batch profile 모두 1위입니다.** 따라서 iBKD가 LG/ALG를 전반적으로
   능가해 공간정보를 가장 잘 보존한다는 원래의 넓은 가설은 현재 지지되지 않습니다.
2. **iBKD–ALG 방향은 batch에 따라 뒤집혔습니다.** Batch 64에서는 iBKD가 ALG보다
   낮았지만 batch 128에서는 크게 높습니다. 좋은 profile만 선택해 iBKD 우위라고
   결론 내릴 수 없습니다.
3. **batch 128 ALG만 18.713%p 급락했습니다.** 같은 checkpoint의 분류 결과에서도
   ALG가 세 seed 모두 epoch 2에 guidance를 종료하면서 크게 낮아졌으므로, 이번
   iBKD–ALG 차이를 iBKD 구조의 순수한 우위로 해석하면 안 됩니다.
4. **iBKD는 KD와 Vanilla보다 두 λ 모두 높습니다.** 이는 spatial guidance가 logit
   KD보다 위치·형태 정보를 더 남긴다는 보조 근거지만, LG보다 낮아서 iBKD 고유의
   우월성 근거는 아닙니다.
5. ALG controller warm-up 20 실행은 이 batch 128 이상 현상의 원인을 보는 사후
   진단이며, 결과가 나오더라도 canonical 결과를 대체하지 않습니다.

## 아직 확정하지 않는 항목

결과 bundle을 받은 뒤 270개 후보 선택 재계산, 90개 probe checkpoint 감사,
global confusion metric 재계산, test-before-selection 기록과 고정 정성 panel을
확인해야 합니다. 그 전까지 이 문서의 수치는 **로그 기반 잠정값**이며 Phase 1 최종
Go/Hold/No-Go 결정도 보류합니다.
