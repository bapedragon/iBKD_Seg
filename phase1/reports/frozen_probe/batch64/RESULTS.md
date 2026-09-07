# Phase 1 Pet batch 64 frozen segmentation probe 결과

상태: **본 실험 완료·감사 통과 — batch 64의 iBKD > ALG 1차 가설은 지지되지 않음**

분류 label만으로 학습한 batch 64 encoder 18개를 고정하고, Oxford-IIIT Pet의 공식
pixel trimap으로 동일한 `Conv2d(192, 2, 1)` probe만 학습했습니다. 이 결과는
encoder feature에서 동물/배경 공간정보가 얼마나 선형적으로 복원되는지 비교합니다.

## 실행 및 protocol 감사

- H200 작업: `706`
- 실행 코드 commit: `a53f22c8ede9220013bef033087f1f9a79f91567`
- LOCK protocol SHA-256:
  `38f743958d1211144495dd9b4c7eb6edd4c12ab1bacbb27c75d38528b3e72143`
- 행렬: 6설정 × encoder seed 3 × probe seed 5 = 선택 probe 90개
- 후보: probe마다 LR `[0.01, 0.03, 0.1]` × 100 epoch = 총 270개
- split: train 2,940 / validation 740 / official test 3,669
- 선택: validation `14×14` 2-class mIoU만 사용
- 공식 test: 90개 선택이 모두 끝난 뒤 probe당 정확히 1회
- encoder gradient tensor 0개, probe gradient tensor 2개, parameter 386개
- 전체 시간: 3,535.088초 = 58분 55.088초
- peak CUDA allocated memory: 134,273,536 byte

독립 curation에서 다음을 모두 재검사했습니다.

- 270개 후보의 100-epoch history와 best epoch/LR 선택 재계산
- probe seed별 동일 초기 state와 epoch별 동일 batch order
- 선택 전 기록에 test 값이 없고, 90개 official test가 정확히 한 번인지 확인
- 90개 `.pt`의 SHA-256, `weights_only=True`, strict-load, 유한값과 test-before-write 확인
- 저장 confusion으로 90개 grid/input metric 전부 재계산
- encoder seed → probe seed의 계층적 집계 재계산
- 사전에 고정한 16개 panel과 80개 원시 mask decode 확인

## 정량 결과

주 metric은 224×224로 복원한 동물/배경 2-class global mIoU입니다. 각 encoder
seed에서 probe seed 5개를 먼저 평균하고, 아래에는 독립 단위인 encoder seed 3개의
`평균 ± 표본 표준편차`를 백분율로 표시했습니다.

| 설정 | Input mIoU | Foreground IoU | Background IoU | Foreground Dice | Grid mIoU |
|---|---:|---:|---:|---:|---:|
| Vanilla | 60.188 ± 0.987 | 49.013 ± 0.944 | 71.362 ± 1.032 | 65.779 ± 0.854 | 56.820 ± 0.706 |
| KD | 72.529 ± 0.768 | 65.047 ± 0.913 | 80.011 ± 0.632 | 78.820 ± 0.669 | 67.050 ± 0.853 |
| **LG** | **83.851 ± 0.057** | **79.127 ± 0.068** | **88.574 ± 0.047** | **88.347 ± 0.043** | **78.507 ± 0.018** |
| ALG | 82.091 ± 1.060 | 76.944 ± 1.315 | 87.238 ± 0.807 | 86.966 ± 0.843 | 76.558 ± 1.039 |
| iBKD λ=0.25 | 80.295 ± 0.291 | 74.593 ± 0.356 | 85.997 ± 0.229 | 85.448 ± 0.234 | 74.570 ± 0.298 |
| iBKD λ=0.5 | 79.733 ± 1.590 | 73.839 ± 1.995 | 85.628 ± 1.186 | 84.941 ± 1.317 | 74.069 ± 1.581 |

Encoder seed별 주 metric 원값은 [per_encoder_seed.csv](per_encoder_seed.csv), 90개
probe 원값은 [raw_results.csv](raw_results.csv)에 있습니다.

## 사전에 정한 paired 비교

| 비교 | Input mIoU 차이 | seed 1 / 2 / 3 차이 | 방향 일관성 |
|---|---:|---:|---|
| iBKD-0.25 − ALG | **-1.796** | -0.335 / -2.957 / -2.095 | 모두 음수 |
| iBKD-0.5 − ALG | **-2.358** | -2.623 / -3.464 / -0.986 | 모두 음수 |
| iBKD-0.25 − LG | -3.555 | -3.337 / -3.804 / -3.525 | 모두 음수 |
| iBKD-0.5 − LG | -4.117 | -5.626 / -4.311 / -2.415 | 모두 음수 |
| iBKD-0.25 − KD | +7.766 | +8.538 / +6.577 / +8.182 | 모두 양수 |
| iBKD-0.5 − KD | +7.204 | +6.250 / +6.070 / +9.292 | 모두 양수 |

단위는 percentage point입니다. probe seed 15개를 독립 표본처럼 취급하지 않았고,
encoder seed가 3개뿐이므로 formal p-value나 “통계적 동급”을 보고하지 않습니다.

## 비학습 baseline

- all-background: input mIoU `32.903%`
- train-mean-mask: input mIoU `61.572%`

KD와 그 이상의 네 spatial-guidance 계열은 mean-mask baseline을 분명히 넘습니다.
Vanilla는 `60.188%`로 mean-mask보다 `1.385`%p 낮아, 이 작은 linear probe에서
Vanilla feature의 공간 복원성이 약하다는 점도 확인됩니다.

## 결론

1. **batch 64의 1차 가설은 No-Go입니다.** 가장 가까운 비교인 iBKD–ALG에서 두
   λ 모두 평균이 낮고, 여섯 paired seed 차이가 전부 음수입니다. 따라서 이 결과를
   “분류 성능은 비슷하지만 iBKD probe가 일관되게 높다”라고 쓸 수 없습니다.
2. **LG가 분류와 probe 모두 가장 높습니다.** 현재 v1 설정에서 iBKD가 공간정보를
   가장 잘 보존한다는 주장은 지지되지 않습니다.
3. **다만 iBKD는 KD와 Vanilla보다 일관되게 높습니다.** 이는 grid-aware spatial
   guidance가 logit KD보다 공간정보를 더 남길 수 있다는 보조 근거입니다. 그러나
   LG/ALG보다 낮으므로 iBKD만의 우월성 근거는 아닙니다.
4. **결과 확인 뒤 protocol을 바꾸면 안 됩니다.** probe head, LR grid, selection,
   mask 규칙을 고쳐 같은 v1 결과를 대체할 수 없습니다. 수정 실험은 명시적인 v2
   ablation으로 분리해야 합니다.
5. **전체 Phase 1 결정은 아직 보류입니다.** 후속 batch 128 probe는 로그상
   완료됐지만 iBKD–ALG 방향이 반대로 나타났고 LG가 다시 1위였습니다. Batch 128
   산출물 감사와 ALG warm-up 20 사후 진단 뒤 최종 Go/Hold/No-Go를 기록합니다.

분류 정확도와 probe mIoU 사이의 인과관계를 이 여섯 점만으로 주장할 수도 없습니다.
Batch 64에서는 LG와 ALG가 분류에서도 iBKD보다 높았고 probe에서도 높았다는 관측만
정직하게 기록합니다.

## 정성 결과와 보존

- 고정 정성 panel: [QUALITATIVE.md](QUALITATIVE.md)
- GitHub Release: [artifact_release.json](artifact_release.json)에 고정한 90개 probe,
  전체 원시 결과, 선택 전 기록, 정성 mask와 로그
- checkpoint audit: [checkpoint_manifest.json](checkpoint_manifest.json)
- 전체 curation 결과: [summary.json](summary.json)
- Git 제외 raw:
  `phase1/results/raw/oxford_iiit_pet/frozen_probe_v1/batch64/`

원본 공유 ZIP은 batch 128 분류 결과와 함께 들어 있었기 때문에 논리적 이름을
`phase1_pet_b128_classification_issue702_and_b64_frozen_probe_issue706_v1.zip`으로
기록했습니다. 사용자 원본 ZIP은 수정·삭제하지 않았습니다.
