# CUB ResNet-50/224 v4 guided seed-1 결과

상태: **H200 issue 727 완료 · 독립 감사 통과 · 부분 매트릭스**

이 실행은 issue 722의 한 ResNet-50/224 scratch Teacher를 공유해 LG, ALG-w20,
iBKD λ=0.25/0.5를 encoder seed 1에서 학습하고 frozen segmentation probe까지
완료했습니다. Batch 128은 잠긴 v3 주 비교의 네 셀이며, batch 64는 사전 표기한
sensitivity profile입니다. 아직 Vanilla/KD와 batch-128 encoder seed 2·3이 없으므로
최종 6방법×3seed 결과가 아닙니다.

## 완결성 및 감사

- 실행시간: 7.08시간
- 분류: 8/8, probe LR 후보: 120/120, validation 선택: 40/40
- 분류 official test: 8/8, probe official test: 40/40
- 새 checkpoint: encoder 8개 + 선택 probe 40개 = 48개
- 48개 모두 파일 SHA-256, `weights_only=True`, strict load, 유한값 감사 통과
- 공용 Teacher hash와 train/validation/test `5,394 / 600 / 5,794` split hash 일치
- 원시 checkpoint는 [검증된 GitHub Release](artifact_release.json)에 보존하고
  Git history에는 hash·요약만 기록

## 200종 분류

| 방법 | batch128 선택 epoch | batch128 test macro top-1 | batch64 선택 epoch | batch64 test macro top-1 |
|---|---:|---:|---:|---:|
| LG | 208 | 24.315% | 214 | 25.862% |
| ALG-w20 | 108 | 21.382% | 105 | 24.641% |
| iBKD λ=0.25 | 239 | 27.036% | 239 | 26.031% |
| iBKD λ=0.5 | 58 | 11.612% | 283 | 19.131% |

두 batch 모두 iBKD λ=0.25의 seed-1 분류 정확도가 네 방법 중 가장 높았습니다.
반면 λ=0.5는 두 batch 모두 크게 낮아, 현재 설정에서는 λ 민감도가 큽니다.

## Frozen segmentation probe

아래 `±`는 한 encoder에서 반복한 **5개 probe seed의 sample SD**입니다. 독립
encoder는 batch별 하나뿐이므로 encoder-seed SD나 방법 간 통계적 유의성으로
해석하면 안 됩니다.

| 방법 | batch128 test input-224 mIoU | batch64 test input-224 mIoU |
|---|---:|---:|
| LG | 73.695 ± 0.147% | 74.007 ± 0.188% |
| ALG-w20 | 70.299 ± 0.252% | 72.402 ± 0.229% |
| iBKD λ=0.25 | 71.972 ± 0.258% | 71.198 ± 0.145% |
| iBKD λ=0.5 | 63.993 ± 0.255% | 68.533 ± 0.194% |

## 현재 해석

- Batch 128에서 iBKD λ=0.25는 ALG-w20보다 `+1.673%p` 높지만,
  LG보다는 `-1.723%p` 낮습니다.
- Batch 64에서는 iBKD λ=0.25가 ALG-w20보다 `-1.204%p`, LG보다
  `-2.809%p` 낮습니다.
- 따라서 seed 1만 놓고 보면 “iBKD가 ALG보다 항상 높다”거나 “guided 방법 중
  iBKD가 공간정보를 가장 잘 보존한다”는 결론은 성립하지 않습니다. Batch 128의
  ALG 대비 우위는 관찰됐지만 LG가 더 높고, batch 64에서는 방향도 바뀝니다.
- 분류에서는 iBKD λ=0.25가 가장 높지만 probe에서는 LG가 가장 높습니다. 즉 이번
  seed-1 결과는 분류 성능과 frozen spatial probe 성능이 같은 순서가 아님을
  보여줍니다.
- Batch 크기 효과도 방법마다 방향과 크기가 다릅니다. Batch 64는 탐색적
  sensitivity이므로 batch 선택 근거로 사후 사용하지 않습니다.

최종 guided 판단은 이미 고정해 실행 중인 batch-128 seed 2·3을 합쳐 encoder
seed `[1,2,3]` 기준으로 내려야 합니다. 이후 Vanilla/KD까지 같은 계약으로 채워야
전체 6방법 비교가 됩니다. Part localization, spatial CKA, attention–GT 분석에는
이번에 보존한 8개 encoder checkpoint를 후속 입력으로 사용할 수 있습니다.
