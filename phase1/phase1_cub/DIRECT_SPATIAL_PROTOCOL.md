# CUB 직접 공간정보 진단 v2

상태: **batch-128 guided 네 방법 × encoder seed 3개 본실험 완료·감사 통과**

이 진단은 분류 성능이나 segmentation mIoU만으로 공간정보 보존을 추정하지 않고,
CUB의 part와 mask 정답을 이용해 encoder 표현을 직접 확인합니다. 비교 대상은 LG,
ALG-w20, iBKD λ=0.25, iBKD λ=0.5입니다.

## 1. Part localization probe — 주 지표

- 분류 checkpoint의 encoder를 strict load 후 완전 freeze
- block 11 pre-norm patch feature `[192,14,14]`
- 공통 `Conv2d(192,15,1,bias=True)` head
- visible이며 실제 이미지 범위 안에 있는 part만 loss·평가에 포함
- Gaussian heatmap σ=1 grid pixel, masked MSE
- LR `[0.01,0.03,0.1]`, 100 epoch, probe seeds `[1,2,3,4,5]`
- validation micro PCK@0.1로 선택; 동률은 낮은 LR, 이른 epoch 순
- 모든 선택 완료 뒤 official-test PCK@0.1을 probe당 1회 평가

정답 part와 예측점의 거리가 `0.1 × max(GT bbox width, height)` 이하면 정답입니다.
주 보고 단위는 각 encoder seed 안에서 probe seed 5개를 먼저 평균한 뒤 독립
encoder seed 3개에 대해 계산한 평균과 표본 표준편차입니다.

## 2. Spatial CKA — 보조 지표

- 고정 validation 600장, random augmentation 없음
- ResNet-50 teacher `layer3`의 14×14 공간 feature와 DeiT blocks 0–11 patch
  feature 비교
- image×spatial position을 관측 단위로 한 centered linear CKA
- float64 streaming accumulator 사용
- CKA는 교사와의 표현 정렬도이며, 단독으로 공간적 유용성을 증명하지 않음

## 3. Attention–GT localization — 보조 지표

- 12개 layer 전체의 CLS-to-patch attention rollout
- head 평균, identity 추가와 row normalization 후 순차 합성
- global micro patch AP, pointing game, foreground attention mass 보고
- 고정 official test와 고정 qualitative image ID를 사용하며 threshold 튜닝 없음

## 데이터·평가 계약

- CUB split `5,394 / 600 / 5,794`
- classification encoder 학습에는 part·box·mask를 사용하지 않음
- train에서 공식 visible이지만 프레임 밖인 image 5007 part 4 한 개는 사전 규칙대로
  제외하고 clipping하지 않음
- validation으로 선택하기 전 official test에 접근하지 않음

본학습 config는
[`seed 1`](configs/cub200_r50_224_b128_seed1_direct_spatial_full_v2.json)과
[`seed 2·3`](configs/cub200_r50_224_b128_seed2_3_direct_spatial_full_v2.json)으로
분리되어 있습니다. 실행에 필요한 불변 metric/checkpoint inventory는
`*_direct_spatial_metric_contract_*.json`에 보존합니다.

## 결과

| 방법 | Part PCK@0.1 ↑ | 위치오차 ↓ | CKA block11 ↑ | Attention AP ↑ |
|---|---:|---:|---:|---:|
| LG | **27.510 ± 2.030%** | **0.2356 ± 0.0094** | **0.4285 ± 0.0166** | 0.2269 ± 0.0678 |
| ALG-w20 | 21.621 ± 2.716% | 0.2725 ± 0.0194 | 0.3768 ± 0.0275 | 0.2457 ± 0.0173 |
| iBKD λ=0.25 | 20.109 ± 2.024% | 0.2946 ± 0.0179 | 0.3184 ± 0.0152 | **0.2816 ± 0.0546** |
| iBKD λ=0.5 | 15.426 ± 1.858% | 0.3407 ± 0.0344 | 0.2351 ± 0.0311 | 0.2408 ± 0.0245 |

주 지표 Part PCK와 CKA는 LG가 1위입니다. Attention 평균은 iBKD-0.25가 높지만
주 지표와 방향이 달라 보조 관측으로만 해석합니다. 따라서 이 프로토콜은 iBKD의
전반적 공간정보 우위를 지지하지 않습니다. 전체 수치·감사는
[3-seed 결과 보고서](reports/direct_spatial/resnet50_224_b128_guided_3seed_v2/RESULTS.md)에
있습니다.
