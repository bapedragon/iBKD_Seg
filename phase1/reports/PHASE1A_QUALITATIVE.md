# Phase 1A Flowers 정성 결과

이 문서는 Phase 1A 전체 실행에서 실제 사용한 Flowers-102 test 이미지와
segmentation probe의 출력을 Git에서 바로 확인하기 위한 기록입니다.

각 PNG는 왼쪽부터 다음 세 칸으로 구성됩니다.

```text
실제 입력 사진 | 자동 생성 pseudo-mask | frozen-feature probe 예측
```

검정은 배경(class 0), 흰색은 전경(class 1)입니다. 입력과 mask는 `224 x 224`, 전체
panel은 `672 x 224`입니다. 모든 그림은 결과를 보기 전에 고정한 test ID
`1, 45, 568, 1737, 3650, 5775, 8189`와 probe seed 1에서 생성했습니다.

## 무엇을 비교했는가

| 방법 | 비교 역할 | 사용한 encoder | 정성 panel용 선택 |
|---|---|---|---|
| Ours | 조건 일치 주 진단군 | researcher-sync seed 1, λ(fusion ratio) 0.5 | LR 0.01, epoch 20 |
| ALG | 조건 일치 주 진단군 | researcher-sync seed 1 | LR 0.1, epoch 91 |
| KD | 조건 불일치 탐색군 | generic-KD seed 42 | LR 0.01, epoch 60 |

각 encoder는 고정했으며 세 방법 모두 동일한
`Conv2d(192, 2, 1, bias=True)` probe만 따로 학습했습니다. Ours λ `0.25` 실행은
이 문서와 그림에 포함되지 않습니다. 정확한 checkpoint SHA-256과 각 PNG의 hash는
[`figures/phase1a/manifest.json`](figures/phase1a/manifest.json)에 있습니다.

## 대표 사례

### ID 1 — 일반적인 pseudo-mask 사례

Ours

![Ours test ID 1: input, pseudo-mask, prediction](figures/phase1a/ours/image_00001.png)

ALG

![ALG test ID 1: input, pseudo-mask, prediction](figures/phase1a/alg/image_00001.png)

KD — 조건 불일치 탐색용

![KD test ID 1: input, pseudo-mask, prediction](figures/phase1a/kd/image_00001.png)

### ID 45 — pseudo-mask가 완전 배경인 실패 사례

Ours

![Ours test ID 45: input, empty pseudo-mask, prediction](figures/phase1a/ours/image_00045.png)

ALG

![ALG test ID 45: input, empty pseudo-mask, prediction](figures/phase1a/alg/image_00045.png)

KD — 조건 불일치 탐색용

![KD test ID 45: input, empty pseudo-mask, prediction](figures/phase1a/kd/image_00045.png)

### ID 568 — pseudo-mask가 거의 완전 전경인 실패 사례

Ours

![Ours test ID 568: input, near-full pseudo-mask, prediction](figures/phase1a/ours/image_00568.png)

ALG

![ALG test ID 568: input, near-full pseudo-mask, prediction](figures/phase1a/alg/image_00568.png)

KD — 조건 불일치 탐색용

![KD test ID 568: input, near-full pseudo-mask, prediction](figures/phase1a/kd/image_00568.png)

## 전체 고정 표본

대표 세 사례를 포함해 고정한 7개 ID의 모든 방법별 panel을 보존했습니다.

| test ID | Ours | ALG | KD (탐색용) |
|---:|---|---|---|
| 1 | [PNG](figures/phase1a/ours/image_00001.png) | [PNG](figures/phase1a/alg/image_00001.png) | [PNG](figures/phase1a/kd/image_00001.png) |
| 45 | [PNG](figures/phase1a/ours/image_00045.png) | [PNG](figures/phase1a/alg/image_00045.png) | [PNG](figures/phase1a/kd/image_00045.png) |
| 568 | [PNG](figures/phase1a/ours/image_00568.png) | [PNG](figures/phase1a/alg/image_00568.png) | [PNG](figures/phase1a/kd/image_00568.png) |
| 1737 | [PNG](figures/phase1a/ours/image_01737.png) | [PNG](figures/phase1a/alg/image_01737.png) | [PNG](figures/phase1a/kd/image_01737.png) |
| 3650 | [PNG](figures/phase1a/ours/image_03650.png) | [PNG](figures/phase1a/alg/image_03650.png) | [PNG](figures/phase1a/kd/image_03650.png) |
| 5775 | [PNG](figures/phase1a/ours/image_05775.png) | [PNG](figures/phase1a/alg/image_05775.png) | [PNG](figures/phase1a/kd/image_05775.png) |
| 8189 | [PNG](figures/phase1a/ours/image_08189.png) | [PNG](figures/phase1a/alg/image_08189.png) | [PNG](figures/phase1a/kd/image_08189.png) |

## 해석 제한

가운데 mask는 사람이 만든 pixel ground truth가 아니라 blue-screen 합성쌍에서 자동
복원한 pseudo-mask입니다. 특히 ID 45와 568처럼 명백한 mask 실패가 있으므로 이
그림과 정량값은 **파이프라인 및 표현 진단**에만 사용합니다. 실제 segmentation
확장성 주장은 공식 trimap을 쓰는 Phase 1B에서 판단합니다.
