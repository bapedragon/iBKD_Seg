# Phase 0.5 최종 결정

상태: **완료 — 파이프라인 통과(GO TO PHASE 1 PREPARATION)**

결정일: 2026-09-04

## 결론

Flowers-102 pseudo-mask를 사용한 frozen spatial-probe 파이프라인이 전체 공식
split에서 정상 작동했습니다. 따라서 Oxford-IIIT Pet 데이터 계약과 조건 일치
분류 encoder 학습을 수행하는 Phase 1 준비로 진행합니다.

이는 **파이프라인이 작동한다는 뜻**입니다. Flowers 방법 순위가 신뢰 가능한 pixel
ground truth에서 재현됐다는 뜻도, iBKD의 세그멘테이션 확장성에 대한 과학적 Go
결정도 아닙니다.

## 실행 계약과 gate

- 고정 프로토콜: [PROTOCOL.md](PROTOCOL.md)
- config: [`configs/flowers102_phase05_v1.json`](configs/flowers102_phase05_v1.json)
- 실행 당시 config SHA-256:
  `2cf353e0e9e6dcc0b0b01b75eb17039dde2a20ac5af07f1c1683ec672c3afae2`
- 공식 split: train 1,020 / validation 1,020 / test 6,149
- 환경: Python 3.13.1, PyTorch 2.11.0, timm 1.0.27, CPU
- 전체 실행 시간: 2,057.49초

finite 학습, frozen encoder, probe-only gradient, validation-only 선택, test-once
평가, 고정 test ID panel 생성과 Phase 0 mask 감사 일치를 모두 확인했습니다.

## 실제 사용한 segmentation head

```text
고정된 DeiT-Tiny feature [B, 192, 14, 14]
                    ↓ Conv2d(192, 2, kernel_size=1, bias=True)
배경/전경 logits             [B,   2, 14, 14]
                    ↓ bilinear upsampling
평가용 logits                [B,   2, 224, 224]
```

- weight `2 x 192 x 1 x 1` + bias `2` = 총 **386개 parameter**
- probe만 학습하고 encoder는 `eval` 상태로 완전히 고정
- unweighted cross entropy, SGD momentum 0.9, weight decay 0
- LR `[0.01, 0.03, 0.1]`, 100 epoch, batch 64, probe seed 1–5

모든 방법에 같은 구조와 선택 규칙을 적용했습니다. 구현은
[`src/ibkd_seg/phase05/probe.py`](../src/ibkd_seg/phase05/probe.py)에 있습니다.
Ours는 `fusion_ratio`(λ) `0.5` checkpoint이며 λ `0.25` 결과는 포함하지 않습니다.

## 전체 test 진단값

`224 x 224` 예측의 global confusion matrix를 5개 probe initialization seed로
요약했습니다.

| 방법 | 역할 | mIoU | 전경 IoU | 전경 Dice | Pixel Acc. |
|---|---|---:|---:|---:|---:|
| Ours | 조건 일치 주 진단군 | 82.764 ± 0.034 | 78.179 ± 0.067 | 87.753 ± 0.042 | 91.295 ± 0.013 |
| ALG | 조건 일치 주 진단군 | 82.295 ± 0.041 | 77.559 ± 0.120 | 87.362 ± 0.076 | 91.045 ± 0.020 |
| KD | 조건 불일치 탐색군 | 81.793 ± 0.068 | 76.947 ± 0.122 | 86.972 ± 0.078 | 90.760 ± 0.027 |
| Train mean-mask | 비영상 baseline | 59.127 | 47.715 | 64.604 | 76.409 |
| All-background | 비영상 baseline | 32.109 | 0.000 | 0.000 | 64.219 |

조건이 일치하는 Ours–ALG 차이는 Ours 기준 mIoU `+0.469%p`, 전경 IoU
`+0.620%p`, Dice `+0.392%p`였습니다. 그러나 encoder-training seed가 하나이고
정답이 자동 pseudo-mask이므로 이 차이에 통계적·과학적 의미를 부여하지 않습니다.

## 정성 확인

고정 test ID 7개의 입력–pseudo-mask–예측 panel을 세 방법 모두 확인했습니다.
[실제 그림](reports/QUALITATIVE.md)에서 직접 볼 수 있습니다.

- 정상 pseudo-mask에서는 선형 probe가 대략적인 꽃 위치와 형태를 복원했습니다.
- ID 45는 pseudo-mask가 완전 배경이지만 probe는 실제 꽃 일부를 예측했습니다.
- ID 568은 pseudo-mask가 거의 완전 전경이지만 probe는 중심 물체 형태를
  예측했습니다.

마지막 두 사례는 aggregate score를 ground-truth segmentation 성능으로 해석하면
안 된다는 Phase 0 결론을 다시 확인합니다.

기계가 읽을 수 있는 집계값은 [`reports/summary.json`](reports/summary.json),
checkpoint·선택 epoch·그림 hash는
[`reports/figures/flowers102/manifest.json`](reports/figures/flowers102/manifest.json)에
고정했습니다. 대용량 cache, probe checkpoint와 전체 history는 Git에서 제외합니다.

## 다음 작업

1. Oxford-IIIT Pet 이미지·품종 label·trimap·split byte/hash 및 대응 감사
2. 경계 픽셀 ignore 규칙을 포함한 Phase 1 config 고정
3. 공통 CNN teacher와 Vanilla/KD/LG/ALG/iBKD 분류 encoder H200 학습 config 고정
4. 여러 encoder seed 학습 뒤 동일 frozen probe 적용

Phase 1 결과 전에는 PASCAL VOC나 강한 decoder 단계로 넘어가지 않습니다.
