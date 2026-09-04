# Phase 1 — Oxford-IIIT Pet 공간 표현 검증

상태: **본 실험 프로토콜 준비 중**

## 핵심 질문

> 품종 분류 정답만으로 학습한 iBKD encoder에 동물의 위치와 형태 정보가
> Vanilla, KD, LG, ALG보다 선형적으로 더 쉽게 읽히는 상태로 남아 있는가?

[Phase 0.5](../phase0.5/README.md)는 Flowers 자동 pseudo-mask로 실행 경로만
확인했습니다. Phase 1은 Oxford-IIIT Pet의 공식 pixel-level trimap을 사용해 실제
공간정보 보존 여부를 판단하는 첫 본 실험입니다.

## 실험 흐름

Oxford-IIIT Pet에는 각 이미지의 품종 label과 pixel trimap이 함께 있습니다.
먼저 모델에는 mask를 보여주지 않고 품종 label만으로 분류 encoder를 학습합니다.
그 뒤 분류 head를 제거하고 encoder를 고정한 다음, 모든 방법에 같은 작은
segmentation probe를 붙입니다.

```text
Pet 이미지 + 품종 라벨
        ↓ 조건을 맞춘 분류 학습
Vanilla / KD / LG / ALG / iBKD encoder
        ↓ 분류 head 제거 및 encoder 고정
최종 feature [B, 192, 14, 14]
        ↓ 동일한 Conv2d(192, 2, 1) probe만 trimap으로 학습
동물 / 배경 예측 비교
```

iBKD 결과가 여러 encoder seed에서 일관되게 높다면, 분류만 학습했는데도 iBKD
feature에 위치·형태 정보가 더 선형적으로 읽기 쉬운 형태로 남았다는 근거가 됩니다.

## 본 실험 전에 고정할 계약

1. 공식 이미지, 품종 label, trimap과 split의 byte size·SHA-256 및 1:1 대응
2. trimap의 동물/배경 정의와 애매한 경계 픽셀 ignore 규칙
3. 공통 CNN teacher, DeiT-Tiny student와 분류 학습 schedule
4. Vanilla/KD/LG/ALG/iBKD의 방법별 고정값과 동일성 범위
5. encoder-training seed, checkpoint 선택 규칙과 test 접근 정책
6. frozen feature layer, normalization 여부와 cache 계약
7. 공통 probe 초기화, LR grid, epoch, seed와 validation 선택 규칙
8. foreground/background IoU, 2-class mIoU, Dice와 비영상 baseline

결과를 바꿀 수 있는 항목이 남아 있으면 본 실험을 시작하지 않습니다. 모든 항목을
문서와 v1 config로 고정하고 smoke gate를 통과한 뒤 H200 실행으로 넘어갑니다.

## 비교가 뜻하는 것

| 비교 | 확인하는 내용 |
|---|---|
| iBKD vs Vanilla | CNN teacher guidance가 공간정보 보존에 도움이 되는가? |
| iBKD vs KD | 최종 예측 전달보다 grid를 유지한 전달이 유리한가? |
| iBKD vs LG | 학습 가능한 정렬이 고정된 공간 정렬보다 유리한가? |
| iBKD vs ALG | iBKD 추가 구조가 adaptive guidance보다 유리한가? |

가장 가까운 주 비교는 조건이 일치한 iBKD–ALG입니다. 조건이 다른 checkpoint나
pilot 결과는 별도의 탐색 결과로 표시합니다.

## 해석 범위

긍정적 결과는 iBKD feature의 **공간정보 선형 복원성**이 더 좋다는 근거입니다.
iBKD가 완성된 segmentation 모델이나 세그멘테이션 전용 KD보다 우수하다는 뜻은
아닙니다. 이후 Phase에서 공간 대조 실험, 공통 decoder fine-tuning, PASCAL VOC와
세그멘테이션 전용 KD 비교가 필요합니다.

## 계산 자원

- 로컬: Pet 데이터 감사, 단위 테스트, smoke probe와 정성 확인
- H200: 공통 teacher, 다섯 분류 encoder의 multi-seed 학습과 전체 probe 반복
