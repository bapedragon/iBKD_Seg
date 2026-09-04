# Phase 1 — Oxford-IIIT Pet 공간 표현 검증

상태: **12-way timing 완료 — batch 64/128 full classification 실행 준비**

현재 LOCK한 프로토콜과 full-run 계약은 [PROTOCOL.md](PROTOCOL.md), 기계가 읽을 수
있는 설정은
[`configs/oxford_iiit_pet_phase1_v1.json`](configs/oxford_iiit_pet_phase1_v1.json)에
있습니다.

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

## 고정한 full classification matrix

데이터 split, test-once, 모델 구조, metric과 probe 방식을 유지합니다. timing
결과에서 12개 조합 모두 OOM 없이 실행되고 batch 64/128의 시간이 비슷했으므로,
성능 수치를 보기 전에 다음 두 batch profile과 두 iBKD λ를 모두 실행·보고하기로
고정했습니다.

```text
(Vanilla, KD, LG, ALG, iBKD-0.25, iBKD-0.5)
× (batch 64, batch 128) × encoder seed (1, 2, 3)
= student 36 runs
```

H200 요청은 batch 64와 batch 128로 나눕니다. 각 요청은 공통 teacher 1회와 해당
batch의 student 18회로 구성합니다. timing 환산 기준 예상시간은 각각 약 8시간
39분과 8시간 20분입니다. 결과가 좋은 batch나 λ만 골라 main 결과로 바꾸지 않고
두 profile을 별도 표로 모두 남깁니다.

## 본 실험 초안에 포함된 공통 계약

1. 공식 이미지, 품종 label, trimap과 split의 byte size·SHA-256 및 1:1 대응
2. trimap의 동물/배경 정의와 애매한 경계 픽셀 ignore 규칙
3. 공통 CNN teacher, DeiT-Tiny student와 분류 학습 schedule
4. Vanilla/KD/LG/ALG/iBKD의 방법별 고정값과 동일성 범위
5. encoder-training seed, checkpoint 선택 규칙과 test 접근 정책
6. frozen feature layer, normalization 여부와 cache 계약
7. 공통 probe 초기화, LR grid, epoch, seed와 validation 선택 규칙
8. foreground/background IoU, 2-class mIoU, Dice와 비영상 baseline

위 항목 중 batch와 λ를 제외한 초안은 2026-09-05에 기록했습니다. 공식 `trainval` 3,680장은 품종별
20장의 고정 validation을 떼어 `train/validation=2,940/740`으로 사용하고, 공식
test 3,669장은 최종 평가 전까지 선택 과정에서 사용하지 않습니다. 분류 encoder는
seed `[1,2,3]`, probe는 각 encoder마다 seed `[1,2,3,4,5]`를 사용합니다.

full-data timing의 runtime만 사용해 H200 작업을 두 개로 나눴습니다. 각 실행 시작
전에 이미지 archive SHA-256, image-label-trimap 대응, split manifest와 method
contract를 검사합니다. 분류 validation으로 checkpoint를 고른 뒤에만 official test를
각 checkpoint당 정확히 한 번 평가합니다.

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
- H200: 완료한 12-way timing, 두 batch profile의 분류 encoder 학습, 이후 전체 probe 반복
