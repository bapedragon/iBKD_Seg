# Phase 1 — 고정된 공간 표현 검사

## 한눈에 보기

Phase 1의 질문은 다음과 같습니다.

> 분류만 학습한 iBKD encoder 안에, 세그멘테이션에 필요한 위치와 형태 정보가
> Vanilla, KD, LG, ALG보다 더 잘 남아 있는가?

Phase 1은 두 단계로 진행합니다.

- **Phase 1A — Flowers 작동 검사:** 자동 마스크로 데이터 처리, 학습, metric,
  시각화가 정상인지 확인합니다. 방법 간 순위는 과학적 결론으로 사용하지 않습니다.
- **Phase 1B — Pet 확장성 검사:** 공식 pixel-level trimap을 가진
  Oxford-IIIT Pet에서 실제 공간정보 보존 여부를 판단합니다.

Phase 1B는 Flowers에서 iBKD가 더 높은 점수를 얻어야 시작하는 것이 아닙니다.
Phase 1A의 **파이프라인이 정상 작동하면**, Flowers 점수와 관계없이 진행합니다.

## Phase 1B에서 무엇을 검사하는가?

Oxford-IIIT Pet의 각 이미지에는 품종 라벨과 픽셀 마스크가 함께 있습니다.
예를 들어 Bengal 고양이 사진에는 다음 두 정답이 존재합니다.

- 분류 정답: `Bengal`
- 마스크 정답: 각 픽셀이 `동물`, `배경`, `경계` 중 어디에 해당하는지

먼저 모델에는 마스크를 보여주지 않고 `Bengal`이라는 분류 정답만으로 encoder를
학습합니다. 학습이 끝나면 encoder를 고정하고, 모든 방법에 똑같은 작은
segmentation probe를 붙여 동물과 배경을 구분하게 합니다.

```text
Pet 이미지 + 품종 라벨
        ↓ 분류 학습
Vanilla / KD / LG / ALG / iBKD encoder
        ↓ encoder 고정
동일한 작은 segmentation probe
        ↓ 마스크로 probe만 학습
동물 / 배경 예측 비교
```

iBKD의 probe 결과가 더 좋다면, 품종 분류만 학습했는데도 iBKD encoder에 동물의
위치와 형태 정보가 더 쉽게 읽을 수 있는 상태로 남았다는 근거가 됩니다.

여기서 **encoder**는 사진을 특징으로 바꾸는 모델의 핵심 부분이고,
**probe**는 그 특징에 공간정보가 남았는지 읽어 보는 작은 검사 도구입니다.
**고정(freeze)**은 probe를 학습하는 동안 encoder를 바꾸지 않는다는 뜻입니다.

## 수행 과정

1. **Pet 데이터 계약 고정**
   - 이미지, 품종 라벨, trimap, split의 대응을 검사합니다.
   - 주 실험은 `동물/배경` 이진 분할로 하고 애매한 경계 픽셀은 제외합니다.
2. **공통 teacher 준비**
   - KD, LG, ALG, iBKD는 같은 Pet 분류용 CNN teacher를 사용합니다.
3. **분류 encoder 학습**
   - 동일한 DeiT-Ti와 학습 조건으로 Vanilla, KD, LG, ALG, iBKD를 학습합니다.
   - 이 단계에서는 품종 라벨만 사용하고 마스크는 사용하지 않습니다.
4. **Encoder 고정**
   - 분류 head를 제거하고 encoder parameter가 학습되지 않도록 고정합니다.
5. **동일한 probe 학습**
   - 최종 `192 x 14 x 14` feature에 동일한 `Conv2d(192, 2, 1)`을 붙입니다.
   - encoder는 그대로 두고 386개 parameter의 probe만 마스크로 학습합니다.
6. **공통 기준으로 평가**
   - foreground IoU, background IoU, 2-class mIoU, Dice를 비교합니다.
   - 여러 seed, 단순 center/mean-mask baseline, 예측 마스크 시각화를 함께 봅니다.
7. **Go / Hold / No-Go 결정**
   - 특히 가장 가까운 비교 대상인 ALG보다 iBKD가 여러 seed에서 일관되게 좋은지
     확인한 뒤 다음 단계 진행 여부를 결정합니다.

## 비교가 뜻하는 것

| 비교 | 확인하는 내용 |
|---|---|
| iBKD vs Vanilla | CNN teacher의 guidance가 공간정보 보존에 도움이 되는가? |
| iBKD vs KD | 최종 예측 전달보다 grid를 유지한 전달이 유리한가? |
| iBKD vs LG | 학습 가능한 정렬이 고정된 공간 정렬보다 유리한가? |
| iBKD vs ALG | iBKD의 추가 구조가 adaptive guidance보다 유리한가? |

## 결과를 어디까지 해석할 수 있는가?

Phase 1B의 긍정적 결과는 **iBKD feature의 공간정보 복원성이 더 좋다**는 근거입니다.
아직 iBKD가 실제 세그멘테이션 모델이나 세그멘테이션 전용 KD보다 우수하다는 뜻은
아닙니다. 이후 Phase에서 공통 decoder fine-tuning, 공간 대조 실험, PASCAL VOC와
세그멘테이션 전용 KD 비교를 진행해야 합니다.

## 계산 자원

- 로컬: 데이터 감사, Phase 1A, 단위 테스트, 작은 probe 점검
- H200: Pet teacher와 조건이 일치하는 다섯 분류 encoder의 multi-seed 학습

## Phase 1A 실행

결과 확인 전에 고정한 전체 계약은 [PROTOCOL.md](PROTOCOL.md), 실행값은
[`configs/flowers102_phase1a_v1.json`](configs/flowers102_phase1a_v1.json)에
있습니다. 먼저 Ours checkpoint의 16/16/16 표본으로 2-epoch smoke test를
실행합니다.

```bash
bash phase1/scripts/run_phase1a_smoke.sh /path/to/IBAM_KD_H200_V2
```

smoke gate가 통과하면 Ours/ALG와 탐색용 KD의 전체 공식 split을 실행합니다.

```bash
bash phase1/scripts/run_phase1a_full.sh /path/to/IBAM_KD_H200_V2
```

feature와 target cache는 `phase1/results/raw/cache/`, 로컬 JSON 보고서는
`phase1/reports/*.local.json`에 저장되며 Git에는 포함되지 않습니다.

**2026-09-04 실행 상태:** smoke와 전체 공식 split 실행이 모두 통과했습니다.
진단값과 해석 제한은 [PHASE1A_DECISION.md](PHASE1A_DECISION.md), 추적 가능한
소형 수치는 [`reports/phase1a_summary.json`](reports/phase1a_summary.json)에
고정했습니다. 실제 Flowers 입력, pseudo-mask와 probe 예측은
[`reports/PHASE1A_QUALITATIVE.md`](reports/PHASE1A_QUALITATIVE.md)에서 함께 볼 수
있습니다.
