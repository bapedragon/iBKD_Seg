# 연구 로드맵

## Phase 0 — 기초 검증 및 프로토콜 확정

Flowers-102 공식 파일, 이미지–마스크 대응 관계, 공식 split, 체크포인트 출처,
strict model loading, frozen feature 계약, 세그멘테이션 metric을 검증합니다.

**2026-09-02 결론:** 구현과 입력 계약은 통과했지만 Flowers 자동 마스크는
ground-truth 품질 gate를 통과하지 못했습니다. Phase 1A는 pseudo-mask 기반
파이프라인 진단으로 제한하고, 과학적 검증인 Phase 1B는 신뢰할 수 있는 GT
경로를 확정할 때까지 보류합니다. 근거는 `phase0/DECISION.md`에 있습니다.

## Phase 1 — Frozen spatial probe

각 DeiT-Ti encoder를 고정하고 최종 `14 x 14` feature grid 위에 동일한
`Conv2d(192, 2, 1)` head를 학습합니다. Phase 1A는 기존 Flowers 체크포인트로
작동만 검사하고, Phase 1B는 조건이 일치하는 Pet 분류 encoder를 새로 학습합니다.
Probe와 encoder seed를 구분해 반복하고 foreground IoU, background IoU,
2-class mIoU, Dice를 보고합니다.

Flowers-102에서 배포한 segmentation은 원 분류 파이프라인이 만든 자동 결과이며
완전한 human ground truth가 아닙니다. 이에 따라 Phase 1을 두 단계로 나눕니다.

- **Phase 1A — Flowers pseudo-mask 진단:** 기존 Flowers 체크포인트로 전체 probe
  파이프라인을 검증합니다. 결과는 자동 마스크에 대한 공간 표현 복원성으로만
  해석합니다.
- **Phase 1B — Pet GT probe:** Oxford-IIIT Pet의 품종 라벨만으로 조건이 일치하는
  Vanilla, KD, LG, ALG, iBKD 분류 encoder를 학습합니다. 이후 encoder를 고정하고
  공식 trimap으로 동일한 작은 probe만 학습하여 위치와 형태 정보의 복원성을
  비교합니다. 자세한 목적과 절차는 `phase1/README.md`에 정리합니다.

**2026-09-04 Phase 1A 결과:** 고정된 v1 프로토콜로 전체 공식 split과 Ours/ALG,
탐색용 KD, 5개 probe seed를 실행해 파이프라인 gate를 통과했습니다. 결과는
`phase1/PHASE1A_DECISION.md`에 있으며 pseudo-mask 방법 순위는 과학적 결론으로
사용하지 않습니다. 다음 단계는 Phase 1B Pet 데이터 계약과 H200 분류 encoder
학습 설정 고정입니다.

**종료 조건:** 조건이 일치하는 iBKD–ALG 비교, 비영상 baseline, 정성 mask,
신뢰 가능한 pixel GT 결과를 함께 검토하여 Go/Hold/No-Go를 기록합니다.

## Phase 2 — 공간적 대조 실험

Mean-mask/center-prior, translation, 고정 grid permutation, layer별 probe,
paired bootstrap confidence interval, 여러 encoder seed를 추가합니다.

**종료 조건:** Phase 1의 차이가 단순 분류 성능이나 데이터 위치 편향이 아니라
공간적으로 유의미하고 재현 가능한 신호인지 판단합니다.

## Phase 3 — 공통 decoder

모든 encoder에 동일한 경량 decoder를 사용합니다. Frozen, partial fine-tuning,
full fine-tuning 조건을 분리합니다. 출력 해상도가 충분히 높아진 뒤에만 boundary
평가를 추가합니다.

## Phase 4 — 표준 semantic segmentation

명시적인 data-scarce 비율을 적용한 PASCAL VOC부터 multi-class benchmark로
확장합니다. 조건을 맞춘 Vanilla, KD, LG, ALG, iBKD와 선별한 segmentation KD
baseline을 비교합니다.

## Phase 5 — Dense iBKD

앞선 Phase에서 확장 가능성이 확인된 경우에만 multi-scale grid alignment,
boundary-aware guidance, encoder–decoder feature transfer 같은 dense task 전용
방법을 설계합니다. 이 단계는 단순 분석이 아니라 새로운 방법론 기여입니다.
