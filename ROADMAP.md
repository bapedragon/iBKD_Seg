# 연구 로드맵

## Phase 1~5 핵심 흐름

공간정보 차이의 존재를 확인한 뒤 원인, 실제 활용성, 외부 일반화, 새 방법론
개발 순서로 단계별 gate를 통과합니다.

| Phase | 확인하려는 핵심 | 앞 Phase에서 넘어오는 논리 |
|---|---|---|
| **1. Frozen spatial probe** | iBKD encoder가 LG/ALG보다 위치·형태 정보를 더 잘 보존하는가? | Pet에서 먼저 확인하고 CUB-200-2011로 데이터셋 의존성을 독립 점검 |
| **2. 공간적 대조 실험** | Phase 1 차이가 shortcut이나 우연이 아니라 실제 공간정보 때문인가? | Phase 1에서 iBKD 우위가 관측될 때만 원인과 통계적 안정성을 검증 |
| **3. 공통 decoder** | 작은 probe뿐 아니라 실제 segmentation 학습에서도 우위가 유지되는가? | 표현 자체의 공간성이 확인될 때 실용적인 decoder 조건으로 확장 |
| **4. 표준 segmentation** | Pet에만 국한되지 않고 표준 multi-class segmentation에서도 일반화되는가? | 실제 segmentation 효과가 확인될 때 외부 데이터셋에서 재검증 |
| **5. Dense iBKD** | segmentation에 특화된 새 iBKD 방법을 설계할 가치가 있는가? | 여러 조건에서 확장성이 확인될 때 새로운 방법론으로 발전 |

## Phase 0 — 기초 검증 및 데이터 감사

Flowers-102 공식 파일, 이미지–마스크 대응, 공식 split, 체크포인트 출처, strict
model loading, frozen feature 계약과 세그멘테이션 metric을 검증합니다.

**2026-09-02 결론:** 구현과 입력 계약은 통과했지만 Flowers 자동 마스크는
ground-truth 품질 gate를 통과하지 못했습니다. Flowers 실행은 Phase 0.5 진단으로
제한하고, 과학적 검증에는 신뢰할 수 있는 pixel GT를 사용합니다. 근거는
[phase0/DECISION.md](phase0/DECISION.md)에 있습니다.

## Phase 0.5 — Flowers pseudo-mask 파이프라인 진단

기존 Flowers Ours/ALG checkpoint와 조건 불일치 탐색용 KD checkpoint를 고정하고,
최종 `192 x 14 x 14` feature에 동일한 `Conv2d(192, 2, 1)` probe를 붙여 전체
데이터·cache·학습·선택·평가·시각화 경로를 검사합니다.

**2026-09-04 결과:** 공식 train/validation/test 전체와 probe seed 5개를 실행해
모든 파이프라인 gate를 통과했습니다. 정량값과 실제 Flowers panel은
[phase0.5/DECISION.md](phase0.5/DECISION.md)에 연결되어 있습니다. 자동 mask에
대한 방법 순위는 논문의 과학적 결론으로 사용하지 않습니다.

**종료 조건:** finite 학습, encoder 고정, validation-only 선택, test-once 정책,
정량 metric과 고정 정성 표본 생성이 모두 확인되면 완료합니다.

## Phase 1-PET — Oxford-IIIT Pet GT frozen spatial probe

**2026-09-07 결론:** 12-way timing 뒤 student batch `64/128`과 iBKD λ
`0.25/0.5`를 모두 사전 고정했습니다. 두 batch의 teacher와 여섯 variant × 3 seed
분류가 각각 19/19 완료됐고 test-once·동일 초기화·동일 teacher·동일 split 계약과
checkpoint 감사를 통과했습니다. 두 frozen probe profile의 선택 probe 180개도
전체 산출물 감사를 통과했습니다. Batch 64에서는 iBKD–ALG 차이가 두 λ 모두
음수였지만, batch 128에서는 두 λ 모두 큰 양수로 방향이 뒤집혔습니다. 동시에
LG는 두 profile 모두 1위였고, epoch 2에 guidance를 종료한 batch 128 ALG의
probe만 batch 64보다 18.713%p 급락했습니다. Controller 종료 판정 warm-up만
`0 → 20`으로 바꾼 사후 진단에서 ALG mIoU가 `80.947%`로 회복되어 iBKD 두
λ보다 높았습니다. 따라서 iBKD의 LG/ALG 우위는 지지되지 않았으며 Phase 1-PET은
**No-Go**, 원래 전제의 Phase 2 진입은 **보류**합니다. 근거는
[분류 profile 비교](phase1/phase1_pet/reports/classification/BATCH_PROFILE_COMPARISON.md),
[batch 64 probe 결과](phase1/phase1_pet/reports/frozen_probe/batch64/RESULTS.md),
[batch 128 결과](phase1/phase1_pet/reports/frozen_probe/batch128/RESULTS.md),
[ALG warm-up 20 진단](phase1/phase1_pet/reports/diagnostics/alg_controller_warmup20_b128/RESULTS.md),
[Phase 1-PET 결정문](phase1/phase1_pet/DECISION.md)에 있습니다. 상세 계약은
[phase1/phase1_pet/PROTOCOL.md](phase1/phase1_pet/PROTOCOL.md)에 있습니다.

Oxford-IIIT Pet의 품종 라벨만 사용해 조건이 일치하는 Vanilla, KD, LG, ALG,
iBKD 분류 encoder를 학습합니다. 이후 모든 encoder를 고정하고 공식 trimap에
동일한 작은 probe만 학습하여 위치와 형태 정보의 선형 복원성을 비교합니다.

Probe와 encoder-training seed를 구분해 반복하고 foreground IoU, background IoU,
2-class mIoU, Dice를 보고합니다. 경계 픽셀 ignore 규칙, split, teacher,
checkpoint 선택과 모든 방법별 고정값은 결과 확인 전에 v1 config로 확정합니다.
자세한 목적과 절차는 [phase1/phase1_pet/README.md](phase1/phase1_pet/README.md)에 있습니다.

**종료 조건:** 조건이 일치하는 iBKD–ALG 비교, 여러 encoder seed, 비영상 baseline,
정성 mask와 공식 pixel GT 결과를 함께 검토해 Go/Hold/No-Go를 기록합니다.

## Phase 1-CUB-200 — CUB-200-2011 독립 반복

**상태: ResNet-50/224 scratch 본실험 v3 LOCK, guided seed-1 부분 결과 감사 완료.** Pet
결과가 특정 데이터셋에만 의존한 것인지 확인하기 위해 CUB-200-2011에서 분류
encoder 학습과 frozen segmentation probe를 독립적으로 반복합니다. Pet의 잠긴
config나 결과를 섞지 않으며, CUB용 데이터·split·mask·ResNet-50 teacher·학습·평가
계약을 full v3로 잠갔습니다. 결과와 관계없이 여섯 설정×세 encoder seed를 모두
수행하고 설정을 바꾸지 않습니다. Smoke 시간으로 10시간 이내 shard만 정하며,
teacher checkpoint 하나는 모든 guided student와 후속 baseline이 공유합니다. 세부 기록은
[phase1/phase1_cub/README.md](phase1/phase1_cub/README.md)입니다.

Guided smoke는 `17/17`로 통과했고, v3 ResNet-50/224 scratch Teacher는 H200 issue
722에서 200 epoch 학습과 독립 감사를 완료했습니다. H200 issue 727에서는 이
Teacher를 공유한 LG, ALG-w20, iBKD λ=0.25/0.5의 batch-128/64 encoder seed 1
분류·frozen probe를 완료하고 48개 새 checkpoint를 감사했습니다. Seed 1에서는
두 batch 모두 LG probe가 가장 높았고 iBKD λ=0.25의 ALG 대비 방향은 batch에 따라
달랐습니다. 따라서 아직 iBKD 우위를 주장하지 않으며, batch-128 seed 2·3과
Vanilla/KD를 같은 잠긴 계약으로 채운 뒤 최종 판정합니다. 상세 수치는
[seed-1 결과 보고서](phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_b64_guided_seed1_v4/RESULTS.md)에 있습니다.

이전 ResNet-56/32 v2 guided shard도 H200 issue 716에서 완료됐지만, v3로 교체한 뒤
회수한 구버전 결과이므로 현재 결과와 합치지 않고 참고 자료로만 보존합니다.

## Phase 2 — 공간적 대조 실험

**상태: 진입 보류.** Phase 1에서 검증하려던 iBKD의 LG/ALG 우위가 관측되지 않아
현재 형태의 원인 검증을 진행할 전제가 없습니다. Phase 1-CUB-200 같은 독립
반복에서 양의 현상이 확인될 경우에만 재개합니다.

Mean-mask/center-prior, translation, 고정 grid permutation, layer별 probe, paired
bootstrap confidence interval과 추가 encoder seed를 사용해 Phase 1 차이가 실제
공간 신호인지 확인합니다.

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
