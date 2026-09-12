# CUB 이미지 loader 실험 프로토콜

상태: **Stage A loader 손상 감사 v1 완료 — Stage B smoke 준비**

이 실험은 완료된 CUB ResNet-50/224 v3 결과를 바꾸지 않는 사후 탐색 실험입니다.
기존 결과는 그대로 보존하고 새 결과에는 모두 `loader_pilot`을 표시합니다. 최종
loader를 정하기 전에는 official test 이미지·분류 정답·mask·part를 열지 않습니다.

## 확인하려는 질문

현재 학생 loader의 강한 증강이 새의 위치·형태 단서를 지나치게 제거해 iBKD의 공간
표현 학습을 방해했는지 확인합니다. 다음 두 요인을 분리합니다.

- L0 ↔ L1: crop 기하는 동일하고 color jitter, RandAugment, random erasing만 제거
- L1 ↔ L2: 광학 증강은 동일하게 약하고 crop 최소 면적만 `0.08`에서 `0.5`로 변경

## 고정 loader 후보

| ID | RandomResizedCrop scale | ratio | flip | 광학 증강 | 역할 |
|---|---:|---:|---:|---|---|
| L0 `l0_current_strong` | `[0.08, 1.0]` | `[0.75, 1.3333…]` | 0.5 | jitter 0.4 + RandAugment + erase 0.25 | 완료된 v3 기준 |
| L1 `l1_matched_weak` | `[0.08, 1.0]` | `[0.75, 1.3333…]` | 0.5 | 없음 | 광학 강도 ablation |
| L2 `l2_conservative_spatial` | `[0.5, 1.0]` | `[0.75, 1.3333…]` | 0.5 | 없음 | crop 강도 ablation |

세 조건 모두 bicubic 224×224, horizontal flip 0.5, ImageNet normalization을
사용합니다. Guided 학습에서는 student와 teacher가 **동일한 crop/flip 결과 tensor**를
받습니다. 독립적인 random crop을 두 모델에 적용하지 않습니다.

## Stage A — 학습 없는 crop 손상 감사

실행 config는
`configs/cub200_r50_224_loader_damage_audit_v1.json`이며 SHA-256은
`6eeff24ba01b188d00f3f8ec7e4de533cd4b531eb95bda352b36fdd9fb9695bf`입니다.

- 대상: derived train 5,394장만 사용하며 이미지마다 5회 crop을 뽑습니다.
- 진단 annotation: mask, bbox, part location은 손상량 계산에만 쓰고 모델 입력이나
  학습에는 쓰지 않습니다.
- 높은 것이 좋은 지표: visible-part retention, 모든 visible part 보존 비율,
  foreground-mask retention, bbox coverage
- 낮은 것이 좋은 지표: foreground 비율 1% 미만 crop 비율, background-only crop 비율
- L0와 L1은 같은 geometry ID와 동일 난수를 사용하므로 crop 관련 수치가 byte-level로
  같아야 합니다.
- 정성 그림은 결과와 무관하게 클래스 번호가 작은 8개 클래스에서 train image ID가
  가장 작은 한 장씩을 고릅니다.
- official validation은 split hash 확인에만 쓰며 감사 metric에는 포함하지 않습니다.
  Official test 이미지는 decode하지 않고 test probe/evaluation record도 만들지 않습니다.
- 이 단계 결과만으로 loader를 선택하지 않습니다.

실행:

```bash
bash phase1/phase1_cub/scripts/run_r50_224_loader_damage_audit.sh
```

완료 gate는 `train=5,394`, profile별 draw `26,970`, profile summary `3/3`,
`L0/L1 geometry exact match=true`, finite metric,
`official_test_images_or_masks_decoded=0`입니다.

H200 issue 745에서 91.27초에 완료 gate를 모두 통과했습니다. L0/L1의 visible-part
보존은 `69.3335%`, foreground-mask 보존은 `69.5441%`였고, L2에서는 각각
`94.0904%`, `93.6316%`로 높아졌습니다. Background-only crop은 L0/L1
`2.1691%`에서 L2 `0.0037%`로 감소했습니다. 상세 해석은
[Stage A 결과 보고서](reports/loader_pilot/damage_audit_v1/RESULTS.md)에 보존합니다.
이 수치만으로 loader를 선택하지 않고 예정대로 Stage B로 진행합니다.

## Stage B — seed-1 validation-only loader pilot

Stage A 결과를 회수·감사한 뒤 다음 잠금 config와 별도 smoke로 실행 경로와 시간을
먼저 확인합니다.

- Config: `configs/cub200_r50_224_b128_loader_pilot_smoke_v1.json`
- SHA-256: `db95bf6eb04410e1da2cfffcc97887086f36c0cc7f766f3f5ba407af417785dc`
- 실행:

```bash
bash phase1/phase1_cub/scripts/run_r50_224_loader_pilot_smoke_b128_seed1.sh
```

Smoke는 2-epoch 최신 checkpoint로 전체 경로만 검사하고 official test를 열지
않습니다. 완료 gate는 분류 checkpoint `12`, segmentation LR 후보 `36`, part LR
후보 `36`, CKA 값 `144`, attention row `12`, 정성 PNG `48`, official-test 평가
`0`입니다.

그 다음 L0/L1/L2 각각에서 LG, ALG-w20, iBKD λ=0.25, iBKD λ=0.5를 encoder seed
1로 동일하게 학습합니다. Scratch ResNet-50/224 issue 722 teacher, batch 128, 학생
300 epoch와 기존 optimizer/controller 값은 바꾸지 않습니다.

각 encoder에는 다음을 동일하게 적용합니다.

1. validation macro top-1으로 분류 checkpoint 선택
2. frozen binary-segmentation probe: 5 probe seeds × LR `[0.01,0.03,0.1]` × 100 epoch
3. frozen part-localization probe: 같은 seed/LR/epoch grid, 주 지표 PCK@0.1
4. validation-only spatial CKA와 attention–GT 보조 분석

Smoke 수치는 구현·시간·메모리 확인 외 어떤 선택에도 쓰지 않습니다. Full pilot의
loader 선택은 결과를 보기 전에 정한 다음 순서를 사용합니다.

1. 네 guided 방법의 **validation Part PCK@0.1 산술평균**이 가장 높은 loader
2. 정확히 동률이면 네 방법의 validation frozen-probe input-224 mIoU 산술평균
3. 다시 정확히 동률이면 더 단순한 loader 순서 `L0 → L1 → L2`

Classification accuracy는 성능 붕괴 여부를 함께 보고하지만 loader 선택의
tie-break로 사용하지 않습니다. CKA와 attention은 해석용 보조 지표이며 선택에
사용하지 않습니다. 이 pilot은 탐색 실험이므로 그 자체를 최종 논문 우위 주장으로
사용하지 않습니다.

## Stage C — 선택 loader 확증 실험

Stage B에서 선택된 loader를 새 버전 protocol로 고정한 뒤 smoke를 통과시키고,
Vanilla, KD, LG, ALG-w20, iBKD λ=0.25/0.5 전부를 encoder seed `[1,2,3]`에서
동일 조건으로 다시 수행합니다. 모든 validation 선택을 완료한 다음에만 official
test를 각 선택 checkpoint당 한 번 평가합니다. L1/L2가 선택되더라도 기존 v3
결과나 config를 덮어쓰거나 섞지 않습니다.

## 해석 규칙

- L2에서 iBKD의 상대 격차가 일관되게 개선되면 강한 crop이 공간학습을 가렸다는
  근거가 됩니다.
- 모든 방법이 비슷하게 개선되고 LG 우위가 유지되면 loader는 과제를 개선했지만
  iBKD 고유 장점의 근거는 아닙니다.
- 변화가 거의 없으면 loader보다 feature alignment, loss 또는 fusion 설계를 먼저
  점검합니다.
- bbox/mask-aware crop은 추가 GT를 쓰므로 이 주 비교에 넣지 않고 별도 oracle
  diagnostic으로만 수행합니다.
