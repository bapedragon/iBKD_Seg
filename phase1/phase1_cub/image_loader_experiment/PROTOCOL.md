# CUB 이미지 loader 실험 프로토콜

상태: **Stage A·B 완료 로그 확인, L2 선택, archive 감사 대기 및 Stage C smoke 준비**

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
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_damage_audit.sh
```

완료 gate는 `train=5,394`, profile별 draw `26,970`, profile summary `3/3`,
`L0/L1 geometry exact match=true`, finite metric,
`official_test_images_or_masks_decoded=0`입니다.

H200 issue 745에서 91.27초에 완료 gate를 모두 통과했습니다. L0/L1의 visible-part
보존은 `69.3335%`, foreground-mask 보존은 `69.5441%`였고, L2에서는 각각
`94.0904%`, `93.6316%`로 높아졌습니다. Background-only crop은 L0/L1
`2.1691%`에서 L2 `0.0037%`로 감소했습니다. 상세 해석은
[Stage A 결과 보고서](reports/damage_audit_v1/RESULTS.md)에 보존합니다.
이 수치만으로 loader를 선택하지 않고 예정대로 Stage B로 진행합니다.

## Stage B — seed-1 validation-only loader pilot

Stage A 결과를 회수·감사한 뒤 잠금 config와 별도 smoke로 실행 경로와 시간을
먼저 확인합니다. H200 issue 746의 v1은 분류 12개와 checkpoint 12개까지 모두
완료했지만, 첫 segmentation probe를 만들 때 runtime schema의 `initialization`
필드가 없어 후보 학습 전에 중단됐습니다. Probe 후보, part, CKA, attention 및
official-test 평가는 모두 0회이므로 v1은 과학적 결과가 아니며 loader 비교에도
사용하지 않습니다. 실패 내역은
[issue 746 실패 기록](reports/failed_smoke_v1_issue746/README.md)에
보존합니다.

- 재실행 Config: `configs/cub200_r50_224_b128_loader_pilot_smoke_v2.json`
- SHA-256: `8ea14d480d64dcc6ad1ab2fafd2754bc22c29867d0b8a20126bcd4ebab537a5f`
- 실행:

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_smoke_b128_seed1.sh
```

v2는 v1의 방법·metric·선택 규칙을 바꾸지 않고 기존 공통 probe 구현이 요구하는
초기화, parameter count, optimizer, scheduler, loss schema만 완성한 기술 수정입니다.
새 컨테이너에서는 issue 746의 임시 출력에 의존하지 않고 12개 분류부터 동일하게
재실행합니다. Smoke는 2-epoch 최신 checkpoint로 전체 경로만 검사하고 official
test를 열지 않습니다. 완료 gate는 분류 checkpoint `12`, segmentation LR 후보
`36`, part LR 후보 `36`, CKA 값 `144`, attention row `12`, 정성 PNG `48`,
official-test 평가 `0`입니다.

V2 smoke는 모든 gate를 통과했습니다. 결과는
[smoke v2 보고서](reports/smoke_v2/RESULTS.md)에 보존하며, 전체
본 pilot의 보수적 외삽이 14시간 54분 8초이므로 10시간 제한을 피하기 위해 다음
세 독립 작업으로 고정했습니다.

- Full config: `configs/cub200_r50_224_b128_seed1_loader_pilot_full_v1.json`
- SHA-256: `97adb9274a4f1a996932915dbb8a715a719b2ae6777aceb40201e96174cbe5a4`
- 작업: L0, L1, L2를 각각 한 이슈에서 실행
- profile별 보수적 예상: 4시간 58분 3초

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_b128_seed1.sh l0_current_strong
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_b128_seed1.sh l1_matched_weak
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_b128_seed1.sh l2_conservative_spatial
```

각 결과는 단독으로 loader를 선택할 수 없으며 세 shard가 모두 완료되어야 합니다.
세 작업 모두 분류와 모든 probe를 validation-only로 수행하고 official test를 열지
않습니다.

### L0 실행시간 확인 뒤의 운영 분할 점검

L0 본 pilot 로그에서 전체 경로가 `3시간 24분 14초`에 완료됐습니다. 이 시간만
H200의 10시간 제한을 피하기 위한 운영 정보로 사용하며, L0의 성능값은 작업 분할이나
protocol 변경에 사용하지 않습니다. 남은 L1·L2를 한 이슈에서 연속 실행할 수 있는지
확인하기 위해 다음 subset smoke를 먼저 수행합니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_smoke_l1_l2_b128_seed1.sh
```

이는 이미 잠긴 v2 smoke에서 L1·L2만 실행하는 운영 점검입니다. 방법, encoder seed,
학습 epoch, probe seed/LR/epoch, validation 선택 규칙, loader 선택 규칙은 바꾸지
않으며 official test도 열지 않습니다. H200 issue 750에서 418.26초에 분류 `8/8`,
segmentation/part 후보 각 `24/24`, 선택 각 `8/8`, CKA `96/96`, attention `8/8`,
정성 PNG `32/32`, official test `0` gate를 통과했고 OOM이나 비정상 종료는 없었습니다.
이 smoke의 성능값은 loader 선택에 사용하지 않습니다.

Smoke 자체의 단순 선형 상한은 `10시간 4분 19초`였지만, 같은 실행기의 L0 상한
`4시간 58분 3초` 대비 실제 `3시간 24분 14초`의 비율 `0.6852`를 운영시간에만
적용하면 L1·L2 합산 예상은 약 `6시간 54분 6초`입니다. 따라서 두 profile을 한
H200 이슈에서 **순차 실행**하도록 운영 분할만 갱신합니다. 최초 full config의
학습·probe·평가·선택 규칙과 SHA-256은 그대로 두며, 변경된 작업 포장은
`configs/cub200_r50_224_b128_seed1_loader_pilot_l1_l2_combined_job_v1.json`에
별도로 잠급니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_loader_pilot_full_l1_l2_b128_seed1.sh
```

L1과 L2는 서로 다른 출력·cache·summary·checkpoint 디렉터리를 사용합니다. 마지막
결합 감사가 두 shard의 원래 completion gate, teacher/checkpoint hash와 총 8개
방법 결과를 재검산합니다. 결합 실행이 완료돼도 L0 archive까지 함께 감사하기 전에는
loader를 선택하지 않습니다.

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

## Stage C0 — 선택 L2의 guided 4방법 × seed 1 예비실험

사전에 고정한 평균 Part PCK 선택 규칙에서 L2가 `0.360609`로 L0 `0.246660`,
L1 `0.217878`보다 높았습니다. 전체 표와 제한은
[Stage B 로그 결과](reports/full_v1_log_snapshot/RESULTS.md)에 기록합니다. 다만
결과 archive와 checkpoint manifest 감사가 아직 남았으므로 이 후속 결과는 최종
논문 결과가 아닌 단일-seed 예비 결과로 한정합니다.

- 방법: LG, ALG-w20, iBKD λ=0.25, iBKD λ=0.5
- encoder seed: 1
- loader: L2 `l2_conservative_spatial`
- 나머지 분류·frozen segmentation probe 설정: 기존 ResNet-50/224 v3와 동일
- 결과 역할: 단일-seed 예비 결과이며 최종 6방법×3seed 확증 결과가 아님
- test: 모든 validation 선택 후 선택 checkpoint당 정확히 한 번만 평가

Smoke config와 예정 full config는 각각
`configs/cub200_r50_224_b128_seed1_l2_guided_preliminary_smoke_v1.json`,
`configs/cub200_r50_224_b128_seed1_l2_guided_preliminary_full_v1.json`에 고정합니다.
SHA-256은 각각
`484783f30c795ecfb8eb1c8766ccc466a8c266ac0e8359fbe272531b310287b6`,
`1ecef8999e7233f1f2dc5c960394fbcb2f8892ae85df6f5de98a36bacd51a7b4`입니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_l2_guided_preliminary_smoke_b128_seed1.sh
```

Smoke는 분류 4개 × 2 epoch와 probe seed 1 × LR 3개 × 2 epoch를 실행하고,
분류 및 선택 probe의 official-test 경로를 각각 4회 확인합니다. 이 2-epoch 수치로
방법, lambda, protocol 또는 이후 실행 여부를 바꾸지 않습니다.

H200 issue 753 smoke는 `classification=4/4`, `probe_candidates=12/12`,
`selected_probes=4/4`, `tasks=16/16`, `status=pass`, `312.17초`로 완료됐습니다.
성능값은 선택에 사용하지 않았으며, smoke 전에 잠근 과학적 설정을 바꾸지 않은 실행
manifest는
`configs/cub200_r50_224_b128_seed1_l2_guided_preliminary_full_execution_v1.json`
에 기록합니다. SHA-256은
`d1138ac889fd6bd7ec6503776be9444d9e7915ec37d0eb0c4ac1692a63c24907`입니다.

```bash
bash phase1/phase1_cub/image_loader_experiment/scripts/run_r50_224_l2_guided_preliminary_full_b128_seed1.sh
```

이 본학습은 분류 4개 × 300 epoch, frozen probe 20개 선택을 위한 LR 후보 60개를
실행합니다. 모든 validation 선택 후 분류 checkpoint 4개와 probe checkpoint 20개를
official test에서 각각 한 번만 평가하고, 새 checkpoint 24개를 보존합니다. Stage B
archive 감사 전에도 예비 실행은 가능하지만, 최종 논문 주장에는 해당 감사가 필요합니다.

## Stage C1 — 향후 최종 확증 실험

Stage C0와 별개로 최종 주장을 하려면 Vanilla, KD를 포함한 6방법과 encoder seed
`[1,2,3]`을 동일 L2 조건에서 모두 다시 수행해야 합니다. 기존 v3 결과나 config를
덮어쓰거나 L0 checkpoint와 섞지 않습니다. 현재 4방법×seed 1 범위는 이 최종
matrix를 대신하지 않습니다.

## 해석 규칙

- L2에서 iBKD의 상대 격차가 일관되게 개선되면 강한 crop이 공간학습을 가렸다는
  근거가 됩니다.
- 모든 방법이 비슷하게 개선되고 LG 우위가 유지되면 loader는 과제를 개선했지만
  iBKD 고유 장점의 근거는 아닙니다.
- 변화가 거의 없으면 loader보다 feature alignment, loss 또는 fusion 설계를 먼저
  점검합니다.
- bbox/mask-aware crop은 추가 GT를 쓰므로 이 주 비교에 넣지 않고 별도 oracle
  diagnostic으로만 수행합니다.
