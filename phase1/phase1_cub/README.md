# Phase 1 — CUB-200-2011 공간 표현 검증

상태: **ResNet-50/224 scratch 본실험 v3 LOCK — guided smoke 및 Teacher 본학습·감사 완료**

이 폴더는 Oxford-IIIT Pet 결과와 섞이지 않도록 CUB-200-2011 독립 반복만
관리합니다. 완료된 Pet 실험과 결과는 [phase1_pet](../phase1_pet/README.md)에
그대로 보존합니다.

## 핵심 질문

> CUB-200-2011의 종 분류 label만으로 학습한 iBKD encoder가 Vanilla, KD, LG,
> ALG보다 새의 위치·형태 정보를 더 선형적으로 복원 가능한 형태로 보존하는가?

## 예정한 실험 흐름

1. CUB-200-2011 분류 label로 조건을 맞춘 encoder를 학습합니다.
2. 분류 validation만 사용해 각 encoder checkpoint를 선택합니다.
3. 선택한 encoder를 완전히 고정합니다.
4. 모든 방법에 동일한 segmentation probe를 붙여 pixel mask로 학습합니다.
5. validation으로 probe를 선택한 뒤 test를 최종 평가합니다.

Pet과 질문 및 비교 원칙은 같지만, 데이터 split과 mask 출처를 포함한 CUB용
프로토콜은 별도로 고정합니다. ResNet-56/32 v2 guided shard는 v3 교체 전에 이미
제출되어 완료됐지만, 결과를 회수하기 전에 최종 프로토콜을 scratch
ResNet-50/224 teacher v3로 바꿨습니다. 따라서 v2 결과는 별도 참고 자료로만
보존하며 Pet 결과나 v2 checkpoint를 CUB v3 본실험에 섞지 않습니다.

## 디렉터리

- `configs/`: 결과를 보기 전에 잠근 CUB 전용 machine-readable 설정
- `scripts/`: 데이터 감사, smoke, 본실험 및 결과 정리 진입점
- `reports/classification/`: 200종 분류 결과와 checkpoint manifest
- `reports/frozen_probe/`: frozen segmentation probe 정량·정성 결과
- `results/raw/`: 로컬 원시 산출물 전용 경로이며 Git에는 포함하지 않음

잠긴 계약과 실행 gate는 [PROTOCOL.md](PROTOCOL.md)에 기록합니다.

## 현재 v3: ResNet-50/224 Teacher 본학습 완료

Guided smoke는 `17/17` 작업을 완료해 실행 경로와 메모리·시간 gate를 통과했습니다.
H200 issue 722에서 모든 guided student가 공유할 단 하나의 Teacher checkpoint도
학습하고 독립 감사를 통과했습니다.

```bash
bash phase1/phase1_cub/scripts/run_r50_224_teacher_full.sh
```

이 전용 실행기는 TorchVision ResNet-50을 `weights=None`으로 생성해 ImageNet
pretrained 없이 scratch로 학습합니다. 입력 224×224, train/validation
5,394/600, batch 128, seed 1, 200 epoch 전체를 학습하며 validation macro top-1으로
checkpoint를 고릅니다. 동률이면 이른 epoch를 선택하고, 선택 완료 및 strict
reload 뒤 official test 5,794장을 정확히 한 번 평가합니다. 기존
ResNet-56/32용 `train_full.py`는 호출하지 않습니다.

기본 결과 경로는 `/app/output/phase1_cub_r50_224_teacher_full_v3`입니다. 공유할
`teacher_best_validation.pt`와 파일/model-state SHA-256, epoch별 CSV,
summary·split·protocol JSON 및 `run.log`가 남습니다. 원시 CUB 데이터는
`/app/scratch`에만 두고 결과 폴더에 복사하지 않습니다. 선택 epoch는 165,
official-test macro top-1은 `41.178%`였고 상세 결과와 checkpoint hash는
[Teacher 결과 보고서](reports/classification/resnet50_224_teacher_v3/RESULTS.md)에
고정했습니다. 재사용할 원본은
[checkpoint Release manifest](reports/classification/resnet50_224_teacher_v3/checkpoint_release.json)로
식별합니다. 다음 단계는 이 checkpoint를 공유하는 v3 학생 분류와 frozen probe입니다.

## v3 guided smoke 기록

이 smoke는 scratch ResNet-50/224 teacher 1개와 batch-128 DeiT-Tiny의 LG,
ALG-w20, iBKD λ=0.25, iBKD λ=0.5를 seed 1에서 각각 2 epoch 실행합니다. 이어서
각 encoder를 strict load·완전 동결하고 LR `[0.01, 0.03, 0.1]` probe를 각각
2 epoch 실행합니다.

```bash
bash phase1/phase1_cub/scripts/run_r50_224_guided_smoke_b128.sh
```

최종 v3는 smoke 결과와 관계없이 Vanilla, KD, LG, ALG-w20, iBKD λ=0.25,
iBKD λ=0.5의 6설정 × encoder seed `[1,2,3]`을 모두 수행하며 설정을 변경하지
않는 것으로 사전 고정했습니다. 이에 따라 이번 smoke는 분류 checkpoint와
validation으로 선택한 probe의 official test 경로까지 한 번씩 실행합니다. 단,
2-epoch 분류 정확도와 IoU는 모두 비과학적 진단값이고 선택이나 논문 결과에
사용할 수 없습니다.

기본 결과 경로는 `/app/output/phase1_cub_r50_224_b128_guided_smoke_v3`이며,
마지막 로그에 4개 방법의 validation/test 진단값, peak CUDA memory, 실제 smoke
시간과 4방법×3seed guided block 선형 외삽을 출력합니다. Batch 128 FP32 iBKD의
28×28 cross-attention이 용량 gate였습니다. Full H200에서 `17/17`로
통과했으며, Teacher 본학습은 25분 42초에 완료됐습니다.

## Batch 128·64 guided→probe profile smoke

완료된 issue 722 Teacher 하나를 그대로 공유하면서 LG, ALG-w20, iBKD λ=0.25,
iBKD λ=0.5의 분류→frozen probe 실행 경로를 batch 128과 64에서 모두 점검하는
통합 smoke를 별도로 준비했습니다.

```bash
bash phase1/phase1_cub/scripts/run_r50_224_guided_probe_smoke_b128_b64.sh
```

Teacher release archive와 checkpoint의 SHA-256 및 model-state SHA-256을 검증한
뒤 batch 128을 먼저, batch 64를 다음에 실행합니다. 각 조건은 seed 1, 분류 2
epoch, probe LR `[0.01, 0.03, 0.1]` × 2 epoch이며, 마지막 로그에 두 배치의 전체
결과·peak memory·선형 시간 외삽과 `32/32` 완료 여부가 나옵니다. 이 smoke는
capacity와 실행시간 확인용입니다. 잠긴 v3 주 비교는 계속 batch 128이고, batch
64 수치나 2-epoch 정확도·IoU로 설정을 선택하거나 논문 결론을 내리지 않습니다.

## Batch 128·64 guided seed-1 full-epoch profile

Smoke에서 확인한 정확히 같은 범위를 한 H200 작업에서 full epoch로 실행합니다.

```bash
bash phase1/phase1_cub/scripts/run_r50_224_guided_probe_full_b128_b64_seed1.sh
```

Issue 722의 ResNet-50/224 scratch Teacher를 다시 학습하지 않고 공유합니다. Batch
128을 먼저 완결한 뒤 batch 64를 실행하며, 각 batch에서 LG, ALG-w20, iBKD
λ=0.25, iBKD λ=0.5를 seed 1로 300 epoch 학습합니다. Validation macro top-1으로
고른 각 encoder를 strict load·완전 동결하고 probe seed `[1,2,3,4,5]`, LR
`[0.01,0.03,0.1]`, 100 epoch를 수행합니다. 각 batch의 20개 validation 선택이
완료된 뒤에만 official test mask를 열고 선택 probe를 각각 한 번 평가합니다.

기본 결과 경로는
`/app/output/phase1_cub_r50_224_b128_b64_guided_probe_seed1_full_v4`입니다. 분류
best checkpoint 8개, 선택 probe checkpoint 40개, CSV/JSON/status와 전체
`run.log`가 남습니다. Smoke 선형 외삽은 약 9시간 10분이어서 10시간 제한 여유가
크지 않습니다. 중간 산출물을 매 조건마다 기록하고 batch 128을 먼저 끝내지만,
환경·다운로드 편차로 제한을 넘을 가능성은 남아 있습니다.

이 결과의 batch 128 네 셀은 동일 설정으로 후속 v3 전체 매트릭스에 편입할 수
있습니다. Batch 64는 sensitivity 결과이며, 어느 쪽도 encoder seed가 하나뿐이므로
이 실행만으로 최종 방법 우위나 encoder-seed 표준편차를 주장하지 않습니다.

## Guided encoder-seed 후속 smoke

Batch 128의 guided 네 방법은 기존 seed 1에 seed 2·3을 추가해 주 실험의 독립
encoder seed 세 개를 완성합니다. Batch 64는 seed 2만 추가하며, seed-1 결과를 본
뒤 결정한 범위이므로 탐색적 sensitivity로만 보고합니다. 정확한 후속 조합을 먼저
다음 smoke로 확인합니다.

```bash
bash phase1/phase1_cub/scripts/run_r50_224_guided_probe_seed_extension_smoke.sh
```

Smoke는 issue 722 Teacher를 재사용하고 `(128,2)`, `(128,3)`, `(64,2)`마다 LG,
ALG-w20, iBKD λ=0.25/0.5를 2 epoch 실행합니다. 각 smoke encoder를 strict
load·freeze한 뒤 probe seed 1, LR `[0.01,0.03,0.1]`, 2 epoch를 수행합니다.
마지막 로그의 완료 기준은 분류 `12/12`, probe 후보 `36/36`, 선택 probe
`12/12`, GPU 작업 `48/48`입니다. Smoke 수치로 방법·lambda·batch·seed를
선택하지 않습니다.

이 smoke는 세 profile `3/3`, 전체 GPU 작업 `48/48`로 통과했습니다. 선형 외삽은
batch 128 seed 2·3을 분류부터 probe까지 약 9시간 4분으로 추정했습니다.

## Batch 128 guided encoder seed 2·3 본학습

```bash
bash phase1/phase1_cub/scripts/run_r50_224_guided_probe_full_b128_seeds2_3.sh
```

이 분할은 batch 128의 LG, ALG-w20, iBKD λ=0.25/0.5를 encoder seed 2와 3에서
각각 300 epoch 학습합니다. 각 validation-best encoder를 strict load·완전 동결한
뒤 probe seed 5개 × LR 3개 × 100 epoch를 실행하고, 각 probe seed의 LR/epoch를
validation으로 고른 뒤 official test를 한 번 평가합니다. 분류 checkpoint 8개와
선택 probe checkpoint 40개, 총 48개 새 checkpoint 및 CSV/JSON/전체 로그를
`/app/output/phase1_cub_r50_224_b128_guided_probe_seeds2_3_full_v5`에 보존합니다.

선형 예상 9시간 4분에 최초 데이터 다운로드·target cache 시간이 추가되므로 10시간
제한 여유는 작습니다. 이 실행의 마지막 완료 기준은 분류 `8/8`, probe 후보
`120/120`, 선택·test probe `40/40`, 새 checkpoint `48`입니다.

본실험에서는 각 encoder마다 probe seed `[1,2,3,4,5]`를 그대로 수행합니다.
Batch 128은 먼저 encoder별 5개 probe 평균을 계산한 뒤 encoder seed 1·2·3 간
평균과 표준편차를 보고합니다. Batch 64의 seed 1·2 결과에는 확증적 통계 검정을
적용하지 않습니다. Part localization, spatial CKA와 attention 진단은 이 실행에
섞지 않고 본실험 checkpoint archive를 회수한 뒤 별도 고정 프로토콜로 수행합니다.

## 이전 v2 smoke 및 보존 결과

다음 비과학적 smoke는 CUB 다운로드와 mask 대응, 고정 validation split, 분류
checkpoint 저장·strict load, encoder freeze, feature cache 및 probe 학습 경로를 한
번에 점검합니다.

```bash
bash phase1/phase1_cub/scripts/run_combined_smoke_b128.sh
```

공식 train 5,994장 중 클래스별 3장, 총 600장을 seed 2027 validation으로
분리하고 나머지 5,394장을 학습에 사용합니다. 공식 test 5,794장은 smoke에서
열거나 평가하지 않습니다. scratch ResNet-56 teacher 1개와 batch-128
DeiT-Tiny의 Vanilla, KD, LG, ALG-w20, iBKD λ=0.25/0.5를 seed 1에서 각각 2 epoch
실행합니다. 이어서 각 smoke encoder를 완전히 동결하고 LR
`[0.01, 0.03, 0.1]`의 probe를 각각 2 epoch 실행합니다.

Pet에서 이미 확인한 ALG의 epoch-2 조기 종료를 CUB에서 다시 주 비교 조건으로
사용하지 않도록, CUB의 유일한 ALG 설정은 controller 종료 판정 warm-up 20으로
사전 고정합니다. 이는 원본 ALG와 구분해 모든 로그와 표에서 `ALG-w20`으로 씁니다.

기본 결과 경로는 `/app/output/phase1_cub_b128_combined_smoke_v2`, 데이터와 큰
feature cache는 `/app/scratch`입니다. 마지막 로그에는 여섯 분류 validation
진단값, 여섯 probe validation 진단값, 선형 시간 외삽과 `25/25` 작업 완료 여부가
출력됩니다. 이 수치로 방법·lambda·checkpoint를 선택하거나 논문 결과를 주장할 수
없습니다. Smoke는 `25/25`로 통과했으며 이 진단값은 본실험 설정 선택에 사용하지
않았습니다.

### v2 guided shard 완료·보존

이 v2 guided 작업은 v3 교체 전에 제출되어 H200 issue 716에서 완료됐습니다.
다만 결과 회수 전에 v3를 최종 프로토콜로 확정했으므로 baseline shard는 실행하지
않고, v2 결과를 v3와 합치지 않습니다. 기록 재현을 위해 코드와 설정은 보존합니다.

```bash
bash phase1/phase1_cub/scripts/run_full_guided_b128.sh
```

- Guided producer: teacher + ALG-w20 + iBKD λ=0.25 + iBKD λ=0.5
- Baseline consumer: 동일 teacher + Vanilla + KD + LG

첫 작업은 ResNet-56 teacher를 seed 1로 한 번 학습하고 세 설정의 encoder seed
`[1,2,3]`을 각각 300 epoch 학습합니다. 이후 모든 encoder를 동결하고 probe seed
`[1,2,3,4,5]`, LR `[0.01,0.03,0.1]`, 100 epoch를 실행합니다. 한 shard의 모든
validation 선택이 끝난 뒤에만 official test mask를 열어 선택된 probe를 한 번씩
평가합니다.

결과 경로는 `/app/output/phase1_cub_b128_full_v2_guided`입니다. 결과에는 teacher 1개,
분류 best checkpoint 9개, 선택된 probe checkpoint 45개, raw CSV와 summary JSON이
포함되며 전체 실행 로그도 `run.log`로 저장됐습니다. 55개 checkpoint와 전체
계약은 독립 감사를 통과했습니다. 구버전 조건에서 probe mIoU는 ALG-w20
`77.882%`, iBKD λ=0.25 `77.546%`, iBKD λ=0.5 `77.538%`였습니다. 해석과 제한은
[v2 보존 보고서](reports/legacy_resnet56_v2_guided/RESULTS.md)에 기록했습니다.
55개 checkpoint와 로그는
[artifact Release manifest](reports/legacy_resnet56_v2_guided/artifact_release.json)로
식별합니다.

## 현재 주의사항

- 현재 실험의 이미지와 segmentation archive, split, binary mask mapping,
  ResNet-50/224 scratch teacher, student, 여섯 방법, seed, checkpoint와 test
  규칙은 [full v3 config](configs/cub200_r50_224_b128_full_v3.json)에 고정했습니다.
- v2 guided 결과는 완료됐지만 최종 v3 결과와 섞지 않으며, v2 baseline shard도
  더 이상 실행하지 않습니다.
- v3 학생은 issue 722의 ResNet-50 checkpoint 파일·model-state hash를 모두
  확인한 뒤 시작하고, 모든 guided 방법이 그 하나를 공유해야 합니다.
- 데이터셋, checkpoint, feature cache와 원시 H200 결과는 Git에 올리지 않습니다.
  검증된 작은 요약·manifest·정성 예시만 `reports/`에 반영합니다.
