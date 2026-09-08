# Phase 1 — CUB-200-2011 공간 표현 검증

상태: **ResNet-50/224 scratch 본실험 v3 LOCK — guided smoke 통과, Teacher 본학습 준비 완료**

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
프로토콜은 별도로 고정합니다. ResNet-56/32 v2는 결과가 나오기 전에 중단했고,
현재 본실험은 더 보편적인 scratch ResNet-50/224 teacher를 쓰는 v3입니다. Pet
결과나 checkpoint는 CUB 본실험에 섞지 않습니다.

## 디렉터리

- `configs/`: 결과를 보기 전에 잠근 CUB 전용 machine-readable 설정
- `scripts/`: 데이터 감사, smoke, 본실험 및 결과 정리 진입점
- `reports/classification/`: 200종 분류 결과와 checkpoint manifest
- `reports/frozen_probe/`: frozen segmentation probe 정량·정성 결과
- `results/raw/`: 로컬 원시 산출물 전용 경로이며 Git에는 포함하지 않음

잠긴 계약과 실행 gate는 [PROTOCOL.md](PROTOCOL.md)에 기록합니다.

## 현재 v3: ResNet-50/224 Teacher 본학습

Guided smoke는 `17/17` 작업을 완료해 실행 경로와 메모리·시간 gate를 통과했습니다.
본실험의 첫 단계는 모든 guided student가 공유할 단 하나의 Teacher checkpoint를
만드는 작업입니다.

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
`/app/scratch`에만 두고 결과 폴더에 복사하지 않습니다.

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
통과했으며, Teacher만 분리한 현재 작업은 smoke peak reserved 15.731 GB를
기준으로 H200 1g.18gb MIG 1개에서 실행합니다.

## 이전 v2 smoke 및 미실행 본실험

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

### v2 단일 teacher를 공유하는 두 단계 본실험

이 v2 작업은 실행하지 않았으며 v3로 대체했습니다. 기록 재현을 위해 코드와
설정은 보존합니다.

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

첫 결과 경로는 `/app/output/phase1_cub_b128_full_v2_guided`입니다. 결과에는 teacher 1개,
분류 best checkpoint 9개, 선택된 probe checkpoint 45개, raw CSV와 summary JSON이
포함되며 전체 실행 로그도 `run.log`로 저장됩니다. 이미지 본체와 segmentation
mask archive, feature cache는 `/app/scratch`에만 두므로 결과 ZIP에 포함되지
않습니다. 결과를 받은 뒤 teacher의 파일·model-state hash를 고정하고 이를 받는
두 번째 baseline 실행기를 추가합니다.

## 현재 주의사항

- 현재 실험의 이미지와 segmentation archive, split, binary mask mapping,
  ResNet-50/224 scratch teacher, student, 여섯 방법, seed, checkpoint와 test
  규칙은 [full v3 config](configs/cub200_r50_224_b128_full_v3.json)에 고정했습니다.
- v2 설정은 미실행 기록이며 v3 결과와 섞지 않습니다.
- Baseline shard는 guided shard의 teacher checkpoint가 hash로 고정되기 전에는
  실행할 수 없습니다. 두 결과의 seed별 초기 student state와 split/config hash도
  일치하는지 확인한 뒤에만 하나의 6설정 결과로 병합합니다.
- 데이터셋, checkpoint, feature cache와 원시 H200 결과는 Git에 올리지 않습니다.
  검증된 작은 요약·manifest·정성 예시만 `reports/`에 반영합니다.
