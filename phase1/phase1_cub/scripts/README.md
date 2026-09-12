# CUB-200-2011 실행 스크립트

- `run_r50_224_loader_damage_audit.sh`: loader pilot Stage A. 학습이나 official-test
  접근 없이 derived train 이미지의 random crop이 part, mask, bbox를 얼마나
  보존하는지 L0/L1/L2에서 감사하고 JSON·CSV·고정 정성 그림을 저장합니다.
- `run_r50_224_loader_pilot_smoke_b128_seed1.sh`: loader pilot Stage B의
  validation-only v2 경로 점검입니다. Issue 746 v1은 분류 12개 뒤 첫 probe 생성
  전에 설정 schema 누락으로 중단됐고, 현재 실행기는 의미를 바꾸지 않은 v2를
  사용합니다. Issue 722 teacher를 재사용해 L0/L1/L2 ×
  LG·ALG-w20·iBKD 두 lambda의 분류 12개를 2 epoch 학습하고, 각 frozen encoder에
  segmentation probe, Part PCK, CKA, attention–GT를 적용합니다. Official test는
  열지 않으며 마지막 로그에 12개 결과와 full pilot 보수적 시간 외삽을 나열합니다.
- `run_r50_224_loader_pilot_smoke_l1_l2_b128_seed1.sh`: L0 본 pilot의 실제 실행
  시간이 확인된 뒤 남은 L1·L2를 한 H200 작업으로 묶을 수 있는지 점검하는 운영용
  subset smoke입니다. 잠긴 v2 smoke의 방법·seed·LR·probe를 그대로 사용하되 L1과
  L2의 8개 학생만 실행합니다. 완료 gate는 분류 `8`, segmentation/part 후보 각
  `24`, 선택 각 `8`, CKA `96`, attention `8`, 정성 PNG `32`, official test `0`이며
  smoke 성능값은 loader 선택에 사용하지 않습니다.
- `run_r50_224_loader_pilot_full_b128_seed1.sh`: 통과한 smoke v2 뒤 잠근 Stage B
  본 pilot 실행기입니다. 인자로 L0/L1/L2 중 정확히 하나를 받아 4개 guided
  학생의 300-epoch 분류, frozen segmentation·part probe 각 5 seeds × 3 LR ×
  100 epoch, validation CKA·attention을 수행합니다. 세 profile 전체의 보수적
  예상이 14시간 54분이므로 profile별 약 4시간 58분의 세 이슈로 나누며 official
  test는 열지 않습니다. 마지막 로그에 4방법 결과와 해당 profile의 사전 고정된
  loader 선택 점수를 모두 나열합니다.
- `run_r50_224_loader_pilot_full_l1_l2_b128_seed1.sh`: L0 실제시간과 issue 750의
  L1·L2 subset smoke를 이용해 10시간 제한 안에서 두 남은 shard를 한 이슈로
  순차 실행하는 운영용 wrapper입니다. 과학적 full config는 변경하지 않고 L1과
  L2의 출력·cache·checkpoint를 완전히 분리합니다. 마지막에 두 shard의 hash와
  completion gate를 다시 감사하고 8개 방법 결과, 두 loader 선택 점수 및 합산
  실행시간을 한 번에 출력합니다. Official test는 열지 않습니다.

데이터 감사, smoke, 분류 본실험, frozen probe와 결과 정리 스크립트를 이 폴더에
둡니다.

- `run_r50_224_guided_smoke_b128.sh`: 현재 v3용. Scratch ResNet-50/224 teacher와
  LG, ALG-w20, iBKD-0.25, iBKD-0.5를 각각 2 epoch 실행하고 네 frozen
  encoder × 세 LR probe를 2 epoch 실행합니다. Classification과 선택된 probe의
  official test 경로, peak memory, 4방법×3seed guided block 시간 외삽까지 기록합니다.
- `run_r50_224_teacher_full.sh`: 현재 v3 Teacher 전용 본학습. TorchVision
  ResNet-50을 `weights=None`인 scratch 상태에서 224×224, batch 128, seed 1,
  200 epoch 학습하고 validation macro top-1 best checkpoint를 고른 뒤 strict
  reload와 official test 1회 평가까지 수행합니다. 기존 ResNet-56/32 본학습
  진입점은 호출하지 않습니다.
- `run_r50_224_guided_probe_smoke_b128_b64.sh`: issue 722에서 완료한 동일
  Teacher checkpoint를 release manifest의 byte/model-state SHA-256으로 검증해
  재사용합니다. LG, ALG-w20, iBKD-0.25, iBKD-0.5의 2-epoch 분류와 공통 frozen
  probe smoke를 batch 128, 64 순서로 수행하고, 마지막에 두 배치의 시간·peak
  memory·진단값과 선형 본실험 시간 외삽을 한 번에 출력합니다. Teacher를 다시
  학습하지 않으며 모든 smoke metric은 비과학적입니다.
- `run_r50_224_guided_probe_full_b128_b64_seed1.sh`: 통과한 batch profile smoke의
  full-epoch 실행기입니다. Issue 722 Teacher를 내려받아 strict 검증한 뒤 batch
  128을 먼저 완결하고 batch 64를 이어서 수행합니다. 각 batch에서 guided 4방법 ×
  encoder seed 1 분류 300 epoch, frozen probe 5 seed × 3 LR × 100 epoch와
  validation 선택·official test 1회 평가를 실행합니다. 각 분류/probe checkpoint를
  즉시 `/app/output`에 저장하고 마지막에 두 batch 결과를 모두 나열합니다.
- `run_r50_224_guided_probe_seed_extension_smoke.sh`: 동일한 issue 722 Teacher를
  재사용해 `(batch 128, seed 2)`, `(batch 128, seed 3)`, `(batch 64, seed 2)`를
  순서대로 점검합니다. 각 profile은 guided 네 방법의 2-epoch 분류와 frozen probe
  seed 1 × LR 3개 × 2 epoch를 수행합니다. 마지막에 12개 분류·12개 선택 probe,
  peak memory와 `batch128 seeds 2·3`/`batch64 seed 2` 본실험 시간 외삽을 모두
  출력합니다. 모든 smoke 수치는 비과학적입니다.
- `run_r50_224_guided_probe_full_b128_seeds2_3.sh`: 통과한 v5 smoke 뒤 batch 128의
  encoder seed 2·3만 실행하는 첫 번째 full partition입니다. Guided 네 방법을
  각각 300 epoch 학습하고 encoder마다 probe seed 5개 × LR 3개 × 100 epoch를
  validation 선택 및 official-test 1회까지 수행합니다. 새 분류 checkpoint 8개와
  선택 probe checkpoint 40개를 `/app/output`에 보존하며 마지막 로그에 두 encoder
  seed의 전체 분류 및 probe 평균을 나열합니다.
- `run_r50_224_direct_spatial_smoke_b128_seed1.sh`: issue 722 Teacher와 issue 727
  batch-128 seed-1 네 encoder를 검증된 Release에서 받아 visible-part heatmap
  probe, 12-block spatial linear CKA, attention rollout–GT mask 지표와 정성 PNG를
  생성합니다. 클래스당 한 장의 train/validation subset만 사용하고 official test는
  열지 않습니다. H200 MIG slice 1개용 비과학적 실행 점검입니다.
- `run_r50_224_direct_spatial_smoke_b128_seeds2_3.sh`: issue 730 Release의
  batch-128 seed 2·3 `LG`, `ALG-w20`, `iBKD-0.25`, `iBKD-0.5` encoder 8개를
  모두 strict-load하고 seed-1과 같은 v2 좌표·part·CKA·attention 경로를 축소
  실행합니다. Part 후보 24개, CKA 96개, attention row 8개와 정성 PNG 32개를
  만들며 official test는 열지 않습니다. H200 MIG slice 1개용 비과학적 smoke입니다.
- `run_r50_224_direct_spatial_full_b128_seeds2_3.sh`: issue 738 smoke 통과 뒤
  잠근 seed 2·3 본실험 실행기입니다. Issue 730 encoder 8개에 seed-1과 동일한
  part probe 5 seed × LR 3개 × 100 epoch, validation CKA와 official-test
  attention을 적용합니다. Validation 선택 40개를 모두 완료한 뒤 test를 열고,
  선택 probe checkpoint 40개와 정성 PNG 64개를 `/app/output`에 보존합니다.
- `run_r50_224_direct_spatial_full_b128_seed1.sh`: 직접 공간정보 진단 v2 seed-1
  본실험 실행기입니다. 두 Release를 같은 hash로 검증하고 4 encoder × part-probe
  seed 5개 × LR 3개 × 100 epoch, validation CKA와 official-test attention을
  수행합니다. 원본에서 visible이지만 이미지 밖인 part는 좌표를 clipping하지 않고
  part loss·선택·PCK에서 공통 제외하며 split별 감사 내역을 출력합니다. v1은 어떤
  metric도 계산하기 전에 image `5007`에서 중단됐으므로 결과로 사용하지 않습니다.

아래 두 스크립트는 대체된 ResNet-56/32 v2 기록용입니다.

- `run_combined_smoke_b128.sh`: ResNet-56 teacher와 Vanilla, KD, LG, ALG-w20,
  iBKD-0.25, iBKD-0.5의 여섯 batch-128 분류 설정을
  각각 2 epoch 실행한 뒤, 여섯 frozen encoder × 세 LR probe를 각각 2 epoch
  실행하는 통합 smoke
- `run_full_guided_b128.sh`: teacher를 한 번 학습하고 ALG-w20, iBKD-0.25,
  iBKD-0.5의 분류 3 seed와 frozen probe 전체를 실행하는 약 8시간 작업

`run_combined_smoke_b128.sh`는 과거 smoke만 완료됐고,
`run_full_guided_b128.sh`는 H200 issue 716에서 완료됐습니다. v2 baseline shard는
실행하지 않으며, v2 결과를 v3에 합치지 않습니다.

## 결과 archive 정리

- `import_h200_archive.py`: ZIP을 untrusted data로 다룹니다. 경로 탈출, symlink,
  암호화·중복 member를 거부하고 전체 CRC와 완료 계약을 확인한 뒤 재현에 필요한
  파일만 ignored raw 경로로 가져옵니다.
- `curate_r50_teacher_result.py`: issue 722의 ResNet-50/224 Teacher checkpoint를
  안전 로드·strict load하고 history·split·protocol·test-once 계약과 hash를
  재검산해 추적 가능한 소형 보고서를 생성합니다.
- `curate_resnet56_v2_guided_result.py`: issue 716의 Teacher, guided encoder와
  선택된 probe checkpoint 55개를 모두 감사하고, encoder seed 기준 분류·probe
  집계와 v2 보존 보고서를 생성합니다.
- `curate_r50_v4_guided_seed1_result.py`: issue 727의 batch-128/64 guided seed-1
  encoder 8개와 선택 probe 40개를 `weights_only=True`로 strict load하고 파일·state
  hash, 유한값, split 및 selection-before-test 계약을 재검산해 v4 부분 보고서를
  생성합니다.
- `curate_r50_v5_guided_seeds2_3_result.py`: issue 730의 batch-128 seed 2·3
  encoder 8개와 선택 probe 40개를 감사하고, issue 727의 감사된 seed 1과 결합해
  guided 네 방법의 3-encoder-seed 집계를 생성합니다.
- `curate_r50_direct_spatial_seed1_v2_result.py`: issue 737의 part-probe checkpoint
  20개를 strict load하고 PCK, CKA, attention–GT, annotation 유효성 및 32개 정성
  이미지 계약을 감사해 seed-1 직접진단 보고서를 생성합니다.
- `curate_r50_direct_spatial_seed2_3_v2_result.py`: issue 739의 part-probe
  checkpoint 40개와 재사용 encoder 8개를 감사하고, issue 737 seed 1과 합쳐
  PCK·CKA·attention의 독립 encoder seed 3개 평균과 sample SD를 생성합니다.

```bash
PYTHONPATH=src python phase1/phase1_cub/scripts/import_h200_archive.py \
  ARCHIVE.zip --kind resnet50-v3-teacher --issue-id 722 \
  --output-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_teacher_v3_issue722 \
  --canonical-bundle-filename phase1_cub_resnet50_224_teacher_v3_issue722.zip

PYTHONPATH=src python phase1/phase1_cub/scripts/curate_r50_teacher_result.py \
  --raw-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_teacher_v3_issue722 \
  --report-dir phase1/phase1_cub/reports/classification/resnet50_224_teacher_v3

PYTHONPATH=src python phase1/phase1_cub/scripts/import_h200_archive.py \
  ARCHIVE.zip --kind resnet50-v4-guided-seed1 --issue-id 727 \
  --output-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_b64_guided_seed1_v4_issue727 \
  --canonical-bundle-filename phase1_cub_resnet50_224_b128_b64_guided_seed1_v4_issue727.zip

PYTHONPATH=src python phase1/phase1_cub/scripts/curate_r50_v4_guided_seed1_result.py \
  --raw-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_b64_guided_seed1_v4_issue727 \
  --report-dir phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_b64_guided_seed1_v4

PYTHONPATH=src python phase1/phase1_cub/scripts/import_h200_archive.py \
  ARCHIVE.zip --kind resnet50-v5-guided-seeds2-3 --issue-id 730 \
  --output-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_guided_seeds2_3_v5_issue730 \
  --canonical-bundle-filename phase1_cub_r50_224_issue730_guided_s23_and_issue737_direct_spatial_v2.zip

PYTHONPATH=src python phase1/phase1_cub/scripts/curate_r50_v5_guided_seeds2_3_result.py \
  --raw-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_guided_seeds2_3_v5_issue730 \
  --seed1-report-dir phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_b64_guided_seed1_v4 \
  --report-dir phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_guided_3seed_v5

PYTHONPATH=src python phase1/phase1_cub/scripts/import_h200_archive.py \
  ARCHIVE.zip --kind resnet50-direct-spatial-v2-seed1 --issue-id 737 \
  --output-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_seed1_direct_spatial_v2_issue737 \
  --canonical-bundle-filename phase1_cub_r50_224_issue730_guided_s23_and_issue737_direct_spatial_v2.zip

PYTHONPATH=src python phase1/phase1_cub/scripts/curate_r50_direct_spatial_seed1_v2_result.py \
  --raw-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_seed1_direct_spatial_v2_issue737 \
  --seed1-report-dir phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_b64_guided_seed1_v4 \
  --report-dir phase1/phase1_cub/reports/direct_spatial/resnet50_224_b128_seed1_v2

PYTHONPATH=src python phase1/phase1_cub/scripts/import_h200_archive.py \
  ARCHIVE.zip --kind resnet50-direct-spatial-v2-seeds2-3 --issue-id 739 \
  --output-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_seed2_3_direct_spatial_v2_issue739 \
  --canonical-bundle-filename phase1_cub_resnet50_224_b128_seed2_3_direct_spatial_v2_issue739.zip

PYTHONPATH=src python phase1/phase1_cub/scripts/curate_r50_direct_spatial_seed2_3_v2_result.py \
  --raw-dir phase1/phase1_cub/results/raw/cub200/resnet50_224_b128_seed2_3_direct_spatial_v2_issue739 \
  --seed1-report-dir phase1/phase1_cub/reports/direct_spatial/resnet50_224_b128_seed1_v2 \
  --guided-report-dir phase1/phase1_cub/reports/frozen_probe/resnet50_224_b128_guided_3seed_v5 \
  --report-dir phase1/phase1_cub/reports/direct_spatial/resnet50_224_b128_guided_3seed_v2
```

v3 Teacher는 완료됐습니다. 이후 학생·probe 본실험 실행기는 checkpoint 파일
SHA-256 `ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3`와
model-state SHA-256
`96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7`을 함께
강제해야 합니다.
