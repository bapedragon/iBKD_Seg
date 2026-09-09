# CUB-200-2011 실행 스크립트

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
```

v3 Teacher는 완료됐습니다. 이후 학생·probe 본실험 실행기는 checkpoint 파일
SHA-256 `ca6860f55f440dbe0018e7cc6d4f70dd257ba48e3cf692553408abdde7f1f3a3`와
model-state SHA-256
`96fea19b4556e1f6736d84e5f6ac139ec508ea04fa1fdd2d643498c382cdfba7`을 함께
강제해야 합니다.
