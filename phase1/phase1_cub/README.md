# Phase 1 — CUB-200-2011

상태: **ResNet-50/224 scratch Teacher, guided 4방법 × batch-128 encoder seed 3개,
frozen probe와 직접 공간정보 진단 완료·감사 통과.** Vanilla/KD는 아직 이 v3
매트릭스에 포함되지 않았습니다.

공식 이미지 분할은 train/validation/test `5,394 / 600 / 5,794`입니다. 분류
checkpoint와 학습형 probe는 validation으로만 선택하고, official test는 선택이
끝난 뒤 정해진 횟수만 평가했습니다.

## 본학습 결과 색인

| 실험 | 상태와 핵심 결과 | 보고서 |
|---|---|---|
| Scratch ResNet-50/224 Teacher | test macro Top-1 41.178% | [Teacher](reports/classification/resnet50_224_teacher_v3/RESULTS.md) |
| Guided batch 128/64, seed 1 | 분류 iBKD-0.25 1위, probe LG 1위 | [Issue 727](reports/frozen_probe/resnet50_224_b128_b64_guided_seed1_v4/RESULTS.md) |
| Guided batch 128, 3 seeds | 분류 iBKD-0.25 24.429%, probe LG 73.470%로 각각 1위 | [Issue 727+730](reports/frozen_probe/resnet50_224_b128_guided_3seed_v5/RESULTS.md) |
| 직접 공간정보 진단, main-L0 3 seeds | 기존 issues 727/730 checkpoint 평가; Part PCK·CKA는 LG 1위 | [Issue 737+739](reports/direct_spatial/resnet50_224_b128_guided_3seed_v2/RESULTS.md) |
| 이미지 loader pilot-L0/L1/L2, seed 1 | 세 loader에서 encoder를 별도 재학습; Part PCK로 L2 선택 | [Loader 결과](image_loader_experiment/reports/full_v1_log_snapshot/RESULTS.md) |
| L2 guided 예비 본실험, seed 1 | 분류 LG 1위, probe ALG-w20 1위 | [L2 결과](image_loader_experiment/reports/l2_guided_preliminary_full_seed1_log_snapshot_v1/RESULTS.md) |
| main-L0 레이어 연결 사후 분석 | 공통 guidance 123 epoch·4연결 방식 classification-only 본실험 진행 중 | [Mechanism](mechanism_analysis/README.md) |
| main-L0 제어 A/A | 완료; 입력·RNG 통제 확인, test 24.761/25.510%로 동일 범위 재현 | [Reproducibility](reproducibility/README.md) |

현재 CUB 결과는 iBKD가 LG/ALG보다 공간정보를 전반적으로 더 잘 보존한다는 가설을
지지하지 않습니다. Guided 3-seed frozen probe와 주 직접지표 Part PCK 모두 LG가
가장 높았습니다. Attention 지표는 iBKD-0.25가 높았지만 주 지표 및 CKA와 방향이
달라 보조 관측으로만 해석합니다.

실험 계보와 `main-L0`, `pilot-L0`, `L2 preliminary`의 구분은
[EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)를 기준으로 합니다.

`main-L0 직접 공간정보 진단`과 loader 실험의 `pilot-L0`는 같은 L0 transform과
같은 네 guided 방법을 사용하지만 같은 실행은 아닙니다. 전자는 이미 완료된
main-L0 3-seed checkpoint를 재사용해 official test까지 진단했고, 후자는 L1/L2와
paired 비교를 위해 seed 1 encoder를 새로 학습해 validation에서만 평가했습니다.
따라서 pilot-L0는 loader 비교의 기준군이며 main-L0 결과와 합치거나 중복 결과로
취급하지 않습니다.

## 본학습 재현 진입점

```bash
# 공용 scratch ResNet-50/224 Teacher
bash phase1/phase1_cub/scripts/run_r50_224_teacher_full.sh

# Guided 분류 → frozen segmentation probe
bash phase1/phase1_cub/scripts/run_r50_224_guided_probe_full_b128_b64_seed1.sh
bash phase1/phase1_cub/scripts/run_r50_224_guided_probe_full_b128_seeds2_3.sh

# 직접 공간정보 진단
bash phase1/phase1_cub/scripts/run_r50_224_direct_spatial_full_b128_seed1.sh
bash phase1/phase1_cub/scripts/run_r50_224_direct_spatial_full_b128_seeds2_3.sh
```

과학 설정은 [PROTOCOL.md](PROTOCOL.md), 직접 공간정보 지표 정의는
[DIRECT_SPATIAL_PROTOCOL.md](DIRECT_SPATIAL_PROTOCOL.md), 이미지 loader 탐색은
[image_loader_experiment](image_loader_experiment/README.md)에 분리했습니다.
현재 같은-seed 변동을 확인하는 결정론 gate는
[reproducibility](reproducibility/README.md)에 분리했습니다.

## 보존 원칙

- Git에는 본학습 결과, 작은 표·그림, protocol config와 hash 감사 manifest만 둡니다.
- 큰 teacher/student/probe checkpoint와 원시 로그는 각 결과 폴더의 검증된
  `checkpoint_release.json` 또는 `artifact_release.json`으로 보존합니다.
- CUB 이미지·segmentation archive와 feature cache는 Git에 넣지 않습니다.
- 본학습이 끝난 선행 실행 점검 config·script·중간 보고서는 제거했습니다.
- 아직 본학습 전인 [mechanism_analysis](mechanism_analysis/README.md)와 Phase 4의
  실행 점검 자료는 이 정리 대상이 아닙니다.
