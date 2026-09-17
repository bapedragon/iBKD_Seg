# CUB 실행·정리 스크립트

## 본학습 실행

- `run_r50_224_teacher_full.sh`: 공용 scratch ResNet-50/224 Teacher
- `run_r50_224_guided_probe_full_b128_b64_seed1.sh`: guided 4방법의 batch
  128/64 encoder seed 1 분류와 frozen probe
- `run_r50_224_guided_probe_full_b128_seeds2_3.sh`: batch 128 encoder seed 2·3
  분류와 frozen probe
- `run_r50_224_direct_spatial_full_b128_seed1.sh`: seed 1 직접 공간정보 진단
- `run_r50_224_direct_spatial_full_b128_seeds2_3.sh`: seed 2·3 직접 공간정보 진단

이미지 loader 관련 진입점은
[`../image_loader_experiment/scripts/`](../image_loader_experiment/scripts/)에 있습니다.

## 결과 반입·정리

- `import_h200_archive.py`: H200 결과 archive의 안전한 반입과 manifest 생성
- `curate_r50_teacher_result.py`: Teacher 결과 감사·정리
- `curate_r50_v4_guided_seed1_result.py`: issue 727 결과 감사·정리
- `curate_r50_v5_guided_seeds2_3_result.py`: issue 730 결과 감사·정리
- `curate_r50_direct_spatial_seed1_v2_result.py`: issue 737 결과 감사·정리
- `curate_r50_direct_spatial_seed2_3_v2_result.py`: issue 739 결과 감사·정리
- `build_direct_spatial_attention_comparison.py`: 사전 고정된 seed-1 이미지 8개의
  원본·GT·방법별 attention rollout을 재배치한 논문용 정성 비교 패널 생성

완료된 본실험의 선행 실행 점검용 shell은 보존하지 않습니다.
