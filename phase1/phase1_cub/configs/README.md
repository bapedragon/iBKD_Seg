# CUB 본실험 설정

| 파일 | 역할 |
|---|---|
| `cub200_r50_224_b128_full_v3.json` | ResNet-50/224 scratch Teacher와 6방법×3seed의 기준 과학 프로토콜 |
| `cub200_r50_224_b128_b64_guided_seed1_full_v4.json` | Guided 4방법, batch 128/64, encoder seed 1 본실험 |
| `cub200_r50_224_b128_s23_b64_s2_guided_full_v5.json` | Batch 128 seed 2·3 및 batch 64 seed 2 후속 본실험 |
| `cub200_r50_224_b128_seed1_direct_spatial_full_v2.json` | 직접 공간정보 진단 seed 1 본실험 |
| `cub200_r50_224_b128_seed2_3_direct_spatial_full_v2.json` | 직접 공간정보 진단 seed 2·3 본실험 |
| `cub200_r50_224_b128_seed1_direct_spatial_metric_contract_v1.json` | Seed 1 metric 정의와 checkpoint inventory의 불변 계약 |
| `cub200_r50_224_b128_seed2_3_direct_spatial_metric_contract_v2.json` | Seed 2·3 metric 정의와 checkpoint inventory의 불변 계약 |

이미지 loader 관련 본실험 설정은
[`../image_loader_experiment/configs/`](../image_loader_experiment/configs/)에만 둡니다.
완료된 본실험의 선행 실행 점검 설정은 제거했으며, 새 실험을 시작할 때만 별도
점검 설정을 만들고 본학습 완료 후 다시 정리합니다.
