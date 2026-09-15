# CUB main-L0 iBKD 결정론 A/A 재현성 프로토콜

## 목적

동일한 main-L0 iBKD λ=0.25, batch 128, seed 1 실행에서 분류 정확도와 controller
종료 epoch가 크게 달라졌습니다. 레이어 연결 방식의 원인을 해석하기 전에, 같은
입력과 코드가 같은 학습 궤적을 만드는지 먼저 확인합니다.

이 실험은 새 성능을 주장하거나 방법을 선택하는 실험이 아닙니다. 기존
`main_l0_v3` 결과는 그대로 보존하며, 여기서 나온 수치는 사후 재현성 진단으로만
사용합니다.

## Smoke A/A

- 서로 독립된 새 Python process `A`, `B`를 순차 실행
- CUB split `5,394 / 600 / 5,794`, main-L0 loader
- issue 722 ResNet-50/224 scratch teacher
- DeiT-Tiny/16, iBKD λ=0.25, learned-all aggregation
- batch 128, seed 1, fp32, full train 5,394장을 2 epoch
- official test 접근 0회

두 process 모두 다음 결정론 설정을 강제합니다.

- `torch.use_deterministic_algorithms(True, warn_only=False)`
- cuDNN benchmark off, deterministic on
- CUDA matmul·cuDNN TF32 off, float32 matmul precision `highest`
- `PYTHONHASHSEED=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, DataLoader `num_workers=0`

## PASS gate

두 실행에서 다음 값이 모두 정확히 같아야 합니다.

- 초기 student·guidance state hash와 teacher 두 hash
- validation split manifest
- epoch별 실제 augmentation 적용 후 입력 tensor stream hash
- wall time·memory를 제외한 loss, accuracy, validation trace
- epoch별 student·guidance state hash와 전체 RNG state hash
- 최종 student·guidance state, controller, aggregation state

Checkpoint 파일 자체의 hash는 기록하지만 직렬화 container byte가 달라질 수 있어
PASS gate로 쓰지 않습니다. 실제 model-state SHA-256을 gate로 사용합니다.

하나라도 다르거나 PyTorch가 비결정론 연산을 발견하면 smoke는 실패합니다. Smoke가
통과해도 300 epoch 전체 재현성을 증명한 것은 아니므로, 동일 설정의 full A/A 두
실행을 별도 protocol로 수행해야 합니다.

정확한 machine-readable 계약은
[`configs/cub200_r50_224_b128_main_l0_ibkd_deterministic_aa_smoke_v1.json`](configs/cub200_r50_224_b128_main_l0_ibkd_deterministic_aa_smoke_v1.json)에
고정합니다.
