# CUB main-L0 iBKD 제어 A/A 재현성 프로토콜

## 목적

동일한 main-L0 iBKD λ=0.25, batch 128, seed 1 실행에서 분류 정확도와 controller
종료 epoch가 크게 달라졌습니다. 레이어 연결 방식의 원인을 해석하기 전에, 같은
입력과 코드가 같은 학습 궤적을 만드는지 먼저 확인합니다.

이 실험은 새 성능을 주장하거나 방법을 선택하는 실험이 아닙니다. 기존
`main_l0_v3` 결과는 그대로 보존하며, 여기서 나온 수치는 사후 재현성 진단으로만
사용합니다.

## 엄격 결정론을 그대로 쓸 수 없는 이유

iBKD의 CBAM에는 `torchvision.ops.deform_conv2d`가 포함됩니다. H200의 CUDA
역전파에서 이 연산은 `compute_grad_input`의 결정론 구현을 제공하지 않습니다.
실제로 fail-closed 설정(`warn_only=False`)을 적용한 선행 smoke는 첫 backward에서
이 오류를 검출하고 중단했습니다.

표준 convolution으로 바꾸거나 gradient를 끊으면 원래 iBKD가 아니므로 그렇게
우회하지 않습니다. v2는 원래 과학 경로를 그대로 유지하고, 알려진 연산을
경고로 명시한 **제어 A/A**입니다. 따라서 “완전한 bitwise 결정론”을 주장하지
않고 같은 입력·RNG 아래 실제 CUDA 수치 변동을 직접 측정합니다.

## Smoke A/A

- 서로 독립된 새 Python process `A`, `B`를 순차 실행
- CUB split `5,394 / 600 / 5,794`, main-L0 loader
- issue 722 ResNet-50/224 scratch teacher
- DeiT-Tiny/16, iBKD λ=0.25, learned-all aggregation
- batch 128, seed 1, fp32, full train 5,394장을 2 epoch
- official test 접근 0회

두 process 모두 다음 제어 설정을 강제합니다.

- `torch.use_deterministic_algorithms(True, warn_only=True)`
- cuDNN benchmark off, deterministic on
- CUDA matmul·cuDNN TF32 off, float32 matmul precision `highest`
- `PYTHONHASHSEED=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, DataLoader `num_workers=0`

## PASS gate

두 실행에서 다음 **실행 제어값**이 모두 정확히 같아야 합니다.

- 초기 student·guidance state hash와 teacher 두 hash
- validation split manifest
- epoch별 실제 augmentation 적용 후 입력 tensor stream hash
- epoch별 전체 RNG state hash
- runtime·환경 계약과 알려진 비결정론 연산 고지

두 실행의 loss·accuracy 차이, epoch별·최종 model-state hash 일치 여부, controller,
aggregation 및 checkpoint hash는 모두 기록하되 2-epoch 실행 gate로 사용하지
않습니다. 이 값들은 CUDA deformable-conv가 만드는 수치 변동의 관측 대상입니다.
임의 허용오차로 PASS를 만들지도 않습니다.

두 실행 중 하나가 실패하거나 실행 제어값이 다르면 smoke는 실패합니다. Smoke가
통과했다는 것은 A/A 본실험 경로가 정상이고 입력·RNG가 통제됐다는 뜻일 뿐입니다.
장기 학습 안정성은 동일 설정의 300-epoch A/A 두 실행에서 최종 성능과 controller
종료 시점의 차이로 판단합니다.

정확한 machine-readable 계약은
[`configs/cub200_r50_224_b128_main_l0_ibkd_controlled_aa_smoke_v2.json`](configs/cub200_r50_224_b128_main_l0_ibkd_controlled_aa_smoke_v2.json)에
고정합니다.
