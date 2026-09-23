# Cityscapes SegFormer-B0 smoke v2 · H200 결과

2026-09-23 사용자 제공 로그를 검사했습니다. 실행 commit은
`56105a6ae686b19a8ec3bce5d89a5f83d0eda4e9`입니다.
**학습 7/7 · checkpoint 재개 7/7 · val2 평가 7/7 통과**입니다.
주 비교 6개와 별도 C2VKD(CLIP-pool) 모두 통과했으며, 파이프라인 총 소요는
564.32초(약 9분 24초)입니다. 설치·자료 준비·진단을 포함한 시간입니다.

## 방법별 확인

실제 batch16·crop512·FP32·TF32 off, 3 update 조건입니다.
환경은 NVIDIA H200 NVL, PyTorch 2.11.0+cu130, CUDA13.0, Python3.10.12입니다.

| 방법 | 학습·복원·재실행 | 최종 총 loss | 학습 구간 peak allocated (GiB) | 3번째 update 시간(초) |
|---|---|---:|---:|---:|
| vanilla | 통과 | 5.821401 | 7.89 | 0.335 |
| lg | 통과 | 6.270872 | 15.65 | 1.006 |
| alg | 통과 | 6.270872 | 15.65 | 1.001 |
| ibkd_lambda025 | 통과 | 6.254478 | 16.95 | 2.562 |
| ibkd_lambda050 | 통과 | 6.252602 | 16.95 | 2.564 |
| fskd | 통과 | 11.672958 | 73.83 | 0.869 |
| c2vkd_clip_pool | 통과 | 3.188081 | 15.71 | 0.892 |

GPU 메모리는 runner의 `train_peak_cuda_bytes`입니다. 개별 loss gradient 진단을 포함한
학습 구간의 PyTorch allocated peak이며 calibration·평가 전체 peak나 프로세스 전체
GPU 사용량은 아닙니다. FSKD가 73.83 GiB로 가장 높았고 이번 H200에서 OOM 없이 완료했습니다.
iBKD 시간에는 명세대로 작은 spatial deformable convolution의 CPU 계산이 포함됩니다.
3번째 update 하나의 시간을 80k 전체 소요 시간으로 확정하지 않습니다.

## 재개와 공통 조건 검증

- checkpoint 복원 직후 student·guide·optimizer·controller·step·RNG가 저장값과 **bitwise 일치**했습니다.
- 재실행한 3번째 update의 모델·guide·optimizer·controller·loss가 기존 기준
  `rtol=2e-5, atol=2e-6`을 통과했습니다. 모든 모델 tensor의 bitwise 일치까지 증명한 검사는 아닙니다.
- 로그에 기록된 3번째 update의 loss 항과 gradient norm은 원래 실행/재실행에서 숫자가 정확히 같습니다.
- student 초기 state·입력 hash가 7개 방법에서 같습니다. KD teacher state hash도 같습니다.
- LG와 ALG의 임시 beta·3개 update의 loss·gradient 및 val2 metric이 같습니다.
  실제 3 update 중 controller 관측은 0회이므로 기대한 결과입니다.
- 모든 개별 loss의 encoder gradient가 유한하며 0보다 큽니다.
  KD teacher·BN과 C2VKD pool의 고정 상태 검사가 통과했습니다.
- controller 186/372 관측 및 373 off 경계는 별도 합성 loss 검사에서 통과했습니다.
  실제 373 update까지 학습한 결과로 해석하지 않습니다.
- 제공된 source/config/asset SHA-256·bytes가 작업공간과 고정 명세에 일치합니다.
- 완전한 행이 남은 6개 방법은 confusion matrix로 19-class mIoU와 pixel accuracy를
  재계산했으며 보고값과 일치했습니다. 같은 2장의 유효 픽셀 수는 3,719,892입니다.

## 로그 범위와 다음 단계

붙여넣은 텍스트의 앞부분이 잘려 Vanilla 행의 첫 부분과 metric 수치는 없습니다.
남은 Vanilla 행에서 복원·재실행 통과를 확인했고, pipeline 마지막 집계가
`passed_methods=7`, `evaluated_methods=7`, `resume_passed_methods=7`임을 확인했습니다.
잘린 수치는 복원하거나 추정해서 채우지 않았습니다.
원시 로그 대신 해시와 추출 요약을 [log_audit.json](log_audit.json)에 기록했습니다.

**이번 판정은 실행 연결·짧은 재개·평가 smoke 통과입니다.** val2 점수나 서로 다른
목적함수의 loss로 방법 순위를 정하지 않습니다. 3-update 뒤 mIoU가 낮다는 이유만으로
장기 학습 실패를 뜻하지 않습니다. 최종 beta는 아직 선택하지 않았습니다.

다음 단계는 전체 runner의 val500·400-step 평가와 장기 checkpoint 재개를 완성하고,
계획된 25-batch 초기 손실 측정으로 beta 후보를 고정하는 것입니다. 그 뒤 2k/10k 선별을
진행합니다. NVIDIA MiT-B0 변환본의 CIRKD Baidu 원본 동일성 확인 또는 본실험 출처 revision,
FSKD 재구현 표시, C2VKD의 추가 CLIP pretraining 조건은 계속 명시해야 합니다.
이 로그 점검으로 새로운 GPU 실행이나 beta 선별을 시작하지 않았습니다.
