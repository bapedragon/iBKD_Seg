# H200 smoke v1 · 요청 819 로그 점검

2026-09-23 사용자 제공 로그를 점검했습니다. 실행 commit은 `d585015d844bc88388f402fd01f911cede266a07`입니다.
판정은 **3-update 학습 연결 7/7, checkpoint 재실행 통과 0/7, val 평가 0/7**입니다.
총 파이프라인 시간은 약 484초입니다. 학습 전체 성공으로 해석하지 않습니다.

## 통과한 부분

의존성 설치와 torchsort CUDA 경로, 단위 테스트 10개, Cityscapes ZIP·train2975/val500
준비, 공식 소스/가중치 bytes·SHA-256·적재가 진행됐습니다. 7개 방법 모두 실제
batch16/crop512로 3 update를 끝냈으며 NaN·Inf·OOM이 기록되지 않았습니다.
코드의 loss 항별 encoder gradient 및 parameter group gradient 검사도 update 전에 통과했습니다.

| 방법 | 총 loss: update 1 → 3 | 첫 실패 tensor의 최대 절대 차이 |
|---|---|---|
| vanilla | 6.607754 → 5.821393 | 8.702e-06 |
| lg | 7.062894 → 6.270872 | 5.972e-06 |
| alg | 7.062894 → 6.270870 | 7.182e-06 |
| ibkd_lambda025 | 7.035095 → 6.254459 | 5.797e-06 |
| ibkd_lambda050 | 7.033696 → 6.252599 | 6.091e-06 |
| fskd | 13.301565 → 11.672947 | 3.088e-06 |
| c2vkd_clip_pool | 3.631072 → 3.188081 | 3.111e-06 |

서로 목적함수가 다른 총 loss를 성능 순위로 비교하지 않습니다. β는 smoke의 3-batch
측정값이며 최종 β가 아닙니다. FSKD/C2VKD의 로그 beta=0은 해당 목적함수가 공통 beta를
사용하지 않아서 기록된 값이며, 증류가 꺼졌다는 뜻이 아닙니다.

## 실패 지점과 확인 범위

모두 `smoke.py`의 재실행 이후 `student.state_dict()` 비교에서 실패했습니다.
원본 3번째 update와 step2 checkpoint에서 재실행한 update의 일부 값이
기준 `rtol=2e-5, atol=2e-6`을 넘었습니다. 표시된 값은 **처음 실패한 tensor**의 차이이며
모델 전체의 최대 오차로 해석할 수 없습니다. v1 오류에는 정확한 key가 없습니다.

그 뒤의 optimizer/guide 비교, frozen 상태 해시 확인과 val2 평가는 실행되지 않았습니다.
따라서 teacher BN 불변·정확한 재개·평가 metric·peak memory를 완료 확인했다고 주장하지 않습니다.
v1 예외 처리에는 손실 이력도 최종 JSON에서 사라지는 문제가 있어, 위 표는 `[B0_STEP]`에서 추출했습니다.
원시 로그는 Git에 넣지 않고 해시와 추출 요약만 [log_audit.json](log_audit.json)에 기록했습니다.

## 원인 판단과 v2 수정

공통 runner는 cuDNN deterministic만 켰고 전체 결정적 연산 설정과 cuBLAS 재현성 설정은
누락했습니다. 실제 CUDA 보간의 backward는 비결정적일 수 있다는
[PyTorch 설명](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.interpolate.html)과도
맞는 양상입니다. 현재 로그만으로 특정 kernel 또는 복원 누락 중 하나로 완전히 확정하지는 않습니다.
LG/ALG의 초기 CE·임시 beta에도 작은 차이가 있어 GPU 수치 재현성이 유력한 원인입니다.

v2에서는 다음을 적용했습니다.

1. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict deterministic algorithms, TF32 off,
   math SDPA를 적용합니다. bilinear 크기·align_corners는 유지하며 PyTorch의 결정적 경로를 사용합니다.
2. CE는 같은 유효 픽셀 평균을 2D class logits로 계산해 CUDA `nll_loss2d` reduction을 피합니다.
   CPU double loss·gradient 동등성 검사를 통과했습니다.
3. iBKD에는 기존 L/16의 `flatmax_cpu_deform_v1` 경로를 연결합니다. 채널 최대값은
   같은 첫 최대 index를 쓰는 flat max이고, 작은 spatial deformable convolution은 CPU에서
   계산하며 autograd 연결을 유지합니다. CPU 수식·gradient 동등성 검사를 통과했습니다.
4. 재실행 **직전**의 모델·guide·optimizer·controller·RNG를 저장값과 bitwise 비교하고,
   재실행 **이후** 비교는 기존 허용오차를 유지합니다. 실패 key를 이름과 함께 기록합니다.
5. 모든 완료 update를 즉시 저장합니다. 재실행 수치 비교가 실패해도 원래 연속 학습의
   3번째 상태를 복원해 val2를 진단하며, 전체 판정은 실패로 유지합니다.

v2의 로컬 검사: 단위 검사 14개 통과, CUDA 전용 1개는 로컬 CUDA 부재로 skip.
실제 가중치와 합성 batch2/crop64의 CPU 연결·bitwise 재실행은 7/7 통과했습니다.
**v2 H200 결과는 아직 없습니다.** 다음 실행은 [H200 이슈 입력안](../../H200_SMOKE_ISSUE.md)을 사용합니다.
