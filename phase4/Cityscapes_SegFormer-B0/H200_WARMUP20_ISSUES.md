# iBKD warm-up 20 · H200 재실험 이슈 2개

**아래 λ=0.25 네 후보 10k 이슈는 이전 계획입니다.** Smoke와 β4개·2k를 완료했고,
2026-09-25 사용자 결정으로 상위 두 후보만 유지합니다. 이번 제출은
[ALG·iBKD 각 1위 통합 10k 이슈](H200_TOP1_10K_ISSUE.md)를 사용합니다. λ=0.5는 아직 실행하지 않습니다.

2026-09-24 준비. **iBKD λ=0.25·0.5만 warm-up 20, ALG는 0 유지**입니다.
[프로토콜](WARMUP20_PROTOCOL.md)에 종료 기준과 새 β 선별 계획을 고정했습니다.
아래는 제출용 입력값이며 외부 이슈를 생성하거나 GPU 실행을 시작한 것은 아닙니다.

각 λ의 기존 β 4개를 **처음부터 10,000 step까지** 실행합니다. 2k는 중간 기록입니다.
기존 warm-up 0의 2k checkpoint를 재개하지 않습니다. 별도 GPU smoke 이슈 없이 각 실행의
내장 입력·재개 검사와 warm-up 보호 검사를 사용합니다.

## 공통 입력값

[H200 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

## 이슈 1: iBKD λ=0.25

제목:

```text
[Request]: Cityscapes B0 iBKD lambda025 warmup20 beta4 10k
```

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda025
```

β=0.387021·0.903048·1.9351·3.87021의 **4개 실행**입니다.
출력은 `/app/output/cityscapes_b0_ibkd_warmup20_v1/lambda025/`입니다.

## 이슈 2: iBKD λ=0.5

제목:

```text
[Request]: Cityscapes B0 iBKD lambda050 warmup20 beta4 10k
```

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda050
```

β=0.4303·1.00403·2.1515·4.303의 **4개 실행**입니다.
출력은 `/app/output/cityscapes_b0_ibkd_warmup20_v1/lambda050/`입니다.

## 학습·선택과 종료

- 186 step마다 β 이전 raw guidance를 관측합니다. Window50·threshold−0.02를 유지합니다.
  첫 3720 step은 guidance를 사용하며, 가장 빠른 CE-only 학습은 step3721입니다.
  20구간 후에도 종료 조건을 만족하지 않으면 계속 guidance를 사용합니다.
- Batch16·crop512·FP32·TF32 off, AdamW LR6e-5·WD1e-4와 **80k poly schedule**을 유지합니다.
  Warm-up 20은 guidance 종료 보호이며 LR warm-up은 0입니다.
- 매 400 step 전체 val500을 평가합니다. 후보 1개당 10k까지 25회, 묶음당 100회입니다.
- 네 후보 모두 10k 완료 시 best val mIoU 내림차순 → best step 오름차순 → candidate ID 순으로
  **λ별 최종 β 1개**를 선택합니다. 미완료 후보가 있으면 selection은 `pending`입니다.
- 입력 32-batch 사전 검사, calibration의 첫 25-batch 해시 대조, step2→3 checkpoint 재실행과
  step500 안정성 확인을 포함합니다. 첫 20구간 보호 위반은 실패 처리합니다.
- `completed`는 목표 10k와 전체 평가 완료, `paused`는 시간 중단, `failed`는 실행 오류입니다.
  `pending`은 아직 실행·선택이 끝나지 않았다는 뜻입니다.

처음 3720 step 동안 iBKD 계산을 유지하므로 기존 조기 종료된 2k 실행 시간으로 새 작업 시간을
단순 추정할 수 없습니다. **4개 × 10k를 한 작업에서 끝낸다고 보장하지 않습니다.**
설치·준비를 포함해 [9시간 40분 예산](JOB_RUNTIME.md) 안에 저장 여유를 두고 중단하며 필요하면 여러 작업으로 재개합니다.
학습과 전체 val 시간을 새 로그로 측정한 뒤 완료 예상 시간을 갱신합니다.

## 결과와 재개

`group_summary.json`·`selection.json`·`run.log`와 `runs/`를 포함한 **출력 묶음 전체**를 보존합니다.
후보별 `summary.json`에는 실제 마지막 loss, best/last 전체 metric, 선택 step, controller 이력,
guidance 종료 step, 2k 경과 metric과 warm-up 보호 확인을 기록합니다.
`resume.json`과 `checkpoints/`에는 optimizer·RNG·입력 위치를 포함한 전체 상태가 있습니다.
마지막 로그 JSON에도 후보별 loss와 평가값·상태를 출력합니다.

다른 작업에서 재개할 경우 이전 **warm-up 20 출력**을 서버에 복원해야 합니다.
예를 들어 실제 복원 경로가 `/app/data/previous_w20/lambda025`라면:

```bash
B0_W20_RESUME_FROM=/app/data/previous_w20/lambda025 \
B0_W20_OUTPUT_BASE=/app/output/cityscapes_b0_ibkd_warmup20_v1_resume1 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda025
```

λ=0.5 재개 예시:

```bash
B0_W20_RESUME_FROM=/app/data/previous_w20/lambda050 \
B0_W20_OUTPUT_BASE=/app/output/cityscapes_b0_ibkd_warmup20_v1_resume1 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda050
```

경로는 실제 복원 위치에 맞춥니다. 다운로드·서버 업로드는 명령이 자동 처리하지 않습니다.
완료 후보는 checkpoint를 재검증하고, 중단 후보는 이어서, 미시작 후보는 초기화부터 실행합니다.
이전 warm-up 0의 pack2/pack3는 프로토콜이 달라 재개를 거부합니다.
새 작업의 PyTorch·CUDA 등 기록된 실행 환경도 같아야 합니다.
