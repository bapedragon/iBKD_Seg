# SegFormer-B0 · 2,000-step 선별 이슈 3개

사용자 지정 묶음으로 2026-09-23 준비했습니다. **별도 GPU smoke 이슈 대신 각 실행에
checkpoint 복원·재실행 검사를 포함**합니다. 통과하면 동일한 학습 상태에서 2,000 step까지
진행합니다. 이 문서는 제출할 입력값이며 외부 이슈나 GPU 작업을 자동 생성한 것이 아닙니다.

## 세 이슈의 공통 입력값

[H200 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

## 이슈 1: Vanilla·FSKD·LG

제목:

```text
[Request]: Cityscapes B0 2k pack1 - Vanilla FSKD LG
```

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_screen2000.sh pack1
```

Vanilla 1개 + FSKD 재구현 고정값 1개 + LG β 4개 = **6개 실행**입니다.
순수 학습의 기존 smoke 속도 환산값은 약 **2시간 54분**입니다.
전체 val 500장 평가 30회와 설치·자료 준비·저장 시간이 추가됩니다.
FSKD에 필요한 torchsort CUDA 확장은 이 묶음에서만 설치합니다.

## 이슈 2: ALG·iBKD λ=0.25

제목:

```text
[Request]: Cityscapes B0 2k pack2 - ALG iBKD lambda025
```

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_screen2000.sh pack2
```

ALG β 4개 + iBKD λ=0.25 β 4개 = **8개 실행**입니다.
Guidance를 계속 계산할 때 순수 학습 환산값은 약 **7시간 55분**입니다.
전체 val 평가 40회가 추가되므로 **한 작업에서 전부 완료하지 못할 가능성이 있습니다.**
9시간 전에 checkpoint를 남기고 종료하며, 남은 실행은 같은 묶음을 재개합니다.
ALG/iBKD가 자연스럽게 guidance를 종료하면 이후 teacher·guide 계산을 생략해 빨라질 수 있습니다.

## 이슈 3: iBKD λ=0.5

제목:

```text
[Request]: Cityscapes B0 2k pack3 - iBKD lambda050
```

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_screen2000.sh pack3
```

iBKD λ=0.5 β 4개 = **4개 실행**입니다.
Guidance를 계속 계산할 때 순수 학습 환산값은 약 **5시간 42분**입니다.
전체 val 평가 20회와 준비·저장 시간이 추가됩니다.

## 사전에 고정한 β와 학습 조건

| 방법 | β 후보 1 | 후보 2 | 후보 3 | 후보 4 |
|---|---:|---:|---:|---:|
| LG·ALG | 0.197479 | 0.460784 | 0.987394 | 1.97479 |
| iBKD λ=0.25 | 0.387021 | 0.903048 | 1.9351 | 3.87021 |
| iBKD λ=0.5 | 0.4303 | 1.00403 | 2.1515 | 4.303 |

이 값은 통과한 [25-batch calibration 결과](reports/h200_beta_calibration_v1/RESULTS.md)에서
계산했습니다. [고정 grid](configs/b0_beta_grid_frozen_v1.json)와
[실행 명세](configs/b0_screen2000_v1.json)에 해시를 연결했습니다. 각 후보는 seed1의 같은
student·guide 초기 상태에서 시작합니다. 이전 후보의 학습 가중치를 다음 후보에 넘기지 않습니다.

- 실제 batch16, crop512, FP32, TF32 off, AdamW LR6e-5·WD1e-4, clipping 없음.
- **80,000-step poly schedule**의 처음 2,000 step입니다. LR 분모를 2,000으로 바꾸지 않습니다.
- 매 400 step 전체 val 500장, 원본 해상도 좌우 1024 정사각형 분할·logits 결합 후 확대.
- best val mIoU가 같은 경우 먼저 나온 checkpoint를 유지합니다.
  네 β 후보가 모두 완료되면 best mIoU 내림차순 → best step 오름차순 → candidate ID 순서로
  **방법별 상위 2개**를 `selection.json`에 기록합니다. 10k 실행은 자동 시작하지 않습니다.
- Vanilla·FSKD는 β 선택을 하지 않으며, 낮은 초기 mIoU를 이유로 제외하지 않습니다.
- ALG/iBKD controller는 186 step마다 raw guidance 평균을 관측합니다.
  window50, threshold−0.02, warm-up0을 유지합니다. 꺼진 이후 guide gradient는 없으며
  guide의 optimizer update/weight decay도 수행하지 않습니다.
- FSKD는 미공개 segmentation 설정을 보완한 재구현입니다. C2VKD(CLIP-pool)는 이 세 묶음에 없습니다.

이번 선별은 smoke/calibration과 같은 **NVIDIA 공식 HF ImageNet-only MiT-B0 역변환**을
초기화 출처로 명시해 사용합니다. CIRKD Baidu 원본과 동일하다고 주장하지 않습니다.
원래 공통 JSON은 보존하고, 이번 실행 명세에서 출처 예외를 모든 비교 조건에 동일하게 승계했습니다.

## 내장 검사와 실패 처리

1. ZIP 또는 준비 데이터의 provenance·실제 파일 해시, 가중치 bytes/SHA-256, 고정 프로토콜을 확인합니다.
2. CIRKD의 serial seed1 난수 선택을 명시적 증강 계획으로 만들고 CPU thread4로 로딩합니다.
   순서와 증강은 prefetch·재개 위치·방법에 영향을 받지 않습니다.
   **첫 25 batch의 전체 tensor hash가 완료한 calibration과 정확히 같은지** 학습 전에 검사합니다.
3. 각 후보에서 step2의 모델·guide·optimizer·BN·RNG·controller·sampler 위치를 실제 파일에 저장합니다.
   이를 복원해 step3을 두 번 계산하고 원래 연속 경로를 보존합니다. 복원은 bitwise,
   재실행은 smoke v2와 같은 `rtol=2e-5, atol=2e-6`으로 비교합니다.
   유효 학습 step은 3이며 재실행을 추가 optimizer step으로 세지 않습니다.
4. 첫 정규 전체 val은 step400입니다. Step500에서 내장 재개·전체 평가·유한성 검사를 확인하고
   같은 실행을 계속합니다. 500 step마다 새로 시작하지 않습니다.
5. 손실·gradient·parameter의 NaN/Inf 또는 재개 불일치면 해당 후보를 실패 처리합니다.
   나머지 후보는 계속하지만, 네 후보가 모두 정상 완료되기 전에는 상위 2개를 확정하지 않습니다.
   유한한 일시적 손실 상승이나 낮은 초기 mIoU는 자동 제외 사유가 아닙니다.

## 출력과 재개

결과는 `/app/output/cityscapes_b0_screen2000_v1/pack1/`처럼 묶음별로 저장됩니다.

| 파일/폴더 | 내용 |
|---|---|
| `group_summary.json`, `run.log` | 모든 후보의 실제 최종 loss·best/last metric·완료 step·상태 |
| `selection.json` | 완료한 방법의 후보 순위·상위 2개, 미완료면 pending |
| `input_preflight.json`, `augmentation_plan.npy` | 실제 입력 일치 검사와 전체 80k 증강·샘플 순서 계획 |
| `runs/<run_id>/summary.json` | 해당 후보의 평가 이력·controller·재개 검사·시간·메모리 |
| `runs/<run_id>/training.jsonl` | step별 손실·β·gradient norm·입력 hash |
| `runs/<run_id>/resume.json`와 `checkpoints/` | 최근/이전 세대의 전체 재개 상태와 best student |

매 100 step, 평가 후, 초기 검사 후 및 시간 종료 때 전체 상태를 저장합니다.
입력·방법·β·코드·환경·프로토콜이 달라지면 재개를 거부합니다.
현재 checkpoint가 손상됐으면 checksum을 통과하는 이전 세대로 복구합니다.
평가 중 시간 제한에 도달하면 부분 val 점수는 버리고, 재개 후 그 step의 전체 val을 다시 수행합니다.

설치·준비를 포함한 9시간 예산에 저장 여유를 두고 중단합니다. `paused`/`pending`은 완료가 아닙니다.
**출력 묶음 전체를 보존**해야 합니다. `best` 가중치만으로 2k/10k 학습을 이어가지 않습니다.

다른 작업에서 이어갈 때는 이전 출력 묶음을 서버에 풀어 놓고, 예를 들어 실제 경로가
`/app/data/previous_screen/pack2`라면 아래처럼 실행합니다. 경로는 실제 복원 위치에 맞춥니다.

```bash
B0_SCREEN_RESUME_FROM=/app/data/previous_screen/pack2 \
B0_SCREEN_OUTPUT_BASE=/app/output/cityscapes_b0_screen2000_v1_resume1 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_screen2000.sh pack2
```

이전 후보의 재개 묶음을 새 출력에 복사한 뒤 완료한 후보는 재검증하고,
중단된 후보는 이어서, 아직 시작하지 않은 후보는 처음부터 실행합니다.
원격 출력의 다운로드·복원은 이 명령이 자동 처리하지 않습니다.

## 준비 검사 결과

로컬 단위 검사 **49 passed, 1 CUDA-only skipped**입니다. 공식 CIRKD 변환과 새 증강 계획의
25개 합성 입력이 bitwise 일치했고, worker/prefetch 재개, 중단된 평가 재시도, 80k LR,
guide off, 동점 처리·후보 완결성도 검사했습니다.
실제 CIRKD/NVIDIA 가중치와 합성 batch2·crop64로 **6개 방법 모두 새 공통 학습 루프의
6 update·3회 평가·step2→3 파일 재개 검사를 통과**했습니다.
이것은 실제 Cityscapes 2k 결과가 아닙니다. 새 runner의 H200 검사는 위 내장 검사에서 수행합니다.
