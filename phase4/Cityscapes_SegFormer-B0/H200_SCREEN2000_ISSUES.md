# SegFormer-B0 · 2,000-step 선별 이슈 3개

**2026-09-24 후속 변경:** 이 문서는 기존 warm-up 0 실행 기록입니다.
iBKD 두 λ의 새 실험에는 [warm-up 20 이슈 2개](H200_WARMUP20_ISSUES.md)를 사용합니다.
ALG는 0을 유지하며, 아래 기존 iBKD checkpoint와 상위 2개 선정을 새 revision에 승계하지 않습니다.

**2026-09-24 pack1 v2는 6/6 완료**했습니다. [결과와 LG 상위 2개](reports/h200_screen2000_v2_pack1/RESULTS.md)를
확인했으며 완료한 pack1 2k 이슈를 다시 제출할 필요는 없습니다.
Pack2도 [후보별 기록](reports/h200_screen2000_v2_pack2/RESULTS.md)에서 8개 후보의 2k 완료를 확인했습니다.
앞부분이 잘린 첨부라 전체 집계는 미확인이며, ALG β=0.197479·0.460784와
iBKD λ=0.25 β=0.903048·3.87021을 후보별 점수로 선정했습니다.
Pack3도 [최종 요약에서 4/4 완료](reports/h200_screen2000_v2_pack3/RESULTS.md)를 확인했고,
iBKD λ=0.5 β=0.4303·1.00403을 선정했습니다.
아래 입력값은 세 묶음의 실행 기록으로 보존합니다. 완료한 2k 이슈를 다시 제출하지 않습니다.
Pack1의 다음 단계는 Vanilla·FSKD·LG β=0.197479·0.460784의 step2000 전체 상태를 이용한 10k 재개입니다.

사용자 지정 묶음으로 2026-09-23 준비했습니다. **별도 GPU smoke 이슈 대신 각 실행에
checkpoint 복원·재실행 검사를 포함**합니다. 통과하면 동일한 학습 상태에서 2,000 step까지
진행합니다. 이 문서는 제출할 입력값이며 외부 이슈나 GPU 작업을 자동 생성한 것이 아닙니다.

## 2026-09-23 v2 수정

첫 v1 pack1은 6개 실행 모두 25 update 후 `Invalid transformed sample`로 실패했습니다.
개별 crop에 유효 라벨이 없으면 로딩을 거부했던 버그를 수정했습니다.
CIRKD처럼 해당 crop을 유지하며 CE·logit KD는 배치의 유효 픽셀만 계산합니다.
Feature guidance는 기존 배치 그대로 계산합니다. 배치 전체가 ignore이면 명시적으로 실패합니다.
β grid·초기화·증강 순서·손실 계수·학습 및 평가 조건은 유지합니다.

**아래 명령은 최신 저장소의 v2를 실행합니다. v1 체크포인트를 재개하지 않고 처음부터 실행합니다.**
세 묶음에 같은 수정본을 적용하며 출력도 `cityscapes_b0_screen2000_v2`로 분리합니다.
별도 smoke 이슈는 추가하지 않습니다. 입력 사전 검사는 32 batch로 늘렸고 학습의 내장 재개 검사는 유지합니다.
이미 시작한 v1 작업에는 이 변경이 자동 반영되지 않습니다.
[실패 분석과 검사 결과](reports/h200_screen2000_v1_pack1_failure/RESULTS.md)를 참고합니다.

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
[Request]: Cityscapes B0 2k v2 pack1 - Vanilla FSKD LG
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
[Request]: Cityscapes B0 2k v2 pack2 - ALG iBKD lambda025
```

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_screen2000.sh pack2
```

ALG β 4개 + iBKD λ=0.25 β 4개 = **8개 실행**입니다.
Guidance를 계속 계산할 때 순수 학습 환산값은 약 **7시간 55분**입니다.
전체 val 평가 40회가 추가되므로 **한 작업에서 전부 완료하지 못할 가능성이 있습니다.**
현재 코드는 [9시간 40분 예산](JOB_RUNTIME.md) 안에 checkpoint를 남기고 종료하며, 남은 실행은 같은 묶음을 재개합니다.
ALG/iBKD가 자연스럽게 guidance를 종료하면 이후 teacher·guide 계산을 생략해 빨라질 수 있습니다.

## 이슈 3: iBKD λ=0.5

제목:

```text
[Request]: Cityscapes B0 2k v2 pack3 - iBKD lambda050
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
[실행 명세](configs/b0_screen2000_v2.json)에 해시를 연결했습니다. 각 후보는 seed1의 같은
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
   v2는 입력만 32 batch까지 읽고 배치별 유효 픽셀 수와 ignore-only crop의 배치·이름을 기록합니다.
   optimizer update 없이 수행하며 학습은 원래 첫 배치부터 시작합니다.
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

결과는 `/app/output/cityscapes_b0_screen2000_v2/pack1/`처럼 묶음별로 저장됩니다.

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

설치·준비를 포함한 현재 9시간 40분 예산에 저장 여유를 두고 중단합니다. `paused`/`pending`은 완료가 아닙니다.
**출력 묶음 전체를 보존**해야 합니다. `best` 가중치만으로 2k/10k 학습을 이어가지 않습니다.

**아래 재개 명령은 v2끼리만 사용합니다.** 다른 작업에서 이어갈 때는 이전 v2 출력 묶음을 서버에 풀어 놓고, 예를 들어 실제 경로가
`/app/data/previous_screen/pack2`라면 아래처럼 실행합니다. 경로는 실제 복원 위치에 맞춥니다.

```bash
B0_SCREEN_RESUME_FROM=/app/data/previous_screen/pack2 \
B0_SCREEN_OUTPUT_BASE=/app/output/cityscapes_b0_screen2000_v2_resume1 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_screen2000.sh pack2
```

이전 후보의 재개 묶음을 새 출력에 복사한 뒤 완료한 후보는 재검증하고,
중단된 후보는 이어서, 아직 시작하지 않은 후보는 처음부터 실행합니다.
원격 출력의 다운로드·복원은 이 명령이 자동 처리하지 않습니다.

## 준비 검사 결과

v2 로컬 단위 검사는 **53 passed, 1 CUDA-only skipped**입니다.
25개 CIRKD 합성 변환의 bitwise 일치, worker/prefetch 재개, 평가 중단·재개, 80k LR,
guide off, 동점 처리·후보 완결성 검사에 더해 아래 회귀 검사를 통과했습니다.

- 26번째 batch에 ignore-only crop을 넣은 직렬·thread 로더에서 27 update 완료.
- CE·logit KD가 배치 유효 픽셀로 정규화되고 ignore-only sample의 logit gradient가 0임을 확인.
- CIRKD 원본과 ignore-only crop 출력이 일치하며, validation 입력에서도 해당 이미지를 허용.
- 실제 CIRKD/NVIDIA 가중치와 합성 batch2·crop64로 **6개 방법 모두 6 update·3회 평가·step2→3 파일 재개 통과**.
  이 검사에서는 replay batch와 validation에 ignore-only sample을 각각 1개 넣었습니다.

CPU 검사에는 실제 실패 이미지가 포함되지 않았으며 H200 2k 완료를 의미하지 않습니다.
v2의 실제 32-batch 입력 확인, 전체 val500, 2k 선별은 재실행 로그로 확인해야 합니다.
