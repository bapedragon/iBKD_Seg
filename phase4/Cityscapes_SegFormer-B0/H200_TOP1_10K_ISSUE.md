# ALG·iBKD λ=0.25 각 1위 β · 10,000-step 이슈

2026-09-25 사용자 결정: 네 후보의 2k 결과에서 **방법별 상위 두 β를 유지**하고,
이번에는 그중 **각 1위 후보 하나씩** 10k를 실행합니다. 아래 한 이슈는 ALG → iBKD 순서의
두 실행입니다. 외부 이슈 등록이나 GPU 실행은 자동으로 시작하지 않았습니다.

| 방법 | 이번 β | 다음 β | 이번 β의 2k mIoU | guidance 보호 |
|---|---:|---:|---:|---:|
| ALG | **0.197479** | 0.460784 | 44.7611% | **0 epoch** |
| iBKD λ=0.25 | **0.387021** | 1.9351 | 45.3352% | **20 epoch** |

ALG는 [기존 pack2](reports/h200_screen2000_v2_pack2/RESULTS.md), iBKD는
[warm-up 20·2k 결과](reports/h200_warmup20_check2000_lambda025/RESULTS.md)의 순위입니다.
ALG 첫 후보의 best는 잘린 로그의 last metric과 selected step에서 확인한 값이라는 기존 한계를
유지합니다. iBKD warm-up 0에서 고른 β=0.903048·3.87021을 이번 후보로 쓰지 않습니다.

## 이슈 전체 입력값

[H200 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes B0 ALG + iBKD lambda025 warmup20 top1 beta 10k` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

코드 실행 명령어 — 체크포인트 자동 확인:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_top1_10k.sh auto
```

각 후보별로 서버의 `/app/output`·`/app/data` 아래에서 압축이 풀린 run 폴더를 찾습니다.
호환되는 전체 checkpoint가 있으면 **가장 많이 진행한 step부터 총 10k까지** 이어가고,
없으면 같은 seed1 초기화부터 총 10k를 실행합니다. 한 방법만 checkpoint가 있어도 각각 판단합니다.
현재 출력 폴더의 같은 작업이 있으면 그 group부터 이어갑니다. 재실행할 때도 같은 명령을 씁니다.

- 파일명이 같아도 warm-up 0 iBKD, smoke val2, 다른 프로토콜은 제외하고 발견 내역을 기록합니다.
- 호환 후보를 찾았는데 checkpoint가 누락·손상됐거나 코드·환경이 다르면 오류로 알립니다.
  이 경우에 조용히 초기화해서 재학습하지 않습니다.
- 로그·mIoU 결과만 있거나 압축 파일만 있는 경우 전체 상태를 복원할 수 없습니다.
  원격 서버의 과거 출력을 자동 다운로드하거나 압축 해제하지 않습니다.
- 탐색은 두 root 아래 최대 9단계이며 데이터셋·weights·checkpoints 내부는 순회하지 않습니다.
  다른 위치라면 `B0_10K_SEARCH_ROOTS=/실제/복원/위치`를 지정하거나 아래 개별 경로를 사용합니다.
- `[B0_START]` 로그에 후보별 `full_checkpoint` 또는 `initialization`, 경로와 기존 step을 출력합니다.
  실제 복원 step은 worker의 `[B0_RESUME]`에 남깁니다.

전체 checkpoint가 2k이면 추가 8k입니다. 강제로 처음부터 반복하려는 경우에만 `auto` 대신
`fresh`를 지정하고 새로운 출력 폴더를 사용합니다.

## 기존 2k에서 이어갈 때

ALG와 warm-up 20 iBKD의 **각 후보 run 폴더 전체**가 필요합니다. 각 폴더에는
`summary.json`, `resume.json`, `checkpoints/`와 best 가중치가 함께 있어야 합니다.
아래 두 경로는 예시이며 실제 복원 위치로 바꿉니다. 로컬 첨부 로그만으로는 재개할 수 없습니다.

```bash
B0_10K_ALG_RESUME_FROM=/app/data/previous_b0/pack2/runs/alg_beta_r003 \
B0_10K_IBKD_RESUME_FROM=/app/data/previous_w20/lambda025/runs/ibkd_lambda025_beta_r003 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_top1_10k.sh resume
```

각 2,000에서 **추가 8,000 step**, 총 10,000까지 진행합니다. Best student만 읽거나 optimizer를
초기화하지 않습니다. 후보·warm-up·학습 코드·프로토콜을 확인한 뒤 전체 상태를 복원하고,
실제 worker가 checkpoint 해시와 Python/PyTorch/CUDA/GPU 등 실행 환경까지 대조합니다.
불일치나 파일 누락은 실패로 표시하며 자동으로 처음부터 다시 시작하지 않습니다.

## 이번 범위와 성공 기준

- `alg_beta_r003`, `ibkd_lambda025_beta_r003` **두 개만** 실행합니다. 2순위·λ=0.5는 실행하지 않습니다.
- iBKD의 첫 3720 update는 guidance를 유지합니다. 조건을 만족할 때 가장 빨라야 3721부터
  CE-only로 바뀌며, 20 epoch에서 반드시 끄는 규칙은 아닙니다. ALG는 기존 controller를 유지합니다.
- Batch16·crop512·FP32·TF32 off, AdamW LR6e-5·WD1e-4, **80k poly LR schedule**을 유지합니다.
- 매 400 step **전체 val500**을 평가합니다. Fresh는 후보별 25회, 2k에서 재개하면 추가 20회입니다.
- 입력 32-batch 점검과 calibration 해시 대조를 실행합니다. Fresh는 step2→3 파일 재실행과
  step500 안정성 검사도 수행하며 재개 시 그 검사 결과와 controller 이력을 승계합니다.
- 두 후보 각각 `completed_steps=last_eval_step=10000`, 전체 group `status=completed`,
  `completed_runs=2`가 완료 기준입니다. Last loss, best/last 전체 metric, selected step,
  guidance stop step과 warm-up 보호 확인을 마지막 JSON 및 파일에 남깁니다.
- **`selection.status=pending`은 정상**입니다. 각 방법의 2순위 후보도 10k를 마친 뒤에만
  best val mIoU → 이른 best step → candidate ID 순으로 최종 β 하나를 고릅니다.
  두 방법 사이에서 하나를 탈락시키거나 이번 각 1위 β를 바로 최종값으로 확정하지 않습니다.

## 예상 소요시간과 기존 중단 설정

**예상 시간은 실측 속도로 계산하며, 사용자 발언의 시간을 실행 제한으로 바꾸지 않습니다.**
9시간 55분을 새 예상값이나 코드 제한으로 설정하지 않았습니다.

2k 실측에서 학습+입력 대기+저장 평균은 ALG 1.018초/step, iBKD on 2.855초/step입니다.
CE-only 속도는 Vanilla의 0.480초/step을 근사치로 사용했고, 전체 val500은 회당
ALG 59.5초·iBKD 69.0초입니다. ALG는 현재 속도를 유지한다고 가정합니다.

| 시작 상태 | iBKD가 3720 부근에서 off | iBKD가 10k까지 on |
|---|---:|---:|
| 두 후보 모두 2k에서 재개 | **약 5시간 30분~6시간 30분** | **약 9시간 30분~10시간 30분** |
| 두 후보 모두 처음부터 | **약 8~9시간** | **약 12~13시간** |

표는 ALG+iBKD 합계이며 준비·복원·환경 변동 여유를 포함한 추정입니다. Guidance off 시점은
미확정이고 ALG도 조기에 꺼지면 더 짧아질 수 있습니다. 한 후보만 재개하면 두 경우 사이입니다.
근거는 기존 [ALG](reports/h200_screen2000_v2_pack2/log_audit.json),
[iBKD warm-up 20](reports/h200_warmup20_check2000_lambda025/log_audit.json),
[Vanilla](reports/h200_screen2000_v2_pack1/log_audit.json)의 시간 기록입니다.

이와 별개로 **현재 코드에는 이전에 적용된 9시간 40분 중단 예산이 남아 있습니다.**
이는 완료 예상 시간이 아니며 이번 정정에서 변경하지 않았습니다. 두 실행이 공유합니다.
[저장 여유](JOB_RUNTIME.md)를 두고 중단하므로 예상 총 소요가 한 작업을 넘으면 재개가 필요합니다.
`paused`는 전체 상태 저장 후 시간 중단, `pending` run은 시간 부족으로 이번 작업에서 미완료입니다.

```text
/app/output/cityscapes_b0_top1_10k_v1/
```

`group_summary.json`, `run.log`, `selection.json`, `runs/`를 포함한 **전체 출력**을 보존합니다.
이 합친 작업의 출력을 다음 서버에 `/app/data/previous_top1_10k`로 복원했다면:

```bash
B0_10K_RESUME_FROM=/app/data/previous_top1_10k \
B0_10K_OUTPUT=/app/output/cityscapes_b0_top1_10k_v1_resume1 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_top1_10k.sh resume
```

완료 후보는 전체 상태를 검증하고, 중단 후보는 해당 step부터, 미시작 후보는 초기화부터
진행합니다. 과거 2k 개별 경로 변수와 합친 group 재개 변수는 동시에 지정하지 않습니다.

## 계획 변경과 로컬 확인

[새 실행 범위 JSON](configs/b0_top2_10k_v1.json)은 λ=0.25의 네 후보 모두 10k 계획을
사용자 결정대로 상위 두 후보 비교로 대체합니다. 다른 학습 설정은 변경하지 않습니다.
기존 고정 JSON의 후보 선별 문구는 checkpoint 식별 보존용 과거 기록이며,
이번 runner의 실행 범위·선별 보류는 새 JSON을 따릅니다. λ=0.5의 후보 범위는 변경하지 않습니다.

학습 worker와 공통 controller 소스는 보존했습니다. 새 작업 묶음의 후보 라우팅,
잘못된 warm-up/후보/코드/smoke 재개 거부, 시간 부족 시 두 번째 실행 보류,
최종 JSON 출력·기존 상태 보호·자동 발견·한 방법만 재개·다른 warm-up 제외를 로컬에서 검사했습니다. **53 passed, 2 skipped**이며 두 skip은
로컬 upstream cache 부재에 따른 기존 입력 대조 검사입니다. 기존 H200 로그의 학습 source hash와
일치하고 Python CLI·shell 문법·문서 링크 검사도 통과했습니다. 별도 GPU smoke 이슈는 추가하지 않습니다.
