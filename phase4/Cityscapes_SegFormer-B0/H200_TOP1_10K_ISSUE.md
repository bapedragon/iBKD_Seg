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
| 작업 시간 예산 | **9시간 40분 — 두 실행 합계** |

코드 실행 명령어 — 기존 출력 복원 없이 실행할 때:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_top1_10k.sh fresh
```

이 명령은 같은 seed1 초기화부터 **각각 총 10,000 step**을 실행합니다. 과거 2k가 자동으로
복원된다고 가정하지 않습니다. 기존 결과와 같은 앞 2k를 반복하는 비용이 포함됩니다.
기존 2k 전체 출력이 서버에 복원돼 있으면 아래 재개 명령을 대신 사용합니다.

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

## 시간과 출력 보존

설치·준비·두 학습·평가·저장이 **하나의 9시간 40분 예산**을 공유합니다.
시간 제한은 [기존 저장 여유](JOB_RUNTIME.md)를 유지합니다. 두 후보 모두 10k를 한 작업에서
끝낸다고 보장하지 않습니다. iBKD가 3720 이후에도 guidance를 유지하면 더 오래 걸릴 수 있습니다.
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
최종 JSON 출력과 기존 상태 보호를 로컬에서 검사했습니다. **48 passed, 2 skipped**이며 두 skip은
로컬 upstream cache 부재에 따른 기존 입력 대조 검사입니다. 기존 H200 로그의 학습 source hash와
일치하고 Python CLI·shell 문법·문서 링크 검사도 통과했습니다. 별도 GPU smoke 이슈는 추가하지 않습니다.
