# iBKD λ=0.25 · warm-up 20 · β 4개 smoke

2026-09-24 사용자 결정: **먼저 짧은 smoke를 통과한 뒤, λ=0.25의 β 4개를 각각
2,000 step까지 다시 실행**합니다. 두 λ의 10k 실행을 바로 시작하지 않습니다.
ALG warm-up 0과 기존 warm-up 0 결과는 유지합니다.

## 지금 제출할 smoke 이슈 전체 입력값

[H200 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes B0 iBKD lambda025 warmup20 beta4 smoke32` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda025 smoke
```

이 문서는 제출용 입력안입니다. 외부 이슈를 생성하거나 GPU 실행을 시작한 것은 아닙니다.

## 이번 이슈의 실행 범위

| 후보 | β | λ | Guidance warm-up | 실제 학습 |
|---|---:|---:|---:|---:|
| beta_r003 | 0.387021 | 0.25 | 20 | 32 step |
| beta_r007 | 0.903048 | 0.25 | 20 | 32 step |
| beta_r015 | 1.9351 | 0.25 | 20 | 32 step |
| beta_r030 | 3.87021 | 0.25 | 20 | 32 step |

네 후보를 각각 같은 초기 student·guide에서 순차 실행합니다. 후보 간 학습 상태를 전달하지 않습니다.
기존 25-batch calibration의 β를 그대로 사용하며 smoke에서 다시 계산하지 않습니다.

- 실제 Cityscapes **batch16·crop512·FP32·TF32 off**, 80k poly LR schedule의 첫 32 step입니다.
- 실제 데이터·가중치 해시 및 32-batch 입력 사전 검사, 첫 25-batch calibration 해시를 대조합니다.
- 각 후보에서 step2의 student·guide·optimizer·RNG·controller·입력 위치를 파일에 저장하고
  step3을 재실행합니다. 복원은 bitwise, 재실행 비교는 `rtol=2e-5, atol=2e-6`입니다.
- 32 update 동안 CE·alignment·fusion·raw/weighted guidance와 β, gradient norm의 유한성,
  student·guide의 gradient 연결 및 frozen teacher 상태를 확인합니다.
- 후보마다 step32 후 **val 첫 2장**을 원래 해상도·좌우 분할 방식으로 평가합니다.
  전체 val500이나 성능 선별 결과가 아닌 진단 지표입니다.
- Warm-up 20, 가장 빠른 종료 경계 3720/3721 및 경계 파일 재개의 **별도 toy/합성 손실 단위 검사**를
  먼저 실행합니다. 실제 32-step GPU 학습이 3720-step 경계를 통과했다는 뜻은 아닙니다.
- 네 후보의 smoke가 끝나면 종료합니다. **2k나 10k 학습은 자동 시작하지 않습니다.**

32 step은 첫 controller 관측인 step186보다 짧으므로 실제 관측 완료 구간 수는 0입니다.
종료 보호는 설정·단위 검사와 학습 중 상태 확인으로 점검하고, 실제 장기 관측은 후속 실행에서 확인합니다.

## 통과 판정과 출력

기본 출력 폴더:

```text
/app/output/cityscapes_b0_ibkd_warmup20_v1_smoke32/lambda025/
```

마지막 JSON 및 `group_summary.json`에서 아래를 확인합니다.

- 전체 `status="completed"`, `completed_runs=4`, `target_steps_per_run=32`, `diagnostic_smoke=true`.
- 네 후보 모두 `completed_steps=32`, `inline_replay.status="passed"`.
- `guidance_warmup_epochs=20`, 마지막 loss의 `guidance_on=true`, β가 해당 후보값과 동일.
- `controller.controller.active=true`, `stop_epoch=null`, `guidance_stop_after_step=null`.
- `diagnostic_metrics.validation_samples=2`, `full_validation=false`, 유한한 loss·평가지표.
- `selection`은 `not_performed`, `selected_epoch`·`selected_step`은 null입니다.

`guidance_protection_status="protected_so_far_boundary_not_reached"`와
`minimum_period_reached=false`, `warmup_protection_passed=null`은 **32-step 점검에서 정상**입니다.
아직 3720 step을 실행하지 않았으므로 전체 보호 기간을 검증했다고 표시하지 않습니다.
부분 평가 점수로 후보 순위를 정하지 않습니다. `paused`는 미완료, `failed`는 오류입니다.

`run.log`, `group_summary.json`, `runs/<run_id>/summary.json`과 재개 상태를 포함한 출력 전체를
보존합니다. 마지막 로그 JSON에는 각 후보의 실제 loss, 진단 metric, controller·재개 상태가 포함됩니다.

## Smoke 통과 후 별도 2,000-step 실행

검사 결과 확인 후 별도 이슈에 사용할 명령은 다음과 같습니다.

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda025 2000
```

네 β 모두 공통 초기화부터 **각 2,000 step**, 매 400 step 전체 val500을 평가하고 종료합니다.
출력은 `/app/output/cityscapes_b0_ibkd_warmup20_v1_check2000/lambda025/`입니다.
Warm-up 20은 3720 step이므로 이 네 실행에서는 **2k 끝까지 guidance가 켜져 있어야** 합니다.
20-epoch 이후의 자동 종료 효과나 최종 성능을 이번 2k만으로 결론내리지는 않습니다.

Smoke의 val2 checkpoint는 2k/10k로 재개할 수 없도록 signature로 분리했습니다.
기존 warm-up 0 checkpoint도 사용하지 않습니다. 반면 새 warm-up 20의 **전체 val500 2k**
checkpoint는 모든 설정·코드·환경이 같다면 이후 10k로 이어갈 수 있습니다.
현재 우선순위는 smoke → λ=0.25 β4개·2k 결과 확인이며, 그 다음 실행은 결과 확인 후 진행합니다.

## 로컬 준비 확인

관련 단위 검사 **42 passed, 2 skipped**입니다. 네 β 각각의 toy model 32-step 학습,
파일 재개·재실행, val2 평가 및 smoke checkpoint의 본실험 재개 거부를 확인했습니다.
두 skip은 로컬에 CIRKD upstream cache가 없어 생략된 기존 입력 대조 검사입니다.
실제 H200 smoke는 아직 실행하지 않았으며 위 이슈에서 확인합니다.
Shell의 smoke/2000/10000 명령 전달과 Python/shell 문법 검사도 통과했습니다.
