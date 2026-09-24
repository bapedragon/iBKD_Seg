# iBKD λ=0.25 · warm-up 20 · β4개 2,000-step 재실험

2026-09-24 [네 β의 H200 smoke 통과](reports/h200_warmup20_smoke32_lambda025/RESULTS.md)를 확인했습니다.
사용자 결정대로 아래 네 후보를 각각 초기화부터 2,000 step까지 실행합니다.
외부 이슈나 GPU 작업은 이 문서를 작성하면서 자동 생성하지 않았습니다.

## 이슈 전체 입력값

[H200 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes B0 iBKD lambda025 warmup20 beta4 2k` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량 | **7 — H200 1장 전체** |

코드 실행 명령어:

```bash
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda025 2000
```

## 실행 조건과 확인 항목

- λ=0.25, β=**0.387021·0.903048·1.9351·3.87021**, 각 2,000 step으로 총 4개 실행입니다.
- Guidance warm-up=20, 186-step 관측, window50·threshold−0.02를 유지합니다.
  **2k는 최소 3720-step 보호 기간 안이므로 마지막까지 β와 guidance가 유지돼야 합니다.**
- 같은 seed1 student·guide 초기화에서 시작합니다. Smoke와 이전 warm-up 0 checkpoint를 쓰지 않습니다.
- Batch16·crop512·FP32·TF32 off, AdamW LR6e-5·WD1e-4, **80k poly schedule**을 유지합니다.
- 매 400 step **전체 val500**을 평가합니다. 후보당 5회, 묶음 전체 20회입니다.
- 초기 입력 대조·step2→3 파일 재실행과 step500 안정성 검사를 다시 수행합니다.
- 네 후보 모두 2k를 완료한 뒤 종료합니다. 자동 10k 실행이나 β 탈락은 수행하지 않습니다.

## 결과 확인과 보존

기본 출력은 다음과 같습니다.

```text
/app/output/cityscapes_b0_ibkd_warmup20_v1_check2000/lambda025/
```

마지막 JSON에서 전체 `status=completed`, `completed_runs=4`, `target_steps_per_run=2000`을
확인합니다. 각 후보는 `completed_steps=2000`, `last_eval_step=2000`, full val500,
`inline_replay.status=passed`, `stability_500_passed=true`, `guidance_on=true`, `stop_epoch=null`이어야 합니다.
`selection=not_performed`는 이번 2k 경과 확인에서 정상입니다. 각 후보의 best/last 전체 metric은 별도로 남습니다.
Warm-up 경계를 아직 지나지 않아 `warmup_protection_passed=null`도 정상입니다.

원본 `group_summary.json`, `run.log`, `runs/`와 그 안의 `resume.json`, `checkpoints/`를 포함한
출력 묶음 전체를 보존합니다. Last loss·metric·controller·실패 상태는 마지막 JSON에도 출력합니다.
기존 첨부처럼 로그 앞부분이 잘리더라도 전체 group summary를 별도로 보존하면 점검에 도움이 됩니다.

## 시간 제한과 재개

이번 smoke의 학습 평균은 2.93~3.45초/step입니다. 같은 속도로 8,000 step을 실행하면
학습만 약 6.5~7.7시간이며 전체 val500 평가·자료 준비·저장 시간이 추가됩니다.
이는 짧은 smoke에 근거한 추정이고 한 번의 작업에서 네 후보 완료를 보장하지 않습니다.
9시간 예산 안에 저장 여유를 두고 중단합니다. `paused`/`pending`은 이어서 실행해야 합니다.

이전 **warm-up 20·전체 val500 실행** 출력이 실제로 `/app/data/previous_w20_2k/lambda025`에
복원돼 있다면 다음처럼 재개합니다. 경로는 실제 복원 위치에 맞춥니다.

```bash
B0_W20_RESUME_FROM=/app/data/previous_w20_2k/lambda025 \
B0_W20_OUTPUT_BASE=/app/output/cityscapes_b0_ibkd_warmup20_v1_check2000_resume1 \
bash phase4/Cityscapes_SegFormer-B0/scripts/run_b0_warmup20.sh lambda025 2000
```

데이터/환경/코드/프로토콜 식별이 같아야 하며, 원격 출력의 다운로드·복원은 자동 처리하지 않습니다.
Smoke 출력은 여기서 재개할 수 없습니다.
