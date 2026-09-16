# H200 본학습 이슈 입력안 — Cityscapes L/16 seed1

사용자가 직접 제출하는 입력안입니다. 이슈를 자동 생성하지 않습니다.
**한 GPU에서 한 방법씩**, 앞 작업이 끝난 뒤 다음 방법을 제출합니다.
네 방법은 같은 student 초기 가중치에서 독립적으로 학습합니다.

[H200 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에
아래 공통 항목과 원하는 방법의 명령을 입력합니다.

| 항목 | 입력값 |
|---|---|
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량(MIG 갯수) | `7` — H200 한 장 전체 |

## 1. Vanilla

제목: `[Request]: Cityscapes Segmenter-L16 Vanilla seed1 본학습 216epoch`

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_full.sh vanilla
```

## 2. LG

제목: `[Request]: Cityscapes DeepLabV3-to-Segmenter-L16 LG seed1 본학습 216epoch`

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_full.sh lg
```

## 3. ALG

제목: `[Request]: Cityscapes DeepLabV3-to-Segmenter-L16 ALG seed1 본학습 216epoch`

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_full.sh alg
```

## 4. iBKD

제목: `[Request]: Cityscapes DeepLabV3-to-Segmenter-L16 iBKD seed1 본학습 216epoch`

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_full.sh ibkd
```

명령은 승인된 사용 기간 안에서 실행합니다. 아래 예상 시간은 예약 승인 자체가 아닙니다.
기간이 짧다면 명령 끝에 `--max-hours 10`과 같이 **실제 허용 시간보다 여유 있게** 붙여
저장 후 멈추게 할 수 있습니다. 기본 명령은 별도 시간 제한 없이 216epoch를 수행합니다.

## 예상 소요 시간

H200 `bapedragon_771` smoke의 초기3step 중 2–3번째 시간을 80,352step으로 외삽한 값입니다.

| 방법 | 순수 학습 계산 시간 | 근거 |
|---|---:|---|
| Vanilla | 약50시간 | 약2.225초/step |
| LG | 약56시간 | 약2.495초/step |
| ALG | 약50–56시간 | guidance 종료 이후 계산 감소, 실제 종료 epoch 미확정 |
| iBKD | 약53–88시간 | 3.96초/step, epoch20 이후 guidance 종료 시점에 따라 감소 |

네 방법 순차 실행은 **순수 학습만 약9–10.5일**입니다.
전체 val500 평가55회, 데이터 로딩, 준비·저장 시간이 추가됩니다.
짧은 smoke의 추정이므로 실제 소요 시간은 달라질 수 있습니다.
첫 epoch 로그와 실제 val 시간을 확인한 뒤 남은 시간을 다시 계산합니다.
Teacher는 학습된 공개 가중치를 사용하므로 teacher 별도 학습 시간은 없습니다.

## 실행 내용

- `/app/data/chaoyang`의 원본 ZIP2개를 검증하고 scratch에 train/val만 준비합니다.
- 원본 소스와 공개 사전학습 모델을 자동 준비·검증합니다. 데이터 ZIP을 다시 다운로드하지 않습니다.
- 성공한 smoke와 같은 L/16·crop768·batch8·FP32·SGD Nesterov 조건으로216epoch를 학습합니다.
- val **pixel accuracy**로 checkpoint를 선택하고 **같은 checkpoint의 mIoU**를 기록합니다.
- 100step마다 저장하며, 중간에 멈춰도 복구 bundle을 보존하면 이어서 실행할 수 있습니다.
- 상세 조건·공개 논문 결과와의 차이는 [고정 프로토콜](OFFICIAL_L16_FULL_PROTOCOL.md)에 있습니다.

## 결과 확인

출력은 방법별로 다음 폴더에 저장됩니다.

```text
/app/output/cityscapes_official_l16_full_v1/<method>_seed1/
  run_*.log
  setup/                         # ZIP/데이터 검증 및 설치 로그
  artifacts/
    summary.json                 # complete / paused, 정확도와 같은 checkpoint의 mIoU
    history.json                 # 확정된 epoch별 loss·validation
    progress.json
    identity.json / config.json / provenance.json
    resume.json
    checkpoints/                 # 최근2세대 + 참조하는 best student
    steps_*.jsonl                # 실행 시도별 상세 학습 로그
```

본학습 완료 표식 예시:

```text
[CITYSCAPES_L16_FULL_DONE] status=complete method=vanilla epochs=216/216 step=80352 ...
```

`status=paused`는 저장 후 일시정지이고 완료가 아닙니다.
비유한 loss 등의 오류는 `status=failed` 및 `failure.json`에 남습니다.
설치·데이터 검사 단계 실패는 해당 setup/run 로그를 확인합니다.
실패 시 beta·LR·batch를 자동 변경하지 않습니다.

## 다른 이슈에서 이어서 실행하는 방법

H200 안내상 종료된 컨테이너는 삭제됩니다. 서버에 보관되는 `/app/output`이 다음 작업의
같은 경로로 자동 연결된다고 가정하지 않습니다.
운영진에게 이전 작업의 **`artifacts` 폴더 전체**를 공유 데이터 폴더로 복사해 달라고 요청합니다.
`resume.json` 하나만 전달하면 가중치가 없어 재개할 수 없습니다.
`checkpoints`의 상대 경로를 유지하고 파일을 수정하지 않습니다.

예를 들어 운영진이 실제로 `/app/data/chaoyang/cityscapes_ibkd_resume/`에
`resume.json`과 `checkpoints/`가 있도록 올렸을 때:

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_official_l16_full.sh ibkd --resume /app/data/chaoyang/cityscapes_ibkd_resume/resume.json
```

이 경로는 예시입니다. 전달받은 실제 경로를 사용합니다. 같은 코드 commit과 런타임 버전이
필요하며 다르면 프로그램이 재개를 거부합니다. Smoke checkpoint로는 본학습을 재개하지 않습니다.
