# L/16 10,000-step 원본 정리 및 iBKD guidance 종료 전후 분석

2026-09-28에 사용자가 제공한 issue 810·812·813·814의 전체 결과 폴더를 정리했습니다.
**기존 결과표의 8개 후보 수치는 모두 원본과 일치합니다.** 이번 작업은 과거
Segmenter-L/16 v18 결과의 파일 검증이며, Tiny 실험이나 새 학습 결과가 아닙니다.

## 원본 보관 위치

저장소 루트 기준:

```text
phase4/phase4_cityscapes/results/raw/cityscapes/
└── l16_crop512_candidate_top2_grid10000_v18/
    ├── pack1_issue810/
    │   ├── h200_issue_810.log
    │   └── output/
    ├── pack2_issue812/
    │   ├── h200_issue_812.log
    │   └── output/
    ├── pack3_issue813/
    │   ├── h200_issue_813.log
    │   └── output/
    └── pack4_issue814/
        ├── h200_issue_814.log
        └── output/
```

각 `output/`에는 기존 `run.log`, 데이터 검사 결과, 설정, provenance,
`artifacts/<run_id>/steps.jsonl` 및 `summary.json` 등을 그대로 보존했습니다.
`bapedragon_<issue>_result.txt`는 `h200_issue_<issue>.log`로,
긴 서버 출력 폴더 이름은 `output/`으로 변경했습니다. 방법·λ·β가 들어 있는
실행별 폴더명은 이미 명확하므로 유지했습니다.

| 보관 폴더 | 실행 1 | 실행 2 |
|---|---|---|
| `pack1_issue810` | LG β=0.05 | iBKD λ=0.25, β=0.25 |
| `pack2_issue812` | LG β=0.02 | iBKD λ=0.25, β=0.5 |
| `pack3_issue813` | ALG β=0.05 | iBKD λ=0.5, β=0.1 |
| `pack4_issue814` | ALG β=0.02 | iBKD λ=0.5, β=0.25 |

사용자가 지정한 Downloads 하위 네 폴더를 위 위치로 이동했으며, **원본 60개 파일을
모두 보존**했습니다. 각 pack의 최상위 결과 로그와 `output/run.log`는 내용이 같지만
둘 다 남겼습니다. 원래 경로, 정리한 경로, byte size와 SHA-256은
[`full_artifact_manifest.json`](full_artifact_manifest.json)에 기록했습니다.
이동한 60개 파일은 이동 전후 크기와 SHA-256이 모두 일치합니다.

원본은 `.gitignore` 대상인 `results/raw/`에 보관합니다. Git에는 결과표, 감사·분석
요약과 파일 해시만 반영합니다. 따라서 새로 clone한 컴퓨터에서 원본을 사용하려면
이 로컬 raw 폴더도 별도로 복사해야 합니다.

## 확인 결과

- JSON 44개와 JSONL 8개를 파싱했습니다. 8개 실행에 각각 1–10,000step이 빠짐없이
  있어 학습 기록은 총 80,000개입니다.
- 모든 실행의 기록된 수치는 유한값이며, `loss = CE + beta × guidance`와 일치합니다
  (부동소수점 반올림 오차 허용).
- 원본 summary, pack summary, 마지막 로그 JSON과 기존 Git 결과가 일치합니다.
- val 500장의 19×19 혼동행렬에서 pixel accuracy, mIoU, class IoU를 다시 계산해
  기존 값과 일치함을 확인했습니다. 이 계산은 저장된 혼동행렬의 검산이며 모델 추론을
  다시 수행한 것은 아닙니다.
- 매 step의 입력 해시로 전체 stream SHA-256을 다시 계산했고, 8개 모두 보고된
  해시와 일치합니다. 실제 이미지 tensor를 다시 생성해서 해시를 확인한 것은 아닙니다.
- 초기 student·teacher의 보고된 state hash는 모든 실행에서 같고, 같은 β의 LG·ALG는
  최종 student hash와 평가 결과도 같습니다. 가중치 파일은 없어 state hash 자체를
  모델에서 다시 계산할 수는 없습니다.
- LG·ALG의 전체 scalar 로그는 완전히 동일하지 않습니다. CE·total loss의 최대 차이는
  `4.7684e-7`, CE 대비 guidance 비율의 최대 차이는 `4.4704e-8`입니다.
  입력 해시·β·LR·guidance·gradient norm은 같았습니다. 따라서 최종 hash가 같다는 사실을
  모든 scalar 로그까지 bitwise 동일하다는 의미로 확장하지 않습니다.
- 학습된 모델 체크포인트(`.pt`, `.pth`, `.ckpt`, `.safetensors`)는 **0개**입니다.
  mIoU는 **10,000step 마지막 val 500장 평가 한 번만** 있습니다.

세부 검증 결과와 구간별 CE 수치는 [`raw_import_audit.json`](raw_import_audit.json)에 있습니다.
이번 검증은 원본 파일과 기록의 일관성 검사이며, 새 반복 학습으로 수행한 재현성 검사는 아닙니다.

## iBKD: 꺼지기 전과 후 중 어느 쪽이 나았나?

**mIoU 기준으로는 판정할 수 없습니다.** 종료 전의 validation 결과와 중간 가중치가
없어 7,441step 전후 mIoU를 비교하거나 사후 평가할 수 없습니다. 아래 CE는 학습 중
정답을 맞추는 정도를 보는 보조 지표이며, validation 성능을 대신하지 않습니다.

비교 구간은 종료 직전 500step(6,941–7,440), 직후 500step(7,441–7,940), 마지막
500step(9,501–10,000)입니다. 각 값은 **step별 minibatch CE의 산술평균**이며,
픽셀 수로 다시 가중한 전체 구간 CE는 아닙니다. 표의 mIoU는 모두 마지막 10,000step 값입니다.

| iBKD λ | β | guidance 종료 | 종료 직전 평균 CE | 종료 직후 평균 CE | 마지막 평균 CE | 최종 mIoU |
|---:|---:|---|---:|---:|---:|---:|
| 0.25 | 0.25 | 7,441step부터 OFF | 0.229697 | 0.225324 | 0.203190 | 64.155% |
| 0.25 | 0.50 | 7,441step부터 OFF | 0.240149 | 0.228090 | 0.207603 | **69.736%** |
| 0.50 | 0.25 | 7,441step부터 OFF | 0.243082 | 0.234873 | 0.213656 | 65.165% |
| 0.50 | 0.10 | 10,000step까지 ON | 해당 없음 | 해당 없음 | 0.210376 | 65.894% |

꺼진 세 실행 모두 직후와 마지막 구간의 평균 CE가 더 낮았습니다. 즉 **가이던스를 끈
뒤에도 학습 CE는 내려갔습니다.** 그러나 추가 학습 자체의 효과, 입력 batch와 augmentation의
변화가 섞여 있고, 같은 λ·β에서 가이던스를 계속 켜 둔 대조 실험이 없습니다.
따라서 이 관찰로 “끄는 것이 유리했다”, “20epoch보다 늦게 끄는 편이 낫다”를 결론낼 수 없습니다.
λ=0.5·β=0.1은 β가 다르므로 종료 효과를 검증하는 동일 조건 대조군이 아닙니다.

총 loss는 종료 시 `beta × guidance`가 0이 되어 구조적으로 내려가므로, 종료 전후
학습 상태를 확인할 때는 total loss보다 CE를 비교했습니다. 종료 후 raw guidance가
양수로 기록되어도 실제 loss 가중치 β가 0이므로 guidance가 계속 적용된다는 뜻은 아닙니다.

### 종료 epoch 표기

이 L/16 실행은 1epoch가 `ceil(2975 / 8) = 372step`입니다. 7,440step이 20epoch의 끝이고,
7,441step은 21epoch의 첫 step입니다. 원본의 `guidance_stop_epoch=21`은 **β=0을 처음
사용한 epoch**를 뜻하며, 20epoch warm-up을 끝낸 뒤 꺼졌다는 설명과 일치합니다.
현재 Tiny 후속 실행의 체크포인트 저장 주기나 epoch 표기 규칙을 이 과거 L/16 파일에
그대로 적용하면 안 됩니다.

### 나중에 종료 효과를 직접 비교하려면

같은 초기 가중치·seed·입력 순서·λ·β를 사용하고, 기존 종료 설정과 guidance 유지 설정을
비교해야 합니다. 종료 전후 동일한 step에서 val 500장 mIoU를 평가하고 체크포인트를
저장해야 합니다. 이미 이번 파일에 그 비교 결과가 있는 것은 아니며, 새 학습을
이번 정리 작업에서 실행하지 않았습니다.
