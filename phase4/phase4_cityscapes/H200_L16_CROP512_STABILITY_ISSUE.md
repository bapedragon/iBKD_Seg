# H200 이슈 입력안 — Cityscapes L/16 crop512 500-step 안정성 검사 v1

이슈는 사용자가 직접 제출합니다. 이 문서와 실행 스크립트는 이슈를 자동 등록하지 않습니다.

[공식 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에
아래 값을 입력합니다.

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes L/16 crop512 LG·ALG·iBKD 500-step stability v1` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량(MIG 갯수) | `7` — H200 한 장 전체 |

코드 실행 명령어:

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_l16_crop512_stability.sh
```

## 목적

3-step smoke에서 LG와 ALG의 guidance loss가 `4.61 → 43.57 → 527.91`로 증가했습니다.
80,000-step 본학습을 시작하기 전에 현재 설정을 자동 변경하지 않고 Vanilla/LG/ALG/iBKD를
각각 최대 500 step 실행하여 손실과 gradient가 안정적인지 판정합니다.

## 고정 조건

- 데이터: fine train 2,975장. test는 접근하지 않습니다.
- Student: ImageNet 사전학습 Segmenter ViT-L/16, 24블록·1024채널,
  mask-transformer decoder 1블록.
- Teacher: MMSeg DeepLabV3 ResNetV1c-101-D8 Cityscapes 80k 공개 checkpoint, 전체 고정.
- 입력: 512×512 random crop, batch 8, FP32.
- 증강: random resize 0.5–2.0, category ratio 0.75, flip, photometric distortion, pad.
- SGD Nesterov, lr0.01, momentum0.9, weight decay0, polynomial schedule 80,000-step 분모.
- LG/ALG 블록 `[0,12,23]`, iBKD 24블록 집계, beta2.5, clipping 없음.
- 각 방법은 같은 student 초기값과 epoch별 데이터 순서·augmentation을 사용합니다.
- hyperparameter, 모델, batch, 정밀도는 실행 중 자동 변경하지 않습니다.

## 실행 및 판정

- 500 step은 train 전체 2,975장을 한 번 순회한 372 batch와 두 번째 epoch의 128 batch입니다.
- 모든 step의 CE, guidance, total loss, unclipped gradient norm, lr, 시간을 `steps.jsonl`에 기록합니다.
- 첫 5 step과 이후 25 step 간격의 값을 로그에 출력합니다.
- guidance projection parameter norm도 지정 milestone에 기록합니다.
- NaN/Inf가 발생하면 해당 방법은 즉시 `unstable`로 기록하지만, 나머지 방법 검사는 계속합니다.
- 500 step을 마친 방법은 고정 val 20장으로 pixel accuracy와 mIoU를 진단합니다.
- 첫 step 대비 peak가 100배를 넘거나 마지막 50-step 중앙값이 10배를 넘으면
  engineering stability gate를 통과하지 못합니다. 이 기준은 성능 선택 기준이 아닙니다.
- 네 방법 모두 안정적이고 동일한 초기 student/첫 batch를 사용한 경우에만
  `all_methods_stable=true`가 됩니다.

## 완료 표식과 출력

실행 자체가 끝나면 마지막에 다음 형식이 출력됩니다.

```text
[CITYSCAPES_L16_STABILITY_DONE] status=completed stable_methods=N/4 all_methods_stable=true|false
```

`status=completed`는 진단이 끝났다는 뜻입니다. 본학습 진행 조건은 반드시
`all_methods_stable=true`입니다.

출력 루트:

```text
/app/output/cityscapes_l16_crop512_stability_v1
```

핵심 파일:

- `artifacts/stability_summary.json`: 네 방법 종합 판정
- `artifacts/<method>/summary.json`: 방법별 안정성 판정과 진단 metric
- `artifacts/<method>/steps.jsonl`: 500-step 원시 학습 기록
- `run.log`: 전체 실행 로그

이 실행의 pixel accuracy와 mIoU는 20장 진단값으로 논문 성능 비교에 사용하지 않습니다.
