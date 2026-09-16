# H200 이슈 입력안 — CUB 직접 segmentation Vanilla/LG/ALG/iBKD smoke v1

이 문서는 H200 요청 이슈에 붙여 넣을 입력안입니다. 자동으로 외부 이슈를 등록하지 않습니다.

[공식 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에
아래 내용을 입력합니다.

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: CUB-200 직접 segmentation Vanilla·LG·ALG·iBKD real-data smoke v1` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량(MIG 갯수) | `7` — H200 한 장 전체 |

코드 실행 명령어:

```bash
bash phase1/phase1_cub_Seg/scripts/run_direct_segmentation_smoke.sh
```

## 목적

기존 `phase1/phase1_cub`의 classification 및 frozen/direct spatial 진단과 별개로,
CUB의 공식 binary segmentation mask를 실제 학습 정답으로 넣었을 때 다음 경로가 모두
작동하는지 먼저 확인합니다.

- Vanilla: segmentation CE만 사용
- LG: CE + first/middle/last feature guidance
- ALG: CE + LG feature guidance + ALG controller
- iBKD: CE + 12개 student block 학습 가중합 + alignment/fusion guidance

이 smoke는 확장 가능성 확인용입니다. 3step 점수는 성능 비교나 논문 수치가 아닙니다.

## 고정 데이터 계약

- CUB-200-2011 공식 이미지 archive와 공식 `segmentations.tgz`를 자동 다운로드하고
  알려진 byte size·MD5·SHA-256으로 확인합니다.
- classification 실험과 같은 official-train 기반 고정 split을 사용합니다.
  `train 5,394 / validation 600`, class당 validation 3장, split seed 2027입니다.
- smoke에서는 결과를 보기 전에 image ID로 고정한 첫 train 6장과 validation 8장만
  실제 tensor로 만듭니다. 전체 5,394/600 image-mask pair의 존재 여부도 확인합니다.
- mask grayscale 값 `> 0`은 bird `1`, 나머지는 background `0`으로 변환합니다.
- RGB와 mask에 동일한 resize 및 고정 horizontal flip을 적용합니다. RGB는 bilinear,
  mask는 nearest-neighbor 보간을 사용합니다.
- official test record는 model input, 선택, metric 계산에 사용하지 않습니다.

## 모델·학습 연결

- Teacher: scratch ResNet-50 encoder + 공통 4-level convolutional decoder, 2 classes.
- Student: scratch DeiT-Tiny/16 12 blocks + 같은 형식의 4-level decoder, 2 classes.
- 먼저 teacher를 동일한 고정 train batch로 3step 학습하고 strict-load 가능한 checkpoint를
  만듭니다. LG/ALG/iBKD는 모두 이 한 teacher state를 frozen/eval 상태로 공유합니다.
- 네 student는 동일 초기 state와 동일한 image/mask tensor를 사용합니다.
- 입력 224×224, batch 2, 각 방법 3step, BF16, AdamW lr `5e-4`, weight decay `0.05`,
  gradient clip `1.0`, guidance beta `2.5`, iBKD fusion ratio `0.25`입니다.
- LG/ALG는 student block `[0, 6, 11]`과 teacher layer2/3/4를 연결합니다.
  iBKD는 12개 block 전체를 사용합니다.
- encoder·decoder·guidance의 nonzero gradient, teacher freeze, optimizer update,
  checkpoint strict reload, 입력/state hash 동일성을 검사합니다.
- OOM 시 batch나 입력 크기, precision을 자동 변경하지 않고 실패합니다.

## segmentation 지표

대표 진단 지표는 dataset-level **2-class mIoU**입니다. Bird와 background IoU를 각각
계산한 뒤 평균합니다. 함께 기록하는 값은 다음과 같습니다.

- `foreground_iou`: bird IoU
- `background_iou`: background IoU
- `foreground_dice`: bird Dice
- `pixel_accuracy`: 전체 픽셀 정확도

Pixel accuracy는 넓은 background를 모두 맞히는 예측도 높게 나올 수 있으므로 대표 지표로
사용하지 않습니다. Smoke의 validation 8장 점수는 metric 구현과 데이터 연결 확인값일 뿐,
방법 순위나 checkpoint 선택에 사용하지 않습니다.

## 결과와 완료 표식

출력 루트: `/app/output/cub_direct_segmentation_smoke_v1`

- `run.log`: 전체 실행 로그
- `artifacts/config.json`: 실제 실행 설정
- `artifacts/dataset_audit.json`: split, pair inventory, 선택 ID와 tensor hash
- `artifacts/teacher/summary.json`, `teacher.pt`: 3step teacher 진단과 checkpoint
- `artifacts/<method>/summary.json`: 방법별 loss, gradient/reload 검사, 진단 metric
- `artifacts/smoke_summary.json`: 네 방법의 교차 동일성 검사 및 최종 요약

성공 시 마지막 로그:

```text
[CUB_DIRECT_SEGMENTATION_SMOKE_DONE] status=passed methods=4/4 scientific_result=false
```

같은 컨테이너에서 재실행하려면 `CUB_SEGMENTATION_SMOKE_OUTPUT_DIR`에 비어 있는 새
출력 경로를 지정합니다. 다운로드·압축 해제 데이터 위치는
`CUB_SEGMENTATION_DATA_DIR`로 바꿀 수 있습니다.
