# H200 이슈 입력안 — Cityscapes Segmenter-L/16 crop512 smoke v3

이슈는 사용자가 직접 제출합니다. 이 문서와 실행 스크립트는 이슈를 자동 등록하지 않습니다.

[공식 요청 양식](https://github.com/Aerodrone-H200/gpu-request/issues/new?template=request-container.yml)에
아래 값을 입력합니다.

| 항목 | 입력값 |
|---|---|
| 제목 | `[Request]: Cityscapes Segmenter-L/16 crop512 LG·ALG·iBKD smoke v3` |
| 사용자 ID | `bapedragon` |
| GitHub 링크 | `https://github.com/bapedragon/iBKD_Seg.git` |
| 이미지 | `pytorch/pytorch:latest` |
| 언어 | `Python` |
| GPU 할당량(MIG 갯수) | `7` — H200 한 장 전체 |

코드 실행 명령어:

```bash
bash phase4/phase4_cityscapes/scripts/run_cityscapes_l16_crop512_smoke.sh
```

## 목적

이 실행은 Cityscapes 원본 train/val과 공개 사전학습 가중치를 사용하여
DeepLabV3-R101-D8 teacher에서 Segmenter-L/16 student로 전달하는
Vanilla/LG/ALG/iBKD 네 경로를 각각 3 update만 실행합니다. 성능 비교가 아니라
crop512에서의 학습 연결, finite loss/gradient, teacher 고정, checkpoint 재개,
GPU 메모리와 step 시간을 확인하는 smoke입니다.

## 고정 조건

- 데이터: fine train 2,975장 / val 500장, 19 trainIds, test 미사용.
- Student: ViT-L/16 24블록·1024채널, **mask-transformer decoder 1블록**.
  Segmenter 논문이 Cityscapes L/16에 명시한 decoder 깊이를 사용합니다.
- Student 초기화: 저자가 지정한 AugReg ImageNet 사전학습 NPZ.
- Teacher: MMSeg DeepLabV3 ResNetV1c-101-D8 Cityscapes 80k 공개 checkpoint, 전체 고정.
- 학습 입력: 원 논문의 crop768 대신 **512×512 random crop**. Random resize 0.5–2.0,
  category ratio 0.75, flip, photometric distortion, pad는 유지합니다.
- Batch 8, FP32, SGD Nesterov, lr0.01, momentum0.9, weight decay0,
  polynomial power0.9, min lr1e-5, 80,000-step 분모를 유지합니다.
- 평가는 원본 val 중 고정된 2장을 512 window/stride512 단일 scale로 수행합니다.
  Pixel accuracy와 mIoU는 데이터·평가 연결 진단값일 뿐 방법 선택에 사용하지 않습니다.
- LG/ALG 블록 `[0,12,23]`, iBKD는 24블록 전체 집계, 공통 beta2.5.
- 메모리를 위해 activation recomputation과 iBKD query chunking을 사용하지만
  loss, key 집합, batch, 정밀도는 줄이지 않습니다.

## 검사 항목

1. `/app/data/chaoyang`의 두 ZIP을 byte size, SHA-256, CRC로 다시 확인합니다.
2. 기존 압축 해제 데이터가 있으면 무결성을 확인하고 재사용합니다.
3. 고정 commit의 원본 Segmenter/MMCV/MMSeg 소스와 공개 가중치 두 개를 검증합니다.
4. 네 방법이 동일한 첫 24장, augmentation tensor, student 초기 state를 사용해야 합니다.
5. 각 방법의 3 step loss, CE, guidance, gradient norm, step 시간과 peak CUDA memory를 기록합니다.
6. 세 번째 update 직전 checkpoint를 strict reload하고 동일 update 재실행 결과를 검사합니다.
7. 실제 decoder depth가 1이고 LR scheduler 분모가 80,000인지 결과 JSON에 기록합니다.

crop, batch, precision 또는 모델을 OOM 시 자동 변경하지 않습니다. 실패하면 해당 설정과
직전 로그를 그대로 남깁니다. smoke 점수로 hyperparameter나 방법을 선택하지 않습니다.

## 완료 표식과 출력

정상 완료 로그:

```text
[CITYSCAPES_OFFICIAL_L16_SMOKE_DONE] status=passed methods=4/4 scientific_result=false
```

출력 루트:

```text
/app/output/cityscapes_l16_crop512_smoke_v3
```

핵심 결과는 `artifacts/smoke_summary.json`과 각 방법의
`artifacts/<method>/summary.json`에 저장됩니다. 후자에는 `decoder_layers=1`,
`schedule_total_steps=80000`, step별 시간, peak memory, 진단 pixel accuracy/mIoU가 포함됩니다.

## 해석 범위

Segmenter 논문의 Cityscapes 설정과 비교해 학습 protocol에서 의도적으로 바꾼 것은
768 crop/inference window를 512로 줄인 부분입니다. Pixel accuracy 우선 보고는 사용자가
정한 평가 정책이며 원 논문의 대표 지표는 mIoU입니다. DeepLab teacher와 LG/ALG/iBKD는
원 Segmenter 논문에 없는 연구 조건입니다. 따라서 이 실행은 Segmenter 공개 수치의 재현이
아니며, 동일한 crop512 예산에서 네 방법을 비교하기 위한 준비 검사입니다.
