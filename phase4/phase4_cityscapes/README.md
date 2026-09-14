# Cityscapes 직접 segmentation 실험

## 현재 작업: DeepLabV3 → Segmenter GPU smoke

2026-09-14 사용자 선택에 따라 **DeepLabV3-ResNet101 → Segmenter-S/16 mask decoder**로
Vanilla/LG/ALG/iBKD 연결을 먼저 확인합니다. Cityscapes 데이터는 아직 준비되지 않았습니다.
이 요청은 **합성 입력·임의 가중치로 실행하는 구조 검증**이며 실제 Cityscapes 성능 평가가 아닙니다.

- [현재 smoke 조건·공식 프로토콜 구분](DEEPLAB_SEGMENTER_SMOKE.md)
- [H200 제출 본문](H200_SMOKE_ISSUE.md)
- [H200 smoke 이슈 #448](https://github.com/Aerodrone-H200/gpu-request/issues/448)
- [GPU smoke 설정](configs/deeplabv3_segmenter_smoke_v1.json)
- 실행: `bash phase4/phase4_cityscapes/scripts/run_deeplabv3_segmenter_smoke.sh`
- 로컬: `PYTHONPATH=src .venv/bin/python -m ibkd_seg.cityscapes.public_smoke --device cpu --cpu-small --output-dir outputs/cityscapes_arch_smoke_new`

아래는 모델 선택 전 만들었던 **ResNet50/DeiT + 자체 convolution decoder pilot 이력**입니다.
그 실행 명령은 DeepLabV3/Segmenter 실험에 사용하지 않습니다.

## 이전 scratch pilot

Vanilla·LG·ALG·iBKD를 같은 DeiT-Tiny encoder와 decoder로 비교하는 실행 구성입니다.
현재 pilot은 Cityscapes 전용 ResNet-50 segmentation teacher를 먼저 학습하고, 선택된 teacher
하나를 모든 guided student가 공유합니다. 모델과 decoder를 모두 픽셀 라벨로 학습합니다.

현재 상태는 **실행 구성 및 로컬 합성 데이터 검증 완료**입니다. 실제 Cityscapes
mIoU, H200 실행시간, 기본 512×512 설정의 GPU 메모리는 아직 측정하지 않았습니다.
기본 설정은 탐색용 `pilot`이며, 논문 결과로 확정된 프로토콜이 아닙니다.

- 현재 설정: [configs/scratch_pixel_accuracy_v2.json](configs/scratch_pixel_accuracy_v2.json)
- 이전 mIoU 선택 설정 보존: [configs/scratch_v1.json](configs/scratch_v1.json)
- 고정 조건과 해석 범위: [PROTOCOL.md](PROTOCOL.md)
- 로컬 검증: [VALIDATION.md](VALIDATION.md)
- 구현: [src/ibkd_seg/cityscapes](../../src/ibkd_seg/cityscapes)

## 기본 구성

| 항목 | 설정 |
|---|---|
| 데이터 | Fine train 2,975장 / val 500장, 19클래스 |
| 초기화 | Teacher와 student 모두 scratch, 외부 pretrained weight 없음 |
| Teacher | ResNet-50 + 4단계 convolution decoder |
| Student | DeiT-Tiny/16 + 공통 4-layer convolution decoder |
| 비교 방법 | Vanilla / LG / ALG / iBKD λ=0.25 |
| 학습 | 512×512 crop, batch 4, 100 epochs, AdamW, BF16 |
| 평가 | 원본 1024×2048, 512×512 window / stride 256, 단일 scale |
| 1차 평가·선택 | 5 epoch마다 val pixel accuracy 평가, 가장 높은 checkpoint 선택 |
| 보조 평가 | 같은 checkpoint의 mIoU·클래스별 IoU |
| Seed | Teacher 1개(seed 1), student seed 1·2·3 |

먼저 seed 1로 teacher 포함 **5개 실행**을 진행할 수 있습니다. 전체 구성은
teacher 1개 + student 4방법 × 3 seed = **13개 실행**입니다. 한 GPU에서 순차 실행합니다.

## 1. 환경과 데이터 준비

저장소 루트 `iBKD_Seg/`에서 실행합니다. H200 서버에서는 CUDA가 지원되는
PyTorch/torchvision 환경을 사용해야 합니다. 기존 의존성을 그대로 사용하며,
MMSegmentation이나 별도 CUDA 확장 패키지는 추가하지 않았습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

[Cityscapes 공식 다운로드](https://www.cityscapes-dataset.com/downloads/)에서 계정으로
로그인해 `leftImg8bit_trainvaltest.zip`과 `gtFine_trainvaltest.zip`을 받습니다.
두 파일을 같은 데이터 루트에 풀어 아래 구조를 만듭니다. 코드가 계정에 로그인하거나
데이터를 자동 다운로드하지는 않습니다.

```text
/app/data/cityscapes/
├── leftImg8bit/
│   ├── train/<city>/*_leftImg8bit.png
│   └── val/<city>/*_leftImg8bit.png
└── gtFine/
    ├── train/<city>/*_gtFine_labelIds.png
    └── val/<city>/*_gtFine_labelIds.png
```

`labelIds`를 입력으로 받습니다. `labelTrainIds`, color mask, instanceIds로
파일을 바꾸면 안 됩니다. 압축 파일에 test 폴더가 있어도 이 runner는 접근하지 않습니다.

## 2. 실행 전 확인

데이터 없이도 CPU에서 전체 코드 경로를 확인할 수 있습니다. 출력 폴더는 새 경로여야 합니다.

```bash
python -m ibkd_seg.cityscapes.run smoke \
  --output-dir outputs/cityscapes_smoke --device cpu
```

실제 GPU에서는 **본 학습 전에** 기본 crop·batch·BF16 설정으로 다음 점검을 실행합니다.
Teacher와 네 방법 각각 3 step을 수행하며 첫 step 이후 속도와 최대 메모리를 기록합니다.
입력과 teacher는 합성이므로 이 결과에 정확도 의미는 없습니다.

```bash
python -m ibkd_seg.cityscapes.run preflight \
  --output /app/output/cityscapes_preflight_pa_v2.json --device cuda
```

OOM이 나면 모든 student가 공유하는 설정을 새 파일로 복사해 batch 등을 조정하고
`protocol_id`도 바꿉니다. 변경한 설정은 이후 모든 명령에 `--config /path/to/new.json`으로
동일하게 전달합니다. 방법 하나에만 작은 batch나 낮은 해상도를 적용하지 않습니다.

데이터 파일 수·해상도·마스크·byte size·SHA-256을 확인합니다.

```bash
python -m ibkd_seg.cityscapes.run audit \
  --data-dir /app/data/cityscapes \
  --output /app/output/cityscapes_manifest.json
```

학습과 독립 평가는 manifest에 기록된 모든 이미지/마스크의 크기와 해시를 다시 확인합니다.
따라서 학습 시작 전에 데이터 검증 시간이 추가됩니다.

## 3. 먼저 seed 1 실행

```bash
python -m ibkd_seg.cityscapes.run matrix \
  --data-dir /app/data/cityscapes \
  --manifest /app/output/cityscapes_manifest.json \
  --output-dir /app/output/cityscapes_scratch_pa_v2 \
  --seeds 1 --device cuda --execute
```

`--execute`를 빼면 실행 명령만 출력합니다. 위 명령은 teacher를 먼저 끝내고
Vanilla → LG → ALG → iBKD 순서로 진행합니다. 실패하면 이후 작업을 중단하고 로그를 남깁니다.

## 4. seed 2·3 추가 또는 중단 후 재개

```bash
python -m ibkd_seg.cityscapes.run matrix \
  --data-dir /app/data/cityscapes \
  --manifest /app/output/cityscapes_manifest.json \
  --output-dir /app/output/cityscapes_scratch_pa_v2 \
  --seeds 1 2 3 --device cuda --resume --execute
```

완료된 실행은 체크포인트 계약을 확인하고 건너뛰며, 미완료 실행은 마지막으로 저장된
epoch 다음부터 재개합니다. 중단된 epoch의 미저장 batch는 다시 계산합니다. 설정,
데이터, teacher, source code가 달라지면 재개를 거부합니다. 수정 실험은 새 출력 폴더를 사용합니다.

## 5. 결과 확인과 재평가

출력 루트에 `results.csv`와 `matrix_summary.json`이 생깁니다. 후자는 seed별 수치와
pixel accuracy와 mIoU의 평균·표본 표준편차를 각각 포함하며, 1 seed일 때 SD는 `null`입니다.

각 실행 폴더에는 다음 파일을 저장합니다.

- `best.pt`: val로 선택한 encoder+decoder. Teacher/IBAM 없이 student 추론 가능.
- `latest.pt`: optimizer, guidance, controller, RNG를 포함하는 재개용 체크포인트.
- `history.json`: loss, guidance 가중치·종료 epoch, pixel accuracy, 클래스별 IoU, 전체 confusion matrix.
- `summary.json`: 정확도로 선택한 epoch의 val pixel accuracy·mIoU, 체크포인트 byte size·SHA-256, 시간·GPU 메모리.
- `config.json`, `identity.json`, `environment.json`: 설정·데이터·코드·초기화 식별 정보.
- 출력 루트의 `logs/`: 실행별 표준 출력과 오류 로그.

```bash
python -m ibkd_seg.cityscapes.run evaluate \
  --data-dir /app/data/cityscapes \
  --manifest /app/output/cityscapes_manifest.json \
  --checkpoint /app/output/cityscapes_scratch_pa_v2/ibkd_seed1/best.pt \
  --output /app/output/cityscapes_ibkd_seed1_recheck.json --device cuda
```

`pixel_accuracy`, `miou`, `class_iou` JSON 값은 0–1 범위이고, `results.csv`는 % 단위입니다.
저장된 점수는 **validation에서 checkpoint를 선택한 결과**입니다. 독립 test 성능이나
Cityscapes 공식 leaderboard 결과로 해석하지 않습니다.

## 정확도·test·모델 선택에 대한 보완

2026-09-14 사용자 요청에 따라 기본 설정을 v2로 올리고 **pixel accuracy를 1차 지표와
checkpoint 선택 기준**으로 변경했습니다. 정확도는 `맞힌 유효 픽셀 수 / 전체 유효 픽셀 수`이며,
이미지 한 장의 Top-1 분류 정확도와 구분합니다. Ignore 픽셀은 분모·분자 모두에서 제외합니다.
mIoU는 동일한 정확도 선택 checkpoint에서 보조 지표로 기록합니다. 각 지표의 최고점을 서로
다른 epoch에서 가져와 섞지 않습니다. 기존 합성 실행은 v1 이력으로 남깁니다.

Cityscapes에는 공식 test **1,525장**이 있습니다. Test의 일반적인 semantic 정답은 비공개이고
공식 서버에 예측을 제출해 평가합니다. 현재 runner는 공개 정답이 있는 train 2,975장과
val 500장만 사용하며 test 예측 생성/제출을 아직 구현하지 않았습니다.
[공식 split 설명](https://github.com/mcordts/cityscapesScripts/blob/master/README.md)

현재의 **ResNet-50·DeiT-Tiny + 자체 decoder 전체 조합은 Cityscapes의 대표 표준 구성이
아닙니다.** 기존 LG/iBKD의 layer·channel 계약을 재사용한 pilot입니다. ResNet-50 자체는
널리 쓰이는 backbone이지만 DeepLabV3+처럼 검증된 decoder와 묶은 구성을 함께 봐야 합니다.
Cityscapes에서 보편적인 비교를 우선한다면 **DeepLabV3+ ResNet-101 teacher →
SegFormer-B0 student**가 후보이며, 이는 표준 개별 모델을 조합한 증류 실험 제안입니다.
이 teacher/student 조합이 LG/ALG/iBKD의 공식 Cityscapes 기준선이라는 의미는 아닙니다.
현재 코드는 아직 이 조합으로 교체하지 않았습니다. SegFormer는 layer마다 채널·해상도가
달라 기존 고정 12-layer 집계를 그대로 연결할 수 없고 alignment 정의가 추가로 필요합니다.
[DeepLabV3+ Cityscapes 구성](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/deeplabv3plus),
[SegFormer 공식 구현](https://github.com/NVlabs/SegFormer)
